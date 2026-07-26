"""High-level analysis utilities for attribution graphs.

This module provides the primary programmatic API for analyzing circuit-tracer
graphs.  It re-exports low-level primitives from :mod:`circuit_tracer.graph` and
adds higher-level helpers (feature ranking, intervention bridging, batch
comparison) so that users have a single, discoverable import path::

    from circuit_tracer.analysis import get_top_features, prune_graph, compute_graph_scores

The standalone functions are also available as convenience methods on
:class:`~circuit_tracer.graph.Graph` (e.g. ``graph.top_features()``).
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch

from circuit_tracer.graph import (
    PruneResult,
    compute_edge_influence,
    compute_graph_scores,
    compute_node_influence,
    prune_graph,
)

if TYPE_CHECKING:
    from circuit_tracer.graph import Graph
    from circuit_tracer.replacement_model.common import Intervention

# ── Re-exports (so users can import everything from one place) ──────
__all__ = [
    # Low-level (from graph.py)
    "prune_graph",
    "PruneResult",
    "compute_graph_scores",
    "compute_node_influence",
    "compute_edge_influence",
    # High-level (defined here)
    "get_top_features",
    "summarize_graph",
    "graph_to_interventions",
    "summarize_interventions",
    "summarize_intervention_results",
    "compare_graphs",
    "find_common_circuit",
    "ComparisonResult",
]

CIRCUIT_SUMMARY_KIND = "circuit-tracer.summary.v1"
INTERVENTION_SUMMARY_KIND = "circuit-tracer.interventions.v1"
INTERVENTION_RESULT_SUMMARY_KIND = "circuit-tracer.intervention-results.v1"


# ── Feature ranking ─────────────────────────────────────────────────


def get_top_features(graph: Graph, n: int = 10) -> tuple[list[tuple[int, int, int]], list[float]]:
    """Extract the top-*n* feature nodes by total multi-hop influence.

    Uses :func:`compute_node_influence` to rank features by their total
    effect on *all* logit targets (direct + indirect paths), weighted by
    each target's probability.

    Args:
        graph: A :class:`~circuit_tracer.graph.Graph` produced by
            :func:`~circuit_tracer.attribute`.
        n: Number of top features to return.

    Returns:
        ``(features, scores)`` where *features* is a list of
        ``(layer, position, feature_idx)`` tuples and *scores* is the
        corresponding influence values.
    """
    n_logits = len(graph.logit_targets)
    n_features = len(graph.selected_features)

    # Build logit weight vector — one entry per node, non-zero only for logits
    logit_weights = torch.zeros(
        graph.adjacency_matrix.shape[0], device=graph.adjacency_matrix.device
    )
    logit_weights[-n_logits:] = graph.logit_probabilities

    # Multi-hop influence across all logit targets
    node_influence = compute_node_influence(graph.adjacency_matrix, logit_weights)
    feature_influence = node_influence[:n_features]

    top_k = min(n, n_features)
    top_values, top_indices = torch.topk(feature_influence, top_k)

    features: list[tuple[int, int, int]] = [
        tuple(graph.active_features[graph.selected_features[i]].tolist())  # type: ignore[misc]
        for i in top_indices
    ]
    scores = top_values.tolist()
    return features, scores


def _tensor_to_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _target_summaries(graph: Graph) -> list[dict]:
    probabilities = _tensor_to_list(graph.logit_probabilities)

    return [
        {
            "token": target.token_str,
            "vocabIndex": int(target.vocab_idx),
            "probability": float(probabilities[index]),
        }
        for index, target in enumerate(graph.logit_targets)
    ]


def _input_summary(graph: Graph) -> dict:
    return {
        "text": graph.input_string,
        "tokenIds": [int(token) for token in _tensor_to_list(graph.input_tokens)],
        "tokenCount": len(graph.input_tokens),
    }


def _model_summary(graph: Graph) -> dict:
    return {
        "name": getattr(graph.cfg, "model_name", None),
        "tokenizer": getattr(graph.cfg, "tokenizer_name", None),
        "layers": int(graph.cfg.n_layers),
        "vocabSize": int(graph.vocab_size),
        "scan": graph.scan_name,
    }


def _activation_lookup(graph: Graph) -> dict[tuple[int, int, int], float]:
    lookup = {}

    for selected_index in graph.selected_features.detach().cpu().tolist():
        feature = tuple(int(value) for value in graph.active_features[selected_index].tolist())
        lookup[feature] = float(graph.activation_values[selected_index].item())

    return lookup


def _top_feature_summaries(graph: Graph, top_n: int) -> list[dict]:
    top_features, influence_scores = get_top_features(graph, n=top_n)
    activations = _activation_lookup(graph)

    return [
        {
            "layer": int(layer),
            "position": int(position),
            "feature": int(feature),
            "influence": float(influence_scores[index]),
            "activation": activations.get((layer, position, feature)),
        }
        for index, (layer, position, feature) in enumerate(top_features)
    ]


def _pruning_summary(graph: Graph, node_threshold: float, edge_threshold: float) -> dict:
    node_mask, edge_mask, cumulative_scores = prune_graph(
        graph,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    return {
        "nodeThreshold": node_threshold,
        "edgeThreshold": edge_threshold,
        "keptNodeCount": int(node_mask.sum().item()),
        "keptEdgeCount": int(edge_mask.sum().item()),
        "maxCumulativeScore": (
            float(cumulative_scores.max().item()) if cumulative_scores.numel() else 0.0
        ),
    }


def summarize_graph(
    graph: Graph,
    top_n: int = 10,
    node_threshold: float | None = 0.8,
    edge_threshold: float | None = 0.98,
) -> dict:
    """Return a JSON-safe summary of a circuit attribution graph.

    The raw ``Graph`` object is a PyTorch-rich research object. This summary is
    the stable interchange layer: it captures the prompt, model identity, target
    logits, top features, and graph-quality metrics without requiring callers to
    deserialize a ``.pt`` file.
    """

    replacement_score, completeness_score = compute_graph_scores(graph)
    n_features = len(graph.selected_features)
    n_tokens = len(graph.input_tokens)
    n_logits = len(graph.logit_targets)
    n_layers = int(graph.cfg.n_layers)
    n_errors = n_layers * n_tokens
    pruning = None

    if node_threshold is not None and edge_threshold is not None:
        pruning = _pruning_summary(graph, node_threshold, edge_threshold)

    return {
        "kind": CIRCUIT_SUMMARY_KIND,
        "version": 1,
        "input": _input_summary(graph),
        "model": _model_summary(graph),
        "targets": _target_summaries(graph),
        "nodeCounts": {
            "selectedFeatures": n_features,
            "activeFeatures": int(len(graph.active_features)),
            "errorNodes": n_errors,
            "tokenNodes": n_tokens,
            "logitNodes": n_logits,
            "totalNodes": int(graph.adjacency_matrix.shape[0]),
        },
        "metrics": {
            "replacementScore": float(replacement_score),
            "completenessScore": float(completeness_score),
        },
        "topFeatures": _top_feature_summaries(graph, top_n),
        "pruning": pruning,
    }


# ── Intervention bridge ─────────────────────────────────────────────


def graph_to_interventions(
    graph: Graph,
    n: int = 10,
    value: float = 0.0,
) -> list[Intervention]:
    """Convert a graph's top features into intervention tuples.

    This bridges the attribution → intervention workflow: find the most
    influential features, then create tuples ready for
    ``model.feature_intervention()``.

    Args:
        graph: An attribution graph.
        n: Number of top features to convert.
        value: Activation value to set.  ``0.0`` (default) means ablation;
            use the feature's original activation for amplification, etc.

    Returns:
        List of ``(layer, position, feature_idx, value)`` tuples.
    """
    features, _ = get_top_features(graph, n)
    return [(layer, pos, feat_idx, value) for layer, pos, feat_idx in features]


def summarize_interventions(
    graph: Graph,
    n: int = 10,
    value: float = 0.0,
) -> dict:
    """Return a JSON-safe intervention plan for the graph's top features.

    The returned summary is an interchange artifact. It records the exact
    ``ReplacementModel.feature_intervention`` tuple payload that would be used
    for feature ablations or feature setting, but it does not execute the model
    or claim an observed causal effect.
    """
    intervention_value = float(value)
    features, influence_scores = get_top_features(graph, n)
    activations = _activation_lookup(graph)

    return {
        "kind": INTERVENTION_SUMMARY_KIND,
        "version": 1,
        "sourceGraph": {
            "kind": CIRCUIT_SUMMARY_KIND,
            "input": _input_summary(graph),
            "model": _model_summary(graph),
            "targets": _target_summaries(graph),
        },
        "runtime": {
            "function": "ReplacementModel.feature_intervention",
            "tupleFormat": ["layer", "position", "feature", "value"],
            "executed": False,
        },
        "interventionType": (
            "feature_ablation" if intervention_value == 0.0 else "feature_set"
        ),
        "value": intervention_value,
        "interventionCount": len(features),
        "interventions": [
            {
                "layer": int(layer),
                "position": int(position),
                "feature": int(feature),
                "value": intervention_value,
                "sourceInfluence": float(influence_scores[index]),
                "sourceActivation": activations.get((layer, position, feature)),
            }
            for index, (layer, position, feature) in enumerate(features)
        ],
    }


def _final_logits(value: torch.Tensor | Sequence) -> torch.Tensor:
    tensor = (
        value.detach().cpu().float()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, dtype=torch.float32)
    )

    if tensor.ndim == 0:
        raise ValueError("Logits must have at least one dimension")

    if tensor.ndim == 1:
        return tensor

    return tensor.reshape(-1, tensor.shape[-1])[-1]


def _token_logit_effects(
    baseline_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    target_token_ids: Sequence[int],
) -> list[dict]:
    effects = []

    for token_id in target_token_ids:
        if token_id < 0 or token_id >= baseline_logits.numel():
            raise ValueError(f"Target token id out of range: {token_id}")

        before = float(baseline_logits[token_id].item())
        after = float(intervened_logits[token_id].item())
        delta = after - before
        effects.append(
            {
                "vocabIndex": int(token_id),
                "beforeLogit": before,
                "afterLogit": after,
                "deltaLogit": delta,
                "absoluteDelta": abs(delta),
            }
        )

    return effects


def _top_logit_shifts(
    baseline_logits: torch.Tensor,
    intervened_logits: torch.Tensor,
    top_k: int,
) -> list[dict]:
    delta = intervened_logits - baseline_logits
    k = max(0, min(top_k, delta.numel()))

    if k == 0:
        return []

    _, indices = torch.topk(delta.abs(), k)

    return [
        {
            "vocabIndex": int(index.item()),
            "beforeLogit": float(baseline_logits[index].item()),
            "afterLogit": float(intervened_logits[index].item()),
            "deltaLogit": float(delta[index].item()),
            "absoluteDelta": float(delta[index].abs().item()),
        }
        for index in indices
    ]


def summarize_intervention_results(
    intervention_summary: Mapping,
    baseline_logits: torch.Tensor | Sequence,
    intervened_logits: torch.Tensor | Sequence,
    target_token_ids: Sequence[int] | None = None,
    top_k: int = 10,
    metadata: Mapping | None = None,
) -> dict:
    """Return a JSON-safe summary of an executed feature intervention.

    ``summarize_interventions`` records an intervention plan. This function
    records observed logit effects after that plan has been executed, preserving
    enough information for another system to compare whether the same causal
    test had a similar effect.
    """
    before = _final_logits(baseline_logits)
    after = _final_logits(intervened_logits)

    if before.shape != after.shape:
        raise ValueError(
            "Baseline and intervened logits must have the same final dimension"
        )

    target_ids = [int(token_id) for token_id in (target_token_ids or [])]
    delta = after - before
    top_before = int(before.argmax().item())
    top_after = int(after.argmax().item())

    return {
        "kind": INTERVENTION_RESULT_SUMMARY_KIND,
        "version": 1,
        "sourcePlan": {
            "kind": intervention_summary.get("kind"),
            "sourceGraph": copy.deepcopy(intervention_summary.get("sourceGraph")),
            "interventionType": intervention_summary.get("interventionType"),
            "value": intervention_summary.get("value"),
            "interventionCount": intervention_summary.get("interventionCount"),
            "interventions": copy.deepcopy(
                intervention_summary.get("interventions", [])
            ),
        },
        "runtime": {
            "function": "ReplacementModel.feature_intervention",
            "executed": True,
        },
        "effects": {
            "vocabSize": int(before.numel()),
            "topBefore": {
                "vocabIndex": top_before,
                "logit": float(before[top_before].item()),
            },
            "topAfter": {
                "vocabIndex": top_after,
                "logit": float(after[top_after].item()),
            },
            "topTokenChanged": top_before != top_after,
            "maxAbsLogitDelta": float(delta.abs().max().item()),
            "meanAbsLogitDelta": float(delta.abs().mean().item()),
            "l2LogitDelta": float(torch.linalg.vector_norm(delta).item()),
            "targetEffects": _token_logit_effects(before, after, target_ids),
            "topLogitShifts": _top_logit_shifts(before, after, top_k),
        },
        "metadata": copy.deepcopy(metadata) if metadata else {},
    }


# ── Batch comparison ────────────────────────────────────────────────


class ComparisonResult(NamedTuple):
    """Result of comparing multiple attribution graphs.

    Attributes:
        shared_features: Features appearing in the top-*n* of **every** graph.
        per_graph_features: Top features for each graph, in input order.
        feature_frequency: Mapping from ``(layer, pos, feature_idx)`` to the
            number of graphs in which the feature appeared in the top-*n*.
        graph_scores: ``(replacement_score, completeness_score)`` for each graph.
    """

    shared_features: list[tuple[int, int, int]]
    per_graph_features: list[list[tuple[int, int, int]]]
    feature_frequency: dict[tuple[int, int, int], int]
    graph_scores: list[tuple[float, float]]


def compare_graphs(
    graphs: Sequence[Graph],
    n_per_graph: int = 20,
) -> ComparisonResult:
    """Compare multiple attribution graphs, finding shared and unique features.

    For each graph, extracts the top-*n_per_graph* features by influence,
    then computes overlap statistics.

    Args:
        graphs: Sequence of :class:`~circuit_tracer.graph.Graph` objects.
        n_per_graph: Number of top features to consider per graph.

    Returns:
        A :class:`ComparisonResult` containing shared features, per-graph
        features, frequency counts, and quality scores.
    """
    per_graph: list[list[tuple[int, int, int]]] = []
    frequency: Counter[tuple[int, int, int]] = Counter()

    for graph in graphs:
        features, _ = get_top_features(graph, n_per_graph)
        per_graph.append(features)
        frequency.update(features)

    # Features present in every graph
    n_graphs = len(graphs)
    shared = [feat for feat, count in frequency.items() if count == n_graphs]

    scores = [compute_graph_scores(graph) for graph in graphs]

    return ComparisonResult(
        shared_features=shared,
        per_graph_features=per_graph,
        feature_frequency=dict(frequency),
        graph_scores=scores,
    )


def find_common_circuit(
    graphs: Sequence[Graph],
    min_frequency: float = 0.5,
    n_per_graph: int = 20,
) -> list[tuple[int, int, int]]:
    """Find features appearing across a minimum fraction of graphs.

    Useful for identifying circuit motifs that are consistent across
    different inputs — the core question in mechanistic interpretability
    ("Does the same circuit fire for all instances of this task?").

    Args:
        graphs: Sequence of :class:`~circuit_tracer.graph.Graph` objects.
        min_frequency: Minimum fraction of graphs a feature must appear in
            (0.0–1.0).
        n_per_graph: Number of top features to consider per graph.

    Returns:
        List of ``(layer, position, feature_idx)`` tuples meeting the
        frequency threshold, sorted by descending frequency.
    """
    result = compare_graphs(graphs, n_per_graph)
    threshold = min_frequency * len(graphs)

    common = [
        (feat, count) for feat, count in result.feature_frequency.items() if count >= threshold
    ]
    # Sort by frequency (descending), then by layer/position for stability
    common.sort(key=lambda x: (-x[1], x[0]))
    return [feat for feat, _ in common]
