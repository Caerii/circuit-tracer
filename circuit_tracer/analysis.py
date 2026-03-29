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

from collections import Counter
from collections.abc import Sequence
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
    "graph_to_interventions",
    "compare_graphs",
    "find_common_circuit",
    "ComparisonResult",
]


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
