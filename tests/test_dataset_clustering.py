"""CPU tests for circuit clustering and dataset summaries."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.dataset import (
    CircuitDataset,
    CircuitRecord,
    cluster_circuits,
    summarize_dataset,
)
from circuit_tracer.graph import Graph
from circuit_tracer.utils import get_default_device


def _make_graph(seed: int, feature_ids: list[int]) -> Graph:
    n_features = len(feature_ids)
    n_tokens, n_logits, n_layers = 2, 1, 2
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits
    adj = torch.zeros(n_total, n_total)
    for i in range(n_features):
        adj[-1, i] = 1.0 - 0.05 * i
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": n_layers,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 8,
            "model_name": "test-model",
            "device": get_default_device(),
        }
    )
    return Graph(
        input_string=f"p{seed}",
        input_tokens=torch.arange(n_tokens),
        active_features=torch.tensor([(0, 0, fid) for fid in feature_ids]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="a", vocab_idx=1)],
        logit_probabilities=torch.ones(1),
        selected_features=torch.arange(n_features),
        activation_values=torch.ones(n_features),
    )


def test_cluster_circuits_groups_similar_graphs():
    ds = CircuitDataset(
        [
            CircuitRecord(prompt="a", graph=_make_graph(1, [10, 11, 12]), label="x"),
            CircuitRecord(prompt="b", graph=_make_graph(2, [10, 11, 13]), label="x"),
            CircuitRecord(prompt="c", graph=_make_graph(3, [50, 51, 52]), label="y"),
        ]
    )
    result = cluster_circuits(ds, n_per_graph=3, method="jaccard", threshold=0.6)
    assert result.to_dict()["kind"] == "circuit-tracer.circuit-clusters.v1"
    assert len(result.labels) == 3
    # First two should share a cluster under a moderate Jaccard threshold
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] != result.labels[0]


def test_summarize_dataset_bootstrap_and_labels():
    ds = CircuitDataset(
        [
            CircuitRecord(prompt="a", graph=_make_graph(1, [10, 11]), label="x"),
            CircuitRecord(prompt="b", graph=_make_graph(2, [10, 12]), label="x"),
            CircuitRecord(prompt="c", graph=_make_graph(3, [20, 21]), label="y"),
        ]
    )
    summary = summarize_dataset(ds, n_per_graph=2, bootstrap=50, seed=0)
    payload = summary.to_dict()
    assert payload["kind"] == "circuit-tracer.dataset-summary.v1"
    assert summary.n_graphs == 3
    assert (0, 0, 10) in summary.feature_frequency
    assert "x" in summary.per_label
    assert summary.mean_replacement_score >= 0.0
