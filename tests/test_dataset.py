"""Tests for CircuitDataset save/load and drift comparison."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.dataset import CircuitDataset, CircuitRecord, compare_datasets
from circuit_tracer.graph import Graph
from circuit_tracer.utils import get_default_device


def _make_graph(seed: int = 0) -> Graph:
    n_features, n_tokens, n_logits, n_layers = 3, 2, 1, 2
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits
    rng = torch.Generator().manual_seed(seed)
    adj = torch.zeros(n_total, n_total)
    for i in range(n_features):
        adj[-1, i] = torch.rand(1, generator=rng).item()
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
        active_features=torch.tensor([(0, 0, 10 + seed), (1, 1, 11), (0, 1, 12)]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="a", vocab_idx=1)],
        logit_probabilities=torch.ones(1),
        selected_features=torch.arange(n_features),
        activation_values=torch.ones(n_features),
    )


def test_dataset_save_load_round_trip(tmp_path):
    ds = CircuitDataset(
        [
            CircuitRecord(prompt="alpha", graph=_make_graph(1), label="a"),
            CircuitRecord(prompt="beta", graph=_make_graph(2), label="b", metadata={"k": 1}),
        ]
    )
    ds.save(tmp_path / "ds")
    loaded = CircuitDataset.load(tmp_path / "ds")
    assert len(loaded) == 2
    assert loaded.prompts == ["alpha", "beta"]
    assert loaded.records[0].label == "a"
    assert loaded.records[1].metadata["k"] == 1
    assert loaded.graphs[0].input_string == "p1"


def test_compare_datasets_paired():
    g1 = _make_graph(1)
    g2 = _make_graph(1)
    baseline = CircuitDataset(
        [
            CircuitRecord(prompt="same", graph=g1),
            CircuitRecord(prompt="only-base", graph=_make_graph(3)),
        ]
    )
    current = CircuitDataset(
        [
            CircuitRecord(prompt="same", graph=g2),
            CircuitRecord(prompt="only-cur", graph=_make_graph(4)),
        ]
    )
    drift = compare_datasets(baseline, current, n_per_graph=3)
    assert drift.paired is True
    assert drift.n_baseline == 1
    assert drift.n_current == 1
    payload = drift.to_dict()
    assert payload["kind"] == "circuit-tracer.dataset-drift.v1"
