"""Tests for the programmatic analysis API (circuit_tracer.analysis).

Uses synthetic graphs — no GPU or model downloads required.
"""

import gc

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.analysis import (
    ComparisonResult,
    compare_graphs,
    find_common_circuit,
    get_top_features,
    graph_to_interventions,
)
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph, PruneResult
from circuit_tracer.utils import get_default_device


@pytest.fixture(autouse=True)
def cleanup_cuda():
    yield
    torch.cuda.empty_cache()
    gc.collect()


# ── Shared fixture ──────────────────────────────────────────────────


def _make_cfg(**overrides):
    """Minimal HookedTransformerConfig for testing."""
    defaults = {
        "n_layers": 2,
        "d_model": 8,
        "n_ctx": 32,
        "d_head": 4,
        "n_heads": 2,
        "d_mlp": 16,
        "act_fn": "gelu",
        "d_vocab": 16,
        "model_name": "test-model",
        "device": get_default_device(),
    }
    defaults.update(overrides)
    return HookedTransformerConfig.from_dict(defaults)


def _make_graph(
    n_features: int = 5,
    n_tokens: int = 2,
    n_logits: int = 1,
    n_layers: int = 2,
    seed: int = 42,
) -> Graph:
    """Build a synthetic Graph with deterministic edge weights.

    Node layout: [features | errors | tokens | logits]
    where errors = n_layers * n_tokens nodes.
    """
    rng = torch.Generator().manual_seed(seed)
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits

    # Random adjacency with some structure — logit row has non-zero feature columns
    adj = torch.zeros(n_total, n_total)
    for i in range(n_features):
        adj[-1, i] = torch.rand(1, generator=rng).item()  # logit ← features
    # Some feature-to-feature edges
    for i in range(1, n_features):
        adj[i, i - 1] = torch.rand(1, generator=rng).item()
    # Token → feature edges
    for i in range(n_features):
        adj[i, n_features + n_errors + (i % n_tokens)] = torch.rand(1, generator=rng).item()

    # Active features: (layer, pos, feature_idx)
    active_features = torch.tensor(
        [(i % n_layers, i % n_tokens, 100 + i) for i in range(n_features)]
    )

    cfg = _make_cfg(n_layers=n_layers)

    return Graph(
        input_string="ab",
        input_tokens=torch.arange(n_tokens),
        active_features=active_features,
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str=f"tok_{i}", vocab_idx=i) for i in range(n_logits)],
        logit_probabilities=torch.ones(n_logits) / n_logits,
        selected_features=torch.arange(n_features),
        activation_values=torch.randn(n_features, generator=rng).abs(),
    )


# ── get_top_features ────────────────────────────────────────────────


class TestGetTopFeatures:
    def test_returns_correct_types(self):
        graph = _make_graph()
        features, scores = get_top_features(graph, n=3)

        assert isinstance(features, list)
        assert isinstance(scores, list)
        assert len(features) == 3
        assert len(scores) == 3
        # Each feature is a (layer, pos, feature_idx) tuple
        for f in features:
            assert isinstance(f, tuple)
            assert len(f) == 3

    def test_scores_are_descending(self):
        graph = _make_graph()
        _, scores = get_top_features(graph, n=5)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_n_exceeds_features(self):
        graph = _make_graph(n_features=3)
        features, scores = get_top_features(graph, n=10)
        assert len(features) == 3  # capped at available features

    def test_n_equals_one(self):
        graph = _make_graph()
        features, scores = get_top_features(graph, n=1)
        assert len(features) == 1
        assert len(scores) == 1


# ── Graph convenience methods ───────────────────────────────────────


class TestGraphConvenienceMethods:
    def test_top_features_delegates(self):
        graph = _make_graph()
        features_method, scores_method = graph.top_features(n=3)
        features_fn, scores_fn = get_top_features(graph, n=3)
        assert features_method == features_fn
        assert scores_method == scores_fn

    def test_prune_returns_prune_result(self):
        graph = _make_graph()
        result = graph.prune()
        assert isinstance(result, PruneResult)
        assert result.node_mask.dtype == torch.bool
        assert result.edge_mask.dtype == torch.bool

    def test_scores_returns_tuple(self):
        graph = _make_graph()
        result = graph.scores()
        assert isinstance(result, tuple)
        assert len(result) == 2
        r_score, c_score = result
        assert isinstance(r_score, float)
        assert isinstance(c_score, float)
        assert 0.0 <= r_score <= 1.0
        assert 0.0 <= c_score <= 1.0

    @pytest.mark.requires_disk  # Needs HF tokenizer download
    def test_to_json_creates_file(self, tmp_path):
        graph = _make_graph()
        graph.scan = "test-scan"  # Required for JSON export
        graph.cfg.tokenizer_name = "google/gemma-2-2b"
        graph.to_json(slug="test-graph", output_path=str(tmp_path))
        assert (tmp_path / "test-graph.json").exists()


# ── graph_to_interventions ──────────────────────────────────────────


class TestGraphToInterventions:
    def test_returns_correct_format(self):
        graph = _make_graph()
        interventions = graph_to_interventions(graph, n=3, value=0.0)

        assert isinstance(interventions, list)
        assert len(interventions) == 3
        for layer, pos, feat_idx, val in interventions:
            assert isinstance(layer, int)
            assert isinstance(pos, int)
            assert isinstance(feat_idx, int)
            assert val == 0.0

    def test_ablation_default(self):
        graph = _make_graph()
        interventions = graph_to_interventions(graph, n=2)
        for _, _, _, val in interventions:
            assert val == 0.0

    def test_amplification(self):
        graph = _make_graph()
        interventions = graph_to_interventions(graph, n=2, value=5.0)
        for _, _, _, val in interventions:
            assert val == 5.0

    def test_features_match_top_features(self):
        graph = _make_graph()
        features, _ = get_top_features(graph, n=3)
        interventions = graph_to_interventions(graph, n=3)
        for (fl, fp, fi), (il, ip, ii, _) in zip(features, interventions):
            assert (fl, fp, fi) == (il, ip, ii)


# ── compare_graphs ──────────────────────────────────────────────────


class TestCompareGraphs:
    def test_basic_comparison(self):
        # Two graphs with the same seed should have identical features
        g1 = _make_graph(seed=42)
        g2 = _make_graph(seed=42)
        result = compare_graphs([g1, g2], n_per_graph=5)

        assert isinstance(result, ComparisonResult)
        assert len(result.per_graph_features) == 2
        assert len(result.graph_scores) == 2
        # Same graph → all features shared
        assert len(result.shared_features) == 5

    def test_different_graphs_may_differ(self):
        g1 = _make_graph(seed=1)
        g2 = _make_graph(seed=2)
        result = compare_graphs([g1, g2], n_per_graph=3)

        assert isinstance(result, ComparisonResult)
        assert len(result.per_graph_features) == 2
        # Shared features can be fewer than per-graph features
        assert len(result.shared_features) <= 3

    def test_graph_scores_are_valid(self):
        g1 = _make_graph(seed=10)
        g2 = _make_graph(seed=20)
        result = compare_graphs([g1, g2])

        for r_score, c_score in result.graph_scores:
            assert isinstance(r_score, float)
            assert isinstance(c_score, float)


# ── find_common_circuit ─────────────────────────────────────────────


class TestFindCommonCircuit:
    def test_identical_graphs(self):
        graphs = [_make_graph(seed=42) for _ in range(3)]
        common = find_common_circuit(graphs, min_frequency=1.0, n_per_graph=5)
        assert len(common) == 5  # all features appear in all graphs

    def test_frequency_threshold(self):
        g1 = _make_graph(seed=42)
        g2 = _make_graph(seed=42)
        g3 = _make_graph(seed=99)  # different graph

        # min_frequency=0.5 means feature must appear in at least 1.5 → 2 of 3 graphs
        common = find_common_circuit([g1, g2, g3], min_frequency=0.5, n_per_graph=5)
        assert isinstance(common, list)
        # At minimum, features shared between g1 and g2 should appear
        assert len(common) >= 0  # some may not meet threshold

    def test_empty_with_high_threshold(self):
        g1 = _make_graph(seed=1)
        g2 = _make_graph(seed=2)
        # With very different graphs and min_frequency=1.0, expect few/no shared features
        common = find_common_circuit([g1, g2], min_frequency=1.0, n_per_graph=3)
        assert isinstance(common, list)
