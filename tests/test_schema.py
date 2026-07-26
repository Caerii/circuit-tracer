"""Golden JSON fixtures + schema validation for interchange documents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.analysis import (
    CIRCUIT_SUMMARY_KIND,
    INTERVENTION_RESULT_SUMMARY_KIND,
    INTERVENTION_SUMMARY_KIND,
    summarize_graph,
    summarize_intervention_results,
    summarize_interventions,
)
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.schema import SchemaError, dump_summary, load_summary, validate_summary
from circuit_tracer.utils import get_default_device

FIXTURES = Path(__file__).parent / "fixtures" / "summaries"


def _make_graph() -> Graph:
    n_features, n_tokens, n_logits, n_layers = 4, 2, 1, 2
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits
    adj = torch.zeros(n_total, n_total)
    for i in range(n_features):
        adj[-1, i] = 0.5 + 0.1 * i
        adj[i, n_features + n_errors + (i % n_tokens)] = 0.2
    cfg = HookedTransformerConfig.from_dict(
        {
            "n_layers": n_layers,
            "d_model": 8,
            "n_ctx": 32,
            "d_head": 4,
            "n_heads": 2,
            "d_mlp": 16,
            "act_fn": "gelu",
            "d_vocab": 16,
            "model_name": "test-model",
            "tokenizer_name": "test-tokenizer",
            "device": get_default_device(),
        }
    )
    return Graph(
        input_string="ab",
        input_tokens=torch.arange(n_tokens),
        active_features=torch.tensor(
            [(i % n_layers, i % n_tokens, 100 + i) for i in range(n_features)]
        ),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="tok_0", vocab_idx=0)],
        logit_probabilities=torch.ones(n_logits),
        selected_features=torch.arange(n_features),
        activation_values=torch.arange(1, n_features + 1, dtype=torch.float32),
        scan_name="test-scan",
    )


def test_summarize_graph_matches_schema_and_fixture(tmp_path):
    graph = _make_graph()
    summary = summarize_graph(graph, top_n=3, node_threshold=None, edge_threshold=None)
    assert validate_summary(summary) == CIRCUIT_SUMMARY_KIND

    fixture_path = FIXTURES / "circuit_summary.v1.json"
    if not fixture_path.exists():
        dump_summary(summary, fixture_path)
    expected = load_summary(fixture_path, expected_kind=CIRCUIT_SUMMARY_KIND)
    # Stable fields only — scores may float-differ slightly across platforms
    assert summary["kind"] == expected["kind"]
    assert summary["input"] == expected["input"]
    assert summary["model"] == expected["model"]
    assert [f["feature"] for f in summary["topFeatures"]] == [
        f["feature"] for f in expected["topFeatures"]
    ]

    out = tmp_path / "summary.json"
    dump_summary(summary, out)
    assert load_summary(out)["kind"] == CIRCUIT_SUMMARY_KIND


def test_intervention_plan_round_trip(tmp_path):
    graph = _make_graph()
    plan = summarize_interventions(graph, n=2, value=0.0)
    assert validate_summary(plan) == INTERVENTION_SUMMARY_KIND
    path = tmp_path / "plan.json"
    dump_summary(plan, path)
    loaded = load_summary(path, expected_kind=INTERVENTION_SUMMARY_KIND)
    assert loaded["interventionCount"] == 2
    assert loaded["interventions"][0]["value"] == 0.0


def test_intervention_results_schema():
    graph = _make_graph()
    plan = summarize_interventions(graph, n=1, value=0.0)
    baseline = torch.zeros(16)
    baseline[0] = 2.0
    intervened = baseline.clone()
    intervened[0] = 1.0
    intervened[3] = 0.5
    result = summarize_intervention_results(plan, baseline, intervened, target_token_ids=[0])
    assert validate_summary(result) == INTERVENTION_RESULT_SUMMARY_KIND
    assert "maxAbsLogitDelta" in result["effects"]


def test_schema_rejects_bad_kind():
    with pytest.raises(SchemaError):
        validate_summary({"kind": "not-a-real-kind", "version": 1})


def test_fixture_files_are_valid():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    # Ensure at least the generated circuit summary fixture validates when present
    for path in FIXTURES.glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        validate_summary(doc)
