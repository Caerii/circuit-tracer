"""CLI subcommand wiring for analysis JSON parity."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer import __main__ as cli
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.utils import get_default_device


def _make_graph(path: Path) -> Path:
    n_features, n_tokens, n_logits, n_layers = 3, 2, 1, 2
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits
    adj = torch.zeros(n_total, n_total)
    adj[-1, 0] = 1.0
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
            "tokenizer_name": "hf-internal-testing/tiny-random-gpt2",
            "device": get_default_device(),
        }
    )
    graph = Graph(
        input_string="hi",
        input_tokens=torch.arange(n_tokens),
        active_features=torch.tensor([(0, 0, 10), (1, 1, 11), (0, 1, 12)]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="a", vocab_idx=1)],
        logit_probabilities=torch.ones(1),
        selected_features=torch.arange(n_features),
        activation_values=torch.ones(n_features),
        scan_name="test-scan",
    )
    graph.to_pt(str(path))
    return path


def test_cli_summarize_and_interventions(tmp_path, capsys):
    graph_path = _make_graph(tmp_path / "g.pt")
    out = tmp_path / "summary.json"
    cli.run_summarize(
        type(
            "Args",
            (),
            {
                "graph": str(graph_path),
                "output": str(out),
                "top_n": 2,
                "node_threshold": 0.8,
                "edge_threshold": 0.98,
                "no_pruning": True,
            },
        )()
    )
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["kind"] == "circuit-tracer.summary.v1"

    plan_out = tmp_path / "plan.json"
    cli.run_interventions(
        type(
            "Args",
            (),
            {
                "graph": str(graph_path),
                "output": str(plan_out),
                "n": 2,
                "value": 0.0,
            },
        )()
    )
    plan = json.loads(plan_out.read_text(encoding="utf-8"))
    assert plan["kind"] == "circuit-tracer.interventions.v1"
