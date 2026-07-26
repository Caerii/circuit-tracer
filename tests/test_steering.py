"""Tests for steering / intervention plan helpers (no GPU required)."""

from __future__ import annotations

import pytest
import torch
from transformer_lens import HookedTransformerConfig

from circuit_tracer.analysis import summarize_interventions
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph
from circuit_tracer.steering import (
    VALIDATION_KIND,
    interventions_from_plan,
    load_intervention_plan,
    save_intervention_plan,
    steer,
    validate_intervention,
)
from circuit_tracer.utils import get_default_device


def _make_graph() -> Graph:
    n_features, n_tokens, n_logits, n_layers = 3, 2, 1, 2
    n_errors = n_layers * n_tokens
    n_total = n_features + n_errors + n_tokens + n_logits
    adj = torch.zeros(n_total, n_total)
    adj[-1, 0] = 1.0
    adj[-1, 1] = 0.5
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
        input_string="hi",
        input_tokens=torch.arange(n_tokens),
        active_features=torch.tensor([(0, 0, 10), (1, 1, 11), (0, 1, 12)]),
        adjacency_matrix=adj,
        cfg=cfg,
        logit_targets=[LogitTarget(token_str="a", vocab_idx=1)],
        logit_probabilities=torch.ones(1),
        selected_features=torch.arange(n_features),
        activation_values=torch.tensor([2.0, 1.0, 0.5]),
    )


class _FakeModel:
    def __call__(self, prompt):
        return torch.zeros(1, 2, 8)

    def feature_intervention(self, prompt, interventions, **kwargs):
        logits = torch.zeros(1, 2, 8)
        logits[0, -1, 1] = float(len(interventions))
        return logits, None

    def feature_intervention_generate(self, prompt, interventions, **kwargs):
        logits = torch.zeros(3, 8)
        return "generated", logits, None


def test_interventions_from_plan_round_trip(tmp_path):
    plan = summarize_interventions(_make_graph(), n=2, value=1.5)
    path = tmp_path / "plan.json"
    save_intervention_plan(plan, path)
    loaded = load_intervention_plan(path)
    tuples = interventions_from_plan(loaded)
    assert len(tuples) == 2
    assert tuples[0][3] == 1.5


def test_steer_builds_amplify_and_suppress():
    model = _FakeModel()
    logits, acts = steer(
        model,
        "hello",
        amplify=[(1, 0, 5)],
        suppress=[(2, 1, 9)],
        amplify_value=3.0,
    )
    assert acts is None
    assert logits.shape[-1] == 8

    text, gen_logits, _ = steer(
        model,
        "hello",
        suppress=[(2, 1, 9)],
        generate=True,
    )
    assert text == "generated"
    assert gen_logits.ndim == 2


def test_steer_requires_actions():
    with pytest.raises(ValueError, match="amplify or suppress"):
        steer(_FakeModel(), "x")


def test_validate_intervention_observe_mode():
    report = validate_intervention(
        _FakeModel(),
        "hello",
        interventions=[(0, 0, 1, 0.0)],
        target_token_ids=[1],
        mode="observe",
    )
    assert report["kind"] == VALIDATION_KIND
    assert report["passed"] is True
    assert report["effects"]["targetEffects"][0]["vocabIndex"] == 1
