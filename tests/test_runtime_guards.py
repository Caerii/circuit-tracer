"""Tests for mapping validation, logit-cache init, and TRACE_CACHING."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from circuit_tracer.utils.tl_nnsight_mapping import ModelMapping, get_mapping, validate_mapping


def test_validate_mapping_resolves_simple_tree():
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(drop=object()),
        ln=SimpleNamespace(scale=object()),
        pre_feedforward_layernorm=object(),
        post_feedforward_layernorm=object(),
        mlp=object(),
    )
    model = SimpleNamespace(
        layers=[layer, layer],
        embed_tokens=SimpleNamespace(weight=torch.zeros(2, 2)),
        config=SimpleNamespace(num_hidden_layers=2, architectures=["ToyForCausalLM"]),
    )
    model.lm_head = SimpleNamespace(weight=torch.zeros(2, 2))
    # Attach lm_head at root for unembed_weight path
    mapping = ModelMapping(
        model_architecture="ToyForCausalLM",
        attention_location_pattern="layers[{layer}].self_attn.drop",
        layernorm_scale_location_patterns=["layers[{layer}].ln.scale"],
        pre_logit_location="layers",
        embed_location="embed_tokens",
        embed_weight="embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "hook_resid_mid": ("layers[{layer}].pre_feedforward_layernorm", "input"),
            "hook_mlp_out": ("layers[{layer}].post_feedforward_layernorm", "output"),
        },
    )
    warnings = validate_mapping(mapping, model=model, n_layers=2)
    assert warnings == []


def test_validate_mapping_reports_missing_path():
    model = SimpleNamespace(
        layers=[SimpleNamespace()],
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
        lm_head=SimpleNamespace(weight=torch.zeros(1)),
        config=SimpleNamespace(num_hidden_layers=1),
    )
    mapping = ModelMapping(
        model_architecture="ToyForCausalLM",
        attention_location_pattern="layers[{layer}].missing_attn",
        layernorm_scale_location_patterns=[],
        pre_logit_location="layers",
        embed_location="embed_tokens",
        embed_weight="embed_tokens.weight",
        unembed_weight="lm_head.weight",
    )
    warnings = validate_mapping(mapping, model=model, n_layers=1)
    assert any("attention_location_pattern" in w for w in warnings)


def test_gemma2_mapping_registered():
    mapping = get_mapping("Gemma2ForCausalLM")
    assert "hook_mlp_out" in mapping.feature_hook_mapping


def test_nnsight_replacement_module_imports_cleanly():
    import circuit_tracer.replacement_model.replacement_model_nnsight as nnsight_rm

    assert nnsight_rm.NNSIGHT_CONFIG.APP.PYMOUNT is False
    assert nnsight_rm.NNSIGHT_CONFIG.APP.CROSS_INVOKER is False


def test_logit_cache_always_starts_empty_and_appends():
    """Regression: feature_intervention_generate indexes logit_cache[0]."""
    cached_logits: list[torch.Tensor] = []
    cached_logits.append(torch.zeros(1, 4))
    assert cached_logits[0].shape[-1] == 4

    # Source-level guard: init must not seed [None]
    from pathlib import Path

    source = Path(
        "circuit_tracer/replacement_model/replacement_model_transformerlens.py"
    ).read_text(encoding="utf-8")
    assert "cached_logits: list[torch.Tensor] = []" in source
    assert "cached_logits = [None]" not in source
    assert "[] if using_past_kv_cache else [None]" not in source
