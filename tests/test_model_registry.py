"""Tests for model registry public API (register_model, get_supported_architectures, etc.)."""

import pytest

from circuit_tracer.utils.tl_nnsight_mapping import (
    _REGISTRY,
    ModelMapping,
    auto_detect_mapping,
    get_mapping,
    get_supported_architectures,
    register_model,
)


class TestGetSupportedArchitectures:
    def test_returns_known_architectures(self):
        archs = get_supported_architectures()
        assert isinstance(archs, list)
        assert "Gemma2ForCausalLM" in archs
        assert "LlamaForCausalLM" in archs

    def test_returns_all_registered(self):
        archs = get_supported_architectures()
        assert len(archs) == len(_REGISTRY)


class TestRegisterModel:
    def test_register_new_architecture(self):
        mapping = ModelMapping(
            model_architecture="TestModelForCausalLM",
            attention_location_pattern="model.layers[{layer}].self_attn",
            layernorm_scale_location_patterns=["model.layers[{layer}].ln"],
            pre_logit_location="model",
            embed_location="model.embed_tokens",
            embed_weight="model.embed_tokens.weight",
            unembed_weight="lm_head.weight",
        )

        try:
            register_model(mapping)
            assert "TestModelForCausalLM" in get_supported_architectures()
            assert get_mapping("TestModelForCausalLM") == mapping
        finally:
            # Clean up to avoid polluting other tests
            _REGISTRY.pop("TestModelForCausalLM", None)

    def test_register_overwrites_existing(self):
        original = get_mapping("Gemma2ForCausalLM")
        replacement = ModelMapping(
            model_architecture="Gemma2ForCausalLM",
            attention_location_pattern="custom_path",
            layernorm_scale_location_patterns=[],
            pre_logit_location="model",
            embed_location="model.embed_tokens",
            embed_weight="model.embed_tokens.weight",
            unembed_weight="lm_head.weight",
        )

        try:
            register_model(replacement)
            assert get_mapping("Gemma2ForCausalLM").attention_location_pattern == "custom_path"
        finally:
            # Restore original
            _REGISTRY["Gemma2ForCausalLM"] = original


class TestAutoDetectMapping:
    @pytest.mark.requires_disk
    def test_detect_gemma(self):
        """Requires network access to download model config."""
        result = auto_detect_mapping("google/gemma-2-2b")
        assert result is not None
        assert result.model_architecture == "Gemma2ForCausalLM"

    @pytest.mark.requires_disk
    def test_detect_unsupported_returns_none(self):
        """An architecture not in the registry should return None."""
        result = auto_detect_mapping("bert-base-uncased")
        assert result is None
