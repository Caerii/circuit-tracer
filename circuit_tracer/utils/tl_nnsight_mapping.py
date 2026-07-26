from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelMapping:
    """Mapping between HuggingFace model internals and circuit-tracer hook points.

    Each field specifies where to find key model components in NNSight's
    traced computation graph.  Patterns containing ``{layer}`` are expanded
    per-layer at runtime.

    To add support for a new architecture, create a ``ModelMapping`` and
    register it with :func:`register_model`::

        from circuit_tracer import ModelMapping, register_model

        register_model(ModelMapping(
            model_architecture="MistralForCausalLM",
            attention_location_pattern="model.layers[{layer}].self_attn...",
            layernorm_scale_location_patterns=[...],
            pre_logit_location="model",
            embed_location="model.embed_tokens",
            embed_weight="model.embed_tokens.weight",
            unembed_weight="lm_head.weight",
            feature_hook_mapping={
                "hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
                "hook_mlp_out": ("model.layers[{layer}].mlp", "output"),
            },
        ))

    See the existing Gemma-2 mapping below for a complete, annotated example.

    Attributes:
        model_architecture: Exact HuggingFace architecture class name
            (e.g. ``"Gemma2ForCausalLM"``).  Must match
            ``AutoConfig.from_pretrained(model).architectures[0]``.
        attention_location_pattern: NNSight envoy path to the attention
            dropout node, with ``{layer}`` placeholder.
        layernorm_scale_location_patterns: Paths to each LayerNorm scale
            operation that should be frozen during attribution.
        pre_logit_location: Path to the module whose output is projected
            to logits (usually ``"model"``).
        embed_location: Path to the token embedding module.
        embed_weight: Path to the embedding weight tensor.
        unembed_weight: Path to the unembedding (lm_head) weight tensor.
        feature_hook_mapping: Maps TransformerLens hook names to
            ``(nnsight_path, "input"|"output")`` tuples, specifying where
            transcoder features read from and write to.
    """

    model_architecture: str
    attention_location_pattern: str
    layernorm_scale_location_patterns: list[str]
    pre_logit_location: str
    embed_location: str
    embed_weight: str
    unembed_weight: str
    feature_hook_mapping: dict[str, tuple[str, Literal["input", "output"]]] = field(
        default_factory=dict
    )


# ── Model mapping registry ───────────────────────────────────────────
# Each entry maps a HuggingFace architecture class name to its NNSight
# location patterns.  New architectures can be added by appending here.

_REGISTRY: dict[str, ModelMapping] = {}


def _register(mapping: ModelMapping) -> ModelMapping:
    """Register a mapping in the global registry and return it."""
    _REGISTRY[mapping.model_architecture] = mapping
    return mapping


_register(
    ModelMapping(
        model_architecture="Gemma2ForCausalLM",
        attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.norm.source.self__norm_0.source.torch_rsqrt_0",
        ],
        pre_logit_location="model",
        embed_location="model.embed_tokens",
        embed_weight="model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "ln2.hook_normalized": (
                "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
                "output",
            ),
            "hook_resid_mid": ("model.layers[{layer}].pre_feedforward_layernorm", "input"),
            "hook_mlp_out": ("model.layers[{layer}].post_feedforward_layernorm", "output"),
        },
    )
)

_register(
    ModelMapping(
        model_architecture="Gemma3ForCausalLM",
        attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].self_attn.q_norm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].self_attn.k_norm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "model.norm.source.self__norm_0.source.torch_rsqrt_0",
        ],
        pre_logit_location="model",
        embed_location="model.embed_tokens",
        embed_weight="model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "ln2.hook_normalized": (
                "model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
                "output",
            ),
            "hook_resid_mid": ("model.layers[{layer}].pre_feedforward_layernorm", "input"),
            "mlp.hook_in": ("model.layers[{layer}].pre_feedforward_layernorm", "output"),
            "hook_mlp_out": ("model.layers[{layer}].post_feedforward_layernorm", "output"),
        },
    )
)

_register(
    ModelMapping(
        model_architecture="Gemma3ForConditionalGeneration",
        attention_location_pattern="language_model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "language_model.layers[{layer}].input_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.layers[{layer}].self_attn.q_norm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.layers[{layer}].self_attn.k_norm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.layers[{layer}].post_attention_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.layers[{layer}].post_feedforward_layernorm.source.self__norm_0.source.torch_rsqrt_0",
            "language_model.norm.source.self__norm_0.source.torch_rsqrt_0",
        ],
        pre_logit_location="language_model",
        embed_location="language_model.embed_tokens",
        embed_weight="language_model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "ln2.hook_normalized": (
                "language_model.layers[{layer}].pre_feedforward_layernorm.source.self__norm_0",
                "output",
            ),
            "hook_resid_mid": ("language_model.layers[{layer}].pre_feedforward_layernorm", "input"),
            "mlp.hook_in": ("language_model.layers[{layer}].pre_feedforward_layernorm", "output"),
            "hook_mlp_out": ("language_model.layers[{layer}].post_feedforward_layernorm", "output"),
        },
    )
)

_register(
    ModelMapping(
        model_architecture="LlamaForCausalLM",
        attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "model.layers[{layer}].input_layernorm.source.mean_0",
            "model.layers[{layer}].post_attention_layernorm.source.mean_0",
            "model.norm.source.mean_0",
        ],
        pre_logit_location="model",
        embed_location="model.embed_tokens",
        embed_weight="model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
            "hook_mlp_out": ("model.layers[{layer}].mlp", "output"),
            "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
            "mlp.hook_out": ("model.layers[{layer}].mlp", "output"),
        },
    )
)

_register(
    ModelMapping(
        model_architecture="Qwen3ForCausalLM",
        attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "model.layers[{layer}].input_layernorm.source.mean_0",
            "model.layers[{layer}].post_attention_layernorm.source.mean_0",
            "model.norm.source.mean_0",
        ],
        pre_logit_location="model",
        embed_location="model.embed_tokens",
        embed_weight="model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
            "mlp.hook_out": ("model.layers[{layer}].mlp", "output"),
        },
    )
)

_register(
    ModelMapping(
        model_architecture="GptOssForCausalLM",
        attention_location_pattern="model.layers[{layer}].self_attn.source.attention_interface_0.source.nn_functional_dropout_0",
        layernorm_scale_location_patterns=[
            "model.layers[{layer}].input_layernorm.source.mean_0",
            "model.layers[{layer}].post_attention_layernorm.source.mean_0",
            "model.norm.source.mean_0",
        ],
        pre_logit_location="model",
        embed_location="model.embed_tokens",
        embed_weight="model.embed_tokens.weight",
        unembed_weight="lm_head.weight",
        feature_hook_mapping={
            "hook_resid_mid": ("model.layers[{layer}].post_attention_layernorm", "input"),
            "mlp.hook_in": ("model.layers[{layer}].post_attention_layernorm", "output"),
            "mlp.hook_out": ("model.layers[{layer}].mlp.source.self_experts_0", "output"),
            "hook_mlp_out": ("model.layers[{layer}].mlp.source.self_experts_0", "output"),
        },
    )
)


def get_mapping(model_architecture: str) -> ModelMapping:
    """Get the NNSight mapping for a given model architecture.

    Args:
        model_architecture: The HuggingFace model architecture class name
            (e.g. ``'Gemma2ForCausalLM'``).

    Returns:
        The mapping configuration for the specified architecture.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    if model_architecture not in _REGISTRY:
        raise ValueError(
            f"Unsupported model architecture: {model_architecture}. Supported: {list(_REGISTRY)}"
        )
    return _REGISTRY[model_architecture]


def get_supported_architectures() -> list[str]:
    """Return the list of supported model architecture names."""
    return list(_REGISTRY)


def register_model(mapping: ModelMapping) -> None:
    """Register a new model architecture for use with circuit-tracer.

    This is the public extension point for adding support for model families
    not yet included in the built-in registry.  After registration, the
    architecture can be used with both the NNSight and TransformerLens backends.

    Args:
        mapping: A :class:`ModelMapping` describing the model's hook points.

    Example::

        from circuit_tracer import ModelMapping, register_model

        register_model(ModelMapping(
            model_architecture="MistralForCausalLM",
            attention_location_pattern="model.layers[{layer}].self_attn...",
            layernorm_scale_location_patterns=[...],
            pre_logit_location="model",
            embed_location="model.embed_tokens",
            embed_weight="model.embed_tokens.weight",
            unembed_weight="lm_head.weight",
        ))
    """
    _register(mapping)


def auto_detect_mapping(model_name: str) -> ModelMapping | None:
    """Check whether a HuggingFace model is already supported.

    Downloads the model config from the Hub (or cache) and looks up its
    architecture class in the registry.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"google/gemma-2-2b"``).

    Returns:
        The :class:`ModelMapping` if the architecture is registered,
        or ``None`` if it is not yet supported.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    architectures = getattr(config, "architectures", None)
    if not architectures:
        return None
    return _REGISTRY.get(architectures[0])


# ── Unified config ────────────────────────────────────────────────────


@dataclass
class UnifiedConfig:
    """A unified config class that supports both TransformerLens and NNsight field names."""

    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    d_vocab: int

    tokenizer_name: str
    model_name: str
    original_architecture: str

    n_key_value_heads: int | None = None
    dtype: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> UnifiedConfig:
        """Create from dictionary."""
        return cls(
            n_layers=config_dict["n_layers"],
            d_model=config_dict["d_model"],
            d_head=config_dict["d_head"],
            n_heads=config_dict["n_heads"],
            d_mlp=config_dict["d_mlp"],
            d_vocab=config_dict["d_vocab"],
            tokenizer_name=config_dict["tokenizer_name"],
            model_name=config_dict["model_name"],
            original_architecture=config_dict["original_architecture"],
            n_key_value_heads=config_dict.get("n_key_value_heads"),
            dtype=config_dict.get("dtype"),
        )


def convert_nnsight_config_to_transformerlens(config: Any) -> UnifiedConfig:
    """Convert an NNsight or HuggingFace config to a ``UnifiedConfig``.

    If *config* is already a ``UnifiedConfig`` it is returned unchanged.
    """
    if isinstance(config, UnifiedConfig):
        return config

    field_mappings = {
        "num_hidden_layers": "n_layers",
        "hidden_size": "d_model",
        "head_dim": "d_head",
        "num_attention_heads": "n_heads",
        "intermediate_size": "d_mlp",
        "vocab_size": "d_vocab",
        "num_key_value_heads": "n_key_value_heads",
        "torch_dtype": "dtype",
    }
    config_dict: dict[str, Any] = config.to_dict()

    if "original_architecture" not in config_dict:
        config_dict["original_architecture"] = config.architectures[0]
    if "tokenizer_name" not in config_dict:
        config_dict["tokenizer_name"] = config.name_or_path
    if "model_name" not in config_dict:
        config_dict["model_name"] = config.name_or_path

    if "text_config" in config_dict:
        config_dict |= config_dict["text_config"]

    for nnsight_field, transformerlens_field in field_mappings.items():
        if transformerlens_field not in config_dict:
            config_dict[transformerlens_field] = config_dict[nnsight_field]

    return UnifiedConfig.from_dict(config_dict)


# Backward-compatible alias
TransformerLens_NNSight_Mapping = ModelMapping
