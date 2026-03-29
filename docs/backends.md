# Backend Guide

`circuit-tracer` supports two backends for model execution: **TransformerLens** (default) and **NNSight**. This guide explains when to use each and the differences between them.

## TransformerLens (default)

The TransformerLens backend wraps the model in a `HookedTransformer`, providing fast hook-based access to intermediate activations.

```python
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
# Equivalent to:
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", backend="transformerlens")
```

**Advantages:**
- Faster execution and lower memory usage
- Well-tested, mature implementation
- Full intervention support (feature steering, constrained layers, attention freezing)

**Limitations:**
- Only supports architectures implemented in TransformerLens (Gemma-2, Llama, GPT-2, etc.)
- Cannot be used with newer architectures until TransformerLens adds support

## NNSight

The NNSight backend wraps the model in an NNSight `LanguageModel`, which can instrument any HuggingFace model.

```python
model = ReplacementModel.from_pretrained("google/gemma-3-4b-pt", "gemma-3", backend="nnsight")
```

**Advantages:**
- Supports most HuggingFace model architectures
- Required for Gemma-3, and any model not in TransformerLens

**Limitations:**
- Slower than TransformerLens (tracing overhead)
- Higher memory usage
- Still experimental — some features may behave differently

## Architecture Support Matrix

| Architecture | TransformerLens | NNSight |
|-------------|:-:|:-:|
| Gemma-2 | yes | yes |
| Gemma-3 | no | yes |
| Gemma-3 (multimodal) | no | yes |
| Llama 3.x | yes | yes |
| Qwen-3 | no | yes |
| GPT-OSS | no | yes |

## Adding a New Backend Architecture

To add NNSight support for a new HuggingFace architecture:

1. **Register the architecture** in `circuit_tracer/utils/tl_nnsight_mapping.py`:

```python
_register(ModelMapping(
    model_architecture="YourModelForCausalLM",
    attention_location_pattern="model.layers[{layer}].self_attn.source...",
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
```

2. **Determine the correct NNSight envoy paths** by inspecting the model's trace graph. The paths depend on how NNSight decomposes the model's forward pass.

3. **Test** with a small model variant to verify attribution produces sensible results.
