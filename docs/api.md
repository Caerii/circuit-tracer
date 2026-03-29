# Programmatic API Guide

This guide covers the Python API for programmatic circuit analysis — the primary way to use circuit-tracer as a library rather than a CLI tool.

## Quick Start

```python
from circuit_tracer import attribute, ReplacementModel, get_top_features

# Load model with transcoders
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

# Compute attribution graph
graph = attribute("The capital of France is", model)

# Analyze the circuit
features, scores = graph.top_features(n=10)
for (layer, pos, feat_idx), score in zip(features, scores):
    print(f"  Layer {layer}, pos {pos}, feature {feat_idx}: {score:.4f}")

# Quality metrics
replacement_score, completeness_score = graph.scores()
print(f"Replacement: {replacement_score:.2%}, Completeness: {completeness_score:.2%}")
```

## Core Workflow

### 1. Load a Model

```python
from circuit_tracer import ReplacementModel

# Using preset shortcuts
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")
model = ReplacementModel.from_pretrained("meta-llama/Llama-3.2-1B", "llama")

# Using explicit HuggingFace repo
model = ReplacementModel.from_pretrained(
    "google/gemma-2-2b",
    "mntss/gemma-scope-transcoders",
    backend="nnsight",  # or "transformerlens" (default)
)
```

### 2. Compute Attribution

```python
from circuit_tracer import attribute

# Auto-select salient logits
graph = attribute("The capital of France is", model)

# Target specific tokens
graph = attribute("The capital of France is", model,
                  attribution_targets=["Paris", "London", "Berlin"])

# Custom contrast directions
from circuit_tracer import CustomTarget
import torch

paris_vec = model.unembed_proj[:, tokenizer.encode("Paris")[0]]
london_vec = model.unembed_proj[:, tokenizer.encode("London")[0]]
graph = attribute("The capital of France is", model,
                  attribution_targets=[
                      CustomTarget("Paris-London", 1.0, paris_vec - london_vec)
                  ])
```

### 3. Analyze the Graph

```python
from circuit_tracer import get_top_features, prune_graph, compute_graph_scores

# Top features by multi-hop influence
features, scores = get_top_features(graph, n=20)

# Or equivalently, as a method:
features, scores = graph.top_features(n=20)

# Prune low-influence nodes and edges
result = graph.prune(node_threshold=0.8, edge_threshold=0.98)
print(f"Kept {result.node_mask.sum()} nodes, {result.edge_mask.sum()} edges")

# Quality metrics
replacement, completeness = graph.scores()
```

### 4. Intervene on Features

```python
from circuit_tracer import graph_to_interventions

# Ablate top 5 features (set activations to 0)
interventions = graph_to_interventions(graph, n=5, value=0.0)
new_logits, _ = model.feature_intervention("The capital of France is", interventions)

# Amplify a specific feature
interventions = [(layer, pos, feat_idx, 5.0)]
new_logits, _ = model.feature_intervention("The capital of France is", interventions)
```

### 5. Export for Visualization

```python
# Save raw graph (for later analysis)
graph.to_pt("my_graph.pt")

# Export pruned JSON (for browser visualization)
graph.scan = "gemma"  # required for JSON export
graph.to_json(slug="france-capital", output_path="./graph_files")
```

## Batch Analysis

Compare circuits across multiple prompts to find consistent patterns:

```python
from circuit_tracer import attribute_batch, compare_graphs, find_common_circuit

# Attribute multiple prompts
prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
]
graphs = attribute_batch(prompts, model, verbose=True)

# Find shared circuit components
result = compare_graphs(graphs, n_per_graph=20)
print(f"Shared features: {len(result.shared_features)}")
print(f"Per-graph scores: {result.graph_scores}")

# Find features appearing in at least 50% of prompts
common = find_common_circuit(graphs, min_frequency=0.5)
for layer, pos, feat_idx in common:
    print(f"  Common feature: layer {layer}, pos {pos}, idx {feat_idx}")
```

## Adding New Model Architectures

Check if a model is supported:

```python
from circuit_tracer import get_supported_architectures, auto_detect_mapping

print(get_supported_architectures())
# ['Gemma2ForCausalLM', 'Gemma3ForCausalLM', ..., 'GptOssForCausalLM']

mapping = auto_detect_mapping("google/gemma-2-2b")
print(mapping)  # ModelMapping(model_architecture='Gemma2ForCausalLM', ...)
```

Register a new architecture:

```python
from circuit_tracer import ModelMapping, register_model

register_model(ModelMapping(
    model_architecture="MistralForCausalLM",
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
    },
))
```

To find the correct NNSight paths for a new model, load it in NNSight and inspect the trace graph. See [backends.md](backends.md) for details.

## API Reference

### Top-Level Exports

| Function | Description |
|----------|-------------|
| `attribute(prompt, model, **kwargs)` | Compute attribution graph for a single prompt |
| `attribute_batch(prompts, model, **kwargs)` | Attribute multiple prompts sequentially |
| `get_top_features(graph, n=10)` | Top features by multi-hop influence |
| `graph_to_interventions(graph, n=10, value=0.0)` | Convert graph features to intervention tuples |
| `prune_graph(graph, node_threshold, edge_threshold)` | Remove low-influence nodes/edges |
| `compute_graph_scores(graph)` | Replacement + completeness metrics |
| `compare_graphs(graphs, n_per_graph=20)` | Compare multiple graphs, find overlap |
| `find_common_circuit(graphs, min_frequency=0.5)` | Features appearing across graphs |
| `register_model(mapping)` | Register new model architecture |
| `auto_detect_mapping(model_name)` | Check if a HF model is supported |
| `get_supported_architectures()` | List registered architecture names |

### Types

| Type | Description |
|------|-------------|
| `Graph` | Attribution graph with adjacency matrix + metadata |
| `ReplacementModel` | Factory for model + transcoder loading |
| `PruneResult` | NamedTuple: `(node_mask, edge_mask, cumulative_scores)` |
| `ComparisonResult` | NamedTuple: `(shared_features, per_graph_features, feature_frequency, graph_scores)` |
| `CustomTarget` | NamedTuple: `(token_str, prob, vec)` for custom attribution directions |
| `Intervention` | Type alias: `tuple[int, int|slice, int, float|Tensor]` |
| `ModelMapping` | Frozen dataclass mapping HF architecture to hook points |

### Graph Methods

| Method | Description |
|--------|-------------|
| `graph.top_features(n=10)` | Shorthand for `get_top_features(graph, n)` |
| `graph.prune(node_threshold, edge_threshold)` | Shorthand for `prune_graph(graph, ...)` |
| `graph.scores()` | Shorthand for `compute_graph_scores(graph)` |
| `graph.to_json(slug, output_path, **kwargs)` | Export pruned JSON for visualization |
| `graph.to_pt(path)` | Save raw graph as `.pt` file |
| `Graph.from_pt(path)` | Load graph from `.pt` file |
| `graph.to(device)` | Move tensors to device |
