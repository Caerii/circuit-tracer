# Architecture Overview

This document describes the internal architecture of `circuit-tracer` for contributors and advanced users.  For the user-facing API, see [api.md](api.md).  For the release roadmap, see [roadmap.md](roadmap.md).

## Core Concepts

### Attribution Graphs

An **attribution graph** represents the computational circuit a language model uses for a given input. Each node is one of:

- **Feature node**: A non-zero transcoder feature at a specific layer and token position
- **Error node**: The MLP reconstruction error at a given layer/position (what transcoders fail to capture)
- **Token node**: An input embedding
- **Logit node**: An output logit target

Edges represent **direct effects** — the causal influence one node has on another through the model's computation.

### Transcoders

Transcoders replace MLP layers with sparse, interpretable feature dictionaries. There are two architectures:

- **Single-layer transcoders (PLTs)**: One transcoder per MLP layer. Each reads from and writes to the same residual stream position.
- **Cross-layer transcoders (CLTs)**: A single transcoder whose features can read from one layer and write to all subsequent layers, capturing cross-layer interactions.

Both are managed by the `ReplacementModel`, which substitutes them in place of the original MLP layers during attribution.

## Module Structure

```
circuit_tracer/
├── __init__.py                      # Public API (lazy imports)
├── _version.py                      # Single source of truth for version
│
├── analysis.py                      # High-level programmatic analysis API
│   - get_top_features(): rank features by multi-hop influence
│   - graph_to_interventions(): attribution → intervention bridge
│   - compare_graphs(), find_common_circuit(): batch comparison
│
├── graph.py                         # Graph class, pruning, influence computation
│   - Graph: stores adjacency matrix, active features, logit targets
│   - Graph.top_features(), .prune(), .scores(), .to_json(): convenience methods
│   - prune_graph(): node + edge thresholding by influence
│   - compute_graph_scores(): replacement & completeness metrics
│
├── attribution/
│   ├── __init__.py                  # Lazy imports (avoids circular deps with graph.py)
│   ├── attribute.py                 # Unified attribution engine
│   │   - attribute(): public entry point for single prompts
│   │   - attribute_batch(): multi-prompt attribution
│   │   - _run_attribution(): backend-agnostic core (uses model interface methods)
│   ├── targets.py                   # LogitTarget, CustomTarget, AttributionTargets
│   ├── context_nnsight.py           # NNSight-specific attribution context
│   └── context_transformerlens.py   # TransformerLens-specific attribution context
│
├── replacement_model/
│   ├── common.py                    # Shared utilities
│   │   - ensure_tokenized(): prompt → token tensor with BOS handling
│   │   - convert_open_ended_interventions(): for generation loops
│   ├── replacement_model.py         # Backend-dispatching factory
│   ├── replacement_model_nnsight.py # NNSight backend
│   └── replacement_model_transformerlens.py  # TransformerLens backend
│
├── transcoder/
│   ├── cross_layer_transcoder.py    # CrossLayerTranscoder (CLT)
│   └── single_layer_transcoder.py   # SingleLayerTranscoder + TranscoderSet
│
├── frontend/
│   ├── local_server.py              # HTTP server for graph visualization
│   ├── graph_models.py              # Pydantic models for JSON graph format
│   └── assets/                      # Static HTML/JS/CSS frontend
│
└── utils/
    ├── tl_nnsight_mapping.py        # Model architecture registry (ModelMapping)
    ├── caching.py                   # Transcoder cache management
    ├── create_graph_files.py        # Graph → JSON conversion
    ├── decode_url_features.py       # Feature URL encoding/decoding
    ├── demo_utils.py                # Notebook helper functions
    ├── disk_offload.py              # GPU memory offloading utilities
    └── hf_utils.py                  # HuggingFace download helpers
```

## Unified Model Interface

Both `NNSightReplacementModel` and `TransformerLensReplacementModel` expose a uniform interface consumed by the attribution engine:

| Method / Property | Purpose |
|-------------------|---------|
| `model.unembed_proj` | Unembedding weight matrix (`W_U`) |
| `model.model_config` | Unified model configuration (`UnifiedConfig`) |
| `model.run_forward_pass(input_ids, batch_size, ctx)` | Execute forward pass, cache residuals |
| `model.get_offload_targets_phase0()` | Modules to offload after transcoder precomputation |
| `model.get_offload_targets_phase1()` | Modules to offload after forward pass |
| `model.get_offload_targets_phase2()` | Modules to offload after building input vectors |
| `model.ensure_tokenized(prompt)` | Convert prompt to token tensor |

This design eliminates code duplication — the attribution logic in `_run_attribution()` is written once and works identically across both backends.

## Model Architecture Registry

Supported model architectures are registered in `utils/tl_nnsight_mapping.py` using frozen `ModelMapping` dataclasses:

```python
from circuit_tracer.utils.tl_nnsight_mapping import get_mapping, get_supported_architectures

# List all supported architectures
get_supported_architectures()
# ['Gemma2ForCausalLM', 'Gemma3ForCausalLM', 'LlamaForCausalLM', ...]

# Get mapping for a specific architecture
mapping = get_mapping("Gemma2ForCausalLM")
mapping.attention_location_pattern  # NNSight envoy path with {layer} placeholder
mapping.feature_hook_mapping        # TransformerLens hook → NNSight location mapping
```

To add support for a new architecture, append a `_register(ModelMapping(...))` call in that file.

## Graph Lifecycle

```
1. model.setup_attribution(prompt)
   └── Precompute transcoder activations, errors, embeddings

2. attribute(model, prompt)
   └── _run_attribution(model, ...)
       ├── Build unembed vectors for logit targets
       ├── Backward pass: compute Jacobian (batched)
       └── Return Graph(adjacency_matrix, active_features, ...)

3. prune_graph(graph, node_threshold, edge_threshold)
   ├── Compute node influence via power iteration
   ├── Threshold nodes by cumulative influence
   ├── Compute edge influence on pruned matrix
   ├── Threshold edges
   └── Iteratively remove orphaned nodes

4. create_graph_files(graph, slug, ...)
   └── Convert to JSON → serve via frontend
```

## Influence Computation

Node influence is computed by normalizing the adjacency matrix and iterating:

```
influence = logit_weights @ A + logit_weights @ A² + logit_weights @ A³ + ...
```

This is equivalent to `logit_weights @ ((I - A)⁻¹ - I)` but computed iteratively for efficiency, converging when the contribution of higher powers drops to zero (guaranteed for acyclic graphs).
