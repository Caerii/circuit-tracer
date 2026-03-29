# circuit-tracer

[![PyPI version](https://img.shields.io/pypi/v/circuit-tracer.svg)](https://pypi.org/project/circuit-tracer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/Caerii/circuit-tracer/actions/workflows/ci.yaml/badge.svg)](https://github.com/Caerii/circuit-tracer/actions/workflows/ci.yaml)

Find, visualize, and intervene on circuits inside language models using features from (cross-layer) MLP transcoders, as introduced by [Ameisen et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [Lindsey et al. (2025)](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).

## What does it do?

1. **Attribution**: Given a model with pre-trained transcoders, compute the direct effect that each non-zero transcoder feature, error node, and input token has on each other feature and output logit — producing a full circuit / attribution graph.
2. **Visualization**: View and annotate the resulting graph in an interactive browser UI (the same frontend used in [the original papers](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).
3. **Intervention**: Set transcoder features to arbitrary values and observe how model output changes, validating the circuits you discover.

---

## Installation

### From PyPI (recommended)

```bash
pip install circuit-tracer
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add circuit-tracer
```

### Optional extras

```bash
# Visualization dependencies (seaborn, ipykernel, ipywidgets)
pip install circuit-tracer[viz]

# Everything (viz + dev tools)
pip install circuit-tracer[all]
```

### From source

```bash
git clone https://github.com/Caerii/circuit-tracer.git
cd circuit-tracer
uv sync --group dev
```

---

## Quick Start

```python
from circuit_tracer import ReplacementModel, Graph, attribute

# Load a model with transcoders
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

# Find the circuit for a prompt
graph = attribute(model, "The capital of France is")

# Save for later analysis
graph.to_pt("france_capital.pt")

# Or load an existing graph
graph = Graph.from_pt("france_capital.pt")
```

For a complete walkthrough, try the [tutorial notebook](https://github.com/Caerii/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb)!

---

## Getting Started

There are three ways to use circuit-tracer:

1. **On Neuronpedia** (no install needed): Use `circuit-tracer` directly on [Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&pinnedIds=27_22605_10%2C20_15589_10%2CE_26865_9%2C21_5943_10%2C23_12237_10%2C20_15589_9%2C16_25_9%2C14_2268_9%2C18_8959_10%2C4_13154_9%2C7_6861_9%2C19_1445_10%2CE_2329_7%2CE_6037_4%2C0_13727_7%2C6_4012_7%2C17_7178_10%2C15_4494_4%2C6_4662_4%2C4_7671_4%2C3_13984_4%2C1_1000_4%2C19_7477_9%2C18_6101_10%2C16_4298_10%2C7_691_10&supernodes=%5B%5B%22state%22%2C%226_4012_7%22%2C%220_13727_7%22%5D%2C%5B%22preposition+followed+by+place+name%22%2C%2219_1445_10%22%2C%2218_6101_10%22%5D%2C%5B%22Texas%22%2C%2220_15589_10%22%2C%2220_15589_9%22%2C%2219_7477_9%22%2C%2216_25_9%22%2C%224_13154_9%22%2C%2214_2268_9%22%2C%227_6861_9%22%5D%2C%5B%22capital+%2F+capital+cities%22%2C%2215_4494_4%22%2C%226_4662_4%22%2C%224_7671_4%22%2C%223_13984_4%22%2C%221_1000_4%22%2C%2221_5943_10%22%2C%2217_7178_10%22%2C%227_691_10%22%2C%2216_4298_10%22%5D%5D&pruningThreshold=0.6&clickedId=21_5943_10&densityThreshold=0.99). Click `+ New Graph` to create your own, or browse existing graphs from the drop-down.

2. **Python script / Jupyter notebook**: Start with the [tutorial notebook](https://github.com/Caerii/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb) (runs on free Colab GPUs). See the **Demos** section below.

3. **Command-line interface**: Run the full attribution-to-visualization pipeline from your terminal. See **CLI Usage** below.

Working with Gemma-2 (2B) is possible with relatively limited GPU resources; Colab GPUs have 15GB of RAM. More GPU RAM allows less offloading and larger batch sizes.

---

## Supported Models & Transcoders

The following transcoders are available. Use the HuggingFace repo name as the `transcoders` argument of `ReplacementModel.from_pretrained`, or as `--transcoder_set` in the CLI.

| Model | Transcoder Type | HuggingFace Repo | Notes |
|-------|----------------|-------------------|-------|
| **Gemma-2 (2B)** | PLT | [`mntss/gemma-scope-transcoders`](https://huggingface.co/mntss/gemma-scope-transcoders) | Originally from GemmaScope |
| **Gemma-2 (2B)** | CLT (426K) | [`mntss/clt-gemma-2-2b-426k`](https://huggingface.co/mntss/clt-gemma-2-2b-426k) | |
| **Gemma-2 (2B)** | CLT (2.5M) | [`mntss/clt-gemma-2-2b-2.5M`](https://huggingface.co/mntss/clt-gemma-2-2b-2.5M) | |
| **Llama-3.2 (1B)** | PLT | [`mntss/transcoder-Llama-3.2-1B`](https://huggingface.co/mntss/transcoder-Llama-3.2-1B) | |
| **Llama-3.2 (1B)** | CLT | [`mntss/clt-llama-3.2-1b-524k`](https://huggingface.co/mntss/clt-llama-3.2-1b-524k) | |
| **Qwen-3** | PLT | [0.6B](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0), [1.7B](https://huggingface.co/mwhanna/qwen3-1.7b-transcoders-lowl0), [4B](https://huggingface.co/mwhanna/qwen3-4b-transcoders), [8B](https://huggingface.co/mwhanna/qwen3-8b-transcoders), [14B](https://huggingface.co/mwhanna/qwen3-14b-transcoders-lowl0) | |
| **GPT-OSS (20B)** | CLT | [`mntss/clt-131k`](https://huggingface.co/mntss/clt-131k) | |
| **Gemma-3** | PLT | [Collection](https://huggingface.co/collections/mwhanna/gemma-scope-2-transcoders-circuit-tracer) | 270M to 27B, PT & IT. Requires `nnsight` backend. |

### Choosing a Backend

By default, `circuit-tracer` uses the **TransformerLens** backend (inherits from `HookedTransformer`). For models not supported by TransformerLens, use the **NNSight** backend:

```python
# TransformerLens (default) — fast, memory-efficient
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")

# NNSight — supports most HuggingFace models
model = ReplacementModel.from_pretrained("google/gemma-3-4b-pt", "gemma-3", backend="nnsight")
```

> **Note**: The NNSight backend is still experimental: it is slower and less memory-efficient, and may not provide all of the functionality of the TransformerLens version.

---

## Demos

All demos live in the [`demos/`](https://github.com/Caerii/circuit-tracer/tree/main/demos) folder. The main tutorial can run on free Colab GPUs.

| Notebook | Description |
|----------|-------------|
| [**circuit_tracing_tutorial.ipynb**](https://github.com/Caerii/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb) | End-to-end tutorial replicating findings from [Lindsey et al.](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |
| [attribute_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/attribute_demo.ipynb) | How to find circuits and visualize them |
| [attribution_targets_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/attribution_targets_demo.ipynb) | Specifying custom attribution targets (specific logits or directions) |
| [intervention_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/intervention_demo.ipynb) | Performing feature interventions on models |
| [gemma_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/gemma_demo.ipynb) | Pre-annotated Gemma-2 (2B) graphs with interventions |
| [gemma_it_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/gemma_it_demo.ipynb) | Instruction-tuned Gemma-2 (2B) with base-model transcoders |
| [llama_demo.ipynb](https://github.com/Caerii/circuit-tracer/blob/main/demos/llama_demo.ipynb) | Llama 3.2 (1B) graphs (requires local GPU, not Colab) |

---

## Caching

To use `lazy_decoder` and `lazy_encoder` options on transcoders, they must be in `circuit-tracer`-compatible format. Rather than downloading pre-converted weights, you can build a local cache:

```python
from circuit_tracer.utils.caching import save_transcoders_to_cache

save_transcoders_to_cache(
    "mwhanna/gemma-scope-2-27b-pt/transcoder_all/width_262k_l0_small",
    cache_dir="~/.cache/",
)
```

Clear the cache with `circuit_tracer.utils.caching.empty_cache`.

---

## CLI Usage

The CLI runs the complete 3-step pipeline: **attribution** -> **graph file creation** -> **visualization server**.

### Basic Usage

```bash
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set gemma \
  --slug gemma-demo \
  --graph_file_dir ./graph_files \
  --server
```

The server URL (e.g. `localhost:8041`) will be printed. If running on a remote machine, enable port forwarding to view the graphs locally.

### Attribution only (save raw graph)

```bash
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set llama \
  --graph_output_path france_capital.pt
```

### CLI Arguments

#### Required

| Argument | Description |
|----------|-------------|
| `--prompt` (`-p`) | Input prompt to analyze |
| `--transcoder_set` (`-t`) | Transcoders to use (HuggingFace repo ID, or shortcut: `gemma`, `llama`) |

Plus at least one output destination:
- `--slug` + `--graph_file_dir` (for visualization), and/or
- `--graph_output_path` (`-o`) (for raw `.pt` graph)

#### Optional

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` (`-m`) | auto | Model name (auto-inferred for `gemma`/`llama` presets) |
| `--max_n_logits` | 10 | Maximum logit nodes to attribute from |
| `--desired_logit_prob` | 0.95 | Cumulative probability threshold for top logits |
| `--batch_size` | 256 | Batch size for backward passes |
| `--max_feature_nodes` | 7500 | Maximum feature nodes |
| `--dtype` | model default | Load dtype (`float32`/`fp32`, `float16`/`fp16`, `bfloat16`/`bf16`) |
| `--offload` | None | Memory optimization (`cpu`, `disk`, or `None`) |
| `--node_threshold` | 0.8 | Node pruning: keep nodes with cumulative influence >= threshold |
| `--edge_threshold` | 0.98 | Edge pruning: keep edges with cumulative influence >= threshold |
| `--port` | 8041 | Local server port |
| `--server` | false | Start visualization server after attribution |
| `--verbose` | false | Show detailed progress |

### Graph Annotation

When using `--server`, the browser opens an interactive visualization:

- **Select a node**: Click on it
- **Pin/unpin to subgraph pane**: Ctrl+click (or Cmd+click)
- **Annotate a node**: Click "Edit" on the right side while a node is selected
- **Group nodes into a supernode**: Hold G and click on nodes
- **Ungroup a supernode**: Hold G and click the x next to it
- **Annotate a supernode**: Click on the label below it

Interventions are also available on Neuronpedia for Gemma-2 (2B): pin at least one node, then click "Steer" in the subgraph.

---

## Project Structure

```
circuit_tracer/
├── __init__.py                  # Public API: attribute, Graph, ReplacementModel
├── _version.py                  # Single source of truth for version
├── graph.py                     # Graph data structures, pruning, influence computation
├── attribution/
│   ├── attribute.py             # Unified attribution engine (both backends)
│   └── targets.py               # LogitTarget, CustomTarget, AttributionTargets
├── replacement_model/
│   ├── common.py                # Shared utilities (tokenization, interventions)
│   ├── replacement_model_nnsight.py
│   └── replacement_model_transformerlens.py
├── transcoder/
│   ├── cross_layer_transcoder.py
│   └── single_layer_transcoder.py
├── frontend/                    # Visualization server and graph models
└── utils/                       # Config mapping, caching, feature decoding, etc.
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and development workflow. We use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Quick dev setup
git clone https://github.com/Caerii/circuit-tracer.git
cd circuit-tracer
uv sync --group dev

# Quality checks
uv run ruff format && uv run ruff check && uv run pyright && uv run pytest tests -q
```

---

## Active Maintainers

- [Alif Jakir](https://github.com/Caerii)

## Cite

```bibtex
@misc{circuit-tracer,
  author = {Hanna, Michael and Piotrowski, Mateusz and Lindsey, Jack and Ameisen, Emmanuel},
  title = {circuit-tracer},
  howpublished = {\url{https://github.com/decoderesearch/circuit-tracer}},
  note = {The first two authors contributed equally and are listed alphabetically.},
  year = {2025}
}
```

Or cite the paper [here](https://aclanthology.org/2025.blackboxnlp-1.14/).
