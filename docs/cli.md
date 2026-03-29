# CLI Reference

The `circuit-tracer` command-line interface runs the complete attribution-to-visualization pipeline.

## Overview

The CLI performs three steps in sequence:

1. **Attribution**: Run the attribution algorithm to find the circuit, computing direct effects between transcoder features, error nodes, tokens, and output logits.
2. **Graph file creation**: Prune the attribution graph and convert it to JSON format for visualization.
3. **Visualization server**: Start a local web server to view and interact with the graph in your browser.

## Basic Usage

```bash
# Full pipeline: attribute, create graph files, and serve
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set gemma \
  --slug gemma-demo \
  --graph_file_dir ./graph_files \
  --server
```

```bash
# Attribution only: save raw graph for later analysis
circuit-tracer attribute \
  --prompt "The capital of France is" \
  --transcoder_set llama \
  --graph_output_path france_capital.pt
```

## Arguments

### Required

| Argument | Short | Description |
|----------|-------|-------------|
| `--prompt` | `-p` | The input prompt to analyze |
| `--transcoder_set` | `-t` | Transcoders to use. Either a HuggingFace repo ID (e.g. `mntss/gemma-scope-transcoders`) or a shortcut (`gemma`, `llama`) |

You must also specify at least one output:
- `--slug` + `--graph_file_dir` for visualization, and/or
- `--graph_output_path` (`-o`) for raw `.pt` graph file

### Optional — Attribution

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--model` | `-m` | auto | Model name (auto-inferred for `gemma`/`llama` presets) |
| `--max_n_logits` | | 10 | Maximum number of logit nodes to attribute from |
| `--desired_logit_prob` | | 0.95 | Cumulative probability threshold for selecting top logits |
| `--batch_size` | | 256 | Batch size for backward passes |
| `--max_feature_nodes` | | 7500 | Maximum number of feature nodes to include |
| `--dtype` | | model default | Load dtype: `float32`/`fp32`, `float16`/`fp16`, `bfloat16`/`bf16` |
| `--offload` | | None | Memory optimization: `cpu`, `disk`, or `None` |
| `--verbose` | | false | Display detailed progress information |

### Optional — Pruning

| Argument | Default | Description |
|----------|---------|-------------|
| `--node_threshold` | 0.8 | Keep minimum nodes whose cumulative influence >= this fraction |
| `--edge_threshold` | 0.98 | Keep minimum edges whose cumulative influence >= this fraction |

### Optional — Server

| Argument | Default | Description |
|----------|---------|-------------|
| `--server` | false | Start the visualization server after attribution |
| `--port` | 8041 | Port for the local visualization server |

## Graph Annotation (Browser UI)

When using `--server`, your browser opens an interactive visualization (the same frontend as the [original papers](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)).

| Action | How |
|--------|-----|
| Select a node | Click on it |
| Pin/unpin to subgraph pane | Ctrl+click (Cmd+click on macOS) |
| Annotate a node | Select it, then click "Edit" in the right panel |
| Group nodes into a supernode | Hold G and click on each node |
| Ungroup a supernode | Hold G and click the x next to the supernode |
| Annotate a supernode | Click on the label below the supernode |

## Remote Usage

If running on a remote machine (e.g. a GPU server), enable SSH port forwarding to view the visualization locally:

```bash
# On your local machine:
ssh -L 8041:localhost:8041 user@remote-host

# Then on the remote machine:
circuit-tracer attribute --prompt "..." --transcoder_set gemma \
  --slug demo --graph_file_dir ./graphs --server --port 8041
```

Then open `http://localhost:8041` in your local browser.

## Interventions on Neuronpedia

For Gemma-2 (2B), you can also perform interventions directly on [Neuronpedia](https://www.neuronpedia.org):
1. Pin at least one node in the graph
2. Click "Steer" in the subgraph panel
3. Adjust feature activations and observe output changes
