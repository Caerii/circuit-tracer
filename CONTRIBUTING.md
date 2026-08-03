# Contributing to circuit-tracer

Thank you for your interest in contributing to circuit-tracer! We appreciate the community involvement we've seen so far and welcome contributions.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE) that covers the project.

## Maintenance Bandwidth

We maintain this project on a **best-effort basis**.

- PR reviews may take time and we cannot guarantee timely responses or merges
- Issues may not receive immediate attention

### API Stability

> **Warning**: This library is under active development and **breaking changes are possible**. The API is not stable and breaking changes may occur in any release.

## Getting Started

We use [uv](https://docs.astral.sh/uv/) for dependency management and task running.

```bash
# Clone and install (with dev dependencies)
git clone https://github.com/decoderesearch/circuit-tracer.git
cd circuit-tracer
uv sync --group dev

# (Optional) Install visualisation extras
uv sync --group dev --group viz

# (Optional) Neuronpedia upload SDK
uv pip install -e ".[neuronpedia]"
```

## Development Workflow

Run all quality checks before submitting a PR:

```bash
# Formatting (auto-fix)
uv run ruff format

# Linting (auto-fix where possible)
uv run ruff check --fix

# Type checking (must pass with 0 errors)
uv run pyright

# Tests (CPU CI suite)
uv run pytest tests -m "not requires_disk and not requires_gpu and not slow" -q
```

### GPU / VRAM-tiered tests

GitHub CI stays CPU-only. On a local CUDA machine (or self-hosted runner):

```bash
# Inspect detected GPUs
uv run python -c "from circuit_tracer.utils.cuda_info import get_cuda_capabilities; print(get_cuda_capabilities())"

# RTX 3080-class (~10GB): smoke + mid-tier GPU tests
uv run pytest -m "requires_gpu and vram_10gb" -q

# Heavy suites (24GB/32GB cards)
uv run pytest -m "requires_gpu and vram_32gb" -q
```

Markers: `requires_gpu`, `vram_8gb`, `vram_10gb`, `vram_24gb`, `vram_32gb`, `slow`, `requires_disk`.

CI runs these same checks on Python 3.10, 3.11, and 3.12.

## Project Structure

```
circuit_tracer/
├── __init__.py              # Public API (lazy imports)
├── _version.py              # Single source of truth for version
├── graph.py                 # Graph data structures and pruning
├── attribution/             # Attribution engine
│   ├── attribute.py         #   Unified attribution (both backends)
│   └── targets.py           #   LogitTarget, CustomTarget, AttributionTargets
├── replacement_model/       # Model wrappers
│   ├── common.py            #   Shared utilities (ensure_tokenized, etc.)
│   ├── replacement_model_nnsight.py
│   └── replacement_model_transformerlens.py
├── transcoder/              # Transcoder architectures
├── frontend/                # Visualisation server
└── utils/                   # Helpers (config mapping, feature decoding, etc.)
```

## How to Contribute

1. **Check existing issues and PRs** to avoid duplicate work
2. For major changes, **open an issue first** to discuss the approach
3. **Make your changes** with clear, descriptive commits
4. **Run the full quality check suite** (see above)
5. Check that relevant demo notebooks still execute correctly, particularly:
   - `demos/circuit_tracing_tutorial.ipynb`
   - `demos/attribute_demo.ipynb`
   - `demos/intervention_demo.ipynb`
6. **Submit a Pull Request** with a clear description of your changes

## What We're Looking For

- Bug fixes
- Performance enhancements
- New features that align with the project's goals
- In future: updates to support new models or transcoders (currently blocked on our pipeline for generating feature activation examples)

## Code of Conduct

Please be respectful and constructive in all interactions.
