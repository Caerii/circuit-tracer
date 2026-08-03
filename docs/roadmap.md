# Roadmap

This document describes the planned release trajectory for circuit-tracer, organized by version.  Each increment is independently shippable and builds on the previous one.

---

## v0.5.0 — Foundation (shipped)

**Theme**: Production-grade packaging and code quality.

### What shipped
- PyPI packaging with hatchling build system and Trusted Publisher OIDC workflow
- Tag-derived versioning via `hatch-vcs`
- Unified attribution engine — merged duplicate NNSight/TransformerLens implementations into one `attribute.py`
- Registry pattern for model architectures (`ModelMapping` frozen dataclass)
- Shared utilities in `replacement_model/common.py` (eliminated code duplication)
- Lazy imports to prevent circular dependency issues
- CI modernization: uv-based workflows, Python 3.10–3.12 matrix
- CHANGELOG.md, CONTRIBUTING.md rewrite, comprehensive docs/

### Why it mattered
Before v0.5.0, circuit-tracer was a research codebase with duplicated code paths, no PyPI presence, and fragile imports.  This release made it installable (`pip install circuit-tracer`) and maintainable.

---

## v0.6.0 — Programmatic Analysis API (current)

**Theme**: Surface existing analysis capabilities as a discoverable public API.

### What ships
- **`circuit_tracer/analysis.py`** — new module with `get_top_features()`, `graph_to_interventions()`, `compare_graphs()`, `find_common_circuit()`, `ComparisonResult`
- **Graph convenience methods** — `graph.top_features()`, `.prune()`, `.scores()`, `.to_json()` delegate to standalone functions
- **`attribute_batch()`** — run attribution across multiple prompts
- **Public intervention bridge** — `graph_to_interventions()` converts top features into intervention tuples
- **Model extensibility API** — `register_model()`, `auto_detect_mapping()` let users add new architectures without reading source
- **18 public exports** from `circuit_tracer` (up from 3)
- **`docs/api.md`** — full programmatic API guide with examples
- New test suites: `test_analysis.py`, `test_model_registry.py`

### Why it matters
Before v0.6.0, a user who called `attribute()` and got a `Graph` had no discoverable way to analyze it without importing internals.  `get_top_features()` was buried in `demo_utils.py`.  This release makes the API self-documenting — every analysis capability is importable from `circuit_tracer` directly.

### Key design decisions
- **Both methods and standalone functions**: `graph.top_features()` for discoverability, `get_top_features(graph)` for composability across multiple graphs.
- **`analysis.py` as separate module**: Keeps `graph.py` focused on data structures + low-level math.
- **Sequential `attribute_batch`**: Establishes the API contract now; parallelization can come later without breaking the interface.

---

## v0.7.0 — Intervention & Steering (in progress)

**Theme**: Close the loop from understanding to action.

### Shipped early (on main, pre-tag)
- **`steer()`** — amplify/suppress API over `feature_intervention` / `feature_intervention_generate`
- **Intervention plan I/O** — `save_intervention_plan` / `load_intervention_plan` / `run_intervention_plan`
- **`validate_intervention()`** — observe logit effects; optional adjacency prediction checks
- **CLI parity** — `summarize`, `interventions`, `export-viz`

### Still planned
- Stronger cross-backend intervention parity tests in CI
- Richer expected-effect APIs for custom target directions

### Why it matters
The mechanistic interpretability workflow is: attribute → identify circuit → intervene → confirm.  v0.6.0 covers the first two steps; v0.7.0 closes the loop with a production-quality intervention API.  This is what enables practical applications like safety monitoring and model behavior modification.

---

## v0.8.0 — Statistical Circuit Analysis (in progress)

**Theme**: Scale from single-prompt to dataset-level analysis.

### Shipped early
- **`attribute_batch(..., max_workers=)`** with multi-model data parallel
- **`CircuitDataset`** save/load + **`compare_datasets()`** drift helper
- **`cluster_circuits()`** — Jaccard / cosine agglomerative clustering of top features
- **`summarize_dataset()`** — feature frequencies, mean influence, bootstrap CIs, per-label breakdown
- **VRAM-tiered GPU helpers** — `get_cuda_capabilities()` + pytest markers for 8/10/24/32GB cards

### Still planned
- Multi-GPU orchestration helpers
- Circuit clustering refinements (soft clustering / embeddings)

### Why it matters
Single-prompt analysis answers "what circuit did the model use here?"  Dataset-level analysis answers "what circuits does the model use for this *class* of inputs?" — the question regulators, auditors, and alignment researchers actually need answered.

---

## v0.9.0 — Universal Model Support (planned)

**Theme**: Make circuit-tracer work on any HuggingFace causal LM.

### Planned additions
- **Auto-mapping from HF config** — infer NNSight hook points from model architecture automatically, reducing `ModelMapping` boilerplate
- **Mapping validation** — `validate_mapping(mapping, model_name)` soft-checks paths *(shipped early on main)*
- **Architecture-specific test harness** — `test_new_architecture(model_name, mapping)` runs a minimal attribution and verifies edge correctness
- **Mixture-of-Experts support** — handle MoE routing in attribution (which expert activates? why?)
- **Community mapping registry** — accept contributed mappings via PRs with standardized test coverage
- **Expanded built-in support**: Mistral, Phi, DeepSeek, Command-R

### Why it matters
If circuit-tracer only works on 6 architectures, it's a research tool.  If it works on every HuggingFace model, it's civilizational infrastructure.  The registry pattern is already in place — this release removes the friction of contributing new architectures.

---

## v1.0.0 — Production Stability (planned)

**Theme**: Stability guarantees and ecosystem integration.

### Planned additions
- **Semantic versioning guarantee** — stable public API with deprecation policy
- **Formal API documentation** — auto-generated from docstrings (Sphinx or mkdocs)
- **Performance benchmarks** — tracked per-release attribution throughput on reference hardware
- **Integration with SAELens, Baukit, pyvene** — interoperability with the broader interpretability ecosystem
- **Standard interchange format** — portable graph format beyond PyTorch pickle (HDF5 or protocol buffers)
- **Neuronpedia deep integration** — `upload_graph_to_neuronpedia` / `fetch_feature` via optional extra *(shipped early)*; batch feature/activation upload still planned

### Why it matters
1.0 is the signal to the community that circuit-tracer is stable enough to build on.  Ecosystem integration means researchers don't have to choose between tools — circuit-tracer becomes the analysis layer that connects them.

---

## Beyond 1.0 — Aspirational

These are high-impact directions that require fundamental research, not just engineering:

- **Streaming/runtime attribution** — monitor circuits during inference for real-time safety
- **Transcoder training pipeline** — train your own transcoders for any model (removes the biggest bottleneck)
- **Attention mechanism attribution** — extend beyond MLP features to attention heads
- **Higher-order interactions** — capture non-linear effects that the current linear attribution misses
- **Vision/multimodal circuits** — trace circuits through vision encoders and cross-modal attention

---

## Contributing

Each version milestone has clear scope.  If you want to contribute, pick a feature from an upcoming version and open an issue to discuss the approach before writing code.  See [CONTRIBUTING.md](../CONTRIBUTING.md) for the development workflow.
