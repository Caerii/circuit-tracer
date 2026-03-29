# Roadmap

This document describes the planned release trajectory for circuit-tracer, organized by version.  Each increment is independently shippable and builds on the previous one.

---

## v0.5.0 — Foundation (shipped)

**Theme**: Production-grade packaging and code quality.

### What shipped
- PyPI packaging with hatchling build system and Trusted Publisher OIDC workflow
- Single-source versioning via `_version.py`
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

## v0.7.0 — Intervention & Steering (planned)

**Theme**: Close the loop from understanding to action.

### Planned additions
- **Unified intervention interface** across both backends — ensure `feature_intervention()` is consistent between TransformerLens and NNSight
- **`steer()` high-level function** — simplified API for feature steering without manually constructing intervention tuples:
  ```python
  from circuit_tracer import steer
  output = steer(model, "The capital of France is",
                 amplify=[(layer, pos, feat_idx, 2.0)],
                 suppress=[(layer, pos, feat_idx)])
  ```
- **Intervention validation** — automatically verify that predicted effects match actual logit changes (the test-by-intervention pattern currently only in tests)
- **`generate_with_interventions()`** — open-ended generation with persistent feature modifications
- **Intervention serialization** — save/load intervention specs as JSON for reproducibility

### Why it matters
The mechanistic interpretability workflow is: attribute → identify circuit → intervene → confirm.  v0.6.0 covers the first two steps; v0.7.0 closes the loop with a production-quality intervention API.  This is what enables practical applications like safety monitoring and model behavior modification.

---

## v0.8.0 — Statistical Circuit Analysis (planned)

**Theme**: Scale from single-prompt to dataset-level analysis.

### Planned additions
- **Parallel `attribute_batch()`** — multi-GPU or batched attribution for throughput on large datasets
- **`CircuitDataset`** — lightweight container for collections of graphs with metadata (prompt, label, model version):
  ```python
  dataset = CircuitDataset.from_prompts(prompts, model, labels=labels)
  dataset.save("ioi_circuits/")
  dataset = CircuitDataset.load("ioi_circuits/")
  ```
- **Circuit clustering** — group prompts by circuit similarity (which prompts use the same computational pathway?)
- **Circuit regression testing** — detect when model fine-tuning changes established circuits:
  ```python
  baseline = CircuitDataset.load("baseline_circuits/")
  current = CircuitDataset.from_prompts(prompts, finetuned_model)
  drift = compare_datasets(baseline, current)
  ```
- **Statistical summaries** — aggregated feature importance across datasets, confidence intervals on circuit composition

### Why it matters
Single-prompt analysis answers "what circuit did the model use here?"  Dataset-level analysis answers "what circuits does the model use for this *class* of inputs?" — the question regulators, auditors, and alignment researchers actually need answered.

---

## v0.9.0 — Universal Model Support (planned)

**Theme**: Make circuit-tracer work on any HuggingFace causal LM.

### Planned additions
- **Auto-mapping from HF config** — infer NNSight hook points from model architecture automatically, reducing `ModelMapping` boilerplate
- **Mapping validation** — `validate_mapping(mapping, model_name)` checks that paths resolve to actual modules:
  ```python
  from circuit_tracer import validate_mapping, ModelMapping
  warnings = validate_mapping(my_mapping, "mistralai/Mistral-7B-v0.3")
  # ["Warning: model.layers[0].self_attn... not found, did you mean ...?"]
  ```
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
- **Neuronpedia deep integration** — bidirectional: export to and import from Neuronpedia in a single call

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
