# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- JSON schema validation helpers (`validate_summary`, `load_summary`, `dump_summary`) and golden fixtures
- CLI `summarize`, `interventions`, and `export-viz` commands (parity with analysis / viz APIs)
- `steer()`, intervention plan load/save/run, and `validate_intervention()`
- `validate_mapping()` for soft-checking `ModelMapping` paths
- Parallel-capable `attribute_batch(..., max_workers=...)` with multi-model data parallel
- `CircuitDataset` + `compare_datasets()` for multi-prompt circuit collections and drift
- `export_graph_for_viz()` with local serve / Neuronpedia hints

## [0.6.0] - 2026-07-26

### Added

- Programmatic analysis API (`get_top_features`, `compare_graphs`, `find_common_circuit`, `graph_to_interventions`)
- Versioned JSON interchange summaries for graphs and interventions (`summarize_graph`, `summarize_interventions`, `summarize_intervention_results`)
- Graph convenience methods (`top_features`, `prune`, `scores`, `summary`, `intervention_summary`, `to_json`, …)
- Sequential `attribute_batch` API
- Public `ModelMapping` registry (`register_model`, `auto_detect_mapping`)
- Shared replacement-model helpers and unified attribution execution path
- Richer docs (API, backends, CLI, architecture, roadmap) and contribution workflow

### Fixed

- Hugging Face config access in NNSight replacement models (`_hf_config` / softcap / architecture resolution)
- TransformerLens `feature_intervention_generate` logit cache init (empty list + append; avoids a leading `None`)

### Changed

- Tag-derived package versioning via `hatch-vcs` (aligned with upstream)
- Stricter publish workflow (full git history for VCS version; no soft-fail on tests)

## [0.5.2] - upstream

See [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer) for earlier history.
