# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Programmatic analysis API (`get_top_features`, `compare_graphs`, `find_common_circuit`, `graph_to_interventions`)
- Versioned JSON interchange summaries for graphs and interventions
- Graph convenience methods (`top_features`, `prune`, `scores`, `summary`, `to_json`, …)
- Sequential `attribute_batch` API
- Public `ModelMapping` registry (`register_model`, `auto_detect_mapping`)
- Shared replacement-model helpers and unified attribution execution path
- Richer PyPI metadata, optional `viz` extras, and Trusted Publishing workflow
