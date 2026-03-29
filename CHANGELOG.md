# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-03-28

### Added

- **PyPI packaging**: hatchling build system, Trusted Publisher workflow, `MANIFEST.in`, `py.typed` PEP 561 marker.
- **Single-source versioning** via `circuit_tracer/_version.py`, read by both hatchling and `__init__.py`.
- **Registry pattern** for model architecture mappings (`ModelMapping` frozen dataclass, `get_mapping()`, `get_supported_architectures()`).
- **Unified attribution interface**: both NNSight and TransformerLens backends now expose `unembed_proj`, `model_config`, `run_forward_pass()`, and `get_offload_targets_phase{0,1,2}()`, allowing a single `_run_attribution()` implementation.
- **Shared utilities** in `replacement_model/common.py`: `ensure_tokenized()` and `convert_open_ended_interventions()` extracted from duplicated code.
- **Lazy imports** in `__init__.py` and `attribution/__init__.py` to prevent circular import issues.
- **CI modernisation**: uv-based workflows, Python 3.10/3.11/3.12 matrix, `requires_gpu` pytest marker.
- **CHANGELOG.md** (this file).
- `circuit-tracer` CLI entry point via `project.scripts`.

### Changed

- **Attribution deduplication**: merged `attribute_nnsight.py` and `attribute_transformerlens.py` (~560 lines combined) into a single `attribute.py` (~170 lines) using the unified model interface.
- **`CONTRIBUTING.md`** rewritten with uv-based developer workflow.
- **`__main__.py`**: fixed `if`/`elif` dispatch bug, extracted magic numbers into named constants.
- **`frontend/local_server.py`**: extracted constants, renamed `format` → `fmt` to avoid shadowing builtin.
- **`frontend/graph_models.py`**: lifted `cantor_pairing` to module-level `_cantor_pairing()`.
- **`utils/demo_utils.py`**: removed duplicate `decode_url_features()` (~65 lines), imports from canonical location.
- **`graph.py`**: `print()` → `logging.warning()`, added type hints to `Graph.__init__` and `.to()`.
- Loosened `nnsight` dependency from `==0.6.1` to `>=0.6.0`.
- Moved `ipykernel`, `ipywidgets`, `seaborn` from core deps to `[viz]` optional extra.
- Ruff config expanded: `I` (isort), `UP` (pyupgrade), `TID` rules; banned `typing.Union`/`Optional`/`Dict`/`Tuple`/`List`.

### Removed

- `attribute_nnsight.py` — merged into unified `attribute.py`.
- `attribute_transformerlens.py` — merged into unified `attribute.py`.
- `utils/salient_logits.py` — exact duplicate of logic in `targets.py`.
- Duplicate `decode_url_features()` and `extract_supernode_features()` from `demo_utils.py`.

### Fixed

- Circular import between `attribution/__init__.py` and `graph.py` via lazy `__getattr__` pattern.
- `__main__.py` second command check was `if` instead of `elif`, causing potential `AttributeError`.
- pyright: resolved all 11 type errors (device nullability, `CrossLayerTranscoder` iterability, overload mismatches).

## [0.4.1] — 2026-03-13

### Added

- Attribution targets encapsulation (`LogitTarget`, `CustomTarget`, `AttributionTargets`).
- Active maintainer section in README.

## [0.4.0] — 2026-02-28

### Changed

- Version bump from 0.1.0 to 0.4.0.

## [0.1.0] — 2026-01-14

- Initial release.
