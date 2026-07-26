"""JSON interchange schema helpers for circuit-tracer summaries.

Validates the stable ``circuit-tracer.*.v1`` document kinds produced by
:mod:`circuit_tracer.analysis` so pipelines and tests can detect drift early.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from circuit_tracer.analysis import (
    CIRCUIT_SUMMARY_KIND,
    INTERVENTION_RESULT_SUMMARY_KIND,
    INTERVENTION_SUMMARY_KIND,
)

__all__ = [
    "CIRCUIT_SUMMARY_KIND",
    "INTERVENTION_SUMMARY_KIND",
    "INTERVENTION_RESULT_SUMMARY_KIND",
    "KNOWN_SUMMARY_KINDS",
    "SchemaError",
    "validate_summary",
    "assert_json_safe",
    "load_summary",
    "dump_summary",
]

KNOWN_SUMMARY_KINDS = frozenset(
    {
        CIRCUIT_SUMMARY_KIND,
        INTERVENTION_SUMMARY_KIND,
        INTERVENTION_RESULT_SUMMARY_KIND,
    }
)


class SchemaError(ValueError):
    """Raised when a summary document does not match its declared kind."""


def assert_json_safe(value: Any, *, path: str = "$") -> None:
    """Raise :class:`SchemaError` if *value* cannot round-trip through JSON."""
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"Value at {path} is not JSON-serializable: {exc}") from exc


def _require_mapping(value: Any, path: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object")
    return value


def _require_list(value: Any, path: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SchemaError(f"{path} must be an array")
    return value


def _require_keys(doc: Mapping, keys: Sequence[str], path: str) -> None:
    missing = [key for key in keys if key not in doc]
    if missing:
        raise SchemaError(f"{path} missing required keys: {missing}")


def _validate_circuit_summary(doc: Mapping) -> None:
    _require_keys(
        doc,
        ("kind", "version", "input", "model", "targets", "nodeCounts", "metrics", "topFeatures"),
        "$",
    )
    if doc["kind"] != CIRCUIT_SUMMARY_KIND:
        raise SchemaError(f"Expected kind {CIRCUIT_SUMMARY_KIND!r}, got {doc['kind']!r}")
    if int(doc["version"]) != 1:
        raise SchemaError(f"Unsupported circuit summary version: {doc['version']}")

    input_doc = _require_mapping(doc["input"], "$.input")
    _require_keys(input_doc, ("text", "tokenIds", "tokenCount"), "$.input")
    _require_list(input_doc["tokenIds"], "$.input.tokenIds")

    model_doc = _require_mapping(doc["model"], "$.model")
    _require_keys(model_doc, ("name", "tokenizer", "layers", "vocabSize", "scan"), "$.model")

    targets = _require_list(doc["targets"], "$.targets")
    for index, target in enumerate(targets):
        target_doc = _require_mapping(target, f"$.targets[{index}]")
        _require_keys(target_doc, ("token", "vocabIndex", "probability"), f"$.targets[{index}]")

    node_counts = _require_mapping(doc["nodeCounts"], "$.nodeCounts")
    _require_keys(
        node_counts,
        (
            "selectedFeatures",
            "activeFeatures",
            "errorNodes",
            "tokenNodes",
            "logitNodes",
            "totalNodes",
        ),
        "$.nodeCounts",
    )

    metrics = _require_mapping(doc["metrics"], "$.metrics")
    _require_keys(metrics, ("replacementScore", "completenessScore"), "$.metrics")

    top_features = _require_list(doc["topFeatures"], "$.topFeatures")
    for index, feature in enumerate(top_features):
        feature_doc = _require_mapping(feature, f"$.topFeatures[{index}]")
        _require_keys(
            feature_doc,
            ("layer", "position", "feature", "influence", "activation"),
            f"$.topFeatures[{index}]",
        )

    if "pruning" in doc and doc["pruning"] is not None:
        pruning = _require_mapping(doc["pruning"], "$.pruning")
        _require_keys(
            pruning,
            (
                "nodeThreshold",
                "edgeThreshold",
                "keptNodeCount",
                "keptEdgeCount",
                "maxCumulativeScore",
            ),
            "$.pruning",
        )


def _validate_intervention_summary(doc: Mapping) -> None:
    _require_keys(
        doc,
        (
            "kind",
            "version",
            "sourceGraph",
            "runtime",
            "interventionType",
            "value",
            "interventionCount",
            "interventions",
        ),
        "$",
    )
    if doc["kind"] != INTERVENTION_SUMMARY_KIND:
        raise SchemaError(f"Expected kind {INTERVENTION_SUMMARY_KIND!r}, got {doc['kind']!r}")
    if int(doc["version"]) != 1:
        raise SchemaError(f"Unsupported intervention summary version: {doc['version']}")

    source = _require_mapping(doc["sourceGraph"], "$.sourceGraph")
    _require_keys(source, ("kind", "input", "model", "targets"), "$.sourceGraph")
    if source["kind"] != CIRCUIT_SUMMARY_KIND:
        raise SchemaError("$.sourceGraph.kind must be circuit-tracer.summary.v1")

    runtime = _require_mapping(doc["runtime"], "$.runtime")
    _require_keys(runtime, ("function", "tupleFormat", "executed"), "$.runtime")

    interventions = _require_list(doc["interventions"], "$.interventions")
    if len(interventions) != int(doc["interventionCount"]):
        raise SchemaError("$.interventionCount does not match $.interventions length")
    for index, item in enumerate(interventions):
        item_doc = _require_mapping(item, f"$.interventions[{index}]")
        _require_keys(
            item_doc,
            ("layer", "position", "feature", "value", "sourceInfluence", "sourceActivation"),
            f"$.interventions[{index}]",
        )


def _validate_intervention_result_summary(doc: Mapping) -> None:
    _require_keys(doc, ("kind", "version", "sourcePlan", "runtime", "effects", "metadata"), "$")
    if doc["kind"] != INTERVENTION_RESULT_SUMMARY_KIND:
        raise SchemaError(
            f"Expected kind {INTERVENTION_RESULT_SUMMARY_KIND!r}, got {doc['kind']!r}"
        )
    if int(doc["version"]) != 1:
        raise SchemaError(f"Unsupported intervention-result summary version: {doc['version']}")

    source_plan = _require_mapping(doc["sourcePlan"], "$.sourcePlan")
    _require_keys(
        source_plan,
        ("kind", "sourceGraph", "interventionType", "value", "interventionCount", "interventions"),
        "$.sourcePlan",
    )

    effects = _require_mapping(doc["effects"], "$.effects")
    _require_keys(
        effects,
        (
            "vocabSize",
            "topBefore",
            "topAfter",
            "topTokenChanged",
            "maxAbsLogitDelta",
            "meanAbsLogitDelta",
            "l2LogitDelta",
            "targetEffects",
            "topLogitShifts",
        ),
        "$.effects",
    )
    _require_list(effects["targetEffects"], "$.effects.targetEffects")
    _require_list(effects["topLogitShifts"], "$.effects.topLogitShifts")


_VALIDATORS = {
    CIRCUIT_SUMMARY_KIND: _validate_circuit_summary,
    INTERVENTION_SUMMARY_KIND: _validate_intervention_summary,
    INTERVENTION_RESULT_SUMMARY_KIND: _validate_intervention_result_summary,
}


def validate_summary(doc: Mapping, *, expected_kind: str | None = None) -> str:
    """Validate a summary document and return its kind.

    Args:
        doc: Parsed JSON object.
        expected_kind: If set, require this exact kind string.

    Returns:
        The document ``kind``.

    Raises:
        SchemaError: On structural or kind mismatches.
    """
    doc = _require_mapping(doc, "$")
    if "kind" not in doc:
        raise SchemaError("Missing required key: kind")

    kind = str(doc["kind"])
    if expected_kind is not None and kind != expected_kind:
        raise SchemaError(f"Expected kind {expected_kind!r}, got {kind!r}")
    if kind not in _VALIDATORS:
        raise SchemaError(f"Unknown summary kind: {kind!r}")

    assert_json_safe(doc)
    _VALIDATORS[kind](doc)
    return kind


def load_summary(path: str | Path, *, expected_kind: str | None = None) -> dict:
    """Load and validate a summary JSON file."""
    with open(path, encoding="utf-8") as handle:
        doc = json.load(handle)
    validate_summary(doc, expected_kind=expected_kind)
    return doc


def dump_summary(doc: Mapping, path: str | Path, *, expected_kind: str | None = None) -> None:
    """Validate and write a summary JSON file."""
    validate_summary(doc, expected_kind=expected_kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2)
        handle.write("\n")
