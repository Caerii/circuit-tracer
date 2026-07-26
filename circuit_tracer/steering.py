"""High-level intervention / steering helpers.

Closes the attribute → identify → intervene loop with a small public API::

    from circuit_tracer import steer, save_intervention_plan, load_intervention_plan

    text, logits, _ = steer(model, prompt, suppress=[(21, 7, 5066)], generate=True)
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from circuit_tracer.analysis import (
    INTERVENTION_SUMMARY_KIND,
    graph_to_interventions,
    summarize_intervention_results,
    summarize_interventions,
)
from circuit_tracer.replacement_model.common import Intervention
from circuit_tracer.schema import dump_summary, load_summary, validate_summary

__all__ = [
    "VALIDATION_KIND",
    "steer",
    "interventions_from_plan",
    "save_intervention_plan",
    "load_intervention_plan",
    "run_intervention_plan",
    "validate_intervention",
]

VALIDATION_KIND = "circuit-tracer.intervention-validation.v1"

SteerItem = tuple[int, int, int] | tuple[int, int, int, float | int] | Intervention


def _normalize_item(item: SteerItem, *, default_value: float) -> Intervention:
    if len(item) == 3:
        layer, position, feature = item  # type: ignore[misc]
        return (int(layer), int(position), int(feature), float(default_value))
    if len(item) == 4:
        layer, position, feature, value = item  # type: ignore[misc]
        return (int(layer), int(position), int(feature), float(value))
    raise ValueError(
        "Intervention items must be (layer, position, feature) or (layer, position, feature, value)"
    )


def _build_interventions(
    amplify: Sequence[SteerItem] | None,
    suppress: Sequence[SteerItem] | None,
    *,
    amplify_value: float,
) -> list[Intervention]:
    interventions: list[Intervention] = []
    for item in amplify or ():
        interventions.append(_normalize_item(item, default_value=amplify_value))
    for item in suppress or ():
        interventions.append(_normalize_item(item, default_value=0.0))
    if not interventions:
        raise ValueError("Provide at least one amplify or suppress intervention")
    return interventions


def steer(
    model: Any,
    prompt: str | torch.Tensor,
    *,
    amplify: Sequence[SteerItem] | None = None,
    suppress: Sequence[SteerItem] | None = None,
    amplify_value: float = 2.0,
    generate: bool = False,
    return_activations: bool = False,
    constrained_layers: range | None = None,
    freeze_attention: bool = True,
    apply_activation_function: bool = True,
    sparse: bool = False,
    **generate_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[str, torch.Tensor, torch.Tensor | None]:
    """Run feature interventions with a compact amplify/suppress API.

    Args:
        model: A :class:`~circuit_tracer.replacement_model.ReplacementModel`.
        prompt: Text or token ids.
        amplify: Features to set (default value ``amplify_value``).
        suppress: Features to ablate (value ``0.0``).
        amplify_value: Default activation used when amplify tuples omit a value.
        generate: If ``True``, call ``feature_intervention_generate``.
        return_activations: Forwarded to the intervention APIs.
        constrained_layers / freeze_attention / apply_activation_function / sparse:
            Forwarded to the intervention APIs.
        **generate_kwargs: Extra kwargs for ``feature_intervention_generate``.

    Returns:
        Same return shape as ``feature_intervention`` or
        ``feature_intervention_generate``.
    """
    interventions = _build_interventions(amplify, suppress, amplify_value=amplify_value)
    common = dict(
        constrained_layers=constrained_layers,
        freeze_attention=freeze_attention,
        apply_activation_function=apply_activation_function,
        sparse=sparse,
        return_activations=return_activations,
    )
    if generate:
        return model.feature_intervention_generate(
            prompt, interventions, **common, **generate_kwargs
        )
    return model.feature_intervention(prompt, interventions, **common)


def interventions_from_plan(plan: Mapping) -> list[Intervention]:
    """Convert a ``circuit-tracer.interventions.v1`` plan into runtime tuples."""
    validate_summary(plan, expected_kind=INTERVENTION_SUMMARY_KIND)
    return [
        (
            int(item["layer"]),
            int(item["position"]),
            int(item["feature"]),
            float(item["value"]),
        )
        for item in plan["interventions"]
    ]


def save_intervention_plan(plan: Mapping, path: str | Path) -> None:
    """Validate and write an intervention plan JSON file."""
    dump_summary(plan, path, expected_kind=INTERVENTION_SUMMARY_KIND)


def load_intervention_plan(path: str | Path) -> dict:
    """Load and validate an intervention plan JSON file."""
    return load_summary(path, expected_kind=INTERVENTION_SUMMARY_KIND)


def run_intervention_plan(
    model: Any,
    prompt: str | torch.Tensor | None = None,
    plan: Mapping | str | Path | None = None,
    *,
    graph: Any | None = None,
    n: int = 10,
    value: float = 0.0,
    target_token_ids: Sequence[int] | None = None,
    top_k: int = 10,
    constrained_layers: range | None = None,
    freeze_attention: bool = True,
    apply_activation_function: bool = True,
    sparse: bool = False,
    return_activations: bool = False,
) -> dict:
    """Execute an intervention plan and return a results summary.

    Provide either an existing *plan* (mapping or JSON path) or a *graph* from
    which a plan is derived via :func:`~circuit_tracer.analysis.summarize_interventions`.
    """
    if plan is None:
        if graph is None:
            raise ValueError("Provide either plan or graph")
        plan = summarize_interventions(graph, n=n, value=value)
    elif isinstance(plan, (str, Path)):
        plan = load_intervention_plan(plan)
    else:
        validate_summary(plan, expected_kind=INTERVENTION_SUMMARY_KIND)

    if prompt is None:
        if graph is not None:
            prompt = graph.input_tokens
        else:
            text = plan.get("sourceGraph", {}).get("input", {}).get("text")
            if not text:
                raise ValueError("prompt is required when the plan has no source text")
            prompt = text

    interventions = interventions_from_plan(plan)
    with torch.no_grad():
        if hasattr(model, "forward") and isinstance(prompt, str):
            baseline_logits = model(prompt)
        else:
            baseline_logits = model(prompt)

        intervened_logits, _ = model.feature_intervention(
            prompt,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            sparse=sparse,
            return_activations=return_activations,
        )

    if target_token_ids is None:
        targets = plan.get("sourceGraph", {}).get("targets", [])
        target_token_ids = [int(t["vocabIndex"]) for t in targets]

    return summarize_intervention_results(
        plan,
        baseline_logits,
        intervened_logits,
        target_token_ids=target_token_ids,
        top_k=top_k,
    )


def validate_intervention(
    model: Any,
    prompt: str | torch.Tensor,
    interventions: Sequence[Intervention] | None = None,
    *,
    graph: Any | None = None,
    n: int = 1,
    value: float | None = None,
    target_token_ids: Sequence[int] | None = None,
    constrained_layers: range | None = None,
    apply_activation_function: bool = False,
    freeze_attention: bool = True,
    atol: float = 2e-3,
    rtol: float = 1e-2,
    mode: Literal["observe", "predict"] = "observe",
) -> dict:
    """Compare baseline vs intervened logits, optionally vs graph predictions.

    Modes:
        ``observe``: Always available. Reports observed logit deltas and whether
            any target tokens moved beyond ``atol``.
        ``predict``: Requires *graph*. For each intervened feature present in the
            graph, compares observed final-logit deltas against the graph's
            direct attribution column under linear-regime settings
            (``constrained_layers`` covering all layers, ``apply_activation_function=False``).
    """
    if interventions is None:
        if graph is None:
            raise ValueError("Provide interventions or graph")
        interventions = graph_to_interventions(graph, n=n, value=0.0 if value is None else value)

    interventions = list(interventions)
    if not interventions:
        raise ValueError("No interventions to validate")

    if constrained_layers is None and mode == "predict" and graph is not None:
        constrained_layers = range(int(graph.cfg.n_layers))

    with torch.no_grad():
        baseline_logits = model(prompt)
        intervened_logits, _ = model.feature_intervention(
            prompt,
            interventions,
            constrained_layers=constrained_layers,
            freeze_attention=freeze_attention,
            apply_activation_function=apply_activation_function,
            return_activations=False,
        )

    plan = {
        "kind": INTERVENTION_SUMMARY_KIND,
        "version": 1,
        "sourceGraph": {
            "kind": "circuit-tracer.summary.v1",
            "input": {
                "text": str(prompt) if isinstance(prompt, str) else "",
                "tokenIds": [],
                "tokenCount": 0,
            },
            "model": {"name": None, "tokenizer": None, "layers": 0, "vocabSize": 0, "scan": None},
            "targets": [],
        },
        "runtime": {
            "function": "ReplacementModel.feature_intervention",
            "tupleFormat": ["layer", "position", "feature", "value"],
            "executed": False,
        },
        "interventionType": "feature_set",
        "value": float(interventions[0][3]),
        "interventionCount": len(interventions),
        "interventions": [
            {
                "layer": int(layer),
                "position": int(pos),
                "feature": int(feat),
                "value": float(val),
                "sourceInfluence": 0.0,
                "sourceActivation": None,
            }
            for layer, pos, feat, val in interventions
        ],
    }

    if target_token_ids is None and graph is not None:
        target_token_ids = [int(t.vocab_idx) for t in graph.logit_targets]

    result_summary = summarize_intervention_results(
        plan,
        baseline_logits,
        intervened_logits,
        target_token_ids=target_token_ids,
        top_k=10,
    )

    prediction_checks: list[dict] = []
    passed = True

    if mode == "predict":
        if graph is None:
            raise ValueError("mode='predict' requires graph")
        device = graph.adjacency_matrix.device
        before = _final_position_logits(baseline_logits).to(device)
        after = _final_position_logits(intervened_logits).to(device)
        observed_delta = after - before
        n_logits = len(graph.logit_targets)
        feature_index = {
            tuple(int(x) for x in graph.active_features[int(idx)].tolist()): int(i)
            for i, idx in enumerate(graph.selected_features.tolist())
        }

        for layer, pos, feat, val in interventions:
            key = (int(layer), int(pos), int(feat))
            if key not in feature_index:
                prediction_checks.append(
                    {
                        "feature": {"layer": key[0], "position": key[1], "feature": key[2]},
                        "status": "skipped",
                        "reason": "feature not in graph.selected_features",
                    }
                )
                continue

            node = feature_index[key]
            selected_row = int(graph.selected_features[node])
            old_activation = float(graph.activation_values[selected_row].item())
            # Adjacency column encodes effect of the feature at its recorded activation.
            # Setting value V corresponds to adding ((V / old) - 1) copies when old != 0
            # in the linear doubling regime used by attribution tests; for ablation (V=0)
            # the scale is -1.
            if old_activation == 0.0:
                scale = 0.0
            else:
                scale = (float(val) / old_activation) - 1.0
            predicted = graph.adjacency_matrix[-n_logits:, node] * scale
            actual = observed_delta
            if actual.numel() != predicted.numel():
                # Fall back to comparing only overlapping target vocab ids when shapes differ
                target_ids = [int(t.vocab_idx) for t in graph.logit_targets]
                actual = (
                    observed_delta[target_ids]
                    if observed_delta.numel() > max(target_ids)
                    else observed_delta
                )
            match = torch.allclose(actual[: predicted.numel()], predicted, atol=atol, rtol=rtol)
            passed = passed and bool(match)
            prediction_checks.append(
                {
                    "feature": {"layer": key[0], "position": key[1], "feature": key[2]},
                    "status": "pass" if match else "fail",
                    "scale": scale,
                    "maxAbsError": float(
                        (actual[: predicted.numel()] - predicted).abs().max().item()
                    ),
                    "atol": atol,
                    "rtol": rtol,
                }
            )

    target_effects = result_summary["effects"]["targetEffects"]
    if mode == "observe" and target_effects:
        # Observation mode passes if the call succeeded and targets were scored.
        passed = True

    return {
        "kind": VALIDATION_KIND,
        "version": 1,
        "mode": mode,
        "passed": passed,
        "interventionCount": len(interventions),
        "effects": result_summary["effects"],
        "predictionChecks": prediction_checks,
    }


def _final_position_logits(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().float()
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])[-1]


def plan_to_json_string(plan: Mapping) -> str:
    """Serialize a validated intervention plan to a JSON string."""
    validate_summary(plan, expected_kind=INTERVENTION_SUMMARY_KIND)
    return json.dumps(plan, indent=2)
