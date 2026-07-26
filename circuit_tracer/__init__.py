from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("circuit-tracer")
except PackageNotFoundError:
    # Source tree / editable install before metadata is available.
    __version__ = "0.0.0"

if TYPE_CHECKING:
    from circuit_tracer.analysis import (
        ComparisonResult,
        compare_graphs,
        find_common_circuit,
        get_top_features,
        graph_to_interventions,
        summarize_graph,
        summarize_intervention_results,
        summarize_interventions,
    )
    from circuit_tracer.attribution.attribute import attribute, attribute_batch
    from circuit_tracer.attribution.targets import CustomTarget
    from circuit_tracer.dataset import CircuitDataset, CircuitRecord, compare_datasets
    from circuit_tracer.graph import Graph, PruneResult, compute_graph_scores, prune_graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.replacement_model.common import Intervention
    from circuit_tracer.schema import SchemaError, dump_summary, load_summary, validate_summary
    from circuit_tracer.steering import (
        load_intervention_plan,
        run_intervention_plan,
        save_intervention_plan,
        steer,
        validate_intervention,
    )
    from circuit_tracer.utils.create_graph_files import create_graph_files, export_graph_for_viz
    from circuit_tracer.utils.tl_nnsight_mapping import (
        ModelMapping,
        auto_detect_mapping,
        get_supported_architectures,
        register_model,
        validate_mapping,
    )

__all__ = [
    # ── Core ────────────────────────────────────────────────────────
    "ReplacementModel",
    "Graph",
    "attribute",
    "__version__",
    # ── Analysis ────────────────────────────────────────────────────
    "get_top_features",
    "summarize_graph",
    "summarize_interventions",
    "summarize_intervention_results",
    "prune_graph",
    "PruneResult",
    "compute_graph_scores",
    "CustomTarget",
    "create_graph_files",
    "export_graph_for_viz",
    "validate_summary",
    "load_summary",
    "dump_summary",
    "SchemaError",
    # ── Intervention ────────────────────────────────────────────────
    "Intervention",
    "graph_to_interventions",
    "steer",
    "save_intervention_plan",
    "load_intervention_plan",
    "run_intervention_plan",
    "validate_intervention",
    # ── Batch & comparison ──────────────────────────────────────────
    "attribute_batch",
    "compare_graphs",
    "find_common_circuit",
    "ComparisonResult",
    "CircuitDataset",
    "CircuitRecord",
    "compare_datasets",
    # ── Model extensibility ─────────────────────────────────────────
    "ModelMapping",
    "register_model",
    "get_supported_architectures",
    "auto_detect_mapping",
    "validate_mapping",
]


def __getattr__(name):
    _lazy_imports = {
        # Core
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
        # Analysis
        "get_top_features": ("circuit_tracer.analysis", "get_top_features"),
        "summarize_graph": ("circuit_tracer.analysis", "summarize_graph"),
        "summarize_interventions": (
            "circuit_tracer.analysis",
            "summarize_interventions",
        ),
        "summarize_intervention_results": (
            "circuit_tracer.analysis",
            "summarize_intervention_results",
        ),
        "prune_graph": ("circuit_tracer.graph", "prune_graph"),
        "PruneResult": ("circuit_tracer.graph", "PruneResult"),
        "compute_graph_scores": ("circuit_tracer.graph", "compute_graph_scores"),
        "CustomTarget": ("circuit_tracer.attribution.targets", "CustomTarget"),
        "create_graph_files": ("circuit_tracer.utils.create_graph_files", "create_graph_files"),
        "export_graph_for_viz": (
            "circuit_tracer.utils.create_graph_files",
            "export_graph_for_viz",
        ),
        "validate_summary": ("circuit_tracer.schema", "validate_summary"),
        "load_summary": ("circuit_tracer.schema", "load_summary"),
        "dump_summary": ("circuit_tracer.schema", "dump_summary"),
        "SchemaError": ("circuit_tracer.schema", "SchemaError"),
        # Intervention
        "Intervention": ("circuit_tracer.replacement_model.common", "Intervention"),
        "graph_to_interventions": ("circuit_tracer.analysis", "graph_to_interventions"),
        "steer": ("circuit_tracer.steering", "steer"),
        "save_intervention_plan": ("circuit_tracer.steering", "save_intervention_plan"),
        "load_intervention_plan": ("circuit_tracer.steering", "load_intervention_plan"),
        "run_intervention_plan": ("circuit_tracer.steering", "run_intervention_plan"),
        "validate_intervention": ("circuit_tracer.steering", "validate_intervention"),
        # Batch & comparison
        "attribute_batch": ("circuit_tracer.attribution.attribute", "attribute_batch"),
        "compare_graphs": ("circuit_tracer.analysis", "compare_graphs"),
        "find_common_circuit": ("circuit_tracer.analysis", "find_common_circuit"),
        "ComparisonResult": ("circuit_tracer.analysis", "ComparisonResult"),
        "CircuitDataset": ("circuit_tracer.dataset", "CircuitDataset"),
        "CircuitRecord": ("circuit_tracer.dataset", "CircuitRecord"),
        "compare_datasets": ("circuit_tracer.dataset", "compare_datasets"),
        # Model extensibility
        "ModelMapping": ("circuit_tracer.utils.tl_nnsight_mapping", "ModelMapping"),
        "register_model": ("circuit_tracer.utils.tl_nnsight_mapping", "register_model"),
        "get_supported_architectures": (
            "circuit_tracer.utils.tl_nnsight_mapping",
            "get_supported_architectures",
        ),
        "auto_detect_mapping": (
            "circuit_tracer.utils.tl_nnsight_mapping",
            "auto_detect_mapping",
        ),
        "validate_mapping": ("circuit_tracer.utils.tl_nnsight_mapping", "validate_mapping"),
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
