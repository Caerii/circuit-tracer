from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("circuit-tracer")
except PackageNotFoundError:
    # Editable / source installs — read directly from the single-source file.
    from circuit_tracer._version import __version__  # type: ignore[assignment]

if TYPE_CHECKING:
    from circuit_tracer.analysis import (
        ComparisonResult,
        compare_graphs,
        find_common_circuit,
        get_top_features,
        graph_to_interventions,
    )
    from circuit_tracer.attribution.attribute import attribute, attribute_batch
    from circuit_tracer.attribution.targets import CustomTarget
    from circuit_tracer.graph import Graph, PruneResult, compute_graph_scores, prune_graph
    from circuit_tracer.replacement_model import ReplacementModel
    from circuit_tracer.replacement_model.common import Intervention
    from circuit_tracer.utils.create_graph_files import create_graph_files
    from circuit_tracer.utils.tl_nnsight_mapping import (
        ModelMapping,
        auto_detect_mapping,
        get_supported_architectures,
        register_model,
    )

__all__ = [
    # ── Core (existing) ─────────────────────────────────────────────
    "ReplacementModel",
    "Graph",
    "attribute",
    "__version__",
    # ── Analysis (v0.6.0) ───────────────────────────────────────────
    "get_top_features",
    "prune_graph",
    "PruneResult",
    "compute_graph_scores",
    "CustomTarget",
    "create_graph_files",
    # ── Intervention (v0.7.0) ───────────────────────────────────────
    "Intervention",
    "graph_to_interventions",
    # ── Batch & comparison (v0.8.0) ─────────────────────────────────
    "attribute_batch",
    "compare_graphs",
    "find_common_circuit",
    "ComparisonResult",
    # ── Model extensibility (v0.9.0) ────────────────────────────────
    "ModelMapping",
    "register_model",
    "get_supported_architectures",
    "auto_detect_mapping",
]


def __getattr__(name):
    _lazy_imports = {
        # Core
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
        # Analysis
        "get_top_features": ("circuit_tracer.analysis", "get_top_features"),
        "prune_graph": ("circuit_tracer.graph", "prune_graph"),
        "PruneResult": ("circuit_tracer.graph", "PruneResult"),
        "compute_graph_scores": ("circuit_tracer.graph", "compute_graph_scores"),
        "CustomTarget": ("circuit_tracer.attribution.targets", "CustomTarget"),
        "create_graph_files": ("circuit_tracer.utils.create_graph_files", "create_graph_files"),
        # Intervention
        "Intervention": ("circuit_tracer.replacement_model.common", "Intervention"),
        "graph_to_interventions": ("circuit_tracer.analysis", "graph_to_interventions"),
        # Batch & comparison
        "attribute_batch": ("circuit_tracer.attribution.attribute", "attribute_batch"),
        "compare_graphs": ("circuit_tracer.analysis", "compare_graphs"),
        "find_common_circuit": ("circuit_tracer.analysis", "find_common_circuit"),
        "ComparisonResult": ("circuit_tracer.analysis", "ComparisonResult"),
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
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
