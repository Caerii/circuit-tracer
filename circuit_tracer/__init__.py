from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("circuit-tracer")
except PackageNotFoundError:
    # Editable / source installs — read directly from the single-source file.
    from circuit_tracer._version import __version__  # type: ignore[assignment]

if TYPE_CHECKING:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.graph import Graph
    from circuit_tracer.replacement_model import ReplacementModel

__all__ = [
    "ReplacementModel",
    "Graph",
    "attribute",
    "__version__",
]


def __getattr__(name):
    _lazy_imports = {
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "Graph": ("circuit_tracer.graph", "Graph"),
        "ReplacementModel": ("circuit_tracer.replacement_model", "ReplacementModel"),
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
