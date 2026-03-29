from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.attribution.targets import (
        AttributionTargets,
        CustomTarget,
        LogitTarget,
        TargetSpec,
    )

__all__ = [
    "attribute",
    "AttributionTargets",
    "CustomTarget",
    "LogitTarget",
    "TargetSpec",
]


def __getattr__(name):
    _lazy_imports = {
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "AttributionTargets": ("circuit_tracer.attribution.targets", "AttributionTargets"),
        "CustomTarget": ("circuit_tracer.attribution.targets", "CustomTarget"),
        "LogitTarget": ("circuit_tracer.attribution.targets", "LogitTarget"),
        "TargetSpec": ("circuit_tracer.attribution.targets", "TargetSpec"),
    }

    if name in _lazy_imports:
        module_name, attr_name = _lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
