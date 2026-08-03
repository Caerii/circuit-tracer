"""Neuronpedia upload and feature-fetch helpers.

Requires the optional extra::

    pip install circuit-tracer[neuronpedia]

Authentication uses ``NEURONPEDIA_API_KEY`` (or an explicit ``api_key`` argument).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "upload_graph_to_neuronpedia",
    "fetch_feature",
    "require_neuronpedia",
]


def require_neuronpedia():
    """Import the official Neuronpedia SDK or raise a helpful error."""
    try:
        import neuronpedia  # noqa: F401  # type: ignore[import-not-found]
        from neuronpedia.np_graph_metadata import NPGraphMetadata  # type: ignore[import-not-found]
        from neuronpedia.np_sae_feature import SAEFeature  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Neuronpedia support requires the optional extra. Install with: "
            "pip install 'circuit-tracer[neuronpedia]'"
        ) from exc
    return NPGraphMetadata, SAEFeature


def _apply_api_key(api_key: str | None) -> None:
    key = api_key or os.environ.get("NEURONPEDIA_API_KEY")
    if not key:
        raise ValueError(
            "Neuronpedia API key required. Set NEURONPEDIA_API_KEY or pass api_key=..."
        )
    import neuronpedia  # type: ignore[import-not-found]

    set_api_key = getattr(neuronpedia, "set_api_key", None)
    if callable(set_api_key):
        set_api_key(key)
    else:
        os.environ["NEURONPEDIA_API_KEY"] = key


def _ensure_graph_json(
    graph_or_json_path: Any,
    *,
    slug: str | None,
    output_dir: str | Path | None,
    node_threshold: float,
    edge_threshold: float,
    scan_name: str | list[str] | None,
) -> tuple[Path, str]:
    path = (
        Path(str(graph_or_json_path))
        if not hasattr(graph_or_json_path, "adjacency_matrix")
        else None
    )
    if path is not None and path.suffix.lower() == ".json":
        return path, path.stem

    from circuit_tracer.utils.create_graph_files import create_graph_files

    if slug is None:
        if path is not None:
            slug = path.stem
        else:
            slug = "circuit-tracer-graph"

    tmp_owned = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="circuit-tracer-np-")
        tmp_owned = True
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_graph_files(
        graph_or_path=graph_or_json_path,
        slug=slug,
        output_path=str(output_dir),
        scan_name=scan_name,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    json_path = output_dir / f"{slug}.json"
    if tmp_owned:
        # Caller still gets a real file path; tempfile dir is left for OS cleanup
        # after process exit. Prefer passing output_dir for persistence.
        pass
    return json_path, slug


def upload_graph_to_neuronpedia(
    graph_or_json_path: Any,
    *,
    model_id: str | None = None,
    slug: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    scan_name: str | list[str] | None = None,
) -> dict[str, Any]:
    """Upload a circuit-tracer graph (``.pt`` / Graph / JSON) to Neuronpedia.

    Args:
        graph_or_json_path: ``Graph``, ``.pt`` path, or frontend ``.json`` path.
        model_id: Optional model id override for the returned metadata dict.
        slug: Slug used when exporting frontend JSON (must be globally unique
            on Neuronpedia when uploading).
        api_key: Neuronpedia API key (defaults to ``NEURONPEDIA_API_KEY``).
        output_dir: Where to write intermediate JSON when exporting from a Graph.
        node_threshold / edge_threshold: Pruning thresholds for export.
        scan_name: Transcoder scan identifier when the Graph lacks one.

    Returns:
        JSON-safe metadata including ``url`` / ``urlEmbed`` when provided by the SDK.
    """
    NPGraphMetadata, _ = require_neuronpedia()
    _apply_api_key(api_key)

    json_path, resolved_slug = _ensure_graph_json(
        graph_or_json_path,
        slug=slug,
        output_dir=output_dir,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        scan_name=scan_name,
    )

    if hasattr(NPGraphMetadata, "upload_file"):
        metadata = NPGraphMetadata.upload_file(str(json_path))
    else:
        metadata = NPGraphMetadata.upload(json_path.read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "kind": "circuit-tracer.neuronpedia-upload.v1",
        "version": 1,
        "slug": getattr(metadata, "slug", None) or resolved_slug,
        "modelId": model_id or getattr(metadata, "model_id", None),
        "url": getattr(metadata, "url", None),
        "urlEmbed": getattr(metadata, "url_embed", None),
        "localJson": str(json_path),
    }
    return result


def fetch_feature(
    model_id: str,
    source: str,
    index: str | int,
    *,
    api_key: str | None = None,
) -> Mapping[str, Any]:
    """Fetch a Neuronpedia SAE feature record (explanations / activations metadata)."""
    _, SAEFeature = require_neuronpedia()
    _apply_api_key(api_key)
    feature = SAEFeature.get(model_id, source, str(index))
    if hasattr(feature, "model_dump"):
        return feature.model_dump()
    if hasattr(feature, "dict"):
        return feature.dict()
    if isinstance(feature, Mapping):
        return feature
    # Best-effort serialization
    return json.loads(json.dumps(feature, default=lambda o: getattr(o, "__dict__", str(o))))
