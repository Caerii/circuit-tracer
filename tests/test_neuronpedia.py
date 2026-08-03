"""Unit tests for Neuronpedia helpers (SDK mocked; no network)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from circuit_tracer import neuronpedia as np_mod


def test_require_neuronpedia_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "neuronpedia" or name.startswith("neuronpedia."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match="circuit-tracer\\[neuronpedia\\]"):
        np_mod.require_neuronpedia()


def test_upload_graph_to_neuronpedia_json(tmp_path, monkeypatch):
    graph_json = tmp_path / "demo.json"
    graph_json.write_text(json.dumps({"metadata": {"slug": "demo"}}), encoding="utf-8")

    fake_meta = SimpleNamespace(
        slug="demo",
        model_id="gemma-2-2b",
        url="https://neuronpedia.org/gemma-2-2b/graph?slug=demo",
        url_embed="https://neuronpedia.org/gemma-2-2b/graph?slug=demo&embed=true",
    )
    fake_np_graph = MagicMock()
    fake_np_graph.upload_file.return_value = fake_meta

    monkeypatch.setenv("NEURONPEDIA_API_KEY", "test-key")
    with (
        patch.object(np_mod, "require_neuronpedia", return_value=(fake_np_graph, MagicMock())),
        patch.object(np_mod, "_apply_api_key"),
    ):
        result = np_mod.upload_graph_to_neuronpedia(graph_json, model_id="gemma-2-2b")

    assert result["kind"] == "circuit-tracer.neuronpedia-upload.v1"
    assert result["url"].startswith("https://neuronpedia.org/")
    fake_np_graph.upload_file.assert_called_once()


def test_fetch_feature_mocked(monkeypatch):
    fake_feature_cls = MagicMock()
    fake_feature_cls.get.return_value = {"modelId": "gemma-2-2b", "index": "1"}
    monkeypatch.setenv("NEURONPEDIA_API_KEY", "test-key")
    with (
        patch.object(np_mod, "require_neuronpedia", return_value=(MagicMock(), fake_feature_cls)),
        patch.object(np_mod, "_apply_api_key"),
    ):
        payload = np_mod.fetch_feature("gemma-2-2b", "0-gemmascope-res-16k", 1)
    assert payload["modelId"] == "gemma-2-2b"
