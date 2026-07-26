"""Dataset-scale circuit analysis containers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from circuit_tracer.analysis import (
    ComparisonResult,
    compare_graphs,
    find_common_circuit,
    get_top_features,
)
from circuit_tracer.attribution.attribute import attribute_batch
from circuit_tracer.graph import Graph

__all__ = [
    "CircuitRecord",
    "CircuitDataset",
    "DatasetDriftResult",
    "compare_datasets",
]

_MANIFEST_NAME = "manifest.json"


@dataclass
class CircuitRecord:
    """One prompt and its optional attribution graph."""

    prompt: str
    graph: Graph | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    graph_path: str | None = None


@dataclass
class DatasetDriftResult:
    """Comparison between a baseline and current circuit dataset."""

    paired: bool
    n_baseline: int
    n_current: int
    shared_features: list[tuple[int, int, int]]
    baseline_only: list[tuple[int, int, int]]
    current_only: list[tuple[int, int, int]]
    comparison: ComparisonResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "circuit-tracer.dataset-drift.v1",
            "version": 1,
            "paired": self.paired,
            "nBaseline": self.n_baseline,
            "nCurrent": self.n_current,
            "sharedFeatures": [list(f) for f in self.shared_features],
            "baselineOnly": [list(f) for f in self.baseline_only],
            "currentOnly": [list(f) for f in self.current_only],
        }


class CircuitDataset:
    """Lightweight collection of prompts/graphs with save/load support."""

    def __init__(self, records: Sequence[CircuitRecord] | None = None):
        self.records: list[CircuitRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    @property
    def prompts(self) -> list[str]:
        return [record.prompt for record in self.records]

    @property
    def labels(self) -> list[str | None]:
        return [record.label for record in self.records]

    @property
    def graphs(self) -> list[Graph]:
        graphs = [record.graph for record in self.records if record.graph is not None]
        if len(graphs) != len(self.records):
            raise ValueError("Not all records have loaded graphs")
        return graphs

    @classmethod
    def from_prompts(
        cls,
        prompts: Sequence[str],
        model: Any,
        *,
        labels: Sequence[str | None] | None = None,
        metadata: Sequence[dict[str, Any]] | None = None,
        attribute: bool = True,
        max_workers: int = 1,
        **attribute_kwargs: Any,
    ) -> CircuitDataset:
        """Build a dataset from prompts, optionally running attribution."""
        if labels is not None and len(labels) != len(prompts):
            raise ValueError("labels length must match prompts")
        if metadata is not None and len(metadata) != len(prompts):
            raise ValueError("metadata length must match prompts")

        graphs: list[Graph | None]
        if attribute:
            graphs = list(
                attribute_batch(
                    prompts,
                    model,
                    max_workers=max_workers,
                    **attribute_kwargs,
                )
            )
        else:
            graphs = [None] * len(prompts)

        records = [
            CircuitRecord(
                prompt=str(prompt),
                graph=graphs[index],
                label=None if labels is None else labels[index],
                metadata={} if metadata is None else dict(metadata[index]),
            )
            for index, prompt in enumerate(prompts)
        ]
        return cls(records)

    def save(self, path: str | Path) -> None:
        """Save graphs as ``.pt`` files plus a JSON manifest."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        graphs_dir = root / "graphs"
        graphs_dir.mkdir(exist_ok=True)

        manifest_records = []
        for index, record in enumerate(self.records):
            graph_rel: str | None = None
            if record.graph is not None:
                graph_rel = f"graphs/{index:04d}.pt"
                record.graph.to_pt(str(graphs_dir / f"{index:04d}.pt"))
                record.graph_path = graph_rel
            manifest_records.append(
                {
                    "prompt": record.prompt,
                    "label": record.label,
                    "metadata": record.metadata,
                    "graph": graph_rel,
                }
            )

        with open(root / _MANIFEST_NAME, "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "records": manifest_records}, handle, indent=2)
            handle.write("\n")

    @classmethod
    def load(cls, path: str | Path, *, load_graphs: bool = True) -> CircuitDataset:
        """Load a dataset previously written by :meth:`save`."""
        root = Path(path)
        with open(root / _MANIFEST_NAME, encoding="utf-8") as handle:
            manifest = json.load(handle)

        records: list[CircuitRecord] = []
        for item in manifest["records"]:
            graph = None
            graph_rel = item.get("graph")
            if load_graphs and graph_rel:
                graph = Graph.from_pt(str(root / graph_rel))
            records.append(
                CircuitRecord(
                    prompt=item["prompt"],
                    graph=graph,
                    label=item.get("label"),
                    metadata=dict(item.get("metadata") or {}),
                    graph_path=graph_rel,
                )
            )
        return cls(records)

    def top_features(self, n_per_graph: int = 20) -> list[list[tuple[int, int, int]]]:
        return [get_top_features(graph, n_per_graph)[0] for graph in self.graphs]

    def common_circuit(
        self, min_frequency: float = 0.5, n_per_graph: int = 20
    ) -> list[tuple[int, int, int]]:
        return find_common_circuit(
            self.graphs, min_frequency=min_frequency, n_per_graph=n_per_graph
        )


def compare_datasets(
    baseline: CircuitDataset,
    current: CircuitDataset,
    *,
    n_per_graph: int = 20,
    pair_by_prompt: bool = True,
) -> DatasetDriftResult:
    """Detect circuit drift between two datasets.

    When *pair_by_prompt* is true, only prompts present in both datasets are
    compared via :func:`~circuit_tracer.analysis.compare_graphs` on the paired
    graph lists. Otherwise feature sets are pooled independently.
    """
    baseline_graphs = baseline.graphs
    current_graphs = current.graphs

    if pair_by_prompt:
        baseline_by_prompt = {record.prompt: record.graph for record in baseline.records}
        current_by_prompt = {record.prompt: record.graph for record in current.records}
        shared_prompts = sorted(set(baseline_by_prompt) & set(current_by_prompt))
        if not shared_prompts:
            raise ValueError("No shared prompts between datasets to pair")
        paired_baseline = [baseline_by_prompt[p] for p in shared_prompts]
        paired_current = [current_by_prompt[p] for p in shared_prompts]
        assert all(g is not None for g in paired_baseline + paired_current)
        comparison = compare_graphs(
            list(paired_baseline) + list(paired_current),  # type: ignore[arg-type]
            n_per_graph=n_per_graph,
        )
        # Recompute separate pools for baseline-only / current-only reporting
        baseline_feats = {
            feat
            for graph in paired_baseline
            for feat in get_top_features(graph, n_per_graph)[0]  # type: ignore[arg-type]
        }
        current_feats = {
            feat
            for graph in paired_current
            for feat in get_top_features(graph, n_per_graph)[0]  # type: ignore[arg-type]
        }
        shared = sorted(baseline_feats & current_feats)
        return DatasetDriftResult(
            paired=True,
            n_baseline=len(paired_baseline),
            n_current=len(paired_current),
            shared_features=shared,
            baseline_only=sorted(baseline_feats - current_feats),
            current_only=sorted(current_feats - baseline_feats),
            comparison=comparison,
        )

    baseline_feats = {
        feat for graph in baseline_graphs for feat in get_top_features(graph, n_per_graph)[0]
    }
    current_feats = {
        feat for graph in current_graphs for feat in get_top_features(graph, n_per_graph)[0]
    }
    return DatasetDriftResult(
        paired=False,
        n_baseline=len(baseline_graphs),
        n_current=len(current_graphs),
        shared_features=sorted(baseline_feats & current_feats),
        baseline_only=sorted(baseline_feats - current_feats),
        current_only=sorted(current_feats - baseline_feats),
        comparison=None,
    )
