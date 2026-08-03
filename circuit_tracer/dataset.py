"""Dataset-scale circuit analysis containers."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from circuit_tracer.analysis import (
    ComparisonResult,
    compare_graphs,
    compute_graph_scores,
    find_common_circuit,
    get_top_features,
)
from circuit_tracer.attribution.attribute import attribute_batch
from circuit_tracer.graph import Graph

__all__ = [
    "CircuitRecord",
    "CircuitDataset",
    "DatasetDriftResult",
    "CircuitClusterResult",
    "DatasetSummary",
    "compare_datasets",
    "cluster_circuits",
    "summarize_dataset",
]

_MANIFEST_NAME = "manifest.json"

FeatureKey = tuple[int, int, int]


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
    shared_features: list[FeatureKey]
    baseline_only: list[FeatureKey]
    current_only: list[FeatureKey]
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


@dataclass
class CircuitClusterResult:
    """Result of clustering circuits by top-feature overlap."""

    labels: list[int]
    distance_matrix: list[list[float]]
    cluster_members: dict[int, list[int]]
    representative_features: dict[int, list[FeatureKey]]
    method: str
    threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "circuit-tracer.circuit-clusters.v1",
            "version": 1,
            "method": self.method,
            "threshold": self.threshold,
            "labels": self.labels,
            "distanceMatrix": self.distance_matrix,
            "clusterMembers": {str(k): v for k, v in self.cluster_members.items()},
            "representativeFeatures": {
                str(k): [list(f) for f in feats]
                for k, feats in self.representative_features.items()
            },
        }


@dataclass
class DatasetSummary:
    """Aggregate / bootstrap statistics over a circuit dataset."""

    n_graphs: int
    n_per_graph: int
    feature_frequency: dict[FeatureKey, float]
    feature_frequency_ci: dict[FeatureKey, tuple[float, float]]
    mean_influence: dict[FeatureKey, float]
    mean_replacement_score: float
    mean_completeness_score: float
    per_label: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "circuit-tracer.dataset-summary.v1",
            "version": 1,
            "nGraphs": self.n_graphs,
            "nPerGraph": self.n_per_graph,
            "featureFrequency": {
                f"{a}:{b}:{c}": v for (a, b, c), v in self.feature_frequency.items()
            },
            "featureFrequencyCi": {
                f"{a}:{b}:{c}": [lo, hi]
                for (a, b, c), (lo, hi) in self.feature_frequency_ci.items()
            },
            "meanInfluence": {f"{a}:{b}:{c}": v for (a, b, c), v in self.mean_influence.items()},
            "meanReplacementScore": self.mean_replacement_score,
            "meanCompletenessScore": self.mean_completeness_score,
            "perLabel": self.per_label,
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

    def top_features(self, n_per_graph: int = 20) -> list[list[FeatureKey]]:
        return [get_top_features(graph, n_per_graph)[0] for graph in self.graphs]

    def common_circuit(self, min_frequency: float = 0.5, n_per_graph: int = 20) -> list[FeatureKey]:
        return find_common_circuit(
            self.graphs, min_frequency=min_frequency, n_per_graph=n_per_graph
        )

    def cluster(
        self,
        n_per_graph: int = 20,
        *,
        method: Literal["jaccard", "cosine"] = "jaccard",
        threshold: float = 0.5,
    ) -> CircuitClusterResult:
        return cluster_circuits(self, n_per_graph=n_per_graph, method=method, threshold=threshold)

    def summarize(
        self,
        n_per_graph: int = 20,
        *,
        bootstrap: int = 1000,
        seed: int = 0,
    ) -> DatasetSummary:
        return summarize_dataset(self, n_per_graph=n_per_graph, bootstrap=bootstrap, seed=seed)


def compare_datasets(
    baseline: CircuitDataset,
    current: CircuitDataset,
    *,
    n_per_graph: int = 20,
    pair_by_prompt: bool = True,
) -> DatasetDriftResult:
    """Detect circuit drift between two datasets."""
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


def _feature_sets_and_weights(
    dataset: CircuitDataset, n_per_graph: int
) -> tuple[list[set[FeatureKey]], list[dict[FeatureKey, float]]]:
    sets: list[set[FeatureKey]] = []
    weights: list[dict[FeatureKey, float]] = []
    for graph in dataset.graphs:
        features, scores = get_top_features(graph, n_per_graph)
        sets.append(set(features))
        weights.append({feat: float(score) for feat, score in zip(features, scores)})
    return sets, weights


def _jaccard_distance(a: set[FeatureKey], b: set[FeatureKey]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def _cosine_distance(a: dict[FeatureKey, float], b: dict[FeatureKey, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(va, vb) / denom)


def _average_linkage_distance(
    dist: np.ndarray, cluster_a: list[int], cluster_b: list[int]
) -> float:
    total = 0.0
    count = 0
    for i in cluster_a:
        for j in cluster_b:
            total += float(dist[i, j])
            count += 1
    return total / count if count else 0.0


def cluster_circuits(
    dataset: CircuitDataset,
    n_per_graph: int = 20,
    *,
    method: Literal["jaccard", "cosine"] = "jaccard",
    threshold: float = 0.5,
) -> CircuitClusterResult:
    """Cluster graphs by top-feature overlap using thresholded agglomerative linkage.

    Default distance is Jaccard on top-*n* feature sets. Clusters merge while the
    average-linkage distance stays ``<= threshold``.
    """
    if len(dataset) == 0:
        raise ValueError("Cannot cluster an empty dataset")

    feature_sets, weights = _feature_sets_and_weights(dataset, n_per_graph)
    n = len(feature_sets)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if method == "jaccard":
                d = _jaccard_distance(feature_sets[i], feature_sets[j])
            elif method == "cosine":
                d = _cosine_distance(weights[i], weights[j])
            else:
                raise ValueError(f"Unknown method: {method}")
            dist[i, j] = dist[j, i] = d

    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = set(range(n))
    next_id = n

    while True:
        best_pair: tuple[int, int] | None = None
        best_dist = float("inf")
        active_list = sorted(active)
        for ai, ca in enumerate(active_list):
            for cb in active_list[ai + 1 :]:
                d = _average_linkage_distance(dist, clusters[ca], clusters[cb])
                if d < best_dist:
                    best_dist = d
                    best_pair = (ca, cb)
        if best_pair is None or best_dist > threshold:
            break
        ca, cb = best_pair
        clusters[next_id] = clusters.pop(ca) + clusters.pop(cb)
        active.remove(ca)
        active.remove(cb)
        active.add(next_id)
        next_id += 1

    labels = [0] * n
    cluster_members: dict[int, list[int]] = {}
    representative_features: dict[int, list[FeatureKey]] = {}
    for label, cluster_id in enumerate(sorted(active)):
        members = sorted(clusters[cluster_id])
        cluster_members[label] = members
        for member in members:
            labels[member] = label
        freq: Counter[FeatureKey] = Counter()
        for member in members:
            freq.update(feature_sets[member])
        representative_features[label] = [feat for feat, _ in freq.most_common(n_per_graph)]

    return CircuitClusterResult(
        labels=labels,
        distance_matrix=dist.tolist(),
        cluster_members=cluster_members,
        representative_features=representative_features,
        method=method,
        threshold=threshold,
    )


def summarize_dataset(
    dataset: CircuitDataset,
    n_per_graph: int = 20,
    *,
    bootstrap: int = 1000,
    seed: int = 0,
) -> DatasetSummary:
    """Aggregate feature frequencies, influences, and bootstrap CIs."""
    graphs = dataset.graphs
    n = len(graphs)
    if n == 0:
        raise ValueError("Cannot summarize an empty dataset")

    per_graph_features: list[list[FeatureKey]] = []
    per_graph_scores: list[dict[FeatureKey, float]] = []
    replacement_scores: list[float] = []
    completeness_scores: list[float] = []

    for graph in graphs:
        features, scores = get_top_features(graph, n_per_graph)
        per_graph_features.append(features)
        per_graph_scores.append({feat: float(score) for feat, score in zip(features, scores)})
        replacement, completeness = compute_graph_scores(graph)
        replacement_scores.append(float(replacement))
        completeness_scores.append(float(completeness))

    frequency_counts: Counter[FeatureKey] = Counter()
    influence_sums: dict[FeatureKey, float] = defaultdict(float)
    for features, scores in zip(per_graph_features, per_graph_scores):
        frequency_counts.update(features)
        for feat, score in scores.items():
            influence_sums[feat] += score

    feature_frequency = {feat: count / n for feat, count in frequency_counts.items()}
    mean_influence = {
        feat: influence_sums[feat] / frequency_counts[feat] for feat in frequency_counts
    }

    rng = np.random.default_rng(seed)
    feature_frequency_ci: dict[FeatureKey, tuple[float, float]] = {}
    feature_list = list(frequency_counts)
    if bootstrap > 0 and feature_list:
        samples = np.zeros((bootstrap, len(feature_list)), dtype=np.float64)
        for b in range(bootstrap):
            indices = rng.integers(0, n, size=n)
            counts: Counter[FeatureKey] = Counter()
            for idx in indices:
                counts.update(per_graph_features[int(idx)])
            for j, feat in enumerate(feature_list):
                samples[b, j] = counts.get(feat, 0) / n
        for j, feat in enumerate(feature_list):
            lo, hi = np.quantile(samples[:, j], [0.025, 0.975])
            feature_frequency_ci[feat] = (float(lo), float(hi))

    per_label: dict[str, dict[str, Any]] = {}
    label_groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset.labels):
        if label is not None:
            label_groups[str(label)].append(index)
    for label, indices in label_groups.items():
        label_freq: Counter[FeatureKey] = Counter()
        for idx in indices:
            label_freq.update(per_graph_features[idx])
        per_label[label] = {
            "nGraphs": len(indices),
            "featureFrequency": {
                f"{a}:{b}:{c}": count / len(indices) for (a, b, c), count in label_freq.items()
            },
            "meanReplacementScore": _finite_mean([replacement_scores[i] for i in indices]),
            "meanCompletenessScore": _finite_mean([completeness_scores[i] for i in indices]),
        }

    return DatasetSummary(
        n_graphs=n,
        n_per_graph=n_per_graph,
        feature_frequency=feature_frequency,
        feature_frequency_ci=feature_frequency_ci,
        mean_influence=mean_influence,
        mean_replacement_score=_finite_mean(replacement_scores),
        mean_completeness_score=_finite_mean(completeness_scores),
        per_label=per_label,
    )


def _finite_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    mean = float(np.nanmean(arr))
    return 0.0 if math.isnan(mean) else mean
