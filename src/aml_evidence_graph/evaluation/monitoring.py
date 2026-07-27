"""Time, segment, uncertainty, and resource evaluation helpers for AML runs."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, TypeVar

import numpy as np
import polars as pl
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score

from aml_evidence_graph.compat import to_polars
from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores

T = TypeVar("T")


@dataclass
class ProcessResourceMonitor:
    """Sample process RSS during a bounded local operation without data inspection."""

    sample_interval_seconds: float = 0.05
    _process: psutil.Process = field(init=False, repr=False)
    _stop_event: Event = field(init=False, repr=False)
    _sampler: Thread = field(init=False, repr=False)
    _rss_start: int = field(init=False, default=0)
    _rss_peak: int = field(init=False, default=0)
    _running: bool = field(init=False, default=False)

    def __enter__(self) -> ProcessResourceMonitor:
        if self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive.")
        self._process = psutil.Process()
        self._rss_start = self._rss_bytes()
        self._rss_peak = self._rss_start
        self._stop_event = Event()
        self._running = True
        self._sampler = Thread(target=self._sample_until_stopped, daemon=True)
        self._sampler.start()
        return self

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def _sample_until_stopped(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._rss_peak = max(self._rss_peak, self._rss_bytes())

    def metrics(self) -> dict[str, float]:
        """Return current/peak resident memory after or during monitoring."""
        if not self._running:
            raise RuntimeError("ProcessResourceMonitor has not been started.")
        current = self._rss_bytes()
        self._rss_peak = max(self._rss_peak, current)
        return {
            "process_rss_start_mb": self._rss_start / 1024 / 1024,
            "process_rss_end_mb": current / 1024 / 1024,
            "process_rss_peak_mb": self._rss_peak / 1024 / 1024,
        }

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        self._sampler.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self._rss_peak = max(self._rss_peak, self._rss_bytes())


def _safe_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Return availability instead of failing a slice with one observed class."""
    if len(labels) == 0:
        return {"available": False, "reason": "empty_slice", "sample_count": 0}
    if len(np.unique(labels)) < 2:
        return {
            "available": False,
            "reason": "single_label_class",
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
        }
    return {"available": True, **evaluate_binary_risk_scores(labels, scores)}


def _aligned_scores(frame: pl.DataFrame, probabilities: Iterable[float]) -> np.ndarray:
    scores = np.asarray(list(probabilities), dtype=float)
    if len(scores) != frame.height:
        raise ValueError("Probabilities must align with the frame.")
    return scores


def monthly_stability_report(
    frame: pl.DataFrame | object,
    probabilities: Iterable[float],
) -> dict[str, dict[str, Any]]:
    """Evaluate scores month-by-month without inventing unavailable ranking metrics."""
    data = to_polars(frame)
    required = {CANONICAL.event_ts, CANONICAL.is_laundering}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Monthly report requires: {', '.join(missing)}")
    scores = _aligned_scores(data, probabilities)
    months = (
        data[CANONICAL.event_ts]
        .cast(pl.Datetime(time_zone="UTC"), strict=True)
        .dt.strftime("%Y-%m")
    )
    labels = data[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    report: dict[str, dict[str, Any]] = {}
    for month in sorted(set(months.to_list())):
        mask = np.asarray(months == month)
        report[month] = _safe_metrics(labels[mask], scores[mask])
    return report


def typology_slice_report(
    frame: pl.DataFrame | object,
    probabilities: Iterable[float],
) -> dict[str, dict[str, Any]]:
    """Compare each positive typology against all negatives in the same period."""
    data = to_polars(frame)
    required = {CANONICAL.is_laundering, CANONICAL.laundering_type}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Typology report requires: {', '.join(missing)}")
    scores = _aligned_scores(data, probabilities)
    labels = data[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    typology_values = data[CANONICAL.laundering_type].cast(pl.Utf8).to_numpy()
    positive_types = sorted(set(typology_values[labels == 1].tolist()))
    return {
        typology: _safe_metrics(
            labels[(labels == 0) | (typology_values == typology)],
            scores[(labels == 0) | (typology_values == typology)],
        )
        for typology in positive_types
    }


def new_account_slice_report(
    frame: pl.DataFrame | object,
    probabilities: Iterable[float],
    *,
    training_accounts: set[str],
) -> dict[str, dict[str, Any]]:
    """Report endpoint novelty using only account membership known in the training period."""
    data = to_polars(frame)
    required = {CANONICAL.sender_account_id, CANONICAL.receiver_account_id}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("New-account report requires: " + ", ".join(missing))
    scores = _aligned_scores(data, probabilities)
    labels = data[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    sender_ids = data[CANONICAL.sender_account_id].cast(pl.Utf8).to_numpy()
    receiver_ids = data[CANONICAL.receiver_account_id].cast(pl.Utf8).to_numpy()
    sender_new = np.asarray([value not in training_accounts for value in sender_ids])
    receiver_new = np.asarray([value not in training_accounts for value in receiver_ids])
    return {
        "sender_new": _safe_metrics(labels[sender_new], scores[sender_new]),
        "sender_seen": _safe_metrics(labels[~sender_new], scores[~sender_new]),
        "receiver_new": _safe_metrics(labels[receiver_new], scores[receiver_new]),
        "receiver_seen": _safe_metrics(labels[~receiver_new], scores[~receiver_new]),
        "either_endpoint_new": _safe_metrics(
            labels[sender_new | receiver_new],
            scores[sender_new | receiver_new],
        ),
        "both_endpoints_seen": _safe_metrics(
            labels[~sender_new & ~receiver_new],
            scores[~sender_new & ~receiver_new],
        ),
    }


def _categorical_slice_report(
    labels: np.ndarray,
    scores: np.ndarray,
    values: np.ndarray,
    *,
    max_categories: int,
) -> dict[str, dict[str, Any]]:
    if max_categories < 1:
        raise ValueError("max_categories must be positive.")
    categories = np.asarray(
        ["__MISSING__" if value is None else str(value) for value in values],
        dtype=object,
    )
    unique, counts = np.unique(categories, return_counts=True)
    order = np.lexsort((unique, -counts))
    selected = set(unique[order][:max_categories].tolist())
    grouped = np.asarray(
        [value if value in selected else "__OTHER_CATEGORIES__" for value in categories],
        dtype=object,
    )
    ordered_categories = sorted(selected)
    if "__OTHER_CATEGORIES__" in grouped:
        ordered_categories.append("__OTHER_CATEGORIES__")
    return {
        category: _safe_metrics(labels[grouped == category], scores[grouped == category])
        for category in ordered_categories
    }


def categorical_slice_report(
    frame: pl.DataFrame | object,
    probabilities: Iterable[float],
    *,
    column: str,
    max_categories: int = 50,
) -> dict[str, dict[str, Any]]:
    """Report bounded categorical slices without dropping rare values silently."""
    data = to_polars(frame)
    required = {CANONICAL.is_laundering, column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Categorical slice report requires: " + ", ".join(missing))
    scores = _aligned_scores(data, probabilities)
    return _categorical_slice_report(
        data[CANONICAL.is_laundering].cast(pl.Int64).to_numpy(),
        scores,
        data[column].to_numpy(),
        max_categories=max_categories,
    )


def paired_categorical_slice_report(
    frame: pl.DataFrame | object,
    probabilities: Iterable[float],
    *,
    left_column: str,
    right_column: str,
    max_categories: int = 50,
) -> dict[str, dict[str, Any]]:
    """Report a bounded pair slice, such as currency or sender/receiver region."""
    data = to_polars(frame)
    required = {CANONICAL.is_laundering, left_column, right_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Paired categorical slice report requires: " + ", ".join(missing))
    scores = _aligned_scores(data, probabilities)
    left = data[left_column].cast(pl.Utf8).fill_null("__MISSING__")
    right = data[right_column].cast(pl.Utf8).fill_null("__MISSING__")
    values = (left + " -> " + right).to_numpy()
    return _categorical_slice_report(
        data[CANONICAL.is_laundering].cast(pl.Int64).to_numpy(),
        scores,
        values,
        max_categories=max_categories,
    )


def bootstrap_ranking_intervals(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    iterations: int = 200,
    random_seed: int = 20260722,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Stratified bootstrap confidence intervals for PR-AUC and ROC-AUC."""
    labels = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be non-empty and aligned.")
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("Bootstrap requires both label classes.")
    if iterations < 20:
        raise ValueError("Use at least 20 bootstrap iterations.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    generator = np.random.default_rng(random_seed)
    pr_auc_values: list[float] = []
    roc_auc_values: list[float] = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [
                generator.choice(positive_indices, size=len(positive_indices), replace=True),
                generator.choice(negative_indices, size=len(negative_indices), replace=True),
            ]
        )
        pr_auc_values.append(float(average_precision_score(labels[sampled], scores[sampled])))
        roc_auc_values.append(float(roc_auc_score(labels[sampled], scores[sampled])))
    alpha = (1 - confidence) / 2

    def interval(values: list[float]) -> dict[str, float | int]:
        return {
            "iterations": iterations,
            "point_estimate": float(np.mean(values)),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1 - alpha)),
        }

    return {"pr_auc": interval(pr_auc_values), "roc_auc": interval(roc_auc_values)}


def measure_runtime(function: Callable[[], T]) -> tuple[T, dict[str, float]]:
    """Measure wall time, Python heap, and sampled process RSS for an operation."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        with ProcessResourceMonitor() as process_monitor:
            value = function()
            process_metrics = process_monitor.metrics()
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return value, {
        "wall_time_ms": (time.perf_counter() - start) * 1_000,
        "python_heap_current_mb": current / 1024 / 1024,
        "python_heap_peak_mb": peak / 1024 / 1024,
        **process_metrics,
    }
