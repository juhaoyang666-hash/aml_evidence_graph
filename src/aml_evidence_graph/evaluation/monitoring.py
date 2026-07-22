"""Time, segment, uncertainty, and resource evaluation helpers for AML runs."""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, TypeVar

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import average_precision_score, roc_auc_score

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


def _safe_metrics(labels: pd.Series, scores: pd.Series) -> dict[str, Any]:
    """Return availability instead of failing a slice with one observed class."""
    if len(labels) == 0:
        return {"available": False, "reason": "empty_slice", "sample_count": 0}
    if labels.nunique() < 2:
        return {
            "available": False,
            "reason": "single_label_class",
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
        }
    return {"available": True, **evaluate_binary_risk_scores(labels, scores)}


def monthly_stability_report(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
) -> dict[str, dict[str, Any]]:
    """Evaluate scores month-by-month without inventing unavailable ranking metrics."""
    required = {CANONICAL.event_ts, CANONICAL.is_laundering}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Monthly report requires: {', '.join(missing)}")
    scores = pd.Series(list(probabilities), index=frame.index, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Probabilities must align with the frame.")
    event_ts = pd.to_datetime(frame[CANONICAL.event_ts], utc=True, errors="raise")
    months = event_ts.dt.strftime("%Y-%m")
    return {
        month: _safe_metrics(
            frame.loc[months.eq(month), CANONICAL.is_laundering].astype(int),
            scores.loc[months.eq(month)],
        )
        for month in sorted(months.unique())
    }


def typology_slice_report(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
) -> dict[str, dict[str, Any]]:
    """Compare each positive typology against all negatives in the same period."""
    required = {CANONICAL.is_laundering, CANONICAL.laundering_type}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Typology report requires: {', '.join(missing)}")
    scores = pd.Series(list(probabilities), index=frame.index, dtype=float)
    labels = frame[CANONICAL.is_laundering].astype(int)
    positive_types = sorted(
        frame.loc[labels.eq(1), CANONICAL.laundering_type].astype(str).unique()
    )
    return {
        typology: _safe_metrics(
            labels.loc[labels.eq(0) | frame[CANONICAL.laundering_type].astype(str).eq(typology)],
            scores.loc[labels.eq(0) | frame[CANONICAL.laundering_type].astype(str).eq(typology)],
        )
        for typology in positive_types
    }


def new_account_slice_report(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    *,
    training_accounts: set[str],
) -> dict[str, dict[str, Any]]:
    """Report endpoint novelty using only account membership known in the training period."""
    required = {CANONICAL.sender_account_id, CANONICAL.receiver_account_id}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("New-account report requires: " + ", ".join(missing))
    scores = pd.Series(list(probabilities), index=frame.index, dtype=float)
    labels = frame[CANONICAL.is_laundering].astype(int)
    sender_new = ~frame[CANONICAL.sender_account_id].astype(str).isin(training_accounts)
    receiver_new = ~frame[CANONICAL.receiver_account_id].astype(str).isin(training_accounts)
    return {
        "sender_new": _safe_metrics(labels.loc[sender_new], scores.loc[sender_new]),
        "sender_seen": _safe_metrics(labels.loc[~sender_new], scores.loc[~sender_new]),
        "receiver_new": _safe_metrics(labels.loc[receiver_new], scores.loc[receiver_new]),
        "receiver_seen": _safe_metrics(labels.loc[~receiver_new], scores.loc[~receiver_new]),
        "either_endpoint_new": _safe_metrics(
            labels.loc[sender_new | receiver_new],
            scores.loc[sender_new | receiver_new],
        ),
        "both_endpoints_seen": _safe_metrics(
            labels.loc[~sender_new & ~receiver_new],
            scores.loc[~sender_new & ~receiver_new],
        ),
    }


def _categorical_slice_report(
    labels: pd.Series,
    scores: pd.Series,
    values: pd.Series,
    *,
    max_categories: int,
) -> dict[str, dict[str, Any]]:
    if max_categories < 1:
        raise ValueError("max_categories must be positive.")
    categories = values.astype("string").fillna("__MISSING__").astype(str)
    counts = (
        categories.value_counts()
        .rename_axis("category")
        .reset_index(name="row_count")
        .sort_values(["row_count", "category"], ascending=[False, True], kind="stable")
    )
    selected_categories = set(counts.head(max_categories)["category"])
    grouped_categories = categories.where(
        categories.isin(selected_categories),
        "__OTHER_CATEGORIES__",
    )
    ordered_categories = sorted(selected_categories)
    if grouped_categories.eq("__OTHER_CATEGORIES__").any():
        ordered_categories.append("__OTHER_CATEGORIES__")
    return {
        category: _safe_metrics(
            labels.loc[grouped_categories.eq(category)],
            scores.loc[grouped_categories.eq(category)],
        )
        for category in ordered_categories
    }


def categorical_slice_report(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    *,
    column: str,
    max_categories: int = 50,
) -> dict[str, dict[str, Any]]:
    """Report bounded categorical slices without dropping rare values silently."""
    required = {CANONICAL.is_laundering, column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Categorical slice report requires: " + ", ".join(missing))
    scores = pd.Series(list(probabilities), index=frame.index, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Probabilities must align with the frame.")
    return _categorical_slice_report(
        frame[CANONICAL.is_laundering].astype(int),
        scores,
        frame[column],
        max_categories=max_categories,
    )


def paired_categorical_slice_report(
    frame: pd.DataFrame,
    probabilities: Iterable[float],
    *,
    left_column: str,
    right_column: str,
    max_categories: int = 50,
) -> dict[str, dict[str, Any]]:
    """Report a bounded pair slice, such as currency or sender/receiver region."""
    required = {CANONICAL.is_laundering, left_column, right_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Paired categorical slice report requires: " + ", ".join(missing))
    scores = pd.Series(list(probabilities), index=frame.index, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("Probabilities must align with the frame.")
    values = (
        frame[left_column].astype("string").fillna("__MISSING__")
        + " -> "
        + frame[right_column].astype("string").fillna("__MISSING__")
    )
    return _categorical_slice_report(
        frame[CANONICAL.is_laundering].astype(int),
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
