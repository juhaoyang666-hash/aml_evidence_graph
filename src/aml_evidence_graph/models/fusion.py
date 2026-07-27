"""OOF-only probability fusion, validation calibration, and alert thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from aml_evidence_graph.compat import to_polars


@dataclass
class OOFFusionModel:
    """A logistic fusioner fitted exclusively from chronological training OOF scores."""

    model_names: tuple[str, ...]
    classifier: LogisticRegression

    def predict_proba(self, scores: pl.DataFrame | pd.DataFrame) -> np.ndarray:
        frame = to_polars(scores)
        _validate_score_frame(frame, self.model_names)
        return self.classifier.predict_proba(frame.select(list(self.model_names)).to_numpy())[:, 1]


@dataclass
class ValidationCalibration:
    """A monotonic calibrator and a validation-selected operational threshold."""

    method: Literal["platt", "isotonic"]
    calibrator: IsotonicRegression | LogisticRegression
    alert_fraction: float
    threshold: float

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_scores, dtype=float)
        if self.method == "platt":
            assert isinstance(self.calibrator, LogisticRegression)
            return self.calibrator.predict_proba(values.reshape(-1, 1))[:, 1]
        assert isinstance(self.calibrator, IsotonicRegression)
        return self.calibrator.predict(values)


def _validate_score_frame(scores: pl.DataFrame, model_names: tuple[str, ...]) -> None:
    missing = sorted(set(model_names).difference(scores.columns))
    if missing:
        raise ValueError(f"Fusion scores are missing model columns: {', '.join(missing)}")
    values = scores.select(list(model_names)).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("All fusion input scores must be finite probabilities in [0, 1].")


def fit_oof_fusion(
    oof_scores: pl.DataFrame | pd.DataFrame,
    labels: pl.Series | pd.Series | np.ndarray,
    *,
    model_names: tuple[str, ...] | None = None,
    random_seed: int = 20260722,
) -> OOFFusionModel:
    """Fit the fusioner on train-period OOF predictions, never validation/test scores."""
    frame = to_polars(oof_scores)
    names = model_names or tuple(frame.columns)
    if not names:
        raise ValueError("At least one component score is required for fusion.")
    _validate_score_frame(frame, names)
    y_true = np.asarray(
        labels.to_list() if isinstance(labels, (pl.Series, pd.Series)) else labels,
        dtype=int,
    )
    if len(y_true) != frame.height or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Fusion labels must align with scores and contain both binary classes.")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=random_seed,
    )
    classifier.fit(frame.select(list(names)).to_numpy(), y_true)
    return OOFFusionModel(model_names=names, classifier=classifier)


def fit_validation_calibration_and_threshold(
    validation_raw_scores: np.ndarray,
    validation_labels: pl.Series | pd.Series | np.ndarray,
    *,
    alert_fraction: float = 0.005,
    method: Literal["auto", "platt", "isotonic"] = "auto",
) -> ValidationCalibration:
    """Fit calibration and choose the alert cutoff only on the validation period."""
    scores = np.asarray(validation_raw_scores, dtype=float)
    labels = np.asarray(
        validation_labels.to_list()
        if isinstance(validation_labels, (pl.Series, pd.Series))
        else validation_labels,
        dtype=int,
    )
    if len(scores) != len(labels) or len(scores) == 0:
        raise ValueError("Validation scores and labels must be non-empty and aligned.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Validation calibration requires both label classes.")
    if not 0 < alert_fraction <= 1:
        raise ValueError("alert_fraction must be in (0, 1].")
    if method not in {"auto", "platt", "isotonic"}:
        raise ValueError("method must be auto, platt, or isotonic.")
    candidates: list[
        tuple[
            Literal["platt", "isotonic"],
            IsotonicRegression | LogisticRegression,
            np.ndarray,
        ]
    ] = []
    if method in {"auto", "platt"}:
        platt = LogisticRegression(max_iter=1_000)
        platt.fit(scores.reshape(-1, 1), labels)
        candidates.append(
            (
                "platt",
                platt,
                platt.predict_proba(scores.reshape(-1, 1))[:, 1],
            )
        )
    if method in {"auto", "isotonic"}:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(scores, labels)
        candidates.append(("isotonic", isotonic, isotonic.predict(scores)))
    chosen_method, calibrator, calibrated = min(
        candidates,
        key=lambda candidate: float(
            np.mean((labels.astype(float) - candidate[2]) ** 2)
        ),
    )
    alert_count = max(1, int(np.ceil(len(calibrated) * alert_fraction)))
    threshold = float(np.sort(calibrated)[-alert_count])
    return ValidationCalibration(
        method=chosen_method,
        calibrator=calibrator,
        alert_fraction=alert_fraction,
        threshold=threshold,
    )
