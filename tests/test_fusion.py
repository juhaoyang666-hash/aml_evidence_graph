import numpy as np
import polars as pl

from aml_evidence_graph.models.fusion import (
    fit_oof_fusion,
    fit_validation_calibration_and_threshold,
)


def test_oof_fusion_and_validation_calibration_keep_periods_separate() -> None:
    oof_scores = pl.DataFrame(
        {
            "catboost": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "graphsage": [0.2, 0.8, 0.1, 0.9, 0.4, 0.6],
        }
    )
    labels = np.array([0, 1, 0, 1, 0, 1])
    fusion = fit_oof_fusion(oof_scores, labels)
    validation_raw = fusion.predict_proba(oof_scores)
    calibration = fit_validation_calibration_and_threshold(
        validation_raw,
        labels,
        alert_fraction=0.5,
    )

    calibrated = calibration.predict_proba(validation_raw)

    assert calibrated.shape == (6,)
    assert 0 <= calibration.threshold <= 1
    assert calibration.method in {"platt", "isotonic"}
