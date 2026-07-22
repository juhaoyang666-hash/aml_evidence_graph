import pandas as pd

from aml_evidence_graph.evaluation.drift import feature_drift_report


def test_feature_drift_reports_numeric_and_categorical_psi() -> None:
    reference = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0],
            "payment_type": ["A", "A", "B", "B"],
        }
    )
    current = pd.DataFrame(
        {
            "amount": [10.0, 11.0, 12.0, 13.0],
            "payment_type": ["B", "B", "B", "C"],
        }
    )

    report = feature_drift_report(
        reference,
        current,
        feature_columns=["amount", "payment_type"],
    )

    assert report["amount"]["method"] == "numeric_psi"
    assert report["amount"]["psi"] > 0
    assert report["payment_type"]["method"] == "categorical_psi"
