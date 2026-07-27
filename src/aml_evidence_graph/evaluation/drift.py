"""Lightweight offline feature-drift reports for AML model monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from aml_evidence_graph.compat import is_numeric_dtype, to_polars, to_polars_series


def population_stability_index(
    reference: pl.Series | object,
    current: pl.Series | object,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins with finite numerical safeguards."""
    if bins < 2:
        raise ValueError("bins must be at least two.")
    reference_values = to_polars_series(reference).cast(pl.Float64, strict=False)
    current_values = to_polars_series(current).cast(pl.Float64, strict=False)
    reference_non_null = reference_values.drop_nulls().to_numpy()
    if len(reference_non_null) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference_non_null, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference_values.drop_nulls().to_numpy(), bins=edges)[0]
    current_counts = np.histogram(current_values.drop_nulls().to_numpy(), bins=edges)[0]
    reference_missing = int(reference_values.null_count())
    current_missing = int(current_values.null_count())
    reference_distribution = np.append(reference_counts, reference_missing).astype(float)
    current_distribution = np.append(current_counts, current_missing).astype(float)
    reference_distribution /= max(reference_distribution.sum(), 1)
    current_distribution /= max(current_distribution.sum(), 1)
    epsilon = 1e-6
    reference_distribution = np.clip(reference_distribution, epsilon, None)
    current_distribution = np.clip(current_distribution, epsilon, None)
    return float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(current_distribution / reference_distribution)
        )
    )


def categorical_population_stability_index(
    reference: pl.Series | object,
    current: pl.Series | object,
    *,
    max_categories: int = 100,
) -> float:
    """Compute PSI for category frequencies, grouping rare/unseen values as OTHER."""
    if max_categories < 1:
        raise ValueError("max_categories must be positive.")
    reference_values = to_polars_series(reference).cast(pl.Utf8).fill_null("__MISSING__")
    current_values = to_polars_series(current).cast(pl.Utf8).fill_null("__MISSING__")
    value_column = reference_values.name
    top_categories = set(
        reference_values.value_counts()
        .sort(["count", value_column], descending=[True, False])
        .head(max_categories)[value_column]
        .to_list()
    )
    reference_grouped = [
        value if value in top_categories else "__OTHER__" for value in reference_values.to_list()
    ]
    current_grouped = [
        value if value in top_categories else "__OTHER__" for value in current_values.to_list()
    ]
    categories = sorted(set(reference_grouped).union(current_grouped))
    reference_counts = {category: 0 for category in categories}
    current_counts = {category: 0 for category in categories}
    for value in reference_grouped:
        reference_counts[value] += 1
    for value in current_grouped:
        current_counts[value] += 1
    epsilon = 1e-6
    ref = np.clip(
        np.asarray([reference_counts[category] for category in categories], dtype=float)
        / max(len(reference_grouped), 1),
        epsilon,
        None,
    )
    cur = np.clip(
        np.asarray([current_counts[category] for category in categories], dtype=float)
        / max(len(current_grouped), 1),
        epsilon,
        None,
    )
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift_report(
    reference: pl.DataFrame | object,
    current: pl.DataFrame | object,
    *,
    feature_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Return PSI and method per feature; schema changes are explicit errors."""
    reference_frame = to_polars(reference)
    current_frame = to_polars(current)
    missing = sorted(
        set(feature_columns).difference(reference_frame.columns).union(
            set(feature_columns).difference(current_frame.columns)
        )
    )
    if missing:
        raise ValueError(f"Feature drift input is missing: {', '.join(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        if is_numeric_dtype(reference_frame.schema[column]):
            result[column] = {
                "method": "numeric_psi",
                "psi": population_stability_index(
                    reference_frame[column],
                    current_frame[column],
                ),
            }
        else:
            result[column] = {
                "method": "categorical_psi",
                "psi": categorical_population_stability_index(
                    reference_frame[column],
                    current_frame[column],
                ),
            }
    return result
