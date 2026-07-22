"""Lightweight offline feature-drift reports for AML model monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Compute PSI using reference quantile bins with finite numerical safeguards."""
    if bins < 2:
        raise ValueError("bins must be at least two.")
    reference_values = pd.to_numeric(reference, errors="coerce")
    current_values = pd.to_numeric(current, errors="coerce")
    reference_non_null = reference_values.dropna()
    if reference_non_null.empty:
        return 0.0
    edges = np.unique(np.quantile(reference_non_null, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts = np.histogram(reference_values.dropna(), bins=edges)[0]
    current_counts = np.histogram(current_values.dropna(), bins=edges)[0]
    reference_missing = int(reference_values.isna().sum())
    current_missing = int(current_values.isna().sum())
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
    reference: pd.Series,
    current: pd.Series,
    *,
    max_categories: int = 100,
) -> float:
    """Compute PSI for category frequencies, grouping rare/unseen values as OTHER."""
    if max_categories < 1:
        raise ValueError("max_categories must be positive.")
    reference_values = reference.astype("string").fillna("__MISSING__")
    current_values = current.astype("string").fillna("__MISSING__")
    top_categories = reference_values.value_counts().head(max_categories).index
    reference_grouped = reference_values.where(
        reference_values.isin(top_categories),
        "__OTHER__",
    )
    current_grouped = current_values.where(
        current_values.isin(top_categories),
        "__OTHER__",
    )
    categories = sorted(set(reference_grouped.unique()).union(current_grouped.unique()))
    reference_distribution = reference_grouped.value_counts().reindex(
        categories,
        fill_value=0,
    )
    current_distribution = current_grouped.value_counts().reindex(categories, fill_value=0)
    epsilon = 1e-6
    ref = np.clip(
        reference_distribution.to_numpy(dtype=float) / len(reference_grouped),
        epsilon,
        None,
    )
    cur = np.clip(
        current_distribution.to_numpy(dtype=float) / len(current_grouped),
        epsilon,
        None,
    )
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Return PSI and method per feature; schema changes are explicit errors."""
    missing = sorted(
        set(feature_columns).difference(reference.columns).union(
            set(feature_columns).difference(current.columns)
        )
    )
    if missing:
        raise ValueError(f"Feature drift input is missing: {', '.join(missing)}")
    result: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            result[column] = {
                "method": "numeric_psi",
                "psi": population_stability_index(reference[column], current[column]),
            }
        else:
            result[column] = {
                "method": "categorical_psi",
                "psi": categorical_population_stability_index(
                    reference[column],
                    current[column],
                ),
            }
    return result
