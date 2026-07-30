from __future__ import annotations

import pytest

from scripts.run_gat_validation_candidate import _exclude_exact_feature_columns


def test_exclude_exact_feature_columns_is_deterministic_and_deduplicated() -> None:
    retained, excluded = _exclude_exact_feature_columns(
        ("amount", "graph_endpoint_min_historical_degree", "graph_degree_imbalance_ratio"),
        ["graph_endpoint_min_historical_degree", "graph_endpoint_min_historical_degree"],
    )

    assert retained == ("amount", "graph_degree_imbalance_ratio")
    assert excluded == ("graph_endpoint_min_historical_degree",)


def test_exclude_exact_feature_columns_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="missing_feature"):
        _exclude_exact_feature_columns(("amount",), ["missing_feature"])
