from pathlib import Path

import pytest

from aml_evidence_graph.data.configuration import (
    DEFAULT_DATA_CONFIG_PATH,
    load_data_configuration,
)


def test_default_data_configuration_locks_time_splits() -> None:
    configuration = load_data_configuration(DEFAULT_DATA_CONFIG_PATH)

    assert configuration.version == "data-v1"
    assert configuration.quality.max_null_rate_for_required_columns == 0.0


def test_data_configuration_rejects_time_split_drift(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text(
        """
version: test-v1
time_splits:
  train_start: "2022-10-08"
  train_end: "2023-04-30"
  validation_start: "2023-05-01"
  validation_end: "2023-06-30"
  test_start: "2023-07-01"
  test_end: "2023-08-23"
quality:
  max_null_rate_for_required_columns: 0.0
  require_binary_label: true
  reject_duplicate_transaction_ids: false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pre-registered time splits"):
        load_data_configuration(path)
