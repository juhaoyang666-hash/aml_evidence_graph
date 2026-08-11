from pathlib import Path

import pytest

from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    catboost_parameters_from_configuration,
    fusion_alert_fraction_from_configuration,
    graphsage_parameters_from_configuration,
    lightgbm_parameters_from_configuration,
    load_model_configuration,
)


def test_default_model_configuration_maps_to_training_parameters() -> None:
    configuration = load_model_configuration(DEFAULT_MODEL_CONFIG_PATH)

    catboost = catboost_parameters_from_configuration(configuration)
    graphsage = graphsage_parameters_from_configuration(configuration)
    lightgbm = lightgbm_parameters_from_configuration(configuration)

    assert "random_seed" not in catboost
    assert configuration["version"] == "models-v1"
    assert catboost["iterations"] == 800
    assert lightgbm["num_leaves"] == 63
    assert graphsage["num_neighbors"] == (15, 10)
    assert graphsage["patience"] == 3
    assert graphsage["history_window_days"] == 30
    assert fusion_alert_fraction_from_configuration(configuration) == 0.005


def test_graphsage_configuration_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        """
catboost: {}
lightgbm: {}
graphsage:
  unsupported_field: true
fusion: {}
version: test-v1
""".strip(),
        encoding="utf-8",
    )
    configuration = load_model_configuration(config_path)

    with pytest.raises(ValueError, match="Unsupported GraphSAGE"):
        graphsage_parameters_from_configuration(configuration)
