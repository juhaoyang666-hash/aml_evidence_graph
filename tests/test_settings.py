from pathlib import Path

import pytest

from aml_evidence_graph.settings import Settings


def test_internal_api_token_requirement_rejects_short_values() -> None:
    short = Settings(AML_INTERNAL_API_TOKEN="short-token")

    with pytest.raises(RuntimeError, match="non-placeholder"):
        short.require_internal_api_token()


def test_internal_api_token_requirement_returns_strong_value() -> None:
    api_token = "a" * 32
    settings = Settings(AML_INTERNAL_API_TOKEN=api_token)

    assert settings.require_internal_api_token() == api_token


def test_blank_optional_environment_values_do_not_enable_private_mode() -> None:
    settings = Settings(
        AML_FEATURE_ROOT="",
        AML_TABLE_MODEL_DIR="",
        AML_GRAPHSAGE_MODEL_PATH="",
        AML_FUSION_DIR="",
        AML_AGENT_CHECKPOINT_PATH="",
        AML_AGENT_AUDIT_PATH="",
        AML_INTERNAL_API_TOKEN="",
        ECNU_API_KEY="",
    )

    assert settings.feature_root is None
    assert settings.table_model_dir is None
    assert settings.graphsage_model_path is None
    assert settings.fusion_dir is None
    assert settings.agent_checkpoint_path is None
    assert settings.agent_audit_path is None
    assert settings.internal_api_token is None
    assert settings.ecnu_api_key is None


def test_private_model_version_must_be_explicit() -> None:
    with pytest.raises(RuntimeError, match="AML_MODEL_VERSION"):
        Settings().require_model_version()

    assert Settings(AML_MODEL_VERSION="table-run-20260722").require_model_version() == (
        "table-run-20260722"
    )


def test_env_example_loads_with_blank_optional_values() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = Settings(_env_file=root / ".env.example")

    assert settings.feature_root is None
    assert settings.internal_api_token is None
    assert settings.llm_input_cost_per_million_tokens_usd is None
    assert settings.agent_checkpoint_path is None
    assert settings.agent_audit_path is None


def test_agent_checkpoint_and_audit_paths_must_be_distinct(tmp_path: Path) -> None:
    shared = tmp_path / "shared.sqlite"
    settings = Settings(
        AML_AGENT_CHECKPOINT_PATH=str(shared),
        AML_AGENT_AUDIT_PATH=str(shared),
    )

    with pytest.raises(RuntimeError, match="must be different files"):
        settings.validate_agent_storage_separation()

    Settings(
        AML_AGENT_CHECKPOINT_PATH=str(tmp_path / "checkpoint.sqlite"),
        AML_AGENT_AUDIT_PATH=str(tmp_path / "audit.sqlite"),
    ).validate_agent_storage_separation()
