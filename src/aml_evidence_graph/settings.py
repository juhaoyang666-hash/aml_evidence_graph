"""Runtime configuration. Secrets are read only from the process environment."""

from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with secure defaults for local development."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_parse_none_str="",
    )

    environment: str = Field(default="development", validation_alias="AML_ENV")
    data_root: Path = Field(default=Path("./data"), validation_alias="AML_DATA_ROOT")
    artifact_root: Path = Field(default=Path("./artifacts"), validation_alias="AML_ARTIFACT_ROOT")
    feature_root: Path | None = Field(
        default=None,
        validation_alias="AML_FEATURE_ROOT",
    )
    table_model_dir: Path | None = Field(
        default=None,
        validation_alias="AML_TABLE_MODEL_DIR",
    )
    graphsage_model_path: Path | None = Field(
        default=None,
        validation_alias="AML_GRAPHSAGE_MODEL_PATH",
    )
    fusion_dir: Path | None = Field(default=None, validation_alias="AML_FUSION_DIR")
    agent_checkpoint_path: Path | None = Field(
        default=None,
        validation_alias="AML_AGENT_CHECKPOINT_PATH",
    )
    agent_audit_path: Path | None = Field(
        default=None,
        validation_alias="AML_AGENT_AUDIT_PATH",
    )
    agent_coordination_path: Path | None = Field(
        default=None,
        validation_alias="AML_AGENT_COORDINATION_PATH",
    )
    evidence_store_path: Path | None = Field(
        default=None,
        validation_alias="AML_EVIDENCE_STORE_PATH",
    )
    graphsage_device: str = Field(default="auto", validation_alias="AML_GRAPHSAGE_DEVICE")
    alert_threshold: float = Field(default=0.5, validation_alias="AML_ALERT_THRESHOLD")
    model_version: str = Field(default="unconfigured", validation_alias="AML_MODEL_VERSION")
    internal_api_token: SecretStr | None = Field(
        default=None,
        validation_alias="AML_INTERNAL_API_TOKEN",
    )
    typology_root: Path = Field(
        default=Path("./knowledge/typologies"),
        validation_alias="AML_TYPOLOGY_ROOT",
    )
    llm_enabled: bool = Field(default=False, validation_alias="AML_LLM_ENABLED")
    llm_base_url: str = Field(
        default="https://chat.ecnu.edu.cn/open/api/v1",
        validation_alias="AML_LLM_BASE_URL",
    )
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="AML_LLM_API_KEY")
    ecnu_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ECNU_API_KEY",
    )
    llm_model: str = Field(default="ecnu-max", validation_alias="AML_LLM_MODEL")
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        le=300,
        validation_alias="AML_LLM_TIMEOUT_SECONDS",
    )
    llm_prompt_config_path: Path = Field(
        default=Path("./configs/prompts/ecnu-risk-evidence-v3.yaml"),
        validation_alias="AML_LLM_PROMPT_CONFIG",
    )
    llm_input_cost_per_million_tokens_usd: float | None = Field(
        default=None,
        validation_alias="AML_LLM_INPUT_COST_PER_MILLION_TOKENS_USD",
    )
    llm_output_cost_per_million_tokens_usd: float | None = Field(
        default=None,
        validation_alias="AML_LLM_OUTPUT_COST_PER_MILLION_TOKENS_USD",
    )

    @field_validator(
        "feature_root",
        "table_model_dir",
        "graphsage_model_path",
        "fusion_dir",
        "agent_checkpoint_path",
        "agent_audit_path",
        "agent_coordination_path",
        "evidence_store_path",
        "internal_api_token",
        "llm_api_key",
        "ecnu_api_key",
        "llm_input_cost_per_million_tokens_usd",
        "llm_output_cost_per_million_tokens_usd",
        mode="before",
    )
    @classmethod
    def blank_optional_values_are_none(cls, value: object) -> object:
        """Treat blank .env.example values as unset optional configuration."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("model_version", mode="before")
    @classmethod
    def blank_model_version_is_unconfigured(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "unconfigured"
        return str(value)

    @staticmethod
    def _require_strong_secret(
        secret: SecretStr | None,
        *,
        setting_name: str,
        minimum_length: int = 32,
    ) -> str:
        """Return a deliberate secret value and reject example placeholders."""
        if secret is None:
            raise RuntimeError(f"{setting_name} must be configured before this operation.")
        value = secret.get_secret_value().strip()
        placeholder_values = {
            "",
            "changeme",
            "replace-with-a-long-random-secret",
            "replace-me",
        }
        if value.lower() in placeholder_values or len(value) < minimum_length:
            raise RuntimeError(
                f"{setting_name} must be a non-placeholder secret of at least "
                f"{minimum_length} characters."
            )
        return value

    def require_llm_api_key(self) -> str:
        """Resolve ECNU first, then the generic API key only when LLM is enabled."""
        key = self.ecnu_api_key or self.llm_api_key
        if key is None:
            raise RuntimeError(
                "ECNU_API_KEY or AML_LLM_API_KEY must be configured before using the LLM."
            )
        return key.get_secret_value()

    def require_internal_api_token(self) -> str:
        """Return a strong token before exposing private evidence endpoints."""
        return self._require_strong_secret(
            self.internal_api_token,
            setting_name="AML_INTERNAL_API_TOKEN",
        )

    def require_model_version(self) -> str:
        """Require an explicit model version before serving private evidence."""
        value = self.model_version.strip()
        if not value or value == "unconfigured":
            raise RuntimeError(
                "AML_MODEL_VERSION must be configured before serving private artifacts."
            )
        return value

    def validate_agent_storage_separation(self) -> None:
        """Require separate files and a durable checkpoint for shared coordination."""
        if self.agent_coordination_path is not None and self.agent_checkpoint_path is None:
            raise RuntimeError(
                "AML_AGENT_COORDINATION_PATH requires AML_AGENT_CHECKPOINT_PATH."
            )
        configured = {
            name: path.resolve()
            for name, path in {
                "AML_AGENT_CHECKPOINT_PATH": self.agent_checkpoint_path,
                "AML_AGENT_AUDIT_PATH": self.agent_audit_path,
                "AML_AGENT_COORDINATION_PATH": self.agent_coordination_path,
                "AML_EVIDENCE_STORE_PATH": self.evidence_store_path,
            }.items()
            if path is not None
        }
        if len(set(configured.values())) != len(configured):
            raise RuntimeError(
                "Agent checkpoint, audit, coordination, and evidence SQLite paths "
                "must be different files."
            )
