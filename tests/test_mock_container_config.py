from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_mock_container_configuration_excludes_private_data_and_secrets() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    compose = yaml.safe_load(
        (PROJECT_ROOT / "docker-compose.demo.yml").read_text(encoding="utf-8")
    )

    assert "COPY configs ./configs" in dockerfile
    assert "COPY knowledge ./knowledge" in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "AML_TOKENIZATION_SECRET" not in dockerfile
    assert "data/" in ignored
    assert "artifacts/" in ignored
    assert "volumes" not in compose["services"]["aml-demo"]
    assert compose["services"]["aml-demo"]["environment"]["AML_LLM_ENABLED"] == "false"
