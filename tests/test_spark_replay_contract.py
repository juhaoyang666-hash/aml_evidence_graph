import os
from pathlib import Path

from aml_evidence_graph.features.spark_replay import (
    REPRESENTATIVE_FEATURES,
    WINDOW_SECONDS,
    _configure_bundled_java,
)


def test_spark_replay_windows_are_fixed_and_versionable() -> None:
    assert WINDOW_SECONDS == {"1d": 86_400, "7d": 604_800, "30d": 2_592_000}
    assert len(REPRESENTATIVE_FEATURES) == 5


def test_bundled_java_can_be_discovered(monkeypatch, tmp_path: Path) -> None:
    java = tmp_path / "Library" / "bin" / "java.exe"
    java.parent.mkdir(parents=True)
    java.touch()
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(
        "aml_evidence_graph.features.spark_replay.sys.prefix", str(tmp_path)
    )
    _configure_bundled_java()

    java_home = Path(os.environ["JAVA_HOME"])
    assert (java_home / "bin" / "java.exe").is_file()
