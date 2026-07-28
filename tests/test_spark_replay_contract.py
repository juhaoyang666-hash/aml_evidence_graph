from aml_evidence_graph.features.spark_replay import WINDOW_SECONDS


def test_spark_replay_windows_are_fixed_and_versionable() -> None:
    assert WINDOW_SECONDS == {"1d": 86_400, "7d": 604_800, "30d": 2_592_000}
