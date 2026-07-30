from pathlib import Path

import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.monitoring import (
    bootstrap_ranking_intervals,
    categorical_slice_report,
    measure_runtime,
    measure_runtime_rss_only,
    monthly_stability_report,
    new_account_slice_report,
    paired_bootstrap_ranking_differences,
    paired_categorical_slice_report,
    typology_slice_report,
)
from aml_evidence_graph.tracking.run import create_run_manifest


def _evaluation_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            CANONICAL.event_ts: pl.Series(
                [
                    "2023-07-01T10:00:00Z",
                    "2023-07-02T10:00:00Z",
                    "2023-08-01T10:00:00Z",
                    "2023-08-02T10:00:00Z",
                ]
            ).str.to_datetime(time_zone="UTC"),
            CANONICAL.is_laundering: [0, 1, 0, 1],
            CANONICAL.laundering_type: ["None", "Structuring", "None", "Layering"],
            CANONICAL.sender_account_id: ["seen", "new", "new", "seen"],
            CANONICAL.receiver_account_id: ["seen", "seen", "new", "new"],
            CANONICAL.payment_type: ["transfer", "transfer", "cash", "cash"],
            CANONICAL.sender_location: ["A", "A", "B", "B"],
            CANONICAL.receiver_location: ["A", "B", "B", "A"],
            CANONICAL.payment_currency: ["USD", "USD", "EUR", "EUR"],
            CANONICAL.received_currency: ["USD", "EUR", "EUR", "USD"],
        }
    )


def test_stability_and_slice_reports_do_not_drop_single_class_segments() -> None:
    frame = _evaluation_frame()
    scores = [0.1, 0.9, 0.2, 0.8]

    monthly = monthly_stability_report(frame, scores)
    typology = typology_slice_report(frame, scores)
    new_accounts = new_account_slice_report(frame, scores, training_accounts={"seen"})

    assert monthly["2023-07"]["available"]
    assert typology["Structuring"]["available"]
    assert new_accounts["sender_new"]["available"]
    assert new_accounts["sender_seen"]["available"]
    assert new_accounts["receiver_new"]["available"]
    assert new_accounts["either_endpoint_new"]["available"]
    single_class = monthly_stability_report(frame.head(1), scores[:1])
    assert single_class["2023-07"]["available"] is False


def test_categorical_reports_keep_a_bounded_other_bucket() -> None:
    frame = _evaluation_frame()
    scores = [0.1, 0.9, 0.2, 0.8]

    payment = categorical_slice_report(
        frame,
        scores,
        column=CANONICAL.payment_type,
        max_categories=1,
    )
    locations = paired_categorical_slice_report(
        frame,
        scores,
        left_column=CANONICAL.sender_location,
        right_column=CANONICAL.receiver_location,
        max_categories=2,
    )

    assert "cash" in payment
    assert "__OTHER_CATEGORIES__" in payment
    assert len(locations) == 3


def test_bootstrap_and_manifest_are_aggregate_only(tmp_path: Path) -> None:
    intervals = bootstrap_ranking_intervals(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.2, 0.8],
        iterations=20,
    )
    source = tmp_path / "source.txt"
    source.write_text("safe aggregate fixture", encoding="utf-8")
    output = tmp_path / "run"
    output.mkdir()
    manifest = create_run_manifest(
        output_dir=output,
        command="test",
        random_seed=1,
        input_paths={"fixture": source},
    )

    rendered = (output / "run_manifest.json").read_text(encoding="utf-8")

    assert intervals["pr_auc"]["lower"] <= intervals["pr_auc"]["upper"]
    assert manifest.run_id in rendered
    assert manifest.run_purpose == "full"
    assert '"run_purpose": "full"' in rendered
    assert "safe aggregate fixture" not in rendered


def test_paired_bootstrap_reports_candidate_minus_baseline() -> None:
    report = paired_bootstrap_ranking_differences(
        [0, 0, 0, 1, 1, 1],
        [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        [0.1, 0.5, 0.4, 0.3, 0.8, 0.7],
        iterations=20,
    )

    assert report["pr_auc_difference"]["point_estimate"] > 0
    assert report["pr_auc_difference"]["iterations"] == 20


def test_manifest_infers_smoke_from_scoped_paths(tmp_path: Path) -> None:
    source = tmp_path / "smoke-input.txt"
    source.write_text("fixture", encoding="utf-8")
    output = tmp_path / "smoke-run"
    output.mkdir()

    manifest = create_run_manifest(
        output_dir=output,
        command="test-smoke",
        random_seed=1,
        input_paths={"fixture": source},
    )

    assert manifest.run_purpose == "smoke"


def test_manifest_infers_smoke_from_command(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("fixture", encoding="utf-8")
    output = tmp_path / "generic-run"
    output.mkdir()

    manifest = create_run_manifest(
        output_dir=output,
        command="aml-train-table --smoke-limit 1000",
        random_seed=1,
        input_paths={"fixture": source},
    )

    assert manifest.run_purpose == "smoke"


def test_runtime_measurement_records_process_memory_without_payloads() -> None:
    value, metrics = measure_runtime(lambda: sum(range(10_000)))

    assert value == 49_995_000
    assert metrics["wall_time_ms"] >= 0
    assert metrics["process_rss_peak_mb"] >= metrics["process_rss_start_mb"]


def test_rss_only_runtime_measurement_omits_python_heap_metrics() -> None:
    value, metrics = measure_runtime_rss_only(lambda: sum(range(10_000)))

    assert value == 49_995_000
    assert metrics["wall_time_ms"] >= 0
    assert metrics["process_rss_peak_mb"] >= metrics["process_rss_start_mb"]
    assert "python_heap_peak_mb" not in metrics
