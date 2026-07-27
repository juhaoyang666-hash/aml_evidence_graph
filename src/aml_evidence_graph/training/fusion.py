"""Persist an OOF-fitted fusioner and validation-only calibration policy."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from aml_evidence_graph.data.contract import CANONICAL
from aml_evidence_graph.evaluation.metrics import evaluate_binary_risk_scores
from aml_evidence_graph.evaluation.monitoring import (
    bootstrap_ranking_intervals,
    categorical_slice_report,
    monthly_stability_report,
    new_account_slice_report,
    paired_categorical_slice_report,
    typology_slice_report,
)
from aml_evidence_graph.models.fusion import (
    OOFFusionModel,
    ValidationCalibration,
    fit_oof_fusion,
    fit_validation_calibration_and_threshold,
)
from aml_evidence_graph.tracking.run import create_run_manifest
from aml_evidence_graph.training.configuration import (
    DEFAULT_MODEL_CONFIG_PATH,
    fusion_alert_fraction_from_configuration,
    load_model_configuration,
)


@dataclass(frozen=True)
class FusionRunSummary:
    """An audit record for a fusioner that never receives test labels."""

    created_at_utc: str
    run_id: str
    component_models: list[str]
    oof_row_count: int
    validation_row_count: int
    alert_fraction: float
    calibrated_threshold: float
    calibration_method: str
    validation_metrics: dict[str, Any]


@dataclass(frozen=True)
class FusionTestEvaluationSummary:
    """One immutable chronological test evaluation of frozen fusion artifacts."""

    created_at_utc: str
    run_id: str
    component_models: list[str]
    test_row_count: int
    calibration_method: str
    calibrated_threshold: float
    test_metrics: dict[str, Any]
    test_monthly_stability: dict[str, dict[str, Any]] | None
    test_typology_slices: dict[str, dict[str, Any]] | None
    test_new_account_slices: dict[str, dict[str, Any]] | None
    test_payment_type_slices: dict[str, dict[str, Any]] | None
    test_location_pair_slices: dict[str, dict[str, Any]] | None
    test_currency_pair_slices: dict[str, dict[str, Any]] | None
    test_bootstrap_intervals: dict[str, dict[str, Any]] | None
    runtime: dict[str, float]


def merge_component_scores(
    frames: list[pl.DataFrame],
    *,
    component_columns: list[str],
) -> pl.DataFrame:
    """Merge score artifacts only by explicit transaction ID equality."""
    if not frames:
        raise ValueError("At least one score frame is required.")
    required = {CANONICAL.transaction_id, CANONICAL.is_laundering}
    for frame in frames:
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Score frame missing required columns: {', '.join(missing)}")
        if frame[CANONICAL.transaction_id].is_duplicated().any():
            raise ValueError("Each score artifact must have unique transaction IDs.")
    key_columns = [CANONICAL.transaction_id, CANONICAL.is_laundering]
    base_keys = set(frames[0].select(key_columns).iter_rows())
    if len(base_keys) != frames[0].height:
        raise ValueError("Each score artifact must have unique transaction and label keys.")
    merged = frames[0]
    for frame in frames[1:]:
        frame_keys = set(frame.select(key_columns).iter_rows())
        if len(frame_keys) != frame.height or frame_keys != base_keys:
            raise ValueError(
                "All component score artifacts must cover exactly the same transaction keys."
            )
        available_components = [
            column for column in component_columns if column in frame.columns
        ]
        merged = merged.join(
            frame.select(
                [
                    CANONICAL.transaction_id,
                    CANONICAL.is_laundering,
                    *available_components,
                ]
            ),
            on=key_columns,
            how="inner",
            validate="1:1",
        )
    missing_components = sorted(set(component_columns).difference(merged.columns))
    if missing_components:
        raise ValueError(
            f"Merged score artifacts are missing components: {', '.join(missing_components)}"
        )
    return merged


def load_persisted_fusion_artifacts(
    fusion_dir: Path,
) -> tuple[OOFFusionModel, ValidationCalibration]:
    """Load only trusted local fusion and calibration artifacts."""
    fusion_path = fusion_dir / "oof_fusion.joblib"
    calibration_path = fusion_dir / "validation_calibration.joblib"
    if not fusion_path.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Fusion directory is missing required persisted artifacts.")
    fusion = joblib.load(fusion_path)
    calibration = joblib.load(calibration_path)
    if not isinstance(fusion, OOFFusionModel):
        raise TypeError("Trusted fusion artifact has an unexpected type.")
    if not isinstance(calibration, ValidationCalibration):
        raise TypeError("Trusted calibration artifact has an unexpected type.")
    return fusion, calibration


def _align_test_context(
    test_scores: pl.DataFrame,
    context: pl.DataFrame,
) -> pl.DataFrame:
    """Join score rows to private feature context through explicit one-to-one keys."""
    key_columns = [CANONICAL.transaction_id, CANONICAL.is_laundering]
    missing = sorted(set(key_columns).difference(context.columns))
    if missing:
        raise ValueError("Test context is missing: " + ", ".join(missing))
    aligned = context.join(
        test_scores,
        on=key_columns,
        how="inner",
        validate="1:1",
        suffix="_score",
    )
    if aligned.height != context.height or aligned.height != test_scores.height:
        raise ValueError("Test context and component scores must cover the same rows.")
    return aligned


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Use overwrite only for private, regenerable fusion artifacts."
        )
    if not output_dir.is_dir():
        raise ValueError(f"Fusion output path must be a directory: {output_dir}")
    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)


def fit_and_persist_fusion(
    oof_scores: pl.DataFrame,
    validation_scores: pl.DataFrame,
    output_dir: Path,
    *,
    component_models: tuple[str, ...],
    random_seed: int = 20260722,
    alert_fraction: float = 0.005,
    input_paths: dict[str, Path] | None = None,
    model_config_path: Path | None = None,
    overwrite: bool = False,
) -> FusionRunSummary:
    """Fit OOF fusion then validation calibration; intentionally accepts no test frame."""
    _prepare_output_dir(output_dir, overwrite=overwrite)
    required = {CANONICAL.transaction_id, CANONICAL.is_laundering, *component_models}
    for name, frame in (("oof", oof_scores), ("validation", validation_scores)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} scores missing: {', '.join(missing)}")
        if frame[CANONICAL.transaction_id].is_duplicated().any():
            raise ValueError(f"{name} scores must have unique transaction IDs.")
    fusion = fit_oof_fusion(
        oof_scores.select(component_models),
        oof_scores[CANONICAL.is_laundering],
        model_names=component_models,
        random_seed=random_seed,
    )
    validation_raw = fusion.predict_proba(validation_scores.select(component_models))
    calibration = fit_validation_calibration_and_threshold(
        validation_raw,
        validation_scores[CANONICAL.is_laundering],
        alert_fraction=alert_fraction,
    )
    validation_calibrated = calibration.predict_proba(validation_raw)
    validation_metrics = evaluate_binary_risk_scores(
        validation_scores[CANONICAL.is_laundering].cast(pl.Int64).to_numpy(),
        validation_calibrated,
    )
    joblib.dump(fusion, output_dir / "oof_fusion.joblib")
    joblib.dump(calibration, output_dir / "validation_calibration.joblib")
    (output_dir / "threshold_policy.json").write_text(
        json.dumps(
            {
                "alert_fraction": calibration.alert_fraction,
                "calibrated_threshold": calibration.threshold,
                "calibration_method": calibration.method,
                "selection_split": "validation",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="aml-fit-fusion",
        random_seed=random_seed,
        input_paths=input_paths or {},
        config_paths={"model_config": model_config_path or DEFAULT_MODEL_CONFIG_PATH},
        metadata={
            "component_models": list(component_models),
            "oof_row_count": oof_scores.height,
            "validation_row_count": validation_scores.height,
        },
    )
    summary = FusionRunSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        run_id=manifest.run_id,
        component_models=list(component_models),
        oof_row_count=oof_scores.height,
        validation_row_count=validation_scores.height,
        alert_fraction=alert_fraction,
        calibrated_threshold=calibration.threshold,
        calibration_method=calibration.method,
        validation_metrics=validation_metrics,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def evaluate_persisted_fusion(
    fusion_dir: Path,
    test_scores: pl.DataFrame,
    output_dir: Path,
    *,
    test_context: pl.DataFrame | None = None,
    training_accounts: set[str] | None = None,
    bootstrap_iterations: int = 0,
    input_paths: dict[str, Path] | None = None,
    overwrite: bool = False,
) -> FusionTestEvaluationSummary:
    """Evaluate frozen fusion artifacts once; test data never enters any fit operation."""
    if bootstrap_iterations and bootstrap_iterations < 20:
        raise ValueError("bootstrap_iterations must be zero or at least 20.")
    _prepare_output_dir(output_dir, overwrite=overwrite)
    started_at = time.perf_counter()
    fusion, calibration = load_persisted_fusion_artifacts(fusion_dir)
    required = {CANONICAL.transaction_id, CANONICAL.is_laundering, *fusion.model_names}
    missing = sorted(required.difference(test_scores.columns))
    if missing:
        raise ValueError("Test component scores are missing: " + ", ".join(missing))
    if test_scores[CANONICAL.transaction_id].is_duplicated().any():
        raise ValueError("Test component scores must have unique transaction IDs.")

    raw_scores = fusion.predict_proba(test_scores.select(fusion.model_names))
    calibrated_scores = calibration.predict_proba(raw_scores)
    labels = test_scores[CANONICAL.is_laundering].cast(pl.Int64).to_numpy()
    test_metrics = evaluate_binary_risk_scores(labels, calibrated_scores)
    score_columns = [
        column
        for column in (
            CANONICAL.transaction_id,
            CANONICAL.event_ts,
            CANONICAL.is_laundering,
            *fusion.model_names,
        )
        if column in test_scores.columns
    ]
    scored_rows = test_scores.select(score_columns).with_columns(
        pl.Series("fusion_raw_probability", raw_scores),
        pl.Series("fusion_calibrated_probability", calibrated_scores),
        pl.Series(
            "is_alert_at_validation_threshold",
            (calibrated_scores >= calibration.threshold).astype(np.int8),
        ),
    )
    scored_rows.write_parquet(output_dir / "test_fusion_scores.parquet")

    monthly: dict[str, dict[str, Any]] | None = None
    typology: dict[str, dict[str, Any]] | None = None
    new_accounts: dict[str, dict[str, Any]] | None = None
    payment_type: dict[str, dict[str, Any]] | None = None
    location_pair: dict[str, dict[str, Any]] | None = None
    currency_pair: dict[str, dict[str, Any]] | None = None
    if test_context is not None:
        aligned_context = _align_test_context(test_scores, test_context)
        monthly = monthly_stability_report(aligned_context, calibrated_scores)
        typology = typology_slice_report(aligned_context, calibrated_scores)
        payment_type = categorical_slice_report(
            aligned_context,
            calibrated_scores,
            column=CANONICAL.payment_type,
        )
        location_pair = paired_categorical_slice_report(
            aligned_context,
            calibrated_scores,
            left_column=CANONICAL.sender_location,
            right_column=CANONICAL.receiver_location,
        )
        currency_pair = paired_categorical_slice_report(
            aligned_context,
            calibrated_scores,
            left_column=CANONICAL.payment_currency,
            right_column=CANONICAL.received_currency,
        )
        if training_accounts is not None:
            new_accounts = new_account_slice_report(
                aligned_context,
                calibrated_scores,
                training_accounts=training_accounts,
            )
    bootstrap = (
        bootstrap_ranking_intervals(labels, calibrated_scores, iterations=bootstrap_iterations)
        if bootstrap_iterations
        else None
    )
    manifest = create_run_manifest(
        output_dir=output_dir,
        command="aml-evaluate-fusion",
        random_seed=20260722,
        input_paths={"fusion_artifacts": fusion_dir, **(input_paths or {})},
        metadata={
            "component_models": list(fusion.model_names),
            "test_row_count": test_scores.height,
            "selection_split": "validation",
            "bootstrap_iterations": bootstrap_iterations,
        },
    )
    summary = FusionTestEvaluationSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        run_id=manifest.run_id,
        component_models=list(fusion.model_names),
        test_row_count=test_scores.height,
        calibration_method=calibration.method,
        calibrated_threshold=calibration.threshold,
        test_metrics=test_metrics,
        test_monthly_stability=monthly,
        test_typology_slices=typology,
        test_new_account_slices=new_accounts,
        test_payment_type_slices=payment_type,
        test_location_pair_slices=location_pair,
        test_currency_pair_slices=currency_pair,
        test_bootstrap_intervals=bootstrap,
        runtime={"wall_time_seconds": time.perf_counter() - started_at},
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def _read_scores(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pl.read_csv(path)
    raise ValueError("Fusion score inputs must be Parquet or CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True, action="append")
    parser.add_argument("--validation", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", required=True, help="Comma-separated score columns.")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--alert-fraction", type=float)
    parser.add_argument("--random-seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_configuration = load_model_configuration(args.model_config)
    component_models = tuple(
        component.strip() for component in args.components.split(",") if component.strip()
    )
    oof_frames = [_read_scores(path) for path in args.oof]
    validation_frames = [_read_scores(path) for path in args.validation]
    oof_scores = (
        merge_component_scores(oof_frames, component_columns=list(component_models))
        if len(oof_frames) > 1
        else oof_frames[0]
    )
    validation_scores = (
        merge_component_scores(validation_frames, component_columns=list(component_models))
        if len(validation_frames) > 1
        else validation_frames[0]
    )
    summary = fit_and_persist_fusion(
        oof_scores,
        validation_scores,
        args.output,
        component_models=component_models,
        alert_fraction=(
            args.alert_fraction
            if args.alert_fraction is not None
            else fusion_alert_fraction_from_configuration(model_configuration)
        ),
        random_seed=args.random_seed,
        model_config_path=args.model_config,
        input_paths={
            **{f"oof_{index}": path for index, path in enumerate(args.oof)},
            **{
                f"validation_{index}": path
                for index, path in enumerate(args.validation)
            },
        },
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "run_id": summary.run_id,
                "threshold": summary.calibrated_threshold,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
