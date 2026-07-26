"""Backfill rule_*_hit columns into an existing PIT feature dataset.

Threshold rules are pure functions of columns that already exist in the PIT output,
so changing `configs/rules/default.yaml` does not require replaying the ~8h causal
history build. This script rewrites each partition in place, adding only the rule
columns and the `_rule_evidence` sidecar that `features.build` would have written.

It refuses to run when a rule references a column the dataset does not contain,
because such a rule genuinely needs a full PIT rebuild.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from aml_evidence_graph.rules.engine import apply_rules, load_rules


def _partition_files(feature_root: Path) -> list[Path]:
    files = sorted(feature_root.glob("event_date=*/split=*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No PIT partitions found beneath {feature_root}")
    return files


def _event_date_of(path: Path, feature_root: Path) -> str:
    for part in path.relative_to(feature_root).parts:
        if part.startswith("event_date="):
            return part.removeprefix("event_date=")
    raise ValueError(f"Cannot infer event_date from {path}")


def backfill(feature_root: Path, rules_path: Path, *, dry_run: bool = False) -> dict[str, int]:
    rules = load_rules(rules_path)
    active = [rule for rule in rules if rule.active]
    if not active:
        raise ValueError(
            f"{rules_path} has no active approved rules; nothing to backfill. "
            "An approved rule needs threshold, approval_reference and backtest_summary."
        )

    files = _partition_files(feature_root)
    available = set(pq.read_schema(files[0]).names)
    missing = sorted({rule.feature for rule in active}.difference(available))
    if missing:
        raise ValueError(
            "These rule features are absent from the PIT dataset, so a full rebuild "
            f"is required: {', '.join(missing)}"
        )

    evidence_dir = feature_root / "_rule_evidence"
    stale_rule_columns = sorted(
        name for name in available if name.startswith("rule_") and name.endswith("_hit")
    )
    total_rows = 0
    total_hits = 0

    for path in files:
        event_date = _event_date_of(path, feature_root)
        table = pq.read_table(path)
        frame = table.to_pandas()
        if stale_rule_columns:
            frame = frame.drop(columns=stale_rule_columns, errors="ignore")
        rule_features, hits = apply_rules(
            frame,
            rules,
            as_of_date=date.fromisoformat(event_date),
        )
        for column in rule_features.columns:
            frame[column] = rule_features[column]
        for rule in active:
            column = f"rule_{rule.rule_id}_hit"
            if column not in frame:
                frame[column] = 0
        total_rows += len(frame)
        total_hits += len(hits)

        if dry_run:
            continue

        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(
            pa.Table.from_pandas(frame, preserve_index=False),
            temporary,
            compression="zstd",
        )
        temporary.replace(path)

        if hits:
            evidence_dir.mkdir(exist_ok=True)
            (evidence_dir / f"event_date={event_date}.json").write_text(
                json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = {
        "partition_count": len(files),
        "row_count": total_rows,
        "rule_hit_count": total_hits,
        "active_rule_count": len(active),
    }
    if not dry_run:
        (feature_root / "_rule_backfill_summary.json").write_text(
            json.dumps(
                {
                    **summary,
                    "rules_path": str(rules_path),
                    "rule_version": active[0].version,
                    "rule_ids": [rule.rule_id for rule in active],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="PIT feature dataset root.")
    parser.add_argument("--rules", type=Path, required=True, help="Versioned rule YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report hit counts without rewriting any partition.",
    )
    parser.add_argument(
        "--backup-evidence",
        action="store_true",
        help="Move an existing _rule_evidence directory aside before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backup_evidence:
        evidence = args.features / "_rule_evidence"
        if evidence.is_dir():
            shutil.move(str(evidence), str(evidence.with_name("_rule_evidence.previous")))
    summary = backfill(args.features, args.rules, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
