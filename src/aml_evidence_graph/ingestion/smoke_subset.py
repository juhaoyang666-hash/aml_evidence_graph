"""Create a small prepared-date subset for smoke end-to-end runs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# Sparse but protocol-valid dates spanning train / validation / test.
# Include two training months so expanding-time OOF can run with
# --minimum-training-months 1 --splits 1 (fit on March, score April).
DEFAULT_SMOKE_DATES = (
    "2023-03-28",
    "2023-03-29",
    "2023-03-30",
    "2023-04-26",
    "2023-04-27",
    "2023-04-28",
    "2023-04-29",
    "2023-04-30",
    "2023-05-01",
    "2023-05-02",
    "2023-07-01",
    "2023-07-02",
)


@dataclass(frozen=True)
class SmokeSubsetSummary:
    created_at_utc: str
    source_root: str
    output_root: str
    copied_dates: list[str]
    missing_dates: list[str]


def prepare_smoke_subset(
    source_root: Path,
    output_root: Path,
    *,
    event_dates: tuple[str, ...] = DEFAULT_SMOKE_DATES,
    overwrite: bool = False,
) -> SmokeSubsetSummary:
    """Copy selected Hive event_date partitions for a fast pipeline smoke."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Prepared source does not exist: {source_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Smoke output already exists: {output_root}. Pass overwrite=True."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    copied: list[str] = []
    missing: list[str] = []
    for event_date in event_dates:
        source_partition = source_root / f"event_date={event_date}"
        if not source_partition.is_dir():
            missing.append(event_date)
            continue
        destination = output_root / f"event_date={event_date}"
        shutil.copytree(source_partition, destination)
        copied.append(event_date)

    if not copied:
        raise FileNotFoundError(
            "None of the requested smoke dates exist under the prepared dataset."
        )
    if missing:
        # Allow partial smoke when a few dates are absent, but record them.
        pass

    summary = SmokeSubsetSummary(
        created_at_utc=datetime.now(UTC).isoformat(),
        source_root=str(source_root),
        output_root=str(output_root),
        copied_dates=copied,
        missing_dates=missing,
    )
    (output_root / "_smoke_subset_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_SMOKE_DATES),
        help="ISO event dates to copy (default covers train/val/test smoke).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_smoke_subset(
        args.input,
        args.output,
        event_dates=tuple(args.dates),
        overwrite=args.overwrite,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
