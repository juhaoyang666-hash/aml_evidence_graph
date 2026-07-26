#!/usr/bin/env python3
"""B4: DuckDB/Polars replay of sender_outgoing_count_7d vs official PIT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from aml_evidence_graph.data.contract import CANONICAL

FEATURE = "sender_outgoing_count_7d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pit-root", type=Path, default=Path("artifacts/pit_features"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("artifacts/prepared_transactions")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/batch_feature_replay"))
    parser.add_argument("--max-days", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    day_dirs = sorted(args.pit_root.glob("event_date=*/split=*"))[-args.max_days :]
    official_parts = []
    for day in day_dirs:
        for part in day.glob("*.parquet"):
            official_parts.append(
                pd.read_parquet(
                    part,
                    columns=[
                        CANONICAL.transaction_id,
                        CANONICAL.event_ts,
                        CANONICAL.sender_account_id,
                        FEATURE,
                    ],
                )
            )
    official = pd.concat(official_parts, ignore_index=True)
    official[CANONICAL.event_ts] = pd.to_datetime(official[CANONICAL.event_ts], utc=True)
    min_day = (official[CANONICAL.event_ts].min() - pd.Timedelta(days=8)).normalize()
    max_day = official[CANONICAL.event_ts].max().normalize()
    prepared_glob = []
    day = min_day
    while day <= max_day:
        prepared_glob.extend(
            str(p)
            for p in (args.prepared_root / f"event_date={day.date()}").glob(
                "split=*/*.parquet"
            )
        )
        day += pd.Timedelta(days=1)
    if not prepared_glob:
        raise FileNotFoundError("No prepared files in lookback window")

    con = duckdb.connect()
    con.register("official_df", official)
    path_sql = "[" + ", ".join(f"'{p}'" for p in prepared_glob) + "]"

    t0 = time.perf_counter()
    duck = con.execute(
        f"""
        WITH hist AS (
          SELECT
            sender_account_id::VARCHAR AS sender_account_id,
            event_ts::TIMESTAMPTZ AS event_ts,
            transaction_id::VARCHAR AS transaction_id
          FROM read_parquet({path_sql})
        ),
        hist_plus AS (
          SELECT
            sender_account_id,
            event_ts,
            transaction_id,
            COUNT(*) OVER (
              PARTITION BY sender_account_id
              ORDER BY event_ts
              RANGE BETWEEN INTERVAL 7 DAY PRECEDING AND INTERVAL 1 MICROSECOND PRECEDING
            ) AS duckdb_value
          FROM hist
        )
        SELECT
          o.transaction_id,
          o.{FEATURE} AS official_value,
          COALESCE(h.duckdb_value, 0) AS duckdb_value
        FROM official_df o
        LEFT JOIN hist_plus h USING (transaction_id)
        """
    ).fetchdf()
    duck_s = time.perf_counter() - t0

    # Polars: same window semantics via group rolling on prepared hist, join to official
    t1 = time.perf_counter()
    hist = pl.concat([pl.read_parquet(p) for p in prepared_glob]).select(
        [
            pl.col(CANONICAL.sender_account_id).cast(pl.Utf8),
            pl.col(CANONICAL.event_ts).cast(pl.Datetime(time_unit="us", time_zone="UTC")),
            pl.col(CANONICAL.transaction_id).cast(pl.Utf8),
        ]
    ).sort([CANONICAL.sender_account_id, CANONICAL.event_ts])
    hist = hist.with_columns(
        pl.col(CANONICAL.event_ts)
        .count()
        .rolling(index_column=CANONICAL.event_ts, period="7d", closed="left")
        .over(CANONICAL.sender_account_id)
        .alias("polars_value")
    )
    # rolling count includes current row depending on closed=; closed=left excludes current if index aligns
    scored = pl.from_pandas(official).select(
        [
            pl.col(CANONICAL.transaction_id).cast(pl.Utf8),
            pl.col(FEATURE).alias("official_value"),
        ]
    )
    polars_df = scored.join(
        hist.select([CANONICAL.transaction_id, "polars_value"]),
        on=CANONICAL.transaction_id,
        how="left",
    ).with_columns(pl.col("polars_value").fill_null(0)).to_pandas()
    polars_s = time.perf_counter() - t1

    duck["match"] = np.isclose(
        duck["official_value"].astype(float), duck["duckdb_value"].astype(float)
    )
    polars_df["match"] = np.isclose(
        polars_df["official_value"].astype(float), polars_df["polars_value"].astype(float)
    )
    summary = {
        "feature": FEATURE,
        "protocol": {
            "max_days_scored": args.max_days,
            "scored_rows": int(len(official)),
            "prepared_files": len(prepared_glob),
            "honest_boundary": (
                "Read-only replay vs official PIT; does not change main-line features. "
                "Match rate may be <1 if same-second isolation differs from engine RANGE."
            ),
        },
        "throughput": {
            "duckdb_seconds": duck_s,
            "polars_seconds": polars_s,
            "duckdb_rows_per_second": len(duck) / max(duck_s, 1e-9),
            "polars_rows_per_second": len(polars_df) / max(polars_s, 1e-9),
        },
        "equality": {
            "duckdb_match_rate": float(duck["match"].mean()),
            "polars_match_rate": float(polars_df["match"].mean()),
            "duckdb_abs_err_mean": float(
                (duck["official_value"].astype(float) - duck["duckdb_value"].astype(float))
                .abs()
                .mean()
            ),
            "polars_abs_err_mean": float(
                (
                    polars_df["official_value"].astype(float)
                    - polars_df["polars_value"].astype(float)
                )
                .abs()
                .mean()
            ),
        },
    }
    duck.to_parquet(args.output_dir / "duckdb_replay.parquet", index=False)
    polars_df.to_parquet(args.output_dir / "polars_replay.parquet", index=False)
    (args.output_dir / "batch_replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
