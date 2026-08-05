#!/usr/bin/env python3
"""Build a DEVELOPMENT set for the prose-boundary defect. Not a holdout.

Holdout v4 showed field names reaching annotation prose on roughly half of unseen cases
while the existing development set leaks on 2 of 26. That set therefore cannot measure a
fix: it is one of the sets the prompt was tuned against. This builds a set whose names
none of prompt v1-v8 was developed on, so a candidate can be iterated against it.

Iterating against these cases is exactly what makes them unusable as evidence of
generalisation. Any promotion still requires a preregistered holdout on a set that has
never been looked at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Vocabulary deliberately disjoint from cases_v1 and holdout sets v1-v4, so a leak here
# cannot be one the prompt was already tuned to avoid.
WIDE_SPECS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...], bool], ...] = (
    (
        "custody",
        ("custody_transfer_context", "dormancy_break_context", "velocity_shift_context"),
        "RULE-CUSTODY-TRANSFER",
        ("TYPOLOGY-MONEY-MULE", "TYPOLOGY-CYCLE"),
        True,
    ),
    (
        "remittance",
        ("remittance_corridor_context", "endorsement_chain_context", "provenance_trail_context"),
        "RULE-REMITTANCE-CORRIDOR",
        ("TYPOLOGY-TRADE-BASED", "TYPOLOGY-FUNNEL-ACCOUNT"),
        False,
    ),
    (
        "disbursement",
        ("disbursement_split_context", "reconciliation_gap_context", "proximity_cluster_context"),
        "RULE-DISBURSEMENT-SPLIT",
        ("TYPOLOGY-PAYMENT-PROCESSOR", "TYPOLOGY-CASH-WITHDRAWAL"),
        True,
    ),
    (
        "aggregation",
        ("aggregation_depth_context", "settlement_lag_context", "beneficiary_reuse_context"),
        "RULE-AGGREGATION-DEPTH",
        ("TYPOLOGY-BENEFICIAL-OWNERSHIP", "TYPOLOGY-VIRTUAL-ASSET"),
        True,
    ),
)

NARROW_SPECS: tuple[tuple[str, str, str], ...] = (
    ("dormancy", "dormancy_break_context", "TYPOLOGY-MONEY-MULE"),
    ("provenance", "provenance_trail_context", "TYPOLOGY-TRADE-BASED"),
    ("proximity", "proximity_cluster_context", "TYPOLOGY-CYCLE"),
)

LOW_SPECS: tuple[tuple[str, str | None, list[str]], ...] = (
    ("no-records", None, ["Every supporting record category is unavailable."]),
    ("empty-list", "reconciliation_gap_context", []),
    ("partial", "settlement_lag_context", ["Timing records are unavailable."]),
    ("scores-only", None, ["Feature records are unavailable."]),
    ("sparse-graph", "beneficiary_reuse_context", ["Network context is incomplete."]),
)

TITLES = {
    "TYPOLOGY-MONEY-MULE": "Money Mule / 跑分过渡账户",
    "TYPOLOGY-FUNNEL-ACCOUNT": "Funnel Account / 异地漏斗账户",
    "TYPOLOGY-PAYMENT-PROCESSOR": "Third-Party Payment Processor Abuse / 第三方支付处理商滥用",
    "TYPOLOGY-TRADE-BASED": "Trade-Based Money Laundering / 贸易型洗钱",
    "TYPOLOGY-VIRTUAL-ASSET": "Virtual-Asset Conversion and Mixing / 虚拟资产转换与混币",
    "TYPOLOGY-BENEFICIAL-OWNERSHIP": "Concealed Beneficial Ownership / 隐匿受益所有人",
    "TYPOLOGY-CYCLE": "Circular / Cycle Transfers",
    "TYPOLOGY-CASH-WITHDRAWAL": "Cash Withdrawal",
}


def _evidence(
    index: int,
    *,
    features: tuple[str, ...],
    rule_id: str | None = None,
    typologies: tuple[str, ...] = (),
    graph: bool = False,
    missing: list[str] | None = None,
    uncertainty: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "alert_id": f"prose-dev-alert-{index:03d}",
        "generated_at": "2026-08-05T05:00:00Z",
        "transaction_id": f"prose-dev-transaction-{index:03d}",
        "event_timestamp": f"2023-08-{index + 1:02d}T12:00:00Z",
        "model_probabilities": {"catboost": 0.31 + index / 100, "gat": 0.42 + index / 100},
        "fusion_probability": 0.37 + index / 100,
        "rule_hits": (
            [
                {
                    "rule_id": rule_id,
                    "rule_version": "2026.1",
                    "feature": features[0],
                    "observed_value": float(index + 2),
                    "threshold": float(index + 1),
                    "operator": "gt",
                    "explanation": "Synthetic development rule hit; not a case conclusion.",
                }
            ]
            if rule_id
            else []
        ),
        "key_features": [
            {
                "name": name,
                "value": float(index + position + 1),
                "window": "30d",
                "source": "prose_boundary_dev_feature",
            }
            for position, name in enumerate(features)
        ],
        "graph_evidence": (
            {
                "source_node_index": index * 2,
                "destination_node_index": index * 2 + 1,
                "historical_source_out_degree": index + 1,
                "historical_destination_in_degree": index + 2,
                "prior_directed_edge_count": index % 3,
                "prior_reverse_edge_count": (index + 1) % 2,
                "two_hop_intermediary_count": index % 4,
                "two_hop_intermediary_node_indices": [],
                "snapshot_as_of": f"2023-08-{index + 1:02d}T00:00:00Z",
                "interpretation_limit": (
                    "Synthetic development evidence; history-only structure is not a "
                    "case conclusion."
                ),
            }
            if graph
            else None
        ),
        "typology_references": [
            {
                "typology_id": typology_id,
                "version": "2026.1",
                "title": TITLES[typology_id],
                "source": "prose-boundary-development",
            }
            for typology_id in typologies
        ],
        "source_versions": {"golden": "llm-prose-boundary-development"},
        "missing_evidence": missing or [],
        "uncertainty_notes": uncertainty or [],
    }


def build_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index, (key, features, rule_id, typologies, graph) in enumerate(WIDE_SPECS):
        cases.append(
            {
                "case_id": f"prose-dev-wide-{index + 1:02d}-{key}",
                "case_category": "typology",
                "evidence": _evidence(
                    index,
                    features=features,
                    rule_id=rule_id,
                    typologies=typologies,
                    graph=graph,
                    uncertainty=[f"Unverified {key} lead awaiting corroboration."],
                ),
                "expected_typology_ids": [typologies[0]],
                "expect_rejected_facts": False,
            }
        )
    for offset, (key, feature, typology_id) in enumerate(NARROW_SPECS, start=4):
        cases.append(
            {
                "case_id": f"prose-dev-typo-{offset - 3:02d}-{key}",
                "case_category": "typology",
                "evidence": _evidence(
                    offset,
                    features=(feature,),
                    typologies=(typology_id,),
                    uncertainty=[f"Unverified {key} lead."],
                ),
                "expected_typology_ids": [typology_id],
                "expect_rejected_facts": False,
            }
        )
    for offset, (name, feature, missing) in enumerate(LOW_SPECS, start=7):
        cases.append(
            {
                "case_id": f"prose-dev-low-{offset - 6:02d}-{name}",
                "case_category": "low_evidence",
                "evidence": _evidence(
                    offset,
                    features=(feature,) if feature else (),
                    graph=name == "sparse-graph",
                    missing=missing,
                ),
                "expected_typology_ids": [],
                "expect_rejected_facts": False,
            }
        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("golden/llm_prose_boundary_dev_cases.json")
    )
    args = parser.parse_args()
    cases = build_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"development cases: {len(cases)}  output: {args.output}")


if __name__ == "__main__":
    main()
