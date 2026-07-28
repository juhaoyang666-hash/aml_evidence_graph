from __future__ import annotations

from pathlib import Path

from aml_evidence_graph.investigation.agent_evaluation import (
    evaluate_agent_cases,
    load_agent_cases,
)


def test_agent_golden_has_sixty_passing_auditable_cases() -> None:
    cases = load_agent_cases(Path("golden/agent_cases_v2.json"))
    metrics = evaluate_agent_cases(cases)

    assert len(cases) == 60
    assert metrics["pass_rate"] == 1.0
    assert metrics["tool_selection_accuracy"] == 1.0
    assert metrics["parameter_validity_accuracy"] == 1.0
    assert metrics["fact_consistency_rate"] == 1.0
    assert metrics["recovery_success_rate"] == 1.0
    assert metrics["token_cost_coverage"]["coverage"] == 0.0
    assert metrics["bad_cases"] == []
