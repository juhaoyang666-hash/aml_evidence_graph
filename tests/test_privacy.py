from aml_evidence_graph.data.privacy import tokenise_identifier


def test_tokenise_identifier_is_deterministic_and_namespaced() -> None:
    first = tokenise_identifier("account-value", secret="test-secret", namespace="account")
    second = tokenise_identifier("account-value", secret="test-secret", namespace="account")
    transaction = tokenise_identifier(
        "account-value",
        secret="test-secret",
        namespace="transaction",
    )

    assert first == second
    assert first != transaction
    assert "account-value" not in first
