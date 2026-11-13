"""Tests for the B2B commercial-contracts bounded domain milestone on v4."""
from __future__ import annotations

import pytest

from claimlayer.v4 import (
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    ClauseRelation,
    EvidenceLayer,
    EvidenceRef,
    FieldEvidence,
    SourceDocument,
    TIER_ANNEX,
    TIER_PRIMARY,
    UnmappedProposal,
    authorize_operation,
    evaluate_dual_readers,
    resolve_scope_and_precedence,
    verify_extraction_fidelity,
)


def _doc(doc_id: str, text: str, *, authority_rank: int = TIER_PRIMARY) -> SourceDocument:
    return SourceDocument(document_id=doc_id, text=text, authority_rank=authority_rank, weight=1.0)


def _ref(doc: SourceDocument, substring: str) -> EvidenceRef:
    start = doc.text.index(substring)
    return EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=start + len(substring),
        quote=substring,
    )


def test_condition_status_prevents_silent_unconditional_collapse():
    # Missing condition_status MUST be rejected when require_explicit_condition_status=True
    with pytest.raises(UnmappedProposal) as exc:
        B2B_CONTRACTS_VOCABULARY.proposition(
            {
                "entity": "Acme Corp",
                "predicate": "payment_terms",
                "value": 30,
                "unit": "days",
                "assertion_kind": "requirement",
            }
        )
    assert exc.value.reason == "condition_status_required"

    # Explicit unknown_condition and unrepresentable_condition are preserved as distinct states
    unk = B2B_CONTRACTS_VOCABULARY.proposition(
        {
            "entity": "Acme Corp",
            "predicate": "payment_terms",
            "value": 30,
            "unit": "days",
            "assertion_kind": "requirement",
            "condition_status": "unknown_condition",
        }
    )
    unrep = B2B_CONTRACTS_VOCABULARY.proposition(
        {
            "entity": "Acme Corp",
            "predicate": "payment_terms",
            "value": 30,
            "unit": "days",
            "assertion_kind": "requirement",
            "condition_status": "unrepresentable_condition",
        }
    )
    assert unk["condition_status"] == "unknown_condition"
    assert unrep["condition_status"] == "unrepresentable_condition"


def test_extraction_fidelity_catches_dropped_proviso_polarity_and_bad_field_evidence():
    doc = _doc(
        "msa_1",
        "Acme shall pay undisputed invoices within 30 days, unless disputed in good faith. "
        "Vendor shall not be liable for indirect damages.",
    )
    layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer.ingest(doc)

    # 1. Truncated proviso + unconditional claim when sentence has "unless"
    res = layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "evidence": [_ref(doc, "Acme shall pay undisputed invoices within 30 days")],
            }
        ],
        attestor="test_app",
    )
    unit = layer.unit(res["unit_ids"][0])
    findings = verify_extraction_fidelity(
        doc.text,
        unit,
        field_evidence=[FieldEvidence("value", _ref(doc, "indirect damages"))],
        require_field_evidence=("entity", "value"),
    )
    codes = {f.code for f in findings}
    assert "TRUNCATED_CLAUSE_CONTEXT" in codes
    assert "UNCAPTURED_CONDITION" in codes
    assert "MISSING_FIELD_EVIDENCE" in codes
    assert "VALUE_NOT_IN_FIELD_EVIDENCE" in codes


def test_dual_reader_evaluation_reports_shared_errors():
    report = evaluate_dual_readers(
        [
            {"gold": "A", "reader_a": "A", "reader_b": "A"},
            {"gold": "B", "reader_a": "X", "reader_b": "X"},  # correlated error!
            {"gold": "C", "reader_a": "C", "reader_b": "Y"},
            {"gold": "D", "reader_a": "D", "reader_b": "D"},
        ]
    )
    assert report["agreement_rate"] == 0.75
    assert report["both_correct_rate"] == 0.50
    assert report["shared_error_rate"] == 0.25
    assert report["agreement_reliability"] == round(2 / 3, 4)


def test_scope_precedence_exceptions_amendments_and_pending_conditions():
    msa = _doc(
        "msa_2025",
        "Clause 4.1: Standard payment terms are 45 days. "
        "Clause 4.2: For Enterprise EU customers, payment terms are 60 days. "
        "Clause 4.3: Payment terms are 15 days if the invoice is disputed.",
    )
    amend = _doc(
        "amend_2026",
        "Amendment Clause 4.1A: Standard payment terms are amended to 30 days effective 2026-01-01.",
    )
    layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer.ingest([msa, amend])
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 45,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "c4_1"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "Clause 4.1: Standard payment terms are 45 days.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 60,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "c4_2", "service_tier": "enterprise", "region": "eu"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "Clause 4.2: For Enterprise EU customers, payment terms are 60 days.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 15,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "conditional",
                    "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": True}],
                    "scope": {"clause_id": "c4_3"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "Clause 4.3: Payment terms are 15 days if the invoice is disputed.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "c4_1a"},
                    "valid_from": "2026-01-01",
                },
                "evidence": [_ref(amend, "Amendment Clause 4.1A: Standard payment terms are amended to 30 days effective 2026-01-01.")],
            },
        ],
        attestor="legal_ops",
    )

    rels = [ClauseRelation("amends_clause", "c4_1a", "c4_1", valid_from="2026-01-01")]
    tiers = {"msa_2025": TIER_PRIMARY, "amend_2026": TIER_PRIMARY}

    # 1. Missing condition context -> PENDING_CONDITION (abstains!)
    r_pending = resolve_scope_and_precedence(
        layer.units,
        entity="Acme",
        predicate="payment_terms",
        vocabulary=B2B_CONTRACTS_VOCABULARY,
        as_of="2026-06-01",
        clause_relations=rels,
        source_tiers=tiers,
        policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
    )
    assert r_pending["status"] == "PENDING_CONDITION"

    # 2. Disputed invoice -> conditional exception (15 days) overrides general amendment (30 days)
    r_exc = resolve_scope_and_precedence(
        layer.units,
        entity="Acme",
        predicate="payment_terms",
        vocabulary=B2B_CONTRACTS_VOCABULARY,
        as_of="2026-06-01",
        query_conditions={"invoice_disputed": True},
        clause_relations=rels,
        source_tiers=tiers,
        policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
    )
    assert r_exc["status"] == "RESOLVED"
    assert r_exc["value"] == {"value": "15", "unit": "calendar_days"}

    # 3. Undisputed Enterprise EU -> more specific scope (60 days) overrides general amendment (30 days)
    r_scope = resolve_scope_and_precedence(
        layer.units,
        entity="Acme",
        predicate="payment_terms",
        vocabulary=B2B_CONTRACTS_VOCABULARY,
        as_of="2026-06-01",
        query_scope={"service_tier": "enterprise", "region": "eu"},
        query_conditions={"invoice_disputed": False},
        clause_relations=rels,
        source_tiers=tiers,
        policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
    )
    assert r_scope["status"] == "RESOLVED"
    assert r_scope["value"] == {"value": "60", "unit": "calendar_days"}

    # 4. Undisputed standard customer in 2026 -> amendment (30 days) supersedes 45 days
    r_amend = resolve_scope_and_precedence(
        layer.units,
        entity="Acme",
        predicate="payment_terms",
        vocabulary=B2B_CONTRACTS_VOCABULARY,
        as_of="2026-06-01",
        query_conditions={"invoice_disputed": False},
        clause_relations=rels,
        source_tiers=tiers,
        policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
    )
    assert r_amend["status"] == "RESOLVED"
    assert r_amend["value"] == {"value": "30", "unit": "calendar_days"}


def test_authorize_operation_blocks_on_gaps_and_policy_limits():
    doc = _doc(
        "msa_clean",
        "Standard payment terms are 30 days. Aggregate liability cap is 500000 USD.",
    )
    layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer.ingest(doc)
    res = layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "evidence": [_ref(doc, "Standard payment terms are 30 days.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "liability_cap",
                    "value": 500000,
                    "unit": "USD",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "evidence": [_ref(doc, "Aggregate liability cap is 500000 USD.")],
            },
        ],
        attestor="legal_ops",
    )
    fe = {
        res["unit_ids"][0]: [
            FieldEvidence("entity", _ref(doc, "Standard payment terms")),
            FieldEvidence("value", _ref(doc, "30 days")),
        ],
        res["unit_ids"][1]: [
            FieldEvidence("entity", _ref(doc, "Aggregate liability cap")),
            FieldEvidence("value", _ref(doc, "500000 USD")),
        ],
    }

    # Read error MUST block authorization even when admitted units look fine
    blocked_read = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("payment_terms", "liability_cap"),
        ingest_summary={"read_errors": [{"document_id": "annex_1", "error_type": "Timeout"}]},
        field_evidences=fe,
        require_field_evidence=("entity", "value"),
    )
    assert blocked_read.status == "RETRY"
    assert blocked_read.allowed is False

    # Policy bound exceeded MUST escalate
    blocked_cap = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("payment_terms", "liability_cap"),
        field_evidences=fe,
        require_field_evidence=("entity", "value"),
        numeric_bounds={"liability_cap": (None, 250000.0)},
    )
    assert blocked_cap.status == "ESCALATE"
    assert blocked_cap.requires_human is True

    # Clean epistemic integrity + within policy bounds -> PASS
    passed = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("payment_terms", "liability_cap"),
        field_evidences=fe,
        require_field_evidence=("entity", "value"),
        numeric_bounds={"liability_cap": (None, 1000000.0)},
    )
    assert passed.status == "PASS"
    assert passed.allowed is True
    assert passed.requires_human is False
