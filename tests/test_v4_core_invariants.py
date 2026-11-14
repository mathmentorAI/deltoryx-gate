"""Failure-family invariant tests for ClaimLayer v4 core unification.

Verifies three structural properties across failure classes rather than isolated examples:
1. Single FieldSpec + UnitRegistry quantity/bound/condition semantics across dimensions.
2. Strict equivalence across ask(), check_claim(), resolve_scope_and_precedence(), and authorize_operation().
3. Persisted 'authorization' state binding and fail-closed verification in VerificationStore.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from claimlayer.semantic.models import EvidenceRef, SourceDocument, StructuredQuery, canonical
from claimlayer.v4.authorization import (
    ClauseRelation,
    authorize_operation,
    execute_authorized_operation,
    resolve_scope_and_precedence,
    verify_authorization_verdict,
)
from claimlayer.v4.domains import B2B_CONTRACTS_VOCABULARY, B2B_DEFAULT_ACCEPTANCE_POLICY
from claimlayer.v4.engine import TIER_PRIMARY, EvidenceLayer
from claimlayer.v4.epistemics import EXPLICIT
from claimlayer.v4.fidelity import FieldEvidence, verify_extraction_fidelity
from claimlayer.v4.models import AcceptancePolicy, EvidenceUnit
from claimlayer.v4.vocabulary import FieldSpec, PredicateSpec, Vocabulary


def _invariant_vocab() -> Vocabulary:
    return Vocabulary(
        name="core-invariants",
        version="4.0",
        predicates=(
            PredicateSpec(
                "duration_days",
                FieldSpec("quantity", dimensions=("duration_calendar_days",)),
                assertion_kinds=("requirement", "fact"),
                conditions=(("window", FieldSpec("quantity", dimensions=("duration_calendar_days",))),),
            ),
            PredicateSpec(
                "multi_dim_value",
                FieldSpec(
                    "quantity",
                    dimensions=(
                        "currency:USD",
                        "currency:EUR",
                        "duration_calendar_days",
                        "duration_business_days",
                        "duration_months",
                        "ratio",
                        "ratio_difference",
                        "annual_rate",
                        "hours_per_week",
                    ),
                ),
                assertion_kinds=("requirement", "fact"),
            ),
            PredicateSpec(
                "multi_currency_cap",
                FieldSpec("quantity", dimensions=("currency:USD", "currency:EUR")),
                assertion_kinds=("requirement",),
            ),
            PredicateSpec(
                "plain_score",
                FieldSpec("number"),
                assertion_kinds=("requirement", "fact"),
            ),
        ),
        scope_keys=("service_tier", "clause_id", "event_id"),
        require_explicit_condition_status=True,
    )


def _layer(tmpdir: str, vocab: Vocabulary | None = None, *, policy: AcceptancePolicy | None = None) -> EvidenceLayer:
    return EvidenceLayer(
        Path(tmpdir) / "ledger.db",
        project_id="inv",
        vocabulary=vocab or _invariant_vocab(),
        policy=policy or AcceptancePolicy(allowed_assertion_kinds=("requirement",)),
    )


def _admit_single(
    layer: EvidenceLayer,
    doc: SourceDocument,
    proposal: dict,
    *,
    quote: str | None = None,
    reliability: float | None = 1.0,
) -> EvidenceUnit:
    q = quote if quote is not None else doc.text
    start = doc.text.index(q)
    ref = EvidenceRef(doc.document_id, doc.revision, start, start + len(q), q)
    summary = layer.ingest_units(
        [{"proposal": proposal, "basis": EXPLICIT, "evidence": [ref]}],
        attestor="test_reader",
        structural_reliability=reliability,
    )
    assert summary["units_admitted"] == 1, f"Admission failed: {summary}"
    return layer.unit(summary["unit_ids"][0])


# ---------------------------------------------------------------------------
# Group 1: Cartesian Unit / Dimension Matrix & Typed Condition / Bound Rules
# ---------------------------------------------------------------------------

DIMENSION_SAMPLES: tuple[tuple[str, str, str], ...] = (
    ("currency:USD", "Acme multi_dim_value obligation is 23 USD.", "USD"),
    ("currency:EUR", "Acme multi_dim_value obligation is 23 EUR.", "EUR"),
    ("duration_calendar_days", "Acme multi_dim_value obligation is 23 days.", "calendar_days"),
    ("duration_business_days", "Acme multi_dim_value obligation is 23 business days.", "business_days"),
    ("duration_months", "Acme multi_dim_value obligation is 23 months.", "months"),
    ("ratio", "Acme multi_dim_value obligation is 23%.", "percent"),
    ("ratio_difference", "Acme multi_dim_value obligation is 23 pp.", "percentage_point"),
    ("annual_rate", "Acme multi_dim_value obligation is 23% per annum.", "percent_per_annum"),
    ("hours_per_week", "Acme multi_dim_value obligation is 23 hours/week.", "hours_per_week"),
)


def test_cartesian_dimension_matrix_in_fidelity():
    vocab = _invariant_vocab()
    for doc_dim, doc_text, doc_unit in DIMENSION_SAMPLES:
        doc = SourceDocument("contract.txt", doc_text, authority_rank=100, weight=1.0)
        ref = EvidenceRef(doc.document_id, doc.revision, 0, len(doc_text), doc_text)
        for prop_dim, _, prop_unit in DIMENSION_SAMPLES:
            prop = vocab.proposition(
                {
                    "entity": "Acme",
                    "predicate": "multi_dim_value",
                    "value": "23",
                    "unit": prop_unit,
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                }
            )
            unit = EvidenceUnit(prop, (ref,), EXPLICIT, "application", {"id": "test"}, 1.0)
            findings = verify_extraction_fidelity(doc_text, unit, vocabulary=vocab)
            if doc_dim == prop_dim:
                assert findings == [], f"Expected match for {doc_unit} vs {prop_unit}: {findings}"
            else:
                assert "VALUE_NOT_IN_EVIDENCE" in {f.code for f in findings}, (
                    f"Cross-dimension leak: prose {doc_text!r} accepted as {prop_unit}!"
                )


def test_exact_within_dimension_unit_conversions_and_dimensionless_separation():
    vocab = _invariant_vocab()

    # Exact within-dimension conversions succeed (2 weeks == 14 calendar_days, 2300 bps == 23 pp)
    for text, unit_name, val in [
        ("Acme multi_dim_value window is 2 weeks.", "calendar_days", "14"),
        ("Acme multi_dim_value delta is 2300 bps.", "percentage_point", "23"),
    ]:
        prop = vocab.proposition(
            {
                "entity": "Acme",
                "predicate": "multi_dim_value",
                "value": val,
                "unit": unit_name,
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            }
        )
        ref = EvidenceRef("d.txt", "rev-1", 0, len(text), text)
        u = EvidenceUnit(prop, (ref,), EXPLICIT, "application", {"id": "test"}, 1.0)
        assert verify_extraction_fidelity(text, u, vocabulary=vocab) == []

    # Quantity predicate MUST reject bare unitless prose numbers; dimensionless predicate MUST reject united prose
    doc_bare = SourceDocument("bare.txt", "Acme duration window is 23.", authority_rank=100, weight=1.0)
    prop_qty = vocab.proposition(
        {
            "entity": "Acme",
            "predicate": "duration_days",
            "value": "23",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        }
    )
    u_qty = EvidenceUnit(
        prop_qty,
        (EvidenceRef("bare.txt", doc_bare.revision, 0, len(doc_bare.text), doc_bare.text),),
        EXPLICIT,
        "application",
        {"id": "test"},
        1.0,
    )
    assert "VALUE_NOT_IN_EVIDENCE" in {
        f.code for f in verify_extraction_fidelity(doc_bare.text, u_qty, vocabulary=vocab)
    }

    doc_united = SourceDocument("usd.txt", "Acme plain score is 23 USD.", authority_rank=100, weight=1.0)
    prop_num = vocab.proposition(
        {
            "entity": "Acme",
            "predicate": "plain_score",
            "value": 23,
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        }
    )
    u_num = EvidenceUnit(
        prop_num,
        (EvidenceRef("usd.txt", doc_united.revision, 0, len(doc_united.text), doc_united.text),),
        EXPLICIT,
        "application",
        {"id": "test"},
        1.0,
    )
    assert "VALUE_NOT_IN_EVIDENCE" in {
        f.code for f in verify_extraction_fidelity(doc_united.text, u_num, vocabulary=vocab)
    }


def test_typed_conditions_and_bounds_reject_unitless_and_cross_dimension_inputs():
    vocab = _invariant_vocab()
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument(
            "msa.txt",
            "Acme duration is 14 days if window is less than 30 days.",
            authority_rank=100,
            weight=1.0,
        )
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Acme",
                "predicate": "duration_days",
                "value": "14",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "conditional",
                "conditions": [
                    {
                        "field": "window",
                        "operator": "lt",
                        "value": {"value": "30", "unit": "calendar_days"},
                    }
                ],
            },
        )

        # Within-dimension unit conversion in query_conditions ("2 weeks" -> 14 calendar_days < 30) -> PASS
        v_ok = authorize_operation(
            layer,
            entity="Acme",
            required_predicates=("duration_days",),
            query_conditions={"window": {"value": 2, "unit": "weeks"}},
            numeric_bounds={"duration_days": {"min": 1, "max": 3, "unit": "weeks"}},
        )
        assert v_ok.status == "PASS"

        # Unitless condition scalar or incompatible unit MUST escalate with INVALID_CONDITION_VALUE
        for bad_cond in (10, {"value": 10}, {"value": 10, "unit": "business_days"}, {"value": 10, "unit": "USD"}):
            v_bad_cond = authorize_operation(
                layer,
                entity="Acme",
                required_predicates=("duration_days",),
                query_conditions={"window": bad_cond},
            )
            assert v_bad_cond.status == "ESCALATE"
            assert "INVALID_CONDITION_VALUE" in {i.code for i in v_bad_cond.issues}

            ask_res = layer.ask(
                {
                    "entity": "Acme",
                    "predicate": "duration_days",
                    "context": {"window": bad_cond},
                },
                conditions="discharge",
            )
            assert ask_res["abstain"] is True
            assert ask_res["excluded"][0]["reason"] == "condition_invalid"

            prec = resolve_scope_and_precedence(
                layer.units,
                entity="Acme",
                predicate="duration_days",
                vocabulary=vocab,
                query_conditions={"window": bad_cond},
                documents={d.document_id: d for d in layer.documents},
                policy=layer.policy,
            )
            assert prec["status"] == "INVALID_CONDITION"

        # Unitless or wrong-dimension numeric_bounds MUST escalate with POLICY_BOUND_UNIT_MISMATCH
        for bad_bound in (
            (None, 20),
            {"max": 20},
            (None, 20, "business_days"),
            (None, 20, "USD"),
            {"max": 20, "unit": "USD"},
        ):
            v_bad_bound = authorize_operation(
                layer,
                entity="Acme",
                required_predicates=("duration_days",),
                query_conditions={"window": {"value": 2, "unit": "weeks"}},
                numeric_bounds={"duration_days": bad_bound},
            )
            assert v_bad_bound.status == "ESCALATE"
            assert "POLICY_BOUND_UNIT_MISMATCH" in {i.code for i in v_bad_bound.issues}

    # Cross-dimension multi-currency predicate: evidence in USD vs bound in EUR -> POLICY_BOUND_UNIT_MISMATCH
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc_usd = SourceDocument("cap.txt", "Acme currency cap is 500 USD.", authority_rank=100, weight=1.0)
        layer.ingest(doc_usd)
        _admit_single(
            layer,
            doc_usd,
            {
                "entity": "Acme",
                "predicate": "multi_currency_cap",
                "value": "500",
                "unit": "USD",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        v_eur_bound = authorize_operation(
            layer,
            entity="Acme",
            required_predicates=("multi_currency_cap",),
            numeric_bounds={"multi_currency_cap": (None, 1000.0, "EUR")},
        )
        assert v_eur_bound.status == "ESCALATE"
        assert "POLICY_BOUND_UNIT_MISMATCH" in {i.code for i in v_eur_bound.issues}


# ---------------------------------------------------------------------------
# Group 2: Strict 4-Gate Equivalence Across All 8 Inadmissibility Causes
# ---------------------------------------------------------------------------

def _assert_all_four_gates_refuse(
    layer: EvidenceLayer,
    *,
    predicate: str,
    claim_value: dict | str | int,
    claim_unit: str | None,
    expected_auth_status: str,
    expected_auth_code: str,
    expected_ask_reason_prefix: str | None,
    as_of: str | None = None,
    query_context: dict | None = None,
    claim_kind: str = "requirement",
) -> None:
    vocab = layer.vocabulary
    q_ctx = dict(query_context or {})

    ask_res = layer.ask(
        {"entity": "Orion", "predicate": predicate, "as_of": as_of, "context": q_ctx},
        conditions="discharge",
    )
    assert ask_res["abstain"] is True, f"ask() unexpectedly answered: {ask_res}"
    if expected_ask_reason_prefix is not None:
        assert any(
            ex["reason"].startswith(expected_ask_reason_prefix) for ex in ask_res["excluded"]
        ), f"Missing {expected_ask_reason_prefix} in ask excluded={ask_res['excluded']}"

    chk = layer.check_claim(
        {
            "entity": "Orion",
            "predicate": predicate,
            "value": claim_value,
            "unit": claim_unit,
            "assertion_kind": claim_kind,
            "condition_status": "unconditional",
        },
        as_of=as_of,
        context=q_ctx,
    )
    assert chk["status"] != "supported", f"check_claim() falsely supported: {chk}"
    assert chk["status"] != "contradicted", f"check_claim() falsely contradicted from inadmissible unit: {chk}"

    prec = resolve_scope_and_precedence(
        layer.units,
        entity="Orion",
        predicate=predicate,
        vocabulary=vocab,
        as_of=as_of,
        query_scope=q_ctx,
        query_conditions=q_ctx,
        documents={d.document_id: d for d in layer.documents},
        policy=layer.policy,
    )
    assert prec["status"] != "RESOLVED", f"resolve_scope_and_precedence() resolved: {prec}"

    verdict = authorize_operation(
        layer,
        entity="Orion",
        required_predicates=(predicate,),
        as_of=as_of,
        query_scope=q_ctx,
        query_conditions=q_ctx,
    )
    assert verdict.status == expected_auth_status, f"Expected {expected_auth_status}, got {verdict}"
    assert verdict.allowed is False
    assert verdict.resolved_values == {}
    assert expected_auth_code in {i.code for i in verdict.issues}


def test_four_gate_equivalence_across_eight_inadmissibility_causes():
    vocab = _invariant_vocab()

    # 1. Wrong numeric value (777 vs 23)
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "777",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="777",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="VALUE_NOT_IN_EVIDENCE",
            expected_ask_reason_prefix="fidelity:VALUE_NOT_IN_EVIDENCE",
        )

    # 2. Incompatible units (23 USD / 23 business days vs 23 calendar_days)
    for bad_text in ("Orion duration is 23 USD.", "Orion duration is 23 business days."):
        with tempfile.TemporaryDirectory() as td:
            layer = _layer(td, vocab)
            doc = SourceDocument("msa.txt", bad_text, authority_rank=100, weight=1.0)
            layer.ingest(doc)
            _admit_single(
                layer,
                doc,
                {
                    "entity": "Orion",
                    "predicate": "duration_days",
                    "value": "23",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
            )
            _assert_all_four_gates_refuse(
                layer,
                predicate="duration_days",
                claim_value="23",
                claim_unit="calendar_days",
                expected_auth_status="RETRY",
                expected_auth_code="VALUE_NOT_IN_EVIDENCE",
                expected_ask_reason_prefix="fidelity:VALUE_NOT_IN_EVIDENCE",
            )

    # 3. Negated sentence claimed as affirmed
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument("msa.txt", "Orion duration shall not be 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "polarity": "affirmed",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="23",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="POLARITY_MISMATCH",
            expected_ask_reason_prefix="fidelity:POLARITY_MISMATCH",
        )

    # 4. Excluded assertion_kind (fact under requirement-only policy, even with bad value)
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab, policy=AcceptancePolicy(allowed_assertion_kinds=("requirement",)))
        doc = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "777",
                "unit": "calendar_days",
                "assertion_kind": "fact",
                "condition_status": "unconditional",
            },
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="777",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="MISSING_REQUIRED_PREDICATE",
            expected_ask_reason_prefix="assertion_kind_not_admitted",
            claim_kind="requirement",
        )

    # 5. Excluded source attribution
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument(
            "memo.txt",
            "Orion duration is 23 days.",
            authority_rank=100,
            weight=1.0,
            attribution="unverified_assertion",
        )
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="23",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="MISSING_REQUIRED_PREDICATE",
            expected_ask_reason_prefix="source_attribution_not_admitted",
        )

    # 6. Expired temporal window
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "valid_until": "2025-12-31T23:59:59Z",
            },
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="23",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="MISSING_REQUIRED_PREDICATE",
            expected_ask_reason_prefix="outside_validity",
            as_of="2026-06-01T00:00:00Z",
        )

    # 7. Stale document revision
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc_v1 = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc_v1)
        _admit_single(
            layer,
            doc_v1,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        doc_v2 = SourceDocument("msa.txt", "Orion duration is 45 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc_v2)
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="23",
            claim_unit="calendar_days",
            expected_auth_status="RETRY",
            expected_auth_code="STALE_EVIDENCE_REVISION",
            expected_ask_reason_prefix="stale_evidence",
        )

    # 8. Unassessed structural_reliability=None
    with tempfile.TemporaryDirectory() as td:
        layer = _layer(td, vocab)
        doc = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            reliability=None,
        )
        _assert_all_four_gates_refuse(
            layer,
            predicate="duration_days",
            claim_value="23",
            claim_unit="calendar_days",
            expected_auth_status="ESCALATE",
            expected_auth_code="UNASSESSED_READER_RELIABILITY",
            expected_ask_reason_prefix=None,
        )


# ---------------------------------------------------------------------------
# Group 3: Persisted Authorization State in VerificationStore & Tamper Checks
# ---------------------------------------------------------------------------

def test_persisted_authorization_state_and_tamper_detection():
    vocab = _invariant_vocab()
    key = b"invariant-signing-key"
    payload = {"action": "set_duration", "days": 23}

    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        layer = _layer(td1, vocab)
        doc = SourceDocument("msa.txt", "Orion duration is 23 days.", authority_rank=100, weight=1.0)
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "scope": {"clause_id": "2.1"},
            },
        )

        verdict = authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("duration_days",),
            numeric_bounds={"duration_days": (None, 30, "calendar_days")},
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        )
        assert verdict.status == "PASS"
        assert verdict.snapshot_digest is not None

        # Persisted record in SQLite VerificationStore matches snapshot & signature
        stored = layer.store.get("authorization", verdict.snapshot_digest)
        assert isinstance(stored, dict)
        assert stored["snapshot_digest"] == verdict.snapshot_digest
        assert canonical(stored["snapshot"]) == canonical(verdict.decision_snapshot)
        assert stored["signature"] == verdict.signature
        head_id, _ = layer.store.head("authorization", "op-auth-01")
        assert head_id == verdict.snapshot_digest

        assert verify_authorization_verdict(
            verdict,
            layer=layer,
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        ) is True

        # A distinct EvidenceLayer with identical inputs that never ran authorize_operation MUST reject
        other_layer = _layer(td2, vocab)
        other_layer.ingest(doc)
        _admit_single(
            other_layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "duration_days",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "scope": {"clause_id": "2.1"},
            },
        )
        assert verify_authorization_verdict(
            verdict,
            layer=other_layer,
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        ) is False

        # Snapshot digest sensitivity across policy/context mutations
        v_other_bound = authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("duration_days",),
            numeric_bounds={"duration_days": (None, 45, "calendar_days")},
            operation_id="op-auth-02",
            payload=payload,
            signing_key=key,
        )
        v_other_rel = authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("duration_days",),
            numeric_bounds={"duration_days": (None, 30, "calendar_days")},
            clause_relations=(ClauseRelation("amends_clause", "9.9", "1.0"),),
            operation_id="op-auth-03",
            payload=payload,
            signing_key=key,
        )
        assert len({verdict.snapshot_digest, v_other_bound.snapshot_digest, v_other_rel.snapshot_digest}) == 3

        # Tampering with persisted 'authorization' row in SQLite MUST fail verification even with valid HMAC
        with layer.store.connect() as db:
            db.execute(
                "UPDATE semantic_records SET payload = '{}' WHERE kind = 'authorization' AND record_id = ?",
                (verdict.snapshot_digest,),
            )
        assert verify_authorization_verdict(
            verdict,
            layer=layer,
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        ) is False

        # Deleting the persisted 'authorization' row MUST fail verification
        with layer.store.connect() as db:
            db.execute(
                "DELETE FROM semantic_records WHERE kind = 'authorization' AND record_id = ?",
                (verdict.snapshot_digest,),
            )
        assert verify_authorization_verdict(
            verdict,
            layer=layer,
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        ) is False

        # Mutated decision_snapshot MUST fail local digest verification even without layer
        tampered_snap = dict(verdict.decision_snapshot)
        tampered_snap["operation_id"] = "op-forged"
        verdict_tampered = replace(verdict, decision_snapshot=tampered_snap)
        assert verify_authorization_verdict(
            verdict_tampered,
            operation_id="op-auth-01",
            payload=payload,
            signing_key=key,
        ) is False


# ---------------------------------------------------------------------------
# Group 5: Post-audit structural closure (stale auth, SQLite tx, grounding, unified core)
# ---------------------------------------------------------------------------


def test_stale_authorization_invalidated_on_conflicting_unit_and_atomic_execution_prevents_replay():
    key = "audit-hmac-secret"
    payload = {"action": "enforce_cure", "days": 23}
    doc = SourceDocument(
        "msa.txt",
        "Orion cure period is 23 days. Under clause 2.2, Orion cure period is 41 days.",
        authority_rank=TIER_PRIMARY,
        weight=1.0,
    )
    with tempfile.TemporaryDirectory() as td:
        layer = EvidenceLayer(
            Path(td) / "ledger.db",
            project_id="stale",
            vocabulary=B2B_CONTRACTS_VOCABULARY,
            policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )
        layer.ingest(doc)
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "scope": {"clause_id": "2.1"},
            },
            quote="Orion cure period is 23 days.",
        )
        v1 = authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("cure_period",),
            operation_id="op-cure-1",
            payload=payload,
            signing_key=key,
        )
        assert v1.status == "PASS"
        assert v1.allowed is True
        assert verify_authorization_verdict(
            v1, layer=layer, operation_id="op-cure-1", payload=payload, signing_key=key
        ) is True

        # Admitting a conflicting unit from the same document and revision MUST invalidate v1 immediately
        _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "41",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "scope": {"clause_id": "2.2"},
            },
            quote="Under clause 2.2, Orion cure period is 41 days.",
        )
        assert verify_authorization_verdict(
            v1, layer=layer, operation_id="op-cure-1", payload=payload, signing_key=key
        ) is False

        v2 = authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("cure_period",),
            operation_id="op-cure-1",
            payload=payload,
            signing_key=key,
        )
        assert v2.status == "ESCALATE"
        assert "UNRESOLVED_CONFLICT" in {i.code for i in v2.issues}
        assert verify_authorization_verdict(
            v1, layer=layer, operation_id="op-cure-1", payload=payload, signing_key=key
        ) is False

        called = []
        with pytest.raises(PermissionError):
            execute_authorized_operation(
                layer,
                v1,
                lambda v: called.append(v.snapshot_digest),
                operation_id="op-cure-1",
                payload=payload,
                signing_key=key,
            )
        assert called == []

        # Fresh layer with one-shot atomic execution prevents replay
        layer_exec = EvidenceLayer(
            Path(td) / "exec.db",
            project_id="exec",
            vocabulary=B2B_CONTRACTS_VOCABULARY,
            policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )
        doc_clean = SourceDocument("msa_clean.txt", "Orion cure period is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        layer_exec.ingest(doc_clean)
        _admit_single(
            layer_exec,
            doc_clean,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        v_ok = authorize_operation(
            layer_exec,
            entity="Orion",
            required_predicates=("cure_period",),
            operation_id="op-exec-1",
            payload=payload,
            signing_key=key,
        )
        assert v_ok.status == "PASS"
        out = execute_authorized_operation(
            layer_exec,
            v_ok,
            lambda v: {"status": "done", "val": v.resolved_values["cure_period"]},
            operation_id="op-exec-1",
            payload=payload,
            signing_key=key,
        )
        assert out == {"status": "done", "val": {"value": "23", "unit": "calendar_days"}}
        assert verify_authorization_verdict(
            v_ok, layer=layer_exec, operation_id="op-exec-1", payload=payload, signing_key=key
        ) is False
        assert verify_authorization_verdict(
            v_ok, layer=layer_exec, operation_id="op-exec-1", payload=payload, signing_key=key, allow_consumed=True
        ) is True
        with pytest.raises(PermissionError):
            execute_authorized_operation(
                layer_exec,
                v_ok,
                lambda v: "replay",
                operation_id="op-exec-1",
                payload=payload,
                signing_key=key,
            )


def test_verification_store_select_is_in_transaction_and_prevents_torn_snapshots_and_partial_ingest(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        layer = EvidenceLayer(
            Path(td) / "tx.db",
            project_id="tx",
            vocabulary=B2B_CONTRACTS_VOCABULARY,
            policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )
        layer.store.put("document", "rev-1", {"n": 1}, head="doc-1")

        # Read connection starts an explicit SQLite transaction so snapshots see a consistent point-in-time
        with layer.store.connect() as db1:
            assert db1.in_transaction is True
            rows_before = db1.execute(
                "SELECT name FROM semantic_heads WHERE project_id=? AND kind='document'",
                ("tx",),
            ).fetchall()
            assert rows_before == [("doc-1",)]

            # Independent VerificationStore commits a second row in WAL mode while db1's read tx is open
            store2 = type(layer.store)(Path(td) / "tx.db", "tx")
            store2.put("document", "rev-2", {"n": 2}, head="doc-2")

            rows_inside_tx = db1.execute(
                "SELECT name FROM semantic_heads WHERE project_id=? AND kind='document' ORDER BY name",
                ("tx",),
            ).fetchall()
            assert rows_inside_tx == [("doc-1",)]

        # Fresh connection observes committed doc-2
        with layer.store.connect() as db_after:
            rows_after = db_after.execute(
                "SELECT name FROM semantic_heads WHERE project_id=? AND kind='document' ORDER BY name",
                ("tx",),
            ).fetchall()
            assert rows_after == [("doc-1",), ("doc-2",)]

        # Mid-ingest batch failure rolls back all documents and units atomically
        layer_atom = EvidenceLayer(
            Path(td) / "atom.db",
            project_id="atom",
            vocabulary=B2B_CONTRACTS_VOCABULARY,
            policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )
        d1 = SourceDocument("d1.txt", "Orion cure period is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        d2 = SourceDocument("d2.txt", "Orion payment terms are 30 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        layer_atom.ingest([d1, d2])

        def failing_fe(*args, **kwargs):
            raise RuntimeError("injected storage fault on second item")

        monkeypatch.setattr(layer_atom, "_store_field_evidence", failing_fe)
        with pytest.raises(RuntimeError, match="injected storage fault"):
            layer_atom.ingest_units(
                [
                    {
                        "proposal": {
                            "entity": "Orion",
                            "predicate": "cure_period",
                            "value": "23",
                            "unit": "calendar_days",
                            "assertion_kind": "requirement",
                            "condition_status": "unconditional",
                        },
                        "evidence": [EvidenceRef(d1.document_id, d1.revision, 0, len(d1.text), d1.text)],
                    },
                    {
                        "proposal": {
                            "entity": "Orion",
                            "predicate": "payment_terms",
                            "value": "30",
                            "unit": "calendar_days",
                            "assertion_kind": "requirement",
                            "condition_status": "unconditional",
                        },
                        "evidence": [EvidenceRef(d2.document_id, d2.revision, 0, len(d2.text), d2.text)],
                        "field_evidence": [FieldEvidence("value", EvidenceRef(d2.document_id, d2.revision, 0, len(d2.text), d2.text))],
                    },
                ],
                attestor="atom_test",
            )
        assert len(layer_atom.units) == 0

        layer_batch = EvidenceLayer(
            Path(td) / "batch.db",
            project_id="batch",
            vocabulary=B2B_CONTRACTS_VOCABULARY,
            policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )
        orig_record = layer_batch._record_doc_read_status
        calls = {"n": 0}

        def fail_on_second_doc(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("mid-ingest document failure")
            return orig_record(*args, **kwargs)

        monkeypatch.setattr(layer_batch, "_record_doc_read_status", fail_on_second_doc)
        with pytest.raises(RuntimeError, match="mid-ingest document failure"):
            layer_batch.ingest([d1, d2])
        snap = layer_batch.snapshot()
        assert snap["documents"] == {}
        assert snap["units"] == []


def test_subject_predicate_and_scope_grounding_rejects_entity_and_predicate_swaps():
    custom_vocab = Vocabulary(
        name="alias-test",
        version="4.0",
        entities=(("Orion", ()), ("Vega", ())),
        predicates=(
            PredicateSpec(
                "cure_period",
                FieldSpec("quantity", dimensions=("duration_calendar_days",)),
                assertion_kinds=("requirement",),
                aliases=("notice window",),
            ),
            PredicateSpec(
                "reporting_period",
                FieldSpec("quantity", dimensions=("duration_calendar_days",)),
                assertion_kinds=("requirement",),
            ),
        ),
        scope_keys=("region", "clause_id", "event_id"),
        require_explicit_condition_status=True,
    )
    with tempfile.TemporaryDirectory() as td:
        # 1. Cross-entity swap in direct text -> ENTITY_NOT_IN_EVIDENCE
        l_ent = EvidenceLayer(
            Path(td) / "ent.db", project_id="ent", vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY
        )
        doc_vega = SourceDocument("vega.txt", "Vega cure period is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        l_ent.ingest(doc_vega)
        _admit_single(
            l_ent,
            doc_vega,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        v_ent = authorize_operation(l_ent, entity="Orion", required_predicates=("cure_period",))
        assert v_ent.status == "RETRY"
        assert "ENTITY_NOT_IN_EVIDENCE" in {i.code for i in v_ent.issues}
        ask_ent = l_ent.ask(StructuredQuery("Orion", "cure_period"))
        assert ask_ent["abstain"] is True
        assert ask_ent["excluded"][0]["reason"] == "fidelity:ENTITY_NOT_IN_EVIDENCE"

        # 2. Multi-sentence document header trap: "Orion MSA." header + quote belonging to "Vega" -> blocked!
        l_multi = EvidenceLayer(
            Path(td) / "multi.db", project_id="multi", vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY
        )
        doc_multi = SourceDocument(
            "multi.txt",
            "Orion Master Agreement. Vega cure period is 23 days.",
            authority_rank=TIER_PRIMARY,
            weight=1.0,
        )
        l_multi.ingest(doc_multi)
        _admit_single(
            l_multi,
            doc_multi,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            quote="Vega cure period is 23 days.",
        )
        v_multi = authorize_operation(l_multi, entity="Orion", required_predicates=("cure_period",))
        assert v_multi.status == "RETRY"
        assert "ENTITY_NOT_IN_EVIDENCE" in {i.code for i in v_multi.issues}

        # 3. Wrong predicate swap ("reporting period" vs "cure_period") -> PREDICATE_NOT_IN_EVIDENCE
        l_pred = EvidenceLayer(
            Path(td) / "pred.db", project_id="pred", vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY
        )
        doc_pred = SourceDocument("pred.txt", "Orion reporting period is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        l_pred.ingest(doc_pred)
        _admit_single(
            l_pred,
            doc_pred,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        v_pred = authorize_operation(l_pred, entity="Orion", required_predicates=("cure_period",))
        assert v_pred.status == "RETRY"
        assert "PREDICATE_NOT_IN_EVIDENCE" in {i.code for i in v_pred.issues}

        # 4. Ungrounded semantic scope ("region": "apj") -> SCOPE_NOT_IN_EVIDENCE
        l_scope = EvidenceLayer(
            Path(td) / "scope.db", project_id="scope", vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY
        )
        doc_scope = SourceDocument("scope.txt", "Orion cure period is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        l_scope.ingest(doc_scope)
        _admit_single(
            l_scope,
            doc_scope,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "scope": {"region": "apj"},
            },
        )
        v_scope = authorize_operation(
            l_scope, entity="Orion", required_predicates=("cure_period",), query_scope={"region": "apj"}
        )
        assert v_scope.status == "RETRY"
        assert "SCOPE_NOT_IN_EVIDENCE" in {i.code for i in v_scope.issues}

        # 5. PredicateSpec aliases allow alternate phrasing ("notice window") -> PASS
        l_alias = EvidenceLayer(
            Path(td) / "alias.db", project_id="alias", vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY
        )
        doc_alias = SourceDocument("alias.txt", "Orion notice window is 14 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        l_alias.ingest(doc_alias)
        _admit_single(
            l_alias,
            doc_alias,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "14",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        v_alias = authorize_operation(l_alias, entity="Orion", required_predicates=("cure_period",))
        assert v_alias.status == "PASS"


def test_single_admissibility_core_unifies_rule_derived_units_across_ask_and_authorize():
    vocab = Vocabulary(
        name="derived-unification",
        version="4.0",
        predicates=(
            PredicateSpec(
                "raw_cure",
                FieldSpec("quantity", dimensions=("duration_calendar_days",)),
                assertion_kinds=("requirement",),
            ),
            PredicateSpec(
                "cure_period",
                FieldSpec("quantity", dimensions=("duration_calendar_days",)),
                assertion_kinds=("requirement",),
            ),
        ),
        scope_keys=("clause_id", "event_id"),
        require_explicit_condition_status=True,
    )
    transfer_rule = (
        "1",
        lambda conclusion, premises: (
            len(premises) == 1
            and premises[0]["predicate"] == "raw_cure"
            and conclusion["predicate"] == "cure_period"
            and conclusion["entity"] == premises[0]["entity"]
            and conclusion["value"] == premises[0]["value"]
            and conclusion["unit"] == premises[0]["unit"]
        ),
    )
    with tempfile.TemporaryDirectory() as td:
        layer = EvidenceLayer(
            Path(td) / "derived.db",
            project_id="drv",
            vocabulary=vocab,
            policy=AcceptancePolicy(allowed_assertion_kinds=("requirement",), tau=0.80, delta=0.50, path_discount=0.95),
            rules={"copy_to_cure": transfer_rule},
        )
        doc = SourceDocument("msa.txt", "Orion raw cure deadline is 23 days.", authority_rank=TIER_PRIMARY, weight=1.0)
        layer.ingest(doc)
        premise_unit = _admit_single(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "raw_cure",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            reliability=0.90,
        )
        derived_id = layer.derive(
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            premises=[premise_unit.unit_id],
            rule="copy_to_cure",
        )
        derived_unit = layer.unit(derived_id)
        assert derived_unit.origin == "rule_derived"
        assert derived_unit.structural_reliability is None

        # 1. Both ask() and authorize_operation() admit the rule_derived unit with discounted weakest-link support 0.95 * 0.90 = 0.855
        ask_res = layer.ask(StructuredQuery("Orion", "cure_period"))
        assert ask_res["status"] == "answered"
        assert ask_res["results"][0]["support"]["S"] == 0.855

        verdict = authorize_operation(layer, entity="Orion", required_predicates=("cure_period",))
        assert verdict.status == "PASS"
        assert verdict.resolved_values["cure_period"] == {"value": "23", "unit": "calendar_days"}

        # 2. Staling the premise document revision causes BOTH ask() and authorize_operation() to reject the derived unit
        doc_v2 = SourceDocument(
            "msa.txt", "Orion raw cure deadline is 30 days.", authority_rank=TIER_PRIMARY, weight=1.0
        )
        layer.ingest(doc_v2)
        ask_stale = layer.ask(StructuredQuery("Orion", "cure_period"))
        assert ask_stale["abstain"] is True
        assert ask_stale["excluded"][0]["reason"] == "premise_stale_evidence"

        verdict_stale = authorize_operation(layer, entity="Orion", required_predicates=("cure_period",))
        assert verdict_stale.status == "RETRY"
        assert "STALE_EVIDENCE_REVISION" in {i.code for i in verdict_stale.issues}

        # 3. Derived unit over a premise with structural_reliability=None is unassessed in BOTH ask() and authorize_operation()
        l_unassessed = EvidenceLayer(
            Path(td) / "unassessed.db",
            project_id="unass",
            vocabulary=vocab,
            policy=AcceptancePolicy(allowed_assertion_kinds=("requirement",), tau=0.80, delta=0.50, path_discount=0.95),
            rules={"copy_to_cure": transfer_rule},
        )
        l_unassessed.ingest(doc)
        u_none = _admit_single(
            l_unassessed,
            doc,
            {
                "entity": "Orion",
                "predicate": "raw_cure",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            reliability=None,
        )
        l_unassessed.derive(
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
            premises=[u_none.unit_id],
            rule="copy_to_cure",
        )
        ask_none = l_unassessed.ask(StructuredQuery("Orion", "cure_period"))
        assert ask_none["abstain"] is True
        assert ask_none["excluded"][0]["reason"] == "premise_unassessed"

        v_none = authorize_operation(l_unassessed, entity="Orion", required_predicates=("cure_period",))
        assert v_none.status == "ESCALATE"
        assert "UNASSESSED_READER_RELIABILITY" in {i.code for i in v_none.issues}
