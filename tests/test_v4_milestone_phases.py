"""Tests for ClaimLayer v4 milestone Phases 1, 3, and 4."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest

from claimlayer import Assertion, Ledger, Policy, Source, verify_receipt
from claimlayer.v4 import (
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    ClauseRelation,
    EvidenceLayer,
    EvidenceRef,
    FieldEvidence,
    LLMProposalReader,
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_VOCABULARY,
    STRICT_READER_PROMPT,
    SourceDocument,
    TIER_PRIMARY,
    authorize_operation,
    render_canonical_claim,
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


class _StrictReaderDouble:
    def __init__(self, proposals_fn):
        self._proposals_fn = proposals_fn
        self.seen_system_prompts: list[str] = []

    @property
    def identity(self):
        return {"provider": "double", "model": "strict-reader-test", "endpoint": "local", "parameters": {}}

    def complete(self, system: str, payload):
        self.seen_system_prompts.append(system)
        return {"output": {"proposals": self._proposals_fn(payload)}}


# ---------------------------------------------------------------------------
# Phase 1: STRICT_READER_PROMPT, field_evidence resolution, fidelity & round-trip
# ---------------------------------------------------------------------------


def test_phase1_strict_reader_prompt_and_field_evidence_extraction():
    doc = _doc(
        "msa_p1",
        "Acme standard payment terms are 30 days, unless the invoice is disputed.",
    )

    def make_props(payload):
        ev_id = payload["fragments"][0]["evidence_id"]
        return [
            {
                "entity": "Acme",
                "predicate": "payment_terms",
                "value": 30,
                "unit": "days",
                "assertion_kind": "requirement",
                "condition_status": "conditional",
                "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": False}],
                "basis": "explicit",
                "evidence": [ev_id],
                "quote": "Acme standard payment terms are 30 days, unless the invoice is disputed.",
                "field_evidence": {
                    "entity": "Acme",
                    "value": "30 days",
                    "conditions": "unless the invoice is disputed",
                },
            }
        ]

    model = _StrictReaderDouble(make_props)
    reader = LLMProposalReader(model, prompt=STRICT_READER_PROMPT, structural_reliability=0.95)
    layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY, reader=reader)
    summary = layer.ingest(doc)
    assert model.seen_system_prompts[0] == STRICT_READER_PROMPT
    assert summary["units_admitted"] == 1
    unit = layer.units[0]
    assert unit.proposition["condition_status"] == "conditional"
    assert unit.proposition["conditions"] == [{"field": "invoice_disputed", "operator": "eq", "value": False}]
    # Verify direct propose() also returns resolved field_evidence spans
    direct = reader.propose(
        {
            "document": {"document_id": doc.document_id, "revision": doc.revision, "text": doc.text},
            "revision": doc.revision,
            "vocabulary": {"name": B2B_CONTRACTS_VOCABULARY.name},
            "profile": {"admissible_bases": ["explicit"]},
        }
    )
    assert len(direct["items"]) == 1
    item = direct["items"][0]
    assert set(item["field_evidence"].keys()) == {"entity", "value", "conditions"}
    assert item["field_evidence"]["entity"]["quote"] == "Acme"
    assert item["field_evidence"]["value"]["quote"] == "30 days"


def test_phase1_direction_of_change_entity_grounding_and_canonical_render():
    doc = _doc(
        "msa_dir",
        "Acme fee was reduced by 10 percent. Beta fee increased by 5 percent.",
    )
    layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer.ingest(doc)
    res = layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "sla_uptime_target",
                    "value": 10,  # positive value contradicts "reduced by"
                    "unit": "percent",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "evidence": [_ref(doc, "Acme fee was reduced by 10 percent.")],
            },
            {
                "proposal": {
                    "entity": "Beta",
                    "predicate": "sla_uptime_target",
                    "value": -5,  # negative value contradicts "increased by"
                    "unit": "percent",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "evidence": [_ref(doc, "Beta fee increased by 5 percent.")],
            },
        ],
        attestor="ops",
    )

    u0 = next(u for u in layer.units if u.proposition["entity"] == "Acme")
    u1 = next(u for u in layer.units if u.proposition["entity"] == "Beta")
    findings_0 = verify_extraction_fidelity(
        doc.text,
        u0,
        field_evidence=[FieldEvidence("entity", _ref(doc, "Beta"))],  # wrong entity span!
    )
    codes_0 = {f.code for f in findings_0}
    assert "DIRECTION_OF_CHANGE_MISMATCH" in codes_0
    assert "ENTITY_NOT_IN_FIELD_EVIDENCE" in codes_0

    findings_1 = verify_extraction_fidelity(doc.text, u1)
    assert any(f.code == "DIRECTION_OF_CHANGE_MISMATCH" for f in findings_1)

    rendered = render_canonical_claim(u0)
    assert "Acme" in rendered
    assert "sla_uptime_target" in rendered
    assert "unconditional" in rendered


# ---------------------------------------------------------------------------
# Phase 3: Trust Boundary, Immutability & Receipt Verification
# ---------------------------------------------------------------------------


def test_phase3_v4_trust_boundary_blocks_tampering_and_read_gaps():
    doc = _doc("msa_tb", "Acme payment terms are 30 days.")
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
                "evidence": [_ref(doc, "Acme payment terms are 30 days.")],
            }
        ],
        attestor="agent_untrusted",
    )
    uid = res["unit_ids"][0]

    # Tampered FieldEvidence (pointing to wrong offset/text) MUST block authorization
    bad_fe = {
        uid: [
            FieldEvidence(
                "entity",
                EvidenceRef(doc.document_id, doc.revision, 0, 4, "FAKE"),
            ),
            FieldEvidence("value", _ref(doc, "30 days")),
        ]
    }
    verdict_bad_fe = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("payment_terms",),
        field_evidences=bad_fe,
        require_field_evidence=("entity", "value"),
    )
    assert verdict_bad_fe.allowed is False
    assert verdict_bad_fe.status == "RETRY"

    # Unread documents in ingest_summary MUST block authorization
    good_fe = {
        uid: [
            FieldEvidence("entity", _ref(doc, "Acme")),
            FieldEvidence("value", _ref(doc, "30 days")),
        ]
    }
    verdict_unread = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("payment_terms",),
        ingest_summary={"unread_documents": ["annex_missing.pdf"]},
        field_evidences=good_fe,
        require_field_evidence=("entity", "value"),
    )
    assert verdict_unread.allowed is False
    assert verdict_unread.status == "RETRY"

    # Verdict object is frozen (agent cannot mutate verdict.allowed = True)
    with pytest.raises(FrozenInstanceError):
        verdict_unread.allowed = True  # type: ignore[misc]


def test_phase3_receipt_verification_rejects_unkeyed_and_tampered_verdicts():
    OPERATOR_KEY = "operator-secret-hmac-key"
    WRONG_KEY = "attacker-key"

    src = Source(id="doc1", text="Acme payment terms are 30 days.", tier="authoritative")
    pol = Policy(policy_id="strict_pol")
    ledger = Ledger([src], signing_key=OPERATOR_KEY)
    assertions = [
        Assertion(
            subject="Acme",
            predicate="payment_terms",
            value="30 days",
            source_id="doc1",
            quote="Acme payment terms are 30 days.",
        )
    ]
    verdict = ledger.verify(assertions, pol)
    assert verify_receipt(verdict, ledger=ledger, policy=pol, signing_key=OPERATOR_KEY, allow_unkeyed=False) is True
    assert verify_receipt(verdict, signing_key=WRONG_KEY, allow_unkeyed=False) is False

    # Unkeyed verdict MUST be rejected when allow_unkeyed=False
    unkeyed_ledger = Ledger([src])
    unkeyed_verdict = unkeyed_ledger.verify(assertions, pol)
    assert verify_receipt(unkeyed_verdict, signing_key=OPERATOR_KEY, allow_unkeyed=False) is False

    # Tampering with serialized verdict payload MUST fail verification
    tampered = verdict.to_dict()
    tampered["confidence"] = 0.9999
    assert verify_receipt(tampered, signing_key=OPERATOR_KEY, allow_unkeyed=False) is False


# ---------------------------------------------------------------------------
# Phase 4: Second Bounded Domain — Securitization
# ---------------------------------------------------------------------------


def test_phase4_securitization_domain_scope_amendment_and_conditional_call():
    psa = _doc(
        "psa_2025",
        (
            "Trust2025 Class A coupon rate is 5.25 percent. "
            "Trust2025 Class C coupon rate is 8.5 percent. "
            "Trust2025 servicer cure period is 30 days under Clause 2.1. "
            "Trust2025 clean up call threshold is 10 percent when pool factor is below threshold."
        ),
    )
    supp = _doc(
        "supp_2026",
        "Supplemental Indenture Clause 2.1A amends Clause 2.1: Trust2025 servicer cure period is 15 days.",
    )

    layer = EvidenceLayer(
        vocabulary=SECURITIZATION_VOCABULARY,
        policy=SECURITIZATION_ACCEPTANCE_POLICY,
    )
    layer.ingest(psa)
    layer.ingest(supp)

    res = layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Trust2025",
                    "predicate": "coupon_rate",
                    "value": 5.25,
                    "unit": "percent",
                    "assertion_kind": "definition",
                    "condition_status": "unconditional",
                    "scope": {"tranche": "Class A"},
                },
                "evidence": [_ref(psa, "Trust2025 Class A coupon rate is 5.25 percent.")],
            },
            {
                "proposal": {
                    "entity": "Trust2025",
                    "predicate": "coupon_rate",
                    "value": 8.5,
                    "unit": "percent",
                    "assertion_kind": "definition",
                    "condition_status": "unconditional",
                    "scope": {"tranche": "Class C"},
                },
                "evidence": [_ref(psa, "Trust2025 Class C coupon rate is 8.5 percent.")],
            },
            {
                "proposal": {
                    "entity": "Trust2025",
                    "predicate": "cure_period",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "2.1"},
                },
                "evidence": [_ref(psa, "Trust2025 servicer cure period is 30 days under Clause 2.1.")],
            },
            {
                "proposal": {
                    "entity": "Trust2025",
                    "predicate": "cure_period",
                    "value": 15,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "2.1A"},
                },
                "evidence": [
                    _ref(supp, "Supplemental Indenture Clause 2.1A amends Clause 2.1: Trust2025 servicer cure period is 15 days.")
                ],
            },
            {
                "proposal": {
                    "entity": "Trust2025",
                    "predicate": "clean_up_call_threshold",
                    "value": 10,
                    "unit": "percent",
                    "assertion_kind": "requirement",
                    "condition_status": "conditional",
                    "conditions": [{"field": "pool_factor_below_threshold", "operator": "eq", "value": True}],
                },
                "evidence": [_ref(psa, "Trust2025 clean up call threshold is 10 percent when pool factor is below threshold.")],
            },
        ],
        attestor="trust_ops",
    )
    uids = res["unit_ids"]
    fe = {
        uids[0]: [FieldEvidence("entity", _ref(psa, "Trust2025")), FieldEvidence("value", _ref(psa, "5.25 percent"))],
        uids[1]: [FieldEvidence("entity", _ref(psa, "Trust2025")), FieldEvidence("value", _ref(psa, "8.5 percent"))],
        uids[2]: [FieldEvidence("entity", _ref(psa, "Trust2025")), FieldEvidence("value", _ref(psa, "30 days"))],
        uids[3]: [FieldEvidence("entity", _ref(supp, "Trust2025")), FieldEvidence("value", _ref(supp, "15 days"))],
        uids[4]: [FieldEvidence("entity", _ref(psa, "Trust2025")), FieldEvidence("value", _ref(psa, "10 percent"))],
    }
    rels = [ClauseRelation("amends_clause", "2.1A", "2.1")]

    # Without pool_factor_below_threshold in query_conditions -> blocked (ESCALATE: PENDING_CONDITION)
    blocked = authorize_operation(
        layer,
        entity="Trust2025",
        required_predicates=("coupon_rate", "cure_period", "clean_up_call_threshold"),
        query_scope={"tranche": "Class A"},
        query_conditions={},
        clause_relations=rels,
        field_evidences=fe,
        require_field_evidence=("entity", "value"),
    )
    assert blocked.allowed is False
    assert blocked.status == "ESCALATE"

    # With pool_factor_below_threshold=True and tranche="Class A" -> PASS with amended cure_period=15 days
    passed = authorize_operation(
        layer,
        entity="Trust2025",
        required_predicates=("coupon_rate", "cure_period", "clean_up_call_threshold"),
        query_scope={"tranche": "Class A"},
        query_conditions={"pool_factor_below_threshold": True},
        clause_relations=rels,
        field_evidences=fe,
        require_field_evidence=("entity", "value"),
    )
    assert passed.allowed is True
    assert passed.status == "PASS"
    assert passed.resolved_values["cure_period"]["value"] == "15"
    assert passed.resolved_values["coupon_rate"]["value"] == "5.25"
    assert passed.resolved_values["coupon_rate"]["unit"] == "percent"
