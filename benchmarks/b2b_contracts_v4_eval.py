"""Three-layer evaluation for the B2B commercial-contracts v4 milestone.

Evaluates separately:
  Layer 1: Reasoning over gold typed units (amendments, exceptions, scope, temporal validity, gaps).
  Layer 2: Extraction fidelity controls + dual-reader correlated-error diagnostics.
  Layer 3: End-to-end operational authorization with abstention (selective risk vs automation coverage).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimlayer.v4 import (
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    ClauseRelation,
    EvidenceLayer,
    EvidenceRef,
    FieldEvidence,
    SourceDocument,
    TIER_PRIMARY,
    authorize_operation,
    evaluate_dual_readers,
    resolve_scope_and_precedence,
    verify_extraction_fidelity,
)


def _ref(doc: SourceDocument, sub: str) -> EvidenceRef:
    idx = doc.text.index(sub)
    return EvidenceRef(doc.document_id, doc.revision, idx, idx + len(sub), sub)


def eval_layer1_reasoning() -> dict[str, int]:
    msa = SourceDocument(
        "msa",
        "C1: Payment terms are 45 days. C2: Enterprise EU payment terms are 60 days. "
        "C3: Payment terms are 15 days if invoice is disputed.",
        authority_rank=TIER_PRIMARY,
        weight=1.0,
    )
    amend = SourceDocument(
        "amend",
        "C1A: Payment terms amended to 30 days.",
        authority_rank=TIER_PRIMARY,
        weight=1.0,
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
                    "scope": {"clause_id": "c1"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "C1: Payment terms are 45 days.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 60,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "c2", "service_tier": "enterprise", "region": "eu"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "C2: Enterprise EU payment terms are 60 days.")],
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
                    "scope": {"clause_id": "c3"},
                    "valid_from": "2025-01-01",
                },
                "evidence": [_ref(msa, "C3: Payment terms are 15 days if invoice is disputed.")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                    "scope": {"clause_id": "c1a"},
                    "valid_from": "2026-01-01",
                },
                "evidence": [_ref(amend, "C1A: Payment terms amended to 30 days.")],
            },
        ],
        attestor="gold",
    )
    rels = [ClauseRelation("amends_clause", "c1a", "c1", valid_from="2026-01-01")]
    tiers = {"msa": TIER_PRIMARY, "amend": TIER_PRIMARY}

    checks = [
        resolve_scope_and_precedence(
            layer.units, entity="Acme", predicate="payment_terms", vocabulary=B2B_CONTRACTS_VOCABULARY,
            as_of="2025-06-01", query_conditions={"invoice_disputed": False}, clause_relations=rels,
            source_tiers=tiers, policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )["value"] == {"value": "45", "unit": "calendar_days"},
        resolve_scope_and_precedence(
            layer.units, entity="Acme", predicate="payment_terms", vocabulary=B2B_CONTRACTS_VOCABULARY,
            as_of="2026-06-01", query_conditions={"invoice_disputed": False}, clause_relations=rels,
            source_tiers=tiers, policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )["value"] == {"value": "30", "unit": "calendar_days"},
        resolve_scope_and_precedence(
            layer.units, entity="Acme", predicate="payment_terms", vocabulary=B2B_CONTRACTS_VOCABULARY,
            as_of="2026-06-01", query_conditions={"invoice_disputed": True}, clause_relations=rels,
            source_tiers=tiers, policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )["value"] == {"value": "15", "unit": "calendar_days"},
        resolve_scope_and_precedence(
            layer.units, entity="Acme", predicate="payment_terms", vocabulary=B2B_CONTRACTS_VOCABULARY,
            as_of="2026-06-01", query_scope={"service_tier": "enterprise", "region": "eu"},
            query_conditions={"invoice_disputed": False}, clause_relations=rels,
            source_tiers=tiers, policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )["value"] == {"value": "60", "unit": "calendar_days"},
        resolve_scope_and_precedence(
            layer.units, entity="Acme", predicate="payment_terms", vocabulary=B2B_CONTRACTS_VOCABULARY,
            as_of="2026-06-01", query_conditions={}, clause_relations=rels,
            source_tiers=tiers, policy=B2B_DEFAULT_ACCEPTANCE_POLICY,
        )["status"] == "PENDING_CONDITION",
    ]
    return {"passed": sum(checks), "total": len(checks)}


def eval_layer2_fidelity() -> dict[str, object]:
    doc = SourceDocument(
        "d1",
        "Payment terms are 30 days unless invoice is disputed. Vendor shall not be liable for indirect loss.",
        authority_rank=TIER_PRIMARY,
        weight=1.0,
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
                "evidence": [_ref(doc, "Payment terms are 30 days")],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_terms",
                    "value": 30,
                    "unit": "days",
                    "assertion_kind": "requirement",
                    "condition_status": "conditional",
                    "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": False}],
                },
                "evidence": [_ref(doc, "Payment terms are 30 days unless invoice is disputed.")],
            },
        ],
        attestor="extractor",
    )
    bad_findings = verify_extraction_fidelity(doc.text, layer.unit(res["unit_ids"][0]))
    good_findings = verify_extraction_fidelity(
        doc.text,
        layer.unit(res["unit_ids"][1]),
        field_evidence=[FieldEvidence("value", _ref(doc, "30 days"))],
        require_field_evidence=("value",),
    )
    dual = evaluate_dual_readers(
        [
            {"gold": "30d_if_undisputed", "reader_a": "30d_if_undisputed", "reader_b": "30d_if_undisputed"},
            {"gold": "30d_if_undisputed", "reader_a": "30d_unconditional", "reader_b": "30d_unconditional"},
            {"gold": "500k_usd", "reader_a": "500k_usd", "reader_b": "500k_usd"},
            {"gold": "15d_if_disputed", "reader_a": "15d_if_disputed", "reader_b": "30d_unconditional"},
        ]
    )
    return {
        "bad_unit_caught": bool(bad_findings),
        "good_unit_clean": not good_findings,
        "dual_reader_metrics": dual,
    }


def eval_layer3_end_to_end() -> dict[str, object]:
    """Measure selective risk (errors among accepted PASS) and automation coverage."""
    doc = SourceDocument(
        "msa",
        "Payment terms are 30 days. Liability cap is 200000 USD.",
        authority_rank=TIER_PRIMARY,
        weight=1.0,
    )
    scenarios = [
        # (expected_safe_to_pass, ingest_err, condition_status, cap_bound)
        (True, None, "unconditional", 500000.0),
        (True, None, "unconditional", 300000.0),
        (False, {"read_errors": [{"document_id": "annex"}]}, "unconditional", 500000.0),
        (False, None, "unknown_condition", 500000.0),
        (False, None, "unrepresentable_condition", 500000.0),
        (False, None, "unconditional", 100000.0),  # exceeds cap -> must ESCALATE
    ]
    accepted = 0
    accepted_errors = 0
    abstained = 0
    for safe_to_pass, ingest_err, cstatus, cap_max in scenarios:
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
                        "condition_status": cstatus,
                    },
                    "evidence": [_ref(doc, "Payment terms are 30 days.")],
                },
                {
                    "proposal": {
                        "entity": "Acme",
                        "predicate": "liability_cap",
                        "value": 200000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(doc, "Liability cap is 200000 USD.")],
                },
            ],
            attestor="eval",
        )
        verdict = authorize_operation(
            layer,
            entity="Acme",
            required_predicates=("payment_terms", "liability_cap"),
            ingest_summary=ingest_err,
            numeric_bounds={"liability_cap": (None, cap_max, "USD")},
        )
        if verdict.allowed:
            accepted += 1
            if not safe_to_pass:
                accepted_errors += 1
        else:
            abstained += 1

    total = len(scenarios)
    return {
        "total_scenarios": total,
        "accepted_pass": accepted,
        "abstained_retry_or_escalate": abstained,
        "automation_coverage": round(accepted / total, 4),
        "selective_risk_on_accepted": round(accepted_errors / accepted, 4) if accepted else 0.0,
    }


def main() -> int:
    report = {
        "layer1_reasoning": eval_layer1_reasoning(),
        "layer2_fidelity": eval_layer2_fidelity(),
        "layer3_end_to_end_abstention": eval_layer3_end_to_end(),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
