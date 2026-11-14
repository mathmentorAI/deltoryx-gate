"""Three-layer empirical benchmark for ClaimLayer v4 (B2B Commercial Contracts).

Separates evaluation into:
  Layer 1 (gold_reasoning): deterministic engine + scope/precedence + authorization
      over gold typed EvidenceUnits (no LLM noise).
  Layer 2 (extraction_fidelity): dual-reader extraction (LLMProposalReader with
      STRICT_READER_PROMPT) checked via verify_extraction_fidelity and
      evaluate_dual_readers (disagreement rate, shared error rate).
  Layer 3 (risk_coverage_curve): end-to-end selective authorization calibrated on
      DEV_CASES over (tau, delta) and evaluated on blind TEST_CASES, reporting
      selective_risk vs automation_coverage.

Supports:
  --dry-run   (default) deterministic scripted dual-reader doubles for CI
  --live      real OpenAICompatibleModel dual-reader API evaluation
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
import sys
from typing import Any, Mapping, Sequence

from claimlayer.semantic.models import canonical
from claimlayer.v4 import (
    AcceptancePolicy,
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    ClauseRelation,
    EvidenceLayer,
    EvidenceRef,
    FieldEvidence,
    LLMProposalReader,
    OpenAICompatibleModel,
    STRICT_READER_PROMPT,
    SourceDocument,
    TIER_ANNEX,
    TIER_PRIMARY,
    authorize_operation,
    evaluate_dual_readers,
    verify_extraction_fidelity,
)
from claimlayer.v4.vocabulary import label_key


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    split: str  # "dev" | "test"
    entity: str
    docs: tuple[SourceDocument, ...]
    required_predicates: tuple[str, ...]
    query_scope: Mapping[str, Any]
    query_conditions: Mapping[str, Any]
    clause_relations: tuple[ClauseRelation, ...]
    numeric_bounds: Mapping[str, tuple[float | None, float | None]]
    gold_units: tuple[dict[str, Any], ...]
    expected_allowed: bool
    expected_status: str  # "PASS" | "RETRY" | "ESCALATE"
    reader_a_output: dict[str, list[dict[str, Any]]]
    reader_b_output: dict[str, list[dict[str, Any]]]


def _ref(doc: SourceDocument, substring: str) -> EvidenceRef:
    start = doc.text.index(substring)
    return EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=start + len(substring),
        quote=substring,
    )


def _fe_from_unit_dict(doc: SourceDocument, entity_q: str, value_q: str) -> list[FieldEvidence]:
    return [
        FieldEvidence("entity", _ref(doc, entity_q)),
        FieldEvidence("value", _ref(doc, value_q)),
    ]


def _build_dev_cases() -> list[EvalCase]:
    # Dev 1: Clean unconditional PASS
    d1 = SourceDocument(
        "dev1_msa",
        "ApexCorp standard payment terms are 30 days. ApexCorp aggregate liability cap is 500000 USD.",
        authority_rank=TIER_PRIMARY,
    )
    # Dev 2: Conditional exception — dispute active -> cannot authorize unconditional payment_terms
    d2 = SourceDocument(
        "dev2_msa",
        "BetaInc payment terms are 30 days, unless the invoice is disputed in good faith.",
        authority_rank=TIER_PRIMARY,
    )
    # Dev 3: Amendment overrides MSA clause + scope
    d3_msa = SourceDocument(
        "dev3_msa",
        "GammaLLC enterprise payment terms are 45 days under Clause 4.1.",
        authority_rank=TIER_PRIMARY,
    )
    d3_amend = SourceDocument(
        "dev3_amend",
        "Amendment 1 replaces Clause 4.1: GammaLLC enterprise payment terms are 30 days.",
        authority_rank=TIER_PRIMARY,
    )
    # Dev 4: Policy bound exceeded -> ESCALATE
    d4 = SourceDocument(
        "dev4_msa",
        "DeltaSystems liability cap is 900000 USD. DeltaSystems payment terms are 30 days.",
        authority_rank=TIER_PRIMARY,
    )
    # Dev 5: Unrepresentable condition -> must block (RETRY)
    d5 = SourceDocument(
        "dev5_msa",
        "EpsilonCo SLA uptime target is 99.9 percent, subject to mutually agreed maintenance windows.",
        authority_rank=TIER_PRIMARY,
    )
    # Dev 6: Contradiction across equal-tier annexes -> ESCALATE
    d6_a = SourceDocument("dev6_annex_a", "ZetaLtd payment terms are 30 days.", authority_rank=TIER_ANNEX)
    d6_b = SourceDocument("dev6_annex_b", "ZetaLtd payment terms are 60 days.", authority_rank=TIER_ANNEX)

    return [
        EvalCase(
            case_id="dev_01_clean_pass",
            split="dev",
            entity="ApexCorp",
            docs=(d1,),
            required_predicates=("payment_terms", "liability_cap"),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={"liability_cap": (None, 600000.0)},
            gold_units=(
                {
                    "proposal": {
                        "entity": "ApexCorp",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d1, "ApexCorp standard payment terms are 30 days.")],
                    "fe": _fe_from_unit_dict(d1, "ApexCorp", "30 days"),
                },
                {
                    "proposal": {
                        "entity": "ApexCorp",
                        "predicate": "liability_cap",
                        "value": 500000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d1, "ApexCorp aggregate liability cap is 500000 USD.")],
                    "fe": _fe_from_unit_dict(d1, "ApexCorp", "500000 USD"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            reader_a_output={
                "dev1_msa": [
                    {
                        "entity": "ApexCorp",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ApexCorp standard payment terms are 30 days.",
                        "field_evidence": {"entity": "ApexCorp", "value": "30 days"},
                    },
                    {
                        "entity": "ApexCorp",
                        "predicate": "liability_cap",
                        "value": 500000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ApexCorp aggregate liability cap is 500000 USD.",
                        "field_evidence": {"entity": "ApexCorp", "value": "500000 USD"},
                    },
                ]
            },
            reader_b_output={
                "dev1_msa": [
                    {
                        "entity": "ApexCorp",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ApexCorp standard payment terms are 30 days.",
                        "field_evidence": {"entity": "ApexCorp", "value": "30 days"},
                    },
                    {
                        "entity": "ApexCorp",
                        "predicate": "liability_cap",
                        "value": 500000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ApexCorp aggregate liability cap is 500000 USD.",
                        "field_evidence": {"entity": "ApexCorp", "value": "500000 USD"},
                    },
                ]
            },
        ),
        EvalCase(
            case_id="dev_02_conditional_proviso",
            split="dev",
            entity="BetaInc",
            docs=(d2,),
            required_predicates=("payment_terms",),
            query_scope={},
            query_conditions={"invoice_disputed": False},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "BetaInc",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "conditional",
                        "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": False}],
                    },
                    "evidence": [_ref(d2, "BetaInc payment terms are 30 days, unless the invoice is disputed in good faith.")],
                    "fe": _fe_from_unit_dict(d2, "BetaInc", "30 days"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            # Reader A drops proviso (caught by fidelity check!), Reader B captures it
            reader_a_output={
                "dev2_msa": [
                    {
                        "entity": "BetaInc",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "BetaInc payment terms are 30 days, unless the invoice is disputed in good faith.",
                        "field_evidence": {"entity": "BetaInc", "value": "30 days"},
                    }
                ]
            },
            reader_b_output={
                "dev2_msa": [
                    {
                        "entity": "BetaInc",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "conditional",
                        "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": False}],
                        "quote": "BetaInc payment terms are 30 days, unless the invoice is disputed in good faith.",
                        "field_evidence": {"entity": "BetaInc", "value": "30 days"},
                    }
                ]
            },
        ),
        EvalCase(
            case_id="dev_03_amendment_override",
            split="dev",
            entity="GammaLLC",
            docs=(d3_msa, d3_amend),
            required_predicates=("payment_terms",),
            query_scope={"service_tier": "enterprise"},
            query_conditions={},
            clause_relations=(
                ClauseRelation("amends_clause", "amend_1", "4.1"),
            ),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "4.1"},
                    },
                    "evidence": [_ref(d3_msa, "GammaLLC enterprise payment terms are 45 days under Clause 4.1.")],
                    "fe": _fe_from_unit_dict(d3_msa, "GammaLLC", "45 days"),
                },
                {
                    "proposal": {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "amend_1"},
                    },
                    "evidence": [_ref(d3_amend, "Amendment 1 replaces Clause 4.1: GammaLLC enterprise payment terms are 30 days.")],
                    "fe": _fe_from_unit_dict(d3_amend, "GammaLLC", "30 days"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            reader_a_output={
                "dev3_msa": [
                    {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "4.1"},
                        "quote": "GammaLLC enterprise payment terms are 45 days under Clause 4.1.",
                        "field_evidence": {"entity": "GammaLLC", "value": "45 days"},
                    }
                ],
                "dev3_amend": [
                    {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "amend_1"},
                        "quote": "Amendment 1 replaces Clause 4.1: GammaLLC enterprise payment terms are 30 days.",
                        "field_evidence": {"entity": "GammaLLC", "value": "30 days"},
                    }
                ],
            },
            reader_b_output={
                "dev3_msa": [
                    {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "4.1"},
                        "quote": "GammaLLC enterprise payment terms are 45 days under Clause 4.1.",
                        "field_evidence": {"entity": "GammaLLC", "value": "45 days"},
                    }
                ],
                "dev3_amend": [
                    {
                        "entity": "GammaLLC",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise", "clause_id": "amend_1"},
                        "quote": "Amendment 1 replaces Clause 4.1: GammaLLC enterprise payment terms are 30 days.",
                        "field_evidence": {"entity": "GammaLLC", "value": "30 days"},
                    }
                ],
            },
        ),
        EvalCase(
            case_id="dev_04_policy_bound_escalate",
            split="dev",
            entity="DeltaSystems",
            docs=(d4,),
            required_predicates=("payment_terms", "liability_cap"),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={"liability_cap": (None, 500000.0)},
            gold_units=(
                {
                    "proposal": {
                        "entity": "DeltaSystems",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d4, "DeltaSystems payment terms are 30 days.")],
                    "fe": _fe_from_unit_dict(d4, "DeltaSystems", "30 days"),
                },
                {
                    "proposal": {
                        "entity": "DeltaSystems",
                        "predicate": "liability_cap",
                        "value": 900000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d4, "DeltaSystems liability cap is 900000 USD.")],
                    "fe": _fe_from_unit_dict(d4, "DeltaSystems", "900000 USD"),
                },
            ),
            expected_allowed=False,
            expected_status="ESCALATE",
            reader_a_output={
                "dev4_msa": [
                    {
                        "entity": "DeltaSystems",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "DeltaSystems payment terms are 30 days.",
                        "field_evidence": {"entity": "DeltaSystems", "value": "30 days"},
                    },
                    {
                        "entity": "DeltaSystems",
                        "predicate": "liability_cap",
                        "value": 900000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "DeltaSystems liability cap is 900000 USD.",
                        "field_evidence": {"entity": "DeltaSystems", "value": "900000 USD"},
                    },
                ]
            },
            reader_b_output={
                "dev4_msa": [
                    {
                        "entity": "DeltaSystems",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "DeltaSystems payment terms are 30 days.",
                        "field_evidence": {"entity": "DeltaSystems", "value": "30 days"},
                    },
                    {
                        "entity": "DeltaSystems",
                        "predicate": "liability_cap",
                        "value": 900000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "DeltaSystems liability cap is 900000 USD.",
                        "field_evidence": {"entity": "DeltaSystems", "value": "900000 USD"},
                    },
                ]
            },
        ),
        EvalCase(
            case_id="dev_05_unrepresentable_condition",
            split="dev",
            entity="EpsilonCo",
            docs=(d5,),
            required_predicates=("sla_uptime_target",),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "EpsilonCo",
                        "predicate": "sla_uptime_target",
                        "value": 99.9,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unrepresentable_condition",
                    },
                    "evidence": [_ref(d5, "EpsilonCo SLA uptime target is 99.9 percent, subject to mutually agreed maintenance windows.")],
                    "fe": _fe_from_unit_dict(d5, "EpsilonCo", "99.9 percent"),
                },
            ),
            expected_allowed=False,
            expected_status="ESCALATE",
            reader_a_output={
                "dev5_msa": [
                    {
                        "entity": "EpsilonCo",
                        "predicate": "sla_uptime_target",
                        "value": 99.9,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unrepresentable_condition",
                        "quote": "EpsilonCo SLA uptime target is 99.9 percent, subject to mutually agreed maintenance windows.",
                        "field_evidence": {"entity": "EpsilonCo", "value": "99.9 percent"},
                    }
                ]
            },
            reader_b_output={
                "dev5_msa": [
                    {
                        "entity": "EpsilonCo",
                        "predicate": "sla_uptime_target",
                        "value": 99.9,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unrepresentable_condition",
                        "quote": "EpsilonCo SLA uptime target is 99.9 percent, subject to mutually agreed maintenance windows.",
                        "field_evidence": {"entity": "EpsilonCo", "value": "99.9 percent"},
                    }
                ]
            },
        ),
        EvalCase(
            case_id="dev_06_contradiction_escalate",
            split="dev",
            entity="ZetaLtd",
            docs=(d6_a, d6_b),
            required_predicates=("payment_terms",),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d6_a, "ZetaLtd payment terms are 30 days.")],
                    "fe": _fe_from_unit_dict(d6_a, "ZetaLtd", "30 days"),
                },
                {
                    "proposal": {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(d6_b, "ZetaLtd payment terms are 60 days.")],
                    "fe": _fe_from_unit_dict(d6_b, "ZetaLtd", "60 days"),
                },
            ),
            expected_allowed=False,
            expected_status="ESCALATE",
            reader_a_output={
                "dev6_annex_a": [
                    {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ZetaLtd payment terms are 30 days.",
                        "field_evidence": {"entity": "ZetaLtd", "value": "30 days"},
                    }
                ],
                "dev6_annex_b": [
                    {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ZetaLtd payment terms are 60 days.",
                        "field_evidence": {"entity": "ZetaLtd", "value": "60 days"},
                    }
                ],
            },
            reader_b_output={
                "dev6_annex_a": [
                    {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ZetaLtd payment terms are 30 days.",
                        "field_evidence": {"entity": "ZetaLtd", "value": "30 days"},
                    }
                ],
                "dev6_annex_b": [
                    {
                        "entity": "ZetaLtd",
                        "predicate": "payment_terms",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ZetaLtd payment terms are 60 days.",
                        "field_evidence": {"entity": "ZetaLtd", "value": "60 days"},
                    }
                ],
            },
        ),
    ]


def _build_test_cases() -> list[EvalCase]:
    # Blind Test 1: Clean PASS (EUR currency, 45 days)
    t1 = SourceDocument(
        "test1_msa",
        "NordicTech payment terms are 45 days. NordicTech liability cap is 250000 EUR.",
        authority_rank=TIER_PRIMARY,
    )
    # Blind Test 2: Conditional exception — both readers share an error (drop condition)!
    # Fidelity check must catch UNCAPTURED_CONDITION and block auto-authorization.
    t2 = SourceDocument(
        "test2_msa",
        "OrionCloud payment terms are 30 days, provided that the invoice is not disputed.",
        authority_rank=TIER_PRIMARY,
    )
    # Blind Test 3: Scope-specific exception (enterprise vs standard)
    t3 = SourceDocument(
        "test3_msa",
        "PulseData standard SLA uptime target is 99.5 percent. PulseData enterprise SLA uptime target is 99.99 percent.",
        authority_rank=TIER_PRIMARY,
    )
    # Blind Test 4: Unknown condition -> must block (RETRY)
    t4 = SourceDocument(
        "test4_msa",
        "QuantumLogistics notice period is 60 days, unless otherwise specified in Schedule B.",
        authority_rank=TIER_PRIMARY,
    )
    # Blind Test 5: Policy bound violation -> ESCALATE
    t5 = SourceDocument(
        "test5_msa",
        "SolsticeAI payment terms are 30 days. SolsticeAI liability cap is 1200000 USD.",
        authority_rank=TIER_PRIMARY,
    )
    # Blind Test 6: Clean PASS with governing_law + auto_renewal
    t6 = SourceDocument(
        "test6_msa",
        "VanguardBio governing law is Delaware. VanguardBio auto renewal is true.",
        authority_rank=TIER_PRIMARY,
    )

    return [
        EvalCase(
            case_id="test_01_clean_eur_pass",
            split="test",
            entity="NordicTech",
            docs=(t1,),
            required_predicates=("payment_terms", "liability_cap"),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={"liability_cap": (None, 500000.0)},
            gold_units=(
                {
                    "proposal": {
                        "entity": "NordicTech",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t1, "NordicTech payment terms are 45 days.")],
                    "fe": _fe_from_unit_dict(t1, "NordicTech", "45 days"),
                },
                {
                    "proposal": {
                        "entity": "NordicTech",
                        "predicate": "liability_cap",
                        "value": 250000,
                        "unit": "EUR",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t1, "NordicTech liability cap is 250000 EUR.")],
                    "fe": _fe_from_unit_dict(t1, "NordicTech", "250000 EUR"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            reader_a_output={
                "test1_msa": [
                    {
                        "entity": "NordicTech",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "NordicTech payment terms are 45 days.",
                        "field_evidence": {"entity": "NordicTech", "value": "45 days"},
                    },
                    {
                        "entity": "NordicTech",
                        "predicate": "liability_cap",
                        "value": 250000,
                        "unit": "EUR",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "NordicTech liability cap is 250000 EUR.",
                        "field_evidence": {"entity": "NordicTech", "value": "250000 EUR"},
                    },
                ]
            },
            reader_b_output={
                "test1_msa": [
                    {
                        "entity": "NordicTech",
                        "predicate": "payment_terms",
                        "value": 45,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "NordicTech payment terms are 45 days.",
                        "field_evidence": {"entity": "NordicTech", "value": "45 days"},
                    },
                    {
                        "entity": "NordicTech",
                        "predicate": "liability_cap",
                        "value": 250000,
                        "unit": "EUR",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "NordicTech liability cap is 250000 EUR.",
                        "field_evidence": {"entity": "NordicTech", "value": "250000 EUR"},
                    },
                ]
            },
        ),
        EvalCase(
            case_id="test_02_correlated_proviso_drop",
            split="test",
            entity="OrionCloud",
            docs=(t2,),
            required_predicates=("payment_terms",),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "OrionCloud",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "conditional",
                        "conditions": [{"field": "invoice_disputed", "operator": "eq", "value": False}],
                    },
                    "evidence": [_ref(t2, "OrionCloud payment terms are 30 days, provided that the invoice is not disputed.")],
                    "fe": _fe_from_unit_dict(t2, "OrionCloud", "30 days"),
                },
            ),
            # Query has no conditions -> conditional unit cannot satisfy unconditional query -> RETRY
            expected_allowed=False,
            expected_status="RETRY",
            # BOTH readers make the correlated mistake of marking it unconditional!
            reader_a_output={
                "test2_msa": [
                    {
                        "entity": "OrionCloud",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "OrionCloud payment terms are 30 days, provided that the invoice is not disputed.",
                        "field_evidence": {"entity": "OrionCloud", "value": "30 days"},
                    }
                ]
            },
            reader_b_output={
                "test2_msa": [
                    {
                        "entity": "OrionCloud",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "OrionCloud payment terms are 30 days, provided that the invoice is not disputed.",
                        "field_evidence": {"entity": "OrionCloud", "value": "30 days"},
                    }
                ]
            },
        ),
        EvalCase(
            case_id="test_03_scoped_sla_pass",
            split="test",
            entity="PulseData",
            docs=(t3,),
            required_predicates=("sla_uptime_target",),
            query_scope={"service_tier": "enterprise"},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.5,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "standard"},
                    },
                    "evidence": [_ref(t3, "PulseData standard SLA uptime target is 99.5 percent.")],
                    "fe": _fe_from_unit_dict(t3, "PulseData", "99.5 percent"),
                },
                {
                    "proposal": {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.99,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise"},
                    },
                    "evidence": [_ref(t3, "PulseData enterprise SLA uptime target is 99.99 percent.")],
                    "fe": _fe_from_unit_dict(t3, "PulseData", "99.99 percent"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            reader_a_output={
                "test3_msa": [
                    {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.5,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "standard"},
                        "quote": "PulseData standard SLA uptime target is 99.5 percent.",
                        "field_evidence": {"entity": "PulseData", "value": "99.5 percent"},
                    },
                    {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.99,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise"},
                        "quote": "PulseData enterprise SLA uptime target is 99.99 percent.",
                        "field_evidence": {"entity": "PulseData", "value": "99.99 percent"},
                    },
                ]
            },
            reader_b_output={
                "test3_msa": [
                    {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.5,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "standard"},
                        "quote": "PulseData standard SLA uptime target is 99.5 percent.",
                        "field_evidence": {"entity": "PulseData", "value": "99.5 percent"},
                    },
                    {
                        "entity": "PulseData",
                        "predicate": "sla_uptime_target",
                        "value": 99.99,
                        "unit": "percent",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "scope": {"service_tier": "enterprise"},
                        "quote": "PulseData enterprise SLA uptime target is 99.99 percent.",
                        "field_evidence": {"entity": "PulseData", "value": "99.99 percent"},
                    },
                ]
            },
        ),
        EvalCase(
            case_id="test_04_unknown_condition_retry",
            split="test",
            entity="QuantumLogistics",
            docs=(t4,),
            required_predicates=("notice_period",),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "QuantumLogistics",
                        "predicate": "notice_period",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unknown_condition",
                    },
                    "evidence": [_ref(t4, "QuantumLogistics notice period is 60 days, unless otherwise specified in Schedule B.")],
                    "fe": _fe_from_unit_dict(t4, "QuantumLogistics", "60 days"),
                },
            ),
            expected_allowed=False,
            expected_status="RETRY",
            reader_a_output={
                "test4_msa": [
                    {
                        "entity": "QuantumLogistics",
                        "predicate": "notice_period",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unknown_condition",
                        "quote": "QuantumLogistics notice period is 60 days, unless otherwise specified in Schedule B.",
                        "field_evidence": {"entity": "QuantumLogistics", "value": "60 days"},
                    }
                ]
            },
            reader_b_output={
                "test4_msa": [
                    {
                        "entity": "QuantumLogistics",
                        "predicate": "notice_period",
                        "value": 60,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unknown_condition",
                        "quote": "QuantumLogistics notice period is 60 days, unless otherwise specified in Schedule B.",
                        "field_evidence": {"entity": "QuantumLogistics", "value": "60 days"},
                    }
                ]
            },
        ),
        EvalCase(
            case_id="test_05_policy_bound_escalate",
            split="test",
            entity="SolsticeAI",
            docs=(t5,),
            required_predicates=("payment_terms", "liability_cap"),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={"liability_cap": (None, 500000.0)},
            gold_units=(
                {
                    "proposal": {
                        "entity": "SolsticeAI",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t5, "SolsticeAI payment terms are 30 days.")],
                    "fe": _fe_from_unit_dict(t5, "SolsticeAI", "30 days"),
                },
                {
                    "proposal": {
                        "entity": "SolsticeAI",
                        "predicate": "liability_cap",
                        "value": 1200000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t5, "SolsticeAI liability cap is 1200000 USD.")],
                    "fe": _fe_from_unit_dict(t5, "SolsticeAI", "1200000 USD"),
                },
            ),
            expected_allowed=False,
            expected_status="ESCALATE",
            reader_a_output={
                "test5_msa": [
                    {
                        "entity": "SolsticeAI",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "SolsticeAI payment terms are 30 days.",
                        "field_evidence": {"entity": "SolsticeAI", "value": "30 days"},
                    },
                    {
                        "entity": "SolsticeAI",
                        "predicate": "liability_cap",
                        "value": 1200000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "SolsticeAI liability cap is 1200000 USD.",
                        "field_evidence": {"entity": "SolsticeAI", "value": "1200000 USD"},
                    },
                ]
            },
            reader_b_output={
                "test5_msa": [
                    {
                        "entity": "SolsticeAI",
                        "predicate": "payment_terms",
                        "value": 30,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "SolsticeAI payment terms are 30 days.",
                        "field_evidence": {"entity": "SolsticeAI", "value": "30 days"},
                    },
                    {
                        "entity": "SolsticeAI",
                        "predicate": "liability_cap",
                        "value": 1200000,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "SolsticeAI liability cap is 1200000 USD.",
                        "field_evidence": {"entity": "SolsticeAI", "value": "1200000 USD"},
                    },
                ]
            },
        ),
        EvalCase(
            case_id="test_06_governing_law_and_renewal_pass",
            split="test",
            entity="VanguardBio",
            docs=(t6,),
            required_predicates=("governing_law", "auto_renewal"),
            query_scope={},
            query_conditions={},
            clause_relations=(),
            numeric_bounds={},
            gold_units=(
                {
                    "proposal": {
                        "entity": "VanguardBio",
                        "predicate": "governing_law",
                        "value": "Delaware",
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t6, "VanguardBio governing law is Delaware.")],
                    "fe": _fe_from_unit_dict(t6, "VanguardBio", "Delaware"),
                },
                {
                    "proposal": {
                        "entity": "VanguardBio",
                        "predicate": "auto_renewal",
                        "value": True,
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "evidence": [_ref(t6, "VanguardBio auto renewal is true.")],
                    "fe": _fe_from_unit_dict(t6, "VanguardBio", "true"),
                },
            ),
            expected_allowed=True,
            expected_status="PASS",
            reader_a_output={
                "test6_msa": [
                    {
                        "entity": "VanguardBio",
                        "predicate": "governing_law",
                        "value": "Delaware",
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                        "quote": "VanguardBio governing law is Delaware.",
                        "field_evidence": {"entity": "VanguardBio", "value": "Delaware"},
                    },
                    {
                        "entity": "VanguardBio",
                        "predicate": "auto_renewal",
                        "value": True,
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "VanguardBio auto renewal is true.",
                        "field_evidence": {"entity": "VanguardBio", "value": "true"},
                    },
                ]
            },
            reader_b_output={
                "test6_msa": [
                    {
                        "entity": "VanguardBio",
                        "predicate": "governing_law",
                        "value": "Delaware",
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                        "quote": "VanguardBio governing law is Delaware.",
                        "field_evidence": {"entity": "VanguardBio", "value": "Delaware"},
                    },
                    {
                        "entity": "VanguardBio",
                        "predicate": "auto_renewal",
                        "value": True,
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "VanguardBio auto renewal is true.",
                        "field_evidence": {"entity": "VanguardBio", "value": "true"},
                    },
                ]
            },
        ),
    ]


class _ScriptedModel:
    """Deterministic ModelAdapter double for --dry-run CI evaluation."""

    def __init__(self, model_name: str, mapping: Mapping[str, list[dict[str, Any]]]) -> None:
        self._model_name = model_name
        self._mapping = mapping

    @property
    def identity(self) -> dict[str, Any]:
        return {"provider": "scripted_dry_run", "model": self._model_name}

    def complete(self, system: str, user_payload: Mapping[str, Any]) -> dict[str, Any]:
        doc_id = user_payload["document_id"]
        first_ev = user_payload["fragments"][0]["evidence_id"]
        raw_items = self._mapping.get(doc_id, [])
        enriched = []
        for item in raw_items:
            p = dict(item)
            p.setdefault("basis", "explicit")
            p.setdefault("evidence", [first_ev])
            enriched.append(p)
        return {"output": {"proposals": enriched}}


class _CapturingReader:
    """Wraps a ProposalReader and caches per-(document_id, revision) propose() outputs."""

    def __init__(self, inner: LLMProposalReader) -> None:
        self._inner = inner
        self.cache: dict[tuple[str, str], Mapping[str, Any]] = {}

    @property
    def identity(self) -> Mapping[str, Any]:
        return self._inner.identity

    @property
    def structural_reliability(self) -> float | None:
        return self._inner.structural_reliability

    def propose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        doc = request["document"]
        key = (doc["document_id"], doc["revision"])
        if key not in self.cache:
            self.cache[key] = self._inner.propose(request)
        return self.cache[key]


class _ReplayReader:
    """Replays captured propose() outputs with an empirically calibrated structural_reliability."""

    def __init__(
        self,
        base_identity: Mapping[str, Any],
        cache: Mapping[tuple[str, str], Mapping[str, Any]],
        structural_reliability: float | None,
    ) -> None:
        self._identity = dict(base_identity)
        self._cache = cache
        self.structural_reliability = structural_reliability

    @property
    def identity(self) -> Mapping[str, Any]:
        return {**self._identity, "structural_reliability": self.structural_reliability}

    def propose(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        doc = request["document"]
        return self._cache[(doc["document_id"], doc["revision"])]


def _canonical_unit_record(case_id: str, u: Any) -> dict[str, Any]:
    p = u.proposition
    return {
        "case_id": case_id,
        "document_id": u.evidence[0].document_id if u.evidence else "",
        "entity": label_key(p.get("entity", "")),
        "predicate": p.get("predicate"),
        "scope": p.get("scope", {}),
        "value": p.get("value"),
        "unit": p.get("unit"),
        "polarity": p.get("polarity"),
        "condition_status": p.get("condition_status"),
        "conditions": p.get("conditions", []),
    }


def _index_records_by_semantic_key(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str, int], dict[str, Any]]:
    counts: dict[tuple[str, str, str, str, str], int] = {}
    indexed: dict[tuple[str, str, str, str, str, int], dict[str, Any]] = {}
    ordered = sorted(records, key=lambda r: canonical(dict(r)))
    for rec in ordered:
        base_key = (
            str(rec["case_id"]),
            str(rec["entity"]),
            str(rec["predicate"]),
            canonical(rec.get("scope", {})),
            str(rec.get("document_id", "")),
        )
        occ = counts.get(base_key, 0)
        counts[base_key] = occ + 1
        indexed[(*base_key, occ)] = dict(rec)
    return indexed


def _eval_layer1_gold_reasoning(cases: Sequence[EvalCase], policy: AcceptancePolicy) -> dict[str, Any]:
    correct = 0
    details = []
    gold_resolved_by_case: dict[str, dict[str, Any]] = {}
    for case in cases:
        layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=policy)
        for doc in case.docs:
            layer.ingest(doc)
        res = layer.ingest_units(
            [
                {
                    "proposal": u["proposal"],
                    "evidence": u["evidence"],
                    "field_evidence": u["fe"],
                }
                for u in case.gold_units
            ],
            attestor="gold_oracle",
        )
        fe_map = {uid: case.gold_units[i]["fe"] for i, uid in enumerate(res["unit_ids"])}
        verdict = authorize_operation(
            layer,
            entity=case.entity,
            required_predicates=case.required_predicates,
            query_scope=case.query_scope,
            query_conditions=case.query_conditions,
            clause_relations=case.clause_relations,
            field_evidences=fe_map,
            require_field_evidence=("entity", "value"),
            numeric_bounds=case.numeric_bounds,
        )
        ok = (verdict.allowed == case.expected_allowed) and (verdict.status == case.expected_status)
        if ok:
            correct += 1
        gold_resolved_by_case[case.case_id] = dict(verdict.resolved_values)
        details.append(
            {
                "case_id": case.case_id,
                "expected": case.expected_status,
                "actual": verdict.status,
                "allowed": verdict.allowed,
                "correct": ok,
            }
        )
    return {
        "total": len(cases),
        "correct": correct,
        "accuracy": round(correct / max(1, len(cases)), 4),
        "details": details,
        "gold_resolved_by_case": gold_resolved_by_case,
    }


def _extract_with_reader(
    reader: Any,
    cases: Sequence[EvalCase],
    policy: AcceptancePolicy,
    gold_resolved_by_case: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_extracted: list[dict[str, Any]] = []
    per_case_verdicts: list[dict[str, Any]] = []

    for case in cases:
        layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=policy, reader=reader)
        combined_read_errors: list[dict[str, Any]] = []
        total_rejected = 0
        for doc in case.docs:
            doc_summary = layer.ingest(doc)
            combined_read_errors.extend(doc_summary.get("read_errors", []))
            total_rejected += int(doc_summary.get("proposals_rejected", 0))
        summary = {"read_errors": combined_read_errors, "proposals_rejected": total_rejected}

        fidelity_findings = []
        for u in layer.units:
            doc = next((d for d in case.docs if d.document_id == u.evidence[0].document_id), None)
            if doc is not None:
                fidelity_findings.extend(verify_extraction_fidelity(doc.text, u))
            all_extracted.append(_canonical_unit_record(case.case_id, u))

        verdict = authorize_operation(
            layer,
            entity=case.entity,
            required_predicates=case.required_predicates,
            query_scope=case.query_scope,
            query_conditions=case.query_conditions,
            clause_relations=case.clause_relations,
            ingest_summary=summary,
            numeric_bounds=case.numeric_bounds,
        )
        expected_vals = (gold_resolved_by_case or {}).get(case.case_id)
        values_match = (
            True
            if (not case.expected_allowed or expected_vals is None)
            else canonical(verdict.resolved_values) == canonical(dict(expected_vals))
        )
        per_case_verdicts.append(
            {
                "case_id": case.case_id,
                "expected_allowed": case.expected_allowed,
                "expected_status": case.expected_status,
                "actual_allowed": verdict.allowed,
                "actual_status": verdict.status,
                "values_match": values_match,
                "fidelity_finding_codes": [f.code for f in fidelity_findings],
            }
        )
    return all_extracted, per_case_verdicts


def _build_gold_flat(cases: Sequence[EvalCase]) -> list[dict[str, Any]]:
    gold: list[dict[str, Any]] = []
    for c in cases:
        layer = EvidenceLayer(vocabulary=B2B_CONTRACTS_VOCABULARY, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
        for doc in c.docs:
            layer.ingest(doc)
        layer.ingest_units(
            [{"proposal": u["proposal"], "evidence": u["evidence"]} for u in c.gold_units],
            attestor="gold_oracle",
        )
        for u in layer.units:
            gold.append(_canonical_unit_record(c.case_id, u))
    return gold


def _align_dual_reader_cases(
    gold_records: Sequence[Mapping[str, Any]],
    ext_a_records: Sequence[Mapping[str, Any]],
    ext_b_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    g_map = _index_records_by_semantic_key(gold_records)
    a_map = _index_records_by_semantic_key(ext_a_records)
    b_map = _index_records_by_semantic_key(ext_b_records)
    all_keys = sorted(set(g_map) | set(a_map) | set(b_map))
    return [
        {
            "gold": g_map.get(k),
            "reader_a": a_map.get(k),
            "reader_b": b_map.get(k),
        }
        for k in all_keys
    ]


def _risk_coverage_at(verdicts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(verdicts)
    accepted = [v for v in verdicts if v["actual_allowed"]]
    abstained = [v for v in verdicts if not v["actual_allowed"]]
    false_pass = [
        v
        for v in accepted
        if (not v["expected_allowed"]) or (not v.get("values_match", True))
    ]
    coverage = len(accepted) / max(1, total)
    selective_risk = (len(false_pass) / len(accepted)) if accepted else 0.0
    return {
        "total": total,
        "accepted": len(accepted),
        "abstained": len(abstained),
        "false_pass": len(false_pass),
        "automation_coverage": round(coverage, 4),
        "selective_risk": round(selective_risk, 4),
    }


def run_benchmark(
    *,
    live: bool = False,
    endpoint_a: str | None = None,
    model_a: str | None = None,
    api_key_a: str | None = None,
    endpoint_b: str | None = None,
    model_b: str | None = None,
    api_key_b: str | None = None,
) -> dict[str, Any]:
    dev_cases = _build_dev_cases()
    test_cases = _build_test_cases()

    if live:
        if not (endpoint_a and model_a and api_key_a and endpoint_b and model_b and api_key_b):
            raise ValueError("Live mode requires endpoint, model, and api_key for both Reader A and Reader B.")
        adapter_a: Any = OpenAICompatibleModel(endpoint=endpoint_a, model=model_a, api_key=api_key_a)
        adapter_b: Any = OpenAICompatibleModel(endpoint=endpoint_b, model=model_b, api_key=api_key_b)
        mode = "live"
    else:
        combined_a: dict[str, list[dict[str, Any]]] = {}
        combined_b: dict[str, list[dict[str, Any]]] = {}
        for c in (*dev_cases, *test_cases):
            combined_a.update(c.reader_a_output)
            combined_b.update(c.reader_b_output)
        adapter_a = _ScriptedModel("dry-run-reader-a", combined_a)
        adapter_b = _ScriptedModel("dry-run-reader-b", combined_b)
        mode = "dry-run"

    raw_reader_a = LLMProposalReader(adapter_a, prompt=STRICT_READER_PROMPT)
    raw_reader_b = LLMProposalReader(adapter_b, prompt=STRICT_READER_PROMPT)
    cap_a = _CapturingReader(raw_reader_a)
    cap_b = _CapturingReader(raw_reader_b)

    # Layer 1: Gold deterministic reasoning
    l1_dev_full = _eval_layer1_gold_reasoning(dev_cases, B2B_DEFAULT_ACCEPTANCE_POLICY)
    l1_test_full = _eval_layer1_gold_reasoning(test_cases, B2B_DEFAULT_ACCEPTANCE_POLICY)
    gold_dev_resolved = l1_dev_full.pop("gold_resolved_by_case")
    gold_test_resolved = l1_test_full.pop("gold_resolved_by_case")

    # Layer 2: Extract ONCE per reader per split; align by semantic key
    ext_a_dev, _ = _extract_with_reader(cap_a, dev_cases, B2B_DEFAULT_ACCEPTANCE_POLICY, gold_dev_resolved)
    ext_b_dev, _ = _extract_with_reader(cap_b, dev_cases, B2B_DEFAULT_ACCEPTANCE_POLICY, gold_dev_resolved)
    gold_dev = _build_gold_flat(dev_cases)
    aligned_dev = _align_dual_reader_cases(gold_dev, ext_a_dev, ext_b_dev)
    dual_dev = evaluate_dual_readers(aligned_dev)

    # Empirical structural_reliability measured from Reader A exact match rate on DEV
    dev_exact_hits = sum(1 for row in aligned_dev if row["gold"] is not None and row["reader_a"] == row["gold"])
    measured_dev_reliability = round(dev_exact_hits / max(1, len(aligned_dev)), 4)

    ext_a_test, _ = _extract_with_reader(cap_a, test_cases, B2B_DEFAULT_ACCEPTANCE_POLICY, gold_test_resolved)
    ext_b_test, _ = _extract_with_reader(cap_b, test_cases, B2B_DEFAULT_ACCEPTANCE_POLICY, gold_test_resolved)
    gold_test = _build_gold_flat(test_cases)
    aligned_test = _align_dual_reader_cases(gold_test, ext_a_test, ext_b_test)
    dual_test = evaluate_dual_readers(aligned_test)

    # Layer 3: Replay captured extractions (zero extra reader calls) over DEV (tau, delta) grid
    calibrated_reader_a = _ReplayReader(
        raw_reader_a.identity,
        cap_a.cache,
        structural_reliability=measured_dev_reliability,
    )
    grid = [(0.50, 0.30), (0.70, 0.20), (0.85, 0.10)]
    dev_curve = []
    for tau, delta in grid:
        pol = AcceptancePolicy(
            tau=tau,
            delta=delta,
            allowed_assertion_kinds=B2B_DEFAULT_ACCEPTANCE_POLICY.allowed_assertion_kinds,
            version=f"sweep-{tau}-{delta}",
        )
        _, v_dev = _extract_with_reader(calibrated_reader_a, dev_cases, pol, gold_dev_resolved)
        metrics = _risk_coverage_at(v_dev)
        dev_curve.append({"tau": tau, "delta": delta, **metrics})

    # Pick lowest selective_risk, tie-break by highest automation_coverage
    best = min(dev_curve, key=lambda r: (r["selective_risk"], -r["automation_coverage"]))
    calibrated_pol = AcceptancePolicy(
        tau=best["tau"],
        delta=best["delta"],
        allowed_assertion_kinds=B2B_DEFAULT_ACCEPTANCE_POLICY.allowed_assertion_kinds,
        version="calibrated",
    )
    _, calibrated_test_verdicts = _extract_with_reader(
        calibrated_reader_a,
        test_cases,
        calibrated_pol,
        gold_test_resolved,
    )
    test_at_calibrated = _risk_coverage_at(calibrated_test_verdicts)

    return {
        "mode": mode,
        "reader_a_dev_measured_reliability": measured_dev_reliability,
        "layer1_gold_reasoning": {"dev": l1_dev_full, "test": l1_test_full},
        "layer2_dual_reader_extraction": {
            "dev": dual_dev,
            "test": dual_test,
        },
        "layer3_risk_coverage": {
            "dev_sweep": dev_curve,
            "calibrated_policy": {"tau": best["tau"], "delta": best["delta"]},
            "blind_test_metrics": test_at_calibrated,
            "blind_test_cases": calibrated_test_verdicts,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ClaimLayer v4 3-Layer B2B Live/Dry-Run Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Run with scripted dual-reader doubles (default if no live keys)")
    parser.add_argument("--live", action="store_true", help="Run against live OpenAI-compatible endpoints")
    parser.add_argument("--endpoint-a", default=os.environ.get("CLAIMLAYER_ENDPOINT_A", ""))
    parser.add_argument("--model-a", default=os.environ.get("CLAIMLAYER_MODEL_A", ""))
    parser.add_argument("--api-key-a", default=os.environ.get("CLAIMLAYER_API_KEY_A", ""))
    parser.add_argument("--endpoint-b", default=os.environ.get("CLAIMLAYER_ENDPOINT_B", ""))
    parser.add_argument("--model-b", default=os.environ.get("CLAIMLAYER_MODEL_B", ""))
    parser.add_argument("--api-key-b", default=os.environ.get("CLAIMLAYER_API_KEY_B", ""))
    args = parser.parse_args(argv)

    use_live = bool(args.live and not args.dry_run)
    report = run_benchmark(
        live=use_live,
        endpoint_a=args.endpoint_a or None,
        model_a=args.model_a or None,
        api_key_a=args.api_key_a or None,
        endpoint_b=args.endpoint_b or None,
        model_b=args.model_b or None,
        api_key_b=args.api_key_b or None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
