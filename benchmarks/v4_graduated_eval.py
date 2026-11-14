"""Three-Layer Evaluation & Four-Arm Comparison Benchmark for ClaimLayer v4 Graduated Engine (Phase 5).

Evaluates:
- Layer 1: Isolated verifiers across all 9 families (arithmetic, quantities, temporal/age intervals,
  comparison, aggregation, explicit_logic, relations, document_authority, open_semantics).
- Layer 2: Extraction & relational association fidelity (subject swap, competing predicate,
  truncated proviso, unit dimension mismatch, polarity error, table alignment).
- Layer 3: Full responses comparing 4 arms:
    1) llm_unverified          (accepts proposed conclusion without verification)
    2) literal_extraction_only (accepts only verbatim literal extractions without derivation verifiers)
    3) graduated_v4_evaluate   (evaluate_response across justification DAG & 9 families)
    4) graduated_v4_authorize  (evaluate_response + authorize operational gate)
  measuring false_support_rate, false_rejection_rate, mean_coverage, abstention_rate,
  latency_ms, and incremental_cache_reuse_ratio.

Independent synthetic benchmark — never copies or tunes to external regression fixtures.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    ClaimNode,
    DeclaredRule,
    DomainProfile,
    EvidenceLayer,
    FieldSpec,
    InferenceStep,
    JustificationGraph,
    PredicateSpec,
    RelationCompositionRule,
    STANDARD_UNITS,
    SemanticEvaluatorConfig,
    Unit,
    VerificationBudget,
    VerificationCache,
    Vocabulary,
    authorize,
    evaluate_response,
)


def _doc(doc_id: str, text: str, *, weight: float = 0.95, dep_group: str | None = None) -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        authority_rank=85,
        weight=weight,
        dependence_group=dep_group,
    )


def _ref(doc: SourceDocument, substring: str) -> EvidenceRef:
    start = doc.text.index(substring)
    return EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=start + len(substring),
        quote=substring,
    )


def _shared_profile() -> DomainProfile:
    units = STANDARD_UNITS.extend([
        Unit("minutes", "duration_hours", "0.0166666666666666666666666667", ("min", "minute")),
    ])
    rule = DeclaredRule(
        rule_id="senior_discount_v1",
        version="1.0",
        kind="implication",
        required_premises=({"predicate": "age_eligible", "value": True, "polarity": 1},),
        conclusion_template={"predicate": "discount_eligible", "value": True},
    )
    rel_rule = RelationCompositionRule(
        rule_id="corp_control_v1",
        left_predicate="controls_entity",
        right_predicate="owns_asset",
        composed_predicate="indirectly_owns",
    )
    vocab = Vocabulary(
        name="graduated-eval-vocab",
        version="1.0",
        units=units,
        predicates=(
            PredicateSpec("cap", FieldSpec("quantity", ("currency:USD",))),
            PredicateSpec("fee", FieldSpec("quantity", ("currency:USD",))),
            PredicateSpec("cure_period", FieldSpec("quantity", ("duration_calendar_days",))),
            PredicateSpec("notice_period", FieldSpec("quantity", ("duration_calendar_days",))),
        ),
        entities=(("Acme", ("Acme Corp",)), ("Beta", ("Beta Inc",))),
    )
    return DomainProfile(
        profile_id="graduated-benchmark-v4",
        version="1.0.0",
        vocabulary=vocab,
        unit_registry=units,
        declared_rules={"senior_discount_v1": rule},
        relation_rules={"corp_control_v1": rel_rule},
        semantic_evaluator=SemanticEvaluatorConfig(
            evaluator_id="benchmark-mock-sem",
            version="0.1",
            enabled=True,
            callable_evaluator=lambda **kw: {"status": "verified", "score": 0.91},
        ),
    )


# ---------------------------------------------------------------------------
# Layer 1: Isolated Verifiers (9 Families)
# ---------------------------------------------------------------------------

def run_layer1_isolated_verifiers(profile: DomainProfile) -> dict[str, Any]:
    d = _doc("l1", "Base is 200 USD. Rate is 15 percent. Born 1980-03-10. Born year 1980. Window 2 hours. Fee1 40 USD. Fee2 60 USD. CoA controls SubB. SubB owns PlantC.")
    cases: list[tuple[str, JustificationGraph, str]] = [
        # 1. arithmetic valid / invalid / div-zero
        ("arith_pct_valid", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", value=200, unit="USD", evidence=(_ref(d, "Base is 200 USD."),), structural_reliability=1.0),
                "r": ClaimNode(claim_id="r", kind="extraction", value=15, unit="percent", evidence=(_ref(d, "Rate is 15 percent."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=30, unit="USD"),
            },
            steps=(InferenceStep(step_id="s1", premises=("b", "r"), conclusion_id="c", operation="percentage_of"),),
            conclusion_id="c",
        ), "supported"),
        ("arith_pct_wrong", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", value=200, unit="USD", evidence=(_ref(d, "Base is 200 USD."),), structural_reliability=1.0),
                "r": ClaimNode(claim_id="r", kind="extraction", value=15, unit="percent", evidence=(_ref(d, "Rate is 15 percent."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=45, unit="USD"),
            },
            steps=(InferenceStep(step_id="s1", premises=("b", "r"), conclusion_id="c", operation="percentage_of"),),
            conclusion_id="c",
        ), "refuted"),
        # 2. quantities valid / mismatch
        ("qty_conv_valid", JustificationGraph(
            claims={
                "h": ClaimNode(claim_id="h", kind="extraction", value=2, unit="hours", evidence=(_ref(d, "Window 2 hours."),), structural_reliability=1.0),
                "m": ClaimNode(claim_id="m", kind="derivation", value=120, unit="minutes"),
            },
            steps=(InferenceStep(step_id="s1", premises=("h",), conclusion_id="m", operation="unit_conversion", parameters={"tolerance": "0.001"}),),
            conclusion_id="m",
        ), "supported"),
        ("qty_dim_mismatch", JustificationGraph(
            claims={
                "h": ClaimNode(claim_id="h", kind="extraction", value=2, unit="hours", evidence=(_ref(d, "Window 2 hours."),), structural_reliability=1.0),
                "m": ClaimNode(claim_id="m", kind="derivation", value=2, unit="USD"),
            },
            steps=(InferenceStep(step_id="s1", premises=("h",), conclusion_id="m", operation="unit_conversion"),),
            conclusion_id="m",
        ), "refuted"),
        # 3. temporal exact age / interval age / exact-from-year inconclusive / impossible age
        ("temp_exact_age", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1980-03-10", evidence=(_ref(d, "Born 1980-03-10."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value=45, unit="years"),
            },
            steps=(InferenceStep(step_id="s1", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), "supported"),
        ("temp_coarse_interval_ok", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1980", evidence=(_ref(d, "Born year 1980."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value={"min": 44, "max": 45}, unit="years"),
            },
            steps=(InferenceStep(step_id="s1", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), "supported"),
        ("temp_coarse_categorical_unknown", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1980", evidence=(_ref(d, "Born year 1980."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value=45, unit="years"),
            },
            steps=(InferenceStep(step_id="s1", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), "inconclusive"),
        ("temp_impossible_age", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1980", evidence=(_ref(d, "Born year 1980."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value=70, unit="years"),
            },
            steps=(InferenceStep(step_id="s1", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), "refuted"),
        # 4. comparison valid / violated
        ("cmp_lt_valid", JustificationGraph(
            claims={
                "f1": ClaimNode(claim_id="f1", kind="extraction", value=40, unit="USD", evidence=(_ref(d, "Fee1 40 USD."),), structural_reliability=1.0),
                "f2": ClaimNode(claim_id="f2", kind="extraction", value=60, unit="USD", evidence=(_ref(d, "Fee2 60 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=True),
            },
            steps=(InferenceStep(step_id="s1", premises=("f1", "f2"), conclusion_id="c", operation="lt"),),
            conclusion_id="c",
        ), "supported"),
        ("cmp_gt_false", JustificationGraph(
            claims={
                "f1": ClaimNode(claim_id="f1", kind="extraction", value=40, unit="USD", evidence=(_ref(d, "Fee1 40 USD."),), structural_reliability=1.0),
                "f2": ClaimNode(claim_id="f2", kind="extraction", value=60, unit="USD", evidence=(_ref(d, "Fee2 60 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=True),
            },
            steps=(InferenceStep(step_id="s1", premises=("f1", "f2"), conclusion_id="c", operation="gt"),),
            conclusion_id="c",
        ), "refuted"),
        # 5. aggregation valid / missing population
        ("agg_sum_valid", JustificationGraph(
            claims={
                "f1": ClaimNode(claim_id="f1", kind="extraction", value=40, unit="USD", event_id="e1", evidence=(_ref(d, "Fee1 40 USD."),), structural_reliability=1.0),
                "f2": ClaimNode(claim_id="f2", kind="extraction", value=60, unit="USD", event_id="e2", evidence=(_ref(d, "Fee2 60 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=100, unit="USD"),
            },
            steps=(InferenceStep(step_id="s1", premises=("f1", "f2"), conclusion_id="c", operation="agg_sum", parameters={"expected_population_ids": ["e1", "e2"]}),),
            conclusion_id="c",
        ), "supported"),
        ("agg_incomplete_unknown", JustificationGraph(
            claims={
                "f1": ClaimNode(claim_id="f1", kind="extraction", value=40, unit="USD", event_id="e1", evidence=(_ref(d, "Fee1 40 USD."),), structural_reliability=1.0),
                "f2": ClaimNode(claim_id="f2", kind="extraction", value=60, unit="USD", event_id="e2", evidence=(_ref(d, "Fee2 60 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", value=100, unit="USD"),
            },
            steps=(InferenceStep(step_id="s1", premises=("f1", "f2"), conclusion_id="c", operation="agg_sum"),),
            conclusion_id="c",
        ), "inconclusive"),
        # 6. explicit_logic valid / bad version
        ("logic_rule_valid", JustificationGraph(
            claims={
                "p": ClaimNode(claim_id="p", kind="extraction", predicate="age_eligible", value=True, polarity=1, evidence=(_ref(d, "Born 1980-03-10."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", predicate="discount_eligible", value=True),
            },
            steps=(InferenceStep(step_id="s1", premises=("p",), conclusion_id="c", operation="explicit_logic", rule_id="senior_discount_v1", rule_version="1.0"),),
            conclusion_id="c",
        ), "supported"),
        ("logic_version_mismatch", JustificationGraph(
            claims={
                "p": ClaimNode(claim_id="p", kind="extraction", predicate="age_eligible", value=True, polarity=1, evidence=(_ref(d, "Born 1980-03-10."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", predicate="discount_eligible", value=True),
            },
            steps=(InferenceStep(step_id="s1", premises=("p",), conclusion_id="c", operation="explicit_logic", rule_id="senior_discount_v1", rule_version="9.9"),),
            conclusion_id="c",
        ), "inconclusive"),
        # 7. relations valid / unauthorized
        ("rel_comp_valid", JustificationGraph(
            claims={
                "e1": ClaimNode(claim_id="e1", kind="extraction", subject="CoA", predicate="controls_entity", value="SubB", evidence=(_ref(d, "CoA controls SubB."),), structural_reliability=1.0),
                "e2": ClaimNode(claim_id="e2", kind="extraction", subject="SubB", predicate="owns_asset", value="PlantC", evidence=(_ref(d, "SubB owns PlantC."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", subject="CoA", predicate="indirectly_owns", value="PlantC"),
            },
            steps=(InferenceStep(step_id="s1", premises=("e1", "e2"), conclusion_id="c", operation="relation_composition"),),
            conclusion_id="c",
        ), "supported"),
        ("rel_unauth_unknown", JustificationGraph(
            claims={
                "e1": ClaimNode(claim_id="e1", kind="extraction", subject="CoA", predicate="knows", value="SubB", evidence=(_ref(d, "CoA controls SubB."),), structural_reliability=1.0),
                "e2": ClaimNode(claim_id="e2", kind="extraction", subject="SubB", predicate="owns_asset", value="PlantC", evidence=(_ref(d, "SubB owns PlantC."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", subject="CoA", predicate="indirectly_owns", value="PlantC"),
            },
            steps=(InferenceStep(step_id="s1", premises=("e1", "e2"), conclusion_id="c", operation="relation_composition"),),
            conclusion_id="c",
        ), "inconclusive"),
        # 8. document_authority superseded
        ("doc_auth_superseded", JustificationGraph(
            claims={"s": ClaimNode(claim_id="s", kind="extraction", value=200, unit="USD", evidence=(_ref(d, "Base is 200 USD."),), structural_reliability=1.0, metadata={"superseded": True})},
            steps=(),
            conclusion_id="s",
        ), "refuted"),
        # 9. open_semantics never becomes deterministic_formal (probabilistic discount yields partially_supported)
        ("open_sem_probabilistic_only", JustificationGraph(
            claims={
                "p": ClaimNode(claim_id="p", kind="extraction", text="Base is 200 USD.", evidence=(_ref(d, "Base is 200 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", text="Reasonable fee."),
            },
            steps=(InferenceStep(step_id="s1", premises=("p",), conclusion_id="c", operation="open_semantics"),),
            conclusion_id="c",
        ), "partially_supported"),  # Probabilistic-only step never receives full deterministic supported/PASS
    ]

    passed = 0
    false_verified_on_invalid = 0
    details: list[dict[str, Any]] = []
    for name, graph, expected_eval in cases:
        res = evaluate_response(
            graph,
            evidence={"l1": d},
            domain_profile=profile,
            budget=VerificationBudget(max_semantic_calls=1),
        )
        ok = res.assessment == expected_eval
        if name == "open_sem_probabilistic_only":
            # Verify guarantee_kind lock AND authorization escalation
            verdict = authorize(res, operation="pay", policy=profile)
            ok = ok and (not verdict.allowed) and any(
                c.guarantee_kind == "probabilistic_model" for c in res.completed_checks if c.family == "open_semantics"
            )
        if ok:
            passed += 1
        elif res.assessment == "supported" and expected_eval != "supported":
            false_verified_on_invalid += 1
        details.append({"case": name, "expected": expected_eval, "observed": res.assessment, "ok": ok})

    return {
        "total": len(cases),
        "passed": passed,
        "accuracy": round(passed / len(cases), 4),
        "false_verified_rate": round(false_verified_on_invalid / len(cases), 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Layer 2: Extraction & Relational Association
# ---------------------------------------------------------------------------

def run_layer2_extraction_fidelity(profile: DomainProfile) -> dict[str, Any]:
    doc_text = (
        "Acme cap is 500 USD and Beta fee is 900 USD. "
        "Acme cure_period is 30 calendar days; Acme notice_period is 15 calendar days. "
        "Acme cap is 250 USD unless payment is overdue. "
        "Beta fee shall not be 100 USD.\n"
        "| Entity | cap | fee |\n"
        "| Acme | 500 USD | 100 USD |\n"
        "| Beta | 800 USD | 900 USD |"
    )
    doc = _doc("l2", doc_text)
    cases: list[tuple[str, ClaimNode, str]] = [
        ("valid_acme_cap", ClaimNode(
            claim_id="l2_1", kind="extraction", subject="Acme", predicate="cap", value=500, unit="USD",
            evidence=(_ref(doc, "Acme cap is 500 USD and Beta fee is 900 USD."),), structural_reliability=1.0,
        ), "supported"),
        ("swapped_subject_beta_fee_as_acme_cap", ClaimNode(
            claim_id="l2_2", kind="extraction", subject="Acme", predicate="cap", value=900, unit="USD",
            evidence=(_ref(doc, "Acme cap is 500 USD and Beta fee is 900 USD."),), structural_reliability=1.0,
        ), "refuted"),
        ("competing_predicate_notice_as_cure", ClaimNode(
            claim_id="l2_3", kind="extraction", subject="Acme", predicate="cure_period", value=15, unit="calendar_days",
            evidence=(_ref(doc, "Acme cure_period is 30 calendar days; Acme notice_period is 15 calendar days."),), structural_reliability=1.0,
        ), "refuted"),
        ("uncaptured_condition_unless", ClaimNode(
            claim_id="l2_4", kind="extraction", subject="Acme", predicate="cap", value=250, unit="USD",
            condition_status="unconditional",
            evidence=(_ref(doc, "Acme cap is 250 USD unless payment is overdue."),), structural_reliability=1.0,
        ), "refuted"),
        ("polarity_mismatch_shall_not", ClaimNode(
            claim_id="l2_5", kind="extraction", subject="Beta", predicate="fee", value=100, unit="USD", polarity=1,
            evidence=(_ref(doc, "Beta fee shall not be 100 USD."),), structural_reliability=1.0,
        ), "refuted"),
        ("unit_mismatch_usd_vs_days", ClaimNode(
            claim_id="l2_6", kind="extraction", subject="Acme", predicate="cure_period", value=30, unit="USD",
            evidence=(_ref(doc, "Acme cure_period is 30 calendar days;"),), structural_reliability=1.0,
        ), "refuted"),
        ("table_row_valid_acme_cap", ClaimNode(
            claim_id="l2_7", kind="extraction", subject="Acme", predicate="cap", value=500, unit="USD",
            evidence=(_ref(doc, "| Entity | cap | fee |\n| Acme | 500 USD | 100 USD |"),), structural_reliability=1.0,
        ), "supported"),
        ("table_cross_column_wrong_beta_cap", ClaimNode(
            claim_id="l2_8", kind="extraction", subject="Acme", predicate="cap", value=800, unit="USD",
            evidence=(_ref(doc, "| Entity | cap | fee |\n| Acme | 500 USD | 100 USD |\n| Beta | 800 USD | 900 USD |"),), structural_reliability=1.0,
        ), "refuted"),
    ]

    passed = 0
    false_pass = 0
    details = []
    for name, node, expected in cases:
        graph = JustificationGraph(claims={node.claim_id: node}, steps=(), conclusion_id=node.claim_id)
        res = evaluate_response(graph, evidence={"l2": doc}, domain_profile=profile)
        ok = res.assessment == expected
        if ok:
            passed += 1
        elif expected == "refuted" and res.assessment == "supported":
            false_pass += 1
        details.append({"case": name, "expected": expected, "observed": res.assessment, "ok": ok})

    return {
        "total": len(cases),
        "passed": passed,
        "accuracy": round(passed / len(cases), 4),
        "false_pass_rate": round(false_pass / len(cases), 4),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Layer 3: Full Response & 4-Arm Comparison
# ---------------------------------------------------------------------------

def run_layer3_arms_comparison(profile: DomainProfile) -> dict[str, Any]:
    doc_primary = _doc(
        "d_main",
        "Applicant Elena was born on 1980-03-10. Base order is 200 USD. Discount is 10 percent. "
        "Primary notice window is 30 days. Invoice A is 40 USD. Invoice B is 60 USD.",
        weight=0.95,
    )
    doc_alt = _doc("d_conflict", "Notice window is 60 days.", weight=0.90)
    doc_year = _doc("d_year", "Applicant Carlos was born in 1982.", weight=0.95)

    # Scenario spec: (id, graph, docs, budget, gold_should_authorize, is_valid_reasoning, requires_derivation)
    scenarios: list[tuple[str, JustificationGraph, dict[str, SourceDocument], VerificationBudget, bool, bool, bool]] = [
        # 1. Valid non-verbatim age derivation -> MUST evaluate supported & authorize PASS
        ("valid_age_derivation", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1980-03-10", evidence=(_ref(doc_primary, "Applicant Elena was born on 1980-03-10."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value=45, unit="years"),
            },
            steps=(InferenceStep(step_id="s", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), {"d_main": doc_primary}, VerificationBudget(), True, True, True),
        # 2. Valid arithmetic net payout: 200 - 20 = 180 USD -> MUST authorize PASS
        ("valid_arithmetic_discount", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", value=200, unit="USD", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "d": ClaimNode(claim_id="d", kind="extraction", value=20, unit="USD", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "net": ClaimNode(claim_id="net", kind="derivation", value=180, unit="USD"),
            },
            steps=(InferenceStep(step_id="s", premises=("b", "d"), conclusion_id="net", operation="sub"),),
            conclusion_id="net",
        ), {"d_main": doc_primary}, VerificationBudget(), True, True, True),
        # 3. Wrong arithmetic calculation -> MUST refute & block
        ("wrong_arithmetic", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", value=200, unit="USD", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "d": ClaimNode(claim_id="d", kind="extraction", value=20, unit="USD", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "net": ClaimNode(claim_id="net", kind="derivation", value=150, unit="USD"),
            },
            steps=(InferenceStep(step_id="s", premises=("b", "d"), conclusion_id="net", operation="sub"),),
            conclusion_id="net",
        ), {"d_main": doc_primary}, VerificationBudget(), False, False, True),
        # 4. Valid coarse year uncertainty interval -> MUST evaluate supported & authorize PASS
        ("valid_age_interval", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1982", evidence=(_ref(doc_year, "Applicant Carlos was born in 1982."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value={"min": 42, "max": 43}, unit="years"),
            },
            steps=(InferenceStep(step_id="s", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), {"d_year": doc_year}, VerificationBudget(), True, True, True),
        # 5. Overconfident categorical age from year-only birth date -> abstain & block
        ("overconfident_year_age", JustificationGraph(
            claims={
                "bd": ClaimNode(claim_id="bd", kind="extraction", value="1982", evidence=(_ref(doc_year, "Applicant Carlos was born in 1982."),), structural_reliability=1.0),
                "ag": ClaimNode(claim_id="ag", kind="derivation", value=43, unit="years"),
            },
            steps=(InferenceStep(step_id="s", premises=("bd",), conclusion_id="ag", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
            conclusion_id="ag",
        ), {"d_year": doc_year}, VerificationBudget(), False, False, True),
        # 6. Recommendation output -> valid for exploration evaluation, MUST block operation
        ("exploratory_recommendation", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", value=200, unit="USD", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "rec": ClaimNode(claim_id="rec", kind="recommendation", text="Consider increasing limit."),
            },
            steps=(InferenceStep(step_id="s", premises=("b",), conclusion_id="rec", operation="open_semantics"),),
            conclusion_id="rec",
        ), {"d_main": doc_primary}, VerificationBudget(max_semantic_calls=1), False, False, True),
        # 7. Conflicting sources (30 days vs 60 days) -> contested, MUST block
        ("contested_notice", JustificationGraph(
            claims={
                "c1": ClaimNode(claim_id="c1", kind="extraction", subject="Vendor", predicate="notice_period", value=30, unit="calendar_days", evidence=(_ref(doc_primary, "Primary notice window is 30 days."),), structural_reliability=1.0),
                "c2": ClaimNode(claim_id="c2", kind="extraction", subject="Vendor", predicate="notice_period", value=60, unit="calendar_days", evidence=(_ref(doc_alt, "Notice window is 60 days."),), structural_reliability=1.0),
            },
            steps=(),
            conclusion_id="c1",
        ), {"d_main": doc_primary, "d_conflict": doc_alt}, VerificationBudget(), False, False, False),
        # 8. Valid complete aggregation -> MUST authorize PASS
        ("valid_agg_sum", JustificationGraph(
            claims={
                "a": ClaimNode(claim_id="a", kind="extraction", value=40, unit="USD", event_id="invA", evidence=(_ref(doc_primary, "Invoice A is 40 USD."),), structural_reliability=1.0),
                "b": ClaimNode(claim_id="b", kind="extraction", value=60, unit="USD", event_id="invB", evidence=(_ref(doc_primary, "Invoice B is 60 USD."),), structural_reliability=1.0),
                "sum": ClaimNode(claim_id="sum", kind="derivation", value=100, unit="USD"),
            },
            steps=(InferenceStep(step_id="s", premises=("a", "b"), conclusion_id="sum", operation="agg_sum", parameters={"expected_population_ids": ["invA", "invB"]}),),
            conclusion_id="sum",
        ), {"d_main": doc_primary}, VerificationBudget(), True, True, True),
        # 9. Missing population aggregation -> inconclusive, MUST block
        ("incomplete_agg_sum", JustificationGraph(
            claims={
                "a": ClaimNode(claim_id="a", kind="extraction", value=40, unit="USD", event_id="invA", evidence=(_ref(doc_primary, "Invoice A is 40 USD."),), structural_reliability=1.0),
                "b": ClaimNode(claim_id="b", kind="extraction", value=60, unit="USD", event_id="invB", evidence=(_ref(doc_primary, "Invoice B is 60 USD."),), structural_reliability=1.0),
                "sum": ClaimNode(claim_id="sum", kind="derivation", value=100, unit="USD"),
            },
            steps=(InferenceStep(step_id="s", premises=("a", "b"), conclusion_id="sum", operation="agg_sum"),),
            conclusion_id="sum",
        ), {"d_main": doc_primary}, VerificationBudget(), False, False, True),
        # 10. Open-semantics only critical step -> block autonomous execution
        ("open_sem_critical_step", JustificationGraph(
            claims={
                "b": ClaimNode(claim_id="b", kind="extraction", text="Base order is 200 USD.", evidence=(_ref(doc_primary, "Base order is 200 USD."),), structural_reliability=1.0),
                "c": ClaimNode(claim_id="c", kind="derivation", text="Implied good faith approval."),
            },
            steps=(InferenceStep(step_id="s", premises=("b",), conclusion_id="c", operation="open_semantics"),),
            conclusion_id="c",
        ), {"d_main": doc_primary}, VerificationBudget(max_semantic_calls=1), False, False, True),
    ]

    arms = {
        "llm_unverified": {"false_support": 0, "false_reject": 0, "abstained": 0, "coverages": [], "latencies_ms": []},
        "literal_extraction_only": {"false_support": 0, "false_reject": 0, "abstained": 0, "coverages": [], "latencies_ms": []},
        "graduated_v4_evaluate": {"false_support": 0, "false_reject": 0, "abstained": 0, "coverages": [], "latencies_ms": []},
        "graduated_v4_authorize": {"false_support": 0, "false_reject": 0, "abstained": 0, "coverages": [], "latencies_ms": []},
    }

    for _, graph, docs, bud, gold_auth, gold_valid, needs_deriv in scenarios:
        # Arm 1: LLM without verifier accepts everything blindly
        t0 = time.perf_counter()
        arm1_accept = True
        dt1 = (time.perf_counter() - t0) * 1000.0
        arms["llm_unverified"]["latencies_ms"].append(dt1)
        arms["llm_unverified"]["coverages"].append(0.0)
        if arm1_accept and not gold_auth:
            arms["llm_unverified"]["false_support"] += 1

        # Arm 2: Strict literal extraction baseline (no derivation/temporal/arithmetic verifier;
        # accepts only verbatim leaf extractions without conflicts or derivations)
        t0 = time.perf_counter()
        conc = graph.claims[graph.conclusion_id]
        has_conflict = len(graph.claims) > 1 and not graph.steps
        arm2_accept = (not needs_deriv) and conc.kind == "extraction" and bool(conc.evidence) and not has_conflict
        dt2 = (time.perf_counter() - t0) * 1000.0
        arms["literal_extraction_only"]["latencies_ms"].append(dt2)
        arms["literal_extraction_only"]["coverages"].append(0.0 if needs_deriv else 1.0)
        if not arm2_accept:
            arms["literal_extraction_only"]["abstained"] += 1
        if arm2_accept and not gold_auth:
            arms["literal_extraction_only"]["false_support"] += 1
        elif (not arm2_accept) and gold_auth:
            arms["literal_extraction_only"]["false_reject"] += 1

        # Arm 3: Graduated evaluate_response
        t0 = time.perf_counter()
        assessment = evaluate_response(graph, evidence=docs, domain_profile=profile, budget=bud)
        dt3 = (time.perf_counter() - t0) * 1000.0
        arms["graduated_v4_evaluate"]["latencies_ms"].append(dt3)
        arms["graduated_v4_evaluate"]["coverages"].append(assessment.verification_coverage)
        eval_accepts_as_supported = assessment.assessment == "supported" and not assessment.critical_unverified_steps
        if not eval_accepts_as_supported:
            arms["graduated_v4_evaluate"]["abstained"] += 1
        if eval_accepts_as_supported and not gold_valid:
            # Note: open_sem_critical_step without authorization may have assessment=='supported' (probabilistic);
            # Arm 4 (authorize) catches all operational risks!
            arms["graduated_v4_evaluate"]["false_support"] += 1
        elif (not eval_accepts_as_supported) and gold_valid:
            arms["graduated_v4_evaluate"]["false_reject"] += 1

        # Arm 4: Graduated evaluate_response + authorize
        t0 = time.perf_counter()
        verdict = authorize(assessment, operation="execute_action", policy=profile)
        dt4 = dt3 + (time.perf_counter() - t0) * 1000.0
        arms["graduated_v4_authorize"]["latencies_ms"].append(dt4)
        arms["graduated_v4_authorize"]["coverages"].append(assessment.verification_coverage)
        if not verdict.allowed:
            arms["graduated_v4_authorize"]["abstained"] += 1
        if verdict.allowed and not gold_auth:
            arms["graduated_v4_authorize"]["false_support"] += 1
        elif (not verdict.allowed) and gold_auth:
            arms["graduated_v4_authorize"]["false_reject"] += 1

    n = len(scenarios)
    invalid_total = sum(1 for s in scenarios if not s[4])
    valid_total = sum(1 for s in scenarios if s[4])

    summary_arms = {}
    for arm_name, rec in arms.items():
        summary_arms[arm_name] = {
            "false_support_rate": round(rec["false_support"] / max(1, invalid_total), 4),
            "false_rejection_rate": round(rec["false_reject"] / max(1, valid_total), 4),
            "abstention_rate": round(rec["abstained"] / n, 4),
            "mean_coverage": round(sum(rec["coverages"]) / n, 4),
            "mean_latency_ms": round(sum(rec["latencies_ms"]) / n, 4),
        }

    # Incremental cache recomputation benchmark on a 2-branch DAG
    docA1 = _doc("dA", "Left amount is 40 USD.")
    docA2 = _doc("dA", "Left amount is 50 USD.")
    docB = _doc("dB", "Right amount is 60 USD.")

    def _branch_graph(da: SourceDocument, a_val: int, tot_val: int) -> JustificationGraph:
        return JustificationGraph(
            claims={
                "a": ClaimNode(claim_id="a", kind="extraction", value=a_val, unit="USD", evidence=(_ref(da, f"Left amount is {a_val} USD."),), structural_reliability=1.0),
                "b": ClaimNode(claim_id="b", kind="extraction", value=60, unit="USD", evidence=(_ref(docB, "Right amount is 60 USD."),), structural_reliability=1.0),
                "a_chk": ClaimNode(claim_id="a_chk", kind="derivation", value=a_val, unit="USD"),
                "b_chk": ClaimNode(claim_id="b_chk", kind="derivation", value=60, unit="USD"),
                "tot": ClaimNode(claim_id="tot", kind="derivation", value=tot_val, unit="USD"),
            },
            steps=(
                InferenceStep(step_id="sa", premises=("a",), conclusion_id="a_chk", operation="unit_conversion"),
                InferenceStep(step_id="sb", premises=("b",), conclusion_id="b_chk", operation="unit_conversion"),
                InferenceStep(step_id="st", premises=("a_chk", "b_chk"), conclusion_id="tot", operation="add"),
            ),
            conclusion_id="tot",
        )

    cache = VerificationCache()
    evaluate_response(_branch_graph(docA1, 40, 100), evidence={"dA": docA1, "dB": docB}, domain_profile=profile, cache=cache)
    inc_res = evaluate_response(_branch_graph(docA2, 50, 110), evidence={"dA": docA2, "dB": docB}, domain_profile=profile, cache=cache)
    total_nodes = inc_res.cache_metrics["reused_nodes"] + inc_res.cache_metrics["recomputed_nodes"]
    reuse_ratio = round(inc_res.cache_metrics["reused_nodes"] / max(1, total_nodes), 4)

    return {
        "scenarios_total": n,
        "arms": summary_arms,
        "incremental_cache": {
            **inc_res.cache_metrics,
            "incremental_cache_reuse_ratio": reuse_ratio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClaimLayer v4 3-layer graduated evaluation benchmark.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    profile = _shared_profile()
    layer1 = run_layer1_isolated_verifiers(profile)
    layer2 = run_layer2_extraction_fidelity(profile)
    layer3 = run_layer3_arms_comparison(profile)

    report = {
        "layer1_isolated_verifiers": layer1,
        "layer2_extraction_fidelity": layer2,
        "layer3_full_response_4_arms": layer3,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=== ClaimLayer v4 Graduated 3-Layer Evaluation ===")
    print(f"Layer 1 (Isolated Verifiers, 9 families): {layer1['passed']}/{layer1['total']} passed (false_verified={layer1['false_verified_rate']})")
    print(f"Layer 2 (Extraction & Relational Fidelity): {layer2['passed']}/{layer2['total']} passed (false_pass={layer2['false_pass_rate']})")
    print("Layer 3 (4-Arm Comparison):")
    for arm, stats in layer3["arms"].items():
        print(
            f"  - {arm:24s} | false_support={stats['false_support_rate']:.3f} "
            f"| false_reject={stats['false_rejection_rate']:.3f} "
            f"| abstain={stats['abstention_rate']:.3f} "
            f"| coverage={stats['mean_coverage']:.3f} "
            f"| latency_ms={stats['mean_latency_ms']:.3f}"
        )
    inc = layer3["incremental_cache"]
    print(
        f"Incremental Cache (partial doc change): reused={inc['reused_nodes']}, "
        f"recomputed={inc['recomputed_nodes']}, reuse_ratio={inc['incremental_cache_reuse_ratio']:.2f}"
    )


if __name__ == "__main__":
    main()
