"""Comprehensive tests for Phases 1–4 of the graduated Justification Graph & Verifier engine."""
from __future__ import annotations

import pytest

from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    AcceptancePolicy,
    ClauseRelation,
    ClaimNode,
    DeclaredRule,
    DomainProfile,
    EmpiricalCalibration,
    EvidenceLayer,
    EvidenceUnit,
    InferenceStep,
    JustificationGraph,
    PolicyConfiguration,
    PredicateSpec,
    RelationCompositionRule,
    SemanticEvaluatorConfig,
    STANDARD_UNITS,
    Unit,
    VerificationBudget,
    VerificationCache,
    VerificationObligation,
    Vocabulary,
    authorize,
    effective_epistemic_kind,
    evaluate_response,
    execute_authorized_operation,
    verify_authorization_verdict,
    FieldSpec,
)


def _doc(doc_id: str, text: str, *, weight: float = 0.9, dep_group: str | None = None, attribution: str = "source_statement") -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        authority_rank=80,
        weight=weight,
        dependence_group=dep_group,
        attribution=attribution,
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


# ---------------------------------------------------------------------------
# Phase 1: Decoupled Evaluation vs Authorization & Epistemic Class Invariants
# ---------------------------------------------------------------------------

def test_evaluate_works_without_policy_and_separates_exploration_from_authorization() -> None:
    doc = _doc("d1", "Orion revenue was 200 USD in 2025. Risk note remains preliminary.")
    c1 = ClaimNode(
        claim_id="rev",
        kind="extraction",
        subject="Orion",
        predicate="revenue",
        value=200,
        unit="USD",
        evidence=(_ref(doc, "Orion revenue was 200 USD in 2025."),),
        structural_reliability=0.95,
    )
    c_rec = ClaimNode(
        claim_id="rec",
        kind="recommendation",
        subject="Orion",
        predicate="credit_recommendation",
        value="expand_limit",
    )
    s1 = InferenceStep(
        step_id="s_rec",
        premises=("rev",),
        conclusion_id="rec",
        operation="open_semantics",
    )
    graph = JustificationGraph(claims={"rev": c1, "rec": c_rec}, steps=(s1,), conclusion_id="rec")

    # Evaluates cleanly with NO authorization policy supplied
    assessment = evaluate_response(graph, evidence={"d1": doc})
    summary = assessment.to_summary_dict()

    assert summary["conclusion_id"] == "rec"
    assert summary["effective_epistemic_kind"] == "recommendation"
    assert summary["calibrated_correctness_probability"] is None
    assert "s_rec" in summary["critical_unverified_steps"]
    assert len(summary["pending_obligations"]) >= 1
    assert summary["verification_coverage"] < 1.0
    assert assessment.uncertainty_index > 0.0

    # Operational authorization refuses PASS because conclusion is recommendation and s_rec is unverified
    verdict = authorize(assessment, operation="approve_credit_increase")
    assert verdict.status == "ESCALATE"
    assert verdict.allowed is False
    codes = {i.code for i in verdict.issues}
    assert "NON_FACTUAL_EPISTEMIC_KIND_FOR_OPERATION" in codes
    assert "CRITICAL_UNVERIFIED_INFERENCE_STEPS" in codes


def test_hypothesis_prediction_recommendation_never_promoted_to_fact_even_with_high_weight() -> None:
    doc = _doc("d_high", "Analyst note suggests projected growth may reach 15 percent.", weight=1.0)
    hyp = ClaimNode(
        claim_id="h1",
        kind="hypothesis",
        subject="Acme",
        predicate="growth_rate",
        value=15,
        unit="percent",
        evidence=(_ref(doc, "Analyst note suggests projected growth may reach 15 percent."),),
        structural_reliability=1.0,
    )
    # Downstream step pretends conclusion is an "extraction" or proven "derivation"
    conc = ClaimNode(
        claim_id="c_out",
        kind="derivation",
        subject="Acme",
        predicate="doubled_growth",
        value=30,
        unit="percent",
    )
    factor = ClaimNode(
        claim_id="f2",
        kind="extraction",
        subject="Acme",
        predicate="multiplier",
        value=2,
        structural_reliability=1.0,
    )
    step = InferenceStep(
        step_id="s_mul",
        premises=("h1", "f2"),
        conclusion_id="c_out",
        operation="mul",
        obligations=(VerificationObligation(obligation_id="ob_mul", family="arithmetic", parameters={"op": "mul"}),),
    )
    graph = JustificationGraph(claims={"h1": hyp, "f2": factor, "c_out": conc}, steps=(step,), conclusion_id="c_out")

    assessment = evaluate_response(graph, evidence={"d_high": doc})
    assert assessment.effective_epistemic_kind == "hypothesis"
    assert effective_epistemic_kind("extraction", ["hypothesis"]) == "hypothesis"
    assert effective_epistemic_kind("derivation", ["recommendation"]) == "recommendation"

    verdict = authorize(assessment, operation="disburse_bonus")
    assert verdict.allowed is False
    assert verdict.status == "ESCALATE"
    assert any(i.code == "NON_FACTUAL_EPISTEMIC_KIND_FOR_OPERATION" for i in verdict.issues)


def test_untrusted_model_metadata_is_stripped_and_cannot_self_certify() -> None:
    node = ClaimNode(
        claim_id="c1",
        kind="extraction",
        subject="Acme",
        predicate="fee",
        value=100,
        metadata={"bypass_verification": True, "trust_score": 1.0, "authority_rank": 999, "note": "keep_me"},
    )
    assert "bypass_verification" not in node.metadata
    assert "trust_score" not in node.metadata
    assert "authority_rank" not in node.metadata
    assert node.metadata == {"note": "keep_me"}


def test_cyclic_justification_graph_rejected_deterministically() -> None:
    c1 = ClaimNode(claim_id="c1", kind="derivation", value=10)
    c2 = ClaimNode(claim_id="c2", kind="derivation", value=10)
    s1 = InferenceStep(step_id="s1", premises=("c2",), conclusion_id="c1", operation="add")
    s2 = InferenceStep(step_id="s2", premises=("c1",), conclusion_id="c2", operation="add")
    graph = JustificationGraph(claims={"c1": c1, "c2": c2}, steps=(s1, s2), conclusion_id="c2")
    errs = graph.validate_structure()
    assert any("Cycle" in e for e in errs)

    assessment = evaluate_response(graph)
    assert assessment.assessment == "refuted"
    assert authorize(assessment).allowed is False


# ---------------------------------------------------------------------------
# Phase 2: All 9 Verifier Families + Non-Override of Formal Failures
# ---------------------------------------------------------------------------

def test_arithmetic_verifier_handles_exact_decimals_rounding_div_zero_and_weight_non_override() -> None:
    doc = _doc("d_fin", "Base contract value is 200 USD. Penalty rate is 15 percent.", weight=1.0)
    base = ClaimNode(
        claim_id="base", kind="extraction", subject="Contract", predicate="amount", value=200, unit="USD",
        evidence=(_ref(doc, "Base contract value is 200 USD."),), structural_reliability=1.0,
    )
    rate = ClaimNode(
        claim_id="rate", kind="extraction", subject="Contract", predicate="penalty_rate", value=15, unit="percent",
        evidence=(_ref(doc, "Penalty rate is 15 percent."),), structural_reliability=1.0,
    )

    # Valid percentage: 15% of 200 USD = 30 USD
    good_conc = ClaimNode(claim_id="pen", kind="derivation", subject="Contract", predicate="penalty", value=30, unit="USD")
    good_step = InferenceStep(step_id="s1", premises=("base", "rate"), conclusion_id="pen", operation="percentage_of")
    good_graph = JustificationGraph(claims={"base": base, "rate": rate, "pen": good_conc}, steps=(good_step,), conclusion_id="pen")
    res_good = evaluate_response(good_graph, evidence={"d_fin": doc})
    assert res_good.assessment == "supported"
    assert authorize(res_good, operation="assess_penalty").status == "PASS"

    # Bad percentage: 15% of 200 USD claimed as 35 USD -> weights=1.0 CANNOT save it!
    bad_conc = ClaimNode(claim_id="pen", kind="derivation", subject="Contract", predicate="penalty", value=35, unit="USD")
    bad_graph = JustificationGraph(claims={"base": base, "rate": rate, "pen": bad_conc}, steps=(good_step,), conclusion_id="pen")
    res_bad = evaluate_response(bad_graph, evidence={"d_fin": doc})
    assert res_bad.assessment == "refuted"
    assert res_bad.evidence_support_index == 0.0
    assert res_bad.conflict_index == 1.0
    verdict_bad = authorize(res_bad, operation="assess_penalty")
    assert verdict_bad.status == "RETRY"
    assert verdict_bad.allowed is False
    assert any(i.code == "FORMAL_CHECK_REFUTED" for i in verdict_bad.issues)

    # Division by zero
    zero_node = ClaimNode(claim_id="z", kind="extraction", value=0, evidence=(_ref(doc, "Base contract value is 200 USD."),), structural_reliability=1.0)
    div_conc = ClaimNode(claim_id="q", kind="derivation", value=0)
    div_step = InferenceStep(step_id="s_div", premises=("base", "z"), conclusion_id="q", operation="div")
    res_div0 = evaluate_response(JustificationGraph(claims={"base": base, "z": zero_node, "q": div_conc}, steps=(div_step,), conclusion_id="q"), evidence={"d_fin": doc})
    assert res_div0.assessment == "refuted"
    assert any("DIVISION_BY_ZERO" in c.reasons for c in res_div0.completed_checks)

    # Explicit rounding within tolerance records assumption and passes
    p_a = ClaimNode(claim_id="a", kind="extraction", value="10.004", unit="USD", evidence=(_ref(doc, "Base contract value is 200 USD."),), structural_reliability=1.0)
    p_b = ClaimNode(claim_id="b", kind="extraction", value="20.004", unit="USD", evidence=(_ref(doc, "Penalty rate is 15 percent."),), structural_reliability=1.0)
    r_conc = ClaimNode(claim_id="r", kind="derivation", value="30.01", unit="USD")
    r_step = InferenceStep(step_id="s_rnd", premises=("a", "b"), conclusion_id="r", operation="add", parameters={"decimals": 2})
    res_rnd = evaluate_response(JustificationGraph(claims={"a": p_a, "b": p_b, "r": r_conc}, steps=(r_step,), conclusion_id="r"), evidence={"d_fin": doc})
    assert res_rnd.assessment == "supported"
    assert any("rounding_half_up_decimals:2" in a for a in res_rnd.assumptions)


def test_quantity_verifier_converts_compatible_units_and_refutes_incompatible_dimensions() -> None:
    ext_units = STANDARD_UNITS.extend([Unit("minutes", "duration_hours", "0.0166666666666666666666666667", ("min", "minute"))])
    profile = DomainProfile(profile_id="sla-ops", unit_registry=ext_units)
    doc = _doc("d_sla", "Response window is 2 hours. Maximum fee is 23 USD.", weight=1.0)

    c_hrs = ClaimNode(
        claim_id="h", kind="extraction", subject="Vendor", predicate="response_window", value=2, unit="hours",
        evidence=(_ref(doc, "Response window is 2 hours."),), structural_reliability=1.0,
    )
    c_mins = ClaimNode(
        claim_id="m", kind="derivation", subject="Vendor", predicate="response_window", value=120, unit="minutes",
    )
    s_conv = InferenceStep(
        step_id="conv", premises=("h",), conclusion_id="m", operation="unit_conversion",
        parameters={"tolerance": "0.0001"},
    )
    res_ok = evaluate_response(
        JustificationGraph(claims={"h": c_hrs, "m": c_mins}, steps=(s_conv,), conclusion_id="m"),
        evidence={"d_sla": doc},
        domain_profile=profile,
    )
    assert res_ok.assessment == "supported"
    assert authorize(res_ok, operation="check_sla", policy=profile).status == "PASS"

    # Incompatible dimension: 23 USD claimed as 23 calendar_days
    c_usd = ClaimNode(
        claim_id="u", kind="extraction", subject="Vendor", predicate="cap", value=23, unit="USD",
        evidence=(_ref(doc, "Maximum fee is 23 USD."),), structural_reliability=1.0,
    )
    c_days = ClaimNode(claim_id="d", kind="derivation", subject="Vendor", predicate="cap", value=23, unit="calendar_days")
    s_bad = InferenceStep(step_id="conv_bad", premises=("u",), conclusion_id="d", operation="unit_conversion")
    res_bad = evaluate_response(
        JustificationGraph(claims={"u": c_usd, "d": c_days}, steps=(s_bad,), conclusion_id="d"),
        evidence={"d_sla": doc},
        domain_profile=profile,
    )
    assert res_bad.assessment == "refuted"
    assert authorize(res_bad, operation="check_sla", policy=profile).allowed is False


def test_temporal_verifier_derives_age_and_uncertainty_intervals_without_verbatim_number_in_quote() -> None:
    # Crucial user acceptance case: birth date is cited, derived age (45) does NOT appear verbatim in text!
    doc_exact = _doc("id_doc", "Applicant Elena Ruiz was born on 1980-03-10.", weight=1.0)
    c_birth = ClaimNode(
        claim_id="birth", kind="extraction", subject="Elena Ruiz", predicate="birth_date",
        value="1980-03-10", evidence=(_ref(doc_exact, "Applicant Elena Ruiz was born on 1980-03-10."),),
        structural_reliability=1.0,
    )
    c_age = ClaimNode(
        claim_id="age", kind="derivation", subject="Elena Ruiz", predicate="age", value=45, unit="years",
    )
    s_age = InferenceStep(
        step_id="calc_age",
        premises=("birth",),
        conclusion_id="age",
        operation="temporal_age",
        parameters={"reference_date": "2025-06-15"},
    )
    graph = JustificationGraph(claims={"birth": c_birth, "age": c_age}, steps=(s_age,), conclusion_id="age")

    assessment = evaluate_response(graph, evidence={"id_doc": doc_exact})
    assert assessment.assessment == "supported"
    assert assessment.verification_coverage == 1.0
    assert authorize(assessment, operation="verify_minimum_age").status == "PASS"

    # Coarse year granularity: "Born in 1980" evaluated on "2025-06-15" -> [44, 45]
    doc_year = _doc("id_year", "Record states Elena Ruiz was born in 1980.", weight=1.0)
    c_birth_y = ClaimNode(
        claim_id="birth_y", kind="extraction", subject="Elena Ruiz", predicate="birth_date",
        value="1980", evidence=(_ref(doc_year, "Record states Elena Ruiz was born in 1980."),),
        structural_reliability=1.0,
    )

    # 1) Categorical exact age "45" without interval/approx -> unknown due to coarse precision
    g_cat = JustificationGraph(
        claims={"birth_y": c_birth_y, "age": c_age},
        steps=(InferenceStep(step_id="calc_y", premises=("birth_y",), conclusion_id="age", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
        conclusion_id="age",
    )
    res_cat = evaluate_response(g_cat, evidence={"id_year": doc_year})
    assert res_cat.assessment == "inconclusive"
    assert res_cat.evidence_support_index == 0.0
    assert any("TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE" in o["reasons"] for o in res_cat.pending_obligations)
    assert authorize(res_cat).allowed is False

    # 2) Explicit uncertainty interval [44, 45] or declared approx_tolerance=1 -> verified!
    c_age_iv = ClaimNode(
        claim_id="age_iv", kind="derivation", subject="Elena Ruiz", predicate="age",
        value={"min": 44, "max": 45}, unit="years",
    )
    g_iv = JustificationGraph(
        claims={"birth_y": c_birth_y, "age_iv": c_age_iv},
        steps=(InferenceStep(step_id="calc_iv", premises=("birth_y",), conclusion_id="age_iv", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
        conclusion_id="age_iv",
    )
    res_iv = evaluate_response(g_iv, evidence={"id_year": doc_year})
    assert res_iv.assessment == "supported"
    assert any("coarse_date_granularity" in a for a in res_iv.assumptions)

    c_age_approx = ClaimNode(
        claim_id="age_ap", kind="derivation", subject="Elena Ruiz", predicate="age",
        value=45, unit="years", approximate=True, approx_tolerance=1.0,
    )
    g_ap = JustificationGraph(
        claims={"birth_y": c_birth_y, "age_ap": c_age_approx},
        steps=(InferenceStep(step_id="calc_ap", premises=("birth_y",), conclusion_id="age_ap", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
        conclusion_id="age_ap",
    )
    assert evaluate_response(g_ap, evidence={"id_year": doc_year}).assessment == "supported"

    # 3) Impossible age "62" -> refuted!
    c_age_bad = ClaimNode(claim_id="age_bad", kind="derivation", subject="Elena Ruiz", predicate="age", value=62, unit="years")
    g_bad = JustificationGraph(
        claims={"birth_y": c_birth_y, "age_bad": c_age_bad},
        steps=(InferenceStep(step_id="calc_bad", premises=("birth_y",), conclusion_id="age_bad", operation="temporal_age", parameters={"reference_date": "2025-06-15"}),),
        conclusion_id="age_bad",
    )
    assert evaluate_response(g_bad, evidence={"id_year": doc_year}).assessment == "refuted"


def test_comparison_aggregation_explicit_logic_relations_and_document_authority_verifiers() -> None:
    # Comparison with unit alignment + interval indecision
    doc = _doc("d_mix", "Cap A is 100 USD. Invoice 1 is 40 USD. Invoice 2 is 60 USD. CoA owns SubB valid 2025. SubB operates PlantC.", weight=0.95)
    cap = ClaimNode(claim_id="cap", kind="extraction", value=100, unit="USD", evidence=(_ref(doc, "Cap A is 100 USD."),), structural_reliability=0.95)
    inv1 = ClaimNode(claim_id="i1", kind="extraction", subject="Inv1", predicate="charge", value=40, unit="USD", event_id="evt-1", evidence=(_ref(doc, "Invoice 1 is 40 USD."),), structural_reliability=0.95)
    inv2 = ClaimNode(claim_id="i2", kind="extraction", subject="Inv2", predicate="charge", value=60, unit="USD", event_id="evt-2", evidence=(_ref(doc, "Invoice 2 is 60 USD."),), structural_reliability=0.95)
    inv2_dup = ClaimNode(claim_id="i2_dup", kind="extraction", subject="Inv2", predicate="charge", value=60, unit="USD", event_id="evt-2", evidence=(_ref(doc, "Invoice 2 is 60 USD."),), structural_reliability=0.95)

    # Aggregation sum with population completeness & deduplicated evt-2 -> 100 USD
    agg_conc = ClaimNode(claim_id="total", kind="derivation", value=100, unit="USD")
    s_agg = InferenceStep(
        step_id="agg_sum", premises=("i1", "i2", "i2_dup"), conclusion_id="total", operation="agg_sum",
        parameters={"expected_population_ids": ["evt-1", "evt-2"]},
    )
    res_agg = evaluate_response(
        JustificationGraph(claims={"i1": inv1, "i2": inv2, "i2_dup": inv2_dup, "total": agg_conc}, steps=(s_agg,), conclusion_id="total"),
        evidence={"d_mix": doc},
    )
    assert res_agg.assessment == "supported"
    assert any("deduplicated_event:evt-2" in a for a in res_agg.assumptions)

    # Missing population completeness triggers abstention (never silent PASS)
    s_agg_incomp = InferenceStep(step_id="agg_inc", premises=("i1", "i2"), conclusion_id="total", operation="agg_sum")
    res_inc = evaluate_response(
        JustificationGraph(claims={"i1": inv1, "i2": inv2, "total": agg_conc}, steps=(s_agg_incomp,), conclusion_id="total"),
        evidence={"d_mix": doc},
    )
    assert res_inc.assessment == "inconclusive"
    assert res_inc.evidence_support_index == 0.0
    assert authorize(res_inc).allowed is False

    # Mixed currency (USD vs EUR) -> REFUTED, never silently dropped
    inv_eur = ClaimNode(claim_id="i_eur", kind="extraction", subject="Inv3", predicate="charge", value=50, unit="EUR", event_id="evt-3", evidence=(_ref(doc, "Invoice 2 is 60 USD."),), structural_reliability=0.95)
    s_agg_eur = InferenceStep(step_id="agg_mix", premises=("i1", "i_eur"), conclusion_id="total", operation="agg_sum", parameters={"population_complete": True})
    res_eur = evaluate_response(
        JustificationGraph(claims={"i1": inv1, "i_eur": inv_eur, "total": agg_conc}, steps=(s_agg_eur,), conclusion_id="total"),
        evidence={"d_mix": doc},
    )
    assert res_eur.assessment == "refuted"

    # Comparison le: 100 USD <= 100 USD -> verified
    cmp_conc = ClaimNode(claim_id="cmp_ok", kind="derivation", value=True)
    s_cmp = InferenceStep(step_id="s_cmp", premises=("total", "cap"), conclusion_id="cmp_ok", operation="le")
    res_cmp = evaluate_response(
        JustificationGraph(
            claims={"i1": inv1, "i2": inv2, "i2_dup": inv2_dup, "cap": cap, "total": agg_conc, "cmp_ok": cmp_conc},
            steps=(s_agg, s_cmp),
            conclusion_id="cmp_ok",
        ),
        evidence={"d_mix": doc},
    )
    assert res_cmp.assessment == "supported"

    # Explicit logic + Relation composition via versioned DomainProfile
    rule = DeclaredRule(
        rule_id="rule_cap_compliant",
        version="2.0",
        kind="implication",
        required_premises=({"predicate": "within_cap", "value": True, "polarity": 1},),
        conclusion_template={"predicate": "payout_eligible", "value": True},
    )
    rel_rule = RelationCompositionRule(
        rule_id="owns_operates_rel",
        left_predicate="owns_subsidiary",
        right_predicate="operates_facility",
        composed_predicate="indirectly_controls",
    )
    profile = DomainProfile(
        profile_id="corp-rules",
        version="1.2.0",
        declared_rules={"rule_cap_compliant": rule},
        relation_rules={"owns_operates_rel": rel_rule},
    )

    e_a = ClaimNode(claim_id="e1", kind="extraction", subject="CoA", predicate="owns_subsidiary", value="SubB", valid_from="2025-01-01", valid_until="2025-12-31", evidence=(_ref(doc, "CoA owns SubB valid 2025."),), structural_reliability=0.95)
    e_b = ClaimNode(claim_id="e2", kind="extraction", subject="SubB", predicate="operates_facility", value="PlantC", valid_from="2025-06-01", valid_until="2025-12-31", evidence=(_ref(doc, "SubB operates PlantC."),), structural_reliability=0.95)
    e_ctrl = ClaimNode(claim_id="ctrl", kind="derivation", subject="CoA", predicate="indirectly_controls", value="PlantC")
    s_rel = InferenceStep(step_id="s_rel", premises=("e1", "e2"), conclusion_id="ctrl", operation="relation_composition")
    res_rel = evaluate_response(
        JustificationGraph(claims={"e1": e_a, "e2": e_b, "ctrl": e_ctrl}, steps=(s_rel,), conclusion_id="ctrl"),
        evidence={"d_mix": doc},
        domain_profile=profile,
    )
    assert res_rel.assessment == "supported"

    p_cap = ClaimNode(claim_id="p_cap", kind="extraction", predicate="within_cap", value=True, polarity=1, evidence=(_ref(doc, "Cap A is 100 USD."),), structural_reliability=0.95)
    c_elig = ClaimNode(claim_id="c_elig", kind="derivation", predicate="payout_eligible", value=True)
    s_log = InferenceStep(step_id="s_log", premises=("p_cap",), conclusion_id="c_elig", operation="explicit_logic", rule_id="rule_cap_compliant", rule_version="2.0")
    res_log = evaluate_response(
        JustificationGraph(claims={"p_cap": p_cap, "c_elig": c_elig}, steps=(s_log,), conclusion_id="c_elig"),
        evidence={"d_mix": doc},
        domain_profile=profile,
    )
    assert res_log.assessment == "supported"

    # Superseded clause or document authority check -> REFUTED
    sup_claim = ClaimNode(
        claim_id="sup", kind="extraction", subject="CoA", predicate="within_cap", value=True,
        evidence=(_ref(doc, "Cap A is 100 USD."),), structural_reliability=0.95, metadata={"superseded_by": "amend-2"},
    )
    res_sup = evaluate_response(
        JustificationGraph(claims={"sup": sup_claim}, steps=(), conclusion_id="sup"),
        evidence={"d_mix": doc},
        domain_profile=profile,
    )
    assert res_sup.assessment == "refuted"
    assert authorize(res_sup).allowed is False


def test_open_semantics_always_probabilistic_and_blocked_for_autonomous_critical_steps() -> None:
    doc = _doc("d_txt", "The supplier will endeavor to deliver promptly.", weight=0.95)
    prem = ClaimNode(
        claim_id="p1", kind="extraction", text="The supplier will endeavor to deliver promptly.",
        evidence=(_ref(doc, "The supplier will endeavor to deliver promptly."),), structural_reliability=0.95,
    )
    conc = ClaimNode(claim_id="c1", kind="derivation", text="Reasonable efforts standard applies.")
    step = InferenceStep(step_id="sem_step", premises=("p1",), conclusion_id="c1", operation="open_semantics")
    graph = JustificationGraph(claims={"p1": prem, "c1": conc}, steps=(step,), conclusion_id="c1")

    def _mock_llm_sem(**kwargs):
        return {"status": "verified", "score": 0.88, "reasons": ["LLM_ENTAILMENT_MATCH"]}

    profile = DomainProfile(
        profile_id="sem-test",
        semantic_evaluator=SemanticEvaluatorConfig(
            evaluator_id="mock-judge", version="0.1", enabled=True, callable_evaluator=_mock_llm_sem
        ),
    )

    # With budget max_semantic_calls=1, open_semantics executes and strictly retains probabilistic_model label
    assessment = evaluate_response(
        graph,
        evidence={"d_txt": doc},
        domain_profile=profile,
        budget=VerificationBudget(max_semantic_calls=1),
    )
    sem_checks = [c for c in assessment.completed_checks if c.family == "open_semantics"]
    assert len(sem_checks) == 1
    assert sem_checks[0].guarantee_kind == "probabilistic_model"
    assert sem_checks[0].status == "verified"

    # Operational authorization refuses PASS because critical step relies solely on probabilistic model
    verdict = authorize(assessment, operation="trigger_legal_notice", policy=profile)
    assert verdict.allowed is False
    assert verdict.status == "ESCALATE"
    assert any(i.code == "PROBABILISTIC_ONLY_CRITICAL_STEP" for i in verdict.issues)


# ---------------------------------------------------------------------------
# Phase 3: Anti-Inflation by Correlation & Conflict vs Ignorance Separation
# ---------------------------------------------------------------------------

def test_correlated_paths_and_reformulations_never_inflate_support_while_independent_sources_do() -> None:
    doc1 = _doc("doc1", "Primary notice window is 30 days. Confirmed 30 day notice. Section 4 says thirty days.", weight=0.80)
    doc2 = _doc("doc2", "Independent audit schedule confirms notice window is 30 days.", weight=0.75)

    p1a = ClaimNode(claim_id="p1a", kind="extraction", subject="Vendor", predicate="notice", value=30, unit="calendar_days", evidence=(_ref(doc1, "Primary notice window is 30 days."),), structural_reliability=1.0)
    p1b = ClaimNode(claim_id="p1b", kind="extraction", subject="Vendor", predicate="notice", value=30, unit="calendar_days", evidence=(_ref(doc1, "Confirmed 30 day notice."),), structural_reliability=1.0)
    p1c = ClaimNode(claim_id="p1c", kind="extraction", subject="Vendor", predicate="notice", value=30, unit="calendar_days", evidence=(_ref(doc1, "Section 4 says thirty days."),), structural_reliability=1.0)
    p2 = ClaimNode(claim_id="p2", kind="extraction", subject="Vendor", predicate="notice", value=30, unit="calendar_days", evidence=(_ref(doc2, "Independent audit schedule confirms notice window is 30 days."),), structural_reliability=1.0)

    # Single path from doc1
    conc1 = ClaimNode(claim_id="c", kind="derivation", subject="Vendor", predicate="notice", value=30, unit="calendar_days")
    s_single = InferenceStep(
        step_id="s1a", premises=("p1a",), conclusion_id="c", operation="comparison",
        parameters={"op": "eq", "threshold": 30, "threshold_unit": "calendar_days"},
    )
    res_single = evaluate_response(
        JustificationGraph(claims={"p1a": p1a, "c": conc1}, steps=(s_single,), conclusion_id="c"),
        evidence={"doc1": doc1},
    )

    # Three correlated reformulations all citing doc1 -> EXACT SAME support as single path!
    s_1b = InferenceStep(step_id="s1b", premises=("p1b",), conclusion_id="c", operation="comparison", parameters={"op": "eq", "threshold": 30, "threshold_unit": "calendar_days"})
    s_1c = InferenceStep(step_id="s1c", premises=("p1c",), conclusion_id="c", operation="comparison", parameters={"op": "eq", "threshold": 30, "threshold_unit": "calendar_days"})
    res_corr = evaluate_response(
        JustificationGraph(claims={"p1a": p1a, "p1b": p1b, "p1c": p1c, "c": conc1}, steps=(s_single, s_1b, s_1c), conclusion_id="c"),
        evidence={"doc1": doc1},
    )
    assert res_corr.evidence_support_index == res_single.evidence_support_index
    assert res_corr.supporting_lineage_count == 1

    # Adding truly independent doc2 MUST increase support via Noisy-OR (unless capped above 0.80)
    s_indep = InferenceStep(step_id="s2", premises=("p2",), conclusion_id="c", operation="comparison", parameters={"op": "eq", "threshold": 30, "threshold_unit": "calendar_days"})
    res_indep = evaluate_response(
        JustificationGraph(claims={"p1a": p1a, "p2": p2, "c": conc1}, steps=(s_single, s_indep), conclusion_id="c"),
        evidence={"doc1": doc1, "doc2": doc2},
    )
    assert res_indep.supporting_lineage_count == 2
    assert res_indep.evidence_support_index > res_single.evidence_support_index


def test_conflict_vs_ignorance_separation_and_optional_calibration() -> None:
    docA = _doc("dA", "Cure period is 30 calendar days.", weight=0.85)
    docB = _doc("dB", "Cure period is 60 calendar days.", weight=0.85)

    conc = ClaimNode(
        claim_id="cA", kind="extraction", subject="Orion", predicate="cure_period", value=30, unit="calendar_days",
        evidence=(_ref(docA, "Cure period is 30 calendar days."),), structural_reliability=1.0,
    )
    other = ClaimNode(
        claim_id="cB", kind="extraction", subject="Orion", predicate="cure_period", value=60, unit="calendar_days",
        evidence=(_ref(docB, "Cure period is 60 calendar days."),), structural_reliability=1.0,
    )
    graph_contested = JustificationGraph(claims={"cA": conc, "cB": other}, steps=(), conclusion_id="cA")
    res_contested = evaluate_response(graph_contested, evidence={"dA": docA, "dB": docB})

    assert res_contested.assessment == "contested"
    assert res_contested.conflict_index > 0.3
    assert res_contested.calibrated_correctness_probability is None
    assert authorize(res_contested).allowed is False

    # Pure ignorance (no evidence) has conflict=0, uncertainty=1.0
    ungrounded = ClaimNode(claim_id="u1", kind="extraction", subject="Orion", predicate="cure_period", value=30)
    res_ignorance = evaluate_response(JustificationGraph(claims={"u1": ungrounded}, steps=(), conclusion_id="u1"))
    assert res_ignorance.conflict_index == 0.0
    assert res_ignorance.uncertainty_index == 1.0
    assert res_ignorance.assessment == "inconclusive"

    # Empirical calibration bins opt-in only
    cal_profile = DomainProfile(
        profile_id="cal",
        calibration=EmpiricalCalibration(calibration_bins=((0.0, 1.0, 0.77),), calibration_dataset_id="holdout-v1"),
    )
    assert evaluate_response(graph_contested, evidence={"dA": docA, "dB": docB}, domain_profile=cal_profile).calibrated_correctness_probability == 0.77


# ---------------------------------------------------------------------------
# Phase 4: Content-Addressed Incremental Cache & Budgets + Layer Integration
# ---------------------------------------------------------------------------

def test_incremental_cache_recomputes_only_changed_branch_and_enforces_budgets() -> None:
    docA_v1 = _doc("docA", "Branch A metric is 40 USD.", weight=0.95)
    docB_v1 = _doc("docB", "Branch B metric is 60 USD.", weight=0.95)

    def _build_graph(da: SourceDocument, db: SourceDocument, valA: int, val_sum: int) -> JustificationGraph:
        la = ClaimNode(claim_id="la", kind="extraction", subject="A", predicate="cost", value=valA, unit="USD", evidence=(_ref(da, f"Branch A metric is {valA} USD."),), structural_reliability=1.0)
        da_node = ClaimNode(claim_id="da_out", kind="derivation", subject="A", predicate="doubled", value=valA, unit="USD")
        lb = ClaimNode(claim_id="lb", kind="extraction", subject="B", predicate="cost", value=60, unit="USD", evidence=(_ref(db, "Branch B metric is 60 USD."),), structural_reliability=1.0)
        db_node = ClaimNode(claim_id="db_out", kind="derivation", subject="B", predicate="doubled", value=60, unit="USD")
        tot = ClaimNode(claim_id="tot", kind="derivation", subject="All", predicate="total", value=val_sum, unit="USD")

        sa = InferenceStep(step_id="sa", premises=("la",), conclusion_id="da_out", operation="unit_conversion")
        sb = InferenceStep(step_id="sb", premises=("lb",), conclusion_id="db_out", operation="unit_conversion")
        s_tot = InferenceStep(step_id="s_tot", premises=("da_out", "db_out"), conclusion_id="tot", operation="add")
        return JustificationGraph(
            claims={"la": la, "da_out": da_node, "lb": lb, "db_out": db_node, "tot": tot},
            steps=(sa, sb, s_tot),
            conclusion_id="tot",
        )

    cache = VerificationCache()
    g1 = _build_graph(docA_v1, docB_v1, 40, 100)
    r1 = evaluate_response(g1, evidence={"docA": docA_v1, "docB": docB_v1}, cache=cache)
    assert r1.assessment == "supported"
    assert r1.cache_metrics["cache_misses"] == 5  # 2 leaves + 3 steps
    assert r1.cache_metrics["cache_hits"] == 0

    # Run 2 identical -> 100% cache hit, 0 recomputed nodes!
    r2 = evaluate_response(g1, evidence={"docA": docA_v1, "docB": docB_v1}, cache=cache)
    assert r2.cache_metrics["cache_hits"] == 5
    assert r2.cache_metrics["recomputed_nodes"] == 0

    # Run 3: ONLY docA changes to 50 USD; docB branch (lb + sb) MUST hit cache!
    docA_v2 = _doc("docA", "Branch A metric is 50 USD.", weight=0.95)
    g2 = _build_graph(docA_v2, docB_v1, 50, 110)
    r3 = evaluate_response(g2, evidence={"docA": docA_v2, "docB": docB_v1}, cache=cache)
    assert r3.assessment == "supported"
    assert r3.cache_metrics["reused_nodes"] == 2  # lb and sb reused without recomputation!
    assert r3.cache_metrics["recomputed_nodes"] == 3  # la, sa, s_tot recomputed!

    # Tight budget (max_steps=1) defers remaining steps with PENDING_BUDGET and blocks authorize()
    r_budget = evaluate_response(g2, evidence={"docA": docA_v2, "docB": docB_v1}, budget=VerificationBudget(max_steps=1))
    assert r_budget.cache_metrics["budget_deferred_steps"] == 2
    assert r_budget.verification_coverage < 1.0
    assert len(r_budget.critical_unverified_steps) == 2
    assert any("PENDING_BUDGET" in o["reasons"] for o in r_budget.pending_obligations)
    assert authorize(r_budget).allowed is False


def test_authorize_binds_to_evidence_layer_hmac_and_three_phase_execution(tmp_path) -> None:
    vocab = Vocabulary(
        name="b2b-exec",
        version="1.0",
        predicates=(PredicateSpec("cap", FieldSpec("quantity", ("currency:USD",))),),
        entities=(("Acme", ("Acme Corp",)),),
    )
    layer = EvidenceLayer(tmp_path / "gate.sqlite", vocabulary=vocab)
    doc = SourceDocument(document_id="msa", text="Acme liability cap is 500 USD. Bonus addition is 100 USD.", authority_rank=90, weight=0.95)
    layer.ingest(doc)

    prop = {
        "entity": "Acme",
        "predicate": "cap",
        "value": {"value": 500, "unit": "USD"},
        "unit": "USD",
        "polarity": "affirmed",
        "assertion_kind": "fact",
        "conditions": [],
        "scope": {},
        "valid_from": None,
        "valid_until": None,
    }
    ing = layer.ingest_units(
        [{"proposal": prop, "evidence": [_ref(doc, "Acme liability cap is 500 USD.")]}],
        attestor="test-reader",
        structural_reliability=0.95,
    )
    uid = ing["unit_ids"][0]

    profile = DomainProfile(profile_id="b2b-exec", vocabulary=vocab)
    leaf = ClaimNode(
        claim_id="cap_src", kind="extraction", subject="Acme", predicate="cap", value=500, unit="USD",
        evidence=(_ref(doc, "Acme liability cap is 500 USD."),), evidence_unit_ids=(uid,), structural_reliability=0.95,
    )
    conc = ClaimNode(claim_id="cap_adj", kind="derivation", subject="Acme", predicate="adjusted_cap", value=600, unit="USD")
    bonus = ClaimNode(
        claim_id="b100", kind="extraction", subject="Acme", predicate="bonus", value=100, unit="USD",
        evidence=(_ref(doc, "Bonus addition is 100 USD."),), structural_reliability=0.95,
    )
    step = InferenceStep(step_id="add_cap", premises=("cap_src", "b100"), conclusion_id="cap_adj", operation="add")
    graph = JustificationGraph(claims={"cap_src": leaf, "b100": bonus, "cap_adj": conc}, steps=(step,), conclusion_id="cap_adj")

    assessment = evaluate_response(graph, evidence=layer, domain_profile=profile)
    assert assessment.assessment == "supported"

    payload = {"account": "acme", "limit": 600}
    verdict = authorize(
        assessment,
        operation="set_limit",
        policy=profile,
        layer=layer,
        payload=payload,
        signing_key="secret-hmac-key",
    )
    assert verdict.status == "PASS"
    assert verify_authorization_verdict(
        verdict, layer=layer, operation_id="set_limit", payload=payload, signing_key="secret-hmac-key"
    ) is True

    executed = []
    out = execute_authorized_operation(
        layer,
        verdict,
        lambda v: (executed.append(v.operation_id), "ok")[1],
        operation_id="set_limit",
        payload=payload,
        signing_key="secret-hmac-key",
    )
    assert out == "ok"
    assert executed == ["set_limit"]


def test_vocabulary_leaf_extraction_fidelity_catches_relational_mismatch() -> None:
    vocab = Vocabulary(
        name="b2b-rel",
        version="1.0",
        predicates=(PredicateSpec("cap", FieldSpec("quantity", ("currency:USD",))),),
        entities=(("Acme", ("Acme",)), ("Beta", ("Beta",))),
    )
    profile = DomainProfile(profile_id="rel-check", vocabulary=vocab)
    doc = _doc("d_rel", "Acme liability cap is 500 USD and Beta fee is 900 USD.", weight=1.0)
    # Hallucinated cross-entity association: assigns Beta's 900 USD fee as Acme's cap
    bad_leaf = ClaimNode(
        claim_id="bad_cap", kind="extraction", subject="Acme", predicate="cap", value=900, unit="USD",
        evidence=(_ref(doc, "Acme liability cap is 500 USD and Beta fee is 900 USD."),), structural_reliability=1.0,
    )
    assessment = evaluate_response(
        JustificationGraph(claims={"bad_cap": bad_leaf}, steps=(), conclusion_id="bad_cap"),
        evidence={"d_rel": doc},
        domain_profile=profile,
    )
    assert assessment.assessment == "refuted"
    assert any(c.status == "refuted" for c in assessment.completed_checks)
    assert authorize(assessment, policy=profile).allowed is False
