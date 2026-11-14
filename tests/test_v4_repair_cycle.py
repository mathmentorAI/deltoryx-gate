"""Comprehensive regression and adversarial tests for ClaimLayer v4 evaluation,
mandatory engine obligations, response-to-graph correspondence, and bounded repair cycles.
"""
from __future__ import annotations

import pytest

from claimlayer.semantic.adapters import OpenAICompatibleModel
from claimlayer.v4 import (
    ClaimNode,
    DomainProfile,
    EvidenceLayer,
    EvidenceRef,
    FieldSpec,
    InferenceStep,
    JustificationGraph,
    LLMGraphProposalAdapter,
    LLMRepairAgent,
    OBLIGATION_GENERATOR_VERSION,
    PredicateSpec,
    RepairBudget,
    RepairOrchestrator,
    RuleGuidedRepairAdapter,
    ScriptedRepairAgent,
    SemanticEvaluatorConfig,
    SourceDocument,
    VerificationObligation,
    Vocabulary,
    authorize,
    evaluate_response,
    review_response,
    verify_response_equations,
    verify_response_graph_correspondence,
    verify_review_freshness,
)


def _doc(doc_id: str, text: str, *, weight: float = 0.95, rank: int = 80, attribution: str = "source_statement") -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        authority_rank=rank,
        weight=weight,
        attribution=attribution,
    )


def _ref(doc: SourceDocument, quote: str) -> EvidenceRef:
    start = doc.text.index(quote)
    return EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=start + len(quote),
        quote=quote,
    )


def _build_anchor_age_graph(
    *,
    subject: str,
    anchor_age: int,
    anchor_year: int,
    reference_year: int,
    claimed_age: int,
    quote_age: str,
    quote_year: str,
    doc: SourceDocument,
    approximate: bool = False,
    predicate: str = "age",
    operation: str = "temporal",
    obligations: tuple[VerificationObligation, ...] = (),
    step_critical: bool = True,
) -> JustificationGraph:
    p_age = ClaimNode(
        claim_id="p_age",
        kind="extraction",
        subject=subject,
        predicate="anchor_age",
        value=anchor_age,
        unit="years",
        evidence=(_ref(doc, quote_age),),
        structural_reliability=0.95,
    )
    p_yr = ClaimNode(
        claim_id="p_yr",
        kind="extraction",
        subject=subject,
        predicate="anchor_date",
        value=anchor_year,
        unit="years",
        evidence=(_ref(doc, quote_year),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="conc_age",
        kind="derivation",
        subject=subject,
        predicate=predicate,
        value=claimed_age,
        unit="years",
        approximate=approximate,
    )
    step = InferenceStep(
        step_id="s_age",
        operation=operation,
        premises=("p_age", "p_yr"),
        conclusion_id="conc_age",
        obligations=obligations,
        parameters={
            "anchor_age": anchor_age,
            "anchor_date": str(anchor_year),
            "reference_date": str(reference_year),
        },
        critical=step_critical,
    )
    return JustificationGraph(
        claims={"p_age": p_age, "p_yr": p_yr, "conc_age": conc},
        steps=(step,),
        conclusion_id="conc_age",
    )


# ---------------------------------------------------------------------------
# Family 1: Incomplete dates / exact age -> RETRY -> qualified ACCEPT (generalized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "subject,text,anchor_age,anchor_year,ref_year,expected_age,q_age,q_year,bad_resp",
    [
        (
            "Warren Buffet",
            "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
            13,
            1933,
            2026,
            106,
            "tenía 13 años",
            "era 1933",
            "Tiene 106 años.",
        ),
        (
            "Elena Torres",
            "Elena Torres obtuvo su primera licencia profesional cuando tenía 20 años y era 2005.",
            20,
            2005,
            2026,
            41,
            "tenía 20 años",
            "era 2005",
            "Tiene 41 años.",
        ),
    ],
)
def test_01_temporal_exact_age_incomplete_dates_retry_then_accept(
    subject: str,
    text: str,
    anchor_age: int,
    anchor_year: int,
    ref_year: int,
    expected_age: int,
    q_age: str,
    q_year: str,
    bad_resp: str,
) -> None:
    doc = _doc("d_bio", text, weight=0.95)
    graph = _build_anchor_age_graph(
        subject=subject,
        anchor_age=anchor_age,
        anchor_year=anchor_year,
        reference_year=ref_year,
        claimed_age=expected_age,
        quote_age=q_age,
        quote_year=q_year,
        doc=doc,
        approximate=False,
    )
    question = f"¿Cuántos años tiene si estamos en {ref_year}?"

    init_review = review_response(
        question, bad_resp, {"d_bio": doc}, graph, authorize_operation_id="approve_age"
    )
    assert init_review.disposition == "RETRY"
    issue_codes = {i.code for i in init_review.issues}
    assert "UNJUSTIFIED_TEMPORAL_PRECISION" in issue_codes
    assert "SOURCE_CONDITIONAL_CONCLUSION" in issue_codes
    assert init_review.required_repairs
    assert any(
        "inventing_missing_dates" in rep.forbidden_shortcuts
        for rep in init_review.required_repairs
    )
    assert init_review.operational_authorization is not None
    assert init_review.operational_authorization.allowed is False
    assert init_review.assessment.calibrated_correctness_probability is None

    # Full bounded repair cycle converges cleanly using generic RuleGuidedRepairAdapter
    orch = RepairOrchestrator(authorize_operation_id="approve_age")
    cycle = orch.run(
        question=question,
        initial_response=bad_resp,
        evidence={"d_bio": doc},
        graph_or_adapter=graph,
        repair_agent=RuleGuidedRepairAdapter(),
    )
    assert cycle.accepted is True
    assert cycle.final_disposition == "ACCEPT"
    assert cycle.stop_reason == "accepted"
    assert cycle.iterations_used == 2
    assert cycle.final_review.assessment.obligation_generator_version == OBLIGATION_GENERATOR_VERSION
    assert cycle.final_review.assessment.response_correspondence_verified is True
    assert cycle.final_review.assessment.verification_coverage == 1.0
    # Exploratory ACCEPT stays strictly separate from operational PASS!
    assert cycle.operational_authorization is not None
    assert cycle.operational_authorization.allowed is False
    assert cycle.operational_authorization.status == "ESCALATE"
    assert any(
        i.code == "EXPLORATORY_QUALIFIED_CONCLUSION_NOT_OPERATIONAL"
        for i in cycle.operational_authorization.issues
    )


# ---------------------------------------------------------------------------
# Family 2: Narrative unattributed source -> RETRY -> attributed ACCEPT
# ---------------------------------------------------------------------------

def test_02_narrative_unattributed_source_retry_then_accept() -> None:
    doc = _doc(
        "d_chron",
        "Meridian Solar reportó ingresos base de 120 USD y un recargo adicional de 30 USD en la nota interna.",
        weight=0.92,
    )
    p1 = ClaimNode(
        claim_id="base_rev",
        kind="extraction",
        subject="Meridian Solar",
        predicate="base_revenue",
        value=120,
        unit="USD",
        evidence=(_ref(doc, "ingresos base de 120 USD"),),
        structural_reliability=0.95,
    )
    p2 = ClaimNode(
        claim_id="extra_rev",
        kind="extraction",
        subject="Meridian Solar",
        predicate="extra_revenue",
        value=30,
        unit="USD",
        evidence=(_ref(doc, "recargo adicional de 30 USD"),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="combined_rev",
        kind="derivation",
        subject="Meridian Solar",
        predicate="combined_revenue",
        value=150,
        unit="USD",
    )
    step = InferenceStep(step_id="s_add", operation="add", premises=("base_rev", "extra_rev"), conclusion_id="combined_rev")
    graph = JustificationGraph(claims={"base_rev": p1, "extra_rev": p2, "combined_rev": conc}, steps=(step,), conclusion_id="combined_rev")

    question = "¿Cuál es el ingreso combinado de Meridian Solar?"
    bad_unattributed = "Es un hecho histórico verificado de forma independiente que Meridian Solar tiene ingresos de 150 USD."
    good_attributed = "Según el texto aportado, el ingreso combinado de Meridian Solar es 150 USD (no verificado externamente)."

    r1 = review_response(question, bad_unattributed, {"d_chron": doc}, graph)
    assert r1.disposition == "RETRY"
    assert any(i.code in ("SOURCE_CONDITIONAL_CONCLUSION", "CONTRADICTORY_SOURCE_ATTRIBUTION") for i in r1.issues)

    # Contradictory attribution ("Según el texto ... hecho histórico verificado") also rejected
    contradictory = "Según el texto aportado, los ingresos de Meridian Solar son 150 USD como hecho histórico verificado."
    r_mid = review_response(question, contradictory, {"d_chron": doc}, graph)
    assert r_mid.disposition == "RETRY"
    assert any(i.code == "CONTRADICTORY_SOURCE_ATTRIBUTION" for i in r_mid.issues)

    r2 = review_response(question, good_attributed, {"d_chron": doc}, graph)
    assert r2.disposition == "ACCEPT"
    assert r2.issues == ()


# ---------------------------------------------------------------------------
# Family 3: Contradictory precision qualification rejected
# ---------------------------------------------------------------------------

def test_03_contradictory_precision_qualification_rejected() -> None:
    doc = _doc("d_bio", "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=True,
    )
    question = "¿Cuántos años tiene si estamos en 2026?"

    fake_hedged = "Según el texto, tendría aproximadamente 106 años en 2026; por tanto, sabemos que tiene exactamente 106 años."
    rv1 = review_response(question, fake_hedged, {"d_bio": doc}, graph)
    assert rv1.disposition == "RETRY"
    assert any(i.code == "CONTRADICTORY_PRECISION_QUALIFICATION" for i in rv1.issues)

    wrong_clause_hedging = "En 2026 aproximadamente, sabemos con exactitud que tiene 106 años según el texto."
    rv2 = review_response(question, wrong_clause_hedging, {"d_bio": doc}, graph)
    assert rv2.disposition == "RETRY"
    assert any(i.code == "CONTRADICTORY_PRECISION_QUALIFICATION" for i in rv2.issues)


# ---------------------------------------------------------------------------
# Family 4: Unsupported factual addition rejected
# ---------------------------------------------------------------------------

def test_04_unsupported_factual_addition_rejected() -> None:
    doc = _doc("d_bio", "Clara Gómez inició su primer contrato cuando tenía 25 años y era 2010.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Clara Gómez",
        anchor_age=25,
        anchor_year=2010,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 25 años",
        quote_year="era 2010",
        doc=doc,
        approximate=True,
    )
    invented = (
        "Según el texto sobre Clara Gómez, nació el 1985-08-30, por lo que tendría aproximadamente 41 años en 2026. "
        "Sin fechas completas no puede determinarse con certeza."
    )
    rv = review_response("¿Cuántos años tiene en 2026?", invented, {"d_bio": doc}, graph)
    assert rv.disposition == "RETRY"
    assert any(i.code == "UNSUPPORTED_FACTUAL_ADDITION" for i in rv.issues)


# ---------------------------------------------------------------------------
# Family 5: Subject swap, calculation alteration, or unit change rejected
# ---------------------------------------------------------------------------

def test_05_subject_calculation_or_unit_mutation_rejected() -> None:
    doc = _doc("d_bio", "Clara Gómez inició su primer contrato cuando tenía 25 años y era 2010. Marcos Silva firmó en 2012.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Clara Gómez",
        anchor_age=25,
        anchor_year=2010,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 25 años",
        quote_year="era 2010",
        doc=doc,
        approximate=True,
    )
    question = "¿Cuántos años tiene Clara Gómez en 2026?"

    swapped_subject = "Según el texto, Marcos Silva tendría aproximadamente 41 años en 2026 (no hay fechas completas)."
    rv_subj = review_response(question, swapped_subject, {"d_bio": doc}, graph)
    assert rv_subj.disposition == "RETRY"
    assert any(i.code == "SUBJECT_CHANGED_IN_REPAIR" for i in rv_subj.issues)

    altered_calc = "Según el texto, Clara Gómez tendría aproximadamente 44 años en 2026 (no hay fechas completas)."
    rv_calc = review_response(question, altered_calc, {"d_bio": doc}, graph)
    assert rv_calc.disposition == "RETRY"
    assert any(i.code in ("CALCULATION_ALTERED_IN_RESPONSE", "CALCULATION_ALTERED_OR_INCONSISTENT") for i in rv_calc.issues)

    swapped_unit = "Según el texto, Clara Gómez tendría aproximadamente 41 meses en 2026 (no hay fechas completas)."
    rv_unit = review_response(question, swapped_unit, {"d_bio": doc}, graph)
    assert rv_unit.disposition == "RETRY"
    assert any(i.code in ("UNIT_CHANGED_IN_RESPONSE", "UNIT_ALTERED_IN_REPAIR") for i in rv_unit.issues)


# ---------------------------------------------------------------------------
# Family 6: Condition dropped -> RETRY -> condition preserved ACCEPT
# ---------------------------------------------------------------------------

def test_06_condition_preservation_retry_then_accept() -> None:
    doc = _doc(
        "d_pol",
        "El cliente tiene derecho a un reembolso de 250 USD, salvo en caso de mora previa.",
        weight=0.95,
    )
    p = ClaimNode(
        claim_id="p_ref",
        kind="extraction",
        subject="Cliente",
        predicate="refund",
        value=250,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de mora previa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "reembolso de 250 USD, salvo en caso de mora previa"),),
        structural_reliability=0.95,
    )
    conc_missing = ClaimNode(
        claim_id="c_ref",
        kind="derivation",
        subject="Cliente",
        predicate="approved_refund",
        value=250,
        unit="USD",
        conditions=(),
        condition_status="unconditional",
    )
    step = InferenceStep(step_id="s_ref", operation="quantities", premises=("p_ref",), conclusion_id="c_ref")
    graph_missing = JustificationGraph(claims={"p_ref": p, "c_ref": conc_missing}, steps=(step,), conclusion_id="c_ref")

    # Even in graph_only mode, engine mandatory condition check blocks silent support
    ev_graph_only = evaluate_response(graph_missing, evidence={"d_pol": doc})
    assert ev_graph_only.assessment != "supported"
    assert any("DROPPED_REQUIRED_CONDITION" in c.reasons for c in ev_graph_only.completed_checks)

    conc_preserved = ClaimNode(
        claim_id="c_ref",
        kind="derivation",
        subject="Cliente",
        predicate="approved_refund",
        value=250,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de mora previa"},),
        condition_status="conditional",
    )
    graph_ok = JustificationGraph(claims={"p_ref": p, "c_ref": conc_preserved}, steps=(step,), conclusion_id="c_ref")

    bad_unconditional = "Según el texto, al Cliente le corresponde un reembolso de 250 USD."
    rv_bad = review_response("¿Qué reembolso aplica al Cliente?", bad_unconditional, {"d_pol": doc}, graph_ok)
    assert rv_bad.disposition == "RETRY"
    assert any(i.code == "CONDITION_DROPPED_IN_RESPONSE" for i in rv_bad.issues)

    good_cond = "Según el texto, al Cliente le corresponde un reembolso de 250 USD, salvo en caso de mora previa."
    rv_ok = review_response("¿Qué reembolso aplica al Cliente?", good_cond, {"d_pol": doc}, graph_ok)
    assert rv_ok.disposition == "ACCEPT"


# ---------------------------------------------------------------------------
# Family 7: Subtotal vs total without complete population
# ---------------------------------------------------------------------------

def test_07_subtotal_vs_total_incomplete_population_retry_then_accept() -> None:
    doc = _doc("d_inv", "Factura 1 es 40 USD. Factura 2 es 60 USD.", weight=0.95)
    i1 = ClaimNode(claim_id="i1", kind="extraction", subject="Factura 1", predicate="charge", value=40, unit="USD", event_id="evt-1", evidence=(_ref(doc, "Factura 1 es 40 USD."),), structural_reliability=0.95)
    i2 = ClaimNode(claim_id="i2", kind="extraction", subject="Factura 2", predicate="charge", value=60, unit="USD", event_id="evt-2", evidence=(_ref(doc, "Factura 2 es 60 USD."),), structural_reliability=0.95)
    conc = ClaimNode(claim_id="tot", kind="derivation", subject="Facturas", predicate="total_amount", value=100, unit="USD")
    step = InferenceStep(step_id="s_agg", operation="agg_sum", premises=("i1", "i2"), conclusion_id="tot")
    graph = JustificationGraph(claims={"i1": i1, "i2": i2, "tot": conc}, steps=(step,), conclusion_id="tot")

    q = "¿Cuál es el total de facturas?"
    bad_total = "Según el texto, el total completo definitivo es 100 USD."
    rv_bad = review_response(q, bad_total, {"d_inv": doc}, graph, authorize_operation_id="pay_invoices")
    assert rv_bad.disposition == "RETRY"
    assert any(i.code in ("SUBTOTAL_PRESENTED_AS_TOTAL", "UNQUALIFIED_SUBTOTAL_CLAIM", "UNVERIFIED_POPULATION_COMPLETENESS") for i in rv_bad.issues)
    assert rv_bad.operational_authorization is not None and rv_bad.operational_authorization.allowed is False

    good_subtotal = "Según el texto, el subtotal parcial de las facturas listadas es 100 USD, sin constancia de completitud de toda la población."
    rv_ok = review_response(q, good_subtotal, {"d_inv": doc}, graph, authorize_operation_id="pay_invoices")
    assert rv_ok.disposition == "ACCEPT"
    # Operational payment remains blocked for an unverified-population subtotal
    assert rv_ok.operational_authorization is not None
    assert rv_ok.operational_authorization.allowed is False
    assert rv_ok.operational_authorization.status == "ESCALATE"


# ---------------------------------------------------------------------------
# Family 8: Material conflict hidden -> RETRY -> disclosed ACCEPT (authorize ESCALATE)
# ---------------------------------------------------------------------------

def test_08_material_conflict_hidden_retry_then_disclosed_accept_while_authorize_escalates() -> None:
    d1 = _doc("d1", "Anexo A indica que el límite de responsabilidad es 500 USD.", weight=0.95)
    d2 = _doc("d2", "Anexo B indica que el límite de responsabilidad es 800 USD.", weight=0.95)
    c1 = ClaimNode(claim_id="cap_a", kind="extraction", subject="Contrato", predicate="liability_cap", value=500, unit="USD", evidence=(_ref(d1, "límite de responsabilidad es 500 USD"),), structural_reliability=0.95)
    c2 = ClaimNode(claim_id="cap_b", kind="extraction", subject="Contrato", predicate="liability_cap", value=800, unit="USD", evidence=(_ref(d2, "límite de responsabilidad es 800 USD"),), structural_reliability=0.95)
    conc = ClaimNode(claim_id="cap_conc", kind="derivation", subject="Contrato", predicate="liability_cap", value=500, unit="USD")
    step = InferenceStep(step_id="s_id", operation="quantities", premises=("cap_a",), conclusion_id="cap_conc")
    graph = JustificationGraph(claims={"cap_a": c1, "cap_b": c2, "cap_conc": conc}, steps=(step,), conclusion_id="cap_conc")

    q = "¿Cuál es el límite de responsabilidad del Contrato?"
    bad_one_sided = "Según el documento, el límite de responsabilidad del Contrato es 500 USD."
    rv_bad = review_response(q, bad_one_sided, {"d1": d1, "d2": d2}, graph, authorize_operation_id="set_limit")
    assert rv_bad.disposition == "RETRY"
    assert any(i.code in ("CONFLICT_HIDDEN_IN_RESPONSE", "UNDISCLOSED_MATERIAL_CONFLICT") for i in rv_bad.issues)

    good_disclosed = "Según los textos, existe un conflicto sin resolver respecto al Contrato: Anexo A indica 500 USD pero Anexo B indica 800 USD."
    rv_ok = review_response(q, good_disclosed, {"d1": d1, "d2": d2}, graph, authorize_operation_id="set_limit")
    assert rv_ok.disposition == "ACCEPT"
    assert rv_ok.assessment.assessment == "contested"
    assert rv_ok.operational_authorization is not None
    assert rv_ok.operational_authorization.status == "ESCALATE"
    assert rv_ok.operational_authorization.allowed is False


# ---------------------------------------------------------------------------
# Family 9: Generic predicate anti-evasion registers AMBIGUOUS_OR_GENERIC_SEMANTIC_TYPE
# ---------------------------------------------------------------------------

def test_09_generic_predicate_cannot_evade_mandatory_obligations() -> None:
    doc = _doc("d_bio", "Elena Torres obtuvo su licencia cuando tenía 20 años y era 2005.", weight=0.95)
    # Adversarial graph uses predicate="number" and operation="add" with empty obligations=()!
    graph = _build_anchor_age_graph(
        subject="Elena Torres",
        anchor_age=20,
        anchor_year=2005,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 20 años",
        quote_year="era 2005",
        doc=doc,
        approximate=False,
        predicate="number",
        operation="add",
        obligations=(),
        step_critical=False,
    )
    rv = review_response("¿Cuántos años tiene en 2026?", "Tiene 41 años.", {"d_bio": doc}, graph)
    assert rv.disposition != "ACCEPT"
    assert rv.assessment.assessment != "supported"
    issue_codes = {i.code for i in rv.issues}
    # Mandatory temporal + semantic uncertainty obligations BOTH fired despite generic label!
    assert "AMBIGUOUS_OR_GENERIC_SEMANTIC_TYPE" in issue_codes
    assert "UNJUSTIFIED_TEMPORAL_PRECISION" in issue_codes or "SOURCE_CONDITIONAL_CONCLUSION" in issue_codes
    assert rv.assessment.mandatory_obligations


# ---------------------------------------------------------------------------
# Family 10: Question evasion rejected vs legitimate pruning accepted
# ---------------------------------------------------------------------------

def test_10_question_evasion_rejected_vs_legitimate_pruning_accepted() -> None:
    doc = _doc("d_bio", "Clara Gómez inició su contrato cuando tenía 25 años y era 2010.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Clara Gómez",
        anchor_age=25,
        anchor_year=2010,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 25 años",
        quote_year="era 2010",
        doc=doc,
        approximate=True,
    )
    q = "¿Cuántos años tiene Clara Gómez en 2026?"

    # Evasion: mentions unrelated 2010 fact or refuses to answer age in 2026
    evasive = "Según el texto sobre Clara Gómez, no puedo responder la edad y solo consta que fue en 2010."
    rv_ev = review_response(q, evasive, {"d_bio": doc}, graph)
    assert rv_ev.disposition == "RETRY"
    assert any(i.code in ("QUESTION_EVADED", "QUESTION_LEFT_UNANSWERED_EVASION") for i in rv_ev.issues)

    # Legitimate pruning: previous attempt had extra unsupported 1985 birth date; pruned repair keeps verified age cleanly
    pruned = (
        "Según el texto sobre Clara Gómez, tendría aproximadamente 41 años en 2026. "
        "No hay fechas completas para determinar una edad exacta."
    )
    rv_pruned = review_response(q, pruned, {"d_bio": doc}, graph)
    assert rv_pruned.disposition == "ACCEPT"


# ---------------------------------------------------------------------------
# Family 11: Wide temporal uncertainty (>2 units) rejects simple "aproximadamente"
# ---------------------------------------------------------------------------

def test_11_wide_temporal_interval_rejects_simple_approximation() -> None:
    # Year-only bounds measured in months -> uncertainty span (~24 months >> 2 months)!
    doc = _doc("d_span", "El proyecto inició en 2020 y cerró en 2026.", weight=0.95)
    p1 = ClaimNode(claim_id="start", kind="extraction", subject="Proyecto K", predicate="start_date", value="2020", evidence=(_ref(doc, "inició en 2020"),), structural_reliability=0.95)
    p2 = ClaimNode(claim_id="end", kind="extraction", subject="Proyecto K", predicate="end_date", value="2026", evidence=(_ref(doc, "cerró en 2026"),), structural_reliability=0.95)
    conc_point = ClaimNode(claim_id="dur", kind="derivation", subject="Proyecto K", predicate="duration", value=72, unit="months", approximate=True)
    step = InferenceStep(
        step_id="s_dur",
        operation="temporal_duration",
        premises=("start", "end"),
        conclusion_id="dur",
        parameters={"start_date": "2020", "end_date": "2026", "metric": "months", "approximate": True},
    )
    graph_point = JustificationGraph(claims={"start": p1, "end": p2, "dur": conc_point}, steps=(step,), conclusion_id="dur")

    q = "¿Cuántos meses duró Proyecto K?"
    simple_approx = "Según el texto sobre Proyecto K, duró aproximadamente 72 meses (sin fechas completas)."
    rv_wide = review_response(q, simple_approx, {"d_span": doc}, graph_point)
    assert rv_wide.disposition == "RETRY"
    assert any(i.code == "APPROXIMATION_HIDES_WIDE_UNCERTAINTY" for i in rv_wide.issues)

    # Explicit derived interval range is accepted
    conc_interval = ClaimNode(
        claim_id="dur",
        kind="derivation",
        subject="Proyecto K",
        predicate="duration",
        value={"min": 60, "max": 84},
        unit="months",
        approximate=True,
    )
    graph_iv = JustificationGraph(claims={"start": p1, "end": p2, "dur": conc_interval}, steps=(step,), conclusion_id="dur")
    explicit_range = "Según el texto sobre Proyecto K, duró entre 60 y 84 meses debido a que no hay fechas completas."
    rv_iv = review_response(q, explicit_range, {"d_span": doc}, graph_iv)
    assert rv_iv.disposition == "ACCEPT"


# ---------------------------------------------------------------------------
# Family 12: Wrong calculation NOT repairable merely by adding "aproximadamente"
# ---------------------------------------------------------------------------

def test_12_incorrect_calculation_not_repaired_by_hedging() -> None:
    doc = _doc("d_bio", "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.", weight=0.95)
    graph_wrong = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=109,  # Wrong calculation (derived interval is [105, 107])!
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=True,
    )
    hedged_wrong = "Según el texto, tendría aproximadamente 109 años en 2026 (sin fechas completas)."
    rv = review_response("¿Cuántos años tiene si estamos en 2026?", hedged_wrong, {"d_bio": doc}, graph_wrong)
    assert rv.disposition == "RETRY"
    assert any(i.code == "TEMPORAL_DERIVATION_CONTRADICTED" for i in rv.issues)
    assert any(
        "adding_approximation_to_wrong_number" in rep.forbidden_shortcuts
        for rep in rv.required_repairs
    )


# ---------------------------------------------------------------------------
# Family 13: Bounded orchestrator stops on budget, duplicates, or stagnation (never ACCEPT)
# ---------------------------------------------------------------------------

def test_13_bounded_orchestrator_stops_without_accepting_on_budget_duplicate_and_stagnation() -> None:
    doc = _doc("d_bio", "Elena Torres obtuvo su primera licencia cuando tenía 20 años y era 2005.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Elena Torres",
        anchor_age=20,
        anchor_year=2005,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 20 años",
        quote_year="era 2005",
        doc=doc,
        approximate=False,
    )
    q = "¿Cuántos años tiene en 2026?"

    # a) Max iterations exhausted with perpetually non-compliant responses
    cycle_budget = RepairOrchestrator(budget=RepairBudget(max_iterations=2, stop_on_stagnant_issues=False)).run(
        question=q,
        initial_response="Tiene 41 años.",
        evidence={"d_bio": doc},
        graph_or_adapter=graph,
        repair_agent=ScriptedRepairAgent(["Tiene exactamente 41 años.", "Aproximadamente 41 años como hecho histórico verificado."]),
    )
    assert cycle_budget.accepted is False
    assert cycle_budget.final_disposition == "ESCALATE"
    assert cycle_budget.stop_reason == "max_iterations_reached"
    assert cycle_budget.iterations_used == 2

    # b) Duplicate candidate response halts immediately
    cycle_dup = RepairOrchestrator(budget=RepairBudget(max_iterations=5, stop_on_duplicate_response=True)).run(
        question=q,
        initial_response="Tiene 41 años.",
        evidence={"d_bio": doc},
        graph_or_adapter=graph,
        repair_agent=ScriptedRepairAgent(["Tiene 41 años."]),
    )
    assert cycle_dup.accepted is False
    assert cycle_dup.final_disposition == "ESCALATE"
    assert cycle_dup.stop_reason == "duplicate_response"
    assert cycle_dup.iterations_used == 1

    # c) Cosmetic edits leaving exact same issues trigger stagnation stop
    cycle_stag = RepairOrchestrator(budget=RepairBudget(max_iterations=5, stop_on_stagnant_issues=True)).run(
        question=q,
        initial_response="Tiene 41 años.",
        evidence={"d_bio": doc},
        graph_or_adapter=graph,
        repair_agent=ScriptedRepairAgent([
            "Tiene exactamente 41 años.",
            "Tiene exactamente 41 años hoy.",
            "Tiene exactamente 41 años ahora.",
        ]),
    )
    assert cycle_stag.accepted is False
    assert cycle_stag.final_disposition == "ESCALATE"
    assert cycle_stag.stop_reason == "stagnant_issues"
    assert cycle_stag.iterations_used == 2


# ---------------------------------------------------------------------------
# Family 14: Cache reuse across repair iterations + source revision invalidation
# ---------------------------------------------------------------------------

def test_14_cache_reuse_across_repairs_and_source_revision_invalidation() -> None:
    doc = _doc("d_bio", "Elena Torres obtuvo su primera licencia cuando tenía 20 años y era 2005.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Elena Torres",
        anchor_age=20,
        anchor_year=2005,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 20 años",
        quote_year="era 2005",
        doc=doc,
        approximate=True,
    )
    orch = RepairOrchestrator()
    cycle = orch.run(
        question="¿Cuántos años tiene en 2026?",
        initial_response="Tiene 41 años.",
        evidence={"d_bio": doc},
        graph_or_adapter=graph,
        repair_agent=RuleGuidedRepairAdapter(),
    )
    assert cycle.accepted is True
    assert cycle.iterations_used == 2
    assert cycle.attempts[0].cache_metrics["cache_misses"] == 3  # 2 leaves + 1 step
    assert cycle.attempts[1].cache_metrics["cache_hits"] == 2  # both premise leaves reused!
    assert cycle.attempts[1].cache_metrics["reused_nodes"] == 2
    assert cycle.attempts[1].cache_metrics["recomputed_nodes"] == 1  # only conclusion step re-evaluated

    # Mutating source document revision invalidates cached leaves
    doc_v2 = _doc("d_bio", "Elena Torres obtuvo su primera licencia cuando tenía 21 años y era 2005.", weight=0.95)
    rv_after_doc_change = review_response(
        "¿Cuántos años tiene en 2026?",
        cycle.final_response,
        {"d_bio": doc_v2},
        graph,
        cache=orch.cache,
    )
    assert rv_after_doc_change.assessment.cache_metrics["cache_hits"] < 2


# ---------------------------------------------------------------------------
# Family 15: Post-approval digest invalidation & operational authorize() separation
# ---------------------------------------------------------------------------

def test_15_post_approval_mutation_invalidates_freshness_and_separates_authorize() -> None:
    doc = _doc("d_bio", "Elena Torres obtuvo su primera licencia cuando tenía 20 años y era 2005.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Elena Torres",
        anchor_age=20,
        anchor_year=2005,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 20 años",
        quote_year="era 2005",
        doc=doc,
        approximate=True,
    )
    q = "¿Cuántos años tiene en 2026?"
    approved_text = "Según el texto sobre Elena Torres, tendría aproximadamente 41 años en 2026. No hay fechas completas para determinar una edad exacta."

    approved = review_response(q, approved_text, {"d_bio": doc}, graph, authorize_operation_id="execute_payout")
    assert approved.disposition == "ACCEPT"
    assert verify_review_freshness(approved, approved_text, q) is True

    # Operational authorization still blocks execution!
    assert approved.operational_authorization is not None
    assert approved.operational_authorization.allowed is False
    assert approved.operational_authorization.status == "ESCALATE"

    # Mutating even one word ("aproximadamente" -> "exactamente") invalidates freshness immediately
    tampered_text = approved_text.replace("aproximadamente ", "")
    assert verify_review_freshness(approved, tampered_text, q) is False
    assert verify_review_freshness(approved, approved_text, "¿Cuántos años tiene en 2030?") is False


# ---------------------------------------------------------------------------
# Family 16: Real adapter contracts (LLMRepairAgent & LLMGraphProposalAdapter via stub transport)
# ---------------------------------------------------------------------------

def test_16_llm_repair_agent_and_graph_adapter_contract() -> None:
    class _StubModel(OpenAICompatibleModel):
        def __init__(self, responses):
            super().__init__(base_url="http://stub.local/v1", model="stub-model", api_key="stub")
            object.__setattr__(self, "_responses", list(responses))
            object.__setattr__(self, "seen_payloads", [])

        def complete(self, system, payload):  # type: ignore[override]
            self.seen_payloads.append((system, payload))
            return {"output": self._responses.pop(0), "usage": {"total_tokens": 42}}

    doc = _doc("d_bio", "Elena Torres obtuvo su primera licencia cuando tenía 20 años y era 2005.", weight=0.95)
    graph = _build_anchor_age_graph(
        subject="Elena Torres",
        anchor_age=20,
        anchor_year=2005,
        reference_year=2026,
        claimed_age=41,
        quote_age="tenía 20 años",
        quote_year="era 2005",
        doc=doc,
        approximate=True,
    )

    # Test LLMGraphProposalAdapter extraction path ("adapter_extracted_graph")
    stub_extractor = _StubModel([graph.to_dict(), graph.to_dict()])
    graph_adapter = LLMGraphProposalAdapter(stub_extractor)

    stub_repair_model = _StubModel([
        {
            "repaired_response": (
                "Según el texto sobre Elena Torres, tendría aproximadamente 41 años en 2026. "
                "Sin fechas completas no puede determinarse una edad exacta."
            )
        }
    ])
    llm_repairer = LLMRepairAgent(stub_repair_model)

    cycle = RepairOrchestrator(authorize_operation_id="approve_age").run(
        question="¿Cuántos años tiene en 2026?",
        initial_response="Tiene 41 años.",
        evidence={"d_bio": doc},
        graph_or_adapter=graph_adapter,
        repair_agent=llm_repairer,
    )
    assert cycle.accepted is True
    assert cycle.attempts[0].review.representation_mode == "adapter_extracted_graph"
    assert cycle.final_review.representation_mode == "adapter_extracted_graph"
    assert len(stub_repair_model.seen_payloads) == 1
    sent_payload = stub_repair_model.seen_payloads[0][1]
    assert sent_payload["required_repairs"]
    assert cycle.operational_authorization is not None and cycle.operational_authorization.allowed is False


# ---------------------------------------------------------------------------
# Family 12: AST equation verification (audit false-repair remediation)
# ---------------------------------------------------------------------------

def test_17_false_equation_in_repaired_response_rejected_vs_true_equation_accepted() -> None:
    doc = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=False,
    )
    q = "¿Cuántos años tiene si estamos en 2026?"

    # Direct unit test on AST equation parser/verifier
    eq_bad, eq_bad_issues = verify_response_equations("El cálculo es 13 + (2026 − 1933) = 13.", graph)
    assert len(eq_bad) == 1
    assert eq_bad[0].is_mathematically_true is False
    assert eq_bad[0].status == "refuted"
    assert eq_bad_issues and eq_bad_issues[0]["code"] == "FALSE_EQUATION_IN_RESPONSE"

    eq_good, eq_good_issues = verify_response_equations("El cálculo es 13 + (2026 − 1933) = 106.", graph)
    assert len(eq_good) == 1
    assert eq_good[0].is_mathematically_true is True
    assert eq_good[0].matches_graph_step is True
    assert eq_good[0].status == "verified"
    assert eq_good_issues == []

    # Full RepairOrchestrator cycle:
    # iter 1: naive -> RETRY
    # iter 2: superficially qualifies + FALSE equation (= 13) -> RETRY
    # iter 3: properly qualified + TRUE grounded equation (= 106) -> ACCEPT
    resp_false_eq = (
        "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
        "la edad exacta no puede determinarse sin fechas completas ni ha sido verificada externamente. "
        "El cálculo es 13 + (2026 − 1933) = 13."
    )
    resp_true_eq = (
        "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
        "la edad exacta no puede determinarse sin fechas completas ni ha sido verificada externamente. "
        "El cálculo es 13 + (2026 − 1933) = 106."
    )
    cycle = RepairOrchestrator(budget=RepairBudget(max_iterations=4)).run(
        question=q,
        initial_response="Tiene 106 años.",
        evidence={"d_buffett": doc},
        graph_or_adapter=graph,
        repair_agent=ScriptedRepairAgent([resp_false_eq, resp_true_eq]),
    )
    assert cycle.iterations_used == 3
    assert cycle.attempts[0].review.disposition == "RETRY"
    assert cycle.attempts[1].review.disposition == "RETRY"
    assert any(i.code == "FALSE_EQUATION_IN_RESPONSE" for i in cycle.attempts[1].review.issues)
    assert cycle.attempts[1].review.assessment.response_correspondence_verified is False
    assert cycle.accepted is True
    assert cycle.final_disposition == "ACCEPT"
    assert cycle.final_review.issues == ()
    assert cycle.final_review.assessment.response_correspondence_verified is True

    # Arithmetically true but ungrounded equation (operands/result unrelated to graph step)
    resp_ungrounded_eq = (
        "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
        "la edad exacta no puede determinarse sin fechas completas ni ha sido verificada externamente. "
        "El cálculo es 50 + 50 = 100."
    )
    rv_ungrounded = review_response(q, resp_ungrounded_eq, {"d_buffett": doc}, graph)
    assert rv_ungrounded.disposition == "RETRY"
    assert any(i.code == "UNGROUNDED_EQUATION_IN_RESPONSE" for i in rv_ungrounded.issues)
    assert rv_ungrounded.assessment.response_correspondence_verified is False

    # Symbolic / unparseable expression cannot receive ACCEPT either
    resp_symbolic_eq = (
        "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
        "la edad exacta no puede determinarse sin fechas completas ni ha sido verificada externamente. "
        "El cálculo es 13 + x = 106."
    )
    rv_symbolic = review_response(q, resp_symbolic_eq, {"d_buffett": doc}, graph)
    assert rv_symbolic.disposition != "ACCEPT"
    assert any(i.code == "UNVERIFIED_EXPRESSION_IN_RESPONSE" for i in rv_symbolic.issues)
    assert rv_symbolic.assessment.response_correspondence_verified is False


# ---------------------------------------------------------------------------
# Family 13: Condition polarity inversion, unrelated condition, and ambiguity
# ---------------------------------------------------------------------------

def _build_conditional_refund_fixture() -> tuple[SourceDocument, JustificationGraph, str]:
    doc = _doc(
        "d_cond",
        "El reembolso aplicable es de 500 USD, salvo en caso de disputa activa.",
        weight=0.95,
    )
    ext = ClaimNode(
        claim_id="r_ext",
        kind="extraction",
        subject="Reembolso",
        predicate="refund",
        value=500,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "El reembolso aplicable es de 500 USD, salvo en caso de disputa activa."),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="r_conc",
        kind="derivation",
        subject="Reembolso",
        predicate="refund",
        value=500,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
    )
    step = InferenceStep(
        step_id="s_ref",
        operation="quantities",
        premises=("r_ext",),
        conclusion_id="r_conc",
    )
    graph = JustificationGraph(
        claims={"r_ext": ext, "r_conc": conc},
        steps=(step,),
        conclusion_id="r_conc",
    )
    return doc, graph, "¿Cuál es el reembolso aplicable?"


def test_18_condition_inversion_and_unrelated_condition_substitution_rejected() -> None:
    doc, graph, q = _build_conditional_refund_fixture()

    # a) Omitted condition -> RETRY
    rv_dropped = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD.",
        {"d_cond": doc},
        graph,
    )
    assert rv_dropped.disposition == "RETRY"
    assert any(i.code == "CONDITION_DROPPED_IN_RESPONSE" for i in rv_dropped.issues)

    # b) Polarity inverted ("incluso en caso de disputa activa") -> RETRY
    rv_inverted = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD, incluso en caso de disputa activa.",
        {"d_cond": doc},
        graph,
    )
    assert rv_inverted.disposition == "RETRY"
    assert any(i.code == "CONDITION_POLARITY_INVERTED" for i in rv_inverted.issues)
    assert rv_inverted.assessment.response_correspondence_verified is False

    # c) Unrelated condition ("si hace sol") -> RETRY with both dropped and unsupported condition
    rv_unrelated = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD, si hace sol.",
        {"d_cond": doc},
        graph,
    )
    assert rv_unrelated.disposition == "RETRY"
    issue_codes = {i.code for i in rv_unrelated.issues}
    assert "CONDITION_DROPPED_IN_RESPONSE" in issue_codes
    assert "UNSUPPORTED_CONDITION_SUBSTITUTION" in issue_codes

    # d) Unmarked ambiguous mention -> never ACCEPT without open_semantics resolution
    rv_ambig = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD; disputa activa.",
        {"d_cond": doc},
        graph,
    )
    assert rv_ambig.disposition != "ACCEPT"
    assert any(i.code == "CONDITION_CORRESPONDENCE_UNRESOLVED" for i in rv_ambig.issues)
    assert rv_ambig.assessment.response_correspondence_verified is False

    seen_checks: list[str] = []

    def _resolve_cond(**kwargs: object) -> dict[str, str]:
        seen_checks.append(str(kwargs.get("check")))
        if kwargs.get("check") == "condition_correspondence":
            return {"status": "verified"}
        return {"status": "unknown"}

    prof_sem = DomainProfile(
        profile_id="cond_sem",
        semantic_evaluator=SemanticEvaluatorConfig(enabled=True, callable_evaluator=_resolve_cond),
    )
    rv_resolved = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD; disputa activa.",
        {"d_cond": doc},
        graph,
        domain_profile=prof_sem,
    )
    assert "condition_correspondence" in seen_checks
    assert rv_resolved.disposition == "ACCEPT"
    assert rv_resolved.assessment.response_correspondence_verified is True

    # e) Proper exception preservation -> ACCEPT
    rv_valid = review_response(
        q,
        "Según el texto, el reembolso aplicable es de 500 USD, salvo en caso de disputa activa.",
        {"d_cond": doc},
        graph,
    )
    assert rv_valid.disposition == "ACCEPT"
    assert rv_valid.issues == ()
    assert rv_valid.assessment.response_correspondence_verified is True


# ---------------------------------------------------------------------------
# Family 14: Conclusion negation & ungrounded non-numeric factual addition
# ---------------------------------------------------------------------------

def test_19_conclusion_negation_and_ungrounded_non_numeric_assertion_rejected() -> None:
    doc = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=True,
    )
    q = "¿Cuántos años tiene si estamos en 2026?"

    negated_resp = (
        "Según el texto, Warren Buffet no tendría aproximadamente 106 años en 2026, "
        "aunque no se puede determinar la edad exacta sin fechas completas."
    )
    rv_neg = review_response(q, negated_resp, {"d_buffett": doc}, graph)
    assert rv_neg.disposition == "RETRY"
    assert any(i.code == "CONCLUSION_POLARITY_INVERTED" for i in rv_neg.issues)
    assert rv_neg.assessment.response_correspondence_verified is False

    hallucinated_non_numeric = (
        "Según el texto, tendría aproximadamente 106 años en 2026 y Warren Buffet es astronauta, "
        "aunque no se puede determinar la edad exacta sin fechas completas."
    )
    rv_halluc = review_response(q, hallucinated_non_numeric, {"d_buffett": doc}, graph)
    assert rv_halluc.disposition == "RETRY"
    assert any(i.code == "UNSUPPORTED_FACTUAL_ADDITION" for i in rv_halluc.issues)
    assert rv_halluc.assessment.response_correspondence_verified is False


# ---------------------------------------------------------------------------
# Family 15: Compositional paraphrase acceptance & perturbation matrix
# ---------------------------------------------------------------------------

def test_20_valid_paraphrases_accepted_across_domains() -> None:
    doc_age = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph_age = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc_age,
        approximate=False,
    )
    paraphrase_age = (
        "A partir del pasaje facilitado, la estimación para Warren Buffet es de unos 106 años en 2026; "
        "faltan el día y el mes para precisar su edad."
    )
    rv_age = review_response(
        "¿Cuántos años tiene si estamos en 2026?",
        paraphrase_age,
        {"d_buffett": doc_age},
        graph_age,
    )
    assert rv_age.disposition == "ACCEPT"
    assert rv_age.issues == ()
    assert rv_age.assessment.response_correspondence_verified is True

    doc_cond, graph_cond, q_cond = _build_conditional_refund_fixture()
    paraphrase_cond = (
        "De acuerdo con el documento proporcionado, el reembolso aplicable es de 500 USD, "
        "solo si no existe disputa activa."
    )
    rv_cond = review_response(q_cond, paraphrase_cond, {"d_cond": doc_cond}, graph_cond)
    assert rv_cond.disposition == "ACCEPT"
    assert rv_cond.issues == ()
    assert rv_cond.assessment.response_correspondence_verified is True


def _build_invoice_subtotal_fixture() -> tuple[SourceDocument, JustificationGraph, str]:
    doc = _doc(
        "d_inv",
        "Factura A es 400 USD y Factura B es 600 USD, salvo en caso de disputa activa.",
        weight=0.95,
    )
    i1 = ClaimNode(
        claim_id="inv_a",
        kind="extraction",
        subject="Factura A",
        predicate="charge",
        value=400,
        unit="USD",
        event_id="evt-a",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "Factura A es 400 USD"),),
        structural_reliability=0.95,
    )
    i2 = ClaimNode(
        claim_id="inv_b",
        kind="extraction",
        subject="Factura B",
        predicate="charge",
        value=600,
        unit="USD",
        event_id="evt-b",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "Factura B es 600 USD"),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="tot_inv",
        kind="derivation",
        subject="Facturas",
        predicate="total_amount",
        value=1000,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
    )
    step = InferenceStep(
        step_id="s_agg",
        operation="agg_sum",
        premises=("inv_a", "inv_b"),
        conclusion_id="tot_inv",
    )
    graph = JustificationGraph(
        claims={"inv_a": i1, "inv_b": i2, "tot_inv": conc},
        steps=(step,),
        conclusion_id="tot_inv",
    )
    return doc, graph, "¿Cuál es el total de facturas?"


def test_21_transformation_perturbation_matrix_zero_false_accept_and_zero_false_reject() -> None:
    doc_age = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph_age = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc_age,
        approximate=False,
    )
    q_age = "¿Cuántos años tiene si estamos en 2026?"
    doc_cond, graph_cond, q_cond = _build_conditional_refund_fixture()
    doc_inv, graph_inv, q_inv = _build_invoice_subtotal_fixture()

    cases = [
        # (name, q, resp, ev, graph, expected_accept)
        (
            "false_equation",
            q_age,
            "Según el texto, tendría aproximadamente 106 años en 2026 sin fechas completas para exactitud. El cálculo es 13 + (2026 − 1933) = 13.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "ungrounded_equation",
            q_age,
            "Según el texto, tendría aproximadamente 106 años en 2026 sin fechas completas para exactitud. El cálculo es 50 + 50 = 100.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "conclusion_negation",
            q_age,
            "Según el texto, Warren Buffet no tendría aproximadamente 106 años en 2026; faltan fechas completas.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "ungrounded_claim_addition",
            q_age,
            "Según el texto, tendría aproximadamente 106 años en 2026 y Warren Buffet es astronauta; faltan el día y el mes para precisar su edad.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "condition_inversion",
            q_cond,
            "Según el texto, el reembolso aplicable es de 500 USD, incluso en caso de disputa activa.",
            {"d_cond": doc_cond},
            graph_cond,
            False,
        ),
        (
            "unrelated_condition",
            q_cond,
            "Según el texto, el reembolso aplicable es de 500 USD, si hace sol.",
            {"d_cond": doc_cond},
            graph_cond,
            False,
        ),
        (
            "role_swap_age_year",
            q_age,
            "Según el texto, Warren Buffet tenía 1933 años en 13 (aproximadamente 106 años en 2026); faltan el día y el mes para precisar su edad.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "predicate_noun_reassignment_shares",
            q_age,
            "Según el texto, Warren Buffet compró 106 acciones en 1933 (aproximadamente 106 años en 2026); faltan el día y el mes para precisar su edad.",
            {"d_buffett": doc_age},
            graph_age,
            False,
        ),
        (
            "entity_value_swap_invoices",
            q_inv,
            "Según el documento, Factura A es 600 USD y Factura B es 400 USD, por lo que el subtotal parcial de las facturas es 1000 USD sin constancia de completitud de población, salvo en caso de disputa activa.",
            {"d_inv": doc_inv},
            graph_inv,
            False,
        ),
        (
            "valid_age_paraphrase_1",
            q_age,
            "A partir del pasaje facilitado, la estimación para Warren Buffet es de unos 106 años en 2026; faltan el día y el mes para precisar su edad.",
            {"d_buffett": doc_age},
            graph_age,
            True,
        ),
        (
            "valid_age_paraphrase_2_with_grounded_equation",
            q_age,
            "Conforme al fragmento aportado, la edad de Warren Buffet rondaría los 106 años en 2026 (13 + (2026 - 1933) = 106), aunque no hay fecha de nacimiento completa ni verificación histórica externa.",
            {"d_buffett": doc_age},
            graph_age,
            True,
        ),
        (
            "valid_cond_paraphrase_1",
            q_cond,
            "De acuerdo con el documento proporcionado, el reembolso aplicable es de 500 USD, solo si no existe disputa activa.",
            {"d_cond": doc_cond},
            graph_cond,
            True,
        ),
        (
            "valid_cond_paraphrase_2",
            q_cond,
            "Según el pasaje facilitado, el reembolso aplicable es de 500 USD, a menos que haya disputa activa.",
            {"d_cond": doc_cond},
            graph_cond,
            True,
        ),
        (
            "valid_paraphrase_invoices_correct",
            q_inv,
            "De acuerdo con el texto, Factura A es 400 USD y Factura B es 600 USD, por lo que el subtotal parcial de las facturas es 1000 USD sin constancia de completitud de población, salvo en caso de disputa activa.",
            {"d_inv": doc_inv},
            graph_inv,
            True,
        ),
        (
            "valid_paraphrase_buffett_event_anchored",
            q_age,
            "Según el pasaje, cuando tenía 13 años en 1933, Warren Buffet compró su primera acción, por lo que rondaría los 106 años en 2026; faltan el día y el mes para precisar su edad.",
            {"d_buffett": doc_age},
            graph_age,
            True,
        ),
    ]

    false_accepts = 0
    false_rejects = 0
    negative_total = sum(1 for item in cases if not item[5])
    positive_total = sum(1 for item in cases if item[5])
    for name, q, resp, ev, graph, expected_accept in cases:
        rv = review_response(q, resp, ev, graph)
        actual_accept = rv.disposition == "ACCEPT" and rv.assessment.response_correspondence_verified
        if not expected_accept and actual_accept:
            false_accepts += 1
        elif expected_accept and not actual_accept:
            false_rejects += 1
            raise AssertionError(f"False reject on {name}: disposition={rv.disposition}, issues={[i.code for i in rv.issues]}")

    false_accept_rate = false_accepts / negative_total
    false_reject_rate = false_rejects / positive_total
    assert false_accept_rate == 0.0
    assert false_reject_rate == 0.0


def test_22_relational_role_swap_and_noun_reassignment_rejected() -> None:
    doc = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=False,
    )
    q = "¿Cuántos años tiene si estamos en 2026?"

    adversarial = [
        "Según el texto, Warren Buffet tenía 1933 años en 13 (aproximadamente 106 años en 2026); faltan fechas completas.",
        "Según el texto, Warren Buffet compró 106 acciones en 1933; tendría aproximadamente 106 años en 2026, sin fechas completas.",
        "Según el texto, Warren Buffet tenía 13 años en 2026; estimación de unos 106 años, faltan fechas completas.",
        "Según el texto, Warren Buffet tenía 106 años en 1933; tendría aproximadamente 106 años en 2026, faltan fechas completas.",
        "Según el texto, Warren Buffet compró su primera acción en 2026; estimación de unos 106 años en 2026, faltan fechas completas.",
    ]
    for bad in adversarial:
        rv = review_response(q, bad, {"d_buffett": doc}, graph)
        assert rv.disposition == "RETRY", f"Expected RETRY for: {bad!r}"
        assert rv.assessment.response_correspondence_verified is False
        assert any(i.code == "RELATIONAL_VALUE_MISMATCH" for i in rv.issues), (
            f"Expected RELATIONAL_VALUE_MISMATCH for {bad!r}, got {[i.code for i in rv.issues]}"
        )

    valid = (
        "Según el texto, cuando tenía 13 años en 1933, Warren Buffet compró su primera acción; "
        "tendría aproximadamente 106 años en 2026, pero faltan el día y el mes para precisar su edad."
    )
    rv_ok = review_response(q, valid, {"d_buffett": doc}, graph)
    assert rv_ok.disposition == "ACCEPT"
    assert rv_ok.assessment.response_correspondence_verified is True
    assert rv_ok.issues == ()


def test_23_invoice_entity_value_swap_rejected_even_when_total_matches() -> None:
    doc_inv, graph_inv, q_inv = _build_invoice_subtotal_fixture()

    swapped = (
        "Según el texto, Factura A es 600 USD y Factura B es 400 USD, por lo que el subtotal parcial de las facturas es 1000 USD "
        "sin constancia de completitud de población, salvo en caso de disputa activa."
    )
    corr_bad = verify_response_graph_correspondence(
        swapped,
        graph_inv,
        question=q_inv,
        documents={"d_inv": doc_inv},
    )
    assert corr_bad["correspondence_ok"] is False
    assert corr_bad["relational_mismatch_in_response"] is True

    rv_bad = review_response(q_inv, swapped, {"d_inv": doc_inv}, graph_inv)
    assert rv_bad.disposition == "RETRY"
    assert rv_bad.assessment.response_correspondence_verified is False
    assert any(i.code == "RELATIONAL_VALUE_MISMATCH" for i in rv_bad.issues)

    valid = (
        "Según el texto, Factura A es 400 USD y Factura B es 600 USD, por lo que el subtotal parcial de las facturas es 1000 USD "
        "sin constancia de completitud de población, salvo en caso de disputa activa."
    )
    rv_ok = review_response(q_inv, valid, {"d_inv": doc_inv}, graph_inv)
    assert rv_ok.disposition == "ACCEPT"
    assert rv_ok.assessment.response_correspondence_verified is True

    cycle = RepairOrchestrator(budget=RepairBudget(max_iterations=3)).run(
        question=q_inv,
        initial_response=swapped,
        evidence={"d_inv": doc_inv},
        graph_or_adapter=graph_inv,
        repair_agent=ScriptedRepairAgent([valid]),
    )
    assert cycle.accepted is True
    assert cycle.final_disposition == "ACCEPT"
    assert cycle.iterations_used == 2
    assert any(i.code == "RELATIONAL_VALUE_MISMATCH" for i in cycle.attempts[0].review.issues)

    # Conflict disclosure must also bind values strictly to each source entity (Anexo A vs Anexo B)
    d1 = _doc("d1", "Anexo A indica que el límite de responsabilidad es 500 USD.", weight=0.95)
    d2 = _doc("d2", "Anexo B indica que el límite de responsabilidad es 800 USD.", weight=0.95)
    c1 = ClaimNode(claim_id="cap_a", kind="extraction", subject="Contrato", predicate="liability_cap", value=500, unit="USD", evidence=(_ref(d1, "límite de responsabilidad es 500 USD"),), structural_reliability=0.95)
    c2 = ClaimNode(claim_id="cap_b", kind="extraction", subject="Contrato", predicate="liability_cap", value=800, unit="USD", evidence=(_ref(d2, "límite de responsabilidad es 800 USD"),), structural_reliability=0.95)
    c_conc = ClaimNode(claim_id="cap_conc", kind="derivation", subject="Contrato", predicate="liability_cap", value=500, unit="USD")
    c_step = InferenceStep(step_id="s_id", operation="quantities", premises=("cap_a",), conclusion_id="cap_conc")
    c_graph = JustificationGraph(claims={"cap_a": c1, "cap_b": c2, "cap_conc": c_conc}, steps=(c_step,), conclusion_id="cap_conc")
    swapped_conflict = "Según los textos, existe un conflicto sin resolver respecto al Contrato: Anexo A indica 800 USD pero Anexo B indica 500 USD."
    rv_c_swap = review_response("¿Cuál es el límite de responsabilidad del Contrato?", swapped_conflict, {"d1": d1, "d2": d2}, c_graph)
    assert rv_c_swap.disposition == "RETRY"
    assert any(i.code == "RELATIONAL_VALUE_MISMATCH" for i in rv_c_swap.issues)


def test_24_unparseable_relational_clause_stays_pending_unless_semantically_resolved() -> None:
    doc = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
        weight=0.95,
    )
    graph = _build_anchor_age_graph(
        subject="Warren Buffet",
        anchor_age=13,
        anchor_year=1933,
        reference_year=2026,
        claimed_age=106,
        quote_age="tenía 13 años",
        quote_year="era 1933",
        doc=doc,
        approximate=False,
    )
    q = "¿Cuántos años tiene si estamos en 2026?"
    ambiguous = "Según el texto, Warren Buffet 13 1933 y rondaría los 106 años en 2026; faltan fechas completas."

    rv_default = review_response(q, ambiguous, {"d_buffett": doc}, graph)
    assert rv_default.disposition != "ACCEPT"
    assert rv_default.assessment.response_correspondence_verified is False
    assert any(i.code == "RELATIONAL_CORRESPONDENCE_UNRESOLVED" for i in rv_default.issues)

    seen_checks: list[str] = []

    def _refute_rel(**kwargs: object) -> dict[str, str]:
        seen_checks.append(str(kwargs.get("check")))
        if kwargs.get("check") == "relational_correspondence":
            return {"status": "refuted"}
        return {"status": "unknown"}

    prof_refute = DomainProfile(
        profile_id="rel_refute",
        semantic_evaluator=SemanticEvaluatorConfig(enabled=True, callable_evaluator=_refute_rel),
    )
    rv_refuted = review_response(q, ambiguous, {"d_buffett": doc}, graph, domain_profile=prof_refute)
    assert "relational_correspondence" in seen_checks
    assert rv_refuted.disposition == "RETRY"
    assert rv_refuted.assessment.response_correspondence_verified is False
    assert any(i.code == "RELATIONAL_VALUE_MISMATCH" for i in rv_refuted.issues)

    def _verify_rel(**kwargs: object) -> dict[str, str]:
        if kwargs.get("check") == "relational_correspondence":
            return {"status": "verified"}
        return {"status": "unknown"}

    prof_verify = DomainProfile(
        profile_id="rel_verify",
        semantic_evaluator=SemanticEvaluatorConfig(enabled=True, callable_evaluator=_verify_rel),
    )
    rv_verified = review_response(q, ambiguous, {"d_buffett": doc}, graph, domain_profile=prof_verify)
    assert rv_verified.disposition == "ACCEPT"
    assert rv_verified.assessment.response_correspondence_verified is True
