#!/usr/bin/env python3
"""Comprehensive regression tests for Harness2 Securitization ClaimLayer v3 + v4 integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _ROOT / "examples" / "harness2_securitization"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from claimlayer import Assertion  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    ClaimNode,
    InferenceStep,
    JustificationGraph,
)
from claimlayer.v4.domains.securitization import (  # noqa: E402
    WaterfallStepSpec,
    verify_waterfall_steps,
)
from harness2_v4_pipeline import (  # noqa: E402
    SecuritizationReportExtractor,
    assert_no_evaluator_only_access,
    build_synthetic_securitization_corpus,
    evaluate_securitization_candidate,
    load_case_documents,
    run_securitization_repair_loop,
)


def _gov_waterfall_steps() -> list[WaterfallStepSpec]:
    return [
        WaterfallStepSpec(
            step_id="wf_class_a_int",
            beneficiary="Class A",
            payment_type="interest",
            position=3,
            before=("wf_class_b_prin", "wf_class_c_prin", "wf_reserve"),
            regime="pre_acceleration",
            document_id="cart-2024-2-indenture.docx",
            clause_id="Sec_5_01_a_3",
        ),
        WaterfallStepSpec(
            step_id="wf_class_b_prin",
            beneficiary="Class B",
            payment_type="principal",
            position=11,
            before=("wf_class_c_prin", "wf_reserve"),
            after=("wf_class_a_int",),
            regime="pre_acceleration",
            document_id="cart-2024-2-indenture.docx",
            clause_id="Sec_5_01_a_11",
        ),
        WaterfallStepSpec(
            step_id="wf_class_c_prin",
            beneficiary="Class C",
            payment_type="principal",
            position=12,
            before=("wf_reserve",),
            after=("wf_class_b_prin",),
            regime="pre_acceleration",
            document_id="cart-2024-2-indenture.docx",
            clause_id="Sec_5_01_a_12",
        ),
        WaterfallStepSpec(
            step_id="wf_reserve",
            beneficiary="Reserve Account",
            payment_type="reserve_replenishment",
            position=13,
            after=("wf_class_b_prin", "wf_class_c_prin"),
            regime="pre_acceleration",
            document_id="cart-2024-2-indenture.docx",
            clause_id="Sec_5_01_a_13",
        ),
    ]


def _make_valid_synthetic_candidate(
    *,
    report_text: str | None = None,
    graph: JustificationGraph | None = None,
    waterfall: dict | None = None,
    assertions: list[Assertion] | None = None,
) -> dict:
    default_text = (
        "Comprehensive Discrepancy Report (CART 2024-2):\n"
        "The executed Indenture (Rank 100) governs over the Offering Memorandum (Rank 60) and Associate Email (Rank 10).\n"
        "1. Note Interest Rates (Section 2.04): Class A-1 Notes bear 5.10% per annum, Class B Notes bear 5.95% per annum, "
        "and Class C Notes ($50,000,000 principal) govern at 6.75% per annum (the OM erroneously states 6.50% per annum in part, "
        "a 25 bps / 0.25 percentage point understatement). Assuming constant principal as a simple annual calculation before "
        "amortization or day-count adjustments, 50,000,000 * 0.0025 = $125,000 per year (illustrative only, not automatic legal damages).\n"
        "2. Waterfall Priority (Section 5.01(a), correcting email's non-existent Section 5.04): Reserve Account replenishment is "
        "Step 13, junior to Class B and Class C principal (steps 11 and 12), whereas in the OM Reserve Account Replenishment is senior (placed ahead of Class B and Class C principal).\n"
        "3. Events of Default Grace Period (Section 10.01): 5 business days (not 5 calendar days or 3 business days).\n"
        "4. Optional Redemption Clean-Up Call (Section 12.01): Servicer clean-up call is 10% of Initial Pool Balance (Optional Redemption is 15% in draft OM excerpt)."
    )
    default_assertions = assertions if assertions is not None else [
        Assertion(
            subject="Class C Notes",
            predicate="coupon",
            value="6.75%",
            source_id="cart-2024-2-indenture.docx",
            quote="Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.",
        ),
        Assertion(
            subject="Waterfall",
            predicate="waterfall",
            value="Step 13 Reserve Account Replenishment junior to Class B and Class C principal",
            source_id="cart-2024-2-indenture.docx",
            quote="Step 13 Reserve Account Replenishment (junior to Class B and Class C principal)",
        ),
        Assertion(
            subject="Grace Period",
            predicate="grace_period",
            value="5 business days",
            source_id="cart-2024-2-indenture.docx",
            quote="Failure to pay interest continuing for 5 business days constitutes an Event of Default.",
        ),
        Assertion(
            subject="Clean-Up Call",
            predicate="clean_up_call",
            value="10%",
            source_id="cart-2024-2-indenture.docx",
            quote="Servicer may redeem Notes when Pool Balance falls to 10% of Initial Pool Balance.",
        ),
    ]
    return {
        "report_text": report_text if report_text is not None else default_text,
        "assertions": default_assertions,
        "required_sections": ["coupon", "waterfall", "grace_period", "clean_up_call"],
        "waterfall": waterfall if waterfall is not None else {"governing_steps": _gov_waterfall_steps()},
        "graph": graph,
        "metadata": {"has_material_discrepancies": True},
    }


def test_01_incomplete_report_omitting_required_sections_rejected():
    corpus = build_synthetic_securitization_corpus()
    candidate = _make_valid_synthetic_candidate(
        report_text="Class C Notes govern at 6.75% per annum ($125,000/yr assuming constant principal simple annual calculation).",
        assertions=[
            Assertion(
                subject="Class C Notes",
                predicate="coupon",
                value="6.75%",
                source_id="cart-2024-2-indenture.docx",
                quote="Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.",
            )
        ],
    )
    res = evaluate_securitization_candidate(
        candidate,
        corpus,
        gate_mode="v3_v4",
        required_sections=["coupon", "waterfall", "grace_period", "clean_up_call"],
    )
    assert res.report_disposition == "RETRY"
    assert not res.accepted
    assert any(i["code"] in ("MISSING_REQUIRED_PREDICATE", "MISSING_REQUIRED_SECTION") for i in res.issues)


def test_02_authority_inversion_and_email_section_trap_rejected():
    corpus = build_synthetic_securitization_corpus()
    inverted_text = (
        "The Offering Memorandum governs over the Indenture because it is dated November 18 vs November 15. "
        "Under Section 5.04(a), Class C Notes bear 6.50% per annum and Step 10 controls."
    )
    candidate = _make_valid_synthetic_candidate(
        report_text=inverted_text,
        assertions=[
            Assertion(
                subject="Class C Notes",
                predicate="coupon",
                value="6.50%",
                source_id="cart-2024-2-offering-memorandum.docx",
                quote="Class C Notes ($50,000,000 principal) 6.50% per annum",
            ),
            Assertion(
                subject="Waterfall",
                predicate="waterfall",
                value="Step 13",
                source_id="cart-2024-2-indenture.docx",
                quote="Step 13 Reserve Account Replenishment (junior to Class B and Class C principal)",
            ),
            Assertion(
                subject="Grace Period",
                predicate="grace_period",
                value="5 business days",
                source_id="cart-2024-2-indenture.docx",
                quote="Failure to pay interest continuing for 5 business days constitutes an Event of Default.",
            ),
            Assertion(
                subject="Clean-Up Call",
                predicate="clean_up_call",
                value="10%",
                source_id="cart-2024-2-indenture.docx",
                quote="Servicer may redeem Notes when Pool Balance falls to 10% of Initial Pool Balance.",
            ),
        ],
    )
    res = evaluate_securitization_candidate(candidate, corpus, gate_mode="v3_v4")
    codes = {i["code"] for i in res.issues}
    assert res.report_disposition == "RETRY"
    assert "AUTHORITY_INVERSION" in codes or "TIER_INSUFFICIENT" in codes
    assert "FALSE_SECTION_CITATION_FROM_EMAIL" in codes


def test_03_coupon_swap_across_classes_rejected():
    corpus = build_synthetic_securitization_corpus()
    swapped_text = (
        "Executed Indenture governs. Section 5.01(a) Step 13 Reserve Account Replenishment. "
        "Class B Notes bear interest at 6.75% per annum and Class C Notes bear interest at 5.95% per annum. "
        "Events of Default grace period is 5 business days. Clean-up call is 10% of Initial Pool Balance."
    )
    candidate = _make_valid_synthetic_candidate(report_text=swapped_text)
    res = evaluate_securitization_candidate(candidate, corpus, gate_mode="v3_v4")
    assert res.report_disposition == "RETRY"
    assert any(i["code"] == "CLASS_COUPON_MISMATCH" for i in res.issues)


def test_04_waterfall_priority_modes_cycle_swap_payment_regime_and_nonconflicting_shift():
    gov = _gov_waterfall_steps()

    # (a) Relative priority inversion (reserve step 6 ahead of Class B & Class C principal steps 11/12)
    inverted_wf = verify_waterfall_steps(
        gov,
        [
            WaterfallStepSpec("wf_reserve", "Reserve Account", "reserve_replenishment", position=6, before=("wf_class_b_prin", "wf_class_c_prin"), regime="pre_acceleration"),
            WaterfallStepSpec("wf_class_b_prin", "Class B", "principal", position=11, regime="pre_acceleration"),
            WaterfallStepSpec("wf_class_c_prin", "Class C", "principal", position=12, regime="pre_acceleration"),
        ],
    )
    assert inverted_wf.status != "verified"
    assert "WATERFALL_PRIORITY_INVERSION" in inverted_wf.reasons

    # (b) Beneficiary swap at step 11 (Class C placed in Class B slot wf_class_b_prin)
    ben_swap = verify_waterfall_steps(
        gov,
        [WaterfallStepSpec("wf_class_b_prin", "Class C", "principal", position=11, regime="pre_acceleration")],
    )
    assert ben_swap.status != "verified"
    assert "WATERFALL_BENEFICIARY_SWAP" in ben_swap.reasons

    # (c) Payment type confusion (Class A Step 3 claimed as principal instead of interest)
    ptype_swap = verify_waterfall_steps(
        gov,
        [WaterfallStepSpec("wf_class_a_int", "Class A", "principal", position=3, regime="pre_acceleration")],
    )
    assert ptype_swap.status != "verified"
    assert "WATERFALL_PAYMENT_TYPE_CONFUSION" in ptype_swap.reasons

    # (d) Cyclic ordering (A before B, B before A)
    cyclic = verify_waterfall_steps(
        [
            WaterfallStepSpec("s1", "Class B", "principal", before=("s2",)),
            WaterfallStepSpec("s2", "Reserve Account", "reserve_replenishment", before=("s1",)),
        ]
    )
    assert cyclic.status != "verified"
    assert "WATERFALL_ORDER_CYCLE" in cyclic.reasons

    # (e) Cross-regime (pre_acceleration vs post_acceleration without qualification)
    cross_reg = verify_waterfall_steps(
        gov,
        [WaterfallStepSpec("wf_reserve", "Reserve Account", "reserve_replenishment", position=13, regime="post_acceleration")],
        allow_cross_regime_if_qualified=False,
    )
    assert cross_reg.status != "verified"
    assert "WATERFALL_INCOMPARABLE_REGIME" in cross_reg.reasons

    # (f) Non-conflicting step numbering shift preserving relative order -> ACCEPTED!
    renumbered_nonconflicting = verify_waterfall_steps(
        gov,
        [
            WaterfallStepSpec("wf_class_a_int", "Class A", "interest", position=4, regime="pre_acceleration"),
            WaterfallStepSpec("wf_class_b_prin", "Class B", "principal", position=14, regime="pre_acceleration"),
            WaterfallStepSpec("wf_class_c_prin", "Class C", "principal", position=15, regime="pre_acceleration"),
            WaterfallStepSpec("wf_reserve", "Reserve Account", "reserve_replenishment", position=16, after=("wf_class_b_prin", "wf_class_c_prin"), regime="pre_acceleration"),
        ],
    )
    assert renumbered_nonconflicting.status == "verified"


def test_05_calendar_vs_business_days_unit_mismatch_rejected():
    corpus = build_synthetic_securitization_corpus()
    bad_text = (
        "Indenture Section 5.01(a) governs. Class C Notes 6.75%. Reserve Account Step 13. "
        "Clean-up call 10%. Note that 5 calendar days equals 5 business days under Section 10.01 grace period."
    )
    candidate = _make_valid_synthetic_candidate(report_text=bad_text)
    res = evaluate_securitization_candidate(candidate, corpus, gate_mode="v3_v4")
    assert res.report_disposition == "RETRY"
    assert any(i["code"] == "UNIT_MISMATCH_CALENDAR_VS_BUSINESS_DAYS" for i in res.issues)


def test_06_percent_vs_percentage_points_and_bps_confusion_rejected():
    corpus = build_synthetic_securitization_corpus()
    bad_text = (
        "Indenture Section 5.01(a) governs. Class C Notes govern at 6.75% while OM states 6.50%, "
        "which is a 25% discrepancy (25 bps = 25%). Reserve replenishment Step 13, 5 business days grace, 10% clean-up call."
    )
    candidate = _make_valid_synthetic_candidate(report_text=bad_text)
    res = evaluate_securitization_candidate(candidate, corpus, gate_mode="v3_v4")
    assert res.report_disposition == "RETRY"
    assert any(i["code"] == "PERCENT_VS_PERCENTAGE_POINT_CONFUSION" for i in res.issues)


def test_07_omitted_assumptions_and_damages_rejected_then_accepted_with_escalate_auth():
    corpus = build_synthetic_securitization_corpus()
    # Unqualified damages + omitted constant_principal assumption -> RETRY
    bad_candidate = _make_valid_synthetic_candidate(
        report_text=(
            "Indenture Section 5.01(a) Step 13 governs. Class C Notes govern at 6.75% per annum vs 6.50% in OM. "
            "On $50,000,000 Class C Notes, the 25 bps difference causes $125,000 per year in automatic legal damages. "
            "Grace period is 5 business days; clean-up call is 10%."
        )
    )
    bad_res = evaluate_securitization_candidate(bad_candidate, corpus, gate_mode="v3_v4")
    bad_codes = {i["code"] for i in bad_res.issues}
    assert bad_res.report_disposition == "RETRY"
    assert "OMITTED_DERIVATION_ASSUMPTIONS" in bad_codes
    assert "UNQUALIFIED_DAMAGES_OR_CASHFLOW_CLAIM" in bad_codes

    # Repaired candidate explicitly qualifies assumptions + full graph -> ACCEPT report, ESCALATE execution auth!
    ind_id = "cart-2024-2-indenture.docx"
    om_id = "cart-2024-2-offering-memorandum.docx"
    eml_id = "associate-review-email.eml"
    ind_txt = corpus.v4_sources[ind_id].text
    om_txt = corpus.v4_sources[om_id].text
    eml_txt = corpus.v4_sources[eml_id].text
    q_a = "The Class A Notes bear interest at 5.10% per annum"
    q_b = "Class B Notes at 5.95% per annum"
    q_prin = "Class C Notes ($50,000,000 principal)"
    q_spread = "Class C 25 bps coupon gap ($125,000/yr)"
    q_c_ind = "Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum"
    q_c_om = "Class C Notes ($50,000,000 principal) 6.50% per annum"
    q_spread_pp = "25 bps / 0.25 percentage points lower than Indenture"
    q_impact_eml = "($125,000/yr)"
    q_wf_c = "Steps 3-7 Class A-1, A-2, A-3, B, and C Interest"
    q_wf = "Step 13 Reserve Account Replenishment (junior to Class B and Class C principal)"
    q_wf_om = "Step 6/10 Reserve Account Replenishment (placed senior/ahead of Class B and Class C principal)"
    q_gp = "Failure to pay interest continuing for 5 business days constitutes an Event of Default."
    q_gp_om = "5 calendar days (or 3 business days) cure period variant."
    q_cc = "Servicer may redeem Notes when Pool Balance falls to 10% of Initial Pool Balance."
    q_cc_om = "Servicer may redeem Notes when Pool Balance falls to 15% of Initial Pool Balance."
    full_graph = JustificationGraph(
        claims={
            "c_a": ClaimNode(
                "c_a",
                "extraction",
                "Class A Notes",
                "coupon_rate",
                5.10,
                "percent_per_annum",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_a), "end": ind_txt.index(q_a) + len(q_a), "quote": q_a},),
            ),
            "c_b": ClaimNode(
                "c_b",
                "extraction",
                "Class B Notes",
                "coupon_rate",
                5.95,
                "percent_per_annum",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_b), "end": ind_txt.index(q_b) + len(q_b), "quote": q_b},),
            ),
            "c_prin": ClaimNode(
                "c_prin",
                "extraction",
                "Class C Notes",
                "note_principal",
                50000000,
                "USD",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_prin), "end": ind_txt.index(q_prin) + len(q_prin), "quote": q_prin},),
            ),
            "c_spread": ClaimNode(
                "c_spread",
                "extraction",
                "Class C Notes",
                "coupon_gap",
                25,
                "basis_point",
                evidence=({"document_id": eml_id, "start": eml_txt.index(q_spread), "end": eml_txt.index(q_spread) + len(q_spread), "quote": q_spread},),
            ),
            "c_spread_pp": ClaimNode(
                "c_spread_pp",
                "extraction",
                "Class C Notes",
                "coupon_spread",
                0.25,
                "percentage_point",
                evidence=({"document_id": om_id, "start": om_txt.index(q_spread_pp), "end": om_txt.index(q_spread_pp) + len(q_spread_pp), "quote": q_spread_pp},),
            ),
            "c_impact_eml": ClaimNode(
                "c_impact_eml",
                "extraction",
                "Class C",
                "annual_interest_impact",
                125000,
                "USD_per_year",
                evidence=({"document_id": eml_id, "start": eml_txt.index(q_impact_eml), "end": eml_txt.index(q_impact_eml) + len(q_impact_eml), "quote": q_impact_eml},),
            ),
            "c_impact": ClaimNode(
                "c_impact",
                "derivation",
                "Class C Notes",
                "annual_interest_impact",
                125000,
                "USD_per_year",
                assumptions=("constant_principal", "simple_annual_calculation"),
            ),
            "c_coup_ind": ClaimNode(
                "c_coup_ind",
                "extraction",
                "Class C Notes",
                "coupon_rate",
                6.75,
                "percent_per_annum",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_c_ind), "end": ind_txt.index(q_c_ind) + len(q_c_ind), "quote": q_c_ind},),
            ),
            "c_coup_om": ClaimNode(
                "c_coup_om",
                "extraction",
                "Class C Notes",
                "coupon_rate",
                6.50,
                "percent_per_annum",
                evidence=({"document_id": om_id, "start": om_txt.index(q_c_om), "end": om_txt.index(q_c_om) + len(q_c_om), "quote": q_c_om},),
            ),
            "c_wf_c": ClaimNode(
                "c_wf_c",
                "extraction",
                "Class C Notes",
                "waterfall_priority",
                "junior",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_wf_c), "end": ind_txt.index(q_wf_c) + len(q_wf_c), "quote": q_wf_c},),
            ),
            "c_wf": ClaimNode(
                "c_wf",
                "extraction",
                "Reserve Account Replenishment",
                "waterfall_priority",
                "junior",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_wf), "end": ind_txt.index(q_wf) + len(q_wf), "quote": q_wf},),
            ),
            "c_wf_om": ClaimNode(
                "c_wf_om",
                "extraction",
                "Reserve Account Replenishment",
                "waterfall_priority",
                "senior",
                evidence=({"document_id": om_id, "start": om_txt.index(q_wf_om), "end": om_txt.index(q_wf_om) + len(q_wf_om), "quote": q_wf_om},),
            ),
            "c_gp": ClaimNode(
                "c_gp",
                "extraction",
                "Grace Period",
                "cure_period",
                5,
                "business_days",
                condition_status="conditional",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_gp), "end": ind_txt.index(q_gp) + len(q_gp), "quote": q_gp},),
            ),
            "c_gp_om_cal": ClaimNode(
                "c_gp_om_cal",
                "extraction",
                "Grace Period",
                "cure_period",
                5,
                "calendar_days",
                condition_status="conditional",
                evidence=({"document_id": om_id, "start": om_txt.index(q_gp_om), "end": om_txt.index(q_gp_om) + len(q_gp_om), "quote": q_gp_om},),
            ),
            "c_gp_om_bd": ClaimNode(
                "c_gp_om_bd",
                "extraction",
                "Grace Period",
                "cure_period",
                3,
                "business_days",
                condition_status="conditional",
                evidence=({"document_id": om_id, "start": om_txt.index(q_gp_om), "end": om_txt.index(q_gp_om) + len(q_gp_om), "quote": q_gp_om},),
            ),
            "c_cc": ClaimNode(
                "c_cc",
                "extraction",
                "Servicer",
                "clean_up_call_threshold",
                10,
                "percent",
                condition_status="conditional",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_cc), "end": ind_txt.index(q_cc) + len(q_cc), "quote": q_cc},),
            ),
            "c_cc_om": ClaimNode(
                "c_cc_om",
                "extraction",
                "Optional Redemption",
                "clean_up_call_threshold",
                15,
                "percent",
                condition_status="conditional",
                evidence=({"document_id": om_id, "start": om_txt.index(q_cc_om), "end": om_txt.index(q_cc_om) + len(q_cc_om), "quote": q_cc_om},),
            ),
        },
        steps=(
            InferenceStep(
                step_id="s1",
                premises=("c_prin", "c_spread"),
                conclusion_id="c_impact",
                operation="multiply",
                assumptions=("constant_principal", "simple_annual_calculation"),
            ),
        ),
        conclusion_id="c_impact",
    )
    good_candidate = _make_valid_synthetic_candidate(graph=full_graph)
    good_res = evaluate_securitization_candidate(good_candidate, corpus, gate_mode="v3_v4")
    # Real recorded cart_synth.json omits Step 1/2/8-12/9/10 spans, so sentence/quantity cross-check
    # MUST fail closed with REFERENCE_REGISTER_GAP (and zero derivation/damage defects remain).
    assert good_res.report_disposition == "ESCALATE"
    assert good_res.accepted is False
    assert {i["code"] for i in good_res.issues} == {"REFERENCE_REGISTER_GAP"}
    assert good_res.operational_authorization["status"] == "ESCALATE"
    assert good_res.operational_authorization["authorized"] is False
    assert good_res.operational_authorization["response_digest"] is not None

    # When complete reader units cover all divergent clauses/quantities on a closed 2-doc securitization corpus,
    # report_disposition reaches ACCEPT while operational authorization still holds at ESCALATE (material discrepancy).
    import hashlib as _hashlib
    from claimlayer.v4 import LLMProposalReader, STRICT_READER_PROMPT
    from harness2_v4_pipeline import _build_corpus_objects as _build_docs

    clean_ind = "The Indenture governs. Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum."
    clean_om = "Draft Offering Memorandum. Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum. The coupon difference is 0.25 percentage point."
    clean_ind_id = "clean-indenture.docx"
    clean_om_id = "clean-om.docx"

    class _Test07Model:
        def __init__(self, mapping):
            self._mapping = mapping

        @property
        def identity(self):
            return {"provider": "test07", "model": "test07-complete", "endpoint": "local", "parameters": {}}

        def complete(self, _system, payload):
            doc_id = payload["document_id"]
            ev_id = payload["fragments"][0]["evidence_id"]
            props = []
            for p in self._mapping.get(doc_id, ()):
                row = dict(p)
                row["evidence"] = [ev_id]
                props.append(row)
            return {"output": {"proposals": props}}

    c_reader = LLMProposalReader(
        model=_Test07Model({
        clean_ind_id: [
            {
                "entity": "Class C Notes",
                "predicate": "note_principal",
                "value": {"value": 50000000, "unit": "USD"},
                "assertion_kind": "definition",
                "condition_status": "unconditional",
                "basis": "explicit",
                "polarity": "affirmed",
                "quote": "Class C Notes ($50,000,000 principal)",
            },
            {
                "entity": "Class C Notes",
                "predicate": "coupon_rate",
                "value": {"value": 6.75, "unit": "percent_per_annum"},
                "assertion_kind": "definition",
                "condition_status": "unconditional",
                "basis": "explicit",
                "polarity": "affirmed",
                "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.",
            },
        ],
        clean_om_id: [
            {
                "entity": "Class C Notes",
                "predicate": "coupon_rate",
                "value": {"value": 6.50, "unit": "percent_per_annum"},
                "assertion_kind": "fact",
                "condition_status": "unconditional",
                "basis": "explicit",
                "polarity": "affirmed",
                "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum.",
            },
            {
                "entity": "Class C Notes",
                "predicate": "coupon_spread",
                "value": {"value": 0.25, "unit": "percentage_point"},
                "assertion_kind": "fact",
                "condition_status": "unconditional",
                "basis": "explicit",
                "polarity": "affirmed",
                "quote": "The coupon difference is 0.25 percentage point.",
            },
        ],
    }),
        prompt=STRICT_READER_PROMPT,
    )
    corpus_complete = _build_docs(
        "test07_complete_sec",
        "test07_task",
        "synthetic_replay",
        [
            ("indenture", clean_ind_id, "governing_contract", "authoritative", 100, True, 1, clean_ind),
            ("offering_memorandum", clean_om_id, "disclosure_document", "supplementary", 40, False, 10, clean_om),
        ],
        common_sha256=_hashlib.sha256((clean_ind + clean_om).encode()).hexdigest(),
        common_chars=len(clean_ind) + len(clean_om),
        total_lines=2,
        reader=c_reader,
    )
    corpus_complete.evidence_layer.ingest(list(corpus_complete.v4_sources.values()))
    c_uids = {u.proposition["predicate"] + ":" + str((u.proposition.get("value") or {}).get("value")): u.unit_id for u in corpus_complete.evidence_layer.units}
    complete_graph = JustificationGraph(
        claims={
            "c_prin": ClaimNode(
                "c_prin", "extraction", "Class C Notes", "note_principal", 50000000, "USD",
                evidence=({"document_id": clean_ind_id, "start": clean_ind.index("Class C Notes"), "end": clean_ind.index("Class C Notes ($50,000,000 principal)") + len("Class C Notes ($50,000,000 principal)"), "quote": "Class C Notes ($50,000,000 principal)"},),
                evidence_unit_ids=(c_uids["note_principal:50000000"],),
            ),
            "c_ind": ClaimNode(
                "c_ind", "extraction", "Class C Notes", "coupon_rate", 6.75, "percent_per_annum",
                evidence=({"document_id": clean_ind_id, "start": clean_ind.index("Class C Notes"), "end": len(clean_ind), "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum."},),
                evidence_unit_ids=(c_uids["coupon_rate:6.75"],),
            ),
            "c_om": ClaimNode(
                "c_om", "extraction", "Class C Notes", "coupon_rate", 6.5, "percent_per_annum",
                evidence=({"document_id": clean_om_id, "start": clean_om.index("Class C Notes"), "end": clean_om.index("6.50% per annum.") + len("6.50% per annum."), "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum."},),
                evidence_unit_ids=(c_uids["coupon_rate:6.5"],),
            ),
            "c_spread": ClaimNode(
                "c_spread", "extraction", "Class C Notes", "coupon_spread", 0.25, "percentage_point",
                evidence=({"document_id": clean_om_id, "start": clean_om.index("The coupon difference"), "end": len(clean_om), "quote": "The coupon difference is 0.25 percentage point."},),
                evidence_unit_ids=(c_uids["coupon_spread:0.25"],),
            ),
            "c_impact": ClaimNode(
                "c_impact", "derivation", "Class C Notes", "annual_interest_impact", 125000, "USD_per_year",
            ),
        },
        steps=(
            InferenceStep(
                step_id="s1",
                premises=("c_prin", "c_spread"),
                conclusion_id="c_impact",
                operation="multiply",
                assumptions=("constant_principal", "simple_annual_calculation"),
            ),
        ),
        conclusion_id="c_impact",
    )
    complete_candidate = {
        "report_text": (
            f"Based on the provided documents, {clean_ind_id} governs over {clean_om_id}. "
            "Class C Notes principal is $50,000,000 and bear interest at 6.75% per annum "
            "(whereas the Draft Offering Memorandum states 6.50% per annum, a coupon difference of 0.25 percentage point). "
            "Assuming constant principal and simple annual calculation, "
            "Class C Notes annual interest impact is $125,000 per year (illustrative only)."
        ),
        "assertions": [
            {
                "subject": "Class C Notes",
                "predicate": "note_principal",
                "value": "$50,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": clean_ind_id,
                "quote": "Class C Notes ($50,000,000 principal)",
            },
            {
                "subject": "Class C Notes",
                "predicate": "coupon_rate",
                "value": "6.75%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": clean_ind_id,
                "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.",
            },
            {
                "subject": "Class C Notes",
                "predicate": "coupon_rate",
                "value": "6.50%",
                "unit": "PERCENT",
                "role": "subordinate_discrepancy",
                "source_id": clean_om_id,
                "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum.",
            },
        ],
        "graph": complete_graph,
    }
    good_res_complete = evaluate_securitization_candidate(complete_candidate, corpus_complete, gate_mode="v3_v4")
    assert good_res_complete.report_disposition == "ACCEPT"
    assert good_res_complete.accepted is True
    assert good_res_complete.operational_authorization["status"] == "ESCALATE"
    assert good_res_complete.operational_authorization["authorized"] is False
    assert good_res_complete.operational_authorization["response_digest"] is not None


def test_08_repair_introducing_new_ungrounded_claim_or_dropping_section_rejected():
    corpus = build_synthetic_securitization_corpus()
    initial = _make_valid_synthetic_candidate(
        report_text=(
            "Indenture governs. Section 5.01(a) Step 13 Reserve Account Replenishment; Class C Notes 6.75% vs OM 6.50% "
            "equals $125,000 per year. Grace period 5 business days. Clean-up call 10%."
        )
    )

    def sloppy_repair(prev, res, iteration):
        # Adds constant_principal note, but drops clean_up_call & adds unqualified damages + Section 5.04!
        return _make_valid_synthetic_candidate(
            report_text=(
                "Section 5.04(a) Step 13 Reserve Account Replenishment; Class C Notes 6.75% vs 6.50% assumes constant principal, "
                "causing $125,000 per year in automatic legal damages. Grace period 5 business days."
            ),
            assertions=prev["assertions"][:3],  # Dropped clean_up_call!
        )

    loop_res = run_securitization_repair_loop(
        corpus=corpus,
        initial_candidate=initial,
        repair_fn=sloppy_repair,
        gate_mode="v3_v4",
        required_sections=["coupon", "waterfall", "grace_period", "clean_up_call"],
        max_iterations=2,
    )
    assert not loop_res["accepted"]
    assert loop_res["final_disposition"] != "ACCEPT"
    second_codes = set(loop_res["attempts"][1]["issue_codes"])
    assert "MISSING_REQUIRED_PREDICATE" in second_codes or "MISSING_REQUIRED_SECTION" in second_codes
    assert "UNQUALIFIED_DAMAGES_OR_CASHFLOW_CLAIM" in second_codes


def test_09_valid_graph_paired_with_wrong_prose_equation_rejected():
    corpus = build_synthetic_securitization_corpus()
    doc_id = "cart-2024-2-indenture.docx"
    om_id = "cart-2024-2-offering-memorandum.docx"
    q_prin = "Class C Notes ($50,000,000 principal)"
    s_prin = corpus.v4_sources[doc_id].text.index(q_prin)
    q_spread = "25 bps / 0.25 percentage points"
    s_spread = corpus.v4_sources[om_id].text.index(q_spread)
    graph = JustificationGraph(
        claims={
            "c_prin": ClaimNode(
                "c_prin",
                "extraction",
                "Class C Notes",
                "note_principal",
                50000000,
                "USD",
                evidence=({"document_id": doc_id, "start": s_prin, "end": s_prin + len(q_prin), "quote": q_prin},),
            ),
            "c_spread": ClaimNode(
                "c_spread",
                "extraction",
                "Class C Notes",
                "coupon_spread",
                25,
                "basis_points_per_annum",
                evidence=({"document_id": om_id, "start": s_spread, "end": s_spread + len(q_spread), "quote": q_spread},),
            ),
            "c_impact": ClaimNode(
                "c_impact",
                "derivation",
                "Class C Notes",
                "annual_interest_impact",
                125000,
                "USD_per_year",
                assumptions=("constant_principal", "simple_annual_calculation"),
            ),
        },
        steps=(
            InferenceStep(
                step_id="s1",
                premises=("c_prin", "c_spread"),
                conclusion_id="c_impact",
                operation="multiply",
                assumptions=("constant_principal", "simple_annual_calculation"),
            ),
        ),
        conclusion_id="c_impact",
    )
    wrong_prose = _make_valid_synthetic_candidate(
        report_text=(
            "Indenture Section 5.01(a) governs. Class C Notes 6.75% vs OM 6.50% (25 bps). "
            "Assuming constant principal as a simple annual calculation, 50000000 * 0.0025 = 50000 USD per year. "
            "Reserve replenishment Step 13; grace period 5 business days; clean-up call 10%."
        ),
        graph=graph,
    )
    res = evaluate_securitization_candidate(wrong_prose, corpus, gate_mode="v3_v4")
    assert not res.accepted
    assert res.report_disposition == "RETRY"
    assert any(
        i["code"]
        in (
            "EQUATION_MISMATCH",
            "VALUE_MISMATCH_IN_RESPONSE",
            "CALCULATION_ALTERED_OR_INCONSISTENT",
            "UNVERIFIED_EXPRESSION_IN_RESPONSE",
            "RELATIONAL_VALUE_MISMATCH",
        )
        for i in res.issues
    )


def test_10_extractor_failure_and_budget_exhaustion_never_accept_and_provenance_isolated():
    corpus = build_synthetic_securitization_corpus()

    # Extractor failure -> RETRY / ESCALATE, never ACCEPT
    ext_fail = run_securitization_repair_loop(
        corpus=corpus,
        initial_candidate={"force_extraction_failure": True, "report_text": "corrupt"},
        repair_fn=lambda prev, res, it: {"force_extraction_failure": True, "report_text": f"still corrupt {it}"},
        gate_mode="v3_v4",
        max_iterations=2,
    )
    assert not ext_fail["accepted"]
    assert ext_fail["final_disposition"] == "ESCALATE"
    assert ext_fail["attempts"][0]["issue_codes"] == ["EXTRACTION_FAILED"]

    # Budget exhaustion on persistently flawed candidate -> never converts RETRY to ACCEPT
    always_flawed = _make_valid_synthetic_candidate(
        report_text="Indenture governs, $125,000/yr automatic legal damages, Section 5.04, Class B 6.75%, Step 13, 5 business days, 10%."
    )
    budget_res = run_securitization_repair_loop(
        corpus=corpus,
        initial_candidate=always_flawed,
        repair_fn=lambda prev, res, it: always_flawed,  # stagnant identical output
        gate_mode="v3_v4",
        max_iterations=3,
    )
    assert not budget_res["accepted"]
    assert budget_res["final_disposition"] in ("ESCALATE", "ABSTAIN")

    # Full-document SHA-256 verification and evaluator_only access barrier
    cart_full = load_case_documents("cart_2024_2")
    assert cart_full.common_characters == 252848
    assert len(cart_full.documents) == 3
    bf_full = load_case_documents("banking_finance_ridgeline")
    assert bf_full.common_characters == 204579
    assert len(bf_full.documents) == 5
    with pytest.raises(PermissionError):
        assert_no_evaluator_only_access(_HARNESS_DIR / "evaluator_only" / "cart_2024_2_ground_truth.json")
