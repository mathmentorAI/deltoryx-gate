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
        "Step 13, junior to Class B and Class C principal (steps 11 and 12), whereas the OM puts Reserve replenishment ahead at Step 6/10.\n"
        "3. Events of Default Grace Period (Section 10.01): 5 business days (not 5 calendar days or 3 business days).\n"
        "4. Optional Redemption Clean-Up Call (Section 12.01): 10% of Initial Pool Balance (versus 15% in draft OM excerpt)."
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
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_b), "end": ind_txt.index(q_b) + len(q_b) + 0, "quote": q_b},),
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
            "c_wf": ClaimNode(
                "c_wf",
                "extraction",
                "Waterfall",
                "waterfall_step",
                "Step 13",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_wf), "end": ind_txt.index(q_wf) + len(q_wf), "quote": q_wf},),
            ),
            "c_wf_om": ClaimNode(
                "c_wf_om",
                "extraction",
                "Waterfall",
                "waterfall_step",
                "Step 6/10",
                evidence=({"document_id": om_id, "start": om_txt.index(q_wf_om), "end": om_txt.index(q_wf_om) + len(q_wf_om), "quote": q_wf_om},),
            ),
            "c_gp": ClaimNode(
                "c_gp",
                "extraction",
                "Grace Period",
                "grace_period",
                5,
                "business_days",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_gp), "end": ind_txt.index(q_gp) + len(q_gp), "quote": q_gp},),
            ),
            "c_gp_om_cal": ClaimNode(
                "c_gp_om_cal",
                "extraction",
                "Grace Period",
                "grace_period",
                5,
                "calendar_days",
                evidence=({"document_id": om_id, "start": om_txt.index(q_gp_om), "end": om_txt.index(q_gp_om) + len(q_gp_om), "quote": q_gp_om},),
            ),
            "c_gp_om_bd": ClaimNode(
                "c_gp_om_bd",
                "extraction",
                "Grace Period",
                "grace_period",
                3,
                "business_days",
                evidence=({"document_id": om_id, "start": om_txt.index(q_gp_om), "end": om_txt.index(q_gp_om) + len(q_gp_om), "quote": q_gp_om},),
            ),
            "c_cc": ClaimNode(
                "c_cc",
                "extraction",
                "Clean-Up Call",
                "cleanup_call_threshold",
                10,
                "percent",
                evidence=({"document_id": ind_id, "start": ind_txt.index(q_cc), "end": ind_txt.index(q_cc) + len(q_cc), "quote": q_cc},),
            ),
            "c_cc_om": ClaimNode(
                "c_cc_om",
                "extraction",
                "Clean-Up Call",
                "cleanup_call_threshold",
                15,
                "percent",
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
    assert good_res.report_disposition == "ACCEPT"
    assert good_res.accepted is True
    assert good_res.operational_authorization["status"] == "ESCALATE"
    assert good_res.operational_authorization["authorized"] is False
    assert good_res.operational_authorization["response_digest"] is not None


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
