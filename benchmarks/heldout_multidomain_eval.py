#!/usr/bin/env python3
"""Blind held-out multi-domain evaluation & statistical SLA suite (Phase 3).

Evaluates ClaimLayer v3 and v3+v4 across unseen held-out corpora in two domains:
  1. Securitization (Meridian Auto Receivables Trust 2026-1)
  2. Banking & Leveraged Finance (Apex Senior Secured Credit Facility 2026)

Strictly separated from known development fixtures (`cart_2024_2`,
`banking_finance_ridgeline`). Supports n >= 3 repetitions, SLA gating
(false_accepts == 0, convergence_rate >= 0.90, false_reject_rate <= 0.10),
and blind adjudication sheet generation for independent human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _ROOT / "src"
_HARNESS_DIR = _ROOT / "examples" / "harness2_securitization"
for _p in (str(_SRC_DIR), str(_HARNESS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from claimlayer.v4 import RecordedProposalReader  # noqa: E402
from harness2_v4_pipeline import (  # noqa: E402
    _build_corpus_objects,
    evaluate_domain_candidate,
    run_securitization_repair_loop,
)


def _build_meridian_securitization_corpus(*, reader: Any = None):
    docs = [
        (
            "D1",
            "meridian-2026-1-indenture.docx",
            "governing_operative_agreement",
            "authoritative",
            100,
            True,
            1,
            (
                "Meridian Auto Receivables Trust 2026-1 Executed Indenture. "
                "Section 2.04 Interest Rates: Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum. "
                "Section 5.01(a) Waterfall: Step 11 Class B principal; Step 12 Reserve Account replenishment. "
                "Section 10.01 Grace Period: 5 business days cure period."
            ),
        ),
        (
            "D2",
            "meridian-2026-1-draft-oc.docx",
            "disclosure_document",
            "standard",
            60,
            False,
            20,
            (
                "Meridian Draft Offering Circular: Class B Notes bear interest at 5.55% per annum. "
                "Reserve Account replenishment occurs at Step 8 ahead of Class B principal."
            ),
        ),
    ]
    eff_reader = (
        reader
        if reader is not None
        else RecordedProposalReader.from_file(_HARNESS_DIR / "recorded_reads" / "meridian.json")
    )
    corpus = _build_corpus_objects(
        case_id="heldout_sec_meridian_2026",
        task_id="structured-finance-securitization/heldout-meridian-2026",
            dataset_role="unseen_heldout_case",
        docs_cfg=docs,
        common_sha256=hashlib.sha256(b"meridian_2026_1_heldout").hexdigest(),
        common_chars=480,
        total_lines=30,
        reader=eff_reader,
    )
    corpus.evidence_layer.ingest(list(corpus.v4_sources.values()))
    return corpus


def _build_apex_banking_corpus(*, reader: Any = None):
    docs = [
        (
            "D1",
            "apex-lender-markup.docx",
            "counterparty_markup",
            "standard",
            60,
            False,
            1,
            "Counterparty Markup: Lender proposes Applicable Margin of 4.90% per annum and Maximum Net Leverage of 3.75x.",
        ),
        (
            "D2",
            "apex-negotiation-playbook.docx",
            "client_playbook_policy",
            "authoritative",
            95,
            True,
            10,
            "Client Playbook: Maximum acceptable Applicable Margin is 4.50% per annum; target Maximum Net Leverage is 4.25x.",
        ),
        (
            "D3",
            "apex-baseline-term-sheet.docx",
            "baseline_term_sheet",
            "authoritative",
            90,
            True,
            20,
            "Baseline Term Sheet: Senior Secured Term Loan Facility Commitment is $150,000,000 at an Applicable Margin of 4.25% per annum.",
        ),
        (
            "D4",
            "apex-partner-scope.eml",
            "client_instructions_scope",
            "standard",
            70,
            False,
            30,
            "Partner Instructions: Focus review on Term Loan Facility Commitment, Applicable Margin, and LTM EBITDA.",
        ),
        (
            "D5",
            "apex-financial-summary.xlsx",
            "quantitative_financial_model",
            "authoritative",
            100,
            True,
            40,
            "Financial Summary: Apex Consolidated Adjusted EBITDA is $30,000,000 for LTM Q4.",
        ),
    ]
    eff_reader = (
        reader
        if reader is not None
        else RecordedProposalReader.from_file(_HARNESS_DIR / "recorded_reads" / "apex.json")
    )
    corpus = _build_corpus_objects(
        case_id="heldout_bank_apex_2026",
        task_id="banking-finance/heldout-apex-2026",
        dataset_role="unseen_heldout_case",
        docs_cfg=docs,
        common_sha256=hashlib.sha256(b"apex_2026_heldout").hexdigest(),
        common_chars=620,
        total_lines=50,
    )
    from claimlayer.v4.domains.banking_finance import BANKING_FINANCE_ACCEPTANCE_POLICY, BANKING_FINANCE_VOCABULARY
    from claimlayer.v4.engine import EvidenceLayer
    bank_layer = EvidenceLayer(
        vocabulary=BANKING_FINANCE_VOCABULARY,
        policy=BANKING_FINANCE_ACCEPTANCE_POLICY,
        reader=eff_reader,
    )
    bank_layer.ingest(list(corpus.v4_sources.values()))
    from dataclasses import replace
    return replace(corpus, evidence_layer=bank_layer)


def _ev_ref(corpus, doc_id: str, quote: str) -> dict[str, Any]:
    src = corpus.v4_sources[doc_id]
    start = src.text.index(quote)
    return {
        "document_id": doc_id,
        "quote": quote,
        "revision": src.revision,
        "start": start,
        "end": start + len(quote),
    }


def _build_heldout_scenarios() -> list[dict[str, Any]]:
    sec = _build_meridian_securitization_corpus()
    sec_q_prin = "Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum."
    sec_q_draft = "Class B Notes bear interest at 5.55% per annum."
    sec_q_wf = "Section 5.01(a) Waterfall: Step 11 Class B principal; Step 12 Reserve Account replenishment."
    sec_q_gp = "Section 10.01 Grace Period: 5 business days cure period."
    sec_valid_graph = {
        "claims": {
            "c_prin": {
                "claim_id": "c_prin",
                "kind": "extraction",
                "subject": "Class B Notes",
                "predicate": "note_principal",
                "value": 40000000,
                "unit": "USD",
                "evidence": [_ev_ref(sec, "meridian-2026-1-indenture.docx", sec_q_prin)],
            },
            "c_coup_ind": {
                "claim_id": "c_coup_ind",
                "kind": "extraction",
                "subject": "Class B Notes",
                "predicate": "coupon_rate",
                "value": 5.80,
                "unit": "percent_per_annum",
                "evidence": [_ev_ref(sec, "meridian-2026-1-indenture.docx", sec_q_prin)],
            },
            "c_coup_draft": {
                "claim_id": "c_coup_draft",
                "kind": "extraction",
                "subject": "Class B Notes",
                "predicate": "coupon_rate",
                "value": 5.55,
                "unit": "percent_per_annum",
                "evidence": [_ev_ref(sec, "meridian-2026-1-draft-oc.docx", sec_q_draft)],
            },
            "c_wf": {
                "claim_id": "c_wf",
                "kind": "extraction",
                "subject": "Class B Notes",
                "predicate": "waterfall_priority",
                "value": "mezzanine",
                "condition_status": "conditional",
                "evidence": [_ev_ref(sec, "meridian-2026-1-indenture.docx", sec_q_wf)],
            },
            "c_gp": {
                "claim_id": "c_gp",
                "kind": "extraction",
                "subject": "Meridian Auto Receivables Trust 2026-1",
                "predicate": "cure_period",
                "value": 5,
                "unit": "business_days",
                "condition_status": "conditional",
                "evidence": [_ev_ref(sec, "meridian-2026-1-indenture.docx", sec_q_gp)],
            },
        },
        "steps": [],
        "conclusion_id": "c_coup_ind",
    }
    sec_clean = {
        "report_text": (
            "Executed Indenture governs over the Draft Offering Circular. "
            "Class B Notes principal is 40000000 USD. "
            "Class B Notes coupon is 5.80% per annum per the executed Indenture "
            "(the Draft Offering Circular states 5.55% per annum for Class B). "
            "Class B Notes waterfall priority is mezzanine. "
            "Meridian Auto Receivables Trust 2026-1 cure period is 5 business days."
        ),
        "assertions": [
            {
                "subject": "Class B Notes",
                "predicate": "note_principal",
                "value": "$40,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "meridian-2026-1-indenture.docx",
                "quote": sec_q_prin,
            },
            {
                "subject": "Class B Notes",
                "predicate": "coupon_rate",
                "value": "5.80%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": "meridian-2026-1-indenture.docx",
                "quote": sec_q_prin,
            },
            {
                "subject": "Class B Notes",
                "predicate": "coupon_rate",
                "value": "5.55%",
                "unit": "PERCENT",
                "role": "subordinate_discrepancy",
                "source_id": "meridian-2026-1-draft-oc.docx",
                "quote": "Class B Notes bear interest at 5.55% per annum.",
            },
        ],
        "graph": sec_valid_graph,
        "metadata": {"has_material_discrepancies": True},
    }
    sec_bad_auth = {
        "report_text": "Class B Notes coupon is 5.55% per Draft OC.",
        "assertions": [
            {
                "subject": "Class B Notes",
                "predicate": "coupon_rate",
                "value": "5.55%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": "meridian-2026-1-draft-oc.docx",
                "quote": "Class B Notes bear interest at 5.55% per annum.",
            }
        ],
        "graph": sec_valid_graph,
    }
    sec_bad_eq = {
        "report_text": "According to the text, Class B Notes is 40000000 USD. 40000000 * 0.0025 = 50000 USD.",
        "assertions": sec_clean["assertions"],
        "graph": sec_valid_graph,
    }
    sec_bad_quote = {
        "report_text": "According to the text, Class B Notes is 40000000 USD.",
        "assertions": [
            {
                "subject": "Class B Notes",
                "predicate": "note_principal",
                "value": "$40,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "meridian-2026-1-indenture.docx",
                "quote": "Fabricated quote not in Meridian Indenture.",
            }
        ],
        "graph": sec_valid_graph,
    }

    bank = _build_apex_banking_corpus()
    bank_q_fac = "Senior Secured Term Loan Facility Commitment is $150,000,000"
    bank_q_ebitda = "Financial Summary: Apex Consolidated Adjusted EBITDA is $30,000,000 for LTM Q4."
    bank_q_pb_margin = "Maximum acceptable Applicable Margin is 4.50% per annum"
    bank_q_base_margin = "at an Applicable Margin of 4.25% per annum"
    bank_q_mk_margin = "Lender proposes Applicable Margin of 4.90% per annum"
    bank_q_pb_lev = "target Maximum Net Leverage is 4.25x"
    bank_q_mk_lev = "Maximum Net Leverage of 3.75x"
    bank_q_mk_dev = "Counterparty Markup: Lender proposes"
    bank_valid_graph = {
        "claims": {
            "c_fac": {
                "claim_id": "c_fac",
                "kind": "extraction",
                "subject": "Term Loan",
                "predicate": "facility_amount",
                "value": 150000000,
                "unit": "USD",
                "evidence": [_ev_ref(bank, "apex-baseline-term-sheet.docx", bank_q_fac)],
            },
            "c_ebitda": {
                "claim_id": "c_ebitda",
                "kind": "extraction",
                "subject": "Apex Consolidated",
                "predicate": "ebitda",
                "value": 30000000,
                "unit": "USD",
                "evidence": [_ev_ref(bank, "apex-financial-summary.xlsx", bank_q_ebitda)],
            },
            "c_m_pb": {
                "claim_id": "c_m_pb",
                "kind": "extraction",
                "subject": "Client Playbook",
                "predicate": "interest_margin",
                "value": 4.50,
                "unit": "percent_per_annum",
                "evidence": [_ev_ref(bank, "apex-negotiation-playbook.docx", bank_q_pb_margin)],
            },
            "c_m_base": {
                "claim_id": "c_m_base",
                "kind": "extraction",
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": 4.25,
                "unit": "percent_per_annum",
                "evidence": [_ev_ref(bank, "apex-baseline-term-sheet.docx", bank_q_base_margin)],
            },
            "c_m_mk": {
                "claim_id": "c_m_mk",
                "kind": "extraction",
                "subject": "Lender",
                "predicate": "interest_margin",
                "value": 4.90,
                "unit": "percent_per_annum",
                "evidence": [_ev_ref(bank, "apex-lender-markup.docx", bank_q_mk_margin)],
            },
            "c_l_pb": {
                "claim_id": "c_l_pb",
                "kind": "extraction",
                "subject": "Client Playbook",
                "predicate": "leverage_ratio",
                "value": 4.25,
                "unit": "multiple",
                "evidence": [_ev_ref(bank, "apex-negotiation-playbook.docx", bank_q_pb_lev)],
            },
            "c_l_mk": {
                "claim_id": "c_l_mk",
                "kind": "extraction",
                "subject": "Lender",
                "predicate": "leverage_ratio",
                "value": 3.75,
                "unit": "multiple",
                "evidence": [_ev_ref(bank, "apex-lender-markup.docx", bank_q_mk_lev)],
            },
            "c_dev": {
                "claim_id": "c_dev",
                "kind": "extraction",
                "subject": "Lender",
                "predicate": "markup_deviation",
                "value": "material_deviation",
                "evidence": [_ev_ref(bank, "apex-lender-markup.docx", bank_q_mk_dev)],
            },
        },
        "steps": [],
        "conclusion_id": "c_fac",
    }
    bank_clean = {
        "report_text": (
            "According to the source documents, Term Loan Commitment is 150000000 USD per Baseline Term Sheet, and "
            "Apex Consolidated Adjusted EBITDA is 30000000 USD per Financial Summary. "
            "Client Playbook governs over Counterparty Markup and Baseline Term Sheet: "
            "Client Playbook interest margin is 4.50% per annum (Term Loan baseline interest margin is 4.25% per annum; "
            "Lender markup interest margin is 4.90% per annum, Lender is material_deviation); "
            "Client Playbook target Maximum Net Leverage is 4.25x, governing over Lender Maximum Net Leverage of 3.75x."
        ),
        "assertions": [
            {
                "subject": "Term Loan",
                "predicate": "facility_amount",
                "value": "$150,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "apex-baseline-term-sheet.docx",
                "quote": bank_q_fac,
            },
            {
                "subject": "Apex",
                "predicate": "ebitda",
                "value": "$30,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "apex-financial-summary.xlsx",
                "quote": bank_q_ebitda,
            },
            {
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": "4.90%",
                "unit": "PERCENT",
                "role": "subordinate_discrepancy",
                "source_id": "apex-lender-markup.docx",
                "quote": bank_q_mk_margin,
            },
        ],
        "graph": bank_valid_graph,
        "metadata": {"has_material_discrepancies": True},
    }
    bank_bad_markup = {
        "report_text": "Lender markup governs over negotiation playbook; Applicable Margin is 4.90%.",
        "assertions": [
            {
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": "4.90%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": "apex-lender-markup.docx",
                "quote": "Lender proposes Applicable Margin of 4.90% per annum",
            }
        ],
        "graph": bank_valid_graph,
    }
    bank_bad_unit = {
        "report_text": "According to the text, the Term Loan is 150000000 USD. Leverage is 3.75% x.",
        "assertions": bank_clean["assertions"],
        "graph": bank_valid_graph,
    }

    return [
        {
            "scenario_id": "sec_meridian_clean",
            "domain": "securitization",
            "corpus": sec,
            # meridian.json genuinely misses Step 8 Reserve Account waterfall in draft-oc.docx,
            # and this candidate omits the waterfall discrepancy -> v3_v4 MUST NOT accept (ESCALATE).
            "kind": "seeded_defect",
            "initial": sec_clean,
            "repair": sec_clean,
        },
        {
            "scenario_id": "sec_meridian_defect_authority",
            "domain": "securitization",
            "corpus": sec,
            "kind": "seeded_defect",
            "initial": sec_bad_auth,
            "repair": sec_bad_auth,
        },
        {
            "scenario_id": "sec_meridian_defect_false_equation",
            "domain": "securitization",
            "corpus": sec,
            "kind": "seeded_defect",
            "initial": sec_bad_eq,
            "repair": sec_bad_eq,
        },
        {
            "scenario_id": "sec_meridian_defect_fabricated_quote",
            "domain": "securitization",
            "corpus": sec,
            "kind": "seeded_defect",
            "initial": sec_bad_quote,
            "repair": sec_bad_quote,
        },
        {
            "scenario_id": "bank_apex_clean",
            "domain": "banking_finance",
            "corpus": bank,
            "kind": "clean_control",
            "initial": bank_clean,
            "repair": bank_clean,
        },
        {
            "scenario_id": "bank_apex_repairable_markup",
            "domain": "banking_finance",
            "corpus": bank,
            "kind": "repairable",
            "initial": bank_bad_markup,
            "repair": bank_clean,
        },
        {
            "scenario_id": "bank_apex_defect_markup_override",
            "domain": "banking_finance",
            "corpus": bank,
            "kind": "seeded_defect",
            "initial": bank_bad_markup,
            "repair": bank_bad_markup,
        },
        {
            "scenario_id": "bank_apex_defect_ratio_unit",
            "domain": "banking_finance",
            "corpus": bank,
            "kind": "seeded_defect",
            "initial": bank_bad_unit,
            "repair": bank_bad_unit,
        },
    ]


def run_heldout_evaluation(
    *,
    repetitions: int = 3,
    out_dir: Path | str | None = None,
    seed: int = 20261115,
) -> dict[str, Any]:
    """Run n >= 3 repetitions across unseen held-out scenarios and compute SLA stats."""
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    scenarios = _build_heldout_scenarios()
    arms = ("v3_only", "v3_v4")
    arm_runs: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    blind_items: list[dict[str, Any]] = []

    for rep in range(1, repetitions + 1):
        for arm in arms:
            false_accepts = 0
            seeded_total = 0
            false_rejects = 0
            clean_total = 0
            converged = 0
            repairable_total = 0
            latencies: list[float] = []
            details: list[dict[str, Any]] = []

            for sc in scenarios:
                corpus = sc["corpus"]
                assert corpus.dataset_role == "unseen_heldout_case"
                if sc["kind"] == "repairable":
                    repairable_total += 1
                    loop_res = run_securitization_repair_loop(
                        corpus=corpus,
                        initial_candidate=sc["initial"],
                        repair_fn=lambda _c, _fb, _it, r=sc["repair"]: r,
                        gate_mode=arm,
                        max_iterations=3,
                    )
                    accepted = bool(loop_res["accepted"])
                    disp = loop_res["final_disposition"]
                    ms = float(loop_res["verifier_ms"])
                    if accepted:
                        converged += 1
                    else:
                        false_rejects += 1
                    clean_total += 1
                else:
                    ver = evaluate_domain_candidate(
                        sc["initial"],
                        corpus,
                        gate_mode=arm,
                        require_graph=(arm == "v3_v4"),
                        require_assertions=True,
                    )
                    accepted = bool(ver.accepted)
                    disp = ver.report_disposition
                    ms = float(ver.verifier_ms)
                    if sc["kind"] == "clean_control":
                        clean_total += 1
                        if not accepted:
                            false_rejects += 1
                    elif sc["kind"] == "seeded_defect":
                        seeded_total += 1
                        if accepted:
                            false_accepts += 1

                latencies.append(ms)
                details.append({
                    "scenario_id": sc["scenario_id"],
                    "domain": sc["domain"],
                    "kind": sc["kind"],
                    "accepted": accepted,
                    "disposition": disp,
                    "verifier_ms": round(ms, 3),
                })
                if rep == 1:
                    blind_items.append({
                        "arm": arm,
                        "scenario_id": sc["scenario_id"],
                        "domain": sc["domain"],
                        "report_text": sc["repair"]["report_text"] if sc["kind"] == "repairable" else sc["initial"]["report_text"],
                        "system_disposition": disp,
                    })

            arm_runs[arm].append({
                "repetition": rep,
                "false_accepts": false_accepts,
                "seeded_total": seeded_total,
                "false_rejects": false_rejects,
                "clean_total": clean_total,
                "converged": converged,
                "repairable_total": repairable_total,
                "mean_verifier_ms": round(statistics.mean(latencies), 3),
                "details": details,
            })

    summary_by_arm: dict[str, Any] = {}
    for arm, runs in arm_runs.items():
        fa_total = sum(r["false_accepts"] for r in runs)
        sd_total = sum(r["seeded_total"] for r in runs)
        fr_total = sum(r["false_rejects"] for r in runs)
        cl_total = sum(r["clean_total"] for r in runs)
        cv_total = sum(r["converged"] for r in runs)
        rp_total = sum(r["repairable_total"] for r in runs)
        ms_vals = [r["mean_verifier_ms"] for r in runs]
        fa_rate = (fa_total / sd_total) if sd_total else 0.0
        fr_rate = (fr_total / cl_total) if cl_total else 0.0
        conv_rate = (cv_total / rp_total) if rp_total else 1.0
        sla_ok = (fa_total == 0) and (conv_rate >= 0.90) and (fr_rate <= 0.10)
        summary_by_arm[arm] = {
            "repetitions": repetitions,
            "false_accepts": fa_total,
            "seeded_total": sd_total,
            "false_accept_rate": round(fa_rate, 4),
            "false_rejects": fr_total,
            "clean_total": cl_total,
            "false_reject_rate": round(fr_rate, 4),
            "repair_convergence_rate": round(conv_rate, 4),
            "verifier_ms_mean": round(statistics.mean(ms_vals), 3),
            "verifier_ms_stdev": round(statistics.stdev(ms_vals), 3) if len(ms_vals) > 1 else 0.0,
            "sla_passed": sla_ok,
        }

    rng = random.Random(seed)
    rng.shuffle(blind_items)
    adjudication_sheet = []
    adjudication_key = []
    for idx, item in enumerate(blind_items, start=1):
        blind_id = f"BLIND-{idx:03d}"
        adjudication_sheet.append({
            "blind_id": blind_id,
            "domain": item["domain"],
            "report_text": item["report_text"],
            "reviewer_accepts_report": None,
            "reviewer_notes": "",
        })
        adjudication_key.append({
            "blind_id": blind_id,
            "arm": item["arm"],
            "scenario_id": item["scenario_id"],
            "system_disposition": item["system_disposition"],
        })

    result = {
        "dataset_role": "unseen_heldout_case",
        "domains": sorted({s["domain"] for s in scenarios}),
        "scenario_count": len(scenarios),
        "repetitions": repetitions,
        "sla_targets": {
            "max_false_accepts": 0,
            "min_convergence_rate": 0.90,
            "max_false_reject_rate": 0.10,
        },
        "summary_by_arm": summary_by_arm,
        "overall_sla_passed": summary_by_arm["v3_v4"]["sla_passed"],
        "v3_v4_strictly_better_than_v3_only": (
            summary_by_arm["v3_v4"]["false_accepts"] < summary_by_arm["v3_only"]["false_accepts"]
        ),
        "runs": arm_runs,
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "heldout_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (out_path / "adjudication_sheet.json").write_text(json.dumps(adjudication_sheet, indent=2) + "\n")
        (out_path / "adjudication_key.json").write_text(json.dumps(adjudication_key, indent=2) + "\n")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--out", default=".build/heldout_eval/latest")
    args = parser.parse_args()
    res = run_heldout_evaluation(repetitions=args.repetitions, out_dir=args.out)
    print(json.dumps({
        "dataset_role": res["dataset_role"],
        "domains": res["domains"],
        "repetitions": res["repetitions"],
        "overall_sla_passed": res["overall_sla_passed"],
        "summary_by_arm": res["summary_by_arm"],
    }, indent=2))
    return 0 if res["overall_sla_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
