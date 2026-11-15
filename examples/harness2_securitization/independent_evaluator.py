#!/usr/bin/env python3
"""Independent External Rubric Evaluator for Harness2 Case 1 (CART 2024-2) and Case 2 (Banking-Finance).

This module lives cleanly outside the agent/runtime boundary and is the ONLY component allowed
to read `evaluator_only/*.json` ground truth files. All metrics include explicit numerators
and denominators (`matched/total`, `errors/denominator`) so results remain fully auditable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

_EVAL_DIR = Path(__file__).resolve().parent / "evaluator_only"


def load_ground_truth(case_id: str) -> dict[str, Any]:
    """Load evaluator-only ground truth JSON for `cart_2024_2` or `banking_finance_ridgeline`."""
    mapping = {
        "cart_2024_2": _EVAL_DIR / "cart_2024_2_ground_truth.json",
        "cart_2024_2_synthetic_replay": _EVAL_DIR / "cart_2024_2_ground_truth.json",
        "banking_finance_ridgeline": _EVAL_DIR / "banking_finance_ground_truth.json",
    }
    if case_id not in mapping:
        raise KeyError(f"Unknown case_id '{case_id}' for independent evaluator.")
    return json.loads(mapping[case_id].read_text(encoding="utf-8"))


def _ratio(num: int, den: int) -> dict[str, Any]:
    return {
        "numerator": num,
        "denominator": den,
        "display": f"{num}/{den}",
        "rate": round(num / den, 4) if den > 0 else None,
    }


def _evaluate_cart_full_report(text: str, gt: Mapping[str, Any]) -> dict[str, Any]:
    """Score a CART 2024-2 full-document discrepancy report against all 8 ground-truth items and 3 email traps."""
    low = text.lower()
    discrepancy_hits: dict[str, bool] = {}

    # 1. Waterfall Reserve Replenishment Step 13 vs Step 10
    discrepancy_hits["CART-D1-WATERFALL"] = (
        ("reserve" in low)
        and bool(re.search(r"step\s*(?:thirteenth|13)\b", low))
        and bool(re.search(r"step\s*(?:tenth|10)\b", low))
    )
    # 2. Class C Coupon 6.75% vs 6.50% (and 25 bps / 0.25 pp)
    discrepancy_hits["CART-D2-CLASS-C-COUPON"] = (
        ("class c" in low)
        and ("6.75%" in low)
        and ("6.50%" in low)
    )
    # 3. Day Count 30/360 vs actual/360
    discrepancy_hits["CART-D3-DAY-COUNT"] = ("30/360" in low) and ("actual/360" in low or "actual number of days" in low)
    # 4. Servicing Fee Base (Pool Balance vs Note Balance)
    discrepancy_hits["CART-D4-SERVICING-FEE-BASE"] = (
        ("servicing fee" in low)
        and ("pool balance" in low)
        and ("note balance" in low or "outstanding note" in low)
    )
    # 5. Clean-Up Call Threshold ($71,230,000 / Initial Pool Balance vs $68,750,000 / Initial Note Balance)
    discrepancy_hits["CART-D5-CLEAN-UP-CALL"] = (
        ("clean-up" in low or "clean up" in low or "optional redemption" in low)
        and ("pool balance" in low)
        and ("note balance" in low)
        and ("71,230,000" in low or "68,750,000" in low or "2,480,000" in low or "10%" in low)
    )
    # 6. Reserve Account Floor (0.50% Initial Note Balance / $3,437,500)
    discrepancy_hits["CART-D6-RESERVE-FLOOR"] = (
        ("reserve account" in low)
        and ("0.50%" in low or "3,437,500" in low or "floor" in low)
    )
    # 7. CNL Trigger July 2027+ (6.00% vs 5.75%)
    discrepancy_hits["CART-D7-CNL-TRIGGER-FINAL"] = (
        ("6.00%" in low)
        and ("5.75%" in low)
        and ("cnl" in low or "cumulative net loss" in low or "trigger" in low)
    )
    # 8. Back-Up Servicer Omission (Granite Loan Servicing LLC & activation-conditioned fee)
    discrepancy_hits["CART-D8-BACKUP-SERVICER-OMISSION"] = (
        ("granite" in low or "back-up servicer" in low or "backup servicer" in low)
        and ("activation" in low or "standby" in low or "60,000" in low or "5,000" in low)
    )

    # Email traps corrected (3 total)
    trap_hits: dict[str, bool] = {}
    trap_hits["EMAIL-TRAP-1-WRONG-SECTION-CITE"] = (
        "5.01" in low
        and (
            "5.04" not in low
            or any(k in low for k in ("does not exist", "non-existent", "no section 5.04", "mis-cite", "miscite", "mistakenly", "incorrectly", "actually section 5.01"))
        )
    )
    # Trap 2: expanded scope beyond associate's 2 items
    trap_hits["EMAIL-TRAP-2-INCOMPLETE-SCOPE"] = sum(
        1 for k in ("CART-D3-DAY-COUNT", "CART-D4-SERVICING-FEE-BASE", "CART-D5-CLEAN-UP-CALL", "CART-D6-RESERVE-FLOOR", "CART-D7-CNL-TRIGGER-FINAL", "CART-D8-BACKUP-SERVICER-OMISSION")
        if discrepancy_hits.get(k)
    ) >= 4
    # Trap 3: Qualified constant-principal calculation, no automatic damages
    has_125 = "$125,000" in low or "125,000" in low
    has_qual = any(k in low for k in ("constant principal", "initial principal", "static principal", "simple annual", "annualized", "before amortization"))
    has_bad_dmg = bool(re.search(r"\b(?:automatic(?:ally)?|guaranteed|certain)\s+(?:legal\s+)?damages\b|\brealized\s+cashflow\s+loss", low))
    trap_hits["EMAIL-TRAP-3-UNQUALIFIED-DOLLAR-IMPACT"] = (not has_125 and not has_bad_dmg) or (has_125 and has_qual and not has_bad_dmg)

    # Error taxonomy & unsupported claims
    subject_errors = 0
    if re.search(r"class\s+b\s+notes?\s+[^.\n]{0,40}?(?:6\.75%|6\.50%)", low) or re.search(r"class\s+c\s+notes?\s+[^.\n]{0,40}?5\.95%", low):
        subject_errors += 1
    if "5.04" in low and not trap_hits["EMAIL-TRAP-1-WRONG-SECTION-CITE"]:
        subject_errors += 1

    authority_inversions = 0
    if re.search(r"(?:offering\s+memorandum|om|email)[^.\n]{0,60}(?:governs?|controls?|supersedes?|prevails?)[^.\n]{0,60}indenture", low):
        authority_inversions += 1
    if re.search(r"class\s+c\s+notes?[^.\n]{0,60}(?:governs?|governed|correct\s+rate|rate\s+is)\s+6\.50%", low):
        authority_inversions += 1

    value_errors = 0
    if authority_inversions > 0 or re.search(r"reserve\s+account[^.\n]{0,50}indenture[^.\n]{0,40}step\s*(?:10|tenth|6|sixth)\b", low):
        value_errors += 1

    unit_errors = 0
    if re.search(r"\bcalendar\s+days?\b[^.\n]{0,35}(?:=|same\s+as|equivalent\s+to)\s+[^.\n]{0,20}\bbusiness\s+days?\b", low):
        unit_errors += 1
    if re.search(r"\b25\s*%\s+(?:discrepancy|difference|spread|gap|higher|lower)\b|\b25\s*(?:bps|basis\s+points?)\s*=\s*25\s*%", low):
        unit_errors += 1

    condition_or_assumption_errors = 0
    if has_125 and not has_qual:
        condition_or_assumption_errors += 1
    if has_bad_dmg:
        condition_or_assumption_errors += 1
    if ("60,000" in low or "5,000" in low) and "activation" not in low and "standby" not in low and "conditional" not in low:
        condition_or_assumption_errors += 1

    priority_waterfall_errors = 0
    if (
        re.search(r"indenture[^.\n]{0,80}reserve\s+account[^.\n]{0,60}(?:senior\s+to|before|ahead\s+of)\s+class\s+[bc]", low)
        or re.search(r"step\s*(?:10|6)\s+(?:is\s+)?(?:in\s+)?the\s+indenture", low)
    ):
        priority_waterfall_errors += 1

    unsupported_or_false_claims = (
        subject_errors + authority_inversions + value_errors + unit_errors
        + condition_or_assumption_errors + priority_waterfall_errors
    )

    return {
        "discrepancy_hits": discrepancy_hits,
        "trap_hits": trap_hits,
        "authority_inversions": authority_inversions,
        "unsupported_or_false_claims": unsupported_or_false_claims,
        "subject_errors": subject_errors,
        "value_errors": value_errors,
        "unit_errors": unit_errors,
        "condition_or_assumption_errors": condition_or_assumption_errors,
        "priority_waterfall_errors": priority_waterfall_errors,
        "total_discrepancies": len(gt["full_document_discrepancies"]),
        "total_traps": len(gt["email_traps"]),
        "claim_slots_checked": 6,
    }


def _evaluate_cart_synthetic_report(text: str, gt: Mapping[str, Any]) -> dict[str, Any]:
    """Score synthetic_replay CART 2024-2 report against the 4 synthetic excerpt discrepancies."""
    low = text.lower()
    discrepancy_hits = {
        "SYN-COUPON": ("class c" in low and "6.75%" in low and ("6.50%" in low or "25 bps" in low or "0.25" in low)),
        "SYN-WATERFALL": ("reserve" in low and ("step 13" in low or "step thirteenth" in low or "junior" in low)),
        "SYN-GRACE-PERIOD": ("5 business days" in low and ("grace" in low or "event of default" in low or "interest" in low)),
        "SYN-CLEANUP-CALL": (("clean-up" in low or "clean up" in low or "optional redemption" in low) and "10%" in low),
    }
    base = _evaluate_cart_full_report(text, gt)
    # Synthetic excerpt doesn't have the 4 full-doc-only items; re-compute email traps specifically for synthetic mode
    has_125 = "$125,000" in low or "125,000" in low
    has_qual = any(k in low for k in ("constant principal", "initial principal", "simple annual", "annualized", "before amortization"))
    has_bad_dmg = bool(re.search(r"\b(?:automatic(?:ally)?|guaranteed)\s+(?:legal\s+)?damages\b", low))
    trap_hits = {
        "SYN-TRAP-AUTHORITY": base["authority_inversions"] == 0,
        "SYN-TRAP-COVERAGE": all(discrepancy_hits.values()),
        "SYN-TRAP-ASSUMPTIONS": (not has_125 and not has_bad_dmg) or (has_125 and has_qual and not has_bad_dmg),
    }
    return {
        **base,
        "discrepancy_hits": discrepancy_hits,
        "trap_hits": trap_hits,
        "total_discrepancies": len(gt["synthetic_replay_discrepancies"]),
        "total_traps": len(trap_hits),
    }


def _evaluate_banking_finance_report(text: str, gt: Mapping[str, Any]) -> dict[str, Any]:
    """Score Banking-Finance Ridgeline markup analysis report against the 12 ground-truth deviations."""
    low = text.lower()
    discrepancy_hits = {
        "BF-1-EXTENSION-OPTIONS": ("extension" in low and ("deleted" in low or "removed" in low or "two" in low or "2032" in low)),
        "BF-2-SOFR-FLOOR": ("sofr floor" in low and ("0.75%" in low or "75 bps" in low)),
        "BF-3-MARGIN-GRID-AND-RATCHET": (("margin" in low) and ("25 bps" in low or "225" in low or "annual" in low or "downward" in low)),
        "BF-4-COMMITMENT-FEE": ("commitment fee" in low and ("0.30%" in low or "0.40%" in low or "0.50%" in low or "grid" in low)),
        "BF-5-LEVERAGE-COVENANT": (("leverage" in low) and ("3.00x" in low or "step-down" in low or "accelerated" in low)),
        "BF-6-FCCR-COVENANT": (("fccr" in low or "fixed charge" in low) and ("1.35x" in low and "1.25x" in low)),
        "BF-7-EBITDA-ADD-BACKS": ("ebitda" in low and ("synerg" in low or "non-recurring" in low) and ("10%" in low or "$3" in low or "3,000,000" in low or "12-month" in low or "12 month" in low)),
        "BF-8-PERMITTED-ACQUISITIONS": (("acquisition" in low) and ("15,000,000" in low or "$15m" in low or "$15 million" in low or "0.25x" in low or "40m" in low or "40,000,000" in low)),
        "BF-9-RESTRICTED-PAYMENTS": (("restricted payment" in low or "distribution" in low) and ("2.50x" in low or "8,000,000" in low or "$8m" in low or "$8 million" in low)),
        "BF-10-MFN-CLAUSE": ("mfn" in low or "most favored nation" in low),
        "BF-11-ECF-SWEEP": (("excess cash flow" in low or "ecf" in low) and ("50%" in low and ("step-down" in low or "flat" in low or "sweep" in low))),
        "BF-12-QUIET-DEFINITIONS-AND-DEFAULTS": sum(1 for k in ("change of control", "cross-default", "cross default", "180", "365", "material adverse effect", "mae", "51%", "35%") if k in low) >= 3,
    }

    authority_inversions = 1 if re.search(r"lender\s+markup[^.\n]{0,50}(?:supersedes|governs?|accepted\s+as\s+is)\s+[^.\n]{0,40}playbook", low) else 0
    unit_errors = 1 if re.search(r"\b25\s*%\s+(?:margin\s+increase|grid\s+increase)\b|\b75\s*%\s+sofr\s+floor\b", low) else 0
    value_errors = 1 if re.search(r"original\s+term\s+sheet[^.\n]{0,40}0\.75%\s+sofr\s+floor", low) else 0
    subject_errors = 1 if re.search(r"minimum\s+fccr[^.\n]{0,30}(?:3\.75x|3\.50x)", low) else 0
    condition_or_assumption_errors = 1 if (
        ("$375" in low or "$1,394" in low or "$401" in low)
        and re.search(r"\b(?:automatic(?:ally)?|guaranteed)\s+(?:legal\s+)?damages\b", low)
    ) else 0

    unsupported = authority_inversions + unit_errors + value_errors + subject_errors + condition_or_assumption_errors
    return {
        "discrepancy_hits": discrepancy_hits,
        "trap_hits": {},
        "authority_inversions": authority_inversions,
        "unsupported_or_false_claims": unsupported,
        "subject_errors": subject_errors,
        "value_errors": value_errors,
        "unit_errors": unit_errors,
        "condition_or_assumption_errors": condition_or_assumption_errors,
        "priority_waterfall_errors": 0,
        "total_discrepancies": len(gt["discrepancies"]),
        "total_traps": 0,
        "claim_slots_checked": 5,
    }


def _score_single_report(case_id: str, text: str, gt: Mapping[str, Any]) -> dict[str, Any]:
    if case_id == "cart_2024_2_synthetic_replay":
        return _evaluate_cart_synthetic_report(text, gt)
    if case_id == "cart_2024_2":
        return _evaluate_cart_full_report(text, gt)
    if case_id == "banking_finance_ridgeline":
        return _evaluate_banking_finance_report(text, gt)
    raise KeyError(f"Unsupported case_id '{case_id}'")


def evaluate_arm_run(
    *,
    case_id: str,
    arm_name: str,
    mode: str,
    budget_regime: str,
    run_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute all Section 7 metrics with explicit denominators from a run result trajectory."""
    gt = load_ground_truth(case_id)
    initial_report = str(run_result.get("initial_report") or "")
    final_report = str(run_result.get("final_report") or "")
    final_disp = str(run_result.get("final_disposition") or "UNKNOWN")
    accepted = run_result.get("accepted")
    gate_assessed = isinstance(accepted, bool) and final_disp != "NOT_VERIFIED"
    op_auth = run_result.get("operational_authorization") or {}
    attempts: Sequence[Mapping[str, Any]] = run_result.get("attempts") or ()
    usage = run_result.get("usage") or {}
    cache_metrics = run_result.get("cache_metrics") or {}

    init_score = _score_single_report(case_id, initial_report, gt)
    final_score = _score_single_report(case_id, final_report, gt)

    total_disc = final_score["total_discrepancies"]
    matched_disc = sum(1 for v in final_score["discrepancy_hits"].values() if v)
    omissions = total_disc - matched_disc

    total_traps = final_score["total_traps"]
    matched_traps = sum(1 for v in final_score["trap_hits"].values() if v)

    init_total_errors = (
        (init_score["total_discrepancies"] - sum(1 for v in init_score["discrepancy_hits"].values() if v))
        + init_score["unsupported_or_false_claims"]
    )
    final_total_errors = omissions + final_score["unsupported_or_false_claims"]

    # Errors fixed vs introduced during repair
    init_missed = {k for k, v in init_score["discrepancy_hits"].items() if not v}
    final_missed = {k for k, v in final_score["discrepancy_hits"].items() if not v}
    fixed_disc = len(init_missed - final_missed)
    introduced_omissions = len(final_missed - init_missed)
    fixed_claims = max(0, init_score["unsupported_or_false_claims"] - final_score["unsupported_or_false_claims"])
    introduced_claims = max(0, final_score["unsupported_or_false_claims"] - init_score["unsupported_or_false_claims"])
    errors_fixed_in_repair = fixed_disc + fixed_claims
    errors_introduced_by_repair = introduced_omissions + introduced_claims

    # Verifier classification diagnostics across attempts & final verdict
    is_final_truly_acceptable = (
        omissions == 0
        and final_score["unsupported_or_false_claims"] == 0
        and (total_traps == 0 or matched_traps == total_traps)
    )
    false_accepts = 1 if (accepted and not is_final_truly_acceptable) else 0
    false_rejects = 1 if ((not accepted) and final_disp in ("RETRY", "ABSTAIN", "ESCALATE") and is_final_truly_acceptable) else 0
    abstentions_or_escalations = 1 if final_disp in ("ABSTAIN", "ESCALATE") else 0

    return {
        "case_id": case_id,
        "dataset_role": gt.get("dataset_role"),
        "arm": arm_name,
        "mode": mode,
        "budget_regime": budget_regime,
        "final_disposition": final_disp,
        "accepted": accepted,
        "stop_reason": run_result.get("stop_reason"),
        "operational_authorization_status": op_auth.get("status"),
        "operational_authorized": bool(op_auth.get("authorized", False)),
        "iterations_used": int(run_result.get("iterations_used", len(attempts))),
        "scorer_status": "heuristic_screen_requires_manual_adjudication",
        "metrics": {
            "discrepancies_identified": _ratio(matched_disc, total_disc),
            "omissions": _ratio(omissions, total_disc),
            "email_traps_corrected": _ratio(matched_traps, total_traps),
            "unsupported_or_false_claims": _ratio(final_score["unsupported_or_false_claims"], final_score["claim_slots_checked"]),
            "authority_inversions": _ratio(final_score["authority_inversions"], 1),
            "error_breakdown": {
                "subject_errors": final_score["subject_errors"],
                "value_errors": final_score["value_errors"],
                "unit_errors": final_score["unit_errors"],
                "condition_or_assumption_errors": final_score["condition_or_assumption_errors"],
                "priority_waterfall_errors": final_score["priority_waterfall_errors"],
            },
            "errors_fixed_in_repair": errors_fixed_in_repair,
            "errors_introduced_by_repair": errors_introduced_by_repair,
            "initial_total_defects": init_total_errors,
            "final_total_defects": final_total_errors,
            "false_accepts": _ratio(false_accepts, int(gate_assessed)),
            "false_rejects": _ratio(false_rejects, int(gate_assessed)),
            "abstentions_or_escalations": _ratio(abstentions_or_escalations, int(gate_assessed)),
        },
        "cost_and_latency": {
            "llm_calls": int(usage.get("llm_calls", 0)),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))),
            "wall_clock_ms": float(usage.get("wall_clock_ms", 0.0)),
            "verifier_ms": float(run_result.get("verifier_ms", 0.0)),
            "api_cost_usd": usage.get("api_cost_usd"),
            "estimated_cost_usd": usage.get("estimated_cost_usd"),
            "local_compute_cost_usd": usage.get("local_compute_cost_usd"),
            "cache_hits": int(cache_metrics.get("hits", 0)),
            "cache_misses": int(cache_metrics.get("misses", 0)),
        },
        "discrepancy_detail": final_score["discrepancy_hits"],
        "trap_detail": final_score["trap_hits"],
    }
