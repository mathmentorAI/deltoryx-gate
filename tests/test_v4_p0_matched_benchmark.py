"""Automated regression suite for P0 (#1 & #3) matched-denominator evaluation & utility SLA."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_DIR = _REPO_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from matched_p0_eval import run_all  # noqa: E402


def test_p0_1_and_p0_3_matched_denominators_and_utility_sla():
    report = run_all(_REPO_ROOT)

    # P0 #1: v1 vs v2 measured over the EXACT same 4 corpora (59 sentence checks) and exact common clean subset (9 checks)
    v1_all = report["p0_1_v1_vs_v2_matched"]["v1"]["matched_all_4_corpora"]
    v2_all = report["p0_1_v1_vs_v2_matched"]["v2"]["matched_all_4_corpora"]
    assert v1_all["corpora_ids"] == v2_all["corpora_ids"] == ["cart_synth", "meridian", "apex", "ridgeline"]
    assert v1_all["sentence_checks"] == v2_all["sentence_checks"] == 59
    assert v1_all["divergent_sections"] == v2_all["divergent_sections"] == 13
    assert v1_all["recall"]["n"] == v2_all["recall"]["n"] == 10

    v1_clean = report["p0_1_v1_vs_v2_matched"]["v1"]["matched_common_clean_2_corpora"]
    v2_clean = report["p0_1_v1_vs_v2_matched"]["v2"]["matched_common_clean_2_corpora"]
    assert v1_clean["corpora_ids"] == v2_clean["corpora_ids"] == ["apex", "ridgeline"]
    assert v1_clean["sentence_checks"] == v2_clean["sentence_checks"] == 9
    assert v1_clean["sentence_any_gap_rate"]["k"] == 0
    assert v2_clean["sentence_any_gap_rate"]["k"] == 0
    assert v1_clean["corpus_escalation_rate"]["k"] == 0
    assert v2_clean["corpus_escalation_rate"]["k"] == 0

    # P0 #3: Utility (PASS / supported / ACCEPT) must co-hold alongside rejection / recall
    p2 = report["p2_fidelity_and_noise_matched"]
    assert p2["utility_admitted_and_supported_positive"]["k"] == p2["utility_admitted_and_supported_positive"]["n"] == 5
    assert p2["adversarial_rejected_fidelity_and_condition"]["k"] == p2["adversarial_rejected_fidelity_and_condition"]["n"] == 4
    assert p2["ocr_clean_no_false_gap"]["k"] == 3
    assert p2["ocr_genuine_change_detected"]["k"] == 1

    p3 = report["p3_recall_and_scope_matched"]
    assert p3["v3_scope_negation_true_reject"]["k"] == p3["v3_scope_negation_true_reject"]["n"] == 8
    assert p3["v3_positive_utility_PASS"]["k"] == p3["v3_positive_utility_PASS"]["n"] == 6
    assert p3["p3_deleted_clause_true_detect"]["k"] == 2
    assert p3["p3_overlap_masking_true_detect"]["k"] == 2

    h = report["p0_3_heldout_multidomain"]
    assert h["v3_v4_false_accept_seeded"]["k"] == 0
    assert h["v3_v4_false_reject_clean"]["k"] == 0
    assert h["v3_v4_sla_passed"] is True
