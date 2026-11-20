"""Offline unit tests for diagnostic Points 2-5 attribution module."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PHASE3_DIR = os.path.join(_REPO_ROOT, "case_studies", "pr_mci_phase3")
if _PHASE3_DIR not in sys.path:
    sys.path.insert(0, _PHASE3_DIR)

import diagnostic_attribution as da  # noqa: E402
from claimlayer.v4.domains.code_change_inference import RelevanceConfig, build_typed_diff_ledger  # noqa: E402


def _make_sample_data():
    long_prefix = "\n".join(f" # boilerplate line {i:03d}" for i in range(120))
    patch_py = (
        f"@@ -1,122 +1,125 @@ def unrelated_top_func():\n"
        f"{long_prefix}\n"
        f"+def compute_retry_backoff(attempt: int) -> float:\n"
        f"+    return float(2 ** attempt)\n"
        f"@@ -300,3 +303,4 @@ def helper_tail():\n"
        f"     pass\n"
        f"+# updated docs note for RetryPolicy\n"
    )
    pr = {
        "pr_id": "9001",
        "repo": "acme/gateway",
        "title": "Add compute_retry_backoff and tests",
        "body": "Implements compute_retry_backoff and removes legacy_backoff.",
        "files": ["src/retry.py", "tests/test_retry.py"],
        "file_patches": [
            {"filename": "src/retry.py", "status": "modified", "patch": patch_py},
            {
                "filename": "tests/test_retry.py",
                "status": "added",
                "patch": "@@ -0,0 +1,3 @@\n+def test_compute_retry_backoff():\n+    assert compute_retry_backoff(2) == 4.0\n",
            },
        ],
    }
    ledger = build_typed_diff_ledger(pr)
    ledgers = {"9001": ledger}
    prs_by_id = {"9001": pr}

    anns = [
        {
            "item_id": "DIAG-001",
            "claim_id": "9001::beh::00",
            "pr_id": "9001",
            "claim_text": "Implements compute_retry_backoff for exponential retries.",
            "annotation_status": "supported",
            "diff_contradicts_claim": False,
            "gold_operation": "implements",
            "gold_target": "compute_retry_backoff",
            "gold_premises": [
                {
                    "file": "src/retry.py",
                    "chunk_id": "diff_9001_0000",
                    "diff_line_no": 122,
                    "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
                }
            ],
            "contradicting_premises": [],
            "confidence": "high",
        },
        {
            "item_id": "DIAG-002",
            "claim_id": "9001::beh::01",
            "pr_id": "9001",
            "claim_text": "Adds unit tests for compute_retry_backoff.",
            "annotation_status": "supported",
            "diff_contradicts_claim": False,
            "gold_operation": "adds_test_for",
            "gold_target": "compute_retry_backoff",
            "gold_premises": [
                {
                    "file": "tests/test_retry.py",
                    "chunk_id": "diff_9001_0001",
                    "diff_line_no": 3,
                    "literal_quote": "+    assert compute_retry_backoff(2) == 4.0",
                }
            ],
            "contradicting_premises": [],
            "confidence": "high",
        },
        {
            "item_id": "DIAG-003",
            "claim_id": "9001::beh::02",
            "pr_id": "9001",
            "claim_text": "Adds Slack webhook notifications.",
            "annotation_status": "no_premises",
            "diff_contradicts_claim": False,
            "gold_operation": "implements",
            "gold_target": "slack_webhook",
            "gold_premises": [],
            "contradicting_premises": [],
            "confidence": "high",
        },
        {
            "item_id": "DIAG-004",
            "claim_id": "9001::beh::03",
            "pr_id": "9001",
            "claim_text": "All local unit tests passed before merge.",
            "annotation_status": "out_of_scope_meta",
            "diff_contradicts_claim": False,
            "gold_operation": "out_of_scope_meta",
            "gold_target": "",
            "gold_premises": [],
            "contradicting_premises": [],
            "confidence": "high",
        },
    ]
    return pr, ledger, prs_by_id, ledgers, anns


def test_oracle_engine_ceiling_and_monotonicity() -> None:
    _, ledger, _, ledgers, anns = _make_sample_data()
    p2_rep, eng_out = da.run_oracle_engine_experiment(anns, ledgers)

    assert p2_rep["accuracy_120"]["k"] == 4
    assert p2_rep["accuracy_120"]["n"] == 4
    assert p2_rep["accuracy_120"]["rate"] == 1.0
    assert p2_rep["monotonicity_check_unruled_derived_count"] == 0
    assert eng_out["DIAG-001"]["status"] == "derived_supported"
    assert eng_out["DIAG-002"]["status"] == "derived_supported"
    assert eng_out["DIAG-003"]["status"] == "no_premises_in_diff"
    assert eng_out["DIAG-004"]["status"] == "unverifiable"

    # Double-check single item monotonicity directly
    unruled = da.run_oracle_engine_item(
        claim_id="m",
        claim_text="Implements compute_retry_backoff",
        gold_operation="implements",
        gold_target="compute_retry_backoff",
        gold_premises=anns[0]["gold_premises"],
        ledger=ledger,
        use_declared_rules=False,
    )
    assert unruled["status"] == "hypothesis_with_verified_premises"


def test_retrieve_claim_hunks_windows_deep_line_past_1600_chars() -> None:
    pr, ledger, _, _, anns = _make_sample_data()
    # Baseline 1600-char prefix cut loses `+def compute_retry_backoff` buried after 120 boilerplate lines
    base_vis = da._baseline_visible_hunks(pr, ledger, frozenset({"acme", "gateway"}))
    assert "+def compute_retry_backoff(attempt: int) -> float:" not in "\n".join(h["patch"] for h in base_vis)

    # Per-claim retriever centers its window around matching symbol inside large @@ hunk
    hunks = da.retrieve_claim_hunks(
        ledger,
        anns[0]["claim_text"],
        repo_tokens=frozenset({"acme", "gateway"}),
        max_hunks=4,
        max_total_chars=6000,
        max_hunk_chars=1200,
    )
    joined = "\n".join(h["patch"] for h in hunks)
    assert "+def compute_retry_backoff(attempt: int) -> float:" in joined
    assert hunks[0]["file"] == "src/retry.py"


def test_verifier_rigidity_reader_eval_and_mutually_exclusive_attribution() -> None:
    _, _, prs_by_id, ledgers, anns = _make_sample_data()
    rel_cfg = RelevanceConfig()

    _, eng_out = da.run_oracle_engine_experiment(anns, ledgers)
    p3_rep, ver_out = da.run_verifier_rigidity_audit(anns, prs_by_id, ledgers, relevance_config=rel_cfg)
    assert p3_rep["premise_level"]["passed"]["k"] == 2
    assert p3_rep["claim_level"]["accuracy_120"]["k"] == 4

    # Construct baseline proposals where DIAG-001 failed in reader (empty) and DIAG-004 misclassified op
    props = {
        "DIAG-001": {"operation": "implements", "target": "compute_retry_backoff", "proposed_premises": [], "reread_attempted": True},
        "DIAG-002": {
            "operation": "adds_test_for",
            "target": "compute_retry_backoff",
            "proposed_premises": [{"file": "tests/test_retry.py", "literal_quote": "+def test_compute_retry_backoff():", "basis": "explicit"}],
            "reread_attempted": False,
        },
        "DIAG-003": {"operation": "implements", "target": "slack_webhook", "proposed_premises": [], "reread_attempted": True},
        "DIAG-004": {
            "operation": "implements",
            "target": "tests",
            "proposed_premises": [{"file": "tests/test_retry.py", "literal_quote": "+def fabricated_nonexistent():", "basis": "explicit"}],
            "reread_attempted": False,
        },
    }
    r_rep, r_out = da.evaluate_reader_proposals(
        label="test_reader",
        anns=anns,
        prs_by_id=prs_by_id,
        ledgers=ledgers,
        proposals_by_item=props,
        relevance_config=rel_cfg,
    )
    assert r_rep["ungrounded_or_fabricated_premises"]["k"] == 1
    assert r_rep["accuracy_120"]["k"] == 2  # DIAG-002, DIAG-003

    attr = da.compute_error_attribution(
        anns=anns,
        engine_outcomes=eng_out,
        verifier_before_outcomes=ver_out,
        verifier_after_outcomes=ver_out,
        baseline_before_outcomes=r_out,
        baseline_after_outcomes=r_out,
    )
    assert attr["baseline_e2e_errors"] == 2
    part = attr["mutually_exclusive_partition"]
    assert sum(part[k]["count"] for k in ("reader_only", "verifier_only", "reader_and_verifier_joint", "engine_only")) == 2
    assert part["reader_only"]["count"] == 2
