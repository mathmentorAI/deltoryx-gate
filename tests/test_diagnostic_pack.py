"""Offline unit and safety tests for the Phase 3 n=120 diagnostic annotation pack."""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PHASE3_DIR = os.path.join(_REPO_ROOT, "case_studies", "pr_mci_phase3")
if _PHASE3_DIR not in sys.path:
    sys.path.insert(0, _PHASE3_DIR)

import diagnostic_pack as dp  # noqa: E402
import pipeline_phase3 as p3  # noqa: E402


class _FakeModel:
    """Deterministic fake cached model keyed cleanly by PR prompt payload."""

    def __init__(self, responses_by_pr: dict[str, dict], literal_by_pr: dict[str, list[dict]]) -> None:
        self.responses_by_pr = responses_by_pr
        self.literal_by_pr = literal_by_pr
        self.identity = {"provider": "fake-test"}

    def complete(self, system: str, payload: dict) -> dict:
        if system == p3.PASS2_BEHAVIOR_PROMPT:
            pr_id = str(payload["pr_id"])
            return {"output": {"behavioral_claims": self.responses_by_pr.get(pr_id, [])}}
        if system == p3.PASS3_REREAD_PROMPT:
            return {"output": {"reread_results": []}}
        # Pass-1 reader contract returns `proposals`
        return {"output": {"proposals": []}}


def _synthetic_prs_and_models():
    prs = []
    beh_by_pr = {}
    for i in range(6):
        pr_id = f"p{100 + i}"
        sym = f"compute_metric_{i}"
        xss_guard = "</script><img src=x onerror=alert(1)>" if i == 0 else "safe"
        prs.append(
            {
                "pr_id": pr_id,
                "agent": "Claude_Code" if i % 2 == 0 else "Devin",
                "repo": f"acme/service-{i}",
                "number": 10 + i,
                "url": f"https://github.com/acme/service-{i}/pull/{10 + i}",
                "title": f"Add {sym} and helper {xss_guard}",
                "body": f"Implements {sym} with details {xss_guard}.",
                "adds": 5,
                "dels": 0,
                "file_patches": [
                    {
                        "filename": f"src/mod_{i}.py",
                        "status": "modified",
                        "patch": (
                            f"@@ -1,2 +1,5 @@\n"
                            f" # {xss_guard}\n"
                            f"+def {sym}(val: int) -> int:\n"
                            f"+    return val + 1\n"
                            f"+def helper_{i}() -> None:\n"
                            f"+    pass\n"
                        ),
                    }
                ],
            }
        )
        beh_by_pr[pr_id] = [
            # Stratum: derived_supported (exact definition line)
            {
                "claim_text": f"Implements `{sym}` to increment input values.",
                "operation": "implements",
                "target": sym,
                "proposed_premises": [
                    {
                        "hunk_id": f"diff_{pr_id}_0000",
                        "file": f"src/mod_{i}.py",
                        "symbol_or_key": sym,
                        "literal_quote": f"+def {sym}(val: int) -> int:",
                        "basis": "explicit",
                    }
                ],
            },
            # Stratum: hypothesis_with_verified_premises (return line without def)
            {
                "claim_text": f"Computes incremented `val` via `helper_{i}`.",
                "operation": "implements",
                "target": f"helper_{i}",
                "proposed_premises": [
                    {
                        "hunk_id": f"diff_{pr_id}_0000",
                        "file": f"src/mod_{i}.py",
                        "symbol_or_key": "val",
                        "literal_quote": f"+def helper_{i}() -> None:",
                        "basis": "explicit",
                    }
                ],
            },
            # Stratum: no_premises_in_diff (empty premises, plus excluded clone candidate)
            {
                "claim_text": f"Adds telemetry exporter for `metric_{i}`.",
                "operation": "implements",
                "target": f"telemetry_exporter_{i}",
                "proposed_premises": [],
            },
            {
                "claim_text": f"Adds dashboard widget `{i}` with custom charts.",
                "operation": "implements",
                "target": f"dashboard_widget_{i}",
                "proposed_premises": [],
            },
        ]

    # Wait: helper_i above matches _DEF_RE (+def helper_i), so use return line for true hypothesis!
    for i in range(6):
        pr_id = f"p{100 + i}"
        beh_by_pr[pr_id][1] = {
            "claim_text": f"Updates `helper_{i}` implementation behavior.",
            "operation": "other_behavior",
            "target": f"helper_{i}",
            "proposed_premises": [
                {
                    "hunk_id": f"diff_{pr_id}_0000",
                    "file": f"src/mod_{i}.py",
                    "symbol_or_key": f"helper_{i}",
                    "literal_quote": f"+def helper_{i}() -> None:",
                    "basis": "explicit",
                }
            ],
        }

    m = _FakeModel(beh_by_pr, {})
    return prs, m, m


def test_stratified_sampling_determinism_exclusion_blinding_and_xss(tmp_path) -> None:
    prs, m1, m2 = _synthetic_prs_and_models()
    excluded = {"p100::beh::02", "p101::beh::03"}
    quotas = {
        "no_premises_in_diff": 4,
        "hypothesis_with_verified_premises": 3,
        "derived_supported": 2,
    }

    ordered1, pools1, diffs1 = dp.sample_diagnostic_claims(
        prs,
        pass1_model=m1,
        pass2_model=m2,
        excluded_claim_ids=excluded,
        strata_quotas=quotas,
        seed=20261121,
    )
    ordered2, pools2, _ = dp.sample_diagnostic_claims(
        prs,
        pass1_model=m1,
        pass2_model=m2,
        excluded_claim_ids=excluded,
        strata_quotas=quotas,
        seed=20261121,
    )
    assert [x["claim_id"] for x in ordered1] == [x["claim_id"] for x in ordered2]
    assert pools1 == {
        "no_premises_in_diff": 10,
        "hypothesis_with_verified_premises": 6,
        "derived_supported": 6,
    }
    assert pools2 == pools1
    assert len(ordered1) == 9
    assert all(x["claim_id"] not in excluded for x in ordered1)

    pack_dir = tmp_path / "ext_pack"
    manifest_path = tmp_path / "git_repo" / "diagnostic_120_manifest.json"
    manifest = dp.write_diagnostic_pack(
        ordered1,
        diffs1,
        out_dir=str(pack_dir),
        manifest_out=str(manifest_path),
        pool_sizes=pools1,
        strata_quotas=quotas,
        excluded_ids=sorted(excluded),
        prereg_meta={"baseline_commit": "3808d82", "preregistration_version": "phase3-v2.0"},
    )
    assert manifest["total_items"] == 9

    # Sanitized manifest must NOT contain third-party PR titles, bodies, diffs, claim texts, or strata
    man_txt = manifest_path.read_text(encoding="utf-8")
    assert "claim_text" not in man_txt
    assert "pr_title" not in man_txt
    assert "stratum" not in man_txt
    assert "compute_metric_" not in man_txt
    assert "no_premises_in_diff" in man_txt  # aggregate quota key only, zero per-item strata

    # Blinded templates must NOT leak model status, operation, target, stratum, or agent
    tmpl_lines = (pack_dir / "annotations_template.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(tmpl_lines) == 9
    for raw_ln in tmpl_lines:
        rec = json.loads(raw_ln)
        assert rec["annotation_status"] == ""
        assert rec["diff_contradicts_claim"] is None
        assert rec["gold_operation"] == ""
        assert rec["gold_target"] == ""
        assert rec["gold_premises"] == []
        assert rec["contradicting_premises"] == []
        for forbidden in ("stratum", "agent", "baseline_d_full_status", "baseline_operation", "proposed_premises"):
            assert forbidden not in rec

    with open(pack_dir / "annotations_template.csv", newline="", encoding="utf-8") as cfh:
        csv_rows = list(csv.DictReader(cfh))
    assert len(csv_rows) == 9
    assert all(r["annotation_status"] == "" and r["gold_operation"] == "" for r in csv_rows)

    # XSS / script-breakout safety in generated JS & HTML viewer
    viewer_html = (pack_dir / "diff_viewer.html").read_text(encoding="utf-8")
    assert ".innerHTML" not in viewer_html
    assert "outerHTML" not in viewer_html
    assert "insertAdjacentHTML" not in viewer_html
    assert (pack_dir / "viewer_index.js").read_text(encoding="utf-8").count("</script") == 0
    for pr_js in (pack_dir / "pr_diffs").glob("pr_*.js"):
        txt = pr_js.read_text(encoding="utf-8")
        assert "</script" not in txt.lower()


def test_validate_annotations_positive_and_negative_plus_cli_viewer(tmp_path) -> None:
    prs, m1, m2 = _synthetic_prs_and_models()
    quotas = {
        "no_premises_in_diff": 2,
        "hypothesis_with_verified_premises": 2,
        "derived_supported": 2,
    }
    ordered, pools, diffs = dp.sample_diagnostic_claims(
        prs[:3],
        pass1_model=m1,
        pass2_model=m2,
        strata_quotas=quotas,
        seed=99,
    )
    pack_dir = tmp_path / "pack"
    dp.write_diagnostic_pack(
        ordered,
        diffs,
        out_dir=str(pack_dir),
        manifest_out=None,
        pool_sizes=pools,
        strata_quotas=quotas,
        excluded_ids=(),
        prereg_meta={"baseline_commit": "3808d82", "preregistration_version": "phase3-v2.0"},
    )

    # CLI view renders numbered diff lines and supports search filtering
    cli_full = dp.render_cli_item_diff(str(pack_dir), "DIAG-001")
    assert "DIAG-001" in cli_full
    assert re.search(r"L0001\s+\|", cli_full)
    cli_search = dp.render_cli_item_diff(str(pack_dir), "DIAG-001", search="return val + 1")
    assert "return val + 1" in cli_search

    # Build valid completed annotations (supported + no_premises + contradicted)
    completed = []
    for idx, rec in enumerate(ordered):
        pr_d = diffs[rec["pr_id"]]
        first_ch = pr_d["chunks"][0]
        add_ln = next(ln for ln in first_ch["lines"] if ln["kind"] == "add")
        prem = {
            "file": first_ch["file"],
            "chunk_id": first_ch["chunk_id"],
            "diff_line_no": add_ln["line_no"],
            "literal_quote": add_ln["text"],
        }
        base = {
            "item_id": rec["item_id"],
            "claim_id": rec["claim_id"],
            "pr_id": rec["pr_id"],
            "confidence": "high",
            "notes": "",
        }
        if idx % 3 == 0:
            completed.append(
                {
                    **base,
                    "annotation_status": "supported",
                    "diff_contradicts_claim": False,
                    "gold_operation": "implements",
                    "gold_target": "sym",
                    "gold_premises": [prem],
                    "contradicting_premises": [],
                }
            )
        elif idx % 3 == 1:
            completed.append(
                {
                    **base,
                    "annotation_status": "no_premises",
                    "diff_contradicts_claim": False,
                    "gold_operation": "implements",
                    "gold_target": "missing_sym",
                    "gold_premises": [],
                    "contradicting_premises": [],
                }
            )
        else:
            completed.append(
                {
                    **base,
                    "annotation_status": "contradicted",
                    "diff_contradicts_claim": True,
                    "gold_operation": "removes_support_for",
                    "gold_target": "sym",
                    "gold_premises": [],
                    "contradicting_premises": [prem],
                }
            )

    jsonl_ok = tmp_path / "ok.jsonl"
    jsonl_ok.write_text("\n".join(json.dumps(x) for x in completed) + "\n", encoding="utf-8")
    ok, summ = dp.validate_annotations(str(jsonl_ok), str(pack_dir))
    assert ok is True, summ
    assert summ["status_counts"] == {"contradicted": 2, "no_premises": 2, "supported": 2}

    # CSV round-trip passes too
    csv_ok = tmp_path / "ok.csv"
    with open(csv_ok, "w", newline="", encoding="utf-8") as cfh:
        w = csv.DictWriter(
            cfh,
            fieldnames=[
                "item_id", "claim_id", "pr_id", "annotation_status", "diff_contradicts_claim",
                "gold_operation", "gold_target", "gold_premises_json", "contradicting_premises_json",
                "confidence", "notes",
            ],
        )
        w.writeheader()
        for c in completed:
            w.writerow(
                {
                    "item_id": c["item_id"],
                    "claim_id": c["claim_id"],
                    "pr_id": c["pr_id"],
                    "annotation_status": c["annotation_status"],
                    "diff_contradicts_claim": str(c["diff_contradicts_claim"]).lower(),
                    "gold_operation": c["gold_operation"],
                    "gold_target": c["gold_target"],
                    "gold_premises_json": json.dumps(c["gold_premises"]),
                    "contradicting_premises_json": json.dumps(c["contradicting_premises"]),
                    "confidence": c["confidence"],
                    "notes": c["notes"],
                }
            )
    ok_csv, summ_csv = dp.validate_annotations(str(csv_ok), str(pack_dir))
    assert ok_csv is True, summ_csv

    # Negative 1: supported without premises
    bad1 = [dict(x) for x in completed]
    bad1[0] = {**bad1[0], "gold_premises": []}
    p_bad1 = tmp_path / "bad1.jsonl"
    p_bad1.write_text("\n".join(json.dumps(x) for x in bad1) + "\n", encoding="utf-8")
    ok_b1, s_b1 = dp.validate_annotations(str(p_bad1), str(pack_dir))
    assert ok_b1 is False
    assert any("requires >= 1 gold_premise" in e for e in s_b1["errors"])

    # Negative 2: fabricated literal_quote not present in diff
    bad2 = [dict(x) for x in completed]
    bad2[0] = {
        **bad2[0],
        "gold_premises": [{**bad2[0]["gold_premises"][0], "literal_quote": "+def fabricated_symbol():"}],
    }
    p_bad2 = tmp_path / "bad2.jsonl"
    p_bad2.write_text("\n".join(json.dumps(x) for x in bad2) + "\n", encoding="utf-8")
    ok_b2, s_b2 = dp.validate_annotations(str(p_bad2), str(pack_dir))
    assert ok_b2 is False
    assert any("not found verbatim" in e for e in s_b2["errors"])

    # Negative 3: contradicted without diff_contradicts_claim=True + missing item + no_premises with premise
    bad3 = [dict(x) for x in completed[:-1]]
    bad3[1] = {**bad3[1], "gold_premises": bad3[0]["gold_premises"]}
    bad3[2] = {**bad3[2], "diff_contradicts_claim": False, "gold_operation": "invalid_op"}
    p_bad3 = tmp_path / "bad3.jsonl"
    p_bad3.write_text("\n".join(json.dumps(x) for x in bad3) + "\n", encoding="utf-8")
    ok_b3, s_b3 = dp.validate_annotations(str(p_bad3), str(pack_dir))
    assert ok_b3 is False
    assert s_b3["missing_count"] == 1
    assert any("must have empty gold_premises" in e for e in s_b3["errors"])
    assert any("requires diff_contradicts_claim=True" in e for e in s_b3["errors"])
    assert any("invalid gold_operation" in e for e in s_b3["errors"])
