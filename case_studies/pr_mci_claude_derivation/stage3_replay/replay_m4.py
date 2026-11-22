"""Stage 3, verifier-first measurement: replay identical inputs (no model calls) through a verifier.

Inputs (all recorded, never regenerated):
  * M4 PR-MCI plans (final raw_proposal per case) for the 20 regression and 10 transfer cases;
  * M4 temporal plans (14);
  * prepared valid temporal plans for T2/T4/T5: the M3 recorded plans that passed (same sources/claims);
  * stage-2 manual derivations (C07, C13; quote and line-range contracts).

Usage: python replay_m4.py --label new|old --out FILE   (PYTHONPATH selects the verifier code)
Verifier configuration is the public-API default: unit_profile auto, pass_min_grade B, no declared
count definitions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CS = HERE.parents[0]
CL = HERE.parents[2]

from claimlayer.v4 import verify_plan  # noqa: E402  (PYTHONPATH decides which code)


def _row(r) -> dict:
    return {"verdict": r.epistemic_verdict, "routing": r.operational_routing.status, "category": r.outcome_category,
            "grade": (r.reliability or {}).get("grade") if isinstance(r.reliability, dict) else None,
            "computed": (r.computed_conclusion or {}).get("value") if isinstance(r.computed_conclusion, dict) else None,
            "defects": sorted({str(d.get("code")) for d in r.defects}),
            "explanation": (r.explanation or "")[:600]}


def pr_sample(live: Path, inputs: Path) -> list[dict]:
    L = json.loads(live.read_text())
    I = {c["case_id"]: c for c in json.loads(inputs.read_text())["cases"]}
    out = []
    for row in L["cases"]:
        cid = row["case_id"]
        resp = row["response"]
        plan = resp["provenance"].get("raw_proposal")
        rec = {"case_id": cid, "recorded": {"verdict": row["verdict"], "routing": row["routing"], "category": row["outcome_category"],
                                            "grade": (resp.get("reliability") or {}).get("grade")}}
        if plan is None:
            rec["replay"] = {"verdict": None, "note": "sin plan registrado"}
        else:
            r = verify_plan(I[cid]["sources"], I[cid]["claim"], plan, unit_profile="auto")
            rec["replay"] = _row(r)
        out.append(rec)
    return out


def temporal(bench: Path, prepared_from: Path, ids: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    B = json.loads(bench.read_text())
    rows = B["core_cases"] + B["heldout_paraphrases"]
    out = []
    for row in rows:
        r = verify_plan(row["sources"], row["claim"], row["raw_proposal"], unit_profile="auto")
        out.append({"case_id": row["case_id"], "expected": row["expected"]["epistemic_verdict"],
                    "recorded": {"verdict": row["observed"]["epistemic_verdict"], "category": row["observed"].get("outcome_category")},
                    "replay": _row(r)})
    P = json.loads(prepared_from.read_text())
    prep = []
    for row in P["core_cases"] + P["heldout_paraphrases"]:
        if row["case_id"] in ids:
            r = verify_plan(row["sources"], row["claim"], row["raw_proposal"], unit_profile="auto")
            prep.append({"case_id": row["case_id"], "plan_source": f"{prepared_from.name} (M3, plan registrado que pasó)",
                         "expected": row["expected"]["epistemic_verdict"], "replay": _row(r)})
    return out, prep


def independent_reference(sample_dir: Path) -> list[dict]:
    """Structural counts of every file diff in the frozen sources vs GitHub's per-file additions /
    deletions (pulls/{n}/files, cached at selection time): a reference independent of the parser."""
    try:
        from claimlayer.v4.diff_evidence import count_diff_lines, parse_unified_diff
    except ImportError:  # older verifier without the full-patch structure
        return []
    out = []
    for case in json.loads((sample_dir / "frozen/inputs_manifest.json").read_text())["cases"]:
        fp = sample_dir / "github_cache" / f"{case['pr_id']}_files.json"
        gh = {f["filename"]: f for f in json.loads(fp.read_text())} if fp.exists() else {}
        for doc in case["sources"]:
            dm = parse_unified_diff(doc["text"])
            if dm is None:
                continue
            for path in dm.files():
                add = count_diff_lines(dm, path=path, side="+")
                rem = count_diff_lines(dm, path=path, side="-")
                ref = gh.get(path)
                out.append({"case_id": case["case_id"], "file": path,
                            "structural": [add["count_git"], rem["count_git"]],
                            "github": [ref["additions"], ref["deletions"]] if ref else None,
                            "truncated": add["truncated"] or rem["truncated"],
                            "unterminated_head": add["unterminated_lines"],
                            "match": bool(ref) and [add["count_git"], rem["count_git"]] == [ref["additions"], ref["deletions"]]})
    return out


def _nl(case: dict, did: str) -> int:
    return len(next(d["text"] for d in case["sources"] if d["document_id"] == did).split("\n"))


def manual_stage3() -> list[dict]:
    """Prepared plans written by hand with the stage-3 contract (typed diff-presence premises,
    structural counts with file scope). No declared count definitions (public-API default)."""
    I = {c["case_id"]: c for c in json.loads((CS / "frozen/inputs_manifest.json").read_text())["cases"]}
    out = []
    c07 = I["C07"]
    path07 = "frontend/packages/db/supabase/tests/database/03-organization_members_rls.test.sql"
    p07 = {"claim_target": {"subject": path07, "predicate": "test cases", "value": 8, "unit": "count", "polarity": 1, "revision": "added"},
           "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "file_01", "file": path07, "side": "added",
                         "line_start": 4, "line_end": _nl(c07, "file_01")}],
           "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                      "parameters": {"line_prefix": "+", "scope": "file", "contains": "SELECT ok("}}],
           "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": []}
    c14 = I["C14"]
    o, r = ".claude/commands/shared/patterns/mcp/orchestration.yml", ".claude/commands/shared/patterns/mcp/registry.yml"

    def p14(conv: str | None) -> dict:
        prm = {"line_prefix": "+", "scope": "file", **({"line_convention": conv} if conv else {})}
        return {"claim_target": {"subject": "mcp/", "predicate": "lines total", "value": 207, "unit": "count", "polarity": 1, "revision": "added"},
                "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "file_09", "file": o, "side": "added",
                              "line_start": 4, "line_end": _nl(c14, "file_09") - 1},
                             {"claim_id": "p2", "kind": "diff_presence", "document_id": "file_10", "file": r, "side": "added",
                              "line_start": 4, "line_end": _nl(c14, "file_10") - 1}],
                "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1", "parameters": prm},
                          {"step_id": "s2", "operation": "count_lines", "premises": ["p2"], "conclusion_id": "c2", "parameters": prm},
                          {"step_id": "s3", "operation": "add", "premises": ["c1", "c2"], "conclusion_id": "c3", "parameters": {}}],
                "derived_conclusion_id": "c3", "missing_premises": [], "assumptions": [], "suggested_rules": []}
    for cid, label, plan in (("C07", "presence + count_lines(contains, file)", p07),
                             ("C14", "presence x2 + count_lines(file) + add, sin convenio", p14(None)),
                             ("C14", "presence x2 + count_lines(file, convenio wc) + add", p14("wc"))):
        case = I[cid]
        res = verify_plan(case["sources"], case["claim"], plan, unit_profile="auto")
        out.append({"case_id": cid, "contract": "stage3", "plan": label, "replay": _row(res)})
    return out


def manual() -> list[dict]:
    sys.path.insert(0, str(CS / "stage2_manual"))
    import manual_derivations as md  # noqa: E402
    import manual_derivations_ranges as mr  # noqa: E402
    out = []
    for name, cid, fn in (("quote_contract", "C07", md.manual_c07),
                          ("quote_contract", "C13", md.manual_c13),
                          ("line_range_contract", "C07", mr.c07_ranges), ("line_range_contract", "C13", mr.c13_ranges)):
        if fn is None:
            continue
        case = md.INPUTS[cid]
        r = verify_plan(case["sources"], case["claim"], fn(), unit_profile="auto")
        out.append({"case_id": cid, "contract": name, "replay": _row(r)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t_rows, prep = temporal(CL / "benchmarks/v4_derivation_planner_live_results_stage2.json",
                            CL / "benchmarks/v4_derivation_planner_live_results_stage1.json",
                            ("T2_simple_refuted_thursday", "T4_birthday_anchor_derived_sunday", "T5_birthday_anchor_refuted_wednesday"))
    res = {
        "label": a.label,
        "verifier_code": sys.modules["claimlayer.v4"].__file__,
        "regression_pr": pr_sample(CS / "results/live_results_stage2.json", CS / "frozen/inputs_manifest.json"),
        "transfer_pr": pr_sample(CS.parent / "pr_mci_transfer_sample/results/live_results_stage2.json",
                                 CS.parent / "pr_mci_transfer_sample/frozen/inputs_manifest.json"),
        "temporal_m4_plans": t_rows,
        "temporal_prepared_plans": prep,
        "manual_derivations": manual(),
        "manual_stage3": manual_stage3() if "count_definitions" in __import__("inspect").signature(verify_plan).parameters else [],
        "independent_reference": {"regression": independent_reference(CS), "transfer": independent_reference(CS.parent / "pr_mci_transfer_sample")},
    }
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
