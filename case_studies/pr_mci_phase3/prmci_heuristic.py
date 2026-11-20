"""Exact reproduction of the PR-MCI heuristic baseline (Gong et al., MSR '26).

Implements the exact weights `(0.3, 0.4, 0.3)` for `(scope_adequacy, file_type_consistency,
task_type_alignment)` and threshold `0.61` from `PR-MCI/code/02_compute_mci.py`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Mapping, Sequence

csv.field_size_limit(10**9)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from claimlayer.v4.domains.code_change_inference import classify_file_category  # noqa: E402

MCI_THRESHOLD_HEURISTIC = 0.61
HEURISTIC_WEIGHTS = (0.3, 0.4, 0.3)

_TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "feat": ("add", "adds", "added", "new", "feature", "implement", "create", "support", "introduce"),
    "fix": ("fix", "fixes", "fixed", "bug", "error", "issue", "resolve", "resolved", "patch", "correct"),
    "docs": ("doc", "docs", "readme", "comment", "documentation", "changelog"),
    "test": ("test", "tests", "spec", "coverage", "unit", "integration", "e2e"),
    "refactor": ("refactor", "cleanup", "clean", "reorganize", "improve", "optimize", "simplify"),
    "chore": ("chore", "update", "bump", "version", "deps", "dependency", "dependencies", "ci", "cd"),
}

_CONV_PREFIX_RE = re.compile(r"^\s*(feat|fix|docs|test|refactor|chore)(?:\([^)]+\))?!?:", re.I)


def infer_task_type(title: str, body: str, files: Sequence[str]) -> str:
    """Deterministically infer PR task_type when upstream metadata does not provide one."""
    t_str = (title or "").strip()
    if m := _CONV_PREFIX_RE.match(t_str):
        return m.group(1).lower()
    msg_l = f"{t_str}\n{body or ''}".lower()
    cats = [classify_file_category(f) for f in files]
    if cats and all(c == "docs" for c in cats):
        return "docs"
    if cats and all(c == "test" for c in cats):
        return "test"
    best_type = "feat"
    best_hits = 0
    for ttype in ("fix", "feat", "refactor", "docs", "test", "chore"):
        hits = sum(1 for kw in _TASK_KEYWORDS[ttype] if kw in msg_l)
        if hits > best_hits:
            best_hits = hits
            best_type = ttype
    return best_type


def scope_adequacy_score(msg_lower: str, churn: int, task_type: str | None = None) -> float:
    msg_len = len(msg_lower.split())
    if churn <= 0:
        return 0.5
    expected_words = int(min(50, max(10, churn / 20)))
    ratio = min(1.0, msg_len / max(1, expected_words))
    if msg_len < 10 and churn > 100:
        if task_type in ("refactor", "chore", "docs"):
            return min(0.6, ratio)
        return 0.2
    return float(ratio)


def task_type_alignment_score(msg_lower: str, task_type: str) -> float:
    kws = _TASK_KEYWORDS.get(task_type, ())
    if not kws:
        return 0.5
    if any(kw in msg_lower for kw in kws):
        return 0.9
    return 0.6


def file_type_consistency_score(
    msg_lower: str,
    *,
    files_test_count: int,
    files_docs_count: int,
    files_config_count: int,
) -> float:
    mentions_tests = any(k in msg_lower for k in ("test", "tests", "unit", "integration", "coverage", "spec", "e2e"))
    mentions_docs = any(k in msg_lower for k in ("doc", "docs", "readme", "documentation", "changelog"))
    mentions_config = any(
        k in msg_lower
        for k in ("config", "ci", "cd", "workflow", "yaml", "yml", "json", "toml", "deps", "dependency", "dependencies")
    )

    def _score_one(has_type: bool, mentions_type: bool) -> float:
        if has_type and mentions_type:
            return 1.0
        if has_type and not mentions_type:
            return 0.6
        if not has_type and mentions_type:
            return 0.2
        return 0.8

    s1 = _score_one(files_test_count > 0, mentions_tests)
    s2 = _score_one(files_docs_count > 0, mentions_docs)
    s3 = _score_one(files_config_count > 0, mentions_config)
    return (s1 + s2 + s3) / 3.0


def score_pr_heuristic(
    pr: Mapping[str, Any],
    *,
    threshold: float = MCI_THRESHOLD_HEURISTIC,
) -> dict[str, Any]:
    """Compute exact PR-MCI heuristic similarity and binary prediction for one PR."""
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    pr_message = str(pr.get("pr_message") or f"{title}\n\n{body}").strip()
    msg_lower = pr_message.lower()

    files = list(pr.get("files") or ())
    if "files_test_count" in pr:
        n_test = int(pr["files_test_count"])
        n_docs = int(pr.get("files_docs_count", 0))
        n_conf = int(pr["files_config_count"]) if "files_config_count" in pr else 0
    else:
        n_test = sum(1 for f in files if classify_file_category(f) == "test")
        n_docs = sum(1 for f in files if classify_file_category(f) == "docs")
        n_conf = sum(1 for f in files if classify_file_category(f) == "config")

    churn = int(pr.get("total_churn") if pr.get("total_churn") is not None else ((pr.get("adds") or 0) + (pr.get("dels") or 0)))
    task_type = str(pr.get("task_type") or "") or infer_task_type(title, body, files)

    sc_scope = scope_adequacy_score(msg_lower, churn, task_type=task_type)
    sc_ft = file_type_consistency_score(
        msg_lower,
        files_test_count=n_test,
        files_docs_count=n_docs,
        files_config_count=n_conf,
    )
    sc_task = task_type_alignment_score(msg_lower, task_type)
    w_s, w_f, w_t = HEURISTIC_WEIGHTS
    sim = w_s * sc_scope + w_f * sc_ft + w_t * sc_task
    return {
        "pr_id": str(pr.get("pr_id") or ""),
        "task_type": task_type,
        "scope_score": round(sc_scope, 6),
        "filetype_score": round(sc_ft, 6),
        "task_score": round(sc_task, 6),
        "similarity_heuristic": round(sim, 6),
        "predicted_heuristic": "misaligned" if sim < threshold else "aligned",
    }


def verify_prmci_csv(csv_path: str) -> dict[str, Any]:
    """Compare our reproduction against `predicted_heuristic` in PR-MCI validation CSV."""
    with open(csv_path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    prmci_dir = os.path.dirname(os.path.dirname(os.path.abspath(csv_path)))
    prs_by_id: dict[str, dict[str, Any]] = {}
    if rows and "pr_message" not in rows[0]:
        v2_dir = os.path.abspath(os.path.join(_THIS_DIR, "..", "pr_mci_claude_code"))
        if v2_dir not in sys.path:
            sys.path.insert(0, v2_dir)
        import parse  # noqa: E402

        for ag in ("Claude_Code", "OpenAI_Codex", "Copilot", "Devin", "Cursor"):
            try:
                for p in parse.load(ag, prmci_dir=prmci_dir):
                    prs_by_id[str(p["pr_id"])] = p
            except Exception:
                continue

    matches = 0
    total = 0
    mismatches = []
    for r in rows:
        if "predicted_heuristic" not in r:
            continue
        pid = str(r.get("pr_id") or "")
        src_rec: Mapping[str, Any] = prs_by_id.get(pid, r)
        total += 1
        res = score_pr_heuristic(src_rec)
        if res["predicted_heuristic"] == r["predicted_heuristic"].lower():
            matches += 1
        else:
            mismatches.append({"pr_id": pid, "ours": res["predicted_heuristic"], "csv": r.get("predicted_heuristic")})
    return {
        "total": total,
        "matches": matches,
        "agreement": round(matches / max(1, total), 6),
        "mismatches_count": len(mismatches),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="PR-MCI heuristic reproduction baseline.")
    ap.add_argument("--prmci-csv", default="")
    ap.add_argument("--aidev-jsonl", default="")
    args = ap.parse_args()

    out: dict[str, Any] = {}
    if args.prmci_csv:
        out["prmci_csv_agreement"] = verify_prmci_csv(args.prmci_csv)
    if args.aidev_jsonl:
        with open(args.aidev_jsonl, encoding="utf-8") as fh:
            prs = [json.loads(ln) for ln in fh if ln.strip()]
        preds = [score_pr_heuristic(p) for p in prs]
        n_mis = sum(1 for p in preds if p["predicted_heuristic"] == "misaligned")
        out["aidev_summary"] = {"n": len(preds), "flagged_misaligned": n_mis, "rate": round(n_mis / max(1, len(preds)), 4)}
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
