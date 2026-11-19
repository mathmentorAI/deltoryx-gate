"""Deterministic evaluation runner for rule-based PR-MCI checkers."""
from __future__ import annotations

import collections
import csv
import json
import math
import os
import sys

csv.field_size_limit(10**9)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_claims  # noqa: E402
import parse  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def metrics(rows, pred, pos) -> dict:
    tp = sum(1 for r in rows if pos(r) and pred(r))
    fp = sum(1 for r in rows if not pos(r) and pred(r))
    fn = sum(1 for r in rows if pos(r) and not pred(r))
    tn = len(rows) - tp - fp - fn
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "n": len(rows),
        "pos": tp + fn,
        "neg": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(P, 4),
        "precision_ci95": [round(x, 4) for x in wilson(tp, tp + fp)],
        "recall": round(R, 4),
        "recall_ci95": [round(x, 4) for x in wilson(tp, tp + fn)],
        "f1": round(F, 4),
        "fp_rate": round(fpr, 4),
        "fp_rate_ci95": [round(x, 4) for x in wilson(fp, fp + tn)],
    }


def _sanitize_record(rec: dict) -> dict:
    return {
        "pr_id": rec["pr_id"],
        "url": rec["url"],
        "verdict": rec["verdict"],
        "signals": {k: (True if k == "U_understated_files" else (len(v) if isinstance(v, list) else v)) for k, v in rec["signals"].items()},
        "claims": [{"kind": c["kind"], "status": c["status"], "has_evidence": bool(c["evidence"])} for c in rec["claims"]],
        "n_claims": rec["n_claims"],
        "n_supported": rec["n_supported"],
        "files_changed": rec["files_changed"],
        "lines_changed": rec["lines_changed"],
        "version": rec.get("version"),
        "evidence_profile": rec.get("evidence_profile", "legal_prose"),
        "receipt_sha256": rec["receipt_sha256"],
    }


def main(agent: str = "Claude_Code", out: str = "results_claude_code.json") -> None:
    prmci_dir = os.environ.get("PRMCI_DIR", parse.SRC)
    prs = {p["pr_id"]: p for p in parse.load(agent, prmci_dir=prmci_dir)}
    with open(f"{prmci_dir}/validation/validation_results.csv", encoding="utf-8-sig") as fh:
        V = [r for r in csv.DictReader(fh) if r["pr_id"] in prs]
    recs = {r["pr_id"]: check_claims.check_pr(prs[r["pr_id"]]) for r in V}
    lab = lambda r: (r["consensus_label"] or r["annotator1_label"]).lower()
    pos_mis = lambda r: r["binary_label"] == "misaligned"
    pos_any = lambda r: lab(r) != "aligned"
    methods = {
        "paper_heuristic": lambda r: r["predicted_heuristic"] == "misaligned",
        "paper_embedding": lambda r: r["predicted_embedding"] == "misaligned",
        "paper_both": lambda r: r["predicted_both"] == "misaligned",
        "deltoryx_gate": lambda r: recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
    }
    res = {"n": len(V), "labels": dict(sorted(collections.Counter(lab(r) for r in V).items())), "metrics": {}}
    for name, f in methods.items():
        res["metrics"][name] = {"misaligned": metrics(V, f, pos_mis), "partial_o_misaligned": metrics(V, f, pos_any)}
    res["signals"] = dict(sorted(collections.Counter(s for r in recs.values() for s in r["signals"]).items()))
    res["claims"] = {
        "total": sum(r["n_claims"] for r in recs.values()),
        "supported": sum(r["n_supported"] for r in recs.values()),
    }
    res["per_pr"] = [
        {
            "pr_id": r["pr_id"],
            "label": lab(r),
            "gate": recs[r["pr_id"]]["verdict"],
            "n_claims": recs[r["pr_id"]]["n_claims"],
            "n_supported": recs[r["pr_id"]]["n_supported"],
            "receipt_sha256": recs[r["pr_id"]]["receipt_sha256"],
            "url": recs[r["pr_id"]]["url"],
        }
        for r in V
    ]
    sanitized = [_sanitize_record(r) for r in recs.values()]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": res, "records": sanitized}, fh, indent=1, ensure_ascii=False, sort_keys=True)
    print(f"n={res['n']} labels={res['labels']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
