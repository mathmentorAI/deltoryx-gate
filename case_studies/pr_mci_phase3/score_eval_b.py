"""Compute inter-annotator agreement (Cohen's kappa) and Evaluation B consensus metrics."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from typing import Any, Mapping, Sequence

LABELS_3 = ("aligned", "partial", "misaligned")


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def cohens_kappa(
    y1: Sequence[str],
    y2: Sequence[str],
    *,
    categories: Sequence[str] = LABELS_3,
    weighted: str | None = None,
) -> dict[str, Any]:
    """Compute unweighted or linear/quadratic weighted Cohen's kappa without external deps."""
    if len(y1) != len(y2):
        raise ValueError(f"Mismatched sequence lengths: {len(y1)} vs {len(y2)}")
    n = len(y1)
    if n == 0:
        return {"n": 0, "observed_agreement": 0.0, "expected_agreement": 0.0, "kappa": 0.0, "matrix": {}}

    k = len(categories)
    idx_map = {c: i for i, c in enumerate(categories)}
    mat = [[0 for _ in range(k)] for _ in range(k)]
    for a, b in zip(y1, y2):
        if a not in idx_map or b not in idx_map:
            raise ValueError(f"Unknown category in pair ({a!r}, {b!r}); allowed={categories}")
        mat[idx_map[a]][idx_map[b]] += 1

    row_tot = [sum(mat[i]) for i in range(k)]
    col_tot = [sum(mat[i][j] for i in range(k)) for j in range(k)]

    # Weight matrix w[i][j] = agreement weight (1.0 on diagonal)
    w = [[0.0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if weighted is None:
                w[i][j] = 1.0 if i == j else 0.0
            elif weighted == "linear":
                w[i][j] = 1.0 - (abs(i - j) / max(1, k - 1))
            elif weighted == "quadratic":
                w[i][j] = 1.0 - ((abs(i - j) / max(1, k - 1)) ** 2)
            else:
                raise ValueError(f"Unsupported weight scheme: {weighted}")

    p_o = sum(w[i][j] * mat[i][j] for i in range(k) for j in range(k)) / n
    p_e = sum(w[i][j] * (row_tot[i] * col_tot[j]) for i in range(k) for j in range(k)) / (n * n)

    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    named_mat = {
        categories[i]: {categories[j]: mat[i][j] for j in range(k)}
        for i in range(k)
    }
    return {
        "n": n,
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "kappa": round(kappa, 4),
        "matrix": named_mat,
    }


def score_annotator_pair(
    ann1_rows: Mapping[str, str],
    ann2_rows: Mapping[str, str],
) -> dict[str, Any]:
    """Compute 3-class, linear-weighted, broad, and strict Cohen's kappa across shared PR IDs."""
    shared = sorted(set(ann1_rows) & set(ann2_rows))
    if not shared:
        raise ValueError("No overlapping item_ids/pr_ids between annotator files.")
    y1 = [ann1_rows[k].strip().lower() for k in shared]
    y2 = [ann2_rows[k].strip().lower() for k in shared]
    for v in y1 + y2:
        if v not in LABELS_3:
            raise ValueError(f"Incomplete or invalid annotation label {v!r}; expected one of {LABELS_3}")

    y1_broad = ["pos" if x != "aligned" else "neg" for x in y1]
    y2_broad = ["pos" if x != "aligned" else "neg" for x in y2]
    y1_strict = ["pos" if x == "misaligned" else "neg" for x in y1]
    y2_strict = ["pos" if x == "misaligned" else "neg" for x in y2]

    disagreements = [k for k, a, b in zip(shared, y1, y2) if a != b]
    return {
        "n_shared": len(shared),
        "exact_agreement": round(sum(1 for a, b in zip(y1, y2) if a == b) / len(shared), 4),
        "disagreements_count": len(disagreements),
        "disagreement_ids": disagreements,
        "kappa_3class_unweighted": cohens_kappa(y1, y2, categories=LABELS_3, weighted=None),
        "kappa_3class_linear_weighted": cohens_kappa(y1, y2, categories=LABELS_3, weighted="linear"),
        "kappa_binary_broad_partial_or_misaligned": cohens_kappa(y1_broad, y2_broad, categories=("neg", "pos"), weighted=None),
        "kappa_binary_strict_misaligned": cohens_kappa(y1_strict, y2_strict, categories=("neg", "pos"), weighted=None),
    }


def compute_binary_eval(preds: Sequence[bool], golds: Sequence[bool]) -> dict[str, Any]:
    tp = sum(1 for p, g in zip(preds, golds) if p and g)
    fp = sum(1 for p, g in zip(preds, golds) if p and not g)
    fn = sum(1 for p, g in zip(preds, golds) if not p and g)
    tn = sum(1 for p, g in zip(preds, golds) if not p and not g)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "precision_ci95": wilson_ci(tp, tp + fp),
        "recall": round(rec, 4),
        "recall_ci95": wilson_ci(tp, tp + fn),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "fpr_ci95": wilson_ci(fp, fp + tn),
    }


def _load_labels_csv(path: str, *, label_col: str = "label", id_col: str = "pr_id") -> dict[str, str]:
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return {str(r[id_col]): str(r[label_col]).strip().lower() for r in rows if r.get(id_col)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Eval B inter-annotator kappa and gold consensus.")
    ap.add_argument("--ann1", required=True, help="Path to completed annotator_1.csv")
    ap.add_argument("--ann2", required=True, help="Path to completed annotator_2.csv")
    ap.add_argument("--adjudication", default="", help="Optional completed adjudication.csv with consensus_label")
    ap.add_argument("--predictions-json", default="", help="Optional Eval B pipeline output JSON (per-PR verdicts)")
    ap.add_argument("--out", default="", help="Optional sanitized output JSON summary path")
    args = ap.parse_args()

    a1 = _load_labels_csv(args.ann1, label_col="label")
    a2 = _load_labels_csv(args.ann2, label_col="label")
    out: dict[str, Any] = {"iaa": score_annotator_pair(a1, a2)}

    if args.adjudication:
        gold = _load_labels_csv(args.adjudication, label_col="consensus_label")
        out["gold_distribution"] = dict(sorted(collections.Counter(gold.values()).items()))
        if args.predictions_json:
            pred_payload = json.load(open(args.predictions_json, encoding="utf-8"))
            rows_by_id = {str(r["pr_id"]): r for r in pred_payload.get("rows", [])}
            shared = sorted(set(gold) & set(rows_by_id))
            y_gold_strict = [gold[k] == "misaligned" for k in shared]
            y_gold_broad = [gold[k] != "aligned" for k in shared]
            y_pred_inc = [rows_by_id[k]["verdict"] == "INCONSISTENT" for k in shared]
            y_pred_inc_rev = [rows_by_id[k]["verdict"] in ("INCONSISTENT", "REVIEW") for k in shared]
            out["model_vs_gold"] = {
                "n": len(shared),
                "strict_misaligned_inconsistent": compute_binary_eval(y_pred_inc, y_gold_strict),
                "broad_partial_or_misaligned_inconsistent": compute_binary_eval(y_pred_inc, y_gold_broad),
                "broad_partial_or_misaligned_inc_or_review": compute_binary_eval(y_pred_inc_rev, y_gold_broad),
            }

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
