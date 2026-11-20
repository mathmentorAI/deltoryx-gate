"""Offline threshold & relevance-guard calibration on AIDev-Dev (`N2_hard` + `originals`).

Strict rules:
- Runs 100% offline in replay mode using external AIDev-Dev JSONL & NIM caches.
- Calibrates `RelevanceConfig` and `VerdictThresholds` ONLY on `N2_hard` and `originals`
  (never touches `Eval A` or `Eval B`).
- Enforces target `originals FPR (INCONSISTENT) <= 10.0%` while minimizing N2 laxity
  leakage and maximizing N2 detection recall.
- Writes strictly sanitized aggregate metrics (zero PR body/diff text) to `results_aidev_dev.json`.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
import re
import sys
from typing import Any, Mapping, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_phase3 as p3  # noqa: E402
import prmci_heuristic  # noqa: E402
from claimlayer.v4.domains.code_change_inference import (  # noqa: E402
    CLAIM_STATES,
    RelevanceConfig,
    VerdictThresholds,
    build_typed_diff_ledger,
    compute_pr_verdict,
    evaluate_i5_structural_signals,
    repo_identifier_tokens,
    specific_tokens,
    verify_behavior_graph,
)

DEFAULT_AIDEV_DIR = os.path.expanduser("~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3")
DEFAULT_OUT = os.path.join(_THIS_DIR, "results_aidev_dev.json")


def _extract_n3_injected_sentence(n3_body: str, orig_body: str) -> str:
    """For N3 mutations, isolate the line(s) added relative to the unmutated original PR body."""
    orig_lines = set((orig_body or "").splitlines())
    added: list[str] = []
    for ln in (n3_body or "").splitlines():
        s = ln.strip()
        if not s or ln in orig_lines:
            continue
        cleaned = re.sub(r"^[-*]\s+", "", s).strip()
        if cleaned:
            added.append(cleaned)
    if added:
        return " ".join(added)
    return (n3_body or "").strip().splitlines()[0].strip() if (n3_body or "").strip() else ""


def _matches_injected_claim(
    injected_text: str,
    claim_text: str,
    target: str | None,
    repo_tokens: frozenset[str],
    *,
    operation: str | None = None,
    mutation_kind: str | None = None,
) -> bool:
    """In-memory matcher associating an extracted behavioral claim with an injected N1/N2/N3 sentence."""
    inj_l = (injected_text or "").strip().lower()
    cl_l = (claim_text or "").strip().lower()
    if not inj_l or not cl_l:
        return False
    if mutation_kind:
        if mutation_kind == "claimed_doc_file_but_zero_docs":
            if not (operation == "documents" or re.search(r"\b(readme|changelog|docs?|documentation)\b", cl_l)):
                return False
        elif mutation_kind == "claimed_tests_but_zero_tests":
            if not (operation == "adds_test_for" or re.search(r"\b(test|tests|unit|integration|e2e|spec)\b", cl_l)):
                return False
        elif mutation_kind == "dependency_version_mismatch":
            if not (operation == "updates_dependency" or re.search(r"\b(99\.9\.9|dependenc\w*|version|bump|upgrade)\b", cl_l)):
                return False
        elif mutation_kind == "polarity_flip_remove_added_symbol":
            if not (operation == "removes_support_for" or re.search(r"\b(remov\w*|delet\w*|drop\w*|deprecat\w*)\b", cl_l)):
                return False
    if cl_l in inj_l or inj_l in cl_l:
        return True
    inj_toks = set(specific_tokens(inj_l, repo_tokens=repo_tokens, min_len=3))
    cl_toks = set(specific_tokens(f"{cl_l} {target or ''}", repo_tokens=repo_tokens, min_len=3))
    if not inj_toks or not cl_toks:
        # Fallback on raw word overlap when injected sentence relies on stopword-heavy words
        w_inj = {w for w in re.findall(r"[a-z0-9_]{3,}", inj_l)}
        w_cl = {w for w in re.findall(r"[a-z0-9_]{3,}", cl_l)}
        if not w_inj or not w_cl:
            return False
        return (len(w_inj & w_cl) / max(1, min(len(w_inj), len(w_cl)))) >= 0.50
    overlap = len(inj_toks & cl_toks)
    return overlap >= 2 or (overlap / max(1, min(len(inj_toks), len(cl_toks)))) >= 0.50


def _collect_base_replay_artifacts(
    prs: Sequence[Mapping[str, Any]],
    *,
    pass1_model: p3.ResilientCachedModel,
    pass2_model: p3.ResilientCachedModel,
    orig_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Run deterministic replay ONCE per PR in parallel to gather ledgers, Pass-1 literal claims, and Pass-2/3 proposals."""

    def _one(pr: Mapping[str, Any]) -> dict[str, Any]:
        rep = p3.evaluate_pr_phase3(
            pr,
            pass1_model=pass1_model,
            pass2_model=pass2_model,
            relevance_config=RelevanceConfig(),
            thresholds=VerdictThresholds(),
            modes=("d_full",),
            include_claim_text_in_memory=True,
        )
        pr_dict = dict(pr)
        if "patch" not in pr_dict:
            pr_dict["patch"] = "\n---FILE---\n".join(
                (fp.get("patch") or "") for fp in pr_dict.get("file_patches", ()) if (fp.get("patch") or "").strip()
            )
        typed_ledger = build_typed_diff_ledger(pr_dict)
        repo_toks = repo_identifier_tokens(pr_dict.get("repo") or pr_dict.get("url"))
        inj_phrase = str(pr_dict.get("injected_phrase") or "")
        if not inj_phrase and pr_dict.get("injection_level") == "N3_mutation":
            orig_pr = (orig_by_id or {}).get(str(pr_dict.get("base_pr_id") or ""))
            orig_body = str((orig_pr or {}).get("body") or "")
            inj_phrase = _extract_n3_injected_sentence(str(pr_dict.get("body") or ""), orig_body)
        return {
            "pr": pr_dict,
            "ledger": typed_ledger,
            "repo_tokens": repo_toks,
            "injected_ref": inj_phrase,
            "literal_claims": rep["literal_claim_statuses"],
            "draft_rejected": rep["draft_rejected"],
            "reader_rejected": rep["reader_rejected"],
            "read_errors": rep["read_errors"],
            "error_escalated_count": sum(1 for lc in rep["literal_claim_statuses"] if lc["status"] == "error_escalated"),
            "raw_beh_with_proposals": rep.get("_in_memory_beh_proposals", []),
        }

    if workers <= 1 or len(prs) <= 4:
        return [_one(pr) for pr in prs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_one, prs))


def _reverify_behaviors_with_relevance(
    artifacts: Sequence[Mapping[str, Any]],
    rel_cfg: RelevanceConfig,
    *,
    include_ablations: bool = True,
) -> list[dict[str, Any]]:
    """Re-verify Pass 2/3 proposals under candidate `RelevanceConfig` cleanly in memory."""
    out: list[dict[str, Any]] = []
    for art in artifacts:
        ledger = art["ledger"]
        repo_toks = art["repo_tokens"]
        beh_d: list[dict[str, Any]] = []
        beh_b: list[dict[str, Any]] = []
        beh_c: list[dict[str, Any]] = []
        for item in art["raw_beh_with_proposals"]:
            # Pass 2+3 combined for d_full / e_full_no_i5
            r_d = verify_behavior_graph(
                claim_id=item["claim_id"],
                claim_text=item["claim_text"],
                operation=item["operation"],
                target=item["target"],
                proposed_premises=item["merged_premises"],
                ledger=ledger,
                repo_tokens=repo_toks,
                relevance_config=rel_cfg,
                use_declared_rules=True,
                reread_attempted=item["reread_attempted"],
            )
            r_d["_is_injected"] = item["is_injected"]
            beh_d.append(r_d)

            if include_ablations:
                r_b = verify_behavior_graph(
                    claim_id=item["claim_id"],
                    claim_text=item["claim_text"],
                    operation=item["operation"],
                    target=item["target"],
                    proposed_premises=item["merged_premises"],
                    ledger=ledger,
                    repo_tokens=repo_toks,
                    relevance_config=rel_cfg,
                    use_declared_rules=False,
                    reread_attempted=item["reread_attempted"],
                )
                r_b["_is_injected"] = item["is_injected"]
                beh_b.append(r_b)

                # Pass 2 only for c_graph_rules_no_reread
                r_c = verify_behavior_graph(
                    claim_id=item["claim_id"],
                    claim_text=item["claim_text"],
                    operation=item["operation"],
                    target=item["target"],
                    proposed_premises=item["p2_premises"],
                    ledger=ledger,
                    repo_tokens=repo_toks,
                    relevance_config=rel_cfg,
                    use_declared_rules=True,
                    reread_attempted=False,
                )
                r_c["_is_injected"] = item["is_injected"]
                beh_c.append(r_c)

        out.append({**art, "beh_d": beh_d, "beh_b": beh_b, "beh_c": beh_c})
    return out


def _annotate_injected_matches(artifacts: Sequence[Mapping[str, Any]]) -> None:
    """Tag in-memory which extracted behavioral claims correspond to the PR's injected sentence."""
    for art in artifacts:
        inj = art["injected_ref"]
        repo_toks = art["repo_tokens"]
        mut_kind = art["pr"].get("mutation_kind")
        for item in art["raw_beh_with_proposals"]:
            item["is_injected"] = bool(inj) and _matches_injected_claim(
                inj,
                item["claim_text"],
                item["target"],
                repo_toks,
                operation=item["operation"],
                mutation_kind=mut_kind,
            )


def _precompute_verdict_inputs(reverified: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Precompute state_counts and threshold-free I5 signals once per (RelevanceConfig, PR)."""
    feats: list[dict[str, Any]] = []
    for art in reverified:
        pr = art["pr"]
        ledger = art["ledger"]
        sc = {st: 0 for st in CLAIM_STATES}
        for lc in art["literal_claims"]:
            st = lc["status"]
            if st in sc:
                sc[st] += 1
        for br in art["beh_d"]:
            st = br["status"]
            if st in sc:
                sc[st] += 1
        i5_base = evaluate_i5_structural_signals(
            pr=pr,
            ledger=ledger,
            thresholds=VerdictThresholds(include_i5=True, i5_understated_min_share=0.0),
        )
        unmentioned_share = -1.0
        has_i5_d = False
        has_i5_p = False
        for sig in i5_base:
            name = sig.get("signal")
            if name == "D_doc_only_but_code":
                has_i5_d = True
            elif name == "P_placeholder_large_diff":
                has_i5_p = True
            elif name == "U_severe_understated_files":
                if "unmentioned_count" in sig and "total_sig_files" in sig:
                    unmentioned_share = int(sig["unmentioned_count"]) / max(1, int(sig["total_sig_files"]))
                else:
                    unmentioned_share = float(sig.get("share", -1.0))
        feats.append(
            {
                "pr_id": str(pr["pr_id"]),
                "truncated": ledger.truncated,
                "error_escalated_count": art["error_escalated_count"],
                "read_errors": bool(art["read_errors"]),
                "files_count": ledger.files_count,
                "draft_rejected_sum": sum(art["draft_rejected"].values()),
                "state_counts": sc,
                "has_i5_d": has_i5_d,
                "has_i5_p": has_i5_p,
                "unmentioned_share": unmentioned_share,
            }
        )
    return feats


def _fast_verdict(feat: Mapping[str, Any], th: VerdictThresholds) -> str:
    """Fast exact equivalent of `compute_pr_verdict(...)['verdict']` for threshold grid search."""
    if feat["truncated"] or feat["error_escalated_count"] >= 1 or feat["read_errors"]:
        return "ESCALATE"
    sc = feat["state_counts"]
    tot_lit = sc["literal_supported"] + sc["literal_unsupported"]
    tot_beh = sc["derived_supported"] + sc["hypothesis_with_verified_premises"] + sc["no_premises_in_diff"]
    tot_chk = tot_lit + tot_beh + sc["premise_contradicted"]
    inc = False
    if feat["files_count"] == 0 and (tot_chk > 0 or sc["unverifiable"] > 0):
        inc = True
    if sc["premise_contradicted"] >= 1:
        inc = True
    n_phan = sc["no_premises_in_diff"]
    if n_phan >= th.k_phantom and tot_beh > 0:
        if (n_phan / tot_beh) >= th.theta_phantom and (
            tot_beh >= 2 or (th.singleton_phantom_inconsistent and sc["literal_supported"] == 0)
        ):
            inc = True
    n_lunsup = sc["literal_unsupported"]
    if n_lunsup >= th.k_lit and tot_lit > 0 and (n_lunsup / tot_lit) >= th.theta_lit:
        inc = True
    if th.include_i5 and (
        feat["has_i5_d"] or feat["has_i5_p"] or (feat["unmentioned_share"] >= th.i5_understated_min_share)
    ):
        inc = True
    if inc:
        return "INCONSISTENT"
    rev = False
    if th.review_on_single_unsupported_or_phantom:
        if (
            n_phan >= th.review_k_phantom
            and tot_beh > 0
            and (n_phan / tot_beh) >= th.review_theta_phantom
        ):
            rev = True
        if (
            n_lunsup >= th.review_k_lit
            and tot_lit > 0
            and (n_lunsup / tot_lit) >= th.review_theta_lit
        ):
            rev = True
    if feat["draft_rejected_sum"] > 0 and (sc["literal_supported"] + sc["derived_supported"] == 0):
        rev = True
    if (
        th.review_on_all_unruled_hypothesis
        and tot_beh >= 2
        and sc["hypothesis_with_verified_premises"] == tot_beh
        and sc["literal_supported"] == 0
    ):
        rev = True
    return "REVIEW" if rev else "CONSISTENT"


def _eval_verdicts(
    reverified: Sequence[Mapping[str, Any]],
    th: VerdictThresholds,
    *,
    mode: str = "d_full",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for art in reverified:
        pr = art["pr"]
        if mode == "a_literal":
            beh: Sequence[Mapping[str, Any]] = ()
            th_eff = VerdictThresholds(**{**th.to_dict(), "include_i5": False})
        elif mode == "b_graph_no_rules":
            beh = art["beh_b"]
            th_eff = VerdictThresholds(**{**th.to_dict(), "include_i5": False})
        elif mode == "c_graph_rules_no_reread":
            beh = art["beh_c"]
            th_eff = VerdictThresholds(**{**th.to_dict(), "include_i5": False})
        elif mode == "e_full_no_i5":
            beh = art["beh_d"]
            th_eff = VerdictThresholds(**{**th.to_dict(), "include_i5": False})
        else:
            beh = art["beh_d"]
            th_eff = th

        v_res = compute_pr_verdict(
            pr=pr,
            ledger=art["ledger"],
            literal_claims=art["literal_claims"],
            behavior_results=beh,
            draft_rejected=art["draft_rejected"],
            reader_rejected=art["reader_rejected"],
            error_escalated_count=art["error_escalated_count"],
            read_errors=art["read_errors"],
            thresholds=th_eff,
            prompt_digests=p3.prompt_digests(),
        )
        rows.append(
            {
                "pr_id": str(pr["pr_id"]),
                "base_pr_id": str(pr.get("base_pr_id") or pr["pr_id"]),
                "agent": str(pr["agent"]),
                "injection_level": str(pr.get("injection_level") or "original"),
                "mutation_kind": pr.get("mutation_kind"),
                "verdict": v_res["verdict"],
                "reasons": v_res["reasons"],
                "state_counts": v_res["state_counts"],
                "i5_signals": v_res["i5_signals"],
                "reader_rejected": v_res["reader_rejected"],
                "beh_claims": beh,
            }
        )
    return rows


def _leakage_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total_prs = len(rows)
    prs_with_matched_inj = 0
    matched_claims = 0
    strict_leaked = 0
    broad_leaked = 0
    caught = 0
    for r in rows:
        inj_claims = [c for c in r["beh_claims"] if c.get("_is_injected")]
        if inj_claims:
            prs_with_matched_inj += 1
        for c in inj_claims:
            matched_claims += 1
            st = c["status"]
            if st == "derived_supported":
                strict_leaked += 1
                broad_leaked += 1
            elif st == "hypothesis_with_verified_premises":
                broad_leaked += 1
            elif st in ("no_premises_in_diff", "premise_contradicted"):
                caught += 1
    return {
        "total_prs": total_prs,
        "prs_with_matched_injected_claim": prs_with_matched_inj,
        "matched_injected_claims": matched_claims,
        "strict_leakage_derived_supported": strict_leaked,
        "strict_leakage_rate": round(strict_leaked / max(1, matched_claims), 4),
        "broad_leakage_any_verified_premise": broad_leaked,
        "broad_leakage_rate": round(broad_leaked / max(1, matched_claims), 4),
        "caught_no_premise_or_contradicted": caught,
        "catch_rate": round(caught / max(1, matched_claims), 4),
    }


def _tier_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    vc = collections.Counter(r["verdict"] for r in rows)
    n_inc = vc.get("INCONSISTENT", 0)
    n_rev = vc.get("REVIEW", 0)
    sc = collections.Counter()
    for r in rows:
        for k, v in r["state_counts"].items():
            sc[k] += v
    by_ag = {}
    for ag in sorted({r["agent"] for r in rows}):
        sub = [r for r in rows if r["agent"] == ag]
        sub_vc = collections.Counter(x["verdict"] for x in sub)
        by_ag[ag] = {
            "n": len(sub),
            "verdicts": dict(sorted(sub_vc.items())),
            "inconsistent_rate": round(sub_vc.get("INCONSISTENT", 0) / max(1, len(sub)), 4),
            "broad_flag_rate": round((sub_vc.get("INCONSISTENT", 0) + sub_vc.get("REVIEW", 0)) / max(1, len(sub)), 4),
        }
    return {
        "n": n,
        "verdicts": dict(sorted(vc.items())),
        "inconsistent_rate": round(n_inc / max(1, n), 4),
        "broad_flag_rate": round((n_inc + n_rev) / max(1, n), 4),
        "state_counts": {k: sc.get(k, 0) for k in CLAIM_STATES},
        "by_agent": by_ag,
        "leakage": _leakage_stats(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate Phase 3 thresholds & relevance guard on AIDev-Dev.")
    ap.add_argument("--aidev-dir", default=DEFAULT_AIDEV_DIR)
    ap.add_argument("--n1-file", default="aidev_dev_n1_easy_50.jsonl")
    ap.add_argument("--pass1-cache", default=os.path.join(DEFAULT_AIDEV_DIR, "caches", "pass1_dev.jsonl"))
    ap.add_argument("--pass2-cache", default=os.path.join(DEFAULT_AIDEV_DIR, "caches", "pass2_dev.jsonl"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    def _read(name: str) -> list[dict[str, Any]]:
        fp = os.path.join(args.aidev_dir, name)
        return [json.loads(ln) for ln in open(fp, encoding="utf-8") if ln.strip()]

    orig_prs = _read("aidev_dev_originals.jsonl")
    n1_prs = _read(args.n1_file)
    n2_prs = _read("aidev_dev_n2_hard.jsonl")
    n3_prs = _read("aidev_dev_n3_mutation.jsonl")

    m1 = p3.ResilientCachedModel(None, args.pass1_cache)
    m2 = p3.ResilientCachedModel(None, args.pass2_cache, allow_pass3_replay_miss_as_empty=True)

    print("Loading & replaying cached AIDev-Dev completions once...")
    orig_by_id = {str(p["pr_id"]): p for p in orig_prs}
    art_orig = _collect_base_replay_artifacts(orig_prs, pass1_model=m1, pass2_model=m2)
    art_n1 = _collect_base_replay_artifacts(n1_prs, pass1_model=m1, pass2_model=m2)
    art_n2 = _collect_base_replay_artifacts(n2_prs, pass1_model=m1, pass2_model=m2)
    art_n3 = _collect_base_replay_artifacts(n3_prs, pass1_model=m1, pass2_model=m2, orig_by_id=orig_by_id)
    for grp in (art_orig, art_n1, art_n2, art_n3):
        _annotate_injected_matches(grp)

    def _binary_op_metrics(pos_flags: Sequence[bool], neg_flags: Sequence[bool]) -> dict[str, Any]:
        tp = sum(1 for x in pos_flags if x)
        fn = len(pos_flags) - tp
        fp = sum(1 for x in neg_flags if x)
        tn = len(neg_flags) - fp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
        }

    # Identify Pass-1-clean originals under default literal rules for diagnostic reporting alongside raw FPR
    base_lit_rows = _eval_verdicts(
        _reverify_behaviors_with_relevance(art_orig, RelevanceConfig(), include_ablations=False),
        VerdictThresholds(),
        mode="a_literal",
    )
    pass1_clean_orig_ids = {r["pr_id"] for r in base_lit_rows if r["verdict"] != "INCONSISTENT"}

    # Stage A: Strict (INCONSISTENT) grid search on N2_hard (positives) vs originals (negatives); N1 & N3 NEVER touch calibration
    rel_grid = [
        RelevanceConfig(
            min_specific_anchors=msa,
            require_target_match_when_present=rtm,
            require_category_compat=True,
            pass3_require_target_grounded_in_claim=True,
        )
        for msa in (2, 1)
        for rtm in (True,)
    ]
    strict_th_grid = [
        VerdictThresholds(
            k_phantom=kp,
            theta_phantom=tp,
            singleton_phantom_inconsistent=sp,
            k_lit=kl,
            theta_lit=tl,
            review_on_single_unsupported_or_phantom=True,
            review_on_all_unruled_hypothesis=False,
            include_i5=True,
            i5_understated_min_share=ush,
        )
        for kp, tp, sp, kl, tl, ush in itertools.product(
            (1, 2, 3, 4),
            (0.34, 0.50, 0.67, 0.75, 0.85),
            (True, False),
            (2, 3, 4, 5, 6, 8),
            (0.50, 0.67, 0.75, 0.80, 0.90),
            (0.75, 0.85, 0.90, 0.95),
        )
    ]

    candidates: list[dict[str, Any]] = []
    cached_revs: dict[tuple[int, bool, bool], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = {}
    parity_checked = False
    for rel_cfg in rel_grid:
        rev_orig = _reverify_behaviors_with_relevance(art_orig, rel_cfg, include_ablations=False)
        rev_n2 = _reverify_behaviors_with_relevance(art_n2, rel_cfg, include_ablations=False)
        n2_leak = _leakage_stats([{"beh_claims": art["beh_d"]} for art in rev_n2])

        feat_o = _precompute_verdict_inputs(rev_orig)
        feat_n2 = _precompute_verdict_inputs(rev_n2)
        cached_revs[(rel_cfg.min_specific_anchors, rel_cfg.require_target_match_when_present, rel_cfg.pass3_require_target_grounded_in_claim)] = (
            rev_orig,
            rev_n2,
            feat_o,
            feat_n2,
            n2_leak,
        )
        if not parity_checked:
            for th_check in (
                VerdictThresholds(),
                VerdictThresholds(
                    k_phantom=3,
                    theta_phantom=0.75,
                    k_lit=5,
                    theta_lit=0.80,
                    review_k_phantom=2,
                    review_theta_phantom=0.34,
                    review_k_lit=3,
                    review_theta_lit=0.50,
                    review_on_all_unruled_hypothesis=False,
                    i5_understated_min_share=0.95,
                ),
            ):
                exact_o = _eval_verdicts(rev_orig, th_check, mode="d_full")
                mismatches = sum(
                    1 for ft, ex in zip(feat_o, exact_o) if _fast_verdict(ft, th_check) != ex["verdict"]
                )
                assert mismatches == 0, f"_fast_verdict parity failure: {mismatches} mismatches on originals"
            parity_checked = True

        orig_p1_clean_mask = [ft["pr_id"] in pass1_clean_orig_ids for ft in feat_o]
        orig_p1_clean_cnt = max(1, sum(1 for c in orig_p1_clean_mask if c))

        for th in strict_th_grid:
            verdicts_o = [_fast_verdict(ft, th) for ft in feat_o]
            verdicts_n2 = [_fast_verdict(ft, th) for ft in feat_n2]

            m_strict = _binary_op_metrics(
                [v == "INCONSISTENT" for v in verdicts_n2],
                [v == "INCONSISTENT" for v in verdicts_o],
            )
            m_broad_default = _binary_op_metrics(
                [v in ("INCONSISTENT", "REVIEW") for v in verdicts_n2],
                [v in ("INCONSISTENT", "REVIEW") for v in verdicts_o],
            )
            orig_p1_clean_inc = (
                sum(1 for v, is_c in zip(verdicts_o, orig_p1_clean_mask) if is_c and v == "INCONSISTENT")
                / orig_p1_clean_cnt
            )

            meets_fpr = m_strict["fpr"] <= 0.10
            meets_leak = n2_leak["strict_leakage_rate"] <= 0.10
            obj = (
                (100.0 if (meets_fpr and meets_leak) else (50.0 if meets_fpr else 0.0))
                + 10.0 * m_strict["f1"]
                - 5.0 * n2_leak["strict_leakage_rate"]
                - 0.5 * m_strict["fpr"]
            )
            candidates.append(
                {
                    "objective": round(obj, 6),
                    "meets_fpr_10pct": meets_fpr,
                    "meets_n2_strict_leak_10pct": meets_leak,
                    "relevance_config": rel_cfg.to_dict(),
                    "thresholds": th.to_dict(),
                    "strict_op_inconsistent": m_strict,
                    "broad_op_default_review": m_broad_default,
                    "orig_pass1_clean_fpr_inconsistent": round(orig_p1_clean_inc, 4),
                    "n2_strict_leakage_rate": n2_leak["strict_leakage_rate"],
                    "n2_broad_leakage_rate": n2_leak["broad_leakage_rate"],
                }
            )

    candidates.sort(
        key=lambda c: (
            c["meets_fpr_10pct"] and c["meets_n2_strict_leak_10pct"],
            c["meets_fpr_10pct"],
            c["strict_op_inconsistent"]["f1"],
            -c["n2_strict_leakage_rate"],
            -c["strict_op_inconsistent"]["fpr"],
            -c["n2_broad_leakage_rate"],
        ),
        reverse=True,
    )
    best_strict = candidates[0]
    best_rel = RelevanceConfig(**best_strict["relevance_config"])
    strict_th_dict = dict(best_strict["thresholds"])

    # Stage B: Decoupled Broad (INCONSISTENT + REVIEW) optimization under pre-fixed FPR(originals) <= 25%
    _, _, feat_o_best, feat_n2_best, _ = cached_revs[
        (best_rel.min_specific_anchors, best_rel.require_target_match_when_present, best_rel.pass3_require_target_grounded_in_claim)
    ]
    orig_p1_clean_mask = [ft["pr_id"] in pass1_clean_orig_ids for ft in feat_o_best]
    orig_p1_clean_cnt = max(1, sum(1 for c in orig_p1_clean_mask if c))

    review_candidates_param_tuples: list[tuple[bool, int, float, int, float, bool]] = []
    for rh in (False, True):
        review_candidates_param_tuples.append((False, 99, 1.0, 99, 1.0, rh))
        for rkp, rtp, rkl, rtl in itertools.product(
            (1, 2, 3, 4, 99),
            (0.0, 0.25, 0.34, 0.50, 0.67, 0.75),
            (1, 2, 3, 4, 5, 99),
            (0.0, 0.25, 0.34, 0.50, 0.67, 0.75),
        ):
            # Collapse unreachable (k=99, non-0.0 theta) duplicates
            if rkp == 99 and rtp != 0.0:
                continue
            if rkl == 99 and rtl != 0.0:
                continue
            review_candidates_param_tuples.append((True, rkp, rtp, rkl, rtl, rh))

    broad_candidates: list[dict[str, Any]] = []
    for rev_sup, rkp, rtp, rkl, rtl, rh in review_candidates_param_tuples:
        th_b = VerdictThresholds(
            **{
                **strict_th_dict,
                "review_on_single_unsupported_or_phantom": rev_sup,
                "review_k_phantom": rkp,
                "review_theta_phantom": rtp,
                "review_k_lit": rkl,
                "review_theta_lit": rtl,
                "review_on_all_unruled_hypothesis": rh,
            }
        )
        v_o = [_fast_verdict(ft, th_b) for ft in feat_o_best]
        v_n2 = [_fast_verdict(ft, th_b) for ft in feat_n2_best]
        m_broad = _binary_op_metrics(
            [v in ("INCONSISTENT", "REVIEW") for v in v_n2],
            [v in ("INCONSISTENT", "REVIEW") for v in v_o],
        )
        m_strict_chk = _binary_op_metrics(
            [v == "INCONSISTENT" for v in v_n2],
            [v == "INCONSISTENT" for v in v_o],
        )
        broad_candidates.append(
            {
                "meets_broad_fpr_25pct": m_broad["fpr"] <= 0.25,
                "review_params": {
                    "review_on_single_unsupported_or_phantom": rev_sup,
                    "review_k_phantom": rkp,
                    "review_theta_phantom": rtp,
                    "review_k_lit": rkl,
                    "review_theta_lit": rtl,
                    "review_on_all_unruled_hypothesis": rh,
                },
                "broad_op_inconsistent_or_review": m_broad,
                "strict_op_inconsistent": m_strict_chk,
                "thresholds": th_b.to_dict(),
            }
        )

    broad_candidates.sort(
        key=lambda c: (
            c["meets_broad_fpr_25pct"],
            c["broad_op_inconsistent_or_review"]["f1"],
            c["broad_op_inconsistent_or_review"]["recall"],
            -c["broad_op_inconsistent_or_review"]["fpr"],
            -c["review_params"]["review_k_phantom"],
            -c["review_params"]["review_k_lit"],
        ),
        reverse=True,
    )
    best_broad = broad_candidates[0]
    best_th = VerdictThresholds(**best_broad["thresholds"])

    # Compute Broad FPR vs F1 Pareto curve across caps
    pareto_caps = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75)
    broad_pareto_curve: list[dict[str, Any]] = []
    for cap in pareto_caps:
        elig = [c for c in broad_candidates if c["broad_op_inconsistent_or_review"]["fpr"] <= cap]
        if not elig:
            broad_pareto_curve.append({"fpr_cap": cap, "reachable": False})
        else:
            top_c = max(
                elig,
                key=lambda c: (
                    c["broad_op_inconsistent_or_review"]["f1"],
                    c["broad_op_inconsistent_or_review"]["recall"],
                    -c["broad_op_inconsistent_or_review"]["fpr"],
                ),
            )
            broad_pareto_curve.append(
                {
                    "fpr_cap": cap,
                    "reachable": True,
                    "broad_op_inconsistent_or_review": top_c["broad_op_inconsistent_or_review"],
                    "review_params": top_c["review_params"],
                }
            )

    best = {
        **best_strict,
        "thresholds": best_th.to_dict(),
        "broad_op_inconsistent_or_review": best_broad["broad_op_inconsistent_or_review"],
        "meets_broad_fpr_25pct": best_broad["meets_broad_fpr_25pct"],
    }

    # Full evaluation across all 4 tiers & all 5 ablations under chosen frozen config
    fin_orig = _reverify_behaviors_with_relevance(art_orig, best_rel)
    fin_n1 = _reverify_behaviors_with_relevance(art_n1, best_rel)
    fin_n2 = _reverify_behaviors_with_relevance(art_n2, best_rel)
    fin_n3 = _reverify_behaviors_with_relevance(art_n3, best_rel)

    ablations_report: dict[str, Any] = {}
    d_full_orig_inc_records: list[dict[str, Any]] = []
    reader_rejected_summary: dict[str, Any] = {}
    for m in p3.ABLATION_MODES:
        r_o = _eval_verdicts(fin_orig, best_th, mode=m)
        r_n1 = _eval_verdicts(fin_n1, best_th, mode=m)
        r_n2 = _eval_verdicts(fin_n2, best_th, mode=m)
        r_n3 = _eval_verdicts(fin_n3, best_th, mode=m)

        # N3 subtype recall breakdown
        n3_subtypes: dict[str, Any] = {}
        for mk in sorted({str(x.get("mutation_kind") or "unknown") for x in r_n3}):
            sub = [x for x in r_n3 if x.get("mutation_kind") == mk]
            n_inc = sum(1 for x in sub if x["verdict"] == "INCONSISTENT")
            n_brd = sum(1 for x in sub if x["verdict"] in ("INCONSISTENT", "REVIEW"))
            n3_subtypes[mk] = {
                "n": len(sub),
                "inconsistent_recall": round(n_inc / max(1, len(sub)), 4),
                "broad_recall": round(n_brd / max(1, len(sub)), 4),
            }

        orig_metrics = _tier_metrics(r_o)
        o_p1_clean = [x for x in r_o if x["pr_id"] in pass1_clean_orig_ids]
        orig_metrics["pass1_clean_n"] = len(o_p1_clean)
        orig_metrics["pass1_clean_inconsistent_rate"] = round(
            sum(1 for x in o_p1_clean if x["verdict"] == "INCONSISTENT") / max(1, len(o_p1_clean)), 4
        )
        orig_metrics["behavioral_i2_i3_rate"] = round(
            sum(
                1
                for x in r_o
                if any(rs.startswith(("I2_premise_contradicted", "I3_phantom_behavior")) for rs in x["reasons"])
            )
            / max(1, len(r_o)),
            4,
        )
        op_strict = _binary_op_metrics(
            [r["verdict"] == "INCONSISTENT" for r in r_n2],
            [r["verdict"] == "INCONSISTENT" for r in r_o],
        )
        op_broad = _binary_op_metrics(
            [r["verdict"] in ("INCONSISTENT", "REVIEW") for r in r_n2],
            [r["verdict"] in ("INCONSISTENT", "REVIEW") for r in r_o],
        )
        ablations_report[m] = {
            "operating_points_n2_vs_originals": {
                "strict_inconsistent": op_strict,
                "broad_inconsistent_or_review": op_broad,
            },
            "originals": orig_metrics,
            "n1_easy_50": _tier_metrics(r_n1),
            "n2_hard": _tier_metrics(r_n2),
            "n3_mutation": {**_tier_metrics(r_n3), "by_mutation_kind": n3_subtypes},
        }
        if m == "d_full":
            d_full_orig_inc_records = [
                {"pr_id": x["pr_id"], "agent": x["agent"], "reasons": x["reasons"]}
                for x in r_o
                if x["verdict"] == "INCONSISTENT"
            ]
            reader_rejected_summary = {}
            for sname, srows in (("originals", r_o), ("n1_easy_50", r_n1), ("n2_hard", r_n2), ("n3_mutation", r_n3)):
                prs_with_rej = [x for x in srows if x.get("reader_rejected")]
                items_total = sum(len(x.get("reader_rejected") or ()) for x in srows)
                esc_rows = [x for x in srows if x["verdict"] == "ESCALATE"]
                reader_rejected_summary[sname] = {
                    "n": len(srows),
                    "prs_with_reader_rejected": len(prs_with_rej),
                    "rejected_items_total": items_total,
                    "escalate_count": len(esc_rows),
                }

    # Heuristic baseline on all 4 Dev splits + N2_hard vs originals F1/FPR
    heur_report: dict[str, Any] = {}
    heur_by_split: dict[str, list[bool]] = {}
    for tname, grp in (("originals", orig_prs), ("n1_easy_50", n1_prs), ("n2_hard", n2_prs), ("n3_mutation", n3_prs)):
        h_preds = [prmci_heuristic.score_pr_heuristic(p) for p in grp]
        flags = [x["predicted_heuristic"] == "misaligned" for x in h_preds]
        heur_by_split[tname] = flags
        n_mis = sum(1 for f in flags if f)
        heur_report[tname] = {
            "n": len(h_preds),
            "flagged_misaligned": n_mis,
            "rate": round(n_mis / max(1, len(h_preds)), 4),
        }
    heur_report["n2_vs_originals"] = _binary_op_metrics(heur_by_split["n2_hard"], heur_by_split["originals"])

    rel_unhardened = RelevanceConfig(
        min_specific_anchors=best_rel.min_specific_anchors,
        require_target_match_when_present=best_rel.require_target_match_when_present,
        require_category_compat=best_rel.require_category_compat,
        pass3_require_target_grounded_in_claim=False,
    )

    def _summarize_pass3_effect(revs_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        out_eff: dict[str, Any] = {}
        for tname, r_list in revs_map.items():
            rr_claims = [c for a in r_list for c in a["beh_d"] if c.get("reread_attempted")]
            inj_rr = [c for c in rr_claims if c.get("_is_injected")]
            sc_rr = collections.Counter(c["status"] for c in rr_claims)
            sc_inj = collections.Counter(c["status"] for c in inj_rr)
            out_eff[tname] = {
                "reread_attempted_claims": len(rr_claims),
                "derived_supported": sc_rr.get("derived_supported", 0),
                "hypothesis_with_verified_premises": sc_rr.get("hypothesis_with_verified_premises", 0),
                "no_premises_in_diff": sc_rr.get("no_premises_in_diff", 0),
                "premise_contradicted": sc_rr.get("premise_contradicted", 0),
                "injected_reread_claims": len(inj_rr),
                "injected_leaked_any_premise": sc_inj.get("derived_supported", 0) + sc_inj.get("hypothesis_with_verified_premises", 0),
                "injected_leaked_derived": sc_inj.get("derived_supported", 0),
            }
        return out_eff

    pass3_reread_effect = {
        "before_target_grounding_guard": _summarize_pass3_effect(
            {
                "originals": _reverify_behaviors_with_relevance(art_orig, rel_unhardened, include_ablations=False),
                "n1_easy_50": _reverify_behaviors_with_relevance(art_n1, rel_unhardened, include_ablations=False),
                "n2_hard": _reverify_behaviors_with_relevance(art_n2, rel_unhardened, include_ablations=False),
                "n3_mutation": _reverify_behaviors_with_relevance(art_n3, rel_unhardened, include_ablations=False),
            }
        ),
        "after_target_grounding_guard": _summarize_pass3_effect(
            {
                "originals": fin_orig,
                "n1_easy_50": fin_n1,
                "n2_hard": fin_n2,
                "n3_mutation": fin_n3,
            }
        ),
    }

    payload = {
        "calibration_scope": "N2_hard (positives) vs originals (negatives) only on AIDev-Dev; N1 (50) and N3 (200) reported only",
        "operating_point_protocol": {
            "strict": "verdict == INCONSISTENT (maps to PR-MCI misaligned)",
            "broad": "verdict in (INCONSISTENT, REVIEW) (maps to PR-MCI partial | misaligned)",
        },
        "selected_config": {
            "relevance_config": best["relevance_config"],
            "thresholds": best["thresholds"],
            "calibration_summary": {
                k: v for k, v in best.items() if k not in ("relevance_config", "thresholds")
            },
        },
        "broad_pareto_curve": broad_pareto_curve,
        "pass3_reread_effect": pass3_reread_effect,
        "pass3_replay_cache_misses": {
            "uncached_claims_count": len(m2.pass3_replay_miss_claims),
            "uncached_prs_count": len({c.split("::beh::", 1)[0] for c in m2.pass3_replay_miss_claims}),
        },
        "top_5_grid_candidates": candidates[:5],
        "originals_inconsistent_d_full": d_full_orig_inc_records,
        "reader_rejected_summary": reader_rejected_summary,
        "ablations": ablations_report,
        "paper_heuristic_baseline": heur_report,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    print("Selected calibrated config:")
    print(json.dumps(payload["selected_config"], indent=2, sort_keys=True))
    print("Full d_full summary (orig FPR / N1(50) / N2(184) / N3(200)):")
    dfull = ablations_report["d_full"]
    print(
        json.dumps(
            {
                "operating_points_n2_vs_originals": dfull["operating_points_n2_vs_originals"],
                "originals_raw_inconsistent": dfull["originals"]["inconsistent_rate"],
                "originals_pass1_clean_fpr": dfull["originals"]["pass1_clean_inconsistent_rate"],
                "originals_raw_broad": dfull["originals"]["broad_flag_rate"],
                "n1_inconsistent_recall": dfull["n1_easy_50"]["inconsistent_rate"],
                "n1_broad_recall": dfull["n1_easy_50"]["broad_flag_rate"],
                "n1_leakage": dfull["n1_easy_50"]["leakage"],
                "n2_inconsistent_recall": dfull["n2_hard"]["inconsistent_rate"],
                "n2_broad_recall": dfull["n2_hard"]["broad_flag_rate"],
                "n2_leakage": dfull["n2_hard"]["leakage"],
                "n3_inconsistent_recall": dfull["n3_mutation"]["inconsistent_rate"],
                "n3_broad_recall": dfull["n3_mutation"]["broad_flag_rate"],
                "n3_leakage": dfull["n3_mutation"]["leakage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
