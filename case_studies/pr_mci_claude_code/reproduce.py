"""Single-command reproducibility driver for the PR-MCI case study on deltoryx-gate.

Reproduces all Phase 1 (Claude Code, n=72) and blind held-out Phase 2 (Codex, Copilot,
Devin, Cursor, n=528) rule, LLM v1, and LLM code_change v2.0 tables from either:
  - offline cached replays (`--reader replay --cache-dir ...`), or
  - live NVIDIA NIM inference (`--reader llm`, requires `NVIDIA_API_KEY`).

Outputs sanitized metric JSONs and SHA-256 receipts only (no PR prose, diffs, caches,
or secrets are written). Supports `--verify-replay` to assert bit-for-bit identical
outputs across two consecutive offline replay runs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

csv.field_size_limit(10**9)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import check_claims  # noqa: E402
import check_claims_v1  # noqa: E402
import eval_v2vocab  # noqa: E402
import llm_gate  # noqa: E402
import parse  # noqa: E402
from claimlayer.v4.reader import LLMProposalReader  # noqa: E402
from claimlayer.v4.triage import verify_draft  # noqa: E402

AGENTS_PHASE1 = ["Claude_Code"]
AGENTS_PHASE2 = ["OpenAI_Codex", "Copilot", "Devin", "Cursor"]


def detect_gate_commit() -> str:
    """Return the short SHA of the latest library commit touching src/claimlayer (or 'unknown')."""
    try:
        out = subprocess.check_output(
            ["git", "-C", _REPO_ROOT, "log", "-n", "1", "--format=%h", "HEAD", "--", "src/claimlayer"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    return "22e3590"


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    return eval_v2vocab.wilson(k, n, z=z)


def compute_metrics(rows: list[dict], pred_fn, pos_fn) -> dict:
    return eval_v2vocab.compute_metrics(rows, pred_fn, pos_fn)


def _metric_block(rows: list[dict], pred_fn) -> dict:
    pos_any = lambda r: r["label"] != "aligned"
    pos_mis = lambda r: r["binary_label"] == "misaligned"
    return {
        "partial_o_misaligned": compute_metrics(rows, pred_fn, pos_any),
        "misaligned": compute_metrics(rows, pred_fn, pos_mis),
    }


def _flip_entry(row_meta: dict, a_verdict: bool, b_verdict: bool) -> dict:
    return {
        "pr_id": row_meta["pr_id"],
        "agent": row_meta["agent"],
        "label": row_meta["label"],
        "binary_label": row_meta["binary_label"],
        "before": "INCONSISTENT" if a_verdict else "CONSISTENT",
        "after": "INCONSISTENT" if b_verdict else "CONSISTENT",
    }


def _write_json(path: str, data: dict) -> str:
    body = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _build_rows_for_agents(prmci_dir: str, agents: list[str], *, legacy_section: bool) -> tuple[list[dict], dict[str, dict]]:
    with open(f"{prmci_dir}/validation/validation_results.csv", encoding="utf-8-sig") as fh:
        val_all = list(csv.DictReader(fh))
    prs_by_id: dict[str, dict] = {}
    agent_by_pid: dict[str, str] = {}
    for ag in agents:
        for p in parse.load(ag, prmci_dir=prmci_dir, legacy_section=legacy_section):
            prs_by_id[p["pr_id"]] = p
            agent_by_pid[p["pr_id"]] = ag
    rows: list[dict] = []
    for r in val_all:
        pid = r["pr_id"]
        if pid not in prs_by_id:
            continue
        lab = (r["consensus_label"] or r["annotator1_label"]).lower()
        rows.append(
            {
                "pr_id": pid,
                "agent": agent_by_pid[pid],
                "url": prs_by_id[pid]["url"],
                "label": lab,
                "binary_label": r["binary_label"],
                "predicted_heuristic": r["predicted_heuristic"],
                "predicted_embedding": r["predicted_embedding"],
                "predicted_both": r["predicted_both"],
            }
        )
    return rows, prs_by_id


def _claim_stats(recs: dict[str, dict]) -> dict:
    tot = sum(x["n_claims"] for x in recs.values())
    sup = sum(x["n_supported"] for x in recs.values())
    f_tot = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "F")
    f_ev = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "F" and c["evidence"])
    f_sup = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "F" and c["status"] == "supported")
    i_tot = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "I")
    i_ev = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "I" and c["evidence"])
    i_sup = sum(1 for x in recs.values() for c in x["claims"] if c["kind"] == "I" and c["status"] == "supported")
    return {
        "n_claims": tot,
        "n_supported": sup,
        "F_total": f_tot,
        "F_with_evidence": f_ev,
        "F_supported": f_sup,
        "F_ev_rejected_by_verify": f_ev - f_sup,
        "I_total": i_tot,
        "I_with_evidence": i_ev,
        "I_supported": i_sup,
        "I_ev_rejected_by_verify": i_ev - i_sup,
    }


def _evaluate_rules_suite(prmci_dir: str, agents: list[str], *, gate_commit: str) -> dict:
    rows_leg, prs_leg = _build_rows_for_agents(prmci_dir, agents, legacy_section=True)
    rows_fix, prs_fix = _build_rows_for_agents(prmci_dir, agents, legacy_section=False)
    assert [r["pr_id"] for r in rows_leg] == [r["pr_id"] for r in rows_fix]

    v1_recs = {r["pr_id"]: check_claims_v1.check_pr(prs_leg[r["pr_id"]]) for r in rows_leg}
    v1_post_recs = {r["pr_id"]: check_claims_v1.check_pr(prs_fix[r["pr_id"]]) for r in rows_fix}
    v2_leg_recs = {r["pr_id"]: check_claims.check_pr(prs_leg[r["pr_id"]], evidence_profile="legal_prose") for r in rows_leg}
    v2_code_recs = {r["pr_id"]: check_claims.check_pr(prs_leg[r["pr_id"]], evidence_profile="code_config") for r in rows_leg}
    v2_post_leg_recs = {
        r["pr_id"]: check_claims.check_pr_corrected(prs_fix[r["pr_id"]], evidence_profile="legal_prose") for r in rows_fix
    }
    v2_post_code_recs = {
        r["pr_id"]: check_claims.check_pr_corrected(prs_fix[r["pr_id"]], evidence_profile="code_config") for r in rows_fix
    }

    def summarize_subset(sub_rows: list[dict]) -> dict:
        methods = {
            "paper_heuristic": lambda r: r["predicted_heuristic"] == "misaligned",
            "paper_embedding": lambda r: r["predicted_embedding"] == "misaligned",
            "paper_both": lambda r: r["predicted_both"] == "misaligned",
            "rules_v1_preregistered": lambda r: v1_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
            "rules_v1_postreview_parse": lambda r: v1_post_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
            "rules_v2_legal_prose": lambda r: v2_leg_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
            "rules_v2_code_config": lambda r: v2_code_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
            "rules_v2_postreview_legal_prose": lambda r: v2_post_leg_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
            "rules_v2_postreview_code_config": lambda r: v2_post_code_recs[r["pr_id"]]["verdict"] == "INCONSISTENT",
        }
        pids = {r["pr_id"] for r in sub_rows}
        sub_recs = lambda rec_map: {pid: rec_map[pid] for pid in sorted(pids)}
        return {
            "n": len(sub_rows),
            "labels": dict(sorted(collections.Counter(r["label"] for r in sub_rows).items())),
            "binary_labels": dict(sorted(collections.Counter(r["binary_label"] for r in sub_rows).items())),
            "metrics": {name: _metric_block(sub_rows, pred) for name, pred in methods.items()},
            "claim_stats": {
                "v1_preregistered": _claim_stats(sub_recs(v1_recs)),
                "v2_legal_prose": _claim_stats(sub_recs(v2_leg_recs)),
                "v2_code_config": _claim_stats(sub_recs(v2_code_recs)),
                "v2_postreview_legal_prose": _claim_stats(sub_recs(v2_post_leg_recs)),
                "v2_postreview_code_config": _claim_stats(sub_recs(v2_post_code_recs)),
            },
        }

    flips_v1 = [
        _flip_entry(r, v1_recs[r["pr_id"]]["verdict"] == "INCONSISTENT", v1_post_recs[r["pr_id"]]["verdict"] == "INCONSISTENT")
        for r in rows_fix
        if v1_recs[r["pr_id"]]["verdict"] != v1_post_recs[r["pr_id"]]["verdict"]
    ]
    flips_legal = [
        _flip_entry(r, v2_leg_recs[r["pr_id"]]["verdict"] == "INCONSISTENT", v2_post_leg_recs[r["pr_id"]]["verdict"] == "INCONSISTENT")
        for r in rows_fix
        if v2_leg_recs[r["pr_id"]]["verdict"] != v2_post_leg_recs[r["pr_id"]]["verdict"]
    ]
    flips_code = [
        _flip_entry(r, v2_code_recs[r["pr_id"]]["verdict"] == "INCONSISTENT", v2_post_code_recs[r["pr_id"]]["verdict"] == "INCONSISTENT")
        for r in rows_fix
        if v2_code_recs[r["pr_id"]]["verdict"] != v2_post_code_recs[r["pr_id"]]["verdict"]
    ]
    flips_profile_orig = [
        _flip_entry(r, v2_leg_recs[r["pr_id"]]["verdict"] == "INCONSISTENT", v2_code_recs[r["pr_id"]]["verdict"] == "INCONSISTENT")
        for r in rows_fix
        if v2_leg_recs[r["pr_id"]]["verdict"] != v2_code_recs[r["pr_id"]]["verdict"]
    ]

    by_agent = {}
    if len(agents) > 1:
        for ag in agents:
            ag_rows = [r for r in rows_fix if r["agent"] == ag]
            by_agent[ag] = summarize_subset(ag_rows)

    per_pr = [
        {
            "pr_id": r["pr_id"],
            "agent": r["agent"],
            "label": r["label"],
            "binary_label": r["binary_label"],
            "v1_preregistered": v1_recs[r["pr_id"]]["verdict"],
            "v2_legal_prose": v2_leg_recs[r["pr_id"]]["verdict"],
            "v2_code_config": v2_code_recs[r["pr_id"]]["verdict"],
            "v2_postreview_legal_prose": v2_post_leg_recs[r["pr_id"]]["verdict"],
            "v2_postreview_code_config": v2_post_code_recs[r["pr_id"]]["verdict"],
            "receipts": {
                "v1_preregistered": v1_recs[r["pr_id"]]["receipt_sha256"],
                "v2_legal_prose": v2_leg_recs[r["pr_id"]]["receipt_sha256"],
                "v2_code_config": v2_code_recs[r["pr_id"]]["receipt_sha256"],
                "v2_postreview_legal_prose": v2_post_leg_recs[r["pr_id"]]["receipt_sha256"],
                "v2_postreview_code_config": v2_post_code_recs[r["pr_id"]]["receipt_sha256"],
            },
        }
        for r in rows_fix
    ]

    return {
        "gate_commit": gate_commit,
        "summary": summarize_subset(rows_fix),
        "by_agent": by_agent,
        "flip_audit": {
            "v1_parse_fix": flips_v1,
            "v2_legal_to_code_config_profile": flips_profile_orig,
            "v2_legal_orig_to_postreview": flips_legal,
            "v2_code_orig_to_postreview": flips_code,
        },
        "rows": per_pr,
    }


def _evaluate_phase1_llm_v1(prmci_dir: str, *, model, gate_commit: str) -> dict:
    # Phase 1 cache was recorded against original parse.section() on Claude Code.
    rows_meta, prs_by_id = _build_rows_for_agents(prmci_dir, AGENTS_PHASE1, legacy_section=True)
    reader = LLMProposalReader(model=model)

    prereg_none, prereg_code, post_none, post_code = {}, {}, {}, {}
    row_out = []
    flips_none, flips_code = [], []

    for rm in rows_meta:
        pid = rm["pr_id"]
        pr = prs_by_id[pid]
        draft = (pr["title"] + "\n\n" + pr["body"]).strip()

        l0, adm0, f0 = llm_gate.build_ledger(pr)
        rep0 = verify_draft(l0, draft, reader=reader, draft_id=f"pr_{pid}")
        v0n, r0n, st0n = llm_gate.verdict(rep0, f0, claim_filter="none")
        v0c, r0c, st0c = llm_gate.verdict(rep0, f0, claim_filter="code")
        prereg_none[pid], prereg_code[pid] = v0n, v0c

        l1, adm1, f1 = llm_gate.build_ledger_corrected(pr)
        rep1 = verify_draft(l1, draft, reader=reader, draft_id=f"pr_{pid}")
        v1n, r1n, st1n = llm_gate.verdict_corrected(rep1, f1, claim_filter="none")
        v1c, r1c, st1c = llm_gate.verdict_corrected(rep1, f1, claim_filter="code")
        post_none[pid], post_code[pid] = v1n, v1c

        if v0n != v1n:
            flips_none.append({**_flip_entry(rm, v0n == "INCONSISTENT", v1n == "INCONSISTENT"), "st_before": st0n, "st_after": st1n})
        if v0c != v1c:
            flips_code.append({**_flip_entry(rm, v0c == "INCONSISTENT", v1c == "INCONSISTENT"), "st_before": st0c, "st_after": st1c})

        row_out.append(
            {
                "pr_id": pid,
                "url": rm["url"],
                "label": rm["label"],
                "binary_label": rm["binary_label"],
                "paper_heuristic": rm["predicted_heuristic"],
                "preregistered_none": {"gate": v0n, "reasons": r0n, "claim_status": st0n, "receipt_sha256": llm_gate._claim_receipt(pid, v0n, r0n, st0n, adm0, rep0)},
                "preregistered_codefilter": {"gate": v0c, "reasons": r0c, "claim_status": st0c, "receipt_sha256": llm_gate._claim_receipt(pid, v0c, r0c, st0c, adm0, rep0)},
                "postreview_none": {"gate": v1n, "reasons": r1n, "claim_status": st1n, "receipt_sha256": llm_gate._claim_receipt(pid, v1n, r1n, st1n, adm1, rep1)},
                "postreview_codefilter": {"gate": v1c, "reasons": r1c, "claim_status": st1c, "receipt_sha256": llm_gate._claim_receipt(pid, v1c, r1c, st1c, adm1, rep1)},
            }
        )

    def sum_claims(field_key: str) -> dict:
        c = collections.Counter()
        for x in row_out:
            c.update(x[field_key]["claim_status"])
        return dict(sorted(c.items()))

    return {
        "gate_commit": gate_commit,
        "vocabulary": {"name": llm_gate.VOC.name, "version": llm_gate.VOC.version, "fingerprint": llm_gate.VOC.fingerprint},
        "reader": getattr(model, "identity", {}),
        "summary": {
            "n": len(rows_meta),
            "metrics": {
                "paper_heuristic": _metric_block(rows_meta, lambda r: r["predicted_heuristic"] == "misaligned"),
                "preregistered_none": _metric_block(rows_meta, lambda r: prereg_none[r["pr_id"]] == "INCONSISTENT"),
                "preregistered_codefilter": _metric_block(rows_meta, lambda r: prereg_code[r["pr_id"]] == "INCONSISTENT"),
                "postreview_none": _metric_block(rows_meta, lambda r: post_none[r["pr_id"]] == "INCONSISTENT"),
                "postreview_codefilter": _metric_block(rows_meta, lambda r: post_code[r["pr_id"]] == "INCONSISTENT"),
            },
            "claim_status": {
                "preregistered_none": sum_claims("preregistered_none"),
                "preregistered_codefilter": sum_claims("preregistered_codefilter"),
                "postreview_none": sum_claims("postreview_none"),
                "postreview_codefilter": sum_claims("postreview_codefilter"),
            },
        },
        "flip_audit": {"none": flips_none, "codefilter": flips_code},
        "rows": row_out,
    }


def _make_model(reader_mode: str, cache_path: str, base_url: str, model_name: str, model_cls):
    if reader_mode == "llm":
        from claimlayer.semantic.adapters import OpenAICompatibleModel

        key = os.environ.get("NVIDIA_API_KEY") or sys.exit("Missing NVIDIA_API_KEY for --reader llm")
        base = OpenAICompatibleModel(
            base_url=base_url,
            model=model_name,
            api_key=key,
            timeout=180,
            retries=3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return model_cls(base, cache_path)
    if not os.path.exists(cache_path):
        sys.exit(f"Missing cache file for offline replay: {cache_path}")
    return model_cls(None, cache_path)


def run_all(args, out_dir: str) -> dict[str, str]:
    assert (
        eval_v2vocab.CODE_CHANGE_VOCAB_V2.fingerprint == eval_v2vocab.EXPECTED_VOCAB_FINGERPRINT
    ), "Frozen vocabulary fingerprint mismatch!"
    gate_commit = detect_gate_commit()
    os.makedirs(out_dir, exist_ok=True)

    c_v1 = args.cache_v1 or os.path.join(args.cache_dir, "prmci_llm_cache.jsonl")
    c_p1 = args.cache_v2_p1 or os.path.join(args.cache_dir, "prmci_llm_v2vocab_phase1_cache.jsonl")
    c_p2 = args.cache_v2_p2 or os.path.join(args.cache_dir, "prmci_llm_v2vocab_phase2_cache.jsonl")

    p1_rules = _evaluate_rules_suite(args.prmci_dir, AGENTS_PHASE1, gate_commit=gate_commit)
    p2_rules = _evaluate_rules_suite(args.prmci_dir, AGENTS_PHASE2, gate_commit=gate_commit)

    m_v1 = _make_model(args.reader, c_v1, args.base_url, args.model, llm_gate.CachedModel)
    p1_llm_v1 = _evaluate_phase1_llm_v1(args.prmci_dir, model=m_v1, gate_commit=gate_commit)

    m_v2_p1 = _make_model(args.reader, c_p1, args.base_url, args.model, eval_v2vocab.CachedModelV2)
    p1_llm_v2 = eval_v2vocab.run_evaluation(
        phase="phase1",
        prmci_dir=args.prmci_dir,
        model=m_v2_p1,
        workers=args.workers,
        legacy_section=True,
        gate_commit=gate_commit,
    )

    m_v2_p2 = _make_model(args.reader, c_p2, args.base_url, args.model, eval_v2vocab.CachedModelV2)
    p2_llm_v2 = eval_v2vocab.run_evaluation(
        phase="phase2",
        prmci_dir=args.prmci_dir,
        model=m_v2_p2,
        workers=args.workers,
        legacy_section=False,
        gate_commit=gate_commit,
    )

    summary = {
        "gate_commit": gate_commit,
        "dataset": "PR-MCI ac6bbf4 (Gong et al., MSR '26)",
        "vocabulary_v2_fingerprint": eval_v2vocab.EXPECTED_VOCAB_FINGERPRINT,
        "phase2_heldout": {
            "n": p2_rules["summary"]["n"],
            "ok_llm_v2": p2_llm_v2["summary"]["ok"],
            "errors_llm_v2": p2_llm_v2["summary"]["errors"],
            "escalated_llm_v2": p2_llm_v2["summary"]["escalated"],
            "truncated_llm_v2": p2_llm_v2["summary"]["truncated"],
            "labels": p2_rules["summary"]["labels"],
            "rules_and_baselines": p2_rules["summary"]["metrics"],
            "llm_v2vocab": {
                "gate_v4_vocab2": p2_llm_v2["summary"]["gate_v4_vocab2"],
                "paper_heuristic": p2_llm_v2["summary"]["paper_heuristic"],
                "claim_status": p2_llm_v2["summary"]["claim_status"],
                "pred_status": p2_llm_v2["summary"]["pred_status"],
                "draft_rejected": p2_llm_v2["summary"]["draft_rejected"],
            },
            "by_agent": {
                ag: {
                    "n": p2_rules["by_agent"][ag]["n"],
                    "labels": p2_rules["by_agent"][ag]["labels"],
                    "rules_and_baselines": p2_rules["by_agent"][ag]["metrics"],
                    "llm_v2vocab": p2_llm_v2["by_agent"][ag]["gate_v4_vocab2"],
                }
                for ag in AGENTS_PHASE2
            },
            "flip_audit_rules": p2_rules["flip_audit"],
        },
        "phase1_development": {
            "n": p1_rules["summary"]["n"],
            "labels": p1_rules["summary"]["labels"],
            "rules_and_baselines": p1_rules["summary"]["metrics"],
            "claim_stats_rules": p1_rules["summary"]["claim_stats"],
            "llm_v1": p1_llm_v1["summary"],
            "llm_v2vocab": p1_llm_v2["summary"],
            "flip_audit_rules": p1_rules["flip_audit"],
            "flip_audit_llm_v1": p1_llm_v1["flip_audit"],
        },
    }

    outputs = {
        "results_phase1_rules.json": p1_rules,
        "results_phase1_llm_v1.json": p1_llm_v1,
        "results_phase1_llm_v2vocab.json": p1_llm_v2,
        "results_phase2_rules.json": p2_rules,
        "results_phase2_llm_v2vocab.json": p2_llm_v2,
        "results_summary.json": summary,
    }
    hashes = {}
    for fname, payload in outputs.items():
        hashes[fname] = _write_json(os.path.join(out_dir, fname), payload)
    return hashes


def main():
    ap = argparse.ArgumentParser(description="Reproduce all PR-MCI case-study tables cleanly.")
    ap.add_argument("--prmci-dir", default=os.environ.get("PRMCI_DIR", os.path.join(_THIS_DIR, "data", "PR-MCI")))
    ap.add_argument("--reader", choices=["replay", "llm"], default="replay")
    ap.add_argument("--cache-dir", default=_THIS_DIR)
    ap.add_argument("--cache-v1", default="")
    ap.add_argument("--cache-v2-p1", default="")
    ap.add_argument("--cache-v2-p2", default="")
    ap.add_argument("--out-dir", default=_THIS_DIR)
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--verify-replay", action="store_true", help="Run offline replay twice and assert bit-identical JSONs.")
    args = ap.parse_args()

    if args.verify_replay:
        if args.reader != "replay":
            sys.exit("--verify-replay requires --reader replay")
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            h1 = run_all(args, d1)
            h2 = run_all(args, d2)
            assert h1 == h2, f"Replay mismatch: {h1} != {h2}"
            for fname in h1:
                b1 = open(os.path.join(d1, fname), "rb").read()
                b2 = open(os.path.join(d2, fname), "rb").read()
                assert b1 == b2, f"Byte mismatch in {fname}"
            print("REPLAY VERIFIED BIT-IDENTICAL across all 6 files (receipts included):")
            for fname, sha in sorted(h1.items()):
                print(f"  {sha}  {fname}")
            return

    hashes = run_all(args, args.out_dir)
    print(f"Reproduced all 6 files in {args.out_dir} (vocab={eval_v2vocab.EXPECTED_VOCAB_FINGERPRINT[:12]}):")
    for fname, sha in sorted(hashes.items()):
        print(f"  {sha}  {fname}")


if __name__ == "__main__":
    main()
