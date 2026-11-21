"""Step 4: score live_results.json against the frozen expected manifest + manual justification review."""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
E = {c["case_id"]: c for c in json.loads((HERE / "frozen/expected_manifest.json").read_text())["cases"]}
I = {c["case_id"]: c for c in json.loads((HERE / "frozen/inputs_manifest.json").read_text())["cases"]}
L = json.loads((HERE / "results/live_results.json").read_text())
R = json.loads((HERE / "results/justification_review.json").read_text())["cases"]
rows = {r["case_id"]: r for r in L["cases"]}
V = ["derived", "refuted", "insufficient_evidence", "unresolved"]


def mstat(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "median": round(statistics.median(xs), 1), "min": round(min(xs), 1), "max": round(max(xs), 1)}


out = {"per_case": [], "metrics": {}, "telemetry": {}, "old_evaluators": {}}
conf = {e: collections.Counter() for e in V[:3]}
for cid in sorted(E):
    e, r, rev = E[cid], rows[cid], R[cid]
    obs = r["verdict"]
    conf[e["expected_verdict"]][obs] += 1
    acceptable = obs == e["expected_verdict"] or obs in e["acceptable_alternative_verdicts"]
    tel = r["response"]["telemetry"]
    out["per_case"].append({
        "case_id": cid, "pr_id": e["pr_id"], "head_sha": I[cid]["head_sha"][:12], "claim": I[cid]["claim"],
        "situation": e["situation"], "expected": e["expected_verdict"], "expected_routing": e["expected_routing"],
        "observed": obs, "observed_routing": r["routing"], "outcome_category": r["outcome_category"],
        "verdict_acceptable": acceptable, "review": rev,
        "first_attempt": (tel["attempt_traces"][0].get("outcome_category") if tel["attempt_traces"] else None),
        "old_gate_v4_vocab2": {"gate": r.get("gate_v4_vocab2", {}).get("gate"), "claim_status": r.get("gate_v4_vocab2", {}).get("claim_status")},
    })
pc = {p["case_id"]: p for p in out["per_case"]}
non_supported = [c for c in E if E[c]["expected_verdict"] in ("refuted", "insufficient_evidence")]
not_refuted = [c for c in E if E[c]["expected_verdict"] != "refuted"]
resolvable = [c for c in E if E[c]["expected_verdict"] in ("derived", "refuted")]
abstained = [c for c in E if rows[c]["verdict"] in ("insufficient_evidence", "unresolved")]
cat = collections.Counter(R[c]["category"] for c in E)
infer = [c for c in E if E[c]["requires_inference"]]
out["metrics"] = {
    "confusion_expected_x_observed": {k: dict(v) for k, v in conf.items()},
    "verdict_acceptable": f"{sum(p['verdict_acceptable'] for p in out['per_case'])}/20",
    "correctly_verified (verdict acceptable + valid justification)": f"{cat['correct_verified']}/20",
    "label_ok_bad_justification": f"{cat['label_ok_bad_justification']}/20",
    "false_acceptances (PASS on refuted|insufficient reference)": f"{sum(rows[c]['routing']=='PASS' for c in non_supported)}/{len(non_supported)}",
    "incorrect_refutations (refuted on non-refuted reference)": f"{sum(rows[c]['verdict']=='refuted' for c in not_refuted)}/{len(not_refuted)}",
    "justified_abstentions (abstention with acceptable verdict + valid justification)": f"{sum(R[c]['category']=='correct_verified' for c in abstained)}/{len(abstained)}",
    "resolvable_cases_ending_in_abstention_or_unresolved": f"{sum(rows[c]['verdict'] in ('insufficient_evidence','unresolved') for c in resolvable)}/{len(resolvable)}",
    "operational_errors (truncated planner output, final)": f"{cat['operational_error']}/20",
    "verifier_abstentions (unresolved by defect/unsupported op, final)": f"{cat['verifier_abstention']}/20",
    "provider_errors": f"{sum(rows[c]['outcome_category']=='provider_error' for c in E)}/20",
    "routing_match": f"{sum(rows[c]['routing']==E[c]['expected_routing'] for c in E)}/20",
    "inference_cases_resolved_with_valid_derivation": f"{sum(R[c]['category']=='correct_verified' and rows[c]['verdict'] in ('derived','refuted') for c in infer)}/{len(infer)}",
    "first_attempt_malformed": f"{sum(1 for c in E for a in rows[c]['response']['telemetry']['attempt_traces'] if a.get('outcome_category')=='malformed_proposal')}/{sum(len(rows[c]['response']['telemetry']['attempt_traces']) for c in E)} attempts",
}
adm = [(c, p) for c in E for p in rows[c]["response"]["justification"]["premises"]]
steps = [(c, s) for c in E for s in rows[c]["response"]["justification"]["verified_steps"]]
out["metrics"]["admitted_premises_final"] = len(adm)
out["metrics"]["admitted_premises_with_literal_quote_check"] = sum("verbatim_span_bound" in (p.get("checks_passed") or []) for _, p in adm)
out["metrics"]["admitted_premises_value_set_by_nemotron_interpretation (non-numeric)"] = sum(not isinstance(p.get("value"), (int, float)) or isinstance(p.get("value"), bool) for _, p in adm)
out["metrics"]["executed_operations"] = dict(collections.Counter(s["operation"] for _, s in steps))
T = [rows[c]["response"]["telemetry"] for c in E]
out["telemetry"] = {
    "calls_total": sum(t["calls"] for t in T), "retries_total": sum(t["retries"] for t in T),
    "total_tokens": mstat([t["total_tokens"] for t in T]),
    "planning_latency_ms": mstat([t["planning_latency_ms"] for t in T]),
    "verifier_latency_ms": mstat([t["verifier_latency_ms"] for t in T]),
    "api_total_latency_ms": mstat([t["total_latency_ms"] for t in T]),
    "client_end_to_end_ms": mstat([rows[c]["client_wall_ms"] for c in E]),
    "tokens_sum": sum(t["total_tokens"] for t in T),
}
g = collections.Counter()
for c in E:
    st = rows[c].get("gate_v4_vocab2", {}).get("claim_status") or {}
    g["supported" if st.get("supported") else ("unverifiable" if st.get("unverifiable") else "other")] += 1
out["old_evaluators"]["gate_v4_vocab2_claim_status"] = dict(g)
out["old_evaluators"]["gate_v4_vocab2_pr_gate"] = dict(collections.Counter(rows[c].get("gate_v4_vocab2", {}).get("gate") for c in E))
out["old_evaluators"]["gate_v4_vocab2_CONSISTENT_on_non_supported_reference"] = f"{sum(rows[c].get('gate_v4_vocab2',{}).get('gate')=='CONSISTENT' for c in non_supported)}/{len(non_supported)}"
out["old_evaluators"]["rules_v2_P_placeholder_fired"] = f"{sum('P_placeholder' in rows[c]['rules_v2_postreview_code_config']['signals'] for c in E)}/20"
(HERE / "results/scored.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: out[k] for k in ("metrics", "telemetry", "old_evaluators")}, indent=1, ensure_ascii=False))
