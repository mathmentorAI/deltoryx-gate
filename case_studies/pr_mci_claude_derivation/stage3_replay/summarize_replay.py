"""Summarise replay_verifier_<old>.json vs replay_verifier_<new>.json against the frozen expected manifests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = {"regression_pr": HERE.parent / "frozen/expected_manifest.json",
       "transfer_pr": HERE.parents[1] / "pr_mci_transfer_sample/frozen/expected_manifest.json"}


def key(r: dict) -> tuple:
    return (r.get("verdict"), r.get("routing"), r.get("category"))


def metrics(rows: list[tuple]) -> dict:
    nonref = [x for x in rows if x[1]["expected_verdict"] != "refuted"]
    nonsup = [x for x in rows if x[1]["expected_verdict"] in ("refuted", "insufficient_evidence")]
    resolv = [x for x in rows if x[1]["expected_verdict"] in ("derived", "refuted")]
    return {
        "false_refutations": f"{sum(x[2].get('verdict') == 'refuted' for x in nonref)}/{len(nonref)}",
        "false_acceptances": f"{sum(x[2].get('routing') == 'PASS' for x in nonsup)}/{len(nonsup)}",
        "verdict_in_acceptable_set": f"{sum(x[2].get('verdict') in [x[1]['expected_verdict']] + x[1]['acceptable_alternative_verdicts'] for x in rows)}/{len(rows)}",
        "derived_or_refuted_with_acceptable_verdict (candidate valid inferences; justification not reviewed)":
            f"{sum(x[2].get('verdict') in ('derived', 'refuted') and x[2].get('verdict') in [x[1]['expected_verdict']] + x[1]['acceptable_alternative_verdicts'] for x in resolv)}/{len(resolv)}",
        "abstentions_on_resolvable": f"{sum(x[2].get('verdict') in ('insufficient_evidence', 'unresolved', None) for x in resolv)}/{len(resolv)}",
    }


def main(old_p: str, new_p: str, out_p: str) -> None:
    o, n = json.loads(Path(old_p).read_text()), json.loads(Path(new_p).read_text())
    S: dict = {"old": o["label"], "new": n["label"]}
    for k in ("regression_pr", "transfer_pr"):
        E = {c["case_id"]: c for c in json.loads(EXP[k].read_text())["cases"]}
        old_rows = [(a["case_id"], E[a["case_id"]], a["replay"]) for a in o[k]]
        new_rows = [(b["case_id"], E[b["case_id"]], b["replay"]) for b in n[k]]
        S[k] = {
            "recorded_M4_reproduced_by_old": f"{sum(key(a['replay']) == key(a['recorded']) for a in o[k])}/{len(o[k])}"
                                             + f" (sin plan registrado: {[a['case_id'] for a in o[k] if a['replay'].get('verdict') is None]})",
            "old": metrics(old_rows), "new": metrics(new_rows),
            "changed": [{"case_id": a["case_id"], "expected": E[a["case_id"]]["expected_verdict"], "old": list(key(a["replay"])),
                         "new": list(key(b["replay"])), "new_computed": b["replay"].get("computed"), "new_defects": b["replay"].get("defects")}
                        for a, b in zip(o[k], n[k]) if key(a["replay"]) != key(b["replay"])],
        }
    S["temporal_m4_plans"] = {"old": f"{sum(r['replay']['verdict'] == r['expected'] for r in o['temporal_m4_plans'])}/14",
                              "new": f"{sum(r['replay']['verdict'] == r['expected'] for r in n['temporal_m4_plans'])}/14"}
    S["temporal_prepared_plans"] = [{"case_id": a["case_id"], "expected": a["expected"], "old": a["replay"]["verdict"],
                                     "new": b["replay"]["verdict"], "new_routing": b["replay"]["routing"], "computed": b["replay"]["computed"]}
                                    for a, b in zip(o["temporal_prepared_plans"], n["temporal_prepared_plans"])]
    S["manual_derivations"] = [{"case_id": a["case_id"], "contract": a["contract"],
                                "old": [a["replay"]["verdict"], a["replay"]["grade"], a["replay"]["computed"]],
                                "new": [b["replay"]["verdict"], b["replay"]["category"], b["replay"]["computed"]]}
                               for a, b in zip(o["manual_derivations"], n["manual_derivations"])]
    S["manual_stage3"] = [{"case_id": x["case_id"], "plan": x["plan"], "verdict": x["replay"]["verdict"],
                           "category": x["replay"]["category"], "computed": x["replay"]["computed"]} for x in n["manual_stage3"]]
    S["independent_reference"] = {k: f"{sum(x['match'] for x in v)}/{len(v)} archivos coinciden con GitHub (adiciones, eliminaciones); truncados: {sum(x['truncated'] for x in v)}"
                                  for k, v in n["independent_reference"].items()}
    Path(out_p).write_text(json.dumps(S, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(S, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
