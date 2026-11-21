"""Stage 2 diagnostic: hand-built derivations for two of the twelve inference cases, sent through
the SAME public contract (verify_plan) and verifier as the planner's proposals. Answers:
does the needed evidence reach the system? can the language express the operation? does the
verifier execute it correctly? Run: PYTHONPATH=claim-layer/src python3 .../manual_derivations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from claimlayer.v4 import verify_plan  # noqa: E402

INPUTS = {c["case_id"]: c for c in json.loads((HERE.parent / "frozen/inputs_manifest.json").read_text())["cases"]}


def doc(cid: str, did: str) -> str:
    return next(s["text"] for s in INPUTS[cid]["sources"] if s["document_id"] == did)


def manual_c07() -> dict:
    """'Added test file X with 8 test cases' -> reproducible count of pgTAP assertions in the added file."""
    block = doc("C07", "file_01")  # whole document: header with path + added hunk
    return {
        "claim_target": {"subject": "frontend/packages/db/supabase/tests/database/03-organization_members_rls.test.sql",
                         "predicate": "test cases", "value": 8, "unit": "count", "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "file_01", "quote": block,
                      "entity": "frontend/packages/db/supabase/tests/database/03-organization_members_rls.test.sql",
                      "event": "SELECT ok", "predicate": "added test file", "value": None, "unit": None,
                      "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+", "contains": "SELECT ok("}}],
        "derived_conclusion_id": "c1", "missing_premises": [],
        "assumptions": ["Cada caso de test pgTAP es una aserción `SELECT ok(` añadida en el archivo nuevo."],
        "suggested_rules": [],
    }


def manual_c13() -> dict:
    """'Implemented `porto create-merchant` command ...' -> the newly added CLI entry registers no such command."""
    block = doc("C13", "file_06")  # src/cli/index.ts, status added: the full file is in the diff
    return {
        "claim_target": {"subject": "porto", "predicate": "registers create-merchant command", "value": True, "unit": None, "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "file_06", "quote": block, "entity": "src/cli/index.ts",
                      "event": "command", "predicate": "CLI entry file", "value": None, "unit": None,
                      "direction": "neutral", "polarity": 1}],
        "steps": [
            {"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
             "parameters": {"line_prefix": "+", "contains": "create-merchant"}},
            {"step_id": "s2", "operation": "registered", "premises": ["c1"], "conclusion_id": "c2", "parameters": {}},
        ],
        "proposed_operations": [{"name": "registered", "description": "Un comando está registrado si su nombre aparece al menos una vez",
                                 "inputs": ["n"], "expression": {"op": "gt", "args": [{"input": "n"}, {"const": 0}]},
                                 "output_type": "boolean", "output_unit": None}],
        "derived_conclusion_id": "c2", "missing_premises": [],
        "assumptions": ["src/cli/index.ts es un archivo nuevo (status added): su contenido completo está en el diff.",
                        "Los comandos del CLI se registran con cli.command(...) en ese archivo de entrada."],
        "suggested_rules": [],
    }


def run(cid: str, plan: dict) -> dict:
    c = INPUTS[cid]
    r = verify_plan(c["sources"], c["claim"], plan)
    out = {"case_id": cid, "claim": c["claim"], "verdict": r.epistemic_verdict, "outcome": r.outcome_category,
           "routing": r.operational_routing.status, "reasons": list(r.operational_routing.reasons),
           "computed": (r.computed_conclusion or {}).get("value"), "reliability": r.reliability,
           "defects": [{"claim_id": d.get("claim_id"), "code": d.get("code"), "message": (d.get("message") or "")[:300]} for d in r.defects],
           "steps": [{k: s.get(k) for k in ("operation", "recomputed_value", "grade")} for s in r.verified_steps],
           "plan": plan}
    return out


if __name__ == "__main__":
    results = [run("C07", manual_c07()), run("C13", manual_c13())]
    for o in results:
        print(o["case_id"], o["verdict"], o["outcome"], o["routing"], "computed=", o["computed"],
              "grade=", (o["reliability"] or {}).get("grade"), [d["code"] for d in o["defects"]][:6], o["steps"])
        for d in o["defects"][:4]:
            print("    ", d["code"], d["message"][:220])
    tag = sys.argv[1] if len(sys.argv) > 1 else "contract_stage1"
    (HERE / f"manual_results_{tag}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str) + "\n")
