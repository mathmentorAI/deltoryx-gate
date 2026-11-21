"""Stage 2 diagnostic, contract after representation support: the same two derivations as
manual_derivations.py, but citing line ranges (no copied text) and binding entity to the diff
file header. Run: PYTHONPATH=claim-layer/src python3 .../manual_derivations_ranges.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from claimlayer.v4 import verify_plan  # noqa: E402
from manual_derivations import INPUTS, doc, run  # noqa: E402


def nlines(cid: str, did: str) -> int:
    return len(doc(cid, did).split("\n"))


def c07_ranges() -> dict:
    path = "frontend/packages/db/supabase/tests/database/03-organization_members_rls.test.sql"
    return {
        "claim_target": {"subject": path, "predicate": "test cases", "value": 8, "unit": "count", "polarity": 1, "revision": "added"},
        "premises": [{"claim_id": "p1", "document_id": "file_01", "line_start": 4, "line_end": nlines("C07", "file_01"),
                      "entity": path, "event": "SELECT ok", "predicate": "added test file", "value": None, "unit": None,
                      "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+", "contains": "SELECT ok("}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def c13_ranges() -> dict:
    return {
        "claim_target": {"subject": "porto", "predicate": "registers create-merchant command", "value": True, "unit": None,
                         "polarity": 1, "revision": "added"},
        "premises": [{"claim_id": "p1", "document_id": "file_06", "line_start": 4, "line_end": nlines("C13", "file_06"),
                      "entity": "src/cli/index.ts", "event": "command", "predicate": "CLI entry file", "value": None,
                      "unit": None, "direction": "neutral", "polarity": 1}],
        "steps": [
            {"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
             "parameters": {"line_prefix": "+", "contains": "create-merchant"}},
            {"step_id": "s2", "operation": "registered", "premises": ["c1"], "conclusion_id": "c2", "parameters": {}},
        ],
        "proposed_operations": [{"name": "registered", "description": "Registrado si el nombre aparece al menos una vez",
                                 "inputs": ["n"], "expression": {"op": "gt", "args": [{"input": "n"}, {"const": 0}]},
                                 "output_type": "boolean", "output_unit": None}],
        "derived_conclusion_id": "c2", "missing_premises": [],
        "assumptions": ["src/cli/index.ts es nuevo (status added): su contenido completo está en el diff."],
        "suggested_rules": [],
    }


if __name__ == "__main__":
    results = [run("C07", c07_ranges()), run("C13", c13_ranges())]
    for o in results:
        print(o["case_id"], o["verdict"], o["outcome"], o["routing"], "computed=", o["computed"], "grade=",
              (o["reliability"] or {}).get("grade"), [d["code"] for d in o["defects"]][:5],
              {k: (o["reliability"] or {}).get("binding", {}).get(k) for k in ("revision_mode", "path")})
    (HERE / "manual_results_contract_stage2.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str) + "\n")
