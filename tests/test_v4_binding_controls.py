"""Stage-1 controls: conclusion binding (subject/attribute/revision), restatement evidence,
reproducible counts, dependency-only blocking. Positive controls sit next to regressions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from claimlayer.v4 import verify_plan
from claimlayer.v4.reconstruction import is_claim_restatement

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "case_studies/pr_mci_claude_derivation"


def _identity_plan(target: dict[str, Any], premise: dict[str, Any]) -> dict[str, Any]:
    return {"claim_target": target, "premises": [premise],
            "steps": [{"step_id": "s1", "operation": "identity", "premises": [premise["claim_id"]], "conclusion_id": "c1", "parameters": {}}],
            "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": []}


# ---- Positive control 1: a valid refutation on the SAME attribute and subject -----------------
def test_positive_control_refutation_same_attribute_is_grade_A() -> None:
    src = {"notas": "La versión publicada del paquete es la 0.0.5 según las notas."}
    plan = _identity_plan(
        {"subject": "paquete", "predicate": "versión publicada", "value": "0.0.4", "unit": None, "polarity": 1},
        {"claim_id": "p1", "document_id": "notas", "quote": "La versión publicada del paquete es la 0.0.5",
         "entity": "paquete", "event": "versión publicada", "predicate": "versión publicada", "value": "0.0.5",
         "unit": None, "direction": "neutral", "polarity": 1})
    r = verify_plan(src, "La versión publicada del paquete es la 0.0.4.", plan)
    assert r.epistemic_verdict == "refuted", (r.outcome_category, r.defects)
    assert r.reliability["grade"] == "A"
    assert r.reliability["binding"]["attribute_binding"] == "matched"
    assert r.reliability["binding"]["subject_binding"] == "lexical"
    assert r.operational_routing.status == "RETRY"


# ---- Regression: refutation by a DIFFERENT attribute (C15 pattern) -----------------------------
def test_regression_attribute_mismatch_cannot_refute() -> None:
    src = {"d": "Changed files:\n- docs/release-notes/v0.0.4.md (added, +55/-0)"}
    plan = _identity_plan(
        {"subject": "docs", "predicate": "add release notes for", "value": "v0.0.4", "unit": None, "polarity": 1},
        {"claim_id": "p1", "document_id": "d", "quote": "- docs/release-notes/v0.0.4.md (added, +55/-0)",
         "entity": "docs/release-notes/v0.0.4.md", "event": "added", "predicate": "file status", "value": "added",
         "unit": None, "direction": "neutral", "polarity": 1})
    r = verify_plan(src, "docs: add release notes for v0.0.4", plan)
    assert r.epistemic_verdict == "unresolved"
    assert r.outcome_category == "conclusion_binding_defect"
    assert any(d["code"] == "CONCLUSION_ATTRIBUTE_MISMATCH" for d in r.defects)


DIFF_SRC = {"file_01": ("File: src/calc.py (status: added)\nUnified diff from base to head:\n"
                        "@@ -0,0 +1,3 @@\n+def double(x):\n+    return 2 * x\n+\n")}
HUNK = "@@ -0,0 +1,3 @@\n+def double(x):\n+    return 2 * x\n+"


# ---- Positive control 2: a quantitative claim proved by a reproducible count over the diff ----
def test_positive_control_count_reproduced_from_diff() -> None:
    plan = {
        "claim_target": {"subject": "src/calc.py", "predicate": "líneas añadidas", "value": 3, "unit": "count", "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "file_01", "quote": HUNK, "entity": "src/calc.py",
                      "event": "double", "predicate": "bloque añadido", "value": None, "unit": None,
                      "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+"}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }
    r = verify_plan(DIFF_SRC, "El archivo src/calc.py añade 3 líneas", plan)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.computed_conclusion["value"] == 3
    assert r.reliability["grade"] == "A"
    assert r.operational_routing.status == "PASS"
    r_bad = verify_plan(DIFF_SRC, "El archivo src/calc.py añade 5 líneas",
                        {**plan, "claim_target": {**plan["claim_target"], "value": 5}})
    assert r_bad.epistemic_verdict == "refuted" and r_bad.computed_conclusion["value"] == 3


# ---- Regression: quoting a number does not prove a count ---------------------------------------
def test_regression_quoted_count_requires_reproducible_count() -> None:
    src = {"t": "SELECT plan(8);\nSELECT ok(true, 'a');"}
    plan = _identity_plan(
        {"subject": "test file", "predicate": "test cases", "value": 8, "unit": "count", "polarity": 1},
        {"claim_id": "p1", "document_id": "t", "quote": "SELECT plan(8);", "entity": "test file",
         "event": "plan", "predicate": "test cases", "value": 8, "unit": "count", "direction": "neutral", "polarity": 1})
    r = verify_plan(src, "The test file has 8 test cases", plan)
    assert r.operational_routing.status != "PASS"
    assert r.epistemic_verdict != "derived"


# ---- Restatement (C14 pattern) and its negative control ----------------------------------------
def test_restatement_detection_is_ordered_and_value_aware() -> None:
    claim = "mcp/: Server registry & orchestration (207 lines total)"
    assert is_claim_restatement("mcp/          # Server registry & orchestration (207 lines total)", claim)
    assert not is_claim_restatement("mcp/ # Server registry & orchestration (209 lines total)", claim)  # different figure
    code = ('+    // Check if we\'re running in a TTY environment\n+    if !term.is_term() {\n+        return Err(err_msg(\n'
            '+            "Interactive review requires a terminal. Use `cargo insta accept` or `cargo insta reject` \\\n')
    assert not is_claim_restatement(code, "Add TTY environment check to `cargo insta review`")


# ---- Dependency-only blocking (NT3 pattern) ----------------------------------------------------
def test_unused_suggested_rule_does_not_override_missing_premise() -> None:
    src = {"doc_budget": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR."}
    plan = {
        "claim_target": {"subject": "Proyecto Norte", "predicate": "margen restante", "value": 350, "unit": "EUR", "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "doc_budget", "quote": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
                      "entity": "Proyecto Norte", "event": "presupuesto aprobado", "predicate": "importe", "value": 1250,
                      "unit": "EUR", "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s_unused", "operation": "python_eval", "premises": ["p1"], "conclusion_id": "c_x"}],
        "derived_conclusion_id": None,
        "missing_premises": [{"role": "gasto_ejecutado", "description": "Falta el gasto ejecutado."}],
        "assumptions": [],
        "suggested_rules": [{"rule_id": "margen_restante_calculo", "description": "margen = presupuesto - gasto"}],
    }
    r = verify_plan(src, "El margen restante del Proyecto Norte es de 350 EUR.", plan)
    assert r.epistemic_verdict == "insufficient_evidence"
    assert r.operational_routing.status == "ESCALATE"
    assert any("UNVERIFIED_SUGGESTED_RULE:margen_restante_calculo" in l for l in r.limitations)


# ---- Replays of the recorded live proposals (not live runs) ------------------------------------
def _replay(fn: str, cid: str):
    inputs = {c["case_id"]: c for c in json.loads((STUDY / "frozen/inputs_manifest.json").read_text())["cases"]}
    live = {r["case_id"]: r for r in json.loads((STUDY / "results" / fn).read_text())["cases"]}
    c = inputs[cid]
    return verify_plan(c["sources"], c["claim"], copy.deepcopy(live[cid]["response"]["provenance"]["raw_proposal"]))


def test_replay_second_measurement_c14_c15() -> None:
    r14 = _replay("live_results_v3.json", "C14")
    assert any(d["code"] == "CLAIM_RESTATEMENT_AS_PREMISE" for d in r14.defects)
    assert r14.operational_routing.status != "PASS"
    r15 = _replay("live_results_v3.json", "C15")
    assert r15.epistemic_verdict == "unresolved" and r15.outcome_category == "conclusion_binding_defect"


def test_replay_temporal_v3_nt3_is_insufficient_evidence() -> None:
    b = json.loads((ROOT / "benchmarks/v4_derivation_planner_live_results_v3.json").read_text())
    row = next(c for c in b["core_cases"] if c["case_id"].startswith("NT3"))
    r = verify_plan(row["sources"], row["claim"], copy.deepcopy(row["raw_proposal"]))
    assert r.epistemic_verdict == "insufficient_evidence" and r.operational_routing.status == "ESCALATE"
