"""Traceable reconstruction: planner-proposed operations, premise fidelity and reliability grades."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from claimlayer.v4 import (
    DERIVATION_PLANNER_PROMPT,
    DERIVATION_PLANNER_PROMPT_V2,
    ExpressionError,
    ReliabilityPolicy,
    evaluate_expression,
    premise_fidelity,
    validate_expression,
    verify_plan,
)
from claimlayer.v4.planner import ClaimDerivationPlanner

ROOT = Path(__file__).resolve().parents[1]
DOCS = {"doc_1": "Ana obtuvo 7 puntos en la prueba.", "doc_2": "Luis obtuvo 9 puntos en la prueba."}
# NOTE: numbers immediately followed by a sentence-final period ("es 9.") are not extracted by the
# current clause number extractor; recorded as a finding, not fixed in this iteration.


def _scores_plan(claimed: int, *, op: dict[str, Any] | None, step_op: str, proposed_value: Any = None) -> dict[str, Any]:
    plan = {
        "claim_target": {"subject": "puntuación", "predicate": "máxima", "value": claimed, "unit": None, "polarity": 1},
        "premises": [
            {"claim_id": "p1", "document_id": "doc_1", "quote": "Ana obtuvo 7 puntos en la prueba", "entity": "Ana",
             "event": "obtuvo", "predicate": "puntuacion", "value": 7, "unit": None, "direction": "neutral", "polarity": 1},
            {"claim_id": "p2", "document_id": "doc_2", "quote": "Luis obtuvo 9 puntos en la prueba", "entity": "Luis",
             "event": "obtuvo", "predicate": "puntuacion", "value": 9, "unit": None, "direction": "neutral", "polarity": 1},
        ],
        "steps": [{"step_id": "s1", "operation": step_op, "premises": ["p1", "p2"], "conclusion_id": "c1",
                   "parameters": {}, "assumptions": [], "proposed_value": proposed_value, "proposed_unit": None}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }
    if op is not None:
        plan["proposed_operations"] = [op]
    return plan


MAX_OP = {"name": "maximo", "description": "Mayor de dos puntuaciones", "inputs": ["a", "b"],
          "expression": {"op": "max", "args": [{"input": "a"}, {"input": "b"}]}, "output_type": "number", "output_unit": None}


def test_expression_language_is_closed_and_exact() -> None:
    e = {"op": "max", "args": [{"const": 1}, {"op": "sub", "args": [{"input": "n"}, {"input": "k"}]}]}
    validate_expression(e, ["n", "k"])
    assert evaluate_expression(e, {"n": 5, "k": 5}) == 1
    assert evaluate_expression(e, {"n": 9, "k": 2}) == 7
    assert evaluate_expression({"op": "add", "args": [{"const": 0.1}, {"const": 0.2}]}, {}) == pytest.approx(0.3)
    assert evaluate_expression({"op": "eq", "args": [{"const": "Create  Account"}, {"const": "create account"}]}, {}) is True
    for bad in ({"op": "__import__", "args": []}, {"input": "x"}, {"op": "add", "args": [{"const": 1}]},
                {"attr": "x"}, {"op": "eval", "args": [{"const": "1"}]}, {"const": [1, 2]}):
        with pytest.raises(ExpressionError):
            validate_expression(bad, ["n"])
    deep: dict[str, Any] = {"input": "n"}
    for _ in range(40):
        deep = {"op": "neg", "args": [deep]}
    with pytest.raises(ExpressionError):
        validate_expression(deep, ["n"])
    with pytest.raises(ExpressionError):
        evaluate_expression({"op": "div", "args": [{"const": 1}, {"const": 0}]}, {})


def test_premise_fidelity_classification() -> None:
    assert premise_fidelity(7) == "literal"
    assert premise_fidelity("miércoles") == "literal"
    assert premise_fidelity(True) == "interpreted"
    assert premise_fidelity(None) == "interpreted"
    assert premise_fidelity({"x": 1}) == "interpreted"


DIFF_OP = {"name": "diferencia_absoluta", "description": "Diferencia absoluta entre dos puntuaciones", "inputs": ["a", "b"],
           "expression": {"op": "abs", "args": [{"op": "sub", "args": [{"input": "a"}, {"input": "b"}]}]},
           "output_type": "number", "output_unit": None}
DIFF_CLAIM = "La diferencia entre las puntuaciones de Ana y Luis es de {} puntos"


def test_composed_operation_is_executed_reverified_and_graded_C() -> None:
    r = verify_plan(DOCS, DIFF_CLAIM.format(2), _scores_plan(2, op=DIFF_OP, step_op="diferencia_absoluta"))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.computed_conclusion["value"] == 2
    assert r.reliability["grade"] == "C"
    assert r.operational_routing.status == "ESCALATE"  # default policy: PASS requires >= B
    assert any(l["kind"] == "step" and l["operation"] == "diferencia_absoluta" for l in r.reliability["weakest_links"])
    assert r.reliability["proposed_operations"][0]["expression"] == DIFF_OP["expression"]
    assert r.verified_steps[0]["proposed_by_planner"] is True and r.verified_steps[0]["grade"] == "C"
    r2 = verify_plan(DOCS, DIFF_CLAIM.format(2), _scores_plan(2, op=DIFF_OP, step_op="diferencia_absoluta"),
                     reliability_policy=ReliabilityPolicy(pass_min_grade="C"))
    assert r2.operational_routing.status == "PASS"
    # Deltoryx computes the value itself: the model cannot impose its own result.
    r3 = verify_plan(DOCS, DIFF_CLAIM.format(3), _scores_plan(3, op=DIFF_OP, step_op="diferencia_absoluta", proposed_value=3))
    assert r3.epistemic_verdict == "refuted" and r3.computed_conclusion["value"] == 2 and r3.reliability["grade"] == "C"
    assert r3.verified_steps[0]["model_disagreed"] is True


def test_proposed_operations_do_not_bypass_mandatory_core_checks() -> None:
    # A model-composed "maximum" is an aggregation: the core still demands population completeness.
    r = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=MAX_OP, step_op="maximo"))
    assert r.epistemic_verdict != "derived"
    assert "UNVERIFIED_POPULATION_COMPLETENESS" in r.operational_routing.reasons
    assert r.operational_routing.status != "PASS"


def test_model_asserted_operation_is_traced_as_grade_D() -> None:
    op = {**MAX_OP, "name": "mayor_segun_modelo", "expression": None}
    r = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=op, step_op="mayor_segun_modelo", proposed_value=9))
    assert r.epistemic_verdict == "derived"
    assert r.reliability["grade"] == "D"
    assert r.operational_routing.status == "ESCALATE"
    assert "model_asserted_operation:mayor_segun_modelo" in r.assumptions
    r_pass = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=op, step_op="mayor_segun_modelo", proposed_value=9),
                         reliability_policy=ReliabilityPolicy(pass_min_grade="D"))
    assert r_pass.operational_routing.status == "PASS"
    r_none = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=op, step_op="mayor_segun_modelo"))
    assert r_none.epistemic_verdict == "unresolved"
    assert r_none.operational_routing.status != "PASS"


def test_invalid_or_shadowing_proposed_operations_are_rejected() -> None:
    shadow = {**MAX_OP, "name": "add"}
    r = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=shadow, step_op="add"))
    assert r.outcome_category == "malformed_proposal"
    assert any(d["code"] == "PROPOSED_OPERATION_SHADOWS_EXISTING" for d in r.defects)
    bad = {**MAX_OP, "expression": {"op": "exec", "args": [{"const": "import os"}]}}
    r2 = verify_plan(DOCS, "La mayor de las dos puntuaciones es de 9 puntos", _scores_plan(9, op=bad, step_op="maximo"))
    assert r2.outcome_category == "malformed_proposal"
    assert any(d["code"] == "INVALID_PROPOSED_EXPRESSION" for d in r2.defects)


def _interpreted_plan(target_value: Any) -> dict[str, Any]:
    return {
        "claim_target": {"subject": "revisión interactiva", "predicate": "comprobación de terminal", "value": target_value, "unit": None, "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "d", "quote": "if !term.is_term() {", "entity": "term",
                      "event": "is_term", "predicate": "comprobacion_terminal", "value": True, "unit": None,
                      "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1", "parameters": {}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def test_identity_over_interpreted_premise_is_grade_E_and_never_PASS_by_default() -> None:
    src = {"d": "fn query() {\n    if !term.is_term() {\n        return Err(e);\n    }\n}"}
    r = verify_plan(src, "La revisión interactiva tiene comprobación de terminal.", _interpreted_plan(True))
    if r.epistemic_verdict == "derived":
        assert r.reliability["grade"] == "E"
        assert r.operational_routing.status == "ESCALATE"
        assert r.reliability["weakest_links"][0]["fidelity"] == "interpreted"
    else:
        assert r.operational_routing.status != "PASS"


def test_incomparable_conclusion_is_unresolved_not_refuted() -> None:
    src = {"d": "Changed files:\n- docs/release-notes/v0.0.4.md (added, +55/-0)"}
    plan = {
        "claim_target": {"subject": "docs", "predicate": "add release notes for", "value": "v0.0.4", "unit": None, "polarity": 1},
        "premises": [{"claim_id": "p1", "document_id": "d", "quote": "Changed files:\n- docs/release-notes/v0.0.4.md (added, +55/-0)",
                      "entity": "docs/release-notes/v0.0.4.md", "event": "changed files", "predicate": "add release notes for", "value": True,
                      "unit": None, "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1", "parameters": {}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }
    r = verify_plan(src, "docs: add release notes for v0.0.4", plan)
    assert r.epistemic_verdict == "unresolved"
    assert r.outcome_category == "incomparable_conclusion"
    assert r.operational_routing.status == "RETRY"


def test_prmci_first_measurement_replay_gains_no_unsafe_pass() -> None:
    base = ROOT / "case_studies/pr_mci_claude_derivation"
    live = json.loads((base / "results/live_results.json").read_text(encoding="utf-8"))
    inputs = {c["case_id"]: c for c in json.loads((base / "frozen/inputs_manifest.json").read_text(encoding="utf-8"))["cases"]}
    expected = {c["case_id"]: c for c in json.loads((base / "frozen/expected_manifest.json").read_text(encoding="utf-8"))["cases"]}
    outcomes = {}
    for row in live["cases"]:
        raw = row["response"]["provenance"]["raw_proposal"]
        if raw is None:
            continue
        c = inputs[row["case_id"]]
        r = verify_plan(c["sources"], c["claim"], copy.deepcopy(raw))
        outcomes[row["case_id"]] = r
        if expected[row["case_id"]]["expected_verdict"] != "derived":
            assert r.operational_routing.status != "PASS", row["case_id"]
    assert outcomes["C03"].operational_routing.status != "PASS"
    assert outcomes["C15"].epistemic_verdict == "unresolved"
    assert outcomes["C15"].outcome_category in {"incomparable_conclusion", "conclusion_binding_defect"}
    assert not any(r.operational_routing.status == "PASS" for r in outcomes.values())


def test_v3_prompt_offers_expression_primitives_and_v2_is_preserved() -> None:
    assert "proposed_operations" in DERIVATION_PLANNER_PROMPT and "proposed_operations" not in DERIVATION_PLANNER_PROMPT_V2

    class _Capture:
        identity = {"provider": "capture"}
        payload: dict[str, Any] = {}

        def complete(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
            _Capture.payload = payload
            return {"claim_target": {}, "premises": [], "steps": []}

    from claimlayer.v4 import DEFAULT_OPERATION_CATALOG, DomainProfile
    from claimlayer.semantic.models import SourceDocument
    ClaimDerivationPlanner(model=_Capture()).plan(sources=[SourceDocument(document_id="d", text="x")], claim="c",
                                                   catalog=DEFAULT_OPERATION_CATALOG, domain_profile=DomainProfile(profile_id="t"))
    assert "max" in _Capture.payload["expression_primitives"]
    ClaimDerivationPlanner(model=_Capture(), prompt=DERIVATION_PLANNER_PROMPT_V2).plan(
        sources=[SourceDocument(document_id="d", text="x")], claim="c", catalog=DEFAULT_OPERATION_CATALOG,
        domain_profile=DomainProfile(profile_id="t"))
    assert "expression_primitives" not in _Capture.payload
