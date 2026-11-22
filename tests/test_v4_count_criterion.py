"""Point 1: a reproducible count proves how many lines match a criterion, not what the criterion
counts. Same rule for deriving and refuting. Synthetic sources only."""
from __future__ import annotations

from typing import Any

from claimlayer.v4 import verify_plan
from claimlayer.v4.count_definitions import CountDefinition

LINES = [
    "File: tests/test_parser.py (status: added)",                 # 1
    "Unified diff from base to head:",                            # 2
    "@@ -0,0 +1,10 @@",                                           # 3
    "+import parser",                                             # 4
    "+",                                                          # 5
    "+def test_empty():",                                         # 6
    "+    assert parser.parse('') == []",                         # 7
    "+def test_one():",                                           # 8
    "+    assert parser.parse('a') == ['a']",                     # 9
    "+def test_two():",                                           # 10
    "+    assert parser.parse('a b') == ['a', 'b']",              # 11
    "+",                                                          # 12
    "+HELPERS = ['x']",                                           # 13
]
SRC = {"d1": "\n".join(LINES)}
PYTEST_FUNCS = CountDefinition("pytest_test_functions", "1.0.0", frozenset({"test", "tests", "caso", "casos", "prueba", "pruebas"}),
                               contains="def test_", line_prefix="+",
                               description="Funciones de test pytest añadidas: líneas '+' que contienen 'def test_'.")
IMPORTS = CountDefinition("python_imports", "1.0.0", frozenset({"import", "imports", "importaciones"}), contains="import ",
                          line_prefix="+")


def _plan(value: int, params: dict[str, Any], predicate: str = "tests añadidos") -> dict[str, Any]:
    return {
        "claim_target": {"subject": "tests/test_parser.py", "predicate": predicate, "value": value, "unit": "count",
                         "polarity": 1, "revision": "added"},
        "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "d1", "file": "tests/test_parser.py",
                      "side": "added", "line_start": 4, "line_end": 13}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+", **params}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def _v(value: int, params: dict[str, Any], defs=(PYTEST_FUNCS, IMPORTS), predicate: str = "tests añadidos"):
    return verify_plan(SRC, f"El archivo tests/test_parser.py añade {value} tests", _plan(value, params, predicate),
                       count_definitions=defs)


def test_control_relevant_declared_criterion_derives() -> None:
    r = _v(3, {"definition": "pytest_test_functions"})
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.computed_conclusion["value"] == 3
    assert r.reliability["grade"] == "B"  # declared rule, not planner interpretation
    assert r.operational_routing.status == "PASS"


def test_control_same_criterion_wrong_value_refutes() -> None:
    r = _v(5, {"definition": "pytest_test_functions"})
    assert r.epistemic_verdict == "refuted", (r.outcome_category, r.defects)
    assert r.computed_conclusion["value"] == 3 and r.reliability["grade"] == "B"


def test_control_irrelevant_criterion_coincident_count_abstains() -> None:
    # 'assert' appears on exactly 3 added lines: the count coincides with the claim but measures
    # something else (and was chosen by the planner).
    r = _v(3, {"contains": "assert"})
    assert r.epistemic_verdict == "insufficient_evidence", (r.outcome_category, r.defects)
    assert r.outcome_category == "count_criterion_interpreted"
    assert r.computed_conclusion["value"] == 3
    assert "3 líneas" in r.explanation and "'assert'" in r.explanation and "Falta" in r.explanation
    assert r.operational_routing.status == "ESCALATE"
    # A declared definition of another attribute that coincides (1 import ≠ 3, use 1) also abstains.
    r2 = _v(1, {"definition": "python_imports"})
    assert r2.epistemic_verdict == "insufficient_evidence" and r2.outcome_category == "count_criterion_not_pertinent"


def test_planner_criterion_abstains_in_both_directions() -> None:
    # Even a pertinent-looking filter chosen by the planner neither derives (3) nor refutes (5).
    for value in (3, 5):
        r = _v(value, {"contains": "def test_"})
        assert r.epistemic_verdict == "insufficient_evidence", (value, r.outcome_category)
        assert r.computed_conclusion["value"] == 3
    # Without any definition registered, the declared path is unavailable.
    r = _v(3, {"definition": "pytest_test_functions"}, defs=())
    assert r.epistemic_verdict == "unresolved" and r.outcome_category == "unsupported_operation_or_rule"


def test_structural_line_count_is_its_own_definition() -> None:
    r = _v(10, {"scope": "file"}, predicate="líneas añadidas")
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.reliability["grade"] == "A"
    # but a line count does not measure tests
    r2 = _v(10, {"scope": "file"})
    assert r2.epistemic_verdict == "insufficient_evidence" and r2.outcome_category == "count_criterion_interpreted"


def test_definition_overrides_planner_filter() -> None:
    r = _v(3, {"definition": "pytest_test_functions", "contains": "assert"})
    assert r.epistemic_verdict == "derived"
