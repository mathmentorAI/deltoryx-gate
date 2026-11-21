"""Stage 2: line-range citations and revision/file/line binding for unified-diff evidence.
Synthetic diffs only (no measured or transfer-sample case data)."""
from __future__ import annotations

from typing import Any

from claimlayer.v4 import DERIVATION_PLANNER_PROMPT, DERIVATION_PLANNER_PROMPT_V3, verify_plan
from claimlayer.v4.diff_evidence import numbered_text, parse_unified_diff, resolve_line_range

DIFF = "\n".join([
    "File: lib/shapes.py (status: modified)",                  # 1
    "Unified diff from base to head:",                         # 2
    "@@ -10,6 +10,8 @@ class Shape:",                          # 3
    "     def area(self):",                                    # 4  context
    "         return 0",                                       # 5  context
    "-    def legacy_perimeter(self):",                        # 6  removed
    "-        return -1",                                      # 7  removed
    "+    def perimeter(self):",                               # 8  added
    "+        return sum(self.sides)",                         # 9  added
    "+    def sides_count(self):",                             # 10 added
    "+        return len(self.sides)",                         # 11 added
    "     def name(self):",                                    # 12 context
])
SRC = {"d1": DIFF}


def _plan(target: dict[str, Any], premise: dict[str, Any], step: dict[str, Any] | None = None) -> dict[str, Any]:
    step = step or {"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1", "parameters": {}}
    return {"claim_target": target, "premises": [premise], "steps": [step], "derived_conclusion_id": "c1",
            "missing_premises": [], "assumptions": [], "suggested_rules": []}


def test_diff_map_sides_files_and_line_numbers() -> None:
    dm = parse_unified_diff(DIFF)
    assert dm is not None
    assert [l.kind for l in dm.lines[3:12]] == ["context", "context", "removed", "removed", "added", "added", "added", "added", "context"]
    assert dm.lines[7].file == "lib/shapes.py" and dm.lines[7].head_line == 12 and dm.lines[5].base_line == 12
    assert resolve_line_range(DIFF, 8, 9) == "+    def perimeter(self):\n+        return sum(self.sides)"
    assert resolve_line_range(DIFF, 0, 2) is None and resolve_line_range(DIFF, 9, 99) is None
    assert numbered_text("a\nb").splitlines() == ["1| a", "2| b"]


def test_positive_control_count_over_line_range_with_file_bound_entity() -> None:
    plan = _plan({"subject": "lib/shapes.py", "predicate": "métodos añadidos", "value": 2, "unit": "count", "polarity": 1, "revision": "added"},
                 {"claim_id": "p1", "document_id": "d1", "line_start": 8, "line_end": 11, "entity": "lib/shapes.py",
                  "event": "def", "predicate": "bloque añadido", "value": None, "unit": None, "direction": "neutral", "polarity": 1},
                 {"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                  "parameters": {"line_prefix": "+", "contains": "def "}})
    r = verify_plan(SRC, "El cambio añade 2 métodos a lib/shapes.py", plan)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.computed_conclusion["value"] == 2 and r.reliability["grade"] == "A"
    assert r.operational_routing.status == "PASS"
    prem = r.premises[0]
    assert prem["quote_source"] == "line_range"
    assert prem["diff_binding"]["files"] == ["lib/shapes.py"] and prem["diff_binding"]["sides"] == ["+"]
    assert prem["diff_binding"]["head_lines"] == [12, 15]
    assert r.reliability["binding"]["revision_mode"] == "added"


def test_regression_base_only_line_cannot_support_head_or_change_claim() -> None:
    plan = _plan({"subject": "lib/shapes.py", "predicate": "define legacy_perimeter", "value": "legacy_perimeter", "unit": None, "polarity": 1},
                 {"claim_id": "p1", "document_id": "d1", "line_start": 6, "line_end": 6, "entity": "lib/shapes.py",
                  "event": "def", "predicate": "define legacy_perimeter", "value": "legacy_perimeter", "unit": None,
                  "direction": "neutral", "polarity": 1})
    r = verify_plan(SRC, "lib/shapes.py define legacy_perimeter", plan)
    assert r.operational_routing.status != "PASS"
    assert any(d["code"] == "BASE_ONLY_EVIDENCE_FOR_HEAD_CLAIM" for d in r.defects), r.defects
    # Positive control: the same base line does support a claim about what was removed.
    r2 = verify_plan(SRC, "lib/shapes.py define legacy_perimeter",
                     {**plan, "claim_target": {**plan["claim_target"], "revision": "removed"}})
    assert not any(d["code"] == "BASE_ONLY_EVIDENCE_FOR_HEAD_CLAIM" for d in r2.defects)
    assert r2.epistemic_verdict == "derived", (r2.outcome_category, r2.defects)


def test_regression_unchanged_context_does_not_prove_a_change() -> None:
    plan = _plan({"subject": "lib/shapes.py", "predicate": "define area", "value": "area", "unit": None, "polarity": 1},
                 {"claim_id": "p1", "document_id": "d1", "line_start": 4, "line_end": 4, "entity": "lib/shapes.py",
                  "event": "def", "predicate": "define area", "value": "area", "unit": None, "direction": "neutral", "polarity": 1})
    r = verify_plan(SRC, "lib/shapes.py define area", plan)  # default revision for diff evidence: 'change'
    assert any(d["code"] == "NO_CHANGED_LINE_EVIDENCE" for d in r.defects)
    assert r.operational_routing.status != "PASS"
    # Positive control: the same context line supports a claim about the head state.
    r2 = verify_plan(SRC, "lib/shapes.py define area", {**plan, "claim_target": {**plan["claim_target"], "revision": "head"}})
    assert r2.epistemic_verdict == "derived", (r2.outcome_category, r2.defects)


def test_invalid_line_range_is_a_premise_defect() -> None:
    plan = _plan({"subject": "Shape", "predicate": "x", "value": "x", "unit": None, "polarity": 1},
                 {"claim_id": "p1", "document_id": "d1", "line_start": 40, "line_end": 50, "entity": "Shape",
                  "event": "x", "predicate": "x", "value": "x", "unit": None, "direction": "neutral", "polarity": 1})
    r = verify_plan(SRC, "Shape x", plan)
    assert any(d["code"] == "INVALID_LINE_RANGE" for d in r.defects)


def test_v4_prompt_requests_numbered_text_and_v3_is_preserved() -> None:
    assert "line_start" in DERIVATION_PLANNER_PROMPT and "line_start" not in DERIVATION_PLANNER_PROMPT_V3
    assert "config/limits.yaml" in DERIVATION_PLANNER_PROMPT  # generic example, unrelated to measured cases
    from claimlayer.semantic.models import SourceDocument
    from claimlayer.v4 import DEFAULT_OPERATION_CATALOG, DomainProfile
    from claimlayer.v4.planner import ClaimDerivationPlanner

    class _Cap:
        identity = {"provider": "cap"}
        payload: dict[str, Any] = {}

        def complete(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
            _Cap.payload = payload
            return {}
    ClaimDerivationPlanner(model=_Cap()).plan(sources=[SourceDocument(document_id="d1", text=DIFF)], claim="c",
                                              catalog=DEFAULT_OPERATION_CATALOG, domain_profile=DomainProfile(profile_id="t"))
    assert _Cap.payload["sources"][0]["numbered_text"].startswith(" 1| File: lib/shapes.py")
