"""Prompt v5: the worked example (synthetic config/limits.yaml) must verify with the stage-3
verifier, and earlier prompts stay available unchanged."""
from __future__ import annotations

from typing import Any

from claimlayer.semantic.models import SourceDocument
from claimlayer.v4 import (DEFAULT_OPERATION_CATALOG, DERIVATION_PLANNER_PROMPT, DERIVATION_PLANNER_PROMPT_V4,
                           DERIVATION_PLANNER_PROMPT_VERSION, DomainProfile, verify_plan)
from claimlayer.v4.count_definitions import CountDefinition
from claimlayer.v4.planner import ClaimDerivationPlanner, ClaimVerifier

DOC = "\n".join(["File: config/limits.yaml (status: modified)", "Unified diff from base to head:", "@@ -1,2 +1,5 @@",
                 " limits:", "+  max_rate: 10", "+  max_burst: 20", "+  max_conn: 5", "   timeout: 3"])


def test_v5_example_plans_verify() -> None:
    assert DERIVATION_PLANNER_PROMPT_VERSION == "v4-derivation-planner-5"
    assert "diff_presence" in DERIVATION_PLANNER_PROMPT and "diff_presence" not in DERIVATION_PLANNER_PROMPT_V4
    count = {"claim_target": {"subject": "config/limits.yaml", "predicate": "líneas añadidas", "value": 3, "unit": "count", "revision": "added"},
             "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "d1", "file": "config/limits.yaml", "side": "added",
                           "line_start": 5, "line_end": 7}],
             "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                        "parameters": {"line_prefix": "+", "scope": "file"}}],
             "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": []}
    r = verify_plan({"d1": DOC}, "El cambio añade tres líneas a config/limits.yaml", count)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    pres = {"claim_target": {"subject": "config/limits.yaml", "predicate": "clave añadida", "value": True, "unit": None, "revision": "added"},
            "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "d1", "file": "config/limits.yaml", "side": "added",
                          "line_start": 5, "line_end": 5, "text": "  max_rate: 10"}],
            "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1", "parameters": {}}],
            "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": []}
    r = verify_plan({"d1": DOC}, "El cambio añade la clave max_rate a config/limits.yaml", pres)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)


def test_count_definitions_reach_the_planner_payload() -> None:
    class _Cap:
        identity = {"provider": "cap"}
        payload: dict[str, Any] = {}

        def complete(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
            _Cap.payload = payload
            return {}
    d = CountDefinition("yaml_limits", "1.0.0", frozenset({"limite", "limites"}), contains="max_", line_prefix="+")
    v = ClaimVerifier(planner=ClaimDerivationPlanner(model=_Cap()), count_definitions=[d], max_retries=0)
    v.planner.plan(sources=[SourceDocument(document_id="d1", text=DOC)], claim="c", catalog=DEFAULT_OPERATION_CATALOG,
                   domain_profile=DomainProfile(profile_id="t"))
    assert _Cap.payload["count_definitions"][0]["definition_id"] == "yaml_limits"
    _Cap.payload = {}
    ClaimDerivationPlanner(model=_Cap()).plan(sources=[SourceDocument(document_id="d1", text=DOC)], claim="c",
                                              catalog=DEFAULT_OPERATION_CATALOG, domain_profile=DomainProfile(profile_id="t"))
    assert _Cap.payload["count_definitions"] == []
