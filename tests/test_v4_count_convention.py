"""Count convention: lines without a trailing newline (git counts them, `wc -l` does not).
Synthetic diffs only."""
from __future__ import annotations

from typing import Any

from claimlayer.v4 import verify_plan

NL = "\\ No newline at end of file"
DIFF = "\n".join([
    "File: etc/hosts.allow (status: added)",
    "Unified diff from base to head:",
    "@@ -0,0 +1,3 @@",
    "+alpha",
    "+beta",
    "+gamma",
    NL,
])
SRC = {"d1": DIFF}


def _plan(value: int, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_target": {"subject": "etc/hosts.allow", "predicate": "líneas añadidas", "value": value,
                         "unit": "count", "polarity": 1, "revision": "added"},
        "premises": [{"claim_id": "p1", "document_id": "d1", "line_start": 4, "line_end": 7,
                      "entity": "etc/hosts.allow", "event": "alpha", "predicate": "bloque añadido",
                      "value": None, "unit": None, "direction": "neutral", "polarity": 1}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": params}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def test_unset_convention_neither_derives_nor_refutes() -> None:
    for v in (3, 2):  # git count and wc count: both convention-dependent
        r = verify_plan(SRC, f"El archivo etc/hosts.allow añade {v} líneas", _plan(v, {"line_prefix": "+"}))
        assert r.epistemic_verdict == "insufficient_evidence", (v, r.outcome_category, r.defects)
        assert r.outcome_category == "count_convention_ambiguous"
        assert r.operational_routing.status == "ESCALATE"
        assert any(d["code"] == "COUNT_CONVENTION_AMBIGUOUS" for d in r.defects)


def test_fixed_convention_resolves() -> None:
    r = verify_plan(SRC, "El archivo etc/hosts.allow añade 3 líneas", _plan(3, {"line_prefix": "+", "line_convention": "git"}))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    r = verify_plan(SRC, "El archivo etc/hosts.allow tiene 2 líneas según wc -l",
                    _plan(2, {"line_prefix": "+", "line_convention": "wc"}))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    r = verify_plan(SRC, "El archivo etc/hosts.allow añade 5 líneas", _plan(5, {"line_prefix": "+", "line_convention": "git"}))
    assert r.epistemic_verdict == "refuted"


def test_terminated_block_is_not_convention_sensitive() -> None:
    src = {"d1": DIFF.rsplit("\n", 1)[0] + "\n+delta"}
    r = verify_plan(src, "El archivo etc/hosts.allow añade 4 líneas", _plan(4, {"line_prefix": "+"}))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
