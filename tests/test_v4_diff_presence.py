"""Point 3: typed diff-presence premise. Asserts a textual modification only: these lines appear on
this side of this file's diff between two revisions. Synthetic sources only."""
from __future__ import annotations

from typing import Any

from claimlayer.v4 import verify_plan

SUMMARY = "Pull request diff summary (base commit aaaaaaa1111111 -> head commit bbbbbbb2222222)\nFiles changed: 2"
DOC_A = "\n".join([
    "File: src/auth.py (status: modified)",      # 1
    "Unified diff from base to head:",           # 2
    "@@ -1,4 +1,4 @@",                           # 3
    " import hashlib",                           # 4 context (both revisions)
    "-TOKEN_TTL = 60",                           # 5 removed (base only)
    "+TOKEN_TTL = 3600",                         # 6 added (head only)
    " def check(token):",                        # 7 context
    "     return verify(token)",                 # 8 context
])
DOC_B = "\n".join([
    "File: src/legacy.py (status: modified)",    # 1
    "Unified diff from base to head:",           # 2
    "@@ -1,1 +1,2 @@",                           # 3
    " x = 1",                                    # 4
    "+TOKEN_TTL = 3600",                         # 5 same text, other file
])
SRC = {"summary": SUMMARY, "a": DOC_A, "b": DOC_B}


def _plan(prem: dict[str, Any], predicate: str = "línea añadida", subject: str = "src/auth.py", value: Any = True) -> dict[str, Any]:
    p = {"claim_id": "p1", "kind": "diff_presence", "document_id": "a", "file": "src/auth.py", "side": "added",
         "line_start": 6, "line_end": 6, "text": "TOKEN_TTL = 3600", **prem}
    return {"claim_target": {"subject": subject, "predicate": predicate, "value": value, "unit": None, "polarity": 1,
                             "revision": "added"},
            "premises": [p],
            "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1", "parameters": {}}],
            "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": []}


CLAIM = "Se añadió la línea TOKEN_TTL = 3600 en src/auth.py"


def _codes(r) -> set[str]:
    return {d.get("code") for d in r.defects}


def test_presence_added_line_derives_without_event_or_unit() -> None:
    r = verify_plan(SRC, CLAIM, _plan({"base_revision": "aaaaaaa", "head_revision": "bbbbbbb2222222"}))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    assert r.reliability["grade"] == "A" and r.operational_routing.status == "PASS"
    prem = r.premises[0]
    assert prem["kind"] == "diff_presence" and prem["event"] == "" and prem["unit"] is None
    dp = prem["diff_presence"]
    assert dp["file"] == "src/auth.py" and dp["side"] == "added" and dp["head_lines"] == [2, 2]
    assert dp["scope"] == "textual_modification" and "ejecución" in dp["limits"]
    assert dp["revision_pair"] == ["aaaaaaa1111111", "bbbbbbb2222222"]


def test_control_same_text_only_in_base() -> None:
    # line 5 is '-TOKEN_TTL = 60' (base only); claimed as added -> premise defect
    r = verify_plan(SRC, "Se añadió la línea TOKEN_TTL = 60 en src/auth.py",
                    _plan({"line_start": 5, "line_end": 5, "text": "TOKEN_TTL = 60"}))
    assert r.epistemic_verdict == "unresolved" and "DIFF_PRESENCE_SIDE_MISMATCH" in _codes(r)
    # declared truthfully as removed, it cannot support an 'added' claim
    r = verify_plan(SRC, "Se añadió la línea TOKEN_TTL = 60 en src/auth.py",
                    _plan({"line_start": 5, "line_end": 5, "text": "TOKEN_TTL = 60", "side": "removed"}))
    assert r.epistemic_verdict != "derived" and _codes(r) & {"CONCLUSION_SIDE_MISMATCH", "BASE_ONLY_EVIDENCE_FOR_HEAD_CLAIM"}
    # ...but it does support a claim about what was removed
    p = _plan({"line_start": 5, "line_end": 5, "text": "TOKEN_TTL = 60", "side": "removed"}, predicate="línea eliminada")
    p["claim_target"]["revision"] = "removed"
    r = verify_plan(SRC, "Se eliminó la línea TOKEN_TTL = 60 de src/auth.py", p)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)


def test_control_same_text_in_other_file() -> None:
    # the premise names src/auth.py but cites document b (src/legacy.py)
    r = verify_plan(SRC, CLAIM, _plan({"document_id": "b", "line_start": 5, "line_end": 5}))
    assert r.epistemic_verdict == "unresolved" and "DIFF_PRESENCE_FILE_MISMATCH" in _codes(r)
    # honest premise about the other file does not bind to the claimed subject
    r = verify_plan(SRC, CLAIM, _plan({"document_id": "b", "line_start": 5, "line_end": 5, "file": "src/legacy.py"}))
    assert r.epistemic_verdict != "derived" and "CONCLUSION_SUBJECT_MISMATCH" in _codes(r)


def test_control_text_on_context_line() -> None:
    r = verify_plan(SRC, "Se añadió la línea import hashlib en src/auth.py",
                    _plan({"line_start": 4, "line_end": 4, "text": "import hashlib"}))
    assert r.epistemic_verdict == "unresolved" and "DIFF_PRESENCE_SIDE_MISMATCH" in _codes(r)
    r = verify_plan(SRC, "Se añadió la línea import hashlib en src/auth.py",
                    _plan({"line_start": 4, "line_end": 4, "text": "import hashlib", "side": "context"}))
    assert r.epistemic_verdict != "derived" and _codes(r) & {"CONCLUSION_SIDE_MISMATCH", "NO_CHANGED_LINE_EVIDENCE"}


def test_header_range_text_mismatch_and_revision_checks() -> None:
    assert "DIFF_PRESENCE_RANGE_NOT_BODY" in _codes(verify_plan(SRC, CLAIM, _plan({"line_start": 3, "line_end": 3})))
    assert "DIFF_PRESENCE_TEXT_MISMATCH" in _codes(verify_plan(SRC, CLAIM, _plan({"text": "TOKEN_TTL = 7200"})))
    assert "DIFF_PRESENCE_REVISION_UNVERIFIED" in _codes(verify_plan(SRC, CLAIM, _plan({"head_revision": "ccccccc"})))
    assert "DIFF_PRESENCE_NOT_A_DIFF" in _codes(verify_plan(SRC, CLAIM, _plan({"document_id": "summary", "line_start": 1, "line_end": 1})))


def test_scope_textual_modification_only() -> None:
    for pred, claim in (("se ejecuta", "La comprobación de src/auth.py se ejecuta con TTL 3600"),
                        ("el test pasa", "El test de src/auth.py pasa"),
                        ("fixes token expiry", "Fixes token expiry in src/auth.py")):
        r = verify_plan(SRC, claim, _plan({}, predicate=pred))
        assert r.epistemic_verdict == "insufficient_evidence", (pred, r.outcome_category, r.defects)
        assert r.outcome_category == "presence_scope_exceeded" and r.operational_routing.status == "ESCALATE"


def test_presence_premise_cannot_prove_a_number() -> None:
    r = verify_plan(SRC, "src/auth.py añade 1 línea", _plan({}, predicate="líneas añadidas", value=1))
    assert r.epistemic_verdict != "derived" and "PRESENCE_REQUIRES_BOOLEAN_TARGET" in _codes(r)


def test_dotted_paths_and_trailing_marker() -> None:
    from claimlayer.v4.diff_evidence import normalize_path, path_matches
    assert normalize_path("./.claude/x.yml") == ".claude/x.yml" and normalize_path(".claude/x.yml") == ".claude/x.yml"
    assert not path_matches("claude/x.yml", [".claude/x.yml"])
    doc = "\n".join(["File: .claude/c.yml (status: added)", "Unified diff from base to head:", "@@ -0,0 +1,2 @@",
                     "+a: 1", "+b: 2", "\\ No newline at end of file"])
    plan = _plan({"document_id": "d", "file": ".claude/c.yml", "line_start": 4, "line_end": 6, "text": None},
                 subject=".claude/c.yml", predicate="archivo añadido")
    plan["premises"][0].pop("text")
    r = verify_plan({"d": doc}, "Se añadió .claude/c.yml", plan)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
