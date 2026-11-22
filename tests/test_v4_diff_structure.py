"""Point 2: diff structure over the full patch, hunk-based counting, checked against an independent
reference (`git diff --numstat` and the files themselves), plus truncation and '++' controls."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from claimlayer.v4 import verify_plan
from claimlayer.v4.diff_evidence import count_diff_lines, parse_unified_diff

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git binary required for the independent reference")

BASE = {
    "a.txt": "".join(f"line {i}\n" for i in range(1, 31)),
    "b.cfg": "--remove-me\nkeep\nold tail",               # base without trailing newline
}
HEAD = {
    "a.txt": "".join(f"line {i}\n" for i in range(1, 31)).replace("line 5\n", "line 5\n++counter\n+++ not a header\n")
             .replace("line 25\n", "line 25 changed\n"),
    "b.cfg": "keep\nnew tail\n++plus",                    # head without trailing newline
    "c.new": "x\ny\nz",                                   # new file without trailing newline
}


def _git_patch(tmp: Path) -> tuple[str, dict[str, tuple[int, int]]]:
    b, h = tmp / "base", tmp / "head"
    for root, files in ((b, BASE), (h, HEAD)):
        root.mkdir()
        for name, txt in files.items():
            (root / name).write_text(txt)
    run = lambda *a: subprocess.run(["git", "-c", "core.quotepath=off", "diff", "--no-index", "--no-color", *a, "base", "head"],
                                    cwd=tmp, capture_output=True, text=True).stdout
    patch = run("-U3")
    numstat = {}
    for ln in run("--numstat").splitlines():
        add, rem, path = ln.split("\t")
        numstat[path.split("/")[-1].rstrip("}").split("=> ")[-1]] = (int(add), int(rem))
    return patch, numstat


def test_structure_matches_git_numstat_and_files(tmp_path: Path) -> None:
    patch, numstat = _git_patch(tmp_path)
    dm = parse_unified_diff(patch)
    assert dm is not None
    for f in ("a.txt", "b.cfg", "c.new"):
        path = "head/" + f
        add = count_diff_lines(dm, path=path, side="+")
        rem = count_diff_lines(dm, path=path, side="-")
        assert (add["count_git"], rem["count_git"]) == numstat[f], (f, add, rem, numstat[f])
        assert not add["truncated"]
    # '++counter' and '+++ not a header' are additions (shown as '+++…'), not file headers
    a_add = [dm.lines[i].content for i in dm.file_indexes("head/a.txt", "added")]
    assert "++counter" in a_add and "+++ not a header" in a_add
    # '--remove-me' is a removal (shown as '---remove-me'), not a header
    assert "--remove-me" in [dm.lines[i].content for i in dm.file_indexes("head/b.cfg", "removed")]
    # Convention, against the file itself: c.new has 3 lines for git, `wc -l` counts its newlines.
    c = count_diff_lines(dm, path="head/c.new", side="+")
    assert c["count_git"] == len(HEAD["c.new"].splitlines()) == 3
    assert c["count_wc"] == HEAD["c.new"].count("\n") == 2 and c["unterminated_lines"] == 1


def test_truncated_patch_keeps_observed_count_and_names_blocker(tmp_path: Path) -> None:
    patch, _ = _git_patch(tmp_path)
    lines = patch.split("\n")
    cut = max(i for i, l in enumerate(lines) if l == "+y")  # inside the c.new hunk (declares 3 head lines)
    truncated = "\n".join(lines[:cut + 1])
    dm = parse_unified_diff(truncated)
    c = count_diff_lines(dm, path="head/c.new", side="+")
    assert c["truncated"] and c["count_git"] == 2
    assert c["truncation"][0]["declared_head"] == 3 and c["truncation"][0]["observed_head"] == 2
    start = next(i for i, l in enumerate(truncated.split("\n")) if l == "+x") + 1
    plan = {
        "claim_target": {"subject": "head/c.new", "predicate": "líneas añadidas", "value": 3, "unit": "count", "polarity": 1,
                         "revision": "added"},
        "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "d", "file": "head/c.new", "side": "added",
                      "line_start": start, "line_end": start + 1}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+", "scope": "file", "line_convention": "git"}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }
    for value in (3, 2):  # neither derive the declared total nor refute it from a truncated patch
        r = verify_plan({"d": truncated}, f"head/c.new añade {value} líneas", {**plan, "claim_target": {**plan["claim_target"], "value": value}})
        assert r.epistemic_verdict == "insufficient_evidence" and r.outcome_category == "count_scope_truncated", r.defects
        assert r.computed_conclusion["value"] == 2 and "truncado" in r.explanation and "2 líneas" in r.explanation


def _manifest_doc() -> str:
    return "\n".join([
        "File: cfg/rules.txt (status: modified)",           # 1
        "Unified diff from base to head:",                   # 2
        "@@ -1,3 +1,4 @@",                                   # 3
        " keep",                                             # 4 context
        "---old",                                            # 5 removed: content '--old'
        "+++new",                                            # 6 added:   content '++new'
        "+second",                                           # 7 added
        " tail",                                             # 8 context
        "\\ No newline at end of file",                      # 9 marker
    ])


def test_plus_plus_content_is_an_addition_and_marker_is_classified() -> None:
    dm = parse_unified_diff(_manifest_doc())
    assert [l.kind for l in dm.lines] == ["header", "header", "hunk", "context", "removed", "added", "added", "context", "marker"]
    assert dm.lines[5].content == "++new" and dm.lines[4].content == "--old"
    assert dm.lines[7].unterminated  # context line followed by the marker
    assert count_diff_lines(dm, path="cfg/rules.txt", side="+")["count_git"] == 2
    assert dm.status_of("cfg/rules.txt") == "modified"


def _range_plan(a: int, b: int, scope: str, value: int) -> dict[str, Any]:
    return {
        "claim_target": {"subject": "cfg/rules.txt", "predicate": "líneas añadidas", "value": value, "unit": "count",
                         "polarity": 1, "revision": "added"},
        "premises": [{"claim_id": "p1", "kind": "diff_presence", "document_id": "d", "file": "cfg/rules.txt", "side": "added",
                      "line_start": a, "line_end": b}],
        "steps": [{"step_id": "s1", "operation": "count_lines", "premises": ["p1"], "conclusion_id": "c1",
                   "parameters": {"line_prefix": "+", "scope": scope}}],
        "derived_conclusion_id": "c1", "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def test_partial_range_abstains_and_file_scope_resolves() -> None:
    src = {"d": _manifest_doc()}
    r = verify_plan(src, "cfg/rules.txt añade 2 líneas", _range_plan(6, 6, "range", 2))
    assert r.epistemic_verdict == "insufficient_evidence" and r.outcome_category == "count_scope_partial"
    assert r.computed_conclusion["value"] == 1
    r = verify_plan(src, "cfg/rules.txt añade 2 líneas", _range_plan(6, 6, "file", 2))
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    r = verify_plan(src, "cfg/rules.txt añade 4 líneas", _range_plan(6, 7, "file", 4))
    assert r.epistemic_verdict == "refuted"


def test_side_inferred_only_when_range_has_one_kind() -> None:
    src = {"d": _manifest_doc()}
    p = _range_plan(6, 7, "file", 2)
    p["steps"][0]["parameters"] = {"scope": "file"}  # no side: range 6-7 holds only additions
    r = verify_plan(src, "cfg/rules.txt añade 2 líneas", p)
    assert r.epistemic_verdict == "derived", (r.outcome_category, r.defects)
    p2 = _range_plan(4, 6, "range", 2)
    p2["premises"][0] = {"claim_id": "p1", "document_id": "d", "line_start": 4, "line_end": 6, "entity": "cfg/rules.txt",
                         "event": "x", "predicate": "bloque", "value": None, "unit": None, "direction": "neutral", "polarity": 1}
    p2["steps"][0]["parameters"] = {}
    r2 = verify_plan(src, "cfg/rules.txt añade 2 líneas", p2)
    assert "COUNT_LINES_DIFF_REQUIRES_SIDE" in str(r2.defects) or r2.epistemic_verdict != "derived"
