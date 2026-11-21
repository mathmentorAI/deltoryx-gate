"""Revision / file / line-range binding for unified-diff evidence.

A source document is treated as a unified diff when it contains hunk headers
(`@@ -a,b +c,d @@`). Each line is classified as added (head only), removed (base only),
context (both revisions) or header, and attributed to a file path taken from a
`File: <path>` line, a `+++ b/<path>` line or a `diff --git a/x b/<path>` line.

Premises may cite `line_start`/`line_end` (1-based, inclusive, over the document text) instead
of copying the text: Deltoryx resolves the literal quote from the document, so multi-line diff
spans are literal by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_LINE = re.compile(r"^File: (.+?)(?: \(status: [^)]*\))?$")
_PLUS_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_DIFF_GIT = re.compile(r"^diff --git a/\S+ b/(\S+)")


@dataclass(frozen=True)
class DiffLine:
    kind: str  # 'added' | 'removed' | 'context' | 'hunk' | 'header' | 'marker'
    file: str | None
    head_line: int | None
    base_line: int | None


@dataclass(frozen=True)
class DiffMap:
    lines: tuple[DiffLine, ...]
    line_offsets: tuple[int, ...] = field(default=())  # char offset of each line start

    def lines_for_span(self, start: int, end: int) -> list[int]:
        """0-based line indexes overlapping character span [start, end)."""
        out = []
        for i, off in enumerate(self.line_offsets):
            nxt = self.line_offsets[i + 1] if i + 1 < len(self.line_offsets) else 10**12
            if off < end and nxt > start:
                out.append(i)
        return out

    def binding_for_span(self, start: int, end: int) -> dict[str, Any]:
        idx = self.lines_for_span(start, end)
        kinds = [self.lines[i].kind for i in idx]
        files = sorted({self.lines[i].file for i in idx if self.lines[i].file})
        sides = sorted({{"added": "+", "removed": "-", "context": " "}[k] for k in kinds if k in ("added", "removed", "context")})
        head = [self.lines[i].head_line for i in idx if self.lines[i].head_line is not None]
        base = [self.lines[i].base_line for i in idx if self.lines[i].base_line is not None]
        return {
            "files": files,
            "sides": sides,  # '+' head only, '-' base only, ' ' both
            "head_lines": [min(head), max(head)] if head else None,
            "base_lines": [min(base), max(base)] if base else None,
            "document_lines": [idx[0] + 1, idx[-1] + 1] if idx else None,
        }


def parse_unified_diff(text: str) -> DiffMap | None:
    raw = text.split("\n")
    if not any(_HUNK.match(l) for l in raw):
        return None
    offsets, pos = [], 0
    for l in raw:
        offsets.append(pos)
        pos += len(l) + 1
    cur_file: str | None = None
    head = base = None
    out: list[DiffLine] = []
    for l in raw:
        m_file = _FILE_LINE.match(l) or _DIFF_GIT.match(l)
        if m_file and head is None or (m_file and _DIFF_GIT.match(l)):
            cur_file = m_file.group(1).strip()
            out.append(DiffLine("header", cur_file, None, None))
            head = base = None
            continue
        if l.startswith("+++ ") and head is None:
            mp = _PLUS_FILE.match(l)
            if mp:
                cur_file = mp.group(1).strip()
            out.append(DiffLine("header", cur_file, None, None))
            continue
        if l.startswith("--- ") and head is None:
            out.append(DiffLine("header", cur_file, None, None))
            continue
        mh = _HUNK.match(l)
        if mh:
            base, head = int(mh.group(1)), int(mh.group(3))
            out.append(DiffLine("hunk", cur_file, None, None))
            continue
        if head is None:
            out.append(DiffLine("header", cur_file, None, None))
            continue
        if l.startswith("\\"):
            out.append(DiffLine("marker", cur_file, None, None))
        elif l.startswith("+"):
            out.append(DiffLine("added", cur_file, head, None)); head += 1
        elif l.startswith("-"):
            out.append(DiffLine("removed", cur_file, None, base)); base += 1
        else:
            out.append(DiffLine("context", cur_file, head, base)); head += 1; base += 1
    return DiffMap(tuple(out), tuple(offsets))


def resolve_line_range(text: str, line_start: Any, line_end: Any) -> str | None:
    """Literal text of 1-based inclusive lines [line_start, line_end]; None if invalid."""
    try:
        a, b = int(line_start), int(line_end)
    except (TypeError, ValueError):
        return None
    lines = text.split("\n")
    if a < 1 or b < a or b > len(lines):
        return None
    return "\n".join(lines[a - 1:b])


def numbered_text(text: str) -> str:
    """Planner view: each line prefixed with its 1-based number (for line_start/line_end)."""
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{str(i).rjust(width)}| {l}" for i, l in enumerate(lines, 1))


def path_matches(entity: str, files: Sequence[str]) -> bool:
    e = (entity or "").strip().strip("`'\"").lstrip("./")
    if not e:
        return False
    return any(f == e or f.endswith("/" + e) or e.endswith("/" + f) for f in files)
