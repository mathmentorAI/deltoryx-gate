"""Revision / file / line-range binding for unified-diff evidence.

A source document is treated as a unified diff when it contains hunk headers
(`@@ -a,b +c,d @@`). The structure is computed over the WHOLE document, hunk by hunk, using the
line counts declared in each hunk header (never by prefix heuristics), so a content line whose
text starts with `++` or `--` (shown as `+++...` / `---...`) is still an addition / removal.
Each document line is one of:

* ``header``  – file preamble (`File: …`, `diff --git …`, `index …`, `--- a/…`, `+++ b/…`, other text
  outside hunks);
* ``hunk``    – a hunk header `@@ … @@`;
* ``added`` / ``removed`` / ``context`` – hunk body lines (head only / base only / both revisions);
* ``marker``  – `\\ No newline at end of file`, attached to the previous body line (that line has no
  terminating newline in its revision).

A hunk whose declared counts are not reached before the document ends (or before a line that
cannot belong to a hunk body) is marked ``truncated``; counts over a truncated file are observed
counts only.

Premises may cite `line_start`/`line_end` (1-based, inclusive, over the document text) instead of
copying the text: Deltoryx resolves the literal quote from the document, so multi-line diff spans
are literal by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_LINE = re.compile(r"^File: (.+?)(?: \(status: ([^)]*)\))?$")
_PLUS_FILE = re.compile(r"^\+\+\+ (?:b/)?(.+)$")
_DIFF_GIT = re.compile(r"^diff --git a/\S+ b/(\S+)")
NO_NEWLINE_MARKER = "\\ No newline at end of file"

SIDE_OF_KIND = {"added": "+", "removed": "-", "context": " "}
KIND_OF_SIDE = {"+": "added", "-": "removed", " ": "context"}


@dataclass(frozen=True)
class DiffLine:
    kind: str  # 'added' | 'removed' | 'context' | 'hunk' | 'header' | 'marker'
    file: str | None
    head_line: int | None
    base_line: int | None
    content: str = ""            # body text without the diff prefix ('' for non-body lines)
    hunk: int | None = None      # index of the hunk this line belongs to
    unterminated: bool = False   # body line followed by the 'No newline' marker


@dataclass(frozen=True)
class HunkInfo:
    index: int
    file: str | None
    header_line: int            # 1-based document line of the '@@' header
    declared_base: int
    declared_head: int
    observed_base: int          # removed + context lines present
    observed_head: int          # added + context lines present
    truncated: bool


@dataclass(frozen=True)
class DiffMap:
    lines: tuple[DiffLine, ...]
    line_offsets: tuple[int, ...] = field(default=())  # char offset of each line start
    hunks: tuple[HunkInfo, ...] = ()
    file_status: tuple[tuple[str, str], ...] = ()     # (path, status) from `File: … (status: …)`

    # ---- span helpers ----------------------------------------------------------------------
    def lines_for_span(self, start: int, end: int) -> list[int]:
        """0-based line indexes overlapping character span [start, end)."""
        out = []
        for i, off in enumerate(self.line_offsets):
            nxt = self.line_offsets[i + 1] if i + 1 < len(self.line_offsets) else 10**12
            if off < end and nxt > start:
                out.append(i)
        return out

    def binding_for_span(self, start: int, end: int) -> dict[str, Any]:
        return self.binding_for_indexes(self.lines_for_span(start, end))

    def binding_for_indexes(self, idx: Sequence[int]) -> dict[str, Any]:
        idx = list(idx)
        kinds = [self.lines[i].kind for i in idx]
        files = sorted({self.lines[i].file for i in idx if self.lines[i].file})
        sides = sorted({SIDE_OF_KIND[k] for k in kinds if k in SIDE_OF_KIND})
        head = [self.lines[i].head_line for i in idx if self.lines[i].head_line is not None]
        base = [self.lines[i].base_line for i in idx if self.lines[i].base_line is not None]
        return {
            "files": files,
            "sides": sides,  # '+' head only, '-' base only, ' ' both
            "head_lines": [min(head), max(head)] if head else None,
            "base_lines": [min(base), max(base)] if base else None,
            "document_lines": [idx[0] + 1, idx[-1] + 1] if idx else None,
        }

    # ---- file-level structure --------------------------------------------------------------
    def files(self) -> list[str]:
        return sorted({l.file for l in self.lines if l.file})

    def file_indexes(self, path: str, kind: str | None = None) -> list[int]:
        return [i for i, l in enumerate(self.lines)
                if l.file == path and (kind is None or l.kind == kind)]

    def file_truncated(self, path: str) -> list[HunkInfo]:
        return [h for h in self.hunks if h.file == path and h.truncated]

    def status_of(self, path: str) -> str | None:
        return dict(self.file_status).get(path)


def parse_unified_diff(text: str) -> DiffMap | None:
    raw = text.split("\n")
    if not any(_HUNK.match(l) for l in raw):
        return None
    offsets, pos = [], 0
    for l in raw:
        offsets.append(pos)
        pos += len(l) + 1
    out: list[DiffLine] = []
    hunks: list[HunkInfo] = []
    statuses: list[tuple[str, str]] = []
    cur_file: str | None = None
    # open hunk state
    h_idx: int | None = None
    rem_base = rem_head = 0
    head = base = 0
    h_meta: dict[str, Any] = {}

    def close_hunk(truncated: bool) -> None:
        nonlocal h_idx
        if h_idx is None:
            return
        hunks.append(HunkInfo(index=h_idx, file=h_meta["file"], header_line=h_meta["line"],
                              declared_base=h_meta["db"], declared_head=h_meta["dh"],
                              observed_base=h_meta["db"] - rem_base, observed_head=h_meta["dh"] - rem_head,
                              truncated=truncated))
        h_idx = None

    for n, l in enumerate(raw):
        if h_idx is not None:
            if l.startswith("\\"):
                if out and out[-1].kind in SIDE_OF_KIND:
                    prev = out[-1]
                    out[-1] = DiffLine(prev.kind, prev.file, prev.head_line, prev.base_line, prev.content,
                                       prev.hunk, unterminated=True)
                out.append(DiffLine("marker", cur_file, None, None, "", h_idx))
                continue
            if rem_base == 0 and rem_head == 0:
                close_hunk(truncated=False)
            else:
                body_kind = None
                if l.startswith("+") and rem_head > 0:
                    body_kind = "added"
                elif l.startswith("-") and rem_base > 0:
                    body_kind = "removed"
                elif (l.startswith(" ") or l == "") and rem_head > 0 and rem_base > 0:
                    body_kind = "context"
                if body_kind is None:
                    # cannot belong to the declared hunk body: the hunk is truncated here
                    close_hunk(truncated=True)
                else:
                    content = l[1:] if l else ""
                    if body_kind == "added":
                        out.append(DiffLine("added", cur_file, head, None, content, h_idx)); head += 1; rem_head -= 1
                    elif body_kind == "removed":
                        out.append(DiffLine("removed", cur_file, None, base, content, h_idx)); base += 1; rem_base -= 1
                    else:
                        out.append(DiffLine("context", cur_file, head, base, content, h_idx))
                        head += 1; base += 1; rem_head -= 1; rem_base -= 1
                    continue
        # outside a hunk
        mh = _HUNK.match(l)
        if mh:
            base, head = int(mh.group(1)), int(mh.group(3))
            db = int(mh.group(2)) if mh.group(2) is not None else 1
            dh = int(mh.group(4)) if mh.group(4) is not None else 1
            h_idx = len(hunks)
            rem_base, rem_head = db, dh
            h_meta = {"file": cur_file, "line": n + 1, "db": db, "dh": dh}
            out.append(DiffLine("hunk", cur_file, None, None, "", h_idx))
            continue
        m_git = _DIFF_GIT.match(l)
        m_file = _FILE_LINE.match(l)
        if m_git:
            cur_file = m_git.group(1).strip()
        elif m_file:
            cur_file = m_file.group(1).strip()
            if m_file.group(2):
                statuses.append((cur_file, m_file.group(2).strip()))
        elif l.startswith("+++ "):
            mp = _PLUS_FILE.match(l)
            if mp and mp.group(1).strip() != "/dev/null":
                cur_file = mp.group(1).strip()
        out.append(DiffLine("header", cur_file, None, None))
    if h_idx is not None:
        close_hunk(truncated=(rem_base > 0 or rem_head > 0))
    return DiffMap(tuple(out), tuple(offsets), tuple(sorted(hunks, key=lambda h: h.index)), tuple(statuses))


def count_diff_lines(
    dmap: DiffMap,
    *,
    path: str,
    side: str,
    indexes: Sequence[int] | None = None,
    contains: str | None = None,
) -> dict[str, Any]:
    """Count body lines of `side` ('+', '-', ' ') for file `path`, over `indexes` (a cited range)
    or over the whole file when `indexes` is None. Structural: headers, hunk headers and markers
    are never counted; prefixes are never used to exclude lines.

    Returns observed counts, the git/wc conventions (a counted line without trailing newline is a
    line for git/diffstat and not for `wc -l`), whether the range covers every line of that side in
    the file, and truncation blockers.
    """
    kind = KIND_OF_SIDE[side]
    all_idx = dmap.file_indexes(path, kind)
    sel = all_idx if indexes is None else [i for i in indexes if dmap.lines[i].file == path and dmap.lines[i].kind == kind]
    if contains is not None:
        sel = [i for i in sel if contains in dmap.lines[i].content]
        universe = [i for i in all_idx if contains in dmap.lines[i].content]
    else:
        universe = all_idx
    unterminated = sum(1 for i in sel if dmap.lines[i].unterminated)
    trunc = dmap.file_truncated(path)
    return {
        "file": path, "side": side, "scope": "file" if indexes is None else "range",
        "count_git": len(sel), "count_wc": len(sel) - unterminated, "unterminated_lines": unterminated,
        "range_covers_file": set(sel) >= set(universe),
        "file_total_observed": len(universe),
        "truncated": bool(trunc),
        "truncation": [{"hunk_header_line": h.header_line, "declared_head": h.declared_head,
                        "observed_head": h.observed_head, "declared_base": h.declared_base,
                        "observed_base": h.observed_base} for h in trunc],
        "head_lines": [dmap.lines[i].head_line for i in sel if dmap.lines[i].head_line is not None][:1]
                      + [dmap.lines[i].head_line for i in sel if dmap.lines[i].head_line is not None][-1:],
    }


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


def normalize_path(p: str) -> str:
    """Trim quotes/spaces and a leading './' or '/' prefix (never leading dots of a name: '.claude/x')."""
    e = (p or "").strip().strip("`'\"")
    while e.startswith("./"):
        e = e[2:]
    return e.lstrip("/")


def path_matches(entity: str, files: Sequence[str]) -> bool:
    e = normalize_path(entity)
    if not e:
        return False
    return any((f := normalize_path(x)) == e or f.endswith("/" + e) or e.endswith("/" + f) for x in files)


def resolve_path(entity: str, files: Sequence[str]) -> str | None:
    e = normalize_path(entity)
    hits = [x for x in files if (f := normalize_path(x)) == e or f.endswith("/" + e) or e.endswith("/" + f)]
    return hits[0] if len(hits) == 1 else None


# ---- premise support ------------------------------------------------------------------------
_REV_PAIR = re.compile(r"base commit ([0-9a-f]{7,40})\s*->\s*head commit ([0-9a-f]{7,40})", re.I)
_KCHAR = {"added": "a", "removed": "r", "context": "c"}


def diff_premise_metadata(dmap: DiffMap, document_id: str, line_start: int, line_end: int) -> dict[str, Any]:
    """Compact, serialisable structure for a premise citing document lines [line_start, line_end]:
    the body rows of every file the range touches (doc line, kind, unterminated, content) plus
    truncation blockers, so count_lines can count structurally over the range or the whole file."""
    idx = range(line_start - 1, line_end)
    range_files = sorted({dmap.lines[i].file for i in idx if dmap.lines[i].file and dmap.lines[i].kind in SIDE_OF_KIND}) \
        or sorted({dmap.lines[i].file for i in idx if dmap.lines[i].file})
    files: dict[str, Any] = {}
    for f in range_files:
        rows = [[i + 1, _KCHAR[l.kind], bool(l.unterminated), l.content]
                for i, l in enumerate(dmap.lines) if l.file == f and l.kind in _KCHAR]
        files[f] = {"rows": rows, "status": dmap.status_of(f),
                    "truncation": [{"hunk_header_line": h.header_line, "declared_head": h.declared_head,
                                    "observed_head": h.observed_head, "declared_base": h.declared_base,
                                    "observed_base": h.observed_base} for h in dmap.file_truncated(f)]}
    return {"document_id": document_id, "range": [line_start, line_end], "range_files": range_files, "files": files}


def find_revision_pair(texts: Iterable[str]) -> tuple[str, str] | None:
    for t in texts:
        m = _REV_PAIR.search(t or "")
        if m:
            return m.group(1).lower(), m.group(2).lower()
    return None


def verify_diff_presence(
    premise: Mapping[str, Any],
    doc_text: str | None,
    dmap: DiffMap | None,
    revision_pair: tuple[str, str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check a typed diff-presence premise against the patch.

    Premise: {kind: 'diff_presence', document_id, file, side: 'added'|'removed'|'context',
    line_start, line_end, text?, base_revision?, head_revision?}. It asserts only a TEXTUAL fact:
    these lines appear on this side of the diff of this file between the two revisions. It does not
    establish execution, behaviour or test results.
    """
    defects: list[dict[str, Any]] = []
    info: dict[str, Any] = {"scope": "textual_modification"}

    def bad(code: str, msg: str, **detail: Any) -> None:
        defects.append({"code": code, "message": msg, "detail": detail})

    if doc_text is None:
        bad("DIFF_PRESENCE_DOCUMENT_NOT_FOUND", "El documento citado no existe.")
        return defects, info
    if dmap is None:
        bad("DIFF_PRESENCE_NOT_A_DIFF", "El documento citado no es un diff unificado.")
        return defects, info
    side = str(premise.get("side") or "")
    if side not in KIND_OF_SIDE.values():
        bad("DIFF_PRESENCE_INVALID_SIDE", f"Lado inválido {side!r}: debe ser added, removed o context.")
        return defects, info
    try:
        a, b = int(premise.get("line_start")), int(premise.get("line_end"))
    except (TypeError, ValueError):
        bad("INVALID_LINE_RANGE", "La premisa de presencia necesita line_start y line_end.")
        return defects, info
    if a < 1 or b < a or b > len(dmap.lines):
        bad("INVALID_LINE_RANGE", f"Rango de líneas inválido {a}-{b}.")
        return defects, info
    rows = [dmap.lines[i] for i in range(a - 1, b)]
    non_body = [a + k for k, l in enumerate(rows) if l.kind not in SIDE_OF_KIND and l.kind != "marker"]
    if non_body:
        bad("DIFF_PRESENCE_RANGE_NOT_BODY", "El rango incluye cabeceras, cabeceras de bloque o marcadores, no líneas del cambio.",
            lines=non_body)
    wrong = [{"line": a + k, "kind": l.kind} for k, l in enumerate(rows) if l.kind in SIDE_OF_KIND and l.kind != side]
    if wrong:
        bad("DIFF_PRESENCE_SIDE_MISMATCH", f"Las líneas citadas no están en el lado {side!r} del diff.", lines=wrong)
    files = sorted({l.file for l in rows if l.file})
    want = resolve_path(str(premise.get("file") or ""), dmap.files())
    if want is None or files != [want]:
        bad("DIFF_PRESENCE_FILE_MISMATCH", f"La ruta {premise.get('file')!r} no es el archivo de las líneas citadas ({files}).",
            cited_files=files)
    exp_text = premise.get("text")
    actual = "\n".join(l.content for l in rows if l.kind in SIDE_OF_KIND)
    if exp_text is not None and str(exp_text) != actual:
        bad("DIFF_PRESENCE_TEXT_MISMATCH", "El texto declarado no coincide con el contenido de las líneas citadas.",
            declared=str(exp_text)[:200], actual=actual[:200])
    for key, pos in (("base_revision", 0), ("head_revision", 1)):
        v = premise.get(key)
        if v:
            if revision_pair is None or not revision_pair[pos].startswith(str(v).lower()) or len(str(v)) < 7:
                bad("DIFF_PRESENCE_REVISION_UNVERIFIED", f"{key} {v!r} no coincide con las revisiones declaradas por el corpus.",
                    declared_pair=revision_pair)
    head = [l.head_line for l in rows if l.head_line is not None]
    base = [l.base_line for l in rows if l.base_line is not None]
    info.update({"file": want, "side": side, "document_lines": [a, b], "content": actual,
                 "head_lines": [min(head), max(head)] if head else None,
                 "base_lines": [min(base), max(base)] if base else None,
                 "revision_pair": list(revision_pair) if revision_pair else None,
                 "limits": "Acredita una modificación textual; no acredita ejecución, comportamiento ni resultado de tests."})
    return defects, info
