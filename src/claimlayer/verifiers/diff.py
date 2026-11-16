"""Deterministic section-level divergence detector for multi-document audits.

Discovers sections where a secondary document (e.g., Offering Memorandum or Counterparty Markup)
diverges quantitatively or structurally from an authoritative baseline (e.g., Indenture or MSA),
automatically generating completeness predicates for Policy.required_predicates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from claimlayer.models import Policy, Source, normalize_key
from claimlayer.verifiers.quote_quantity import (
    _canonicalize_layout_text,
    _normalize_pair_with_maps,
    extract_dates,
    extract_quantities,
)

_CROSS_REF_CONTINUATION_RE = re.compile(
    r"^(?:above|below|hereof|hereto|thereof|herein|therein|of\b|in\b|to\b|for\b|and\b|or\b|"
    r"shall\b|is\b|was\b|are\b|applies\b|governs\b|sets\b|provides\b|"
    r"anterior\b|siguiente\b|del\b|de\s+la\b|establece\b)",
    re.IGNORECASE,
)

# Named headers: Section 2.04, ARTICLE IV, Cláusula 4, Cláusula Primera, Anexo A, etc.
_NAMED_SECTION_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}\s*)?"
    r"(?P<kw>Section|Sec\.|Article|Art\.|Clause|Cláusula|Clausula|Artículo|Articulo|"
    r"Item|Schedule|Annex|Anexo|Exhibit|Apartado|Punto|§)\s*"
    r"(?P<num>\d+(?:\.\d+)*|"
    r"primera|segunda|tercera|cuarta|quinta|sexta|s[eé]ptima|octava|novena|d[eé]cima|"
    r"first|second|third|fourth|fifth|"
    r"[IVXLCDM]+|[A-Z](?:\.\d+)*)\b"
    r"(?:(?P<sep>\s*[:.\-—]\s*)(?P<title>[^\n]+)|[ \t]+(?P<bare_title>[A-ZÁÉÍÓÚÑ][^\n]*)|[ \t]*$)",
    re.IGNORECASE,
)

# Bare numbered or ordinal headers at line start: "4.01 Events of Default", "PRIMERA.- Objeto", "1.- Objeto", "1: Purpose"
# Note: bare "1)" at line start is a list enumeration marker equivalent to "(1)"/"1." rather than a standalone section break.
_BARE_NUMBER_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}\s*)?"
    r"(?:"
    r"(?P<num>\d+\.\d+(?:\.\d+)*)(?P<num_sep>\s*[:.\-—)]\s*|\s+)"
    r"|(?P<ord>[A-ZÁÉÍÓÚÑ]{4,18}|\d+)(?P<ord_sep>\s*(?:\.\s*[-—:]|[:—]))\s*"
    r")"
    r"(?P<title>[A-ZÁÉÍÓÚÑ][^\n]+)"
)

_TRAILING_CONTINUATION_WORD_RE = re.compile(
    r"\b(?:to|in|under|of|from|by|with|for|and|or|see|per|pursuant|bajo|seg[uú]n|conforme|a|de|en|por|para|con|y|o)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SectionDivergence:
    """A deterministic discrepancy detected between an authoritative and secondary section."""

    section_key: str
    section_label: str
    primary_source_id: str
    secondary_source_id: str
    primary_quantities: tuple[str, ...]
    secondary_quantities: tuple[str, ...]
    reason: str
    primary_span: tuple[int, int] | None = None
    secondary_span: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SectionComparisonReport:
    """Full comparison output including per-document character coverage and stripped PDF furniture."""

    divergences: tuple[SectionDivergence, ...]
    primary_coverage: float
    secondary_coverage: float
    primary_uncovered_spans: tuple[tuple[int, int], ...] = ()
    secondary_uncovered_spans: tuple[tuple[int, int], ...] = ()
    primary_excluded_furniture: tuple[dict[str, Any], ...] = ()
    secondary_excluded_furniture: tuple[dict[str, Any], ...] = ()
    exclusion_symmetric: bool = True
    exclusion_asymmetry: bool = False
    exclusion_symmetry_check: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.exclusion_symmetry_check is None:
            object.__setattr__(
                self,
                "exclusion_symmetry_check",
                {
                    "symmetric": bool(self.exclusion_symmetric),
                    "asymmetry_detected": bool(self.exclusion_asymmetry),
                    "primary_excluded_only": [],
                    "secondary_excluded_only": [],
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergences": [d.to_dict() for d in self.divergences],
            "primary_coverage": self.primary_coverage,
            "secondary_coverage": self.secondary_coverage,
            "primary_uncovered_spans": [list(sp) for sp in self.primary_uncovered_spans],
            "secondary_uncovered_spans": [list(sp) for sp in self.secondary_uncovered_spans],
            "primary_excluded_furniture": [dict(x) for x in self.primary_excluded_furniture],
            "secondary_excluded_furniture": [dict(x) for x in self.secondary_excluded_furniture],
            "exclusion_symmetric": bool(self.exclusion_symmetric),
            "exclusion_asymmetry": bool(self.exclusion_asymmetry),
            "exclusion_symmetry_check": dict(self.exclusion_symmetry_check),
        }


@dataclass(frozen=True)
class _ParsedSection:
    key: str
    label: str
    title_key: str
    body: str
    body_without_header: str
    start: int = 0
    end: int = 0


def _strip_leading_header_id(text: str, sec_id: str) -> str:
    pattern = re.compile(r"^\s*(?:#{1,6}\s*)?" + re.escape(sec_id) + r"\s*[:.\-—]?\s*", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


def _is_wrapped_sentence_continuation(
    text: str,
    line_start: int,
    *,
    has_explicit_sep: bool = False,
) -> bool:
    """Return True if the line before `line_start` ends mid-sentence (comma or lowercase word without punctuation)."""
    if line_start <= 0:
        return False
    prev_chunk = text[:line_start].rstrip("\r\n")
    last_nl = prev_chunk.rfind("\n")
    prev_line = prev_chunk[last_nl + 1 :].strip()
    if not prev_line:
        return False
    if prev_line.endswith(","):
        return True
    if _TRAILING_CONTINUATION_WORD_RE.search(prev_line):
        return True
    # If previous line is an all-caps header (e.g. "ARTICLE IV" or "EVENTS OF DEFAULT"), next line is a section header
    if prev_line.isupper():
        return False
    if has_explicit_sep:
        return False
    # If previous line does not end with sentence-closing punctuation (., :, ;, !, ?, ), ])
    if prev_line[-1] not in ".:;!?)]\"'":
        return True
    return False


def _split_into_sections(text: str) -> dict[str, _ParsedSection]:
    """Map normalized section_key -> _ParsedSection (ignoring inline cross-references)."""
    if not text:
        return {}

    candidates: list[tuple[int, int, str, str]] = []
    occupied_starts: set[int] = set()

    for m in _NAMED_SECTION_HEADER_RE.finditer(text):
        sep = m.group("sep")
        if _is_wrapped_sentence_continuation(text, m.start(), has_explicit_sep=bool(sep)):
            continue
        kw = m.group("kw").rstrip(".")
        num = m.group("num")
        raw_title = (m.group("title") or m.group("bare_title") or "").strip()

        # Without explicit header punctuation (: . - —), a title starting with a lowercase word is an inline verb ("Section 7.01 requires...")
        if not sep and raw_title and raw_title[0].islower():
            continue
        # Filter out inline cross-references like "Section 7.01 above..." at start of wrapped line
        if raw_title and _CROSS_REF_CONTINUATION_RE.match(raw_title):
            continue
        # Ensure single-token Roman numeral match is uppercase unless explicit separator is present
        if not num[0].isdigit() and not sep and not num.isupper() and len(num) <= 4:
            continue

        sec_id = f"{kw} {num}"
        candidates.append((m.start(), m.end(), sec_id, raw_title))
        occupied_starts.add(m.start())

    for m in _BARE_NUMBER_HEADER_RE.finditer(text):
        has_sep = bool(m.group("ord_sep")) or bool((m.group("num_sep") or "").strip())
        if m.start() in occupied_starts or _is_wrapped_sentence_continuation(
            text, m.start(), has_explicit_sep=has_sep
        ):
            continue
        num = m.group("num")
        ord_tok = m.group("ord")
        raw_title = (m.group("title") or "").strip()
        if raw_title and (raw_title[0].islower() or _CROSS_REF_CONTINUATION_RE.match(raw_title)):
            continue
        if num:
            sec_id = f"Section {num}"
        elif ord_tok and ord_tok.isdigit():
            sec_id = f"Section {ord_tok}"
        else:
            sec_id = f"Cláusula {ord_tok}"
        candidates.append((m.start(), m.end(), sec_id, raw_title))

    if not candidates:
        return {}

    candidates.sort(key=lambda item: item[0])
    # Deduplicate overlapping start offsets while preserving section order.
    dedup_candidates: list[tuple[int, int, str, str]] = []
    for cand in candidates:
        if dedup_candidates and cand[0] <= dedup_candidates[-1][0]:
            continue
        dedup_candidates.append(cand)

    sections: dict[str, _ParsedSection] = {}
    occ_counts: dict[str, int] = {}

    first_start = dedup_candidates[0][0]
    if first_start > 0 and text[:first_start].strip():
        pre_body = text[:first_start].strip()
        sections["unsegmented_remainder"] = _ParsedSection(
            key="unsegmented_remainder",
            label="UNSEGMENTED_REMAINDER: Pre-Header Remainder",
            title_key="unsegmented_remainder",
            body=pre_body,
            body_without_header=pre_body,
            start=0,
            end=first_start,
        )

    for idx, (start, _hdr_end, sec_id, raw_title) in enumerate(dedup_candidates):
        cov_start = 0 if (idx == 0 and "unsegmented_remainder" not in sections) else start
        end = dedup_candidates[idx + 1][0] if idx + 1 < len(dedup_candidates) else len(text)
        clean_id = re.sub(r"\s+", " ", sec_id.strip())
        short_title = raw_title.split(".")[0].strip() if "." in raw_title else raw_title.rstrip(".")
        label = f"{clean_id}: {short_title}" if short_title else clean_id
        base_key = normalize_key(clean_id)
        occ = occ_counts.get(base_key, 0) + 1
        occ_counts[base_key] = occ
        key = base_key if occ == 1 else f"{base_key}__occ{occ}"
        body = text[start:end].strip()
        body_no_hdr = _strip_leading_header_id(body, clean_id)
        title_key = normalize_key(short_title)

        parsed = _ParsedSection(
            key=key,
            label=label if occ == 1 else f"{label} (occurrence {occ})",
            title_key=title_key,
            body=body,
            body_without_header=body_no_hdr,
            start=cov_start,
            end=end,
        )
        sections[key] = parsed

    return sections


def _compute_char_coverage(
    text_len: int,
    segments: Mapping[str, _ParsedSection],
) -> tuple[float, tuple[tuple[int, int], ...]]:
    """Compute fraction of `[0, text_len)` covered by *segments* and return any uncovered gaps."""
    if text_len <= 0:
        return 1.0, ()
    if not segments:
        return 0.0, ((0, text_len),)
    intervals = sorted((max(0, min(text_len, seg.start)), max(0, min(text_len, seg.end))) for seg in segments.values())
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    uncovered: list[tuple[int, int]] = []
    cursor = 0
    covered_len = 0
    for s, e in merged:
        if s > cursor:
            uncovered.append((cursor, s))
        covered_len += e - s
        cursor = e
    if cursor < text_len:
        uncovered.append((cursor, text_len))
    return covered_len / text_len, tuple(uncovered)


def _partition_document_segments(
    text: str,
    *,
    unsegmented_key: str = "document_body",
    unsegmented_label: str = "Document Body (Unsegmented)",
) -> tuple[dict[str, _ParsedSection], float, tuple[tuple[int, int], ...]]:
    """Partition *text* so 100% of characters fall into a named section or residual block."""
    if not text:
        return {}, 1.0, ()
    sections = _split_into_sections(text)
    if not sections:
        body = text.strip()
        sections = {
            unsegmented_key: _ParsedSection(
                key=unsegmented_key,
                label=unsegmented_label,
                title_key=unsegmented_key,
                body=body,
                body_without_header=body,
                start=0,
                end=len(text),
            )
        }
    cov, uncovered = _compute_char_coverage(len(text), sections)
    return sections, cov, uncovered


def _quantity_tokens(body: str) -> set[str]:
    tokens: set[str] = set()
    for q in extract_quantities(body):
        if q.unit != "NONE" or abs(q.magnitude) >= 100:
            norm_mag = q.magnitude.normalize()
            mag_str = format(norm_mag, "f")
            if "." in mag_str:
                mag_str = mag_str.rstrip("0").rstrip(".")
            tokens.add(f"{mag_str}:{q.unit}")
    for d in extract_dates(body):
        tokens.add(f"DATE:{d}")
    return tokens


def _map_norm_span(
    starts: list[int],
    ends: list[int],
    norm_span: tuple[int, int] | None,
    raw_len: int,
) -> tuple[int, int] | None:
    if norm_span is None:
        return None
    s, e = norm_span
    if not starts or not ends or e <= s:
        return (0, raw_len)
    mapped_s = starts[min(max(s, 0), len(starts) - 1)]
    mapped_e = ends[min(max(e - 1, 0), len(ends) - 1)]
    return (mapped_s, max(mapped_s, mapped_e))


def _exclusion_symmetry_report(
    p_furn: tuple[dict[str, Any], ...],
    s_furn: tuple[dict[str, Any], ...],
) -> tuple[bool, bool, dict[str, Any]]:
    """Verify whether normalized excluded furniture records match across both versions."""
    p_norms = sorted(_canonicalize_layout_text(str(x.get("text", ""))) for x in p_furn if str(x.get("text", "")).strip())
    s_norms = sorted(_canonicalize_layout_text(str(x.get("text", ""))) for x in s_furn if str(x.get("text", "")).strip())
    p_set = sorted(set(p_norms))
    s_set = sorted(set(s_norms))
    p_only = [k for k in p_set if k not in s_set]
    s_only = [k for k in s_set if k not in p_set]
    symmetric = (p_norms == s_norms)
    return symmetric, (not symmetric), {
        "symmetric": symmetric,
        "asymmetry_detected": not symmetric,
        "primary_excluded_count": len(p_norms),
        "secondary_excluded_count": len(s_norms),
        "primary_excluded_only": p_only,
        "secondary_excluded_only": s_only,
    }


def extract_divergent_sections_with_coverage(
    primary: Source,
    secondary: Source,
    *,
    fail_on_incomplete_coverage: bool = True,
) -> SectionComparisonReport:
    """Compare *primary* and *secondary* guaranteeing 100% character coverage and symmetric normalization."""
    (p_norm, p_starts, p_ends, p_furn), (s_norm, s_starts, s_ends, s_furn) = _normalize_pair_with_maps(
        primary.text,
        secondary.text,
        apply_furniture_filter=True,
    )
    excl_sym, excl_asym, excl_check = _exclusion_symmetry_report(p_furn, s_furn)

    p_has_headers = bool(_split_into_sections(p_norm))
    s_has_headers = bool(_split_into_sections(s_norm))

    prim_secs, p_cov, p_uncov = _partition_document_segments(
        p_norm,
        unsegmented_key="document_body" if (not p_has_headers and not s_has_headers) else "unsegmented_remainder",
        unsegmented_label="Document Body (Unsegmented)"
        if (not p_has_headers and not s_has_headers)
        else "UNSEGMENTED_REMAINDER: Unsegmented Document",
    )
    sec_secs, s_cov, s_uncov = _partition_document_segments(
        s_norm,
        unsegmented_key="document_body" if (not p_has_headers and not s_has_headers) else "unsegmented_remainder",
        unsegmented_label="Document Body (Unsegmented)"
        if (not p_has_headers and not s_has_headers)
        else "UNSEGMENTED_REMAINDER: Unsegmented Document",
    )

    divergences: list[SectionDivergence] = []

    if fail_on_incomplete_coverage:
        for us, ue in p_uncov:
            seg = p_norm[us:ue].strip()
            if seg:
                divergences.append(
                    SectionDivergence(
                        section_key="unsegmented_remainder",
                        section_label="UNSEGMENTED_REMAINDER: Uncovered Primary Span",
                        primary_source_id=primary.id,
                        secondary_source_id=secondary.id,
                        primary_quantities=tuple(sorted(_quantity_tokens(seg))),
                        secondary_quantities=(),
                        reason=f"Fail-closed uncovered primary span [{us}, {ue}] in '{primary.id}'.",
                        primary_span=_map_norm_span(p_starts, p_ends, (us, ue), len(primary.text)),
                        secondary_span=None,
                    )
                )
        for us, ue in s_uncov:
            seg = s_norm[us:ue].strip()
            if seg:
                divergences.append(
                    SectionDivergence(
                        section_key="unsegmented_remainder",
                        section_label="UNSEGMENTED_REMAINDER: Uncovered Secondary Span",
                        primary_source_id=primary.id,
                        secondary_source_id=secondary.id,
                        primary_quantities=(),
                        secondary_quantities=tuple(sorted(_quantity_tokens(seg))),
                        reason=f"Fail-closed uncovered secondary span [{us}, {ue}] in '{secondary.id}'.",
                        primary_span=None,
                        secondary_span=_map_norm_span(s_starts, s_ends, (us, ue), len(secondary.text)),
                    )
                )

    p_canon = _canonicalize_layout_text(p_norm)
    s_canon = _canonicalize_layout_text(s_norm)
    p_norm_all_key = normalize_key(p_canon)
    s_norm_all_key = normalize_key(s_canon)
    if (
        not p_uncov
        and not s_uncov
        and p_canon == s_canon
        and _quantity_tokens(p_canon) == _quantity_tokens(s_canon)
    ):
        return SectionComparisonReport(
            divergences=tuple(divergences),
            primary_coverage=p_cov,
            secondary_coverage=s_cov,
            primary_uncovered_spans=p_uncov,
            secondary_uncovered_spans=s_uncov,
            primary_excluded_furniture=p_furn,
            secondary_excluded_furniture=s_furn,
            exclusion_symmetric=excl_sym,
            exclusion_asymmetry=excl_asym,
            exclusion_symmetry_check=excl_check,
        )

    # Unsegmented pair (both lack headers)
    if not p_has_headers and not s_has_headers:
        p_quants = _quantity_tokens(p_canon)
        s_quants = _quantity_tokens(s_canon)
        if p_quants != s_quants or p_canon != s_canon:
            divergences.append(
                SectionDivergence(
                    section_key="document_body",
                    section_label="Document Body (Unsegmented)",
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=tuple(sorted(s_quants)),
                    reason=(
                        f"Unsegmented document divergence between '{primary.id}' and '{secondary.id}'."
                    ),
                    primary_span=(0, len(primary.text)),
                    secondary_span=(0, len(secondary.text)),
                )
            )
        return SectionComparisonReport(
            divergences=tuple(divergences),
            primary_coverage=p_cov,
            secondary_coverage=s_cov,
            primary_uncovered_spans=p_uncov,
            secondary_uncovered_spans=s_uncov,
            primary_excluded_furniture=p_furn,
            secondary_excluded_furniture=s_furn,
            exclusion_symmetric=excl_sym,
            exclusion_asymmetry=excl_asym,
            exclusion_symmetry_check=excl_check,
        )

    matched_sec_keys: set[str] = set()
    p_remainder_only = not p_has_headers
    s_remainder_only = not s_has_headers

    for key in sorted(prim_secs.keys()):
        if key == "unsegmented_remainder" and p_remainder_only:
            # Handled after section alignment as fail-closed fallback only if no divergences
            # were emitted while whole-document text/quantities still differ.
            continue
        p_sec = prim_secs[key]
        p_quants = _quantity_tokens(p_sec.body)

        s_sec: _ParsedSection | None = None
        if key in sec_secs:
            s_sec = sec_secs[key]
            matched_sec_keys.add(key)
        elif key != "unsegmented_remainder":
            # Check for renumbered section with identical title or identical canonical body content
            p_b_canon = _canonicalize_layout_text(p_sec.body_without_header)
            for cand_key, cand_sec in sec_secs.items():
                if cand_key in matched_sec_keys or cand_key in prim_secs or cand_key == "unsegmented_remainder":
                    continue
                same_body = p_b_canon == _canonicalize_layout_text(cand_sec.body_without_header)
                same_title = bool(p_sec.title_key) and p_sec.title_key == cand_sec.title_key
                if same_body or same_title:
                    s_sec = cand_sec
                    matched_sec_keys.add(cand_key)
                    break

        if s_sec is None:
            p_sec_canon = _canonicalize_layout_text(p_sec.body)
            p_sec_body_canon = _canonicalize_layout_text(p_sec.body_without_header)
            # Layout-independent rule (P0): a section whose normalized body is intact in counterpart is NOT missing.
            if (
                p_sec_canon
                and (
                    (len(p_sec_canon) >= 16 and p_sec_canon in s_canon)
                    or (len(p_sec_body_canon) >= 16 and p_sec_body_canon in s_canon)
                )
                and _quantity_tokens(p_sec_canon).issubset(_quantity_tokens(s_canon))
            ):
                continue
            divergences.append(
                SectionDivergence(
                    section_key=key,
                    section_label=p_sec.label,
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=(),
                    reason=f"Section '{p_sec.label}' present in '{primary.id}' is missing from '{secondary.id}'.",
                    primary_span=_map_norm_span(p_starts, p_ends, (p_sec.start, p_sec.end), len(primary.text)),
                    secondary_span=None,
                )
            )
            continue

        s_quants = _quantity_tokens(s_sec.body)
        if p_quants != s_quants:
            divergences.append(
                SectionDivergence(
                    section_key=key,
                    section_label=p_sec.label,
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=tuple(sorted(s_quants)),
                    reason=(
                        f"Quantitative divergence in '{p_sec.label}': "
                        f"{primary.id}={sorted(p_quants)} vs {secondary.id}={sorted(s_quants)}."
                    ),
                    primary_span=_map_norm_span(p_starts, p_ends, (p_sec.start, p_sec.end), len(primary.text)),
                    secondary_span=_map_norm_span(s_starts, s_ends, (s_sec.start, s_sec.end), len(secondary.text)),
                )
            )
        elif _canonicalize_layout_text(p_sec.body_without_header) != _canonicalize_layout_text(s_sec.body_without_header):
            divergences.append(
                SectionDivergence(
                    section_key=key,
                    section_label=p_sec.label,
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=tuple(sorted(s_quants)),
                    reason=f"Textual/structural divergence in '{p_sec.label}' between '{primary.id}' and '{secondary.id}'.",
                    primary_span=_map_norm_span(p_starts, p_ends, (p_sec.start, p_sec.end), len(primary.text)),
                    secondary_span=_map_norm_span(s_starts, s_ends, (s_sec.start, s_sec.end), len(secondary.text)),
                )
            )

    # Also detect sections or residual blocks present in the secondary document
    for sec_key in sorted(sec_secs.keys()):
        if sec_key in matched_sec_keys:
            continue
        if sec_key == "unsegmented_remainder" and s_remainder_only:
            continue
        s_sec = sec_secs[sec_key]
        s_sec_canon = _canonicalize_layout_text(s_sec.body)
        s_sec_body_canon = _canonicalize_layout_text(s_sec.body_without_header)
        if (
            s_sec_canon
            and (
                (len(s_sec_canon) >= 16 and s_sec_canon in p_canon)
                or (len(s_sec_body_canon) >= 16 and s_sec_body_canon in p_canon)
            )
            and _quantity_tokens(s_sec_canon).issubset(_quantity_tokens(p_canon))
        ):
            continue
        s_quants = _quantity_tokens(s_sec.body)
        divergences.append(
            SectionDivergence(
                section_key=sec_key,
                section_label=s_sec.label,
                primary_source_id=primary.id,
                secondary_source_id=secondary.id,
                primary_quantities=(),
                secondary_quantities=tuple(sorted(s_quants)),
                reason=(
                    f"Section '{s_sec.label}' added in '{secondary.id}' is absent from governing '{primary.id}'."
                ),
                primary_span=None,
                secondary_span=_map_norm_span(s_starts, s_ends, (s_sec.start, s_sec.end), len(secondary.text)),
            )
        )

    # Fail-closed guard for asymmetric pairs (one doc has headers, counterpart is all-prose)
    if (p_remainder_only or s_remainder_only) and not divergences:
        p_all_quants = _quantity_tokens(p_canon)
        s_all_quants = _quantity_tokens(s_canon)
        if p_all_quants != s_all_quants or p_canon != s_canon or p_norm_all_key != s_norm_all_key:
            divergences.append(
                SectionDivergence(
                    section_key="unsegmented_remainder",
                    section_label="UNSEGMENTED_REMAINDER: Asymmetric Unsegmented Divergence",
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_all_quants)),
                    secondary_quantities=tuple(sorted(s_all_quants)),
                    reason=(
                        f"Asymmetric unsegmented document divergence between '{primary.id}' and '{secondary.id}'."
                    ),
                    primary_span=(0, len(primary.text)),
                    secondary_span=(0, len(secondary.text)),
                )
            )

    return SectionComparisonReport(
        divergences=tuple(divergences),
        primary_coverage=p_cov,
        secondary_coverage=s_cov,
        primary_uncovered_spans=p_uncov,
        secondary_uncovered_spans=s_uncov,
        primary_excluded_furniture=p_furn,
        secondary_excluded_furniture=s_furn,
        exclusion_symmetric=excl_sym,
        exclusion_asymmetry=excl_asym,
        exclusion_symmetry_check=excl_check,
    )


def extract_divergent_sections(
    primary: Source,
    secondary: Source,
) -> list[SectionDivergence]:
    """Identify sections where *secondary* diverges quantitatively, structurally, or adds/omits provisions."""
    return list(extract_divergent_sections_with_coverage(primary, secondary).divergences)


def build_comparison_policy(
    primary: Source,
    secondary: Source,
    *,
    policy_id: str = "document_comparison",
    min_tier_rank: int | None = None,
) -> tuple[Policy, list[SectionDivergence]]:
    """Build a deterministic Policy requiring the agent to audit every divergent section
    and ground governing values in the higher-authority primary instrument."""
    divergences = extract_divergent_sections(primary, secondary)
    required_preds = [d.section_key for d in divergences]
    effective_min_rank = min_tier_rank if min_tier_rank is not None else primary.rank
    policy = Policy(
        policy_id=policy_id,
        min_tier_rank=effective_min_rank,
        required_predicates=required_preds,
        required_source_ids=[primary.id],
        required_value_tokens={d.section_key: sorted(set(d.primary_quantities) - set(d.secondary_quantities))
                               for d in divergences if set(d.primary_quantities) - set(d.secondary_quantities)},
    )
    return policy, divergences
