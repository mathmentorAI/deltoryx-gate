"""Deterministic section-level divergence detector for multi-document audits.

Discovers sections where a secondary document (e.g., Offering Memorandum or Counterparty Markup)
diverges quantitatively or structurally from an authoritative baseline (e.g., Indenture or MSA),
automatically generating completeness predicates for Policy.required_predicates.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from claimlayer.models import Policy, Source, normalize_key
from claimlayer.verifiers.quote_quantity import extract_dates, extract_quantities

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

# Bare numbered or ordinal headers at line start: "4.01 Events of Default", "PRIMERA.- Objeto", "1.- Objeto", "1) Purpose"
_BARE_NUMBER_HEADER_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}\s*)?"
    r"(?:"
    r"(?P<num>\d+\.\d+(?:\.\d+)*)(?P<num_sep>\s*[:.\-—)]\s*|\s+)"
    r"|(?P<ord>[A-ZÁÉÍÓÚÑ]{4,18}|\d+)(?P<ord_sep>\s*(?:\.\s*[-—:]|[:—)]))\s*"
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
    sections: dict[str, _ParsedSection] = {}

    for idx, (start, _hdr_end, sec_id, raw_title) in enumerate(candidates):
        end = candidates[idx + 1][0] if idx + 1 < len(candidates) else len(text)
        clean_id = re.sub(r"\s+", " ", sec_id.strip())
        # Extract short title (up to first period if body follows on same line)
        short_title = raw_title.split(".")[0].strip() if "." in raw_title else raw_title.rstrip(".")
        label = f"{clean_id}: {short_title}" if short_title else clean_id
        key = normalize_key(clean_id)
        body = text[start:end].strip()
        body_no_hdr = _strip_leading_header_id(body, clean_id)
        title_key = normalize_key(short_title)

        parsed = _ParsedSection(
            key=key,
            label=label,
            title_key=title_key,
            body=body,
            body_without_header=body_no_hdr,
            start=start,
            end=end,
        )
        # If duplicate key appears, keep the longer substantive section (never overwrite with short reference)
        existing = sections.get(key)
        if existing is None or len(parsed.body) > len(existing.body):
            sections[key] = parsed

    return sections


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


def extract_divergent_sections(
    primary: Source,
    secondary: Source,
) -> list[SectionDivergence]:
    """Identify sections where *secondary* diverges quantitatively, structurally, or adds/omits provisions."""
    prim_secs = _split_into_sections(primary.text)
    sec_secs = _split_into_sections(secondary.text)
    divergences: list[SectionDivergence] = []

    # Fail-closed when documents have no structured headers
    if not prim_secs and not sec_secs:
        p_quants = _quantity_tokens(primary.text)
        s_quants = _quantity_tokens(secondary.text)
        if p_quants != s_quants or normalize_key(primary.text) != normalize_key(secondary.text):
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
        return divergences

    matched_sec_keys: set[str] = set()

    for key in sorted(prim_secs.keys()):
        p_sec = prim_secs[key]
        p_quants = _quantity_tokens(p_sec.body)

        s_sec: _ParsedSection | None = None
        if key in sec_secs:
            s_sec = sec_secs[key]
            matched_sec_keys.add(key)
        else:
            # Check for renumbered section with identical title or identical body content
            for cand_key, cand_sec in sec_secs.items():
                if cand_key in matched_sec_keys or cand_key in prim_secs:
                    continue
                same_body = normalize_key(p_sec.body_without_header) == normalize_key(
                    cand_sec.body_without_header
                )
                same_title = bool(p_sec.title_key) and p_sec.title_key == cand_sec.title_key
                if same_body or same_title:
                    s_sec = cand_sec
                    matched_sec_keys.add(cand_key)
                    break

        if s_sec is None:
            divergences.append(
                SectionDivergence(
                    section_key=key,
                    section_label=p_sec.label,
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=(),
                    reason=f"Section '{p_sec.label}' present in '{primary.id}' is missing from '{secondary.id}'.",
                    primary_span=(p_sec.start, p_sec.end),
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
                    primary_span=(p_sec.start, p_sec.end),
                    secondary_span=(s_sec.start, s_sec.end),
                )
            )
        elif normalize_key(p_sec.body_without_header) != normalize_key(s_sec.body_without_header):
            divergences.append(
                SectionDivergence(
                    section_key=key,
                    section_label=p_sec.label,
                    primary_source_id=primary.id,
                    secondary_source_id=secondary.id,
                    primary_quantities=tuple(sorted(p_quants)),
                    secondary_quantities=tuple(sorted(s_quants)),
                    reason=f"Textual/structural divergence in '{p_sec.label}' between '{primary.id}' and '{secondary.id}'.",
                    primary_span=(p_sec.start, p_sec.end),
                    secondary_span=(s_sec.start, s_sec.end),
                )
            )

    # Also detect sections added only in the secondary document
    for sec_key in sorted(sec_secs.keys()):
        if sec_key in matched_sec_keys:
            continue
        s_sec = sec_secs[sec_key]
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
                secondary_span=(s_sec.start, s_sec.end),
            )
        )

    return divergences


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
