"""Corpus-derived verification obligations and reference register (v4).

Computes deterministic verification obligations directly from admitted
``EvidenceLayer`` units and the domain ``Vocabulary`` before evaluating any
agent-supplied graph, assertions, or metadata flags.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import deque
from decimal import Decimal, InvalidOperation
import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..models import Source
from ..semantic.models import SourceDocument, canonical, digest
from ..verifiers.diff import build_comparison_policy
from ..verifiers.quote_quantity import (
    _SECTION_HEADER_ONLY_RE,
    _STRUCTURAL_ID_RE,
    extract_dates_with_spans,
    extract_quantities,
    segment_sentences_with_spans,
)
from .domain_profile import DomainProfile
from .engine import EvidenceLayer
from .justification import VerificationObligation
from .vocabulary import Vocabulary, label_key, value_relation


@dataclass(frozen=True)
class ReferenceDivergenceCollection:
    """Single deterministic divergence computation shared across production and measurement."""

    divergences: tuple[Any, ...] = ()
    compared_document_ids: tuple[str, ...] = ()
    governing_document_id: str | None = None
    excluded_documents: tuple[Mapping[str, Any], ...] = ()
    domain_profile_fingerprint: str | None = None
    document_texts: Mapping[str, str] = field(default_factory=dict)
    document_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _doc_attr(doc: Any, name: str, default: Any = "") -> Any:
    if isinstance(doc, Mapping):
        return doc.get(name, default)
    return getattr(doc, name, default)


def collect_reference_divergences(
    corpus_documents: Sequence[Any] | Mapping[str, Any],
    domain_profile: DomainProfile | None = None,
    *,
    min_tier_rank: int | None = None,
) -> ReferenceDivergenceCollection:
    """Collect cross-document divergences under explicit DomainProfile comparison exclusions.

    Both ``evaluate_securitization_candidate`` (production) and
    ``benchmarks/measure_reader_recall.py`` (measurement) MUST call this helper so
    production and offline recall measure the exact same document comparison set.
    """
    raw_docs = list(corpus_documents.values()) if isinstance(corpus_documents, Mapping) else list(corpus_documents or ())
    if not raw_docs:
        fp = domain_profile.fingerprint() if domain_profile is not None else None
        return ReferenceDivergenceCollection(domain_profile_fingerprint=fp)

    fp = domain_profile.fingerprint() if domain_profile is not None else None
    profile_id = domain_profile.profile_id if domain_profile is not None else None
    excluded_roles = dict(getattr(getattr(domain_profile, "policy", None), "excluded_comparison_roles", None) or {})

    doc_texts: dict[str, str] = {}
    doc_meta: dict[str, dict[str, Any]] = {}
    included_docs: list[Any] = []
    candidate_excluded: list[dict[str, Any]] = []
    for doc in raw_docs:
        did = str(
            _doc_attr(doc, "filename", "")
            or _doc_attr(doc, "document_id", "")
            or _doc_attr(doc, "id", "")
        )
        text = str(_doc_attr(doc, "text", "") or "")
        role = str(_doc_attr(doc, "role", "") or "").strip()
        title = str(_doc_attr(doc, "title", "") or "").strip()
        if did:
            doc_texts[did] = text
            doc_meta[did] = {"role": role, "title": title}
        if role and role in excluded_roles:
            candidate_excluded.append({
                "document_id": did,
                "role": role,
                "authority_rank": int(_doc_attr(doc, "authority_rank", 50) or 50),
                "excluding_rule": f"PolicyConfiguration.excluded_comparison_roles[{role!r}]: {excluded_roles[role]}",
                "domain_profile_id": profile_id,
                "domain_profile_fingerprint": fp,
            })
        else:
            included_docs.append(doc)

    # Preserve comparison safety if exclusions would leave fewer than two docs.
    if len(included_docs) >= 2:
        comp_docs = included_docs
        excluded_records = tuple(candidate_excluded)
    else:
        comp_docs = raw_docs
        excluded_records = ()

    compared_ids = tuple(
        str(_doc_attr(d, "filename", "") or _doc_attr(d, "document_id", "") or _doc_attr(d, "id", ""))
        for d in comp_docs
    )
    if len(comp_docs) < 2:
        return ReferenceDivergenceCollection(
            compared_document_ids=compared_ids,
            governing_document_id=compared_ids[0] if compared_ids else None,
            excluded_documents=excluded_records,
            domain_profile_fingerprint=fp,
            document_texts=doc_texts,
            document_metadata=doc_meta,
        )

    gov = max(comp_docs, key=lambda d: int(_doc_attr(d, "authority_rank", 50) or 50))
    gov_id = str(_doc_attr(gov, "filename", "") or _doc_attr(gov, "document_id", "") or _doc_attr(gov, "id", ""))
    gov_rank = int(_doc_attr(gov, "authority_rank", 50) or 50)
    gov_tier = str(_doc_attr(gov, "authority_tier", "") or _doc_attr(gov, "tier", "") or "authoritative")
    gov_key = str(_doc_attr(gov, "doc_key", "") or gov_id)
    gov_source = Source(
        id=gov_id,
        text=str(_doc_attr(gov, "text", "") or ""),
        tier=gov_tier,
        authority_rank=gov_rank,
        title=f"{gov_key}: {gov_id}",
    )

    eff_min_rank = gov_rank if min_tier_rank is None else min_tier_rank
    all_divs: list[Any] = []
    for doc in comp_docs:
        sub_id = str(_doc_attr(doc, "filename", "") or _doc_attr(doc, "document_id", "") or _doc_attr(doc, "id", ""))
        if sub_id == gov_id:
            continue
        sub_rank = int(_doc_attr(doc, "authority_rank", 50) or 50)
        sub_tier = str(_doc_attr(doc, "authority_tier", "") or _doc_attr(doc, "tier", "") or "supplementary")
        sub_key = str(_doc_attr(doc, "doc_key", "") or sub_id)
        sub_source = Source(
            id=sub_id,
            text=str(_doc_attr(doc, "text", "") or ""),
            tier=sub_tier,
            authority_rank=sub_rank,
            title=f"{sub_key}: {sub_id}",
        )
        _, divs = build_comparison_policy(
            gov_source,
            sub_source,
            policy_id=f"ref_cmp::{gov_id}::{sub_id}",
            min_tier_rank=eff_min_rank,
        )
        all_divs.extend(divs)

    return ReferenceDivergenceCollection(
        divergences=tuple(all_divs),
        compared_document_ids=compared_ids,
        governing_document_id=gov_id,
        excluded_documents=excluded_records,
        domain_profile_fingerprint=fp,
        document_texts=doc_texts,
        document_metadata=doc_meta,
    )


@dataclass(frozen=True)
class TargetedRereadConfig:
    """Budget and execution controls for blind gap-targeted re-reading."""

    enabled: bool = True
    max_attempts_per_document: int = 3
    max_total_attempts: int = 12


@dataclass(frozen=True)
class CorpusReferenceRegister:
    """Reference register computed solely from admitted corpus evidence."""

    available: bool
    entity: str | None = None
    required_predicates: tuple[str, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    document_ids: tuple[str, ...] = ()
    unit_count: int = 0
    has_material_conflicts: bool = False
    has_authority_overrides: bool = False
    reason: str = ""
    incomplete: bool = False
    incomplete_documents: tuple[str, ...] = ()
    document_statuses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    gap_divergences: tuple[Mapping[str, Any], ...] = ()
    section_divergences: tuple[Mapping[str, Any], ...] = ()
    admitted_unit_refs: tuple[Mapping[str, Any], ...] = ()
    excluded_documents: tuple[Mapping[str, Any], ...] = ()
    domain_profile_fingerprint: str | None = None
    reread_provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(o.obligation_id for o in self.obligations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "entity": self.entity,
            "required_predicates": list(self.required_predicates),
            "obligation_ids": list(self.obligation_ids),
            "document_ids": list(self.document_ids),
            "unit_count": self.unit_count,
            "has_material_conflicts": self.has_material_conflicts,
            "has_authority_overrides": self.has_authority_overrides,
            "reason": self.reason,
            "incomplete": self.incomplete,
            "incomplete_documents": list(self.incomplete_documents),
            "document_statuses": {k: dict(v) for k, v in self.document_statuses.items()},
            "gap_divergences": [dict(g) for g in self.gap_divergences],
            "section_divergences": [dict(s) for s in self.section_divergences],
            "excluded_documents": [dict(e) for e in self.excluded_documents],
            "domain_profile_fingerprint": self.domain_profile_fingerprint,
            "reread_provenance": dict(self.reread_provenance),
        }


def _unit_authority_rank(layer: EvidenceLayer, unit: Any) -> int:
    ranks: list[int] = []
    for ref in getattr(unit, "evidence", ()) or ():
        try:
            doc = layer._document(ref.document_id, ref.revision, current=False)
            ranks.append(int(doc.authority_rank))
        except Exception:
            continue
    return max(ranks) if ranks else 0


def _unit_spans(unit: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for ref in getattr(unit, "evidence", ()) or ():
        spans.append({
            "document_id": str(ref.document_id),
            "revision": str(ref.revision),
            "start": int(ref.start),
            "end": int(ref.end),
            "quote": str(ref.quote),
        })
    return spans


def _units_incompatible(vocabulary: Vocabulary, spec: Any, u1: Any, u2: Any) -> bool:
    p1 = u1.proposition
    p2 = u2.proposition
    if u1.assertion_key == u2.assertion_key:
        return False
    u1_unit = p1.get("unit")
    u2_unit = p2.get("unit")
    if u1_unit is not None and u2_unit is not None:
        if vocabulary.units.dimension_of(u1_unit) != vocabulary.units.dimension_of(u2_unit):
            # Incompatible counting conventions on same predicate (e.g. calendar_days vs business_days)
            if p1.get("predicate") == p2.get("predicate") and u1_unit != u2_unit:
                return True
            return False
    rel = value_relation(spec.value, p1.get("value"), p2.get("value"))
    if rel == "equal":
        return p1.get("polarity") != p2.get("polarity")
    return rel == "incompatible"


def _spans_overlap(
    doc_a: str, span_a: tuple[int, int] | None, doc_b: str, span_b: tuple[int, int] | None
) -> bool:
    if not doc_a or doc_a != doc_b:
        return False
    if span_a is None or span_b is None:
        return True
    s1, e1 = int(span_a[0]), int(span_a[1])
    s2, e2 = int(span_b[0]), int(span_b[1])
    return s1 < e2 and s2 < e1


def _obligation_overlaps_doc_span(
    ob: VerificationObligation, doc_id: str, span: tuple[int, int] | None
) -> bool:
    spans = (ob.parameters or {}).get("evidence_spans") or ()
    for sp in spans:
        sp_doc = str(sp.get("document_id", ""))
        sp_range = (
            (int(sp["start"]), int(sp["end"]))
            if sp.get("start") is not None and sp.get("end") is not None
            else None
        )
        if _spans_overlap(sp_doc, sp_range, doc_id, span):
            return True
    return False


_DEAL_SERIES_YEAR_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9&\-]{1,}\s+(?:19|20)\d{2}-\d+\b"
)
_MULTI_TRANCHE_RE = re.compile(
    r"\b(?:Class|Classes|Tranche|Tranches|Series|Tier)\s+"
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?(?:\s*(?:,|and|&|/)\s*(?:Class\s+)?[A-Za-z0-9\-]+)+",
    re.IGNORECASE,
)
_STEP_RANGE_RE = re.compile(
    r"\b(?:Steps?|Priorities)\s+(?P<n1>\d+)\s*(?:[-–]|to|through|and)\s*(?P<n2>\d+)\b",
    re.IGNORECASE,
)
_STEP_SINGLE_RE = re.compile(
    r"\b(?:Step|Priority)\s*(?:#\s*)?(?P<num>\d+)\b",
    re.IGNORECASE,
)
_SENTENCE_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "in", "into", "is", "it", "its", "of", "on", "or", "per", "shall", "that",
    "the", "their", "this", "to", "was", "were", "will", "with", "any", "all",
})


def _mask_preserving_len(match: re.Match[str]) -> str:
    return "~" * (match.end() - match.start())


def _fmt_decimal_compact(val: Decimal) -> str:
    txt = format(val.normalize(), "f")
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    return txt or "0"


def _spans_intersect(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return int(a[0]) < int(b[1]) and int(b[0]) < int(a[1])


def _extract_sentence_quantity_tokens(
    sentence: str, sentence_start: int = 0
) -> list[dict[str, Any]]:
    """Extract normalized quantity, step/priority, and date tokens with document offsets.

    Uses v3's ``extract_quantities`` and ``extract_dates_with_spans`` while masking
    structural identifiers (section/clause/series/tranche/deal-year references).
    """
    if not sentence:
        return []

    masked = _STRUCTURAL_ID_RE.sub(_mask_preserving_len, sentence)
    masked = _DEAL_SERIES_YEAR_RE.sub(_mask_preserving_len, masked)
    masked = _MULTI_TRANCHE_RE.sub(_mask_preserving_len, masked)

    tokens: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    def _overlaps_occupied(s: int, e: int) -> bool:
        return any(_spans_intersect((s, e), occ) for occ in occupied)

    # 1. Explicit step/priority ranges and numbers ("Step 12", "Steps 8-12", "Priority 8")
    for m in _STEP_RANGE_RE.finditer(masked):
        s1, e1 = m.start("n1"), m.end("n1")
        s2, e2 = m.start("n2"), m.end("n2")
        if not _overlaps_occupied(m.start(), m.end()):
            occupied.append((m.start(), m.end()))
            tokens.append({
                "token": f"STEP:{int(m.group('n1'))}",
                "span": (sentence_start + s1, sentence_start + e1),
                "raw": m.group("n1"),
            })
            tokens.append({
                "token": f"STEP:{int(m.group('n2'))}",
                "span": (sentence_start + s2, sentence_start + e2),
                "raw": m.group("n2"),
            })

    for m in _STEP_SINGLE_RE.finditer(masked):
        s_num, e_num = m.start("num"), m.end("num")
        if not _overlaps_occupied(s_num, e_num):
            occupied.append((m.start(), m.end()))
            tokens.append({
                "token": f"STEP:{int(m.group('num'))}",
                "span": (sentence_start + s_num, sentence_start + e_num),
                "raw": m.group("num"),
            })

    # 2. Dates via v3 extract_dates_with_spans
    for (ds, de), iso in extract_dates_with_spans(masked, allow_slash=True):
        if _overlaps_occupied(ds, de):
            continue
        occupied.append((ds, de))
        tokens.append({
            "token": f"DATE:{iso}",
            "span": (sentence_start + ds, sentence_start + de),
            "raw": sentence[ds:de],
        })

    # 3. Quantities via v3 extract_quantities (1-to-1 whitespace length preserved for exact spans)
    len_safe_sentence = re.sub(r"\s", " ", masked)
    for pq in extract_quantities(
        len_safe_sentence,
        apply_table_scale=False,
    ):
        qs, qe = int(pq.span[0]), int(pq.span[1])
        qs = max(0, min(len(sentence), qs))
        qe = max(qs, min(len(sentence), qe))
        # Trim outer wrapper punctuation such as "(" or "," so spans hug the quantity text
        while qs < qe and sentence[qs] in "([{ \t,;:\"'":
            qs += 1
        while qe > qs and sentence[qe - 1] in ")]} \t,;:\"'":
            qe -= 1
        if qs >= qe or _overlaps_occupied(qs, qe):
            continue
        # Keep typed quantities (amounts, %, bps, durations + counting basis, multiples) and large bare numbers >= 100
        if pq.unit == "NONE" and abs(pq.magnitude) < Decimal("100"):
            continue
        occupied.append((qs, qe))
        tok_key = f"{_fmt_decimal_compact(pq.magnitude)}:{pq.unit}"
        tokens.append({
            "token": tok_key,
            "span": (sentence_start + qs, sentence_start + qe),
            "raw": sentence[qs:qe],
        })

    tokens.sort(key=lambda t: (int(t["span"][0]), int(t["span"][1]), str(t["token"])))
    return tokens


_OCR_LIGATURE_MAP: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
}

_OCR_PUNCT_MAP: dict[str, str] = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
}

_OCR_STRIP_CHARS = frozenset({"\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"})


def normalize_pdf_ocr_noise(text: str) -> str:
    """Normalize PDF/OCR extraction noise (ligatures, line-break hyphens, wraps, smart punctuation)."""
    norm_text, _, _ = _normalize_pdf_ocr_with_map(text or "", base_offset=0)
    return norm_text


def _normalize_pdf_ocr_with_map(
    text: str,
    *,
    base_offset: int = 0,
) -> tuple[str, list[int], list[int]]:
    """Return `(normalized_text, norm_char_to_orig_start, norm_char_to_orig_end)`."""
    if not text:
        return "", [], []

    # Step 1: per-character ligatures, typographic punctuation, soft hyphens, CRLF -> LF
    stage1: list[tuple[str, int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        orig_s = base_offset + i
        if ch == "\r":
            orig_e = base_offset + i + (2 if (i + 1 < n and text[i + 1] == "\n") else 1)
            stage1.append(("\n", orig_s, orig_e))
            i += 2 if (i + 1 < n and text[i + 1] == "\n") else 1
            continue
        i += 1
        if ch in _OCR_STRIP_CHARS:
            continue
        if ch in _OCR_LIGATURE_MAP:
            for rep_ch in _OCR_LIGATURE_MAP[ch]:
                stage1.append((rep_ch, orig_s, orig_s + 1))
            continue
        if ch in _OCR_PUNCT_MAP:
            stage1.append((_OCR_PUNCT_MAP[ch], orig_s, orig_s + 1))
            continue
        nfkc = unicodedata.normalize("NFKC", ch)
        for rep_ch in nfkc:
            stage1.append((rep_ch, orig_s, orig_s + 1))

    # Step 2: join line-break hyphens and intra-paragraph wrapped lines
    stage2: list[tuple[str, int, int]] = []
    m = len(stage1)
    j = 0
    while j < m:
        ch, orig_s, orig_e = stage1[j]
        # Line-break hyphen: letter + '-' + optional spaces + '\n' + optional spaces + continuation letter
        if (
            ch == "-"
            and stage2
            and stage2[-1][0].isalpha()
        ):
            k = j + 1
            while k < m and stage1[k][0] in (" ", "\t"):
                k += 1
            if k < m and stage1[k][0] == "\n":
                k += 1
                while k < m and stage1[k][0] in (" ", "\t"):
                    k += 1
                if k < m and stage1[k][0].isalpha():
                    next_ch = stage1[k][0]
                    # Rejoin when continuation starts lowercase (word split across lines) or ALL-CAPS wrap
                    all_caps_wrap = next_ch.isupper() and len(stage2) >= 2 and stage2[-1][0].isupper() and stage2[-2][0].isupper()
                    if next_ch.islower() or all_caps_wrap:
                        j = k
                        continue

        # Intra-paragraph newline: single '\n' mid-sentence wrapped across PDF lines
        if ch == "\n" and stage2:
            k = j + 1
            while k < m and stage1[k][0] in (" ", "\t"):
                k += 1
            prev_line_chars: list[str] = []
            idx_p = len(stage2) - 1
            while idx_p >= 0 and stage2[idx_p][0] != "\n":
                prev_line_chars.append(stage2[idx_p][0])
                idx_p -= 1
            prev_line = "".join(reversed(prev_line_chars)).strip()
            if (
                prev_line
                and prev_line[-1] not in ".:;!?"
                and "|" not in prev_line
                and k < m
                and stage1[k][0] != "\n"
            ):
                # Look ahead to check next line is not a bullet, numbered list, or pipe row
                next_line_chars: list[str] = []
                idx_n = k
                while idx_n < m and stage1[idx_n][0] != "\n" and len(next_line_chars) < 24:
                    next_line_chars.append(stage1[idx_n][0])
                    idx_n += 1
                next_prefix = "".join(next_line_chars).lstrip()
                is_structural_next = bool(
                    next_prefix.startswith(("|", "- ", "* ", "•"))
                    or re.match(r"^(?:\(\d+\)|\(\w\)|\d+\.\s)", next_prefix)
                )
                # Real headings do not wrap onto a lowercase continuation word
                header_only_prev = _SECTION_HEADER_ONLY_RE.match(prev_line) is not None and not (
                    next_prefix and next_prefix[0].islower()
                )
                if not header_only_prev and not is_structural_next and next_prefix and (
                    next_prefix[0].islower()
                    or re.match(r"^(?:[A-Za-z0-9$\"'(])", next_prefix)
                ):
                    if stage2[-1][0] != " ":
                        stage2.append((" ", orig_s, orig_e))
                    j = k
                    continue

        stage2.append((ch, orig_s, orig_e))
        j += 1

    norm_text = "".join(c for c, _, _ in stage2)
    orig_starts = [s for _, s, _ in stage2]
    orig_ends = [e for _, _, e in stage2]
    return norm_text, orig_starts, orig_ends


def _norm_sentence_eq(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", normalize_pdf_ocr_noise(text).strip().lower())
    return cleaned.rstrip(" .;:")


def _sentence_content_words(text: str) -> set[str]:
    norm = normalize_pdf_ocr_noise(text).lower()
    words = {w for w in re.findall(r"[^\W\d_]{2,}", norm) if w not in _SENTENCE_STOPWORDS}
    return words


def _align_sentences(
    pri_sents: Sequence[tuple[int, int, str]],
    sec_sents: Sequence[tuple[int, int, str]],
    *,
    single_pair_min_shared: int = 2,
    min_jaccard: float = 0.20,
) -> tuple[
    list[tuple[tuple[int, int, str], tuple[int, int, str]]],
    list[tuple[int, int, str]],
    list[tuple[int, int, str]],
]:
    """Align sentences between two text regions (exact matches cancelled, remainder paired by content similarity)."""
    used_pri: set[int] = set()
    used_sec: set[int] = set()

    sec_norms_map: dict[str, deque[int]] = {}
    for si, s in enumerate(sec_sents):
        sn = _norm_sentence_eq(s[2])
        if sn:
            sec_norms_map.setdefault(sn, deque()).append(si)
    for pi, p in enumerate(pri_sents):
        pn = _norm_sentence_eq(p[2])
        if not pn:
            used_pri.add(pi)
            continue
        q = sec_norms_map.get(pn)
        if q:
            si = q.popleft()
            used_pri.add(pi)
            used_sec.add(si)

    rem_pri_idx = [
        i
        for i, (_, _, t) in enumerate(pri_sents)
        if i not in used_pri and not _SECTION_HEADER_ONLY_RE.match(t.strip())
    ]
    rem_sec_idx = [
        i
        for i, (_, _, t) in enumerate(sec_sents)
        if i not in used_sec and not _SECTION_HEADER_ONLY_RE.match(t.strip())
    ]

    pri_words = {i: _sentence_content_words(pri_sents[i][2]) for i in rem_pri_idx}
    sec_words = {j: _sentence_content_words(sec_sents[j][2]) for j in rem_sec_idx}

    candidates: list[tuple[float, int, int, int, int]] = []
    for pi in rem_pri_idx:
        pw = pri_words[pi]
        if not pw:
            continue
        for si in rem_sec_idx:
            sw = sec_words[si]
            if not sw:
                continue
            shared = len(pw & sw)
            union = len(pw | sw)
            jacc = shared / union if union else 0.0
            req_shared = 1 if (len(rem_pri_idx) == 1 and len(rem_sec_idx) == 1) else single_pair_min_shared
            if shared >= req_shared and (jacc >= min_jaccard or (len(rem_pri_idx) == 1 and len(rem_sec_idx) == 1)):
                candidates.append((-jacc, -shared, abs(pi - si), pi, si))

    candidates.sort()
    paired_pri: set[int] = set()
    paired_sec: set[int] = set()
    pairs: list[tuple[tuple[int, int, str], tuple[int, int, str]]] = []
    for _, _, _, pi, si in candidates:
        if pi in paired_pri or si in paired_sec:
            continue
        paired_pri.add(pi)
        paired_sec.add(si)
        pairs.append((pri_sents[pi], sec_sents[si]))

    unmatched_pri = [pri_sents[i] for i in rem_pri_idx if i not in paired_pri]
    unmatched_sec = [sec_sents[j] for j in rem_sec_idx if j not in paired_sec]
    return pairs, unmatched_pri, unmatched_sec


_DIFF_WORD_RE = re.compile(r"[^\W_]+(?:[.,_-][^\W_]+)*")


def _tokenize_with_spans(sentence: str, base_offset: int) -> list[tuple[str, int, int, str]]:
    """Tokenize *sentence* into alphanumeric spans, ignoring whitespace, casing, and punctuation."""
    out: list[tuple[str, int, int, str]] = []
    for m in _DIFF_WORD_RE.finditer(sentence or ""):
        raw = m.group(0)
        out.append((raw.lower(), int(base_offset) + m.start(), int(base_offset) + m.end(), raw))
    return out


def _differing_spans_from_obligations(
    reader_obligations: Sequence[VerificationObligation],
    admitted_units: Sequence[Any] = (),
    stored_fe_map: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Collect spans in each document that belong to differing-value (conflict/override) obligations."""
    diff_spans: dict[str, list[tuple[int, int]]] = {}
    conflicting_uids: set[str] = set()
    for ob in reader_obligations:
        kind = str((ob.parameters or {}).get("corpus_obligation_kind", "") or "")
        if kind not in ("material_conflict", "authority_override"):
            continue
        for uid in (ob.parameters or {}).get("unit_ids") or ():
            conflicting_uids.add(str(uid))
        for sp in (ob.parameters or {}).get("evidence_spans") or ():
            did = str(sp.get("document_id", "") or "")
            s_pos = sp.get("start")
            e_pos = sp.get("end")
            if did and s_pos is not None and e_pos is not None:
                diff_spans.setdefault(did, []).append((int(s_pos), int(e_pos)))

    if isinstance(stored_fe_map, Mapping) and conflicting_uids:
        for uid in conflicting_uids:
            fe_for_unit = stored_fe_map.get(uid)
            if not isinstance(fe_for_unit, Mapping):
                continue
            for fname, fref in fe_for_unit.items():
                if not str(fname).startswith("value[") or not isinstance(fref, Mapping):
                    continue
                r_doc = str(fref.get("document_id", "") or "")
                r_s = fref.get("start")
                r_e = fref.get("end")
                if r_doc and r_s is not None and r_e is not None:
                    diff_spans.setdefault(r_doc, []).append((int(r_s), int(r_e)))
    return diff_spans


def _is_doc_id_diff_fragment(
    tokens: Sequence[tuple[str, int, int, str]],
    doc_id: str,
    aliases: Sequence[str],
    *,
    doc_meta: Mapping[str, Any] | None = None,
    other_doc_id: str = "",
    other_meta: Mapping[str, Any] | None = None,
    single_side: bool = False,
) -> bool:
    """Return True if *tokens* structurally match *doc_id*'s identifier/title/role aliases."""
    if not tokens:
        return False
    norm_phrase = " ".join(t[0] for t in tokens)
    raw_lower = doc_id.strip().lower()
    stem_lower = _DOC_EXT_RE.sub("", raw_lower)
    stem_words = " ".join(_doc_stem_words(doc_id))
    exact_ids = {x for x in (raw_lower, stem_lower, stem_words) if x}
    if norm_phrase in exact_ids:
        return True
    alias_set = {a.strip().lower() for a in aliases if a.strip()}
    if single_side:
        if norm_phrase in alias_set and (len(tokens) >= 2 or "." in norm_phrase):
            return True
    elif norm_phrase in alias_set:
        return True
    # Check if every token belongs exclusively to this document's structural metadata (id/title/role)
    if len(tokens) >= 2:
        title = str((doc_meta or {}).get("title", "") or "")
        role = str((doc_meta or {}).get("role", "") or "")
        o_title = str((other_meta or {}).get("title", "") or "")
        o_role = str((other_meta or {}).get("role", "") or "")
        own_words = set(_doc_stem_words(doc_id, title, role)) - _GENERIC_DOC_WORDS
        other_words = set(_doc_stem_words(other_doc_id, o_title, o_role)) if other_doc_id else set()
        unique_own = own_words - other_words
        if unique_own and all(t[0] in unique_own for t in tokens):
            return True
    return False


def _unit_predicates_on_doc(admitted_units: Sequence[Any], doc_id: str) -> set[str]:
    preds: set[str] = set()
    for u in admitted_units:
        for ref in getattr(u, "evidence", ()) or ():
            if str(getattr(ref, "document_id", "") or "") == doc_id:
                pred = str((getattr(u, "proposition", {}) or {}).get("predicate", "") or "")
                if pred:
                    preds.add(pred)
    return preds


def _covered_by_disjoint_single_doc_unit(
    admitted_units: Sequence[Any],
    doc_id: str,
    other_doc_preds: set[str],
    target_span: tuple[int, int],
) -> bool:
    for u in admitted_units:
        pred = str((getattr(u, "proposition", {}) or {}).get("predicate", "") or "")
        if not pred or pred in other_doc_preds:
            continue
        for ref in getattr(u, "evidence", ()) or ():
            if str(getattr(ref, "document_id", "") or "") != doc_id:
                continue
            r_s = getattr(ref, "start", None)
            r_e = getattr(ref, "end", None)
            if r_s is not None and r_e is not None and _spans_intersect((int(r_s), int(r_e)), target_span):
                return True
    return False


def _bridges_two_differing_spans(
    diff_spans: Sequence[tuple[int, int]],
    tok_span: tuple[int, int],
    sent_text: str,
    sent_base: int,
) -> bool:
    """Return True if a 1-token insert/delete sits directly between two distinct differing spans."""
    ts, te = int(tok_span[0]), int(tok_span[1])
    left = [sp for sp in diff_spans if int(sp[0]) >= sent_base and int(sp[1]) <= ts]
    right = [sp for sp in diff_spans if int(sp[0]) >= te and int(sp[1]) <= sent_base + len(sent_text)]
    if not left or not right:
        return False
    nearest_left = max(left, key=lambda sp: int(sp[1]))
    nearest_right = min(right, key=lambda sp: int(sp[0]))
    if nearest_left == nearest_right:
        return False
    # Verify only whitespace/punctuation separates nearest_left -> token -> nearest_right
    gap_l = sent_text[max(0, int(nearest_left[1]) - sent_base) : max(0, ts - sent_base)]
    gap_r = sent_text[max(0, te - sent_base) : max(0, int(nearest_right[0]) - sent_base)]
    return not re.search(r"[A-Za-z0-9]", gap_l) and not re.search(r"[A-Za-z0-9]", gap_r)


def _build_merged_span_index(
    spans_by_doc: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, tuple[list[int], list[tuple[int, int]]]]:
    """Build per-document sorted disjoint half-open interval index for O(log N) lookups."""
    index: dict[str, tuple[list[int], list[tuple[int, int]]]] = {}
    for doc_id, raw_spans in spans_by_doc.items():
        clean = sorted((int(s), int(e)) for s, e in raw_spans if int(s) < int(e))
        if not clean:
            continue
        merged: list[tuple[int, int]] = [clean[0]]
        for s, e in clean[1:]:
            ps, pe = merged[-1]
            if s <= pe:
                if e > pe:
                    merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        index[doc_id] = ([m[0] for m in merged], merged)
    return index


def _span_index_intersects(
    index: Mapping[str, tuple[list[int], list[tuple[int, int]]]],
    doc_id: str,
    target_span: tuple[int, int],
) -> bool:
    """Return True iff *target_span* intersects any indexed span in *doc_id*."""
    ts, te = int(target_span[0]), int(target_span[1])
    if ts >= te:
        return False
    entry = index.get(doc_id)
    if not entry:
        return False
    starts, merged = entry
    pos = bisect_left(starts, te) - 1
    return pos >= 0 and merged[pos][1] > ts


def _unit_covers_doc_span(
    unit_spans_by_doc: Mapping[str, Sequence[tuple[int, int]]],
    doc_id: str,
    target_span: tuple[int, int],
) -> bool:
    for u_s, u_e in unit_spans_by_doc.get(doc_id, ()):
        if _spans_intersect((u_s, u_e), target_span):
            return True
    return False


def _divergence_gaps_for_item(
    div: Any,
    doc_texts: Mapping[str, str],
    admitted_units: Sequence[Any],
    unit_spans_by_doc: Mapping[str, Sequence[tuple[int, int]]],
    reader_obligations: Sequence[VerificationObligation],
    *,
    differing_spans_by_doc: Mapping[str, Sequence[tuple[int, int]]] | None = None,
    doc_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    unit_span_index: Mapping[str, tuple[list[int], list[tuple[int, int]]]] | None = None,
    diff_span_index: Mapping[str, tuple[list[int], list[tuple[int, int]]]] | None = None,
    doc_pred_spans: Mapping[str, Sequence[tuple[str, int, int]]] | None = None,
    doc_pred_sets: Mapping[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Compute fine-grained quantity & textual reference-register gaps for a divergence."""
    pri_id = str(getattr(div, "primary_source_id", "") or "")
    sec_id = str(getattr(div, "secondary_source_id", "") or "")
    sec_key = str(getattr(div, "section_key", "") or "")
    sec_label = str(getattr(div, "section_label", "") or "")
    pri_span = getattr(div, "primary_span", None)
    sec_span = getattr(div, "secondary_span", None)

    pri_full = doc_texts.get(pri_id, "")
    sec_full = doc_texts.get(sec_id, "")

    # Fallback when full source texts were not provided (synthetic minimal unit tests).
    if not pri_full and not sec_full:
        if _divergence_covered_by_reader_obligations(div, reader_obligations):
            return []
        return [{
            "kind": "section",
            "document_id": sec_id or pri_id,
            "offset": int(sec_span[0]) if sec_span is not None else (int(pri_span[0]) if pri_span is not None else 0),
            "sentence": sec_label or sec_key,
            "token": "SECTION_DIVERGENCE",
            "section_key": sec_key,
            "section_label": sec_label,
            "primary_source_id": pri_id,
            "secondary_source_id": sec_id,
            "primary_span": list(pri_span) if pri_span is not None else None,
            "secondary_span": list(sec_span) if sec_span is not None else None,
            "reason": str(getattr(div, "reason", "") or ""),
        }]

    pri_base = int(pri_span[0]) if pri_span is not None else 0
    sec_base = int(sec_span[0]) if sec_span is not None else 0
    pri_chunk = pri_full[int(pri_span[0]) : int(pri_span[1])] if (pri_full and pri_span is not None) else pri_full
    sec_chunk = sec_full[int(sec_span[0]) : int(sec_span[1])] if (sec_full and sec_span is not None) else sec_full

    pri_norm_chunk, pri_char_s, pri_char_e = _normalize_pdf_ocr_with_map(pri_chunk, base_offset=pri_base)
    sec_norm_chunk, sec_char_s, sec_char_e = _normalize_pdf_ocr_with_map(sec_chunk, base_offset=sec_base)

    def _map_norm_span(char_s: Sequence[int], char_e: Sequence[int], ns: int, ne: int, default_base: int) -> tuple[int, int]:
        if not char_s or ns >= ne or ns < 0 or ns >= len(char_s):
            return default_base, default_base
        end_idx = min(ne - 1, len(char_e) - 1)
        return int(char_s[ns]), max(int(char_s[ns]), int(char_e[end_idx]))

    # Track norm_start for each segmented sentence while keeping original doc offsets on tuple[0..1]
    raw_pri_sents = segment_sentences_with_spans(pri_norm_chunk, base_offset=0) if pri_norm_chunk else []
    raw_sec_sents = segment_sentences_with_spans(sec_norm_chunk, base_offset=0) if sec_norm_chunk else []

    pri_sents: list[tuple[int, int, str]] = []
    pri_norm_base_by_sent: dict[tuple[int, int], int] = {}
    for ns, ne, stxt in raw_pri_sents:
        os, oe = _map_norm_span(pri_char_s, pri_char_e, ns, ne, pri_base)
        pri_sents.append((os, oe, stxt))
        pri_norm_base_by_sent[(os, oe)] = ns

    sec_sents: list[tuple[int, int, str]] = []
    sec_norm_base_by_sent: dict[tuple[int, int], int] = {}
    for ns, ne, stxt in raw_sec_sents:
        os, oe = _map_norm_span(sec_char_s, sec_char_e, ns, ne, sec_base)
        sec_sents.append((os, oe, stxt))
        sec_norm_base_by_sent[(os, oe)] = ns

    def _orig_sent_span(doc_id: str, sent: tuple[int, int, str], rel_s: int, rel_e: int, base_sent_offset: int) -> tuple[int, int]:
        """Map a span computed inside `sent[2]` back into original document offsets."""
        if doc_id == pri_id:
            n_base = pri_norm_base_by_sent.get((int(sent[0]), int(sent[1])))
            if n_base is not None:
                delta_s = max(0, rel_s - base_sent_offset)
                delta_e = max(delta_s, rel_e - base_sent_offset)
                return _map_norm_span(pri_char_s, pri_char_e, n_base + delta_s, n_base + delta_e, int(sent[0]))
        else:
            n_base = sec_norm_base_by_sent.get((int(sent[0]), int(sent[1])))
            if n_base is not None:
                delta_s = max(0, rel_s - base_sent_offset)
                delta_e = max(delta_s, rel_e - base_sent_offset)
                return _map_norm_span(sec_char_s, sec_char_e, n_base + delta_s, n_base + delta_e, int(sent[0]))
        return rel_s, rel_e

    both_bounded = pri_span is not None and sec_span is not None
    pairs, unmatched_pri, unmatched_sec = _align_sentences(
        pri_sents,
        sec_sents,
        single_pair_min_shared=2 if both_bounded else 2,
        min_jaccard=0.20 if both_bounded else 0.25,
    )

    gaps: list[dict[str, Any]] = []

    def _emit_quantity_gap(
        doc_id: str,
        q: Mapping[str, Any],
        sent: tuple[int, int, str],
        counterpart_id: str,
        counterpart_sent: tuple[int, int, str] | None,
    ) -> None:
        qs, qe = int(q["span"][0]), int(q["span"][1])
        gaps.append({
            "kind": "quantity",
            "document_id": doc_id,
            "offset": qs,
            "span": [qs, qe],
            "token": str(q["token"]),
            "raw_token": str(q.get("raw", "")),
            "sentence": sent[2],
            "sentence_span": [int(sent[0]), int(sent[1])],
            "counterpart_document_id": counterpart_id,
            "counterpart_sentence": counterpart_sent[2] if counterpart_sent is not None else None,
            "section_key": sec_key,
            "section_label": sec_label,
            "primary_source_id": pri_id,
            "secondary_source_id": sec_id,
        })

    eff_diff_spans = (
        differing_spans_by_doc
        if differing_spans_by_doc is not None
        else _differing_spans_from_obligations(reader_obligations, admitted_units)
    )
    eff_unit_idx = unit_span_index if unit_span_index is not None else _build_merged_span_index(unit_spans_by_doc)
    eff_diff_idx = diff_span_index if diff_span_index is not None else _build_merged_span_index(eff_diff_spans)
    pri_aliases, sec_aliases = _pair_doc_aliases(pri_id, sec_id, doc_metadata)
    pri_meta = (doc_metadata or {}).get(pri_id, {})
    sec_meta = (doc_metadata or {}).get(sec_id, {})
    pri_preds = (doc_pred_sets[pri_id] if doc_pred_sets and pri_id in doc_pred_sets else _unit_predicates_on_doc(admitted_units, pri_id))
    sec_preds = (doc_pred_sets[sec_id] if doc_pred_sets and sec_id in doc_pred_sets else _unit_predicates_on_doc(admitted_units, sec_id))
    disjoint_pri_idx: dict[str, tuple[list[int], list[tuple[int, int]]]] | None = None
    disjoint_sec_idx: dict[str, tuple[list[int], list[tuple[int, int]]]] | None = None

    def _get_disjoint_idx(target_doc: str, other_preds: set[str]) -> dict[str, tuple[list[int], list[tuple[int, int]]]]:
        if doc_pred_spans is not None:
            spans = [(s, e) for p, s, e in doc_pred_spans.get(target_doc, ()) if p and p not in other_preds]
            return _build_merged_span_index({target_doc: spans})
        spans = []
        for u in admitted_units:
            pred = str((getattr(u, "proposition", {}) or {}).get("predicate", "") or "")
            if not pred or pred in other_preds:
                continue
            for ref in getattr(u, "evidence", ()) or ():
                if str(getattr(ref, "document_id", "") or "") != target_doc:
                    continue
                r_s = getattr(ref, "start", None)
                r_e = getattr(ref, "end", None)
                if r_s is not None and r_e is not None:
                    spans.append((int(r_s), int(r_e)))
        return _build_merged_span_index({target_doc: spans})

    # 1. Aligned differing sentence pairs
    for p_sent, s_sent in pairs:
        p_norm = _norm_sentence_eq(p_sent[2])
        s_norm = _norm_sentence_eq(s_sent[2])
        if p_norm == s_norm:
            continue
        p_quants = _extract_sentence_quantity_tokens(p_sent[2], p_sent[0])
        for q in p_quants:
            os, oe = _orig_sent_span(pri_id, p_sent, int(q["span"][0]), int(q["span"][1]), int(p_sent[0]))
            q["span"] = [os, oe]
        s_quants = _extract_sentence_quantity_tokens(s_sent[2], s_sent[0])
        for q in s_quants:
            os, oe = _orig_sent_span(sec_id, s_sent, int(q["span"][0]), int(q["span"][1]), int(s_sent[0]))
            q["span"] = [os, oe]
        p_keys = {q["token"] for q in p_quants}
        s_keys = {q["token"] for q in s_quants}
        p_diff = [q for q in p_quants if q["token"] not in s_keys]
        s_diff = [q for q in s_quants if q["token"] not in p_keys]

        p_diff_q_spans = [(int(q["span"][0]), int(q["span"][1])) for q in p_diff]
        s_diff_q_spans = [(int(q["span"][0]), int(q["span"][1])) for q in s_diff]

        for q in p_diff:
            q_span = (int(q["span"][0]), int(q["span"][1]))
            if not _span_index_intersects(eff_unit_idx, pri_id, q_span):
                _emit_quantity_gap(pri_id, q, p_sent, sec_id, s_sent)
        for q in s_diff:
            q_span = (int(q["span"][0]), int(q["span"][1]))
            if not _span_index_intersects(eff_unit_idx, sec_id, q_span):
                _emit_quantity_gap(sec_id, q, s_sent, pri_id, p_sent)

        # Token-level diff check on ALL differing aligned sentence pairs (even when quantities differ).
        p_tokens = [
            (norm_t, *_orig_sent_span(pri_id, p_sent, ts, te, int(p_sent[0])), raw_t)
            for norm_t, ts, te, raw_t in _tokenize_with_spans(p_sent[2], int(p_sent[0]))
        ]
        s_tokens = [
            (norm_t, *_orig_sent_span(sec_id, s_sent, ts, te, int(s_sent[0])), raw_t)
            for norm_t, ts, te, raw_t in _tokenize_with_spans(s_sent[2], int(s_sent[0]))
        ]
        sm = difflib.SequenceMatcher(
            None,
            [t[0] for t in p_tokens],
            [t[0] for t in s_tokens],
            autojunk=False,
        )
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            p_slice = p_tokens[i1:i2]
            s_slice = s_tokens[j1:j2]

            p_non_qty = [
                tok
                for tok in p_slice
                if not any(_spans_intersect((tok[1], tok[2]), qs) for qs in p_diff_q_spans)
            ]
            s_non_qty = [
                tok
                for tok in s_slice
                if not any(_spans_intersect((tok[1], tok[2]), qs) for qs in s_diff_q_spans)
            ]
            p_uncov = [
                tok
                for tok in p_non_qty
                if not _span_index_intersects(eff_diff_idx, pri_id, (tok[1], tok[2]))
            ]
            s_uncov = [
                tok
                for tok in s_non_qty
                if not _span_index_intersects(eff_diff_idx, sec_id, (tok[1], tok[2]))
            ]

            # Strip leading/trailing or full structural doc-identifier tokens from uncovered slices.
            def _strip_doc_prefix(
                uncov: list[tuple[str, int, int, str]],
                did: str,
                aliases: Sequence[str],
                d_meta: Mapping[str, Any],
                o_did: str,
                o_meta: Mapping[str, Any],
                single: bool,
            ) -> tuple[list[tuple[str, int, int, str]], bool]:
                if not uncov:
                    return uncov, False
                if _is_doc_id_diff_fragment(
                    uncov, did, aliases, doc_meta=d_meta, other_doc_id=o_did, other_meta=o_meta, single_side=single
                ):
                    return [], True
                # Also strip a doc-id prefix subsequence (e.g. "Baseline Term Sheet:" before another clause)
                for k in range(len(uncov) - 1, 0, -1):
                    if _is_doc_id_diff_fragment(
                        uncov[:k],
                        did,
                        aliases,
                        doc_meta=d_meta,
                        other_doc_id=o_did,
                        other_meta=o_meta,
                        single_side=True,
                    ):
                        return uncov[k:], True
                return uncov, False

            single_side_orig = tag in ("insert", "delete")
            p_uncov, p_had_doc_ref = _strip_doc_prefix(
                p_uncov, pri_id, pri_aliases, pri_meta, sec_id, sec_meta, single_side_orig
            )
            s_uncov, s_had_doc_ref = _strip_doc_prefix(
                s_uncov, sec_id, sec_aliases, sec_meta, pri_id, pri_meta, single_side_orig
            )

            # One-sided extra clause already admitted cleanly on a predicate absent from counterpart doc
            if p_uncov and not s_uncov and (tag == "delete" or s_had_doc_ref or not s_non_qty):
                if disjoint_pri_idx is None:
                    disjoint_pri_idx = _get_disjoint_idx(pri_id, sec_preds)
                p_uncov = [
                    tok
                    for tok in p_uncov
                    if not _span_index_intersects(disjoint_pri_idx, pri_id, (tok[1], tok[2]))
                ]
            if s_uncov and not p_uncov and (tag == "insert" or p_had_doc_ref or not p_non_qty):
                if disjoint_sec_idx is None:
                    disjoint_sec_idx = _get_disjoint_idx(sec_id, pri_preds)
                s_uncov = [
                    tok
                    for tok in s_uncov
                    if not _span_index_intersects(disjoint_sec_idx, sec_id, (tok[1], tok[2]))
                ]

            # 1-token structural clause join (e.g. "; " vs " and ") strictly bridging two differing spans
            if tag == "delete" and len(p_uncov) == 1 and not s_uncov:
                if _bridges_two_differing_spans(
                    eff_diff_spans.get(pri_id, ()), (p_uncov[0][1], p_uncov[0][2]), p_sent[2], int(p_sent[0])
                ):
                    p_uncov = []
            if tag == "insert" and len(s_uncov) == 1 and not p_uncov:
                if _bridges_two_differing_spans(
                    eff_diff_spans.get(sec_id, ()), (s_uncov[0][1], s_uncov[0][2]), s_sent[2], int(s_sent[0])
                ):
                    s_uncov = []

            p_frag_text = " ".join(t[3] for t in p_uncov) if p_uncov else " ".join(t[3] for t in p_slice)
            s_frag_text = " ".join(t[3] for t in s_uncov) if s_uncov else " ".join(t[3] for t in s_slice)
            p_hunk_span = (
                [p_uncov[0][1], p_uncov[-1][2]]
                if p_uncov
                else ([p_slice[0][1], p_slice[-1][2]] if p_slice else [int(p_sent[0]), int(p_sent[0])])
            )
            s_hunk_span = (
                [s_uncov[0][1], s_uncov[-1][2]]
                if s_uncov
                else ([s_slice[0][1], s_slice[-1][2]] if s_slice else [int(s_sent[0]), int(s_sent[0])])
            )

            if p_uncov:
                gaps.append({
                    "kind": "textual",
                    "document_id": pri_id,
                    "offset": int(p_uncov[0][1]),
                    "span": [int(p_uncov[0][1]), int(p_uncov[-1][2])],
                    "token": "TEXTUAL_DIFF",
                    "changed_text": pri_full[p_uncov[0][1] : p_uncov[-1][2]] if pri_full else p_frag_text,
                    "sentence": p_sent[2],
                    "sentence_span": [int(p_sent[0]), int(p_sent[1])],
                    "counterpart_document_id": sec_id,
                    "counterpart_offset": int(s_hunk_span[0]),
                    "counterpart_span": [int(s_hunk_span[0]), int(s_hunk_span[1])],
                    "counterpart_changed_text": (
                        sec_full[s_hunk_span[0] : s_hunk_span[1]]
                        if (sec_full and s_hunk_span[1] > s_hunk_span[0])
                        else s_frag_text
                    ),
                    "counterpart_sentence": s_sent[2],
                    "section_key": sec_key,
                    "section_label": sec_label,
                    "primary_source_id": pri_id,
                    "secondary_source_id": sec_id,
                })
            if s_uncov:
                gaps.append({
                    "kind": "textual",
                    "document_id": sec_id,
                    "offset": int(s_uncov[0][1]),
                    "span": [int(s_uncov[0][1]), int(s_uncov[-1][2])],
                    "token": "TEXTUAL_DIFF",
                    "changed_text": sec_full[s_uncov[0][1] : s_uncov[-1][2]] if sec_full else s_frag_text,
                    "sentence": s_sent[2],
                    "sentence_span": [int(s_sent[0]), int(s_sent[1])],
                    "counterpart_document_id": pri_id,
                    "counterpart_offset": int(p_hunk_span[0]),
                    "counterpart_span": [int(p_hunk_span[0]), int(p_hunk_span[1])],
                    "counterpart_changed_text": (
                        pri_full[p_hunk_span[0] : p_hunk_span[1]]
                        if (pri_full and p_hunk_span[1] > p_hunk_span[0])
                        else p_frag_text
                    ),
                    "counterpart_sentence": p_sent[2],
                    "section_key": sec_key,
                    "section_label": sec_label,
                    "primary_source_id": pri_id,
                    "secondary_source_id": sec_id,
                })

    # 2. Unmatched sentences with quantities (in explicit bounded sections, or on the bounded divergent side)
    if pri_span is not None:
        for p_sent in unmatched_pri:
            for q in _extract_sentence_quantity_tokens(p_sent[2], p_sent[0]):
                os, oe = _orig_sent_span(pri_id, p_sent, int(q["span"][0]), int(q["span"][1]), int(p_sent[0]))
                q["span"] = [os, oe]
                q_span = (os, oe)
                if not _span_index_intersects(eff_unit_idx, pri_id, q_span):
                    _emit_quantity_gap(pri_id, q, p_sent, sec_id, None)
    if sec_span is not None:
        for s_sent in unmatched_sec:
            for q in _extract_sentence_quantity_tokens(s_sent[2], s_sent[0]):
                os, oe = _orig_sent_span(sec_id, s_sent, int(q["span"][0]), int(q["span"][1]), int(s_sent[0]))
                q["span"] = [os, oe]
                q_span = (os, oe)
                if not _span_index_intersects(eff_unit_idx, sec_id, q_span):
                    _emit_quantity_gap(sec_id, q, s_sent, pri_id, None)

    return gaps


def _obligation_overlaps_doc_span_raw(unit: Any, doc_id: str, span: tuple[int, int]) -> bool:
    for ref in getattr(unit, "evidence", ()) or ():
        if str(getattr(ref, "document_id", "") or "") != doc_id:
            continue
        r_start = getattr(ref, "start", None)
        r_end = getattr(ref, "end", None)
        if r_start is not None and r_end is not None:
            if _spans_overlap(doc_id, (int(r_start), int(r_end)), doc_id, span):
                return True
    return False


def _divergence_covered_by_reader_obligations(
    div: Any, reader_obligations: Sequence[VerificationObligation]
) -> bool:
    pri_id = str(getattr(div, "primary_source_id", "") or "")
    sec_id = str(getattr(div, "secondary_source_id", "") or "")
    pri_span = getattr(div, "primary_span", None)
    sec_span = getattr(div, "secondary_span", None)
    pri_quants = tuple(getattr(div, "primary_quantities", ()) or ())
    sec_quants = tuple(getattr(div, "secondary_quantities", ()) or ())

    pri_matches = (
        [ob for ob in reader_obligations if _obligation_overlaps_doc_span(ob, pri_id, pri_span)]
        if pri_id and pri_span is not None
        else []
    )
    sec_matches = (
        [ob for ob in reader_obligations if _obligation_overlaps_doc_span(ob, sec_id, sec_span)]
        if sec_id and sec_span is not None
        else []
    )

    if pri_span is not None and sec_span is not None:
        if not pri_matches or not sec_matches:
            return False
        # If both sections have distinct quantitative tokens, require that the reader
        # extracted at least one overlapping conflict or override obligation (not just an unrelated shared number).
        if pri_quants and sec_quants and set(pri_quants) != set(sec_quants):
            has_conflict_ob = any(
                (ob.parameters or {}).get("corpus_obligation_kind")
                in ("material_conflict", "authority_override")
                for ob in (pri_matches + sec_matches)
            )
            return has_conflict_ob
        return True
    if pri_span is not None:
        return bool(pri_matches)
    if sec_span is not None:
        return bool(sec_matches)
    return False


def _compute_register_state(
    evidence: EvidenceLayer,
    vocabulary: Vocabulary,
    *,
    question: str | Mapping[str, Any] | None,
    entity: str | None,
    required_predicates: Sequence[str] | None,
    eff_divergences: Sequence[Any],
    divergence_collection: ReferenceDivergenceCollection | None,
    expected_documents: Mapping[str, Any] | Sequence[str] | None,
) -> dict[str, Any] | None:
    admitted = sorted(evidence.units, key=lambda u: str(u.unit_id))
    if not admitted:
        return None

    target_entity = entity
    if target_entity is None and isinstance(question, Mapping):
        raw_ent = question.get("entity")
        if isinstance(raw_ent, str) and raw_ent.strip():
            target_entity = raw_ent.strip()

    # Determine mandatory predicates from explicit override or Vocabulary.required flags
    if required_predicates is not None:
        req_preds = tuple(dict.fromkeys(str(p) for p in required_predicates))
    else:
        vocab_req = tuple(p.name for p in vocabulary.predicates if getattr(p, "required", False))
        req_preds = vocab_req if vocab_req else tuple(
            sorted({str(u.proposition.get("predicate", "")) for u in admitted if u.proposition.get("predicate")})
        )

    doc_ids = tuple(sorted({
        str(ref.document_id)
        for u in admitted
        for ref in (getattr(u, "evidence", ()) or ())
    }))

    # Filter units for entity if target_entity is specified and matches at least one unit
    relevant_units = admitted
    if target_entity is not None:
        ent_units = [u for u in admitted if str(u.proposition.get("entity", "")) == target_entity]
        if ent_units:
            relevant_units = ent_units

    obligations: list[VerificationObligation] = []
    seen_ids: set[str] = set()

    # Pre-cache document authority ranks once per (document_id, revision) so SQLite is never queried per-unit/per-pair.
    _MISSING_RANK = object()
    doc_rank_by_rev: dict[tuple[str, str], Any] = {}
    for sdoc in getattr(evidence, "documents", ()) or ():
        did = str(getattr(sdoc, "document_id", "") or "")
        rev = str(getattr(sdoc, "revision", "") or "")
        if did:
            doc_rank_by_rev[(did, rev)] = int(getattr(sdoc, "authority_rank", 0) or 0)

    unit_rank_cache: dict[str, int] = {}
    unit_spans_cache: dict[str, list[dict[str, Any]]] = {}

    def _unit_rank(u: Any) -> int:
        uid = str(u.unit_id)
        cached = unit_rank_cache.get(uid)
        if cached is not None:
            return cached
        ranks: list[int] = []
        for ref in getattr(u, "evidence", ()) or ():
            key = (ref.document_id, ref.revision)
            rk = doc_rank_by_rev.get(key, _MISSING_RANK)
            if rk is _MISSING_RANK:
                try:
                    doc = evidence._document(ref.document_id, ref.revision, current=False)
                    rk = int(doc.authority_rank)
                except Exception:
                    rk = None
                doc_rank_by_rev[key] = rk
            if rk is not None:
                ranks.append(rk)
        res = max(ranks) if ranks else 0
        unit_rank_cache[uid] = res
        return res

    def _get_unit_spans(u: Any) -> list[dict[str, Any]]:
        uid = str(u.unit_id)
        cached = unit_spans_cache.get(uid)
        if cached is not None:
            return cached
        sp = _unit_spans(u)
        unit_spans_cache[uid] = sp
        return sp

    # Pre-cache context_key once per admitted unit (avoid repeated canonical() JSON encoding in loops)
    unit_ctx_cache: dict[str, str] = {str(u.unit_id): str(u.context_key) for u in admitted}

    # (i) Mandatory predicates that have at least one admissible unit
    by_predicate: dict[str, list[Any]] = {}
    for u in relevant_units:
        pred = str(u.proposition.get("predicate", ""))
        if pred:
            by_predicate.setdefault(pred, []).append(u)

    for pred in req_preds:
        pred_units = by_predicate.get(pred, ())
        if not pred_units:
            continue
        # Pick highest-authority unit(s) as primary source span anchor
        ranked = sorted(
            pred_units,
            key=lambda u: (-_unit_rank(u), str(u.unit_id)),
        )
        top_unit = ranked[0]
        ctx_key = unit_ctx_cache[str(top_unit.unit_id)]
        ob_id = f"corpus::predicate::{pred}::{digest(ctx_key)[:12]}"
        if ob_id in seen_ids:
            continue
        seen_ids.add(ob_id)
        obligations.append(
            VerificationObligation(
                obligation_id=ob_id,
                family="document_authority",
                description=(
                    f"Cover admitted corpus predicate '{pred}' for entity "
                    f"'{top_unit.proposition.get('entity', '')}'."
                ),
                critical=True,
                origin="corpus",
                parameters={
                    "corpus_obligation_kind": "mandatory_predicate",
                    "entity": str(top_unit.proposition.get("entity", "")),
                    "predicate": pred,
                    "context_key": ctx_key,
                    "expected_value": top_unit.proposition.get("value"),
                    "expected_unit": top_unit.proposition.get("unit"),
                    "unit_ids": [str(u.unit_id) for u in ranked],
                    "evidence_spans": _get_unit_spans(top_unit),
                },
            )
        )

    # Group by predicate to detect same-context and compatible-scope cross-document conflicts/overrides
    by_pred: dict[str, list[Any]] = {}
    for u in relevant_units:
        pred = str(u.proposition.get("predicate", ""))
        by_pred.setdefault(pred, []).append(u)

    def _unit_doc_id(u: Any) -> str:
        evs = getattr(u, "evidence", ()) or ()
        return str(evs[0].document_id) if evs else ""

    def _unit_scope(u: Any) -> Mapping[str, Any]:
        sc = u.proposition.get("scope")
        return sc if isinstance(sc, Mapping) else {}

    _SUB_ENTITY_QUAL_RE = re.compile(r"\b(class|tranche|series|tier|facility)\s+([a-z0-9-]+)\b", re.IGNORECASE)

    def _norm(val: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (val or "").strip().lower()).strip("_")

    def _entity_qualifier(ent_txt: str) -> tuple[str, str] | None:
        m = _SUB_ENTITY_QUAL_RE.search(ent_txt or "")
        return (m.group(1).lower(), m.group(2).lower()) if m else None

    def _scopes_compatible(s1: Mapping[str, Any], s2: Mapping[str, Any]) -> bool:
        for k in set(s1) & set(s2):
            if _norm(str(s1[k])) != _norm(str(s2[k])):
                return False
        return True

    has_conflicts = False
    has_overrides = False

    for pred in sorted(by_pred):
        pred_units = by_pred[pred]
        if len(pred_units) < 2:
            continue
        try:
            spec = vocabulary.predicate(pred)
        except Exception:
            continue

        unit_docs = [_unit_doc_id(u) for u in pred_units]
        unit_ents_norm = [_norm(str(u.proposition.get("entity", ""))) for u in pred_units]
        unit_ctxs = [unit_ctx_cache[str(u.unit_id)] for u in pred_units]
        unit_valsigs = [
            (canonical(u.proposition.get("value")), u.proposition.get("unit"), u.proposition.get("polarity"))
            for u in pred_units
        ]

        # Track per-document normalized entities and group indices by context/entity/val_sig to avoid O(N^2) scans.
        doc_entities: dict[str, set[str]] = {}
        ctx_val_buckets: dict[str, dict[tuple[str, Any, Any], list[int]]] = {}
        ent_val_buckets: dict[str, dict[tuple[str, Any, Any], list[int]]] = {}
        for idx in range(len(pred_units)):
            d_i = unit_docs[idx]
            e_i = unit_ents_norm[idx]
            vs_i = unit_valsigs[idx]
            doc_entities.setdefault(d_i, set()).add(e_i)
            ctx_val_buckets.setdefault(unit_ctxs[idx], {}).setdefault(vs_i, []).append(idx)
            if d_i:
                ent_val_buckets.setdefault(e_i, {}).setdefault(vs_i, []).append(idx)

        candidate_pairs: set[tuple[int, int]] = set()

        def _collect_cross_val_pairs(
            val_map: Mapping[tuple[str, Any, Any], list[int]],
            *,
            require_diff_doc: bool,
        ) -> None:
            if len(val_map) < 2:
                return
            groups = list(val_map.values())
            m = len(groups)
            for gi in range(m):
                g1 = groups[gi]
                for gj in range(gi + 1, m):
                    g2 = groups[gj]
                    for a in g1:
                        da = unit_docs[a]
                        for b in g2:
                            if require_diff_doc and (not da or not unit_docs[b] or da == unit_docs[b]):
                                continue
                            candidate_pairs.add((a, b) if a < b else (b, a))

        for val_map in ctx_val_buckets.values():
            _collect_cross_val_pairs(val_map, require_diff_doc=False)
        for val_map in ent_val_buckets.values():
            _collect_cross_val_pairs(val_map, require_diff_doc=True)

        singleton_docs = {d for d, ents in doc_entities.items() if d and len(ents) == 1}
        if len(singleton_docs) >= 2:
            singleton_val_buckets: dict[tuple[str, Any, Any], list[int]] = {}
            for idx in range(len(pred_units)):
                if unit_docs[idx] in singleton_docs:
                    singleton_val_buckets.setdefault(unit_valsigs[idx], []).append(idx)
            _collect_cross_val_pairs(singleton_val_buckets, require_diff_doc=True)

        for i, j in sorted(candidate_pairs):
            u1, u2 = pred_units[i], pred_units[j]
            d1, d2 = unit_docs[i], unit_docs[j]
            if d1 and d2 and d1 == d2:
                # Skip two alternative unit-readings from overlapping character spans in the
                # same document (e.g. parenthetical alternatives in the same clause).
                evs1 = getattr(u1, "evidence", ()) or ()
                evs2 = getattr(u2, "evidence", ()) or ()
                if evs1 and evs2:
                    s1 = getattr(evs1[0], "start", None)
                    e1_pos = getattr(evs1[0], "end", None)
                    s2 = getattr(evs2[0], "start", None)
                    e2_pos = getattr(evs2[0], "end", None)
                    if (
                        s1 is not None
                        and e1_pos is not None
                        and s2 is not None
                        and e2_pos is not None
                        and max(int(s1), int(s2)) < min(int(e1_pos), int(e2_pos))
                    ):
                        continue
            same_ctx = unit_ctxs[i] == unit_ctxs[j]
            if not same_ctx:
                if not d1 or not d2 or d1 == d2:
                    continue
                s1, s2 = _unit_scope(u1), _unit_scope(u2)
                if not _scopes_compatible(s1, s2):
                    continue
                e1, e2 = str(u1.proposition.get("entity", "")), str(u2.proposition.get("entity", ""))
                q1, q2 = _entity_qualifier(e1), _entity_qualifier(e2)
                if q1 and q2 and q1[0] == q2[0] and q1[1] != q2[1]:
                    continue
                same_ent = unit_ents_norm[i] == unit_ents_norm[j]
                singleton_per_doc = d1 in singleton_docs and d2 in singleton_docs
                if not (same_ent or singleton_per_doc):
                    continue
            ctx_key = unit_ctxs[i] if same_ctx else f"{pred}::{unit_ents_norm[i]}"
            if not _units_incompatible(vocabulary, spec, u1, u2):
                continue
            r1 = _unit_rank(u1)
            r2 = _unit_rank(u2)
            # (ii) Material conflict for every incompatible pair on the same context_key
            has_conflicts = True
            pair_sorted = sorted([u1, u2], key=lambda u: str(u.unit_id))
            a_u, b_u = pair_sorted[0], pair_sorted[1]
            ob_id = f"corpus::conflict::{pred}::{digest([a_u.unit_id, b_u.unit_id])[:12]}"
            if ob_id not in seen_ids:
                seen_ids.add(ob_id)
                obligations.append(
                    VerificationObligation(
                        obligation_id=ob_id,
                        family="document_authority",
                        description=(
                            f"Disclose material conflict for '{pred}' between incompatible "
                            "admitted corpus values (present both values and governance/unresolved status)."
                        ),
                        critical=True,
                        origin="corpus",
                        parameters={
                            "corpus_obligation_kind": "material_conflict",
                            "entity": str(a_u.proposition.get("entity", "")),
                            "predicate": pred,
                            "context_key": ctx_key,
                            "unit_ids": [str(a_u.unit_id), str(b_u.unit_id)],
                            "conflicting_values": [
                                {
                                    "unit_id": str(a_u.unit_id),
                                    "document_id": str(a_u.evidence[0].document_id) if getattr(a_u, "evidence", ()) else "",
                                    "value": a_u.proposition.get("value"),
                                    "unit": a_u.proposition.get("unit"),
                                    "authority_rank": _unit_rank(a_u),
                                },
                                {
                                    "unit_id": str(b_u.unit_id),
                                    "document_id": str(b_u.evidence[0].document_id) if getattr(b_u, "evidence", ()) else "",
                                    "value": b_u.proposition.get("value"),
                                    "unit": b_u.proposition.get("unit"),
                                    "authority_rank": _unit_rank(b_u),
                                },
                            ],
                            "evidence_spans": _get_unit_spans(a_u) + _get_unit_spans(b_u),
                        },
                    )
                )
            if r1 != r2:
                # (iii) Higher authority_rank overrides lower authority_rank
                has_overrides = True
                winner, loser = (u1, u2) if r1 > r2 else (u2, u1)
                w_rank, l_rank = max(r1, r2), min(r1, r2)
                ob_id = f"corpus::override::{pred}::{digest([winner.unit_id, loser.unit_id])[:12]}"
                if ob_id in seen_ids:
                    continue
                seen_ids.add(ob_id)
                obligations.append(
                    VerificationObligation(
                        obligation_id=ob_id,
                        family="document_authority",
                        description=(
                            f"Cover authority override for '{pred}': higher authority_rank ({w_rank}) "
                            f"supersedes lower authority_rank ({l_rank}) while disclosing the discrepancy."
                        ),
                        critical=True,
                        origin="corpus",
                        parameters={
                            "corpus_obligation_kind": "authority_override",
                            "entity": str(winner.proposition.get("entity", "")),
                            "predicate": pred,
                            "context_key": ctx_key,
                            "unit_ids": [str(winner.unit_id), str(loser.unit_id)],
                            "winner_unit_id": str(winner.unit_id),
                            "loser_unit_id": str(loser.unit_id),
                            "winner_document_id": str(winner.evidence[0].document_id) if getattr(winner, "evidence", ()) else "",
                            "loser_document_id": str(loser.evidence[0].document_id) if getattr(loser, "evidence", ()) else "",
                            "winner_value": winner.proposition.get("value"),
                            "winner_unit": winner.proposition.get("unit"),
                            "winner_authority_rank": w_rank,
                            "loser_value": loser.proposition.get("value"),
                            "loser_unit": loser.proposition.get("unit"),
                            "loser_authority_rank": l_rank,
                            "evidence_spans": _get_unit_spans(winner) + _get_unit_spans(loser),
                        },
                    )
                )

    # Collect document full texts from divergence_collection, evidence layer, and expected_documents
    doc_texts: dict[str, str] = {}
    if divergence_collection is not None and divergence_collection.document_texts:
        doc_texts.update({str(k): str(v) for k, v in divergence_collection.document_texts.items() if v})
    for sdoc in getattr(evidence, "documents", ()) or ():
        did = str(getattr(sdoc, "document_id", "") or "")
        txt = str(getattr(sdoc, "text", "") or "")
        if did and txt and did not in doc_texts:
            doc_texts[did] = txt
    if isinstance(expected_documents, Mapping):
        for did, sdoc in expected_documents.items():
            txt = str(getattr(sdoc, "text", "") or "")
            if str(did) and txt and str(did) not in doc_texts:
                doc_texts[str(did)] = txt

    unit_spans_by_doc: dict[str, list[tuple[int, int]]] = {}
    doc_pred_spans: dict[str, list[tuple[str, int, int]]] = {}
    doc_pred_sets: dict[str, set[str]] = {}
    stored_fe_map = getattr(evidence, "field_evidences", {}) or {}
    for u in admitted:
        u_pred = str((getattr(u, "proposition", {}) or {}).get("predicate", "") or "")
        for ref in getattr(u, "evidence", ()) or ():
            r_doc = str(getattr(ref, "document_id", "") or "")
            r_s = getattr(ref, "start", None)
            r_e = getattr(ref, "end", None)
            if r_doc and u_pred:
                doc_pred_sets.setdefault(r_doc, set()).add(u_pred)
            if r_doc and r_s is not None and r_e is not None:
                is_s, ie_e = int(r_s), int(r_e)
                unit_spans_by_doc.setdefault(r_doc, []).append((is_s, ie_e))
                if u_pred:
                    doc_pred_spans.setdefault(r_doc, []).append((u_pred, is_s, ie_e))
        fe_for_unit = stored_fe_map.get(str(u.unit_id), {}) if isinstance(stored_fe_map, Mapping) else {}
        if isinstance(fe_for_unit, Mapping):
            for fname, fref in fe_for_unit.items():
                if not str(fname).startswith("value[") or not isinstance(fref, Mapping):
                    continue
                r_doc = str(fref.get("document_id", "") or "")
                r_s = fref.get("start")
                r_e = fref.get("end")
                if r_doc and r_s is not None and r_e is not None:
                    unit_spans_by_doc.setdefault(r_doc, []).append((int(r_s), int(r_e)))

    # Cross-check reader-derived obligations against sentence/quantity divergences
    reader_obligations = list(obligations)
    differing_spans_by_doc = _differing_spans_from_obligations(
        reader_obligations,
        admitted,
        stored_fe_map,
    )
    unit_span_index = _build_merged_span_index(unit_spans_by_doc)
    diff_span_index = _build_merged_span_index(differing_spans_by_doc)

    # Pre-index reader obligations by document span for O(log R) divergence UID lookup.
    rob_unbounded_uids: dict[str, set[str]] = {}
    _rob_bounded_raw: dict[str, list[tuple[int, int, tuple[str, ...]]]] = {}
    for rob in reader_obligations:
        r_params = rob.parameters or {}
        r_uids = tuple(str(uid) for uid in (r_params.get("unit_ids") or ()))
        if not r_uids:
            continue
        for sp in r_params.get("evidence_spans") or ():
            sp_doc = str(sp.get("document_id", "") or "")
            if not sp_doc:
                continue
            sp_s, sp_e = sp.get("start"), sp.get("end")
            if sp_s is None or sp_e is None:
                rob_unbounded_uids.setdefault(sp_doc, set()).update(r_uids)
            else:
                _rob_bounded_raw.setdefault(sp_doc, []).append((int(sp_s), int(sp_e), r_uids))

    _ROB_BLK = 64
    rob_bounded_blocks: dict[str, list[tuple[int, int, list[tuple[int, int, tuple[str, ...]]]]]] = {}
    for sp_doc, entries in _rob_bounded_raw.items():
        entries.sort(key=lambda x: x[0])
        blocks = []
        for idx in range(0, len(entries), _ROB_BLK):
            blk = entries[idx : idx + _ROB_BLK]
            blocks.append((blk[0][0], max(item[1] for item in blk), blk))
        rob_bounded_blocks[sp_doc] = blocks

    def _collect_rob_uids_in_span(doc_id: str, span: tuple[int, int] | None, out: set[str]) -> None:
        if not doc_id or span is None:
            return
        qs, qe = int(span[0]), int(span[1])
        out.update(rob_unbounded_uids.get(doc_id, ()))
        for b_min_s, b_max_e, blk in rob_bounded_blocks.get(doc_id, ()):
            if b_min_s >= qe:
                break
            if b_max_e <= qs:
                continue
            for es, ee, euids in blk:
                if es >= qe:
                    break
                if ee > qs:
                    out.update(euids)
    div_doc_meta: dict[str, dict[str, Any]] = {}
    if divergence_collection is not None and divergence_collection.document_metadata:
        for did, meta in divergence_collection.document_metadata.items():
            div_doc_meta[str(did)] = dict(meta)
    for did, st in (getattr(evidence, "document_read_statuses", {}) or {}).items():
        if str(did) not in div_doc_meta and isinstance(st, Mapping):
            div_doc_meta[str(did)] = {"role": st.get("role", ""), "title": st.get("title", "")}
    gap_divergences: list[dict[str, Any]] = []
    seen_gap_keys: set[tuple[Any, ...]] = set()
    sec_div_records: list[dict[str, Any]] = []
    for div in eff_divergences or ():
        pri_id = str(getattr(div, "primary_source_id", "") or "")
        sec_id = str(getattr(div, "secondary_source_id", "") or "")
        sec_key = str(getattr(div, "section_key", "") or "")
        sec_label = str(getattr(div, "section_label", "") or "")
        pri_span = getattr(div, "primary_span", None)
        sec_span = getattr(div, "secondary_span", None)
        pri_quants = list(getattr(div, "primary_quantities", ()) or ())
        sec_quants = list(getattr(div, "secondary_quantities", ()) or ())
        div_rec = {
            "section_key": sec_key,
            "section_label": sec_label,
            "primary_source_id": pri_id,
            "secondary_source_id": sec_id,
            "primary_span": list(pri_span) if pri_span is not None else None,
            "secondary_span": list(sec_span) if sec_span is not None else None,
            "primary_quantities": pri_quants,
            "secondary_quantities": sec_quants,
            "reason": str(getattr(div, "reason", "") or ""),
        }
        sec_div_records.append(div_rec)
        for gap_item in _divergence_gaps_for_item(
            div,
            doc_texts,
            admitted,
            unit_spans_by_doc,
            reader_obligations,
            differing_spans_by_doc=differing_spans_by_doc,
            doc_metadata=div_doc_meta,
            unit_span_index=unit_span_index,
            diff_span_index=diff_span_index,
            doc_pred_spans=doc_pred_spans,
            doc_pred_sets=doc_pred_sets,
        ):
            gkey = (
                str(gap_item.get("kind", "")),
                str(gap_item.get("document_id", "")),
                int(gap_item.get("offset", 0)),
                str(gap_item.get("token", "")),
            )
            if gkey not in seen_gap_keys:
                seen_gap_keys.add(gkey)
                gap_divergences.append(gap_item)

        # Register every v3 section divergence as an origin='corpus' obligation (family='section_divergence')
        div_ob_id = f"corpus::section_divergence::{sec_key}::{digest([pri_id, sec_id, sec_key])[:12]}"
        if div_ob_id not in seen_ids:
            seen_ids.add(div_ob_id)
            div_spans: list[dict[str, Any]] = []
            if pri_id and pri_span is not None:
                div_spans.append({
                    "document_id": pri_id,
                    "start": int(pri_span[0]),
                    "end": int(pri_span[1]),
                    "quote": sec_label,
                })
            if sec_id and sec_span is not None:
                div_spans.append({
                    "document_id": sec_id,
                    "start": int(sec_span[0]),
                    "end": int(sec_span[1]),
                    "quote": sec_label,
                })
            ov_uids: set[str] = set()
            _collect_rob_uids_in_span(pri_id, pri_span, ov_uids)
            _collect_rob_uids_in_span(sec_id, sec_span, ov_uids)
            overlapping_uids = sorted(ov_uids)
            obligations.append(
                VerificationObligation(
                    obligation_id=div_ob_id,
                    family="section_divergence",
                    description=(
                        f"Cover divergent section '{sec_label}' between '{pri_id}' and '{sec_id}' "
                        "in both verified graph leaves and linked response text."
                    ),
                    critical=True,
                    origin="corpus",
                    parameters={
                        "corpus_obligation_kind": "section_divergence",
                        "section_key": sec_key,
                        "section_label": sec_label,
                        "primary_source_id": pri_id,
                        "secondary_source_id": sec_id,
                        "primary_span": list(pri_span) if pri_span is not None else None,
                        "secondary_span": list(sec_span) if sec_span is not None else None,
                        "primary_quantities": pri_quants,
                        "secondary_quantities": sec_quants,
                        "unit_ids": overlapping_uids,
                        "evidence_spans": div_spans,
                    },
                )
            )

    admitted_refs = tuple(
        {
            "unit_id": str(u.unit_id),
            "document_id": str(sp.get("document_id", "")),
            "start": sp.get("start"),
            "end": sp.get("end"),
            "entity": str(u.proposition.get("entity", "")),
            "predicate": str(u.proposition.get("predicate", "")),
            "value": u.proposition.get("value"),
            "unit": u.proposition.get("unit"),
            "condition_status": str(u.proposition.get("condition_status", "unconditional")),
        }
        for u in admitted
        for sp in _get_unit_spans(u)
    )

    return {
        "entity": target_entity,
        "required_predicates": req_preds,
        "obligations": tuple(obligations),
        "document_ids": doc_ids,
        "unit_count": len(admitted),
        "has_material_conflicts": has_conflicts,
        "has_authority_overrides": has_overrides,
        "gap_divergences": tuple(gap_divergences),
        "section_divergences": tuple(sec_div_records),
        "admitted_unit_refs": admitted_refs,
    }


def _gap_key(gap: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(gap.get("document_id", "")),
        int(gap.get("offset", 0)),
        str(gap.get("token", "")),
        str(gap.get("kind", "")),
    )


def _execute_targeted_rereads(
    evidence: EvidenceLayer,
    vocabulary: Vocabulary,
    *,
    initial_gaps: Sequence[Mapping[str, Any]],
    rereader: Any,
    config: TargetedRereadConfig,
    expected_documents: Mapping[str, Any] | Sequence[str] | None,
    divergence_collection: ReferenceDivergenceCollection | None,
) -> dict[str, Any]:
    """Run blind bounded second reads ONLY on uncovered gap sentences and return provenance."""
    docs_by_id: dict[str, SourceDocument] = {d.document_id: d for d in evidence.documents}
    if isinstance(expected_documents, Mapping):
        for did, sdoc in expected_documents.items():
            if isinstance(sdoc, SourceDocument) and did not in docs_by_id:
                docs_by_id[str(did)] = sdoc
    if divergence_collection is not None and divergence_collection.document_texts:
        for did, txt in divergence_collection.document_texts.items():
            if did not in docs_by_id and txt:
                docs_by_id[str(did)] = SourceDocument(document_id=str(did), text=str(txt))

    # Group uncovered gaps deterministically by (document_id, sentence_span)
    grouped: dict[tuple[str, tuple[int, int]], list[dict[str, Any]]] = {}
    for g in initial_gaps:
        doc_id = str(g.get("document_id") or "")
        s_span_raw = g.get("sentence_span") or g.get("span") or [0, 0]
        s_span = (int(s_span_raw[0]), int(s_span_raw[1]))
        grouped.setdefault((doc_id, s_span), []).append(dict(g))

    ordered_targets = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1][0], kv[0][1][1]))
    attempts_by_doc: dict[str, int] = {}
    total_attempts = 0
    attempt_records: list[dict[str, Any]] = []
    reread_ident: dict[str, Any] = dict(
        getattr(rereader, "reread_identity", None) or getattr(rereader, "identity", {}) or {}
    )
    vocab_desc = vocabulary.describe()
    prof_desc = evidence.profile.describe()

    for (doc_id, s_span), grp_gaps in ordered_targets:
        sdoc = docs_by_id.get(doc_id)
        doc_count = attempts_by_doc.get(doc_id, 0)
        tokens = [str(g.get("token", "")) for g in grp_gaps]
        if sdoc is None:
            attempt_records.append({
                "document_id": doc_id,
                "sentence_span": list(s_span),
                "gap_tokens": tokens,
                "status": "document_unavailable",
                "proposals_returned": 0,
                "proposals_admitted": 0,
                "proposals_rejected": 0,
                "errors": [{"error_type": "SourceDocumentMissing", "document_id": doc_id}],
            })
            continue
        if doc_count >= config.max_attempts_per_document or total_attempts >= config.max_total_attempts:
            attempt_records.append({
                "document_id": doc_id,
                "sentence_span": list(s_span),
                "gap_tokens": tokens,
                "status": "skipped_budget_exceeded",
                "proposals_returned": 0,
                "proposals_admitted": 0,
                "proposals_rejected": 0,
                "errors": [],
            })
            continue

        attempts_by_doc[doc_id] = doc_count + 1
        total_attempts += 1
        sent_text = str(grp_gaps[0].get("sentence") or sdoc.text[s_span[0]:s_span[1]])
        # Blind request: contains ONLY source document slice, gap target, vocabulary, and profile.
        request = {
            "task": "targeted_reread",
            "contract": "claimlayer-v4-reread/1",
            "reader": reread_ident,
            "document": {
                "document_id": sdoc.document_id,
                "revision": sdoc.revision,
                "text": sdoc.text,
            },
            "target": {
                "document_id": doc_id,
                "sentence_span": list(s_span),
                "sentence": sent_text,
                "gaps": [
                    {
                        "kind": str(g.get("kind", "")),
                        "offset": int(g.get("offset", 0)),
                        "span": list(g.get("span") or []),
                        "token": str(g.get("token", "")),
                        "raw_token": str(g.get("raw_token") or g.get("token") or ""),
                    }
                    for g in grp_gaps
                ],
            },
            "vocabulary": vocab_desc,
            "profile": prof_desc,
        }
        try:
            rr_res = rereader.propose_targeted(request)
        except Exception as exc:
            attempt_records.append({
                "document_id": doc_id,
                "sentence_span": list(s_span),
                "gap_tokens": tokens,
                "status": "reread_error",
                "proposals_returned": 0,
                "proposals_admitted": 0,
                "proposals_rejected": 0,
                "errors": [{"error_type": type(exc).__name__, "message": str(exc)}],
            })
            continue

        if isinstance(rr_res, Mapping) and rr_res.get("reread_identity"):
            reread_ident = dict(rr_res["reread_identity"])
        rr_items = list(rr_res.get("items") or ())
        rr_rejected_bind = list(rr_res.get("rejected") or ())
        rr_errors = list(rr_res.get("errors") or ())
        producer = {
            "kind": "targeted_reread",
            "identity": reread_ident,
            "target_document": doc_id,
            "target_span": list(s_span),
        }
        for bad in rr_rejected_bind:
            evidence._reject(
                bad.get("proposal"),
                bad.get("basis"),
                producer,
                {"reason": bad.get("reason", "reader_contract"), "path": "", "detail": bad.get("detail")},
                {"proposals_rejected": 0, "rejection_reasons": {}},
            )
        admit_sum = evidence.admit_reread_items(
            rr_items,
            producer=producer,
            structural_reliability=getattr(rereader, "structural_reliability", None),
        )
        attempt_records.append({
            "document_id": doc_id,
            "sentence_span": list(s_span),
            "gap_tokens": tokens,
            "status": "ok" if not rr_errors else "reread_error",
            "proposals_returned": len(rr_items) + len(rr_rejected_bind),
            "proposals_admitted": int(admit_sum["units_admitted"]),
            "proposals_rejected": len(rr_rejected_bind) + int(admit_sum["proposals_rejected"]),
            "rejection_reasons": dict(admit_sum["rejection_reasons"]),
            "errors": rr_errors,
        })

    return {
        "performed": True,
        "reader": reread_ident,
        "budget_per_document": config.max_attempts_per_document,
        "budget_total": config.max_total_attempts,
        "attempts_total": total_attempts,
        "attempts_by_document": attempts_by_doc,
        "attempts": attempt_records,
    }


def build_corpus_reference_register(
    evidence: EvidenceLayer | None,
    vocabulary: Vocabulary,
    *,
    question: str | Mapping[str, Any] | None = None,
    entity: str | None = None,
    required_predicates: Sequence[str] | None = None,
    comparison_divergences: Sequence[Any] = (),
    divergence_collection: ReferenceDivergenceCollection | None = None,
    min_relevant_authority_rank: int = 1,
    expected_documents: Mapping[str, Any] | Sequence[str] | None = None,
    excluded_documents: Sequence[Mapping[str, Any]] = (),
    domain_profile_fingerprint: str | None = None,
    reread_config: TargetedRereadConfig | None = None,
    reread_reader: Any = None,
) -> CorpusReferenceRegister:
    """Build a deterministic reference register and ``origin='corpus'`` obligations.

    Does not inspect the candidate response, justification graph, assertions, or
    any agent-supplied metadata flags.
    """
    if isinstance(comparison_divergences, ReferenceDivergenceCollection):
        divergence_collection = comparison_divergences
        comparison_divergences = ()
    eff_excluded = tuple(
        dict(e)
        for e in (
            excluded_documents
            or (divergence_collection.excluded_documents if divergence_collection is not None else ())
        )
    )
    eff_profile_fp = (
        domain_profile_fingerprint
        or (divergence_collection.domain_profile_fingerprint if divergence_collection is not None else None)
    )
    eff_divergences = (
        divergence_collection.divergences
        if divergence_collection is not None and not comparison_divergences
        else comparison_divergences
    )

    if not isinstance(evidence, EvidenceLayer):
        return CorpusReferenceRegister(
            available=False,
            entity=entity,
            reason="REFERENCE_REGISTER_MISSING: no EvidenceLayer available.",
            excluded_documents=eff_excluded,
            domain_profile_fingerprint=eff_profile_fp,
        )

    doc_statuses = {k: dict(v) for k, v in (getattr(evidence, "document_read_statuses", {}) or {}).items()}
    if expected_documents is not None:
        if isinstance(expected_documents, Mapping):
            for doc_id, sdoc in expected_documents.items():
                did = str(doc_id)
                if did not in doc_statuses:
                    rank = int(getattr(sdoc, "authority_rank", min_relevant_authority_rank))
                    rev = str(getattr(sdoc, "revision", "") or "")
                    doc_statuses[did] = {
                        "document_id": did,
                        "revision": rev,
                        "authority_rank": rank,
                        "status": "not_read",
                        "errors": [],
                    }
        else:
            for doc_id in expected_documents:
                did = str(doc_id)
                if did not in doc_statuses:
                    doc_statuses[did] = {
                        "document_id": did,
                        "revision": "",
                        "authority_rank": int(min_relevant_authority_rank),
                        "status": "not_read",
                        "errors": [],
                    }
    relevant_incomplete = tuple(
        sorted(
            doc_id
            for doc_id, st in doc_statuses.items()
            if int(st.get("authority_rank", 0)) >= int(min_relevant_authority_rank)
            and st.get("status") != "read_ok"
        )
    )
    has_read_failures = any(
        st.get("status") == "read_failed"
        for doc_id, st in doc_statuses.items()
        if int(st.get("authority_rank", 0)) >= int(min_relevant_authority_rank)
    )
    has_reader_incomplete = bool(
        has_read_failures or (relevant_incomplete and getattr(evidence, "reader", None) is not None)
    )

    state = _compute_register_state(
        evidence,
        vocabulary,
        question=question,
        entity=entity,
        required_predicates=required_predicates,
        eff_divergences=eff_divergences,
        divergence_collection=divergence_collection,
        expected_documents=expected_documents,
    )
    if state is None:
        return CorpusReferenceRegister(
            available=False,
            entity=entity,
            reason="REFERENCE_REGISTER_MISSING: EvidenceLayer has no admitted units.",
            incomplete=has_reader_incomplete,
            incomplete_documents=relevant_incomplete if has_reader_incomplete else (),
            document_statuses=doc_statuses,
            excluded_documents=eff_excluded,
            domain_profile_fingerprint=eff_profile_fp,
        )

    eff_config = reread_config if reread_config is not None else TargetedRereadConfig()
    eff_rereader = reread_reader if reread_reader is not None else getattr(evidence, "reader", None)
    reread_supported = (
        eff_rereader is not None
        and callable(getattr(eff_rereader, "propose_targeted", None))
        and bool(getattr(eff_rereader, "has_reread_support", True))
    )
    initial_gaps = tuple(state["gap_divergences"])
    reread_prov: dict[str, Any] = {
        "performed": False,
        "initial_gap_count": len(initial_gaps),
        "closed_gap_count": 0,
        "open_gap_count": len(initial_gaps),
        "closed_gaps": [],
        "open_gaps": [
            {"document_id": g.get("document_id"), "offset": g.get("offset"), "token": g.get("token"), "kind": g.get("kind")}
            for g in initial_gaps
        ],
    }

    if eff_config.enabled and initial_gaps and reread_supported:
        prov = _execute_targeted_rereads(
            evidence,
            vocabulary,
            initial_gaps=initial_gaps,
            rereader=eff_rereader,
            config=eff_config,
            expected_documents=expected_documents,
            divergence_collection=divergence_collection,
        )
        new_state = _compute_register_state(
            evidence,
            vocabulary,
            question=question,
            entity=entity,
            required_predicates=required_predicates,
            eff_divergences=eff_divergences,
            divergence_collection=divergence_collection,
            expected_documents=expected_documents,
        )
        if new_state is not None:
            state = new_state
        post_gaps = tuple(state["gap_divergences"])
        post_keys = {_gap_key(g) for g in post_gaps}
        closed = [g for g in initial_gaps if _gap_key(g) not in post_keys]
        reread_prov = {
            **prov,
            "initial_gap_count": len(initial_gaps),
            "closed_gap_count": len(closed),
            "open_gap_count": len(post_gaps),
            "closed_gaps": [
                {"document_id": g.get("document_id"), "offset": g.get("offset"), "token": g.get("token"), "kind": g.get("kind")}
                for g in closed
            ],
            "open_gaps": [
                {"document_id": g.get("document_id"), "offset": g.get("offset"), "token": g.get("token"), "kind": g.get("kind")}
                for g in post_gaps
            ],
        }

    return CorpusReferenceRegister(
        available=True,
        entity=state["entity"],
        required_predicates=state["required_predicates"],
        obligations=state["obligations"],
        document_ids=state["document_ids"],
        unit_count=state["unit_count"],
        has_material_conflicts=state["has_material_conflicts"],
        has_authority_overrides=state["has_authority_overrides"],
        reason="Corpus reference register built deterministically from EvidenceLayer.",
        incomplete=bool(relevant_incomplete),
        incomplete_documents=relevant_incomplete,
        document_statuses=doc_statuses,
        gap_divergences=state["gap_divergences"],
        section_divergences=state["section_divergences"],
        admitted_unit_refs=state["admitted_unit_refs"],
        excluded_documents=eff_excluded,
        domain_profile_fingerprint=eff_profile_fp,
        reread_provenance=reread_prov,
    )


_GOVERNANCE_DISCLOSURE_RE = re.compile(
    r"\b(?:governs?|governed\s+by|governing|controls?|controlled\s+by|controlling|"
    r"supersedes?|superseded\s+by|overridden\s+by|prevails?|prevailing|"
    r"higher\s+authority|priority\s+over|takes?\s+precedence|according\s+to\s+[^.;,]{1,60}\s+over)\b",
    re.IGNORECASE,
)
_GOVERNANCE_ACTIVE_RE = re.compile(
    r"\b(?:governs?|governing|controls?|controlling|supersedes?|superseding|prevails?|prevailing|takes?\s+precedence|priority)\b",
    re.IGNORECASE,
)
_GOVERNANCE_PASSIVE_RE = re.compile(
    r"\b(?:governed|controlled|superseded|overridden|prevailed)\s+by\b",
    re.IGNORECASE,
)
_UNRESOLVED_DISCLOSURE_RE = re.compile(
    r"\b(?:unresolved|conflict(?:ing)?|discrepanc(?:y|ies)|contradict(?:s|ory|ion)?|"
    r"inconsistent|cannot\s+be\s+resolved|not\s+resolved|unclear\s+which)\b",
    re.IGNORECASE,
)
_STRICT_UNRESOLVED_RE = re.compile(
    r"\b(?:unresolved|unresolvable|cannot\s+be\s+resolved|not\s+resolved|unclear\s+which|"
    r"neither\s+(?:source|document)\s+governs|no\s+governing\s+(?:source|document))\b",
    re.IGNORECASE,
)
_DOC_EXT_RE = re.compile(r"\.(?:docx|pdf|xlsx|eml|txt|md|json)$", re.IGNORECASE)
_GENERIC_DOC_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "document", "source", "text", "notes", "file", "sheet",
})


def _doc_stem_words(doc_id: str, title: str = "", role: str = "") -> list[str]:
    stem = _DOC_EXT_RE.sub("", str(doc_id or ""))
    raw = f"{stem} {title} {role}".strip()
    return [w.lower() for w in re.findall(r"[A-Za-z]+", raw) if w]


def _pair_doc_aliases(
    doc_a: str,
    doc_b: str,
    all_doc_statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build pair-disjoint surface aliases for two documents purely from structural metadata."""
    statuses = all_doc_statuses or {}
    # Count word frequency across all registered documents so shared deal/prefix words drop automatically
    word_df: dict[str, int] = {}
    for did, meta in statuses.items():
        title = str((meta or {}).get("title", "") or "")
        role = str((meta or {}).get("role", "") or "")
        for w in set(_doc_stem_words(did, title, role)):
            word_df[w] = word_df.get(w, 0) + 1

    def _aliases_for(did: str, other_did: str) -> tuple[str, ...]:
        meta = statuses.get(did, {}) or {}
        other_meta = statuses.get(other_did, {}) or {}
        title = str(meta.get("title", "") or "")
        other_title = str(other_meta.get("title", "") or "")
        role = str(meta.get("role", "") or "")
        other_role = str(other_meta.get("role", "") or "")
        words = _doc_stem_words(did, title, role)
        other_words = set(_doc_stem_words(other_did, other_title, other_role))
        stem_norm = " ".join(_doc_stem_words(did))
        other_stem_norm = " ".join(_doc_stem_words(other_did, other_title, other_role))
        cands: list[str] = [did.lower()]
        if stem_norm and stem_norm != other_stem_norm:
            cands.append(stem_norm)
        if title.strip():
            cands.append(title.strip().lower())
        role_words = [w for w in re.findall(r"[A-Za-z]+", role.lower()) if w]
        if role_words:
            cands.append(" ".join(role_words))
            if len(role_words) >= 2:
                cands.append(" ".join(role_words[:2]))
        unique_words = [
            w for w in words
            if w not in other_words and w not in _GENERIC_DOC_WORDS and not w.isdigit()
        ]
        # Phrases of consecutive unique words (e.g. "offering memorandum", "draft oc", "master agreement")
        if len(unique_words) >= 2:
            cands.append(" ".join(unique_words))
            acronym = "".join(w[0] for w in unique_words if w.isalpha())
            if len(acronym) == 2 and acronym not in other_words:
                cands.append(acronym)
        for idx, w in enumerate(unique_words):
            if len(w) >= 4 and word_df.get(w, 0) <= 1 and w not in other_words:
                cands.append(w)
            if idx + 1 < len(unique_words):
                bi = f"{w} {unique_words[idx + 1]}"
                cands.append(bi)
        # Keep only candidates that do not match inside the other document's stem words
        filtered: list[str] = []
        seen_c: set[str] = set()
        for c in sorted(set(cands), key=len, reverse=True):
            c = c.strip()
            if not c or c in seen_c:
                continue
            if re.search(rf"\b{re.escape(c)}\b", other_stem_norm):
                continue
            seen_c.add(c)
            filtered.append(c)
        return tuple(filtered)

    return _aliases_for(doc_a, doc_b), _aliases_for(doc_b, doc_a)


def _text_matches_alias(text: str, aliases: Sequence[str]) -> bool:
    if not text or not aliases:
        return False
    low = text.lower()
    return any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases)


def _value_in_segment_with_unit(val: Any, seg: str) -> bool:
    """Check whether *val* (with its unit suffix if present) appears in *seg* without cross-unit false hits."""
    if val is None or not seg:
        return False
    unit_str = ""
    raw_val = val
    if isinstance(val, Mapping) and "value" in val:
        raw_val = val.get("value")
        unit_str = str(val.get("unit") or "").lower()
    if not _value_in_text(raw_val, seg):
        return False
    if not unit_str:
        return True
    try:
        dec = Decimal(str(raw_val).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return True
    # Check at least one numeric match in seg has compatible trailing/leading unit context
    for m in re.finditer(r"([$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|per\s+annum\b|x\b|bps?\b|business\s+days?|calendar\s+days?|days?|usd\b)?", seg, re.IGNORECASE):
        num_raw = m.group(1).lstrip("$€£ ").replace(",", "")
        try:
            if Decimal(num_raw) != dec:
                continue
        except (InvalidOperation, ValueError):
            continue
        has_dollar = m.group(1).lstrip().startswith(("$", "€", "£"))
        sfx = (m.group(2) or "").strip().lower()
        if "percent" in unit_str:
            if sfx == "x" or has_dollar or sfx == "usd" or "day" in sfx or sfx.startswith("bp"):
                continue
            return True
        if unit_str == "multiple":
            if sfx in ("%", "percent", "per annum") or has_dollar or sfx == "usd" or "day" in sfx:
                continue
            return True
        if "day" in unit_str:
            if sfx in ("%", "percent", "per annum", "x") or has_dollar or sfx == "usd":
                continue
            return True
        if unit_str in ("usd", "usd_per_year"):
            if sfx in ("%", "percent", "x") or "day" in sfx:
                continue
            return True
        return True
    return False


def _check_conflict_disclosure_content(
    *,
    text: str,
    obligation_id: str,
    predicate: str,
    winner_doc: str,
    winner_val: Any,
    winner_rank: int,
    loser_doc: str,
    loser_val: Any,
    loser_rank: int,
    doc_statuses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Verify directional governance and value-to-document attribution in conflict prose."""
    if not text or not winner_doc or not loser_doc or winner_doc == loser_doc:
        return []
    win_aliases, lose_aliases = _pair_doc_aliases(winner_doc, loser_doc, doc_statuses)
    content_issues: list[dict[str, Any]] = []
    unequal_ranks = winner_rank != loser_rank

    if unequal_ranks:
        # 1a. Declaring "unresolved" when authority_rank decides a winner is invalid
        if _STRICT_UNRESOLVED_RE.search(text):
            content_issues.append({
                "code": "GOVERNING_SOURCE_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "expected_governing_document": winner_doc,
                "winner_authority_rank": winner_rank,
                "loser_authority_rank": loser_rank,
                "message": (
                    f"Text declares '{predicate}' conflict unresolved, but '{winner_doc}' "
                    f"(authority_rank={winner_rank}) supersedes '{loser_doc}' (authority_rank={loser_rank})."
                ),
                "remediation": (
                    f"State that '{winner_doc}' (authority_rank={winner_rank}) governs over "
                    f"'{loser_doc}' (authority_rank={loser_rank}); do not claim it is unresolved."
                ),
            })

        # 1b. Check grammatical governance claims in sentences/clauses
        clauses = [c.strip() for c in re.split(r"[.;\n]+", text) if c.strip()]
        gov_mismatch_found = False
        for cl in clauses:
            for mp in _GOVERNANCE_PASSIVE_RE.finditer(cl):
                after = re.split(r"[,;:]|\bover\b", cl[mp.end() :], maxsplit=1)[0]
                if _text_matches_alias(after, lose_aliases) and not _text_matches_alias(after, win_aliases):
                    gov_mismatch_found = True
                    break
            if gov_mismatch_found:
                break
            for ma in _GOVERNANCE_ACTIVE_RE.finditer(cl):
                prefix = cl[: ma.start()]
                if prefix.rstrip().endswith(("is", "are", "was", "were", "be", "been")):
                    continue
                lhs = re.split(r"[:,;]|(?:\b(?:whereas|while|however|although|but)\b)", prefix)[-1]
                # If comma separates independent subject before active governance, keep last comma-segment containing a doc alias
                comma_parts = [p.strip() for p in lhs.split(",") if p.strip()]
                subj_span = lhs
                for part in reversed(comma_parts):
                    if _text_matches_alias(part, win_aliases) or _text_matches_alias(part, lose_aliases):
                        subj_span = part
                        break
                if _text_matches_alias(subj_span, lose_aliases) and not _text_matches_alias(subj_span, win_aliases):
                    gov_mismatch_found = True
                    break
            if gov_mismatch_found:
                break
        if gov_mismatch_found:
            content_issues.append({
                "code": "GOVERNING_SOURCE_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "expected_governing_document": winner_doc,
                "claimed_subordinate_document": loser_doc,
                "message": (
                    f"Response claims subordinate document '{loser_doc}' (rank={loser_rank}) governs "
                    f"instead of '{winner_doc}' (rank={winner_rank}) for '{predicate}'."
                ),
                "remediation": (
                    f"Correct governing attribution: state that '{winner_doc}' governs over '{loser_doc}'."
                ),
            })

    # 2. Check value-to-document attribution across local segments & parenthetical attachments
    if not _value_in_text(winner_val, str(loser_val or "")) or not _value_in_text(loser_val, str(winner_val or "")):
        segments = [
            s.strip()
            for s in re.split(
                r"[;\n()]+|(?<!\d)[.,](?!\d)|\.(?=\s|$)|\b(?:whereas|while|however|although|but|vs\.?|versus)\b|"
                r"\band\b(?=\s*(?:in|per|under|according|the\s+[a-z0-9_-]+\s+(?:states?|provides?|proposes?|specifies?|carries?|shows?|gives?)))",
                text,
                flags=re.IGNORECASE,
            )
            if s.strip()
        ]
        # Also include tight `<doc> (<val>)` pairs where parenthetical carries one value without naming the other doc
        for pm in re.finditer(r"([^.;\n()]{3,80})\(([^()]{1,60})\)", text):
            before, inside = pm.group(1), pm.group(2)
            tail = re.split(r"[,:;]|\b(?:over|vs\.?|versus|than|and)\b", before, flags=re.IGNORECASE)[-1]
            if (
                (_text_matches_alias(tail, win_aliases) or _text_matches_alias(tail, lose_aliases))
                and not (_text_matches_alias(inside, win_aliases) or _text_matches_alias(inside, lose_aliases))
            ):
                segments.append(f"{tail} {inside}")

        val_mismatch_found = False
        for seg in segments:
            seg_w = _text_matches_alias(seg, win_aliases)
            seg_l = _text_matches_alias(seg, lose_aliases)
            has_w_val = _value_in_segment_with_unit(winner_val, seg)
            has_l_val = _value_in_segment_with_unit(loser_val, seg)
            if seg_w and not seg_l and has_l_val and not has_w_val:
                val_mismatch_found = True
                break
            if seg_l and not seg_w and has_w_val and not has_l_val:
                val_mismatch_found = True
                break
        if val_mismatch_found:
            content_issues.append({
                "code": "VALUE_ATTRIBUTION_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "winner_document_id": winner_doc,
                "winner_value": winner_val,
                "loser_document_id": loser_doc,
                "loser_value": loser_val,
                "message": (
                    f"Response swaps or misattributes conflicting '{predicate}' values across "
                    f"'{winner_doc}' ({winner_val}) and '{loser_doc}' ({loser_val})."
                ),
                "remediation": (
                    f"Attribute '{winner_val}' strictly to '{winner_doc}' and '{loser_val}' strictly to '{loser_doc}'."
                ),
            })

    return content_issues


def _value_in_text(val: Any, text: str) -> bool:
    if val is None or not text:
        return False
    if isinstance(val, Mapping) and "value" in val:
        val = val["value"]
    raw = str(val).strip()
    if not raw:
        return False
    try:
        dec = Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return raw.lower() in text.lower()
    # Numeric values must match an exact parsed numeric token (never substring inside "13" or "50")
    for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        tok = m.group(0).replace(",", "")
        try:
            if Decimal(tok) == dec:
                return True
        except (InvalidOperation, ValueError):
            continue
    return False


def _claim_matches_obligation(node: Any, ob: VerificationObligation) -> bool:
    if getattr(node, "kind", "") == "opinion" or getattr(node, "assertion_kind", "") == "opinion":
        return False
    params = ob.parameters or {}
    ob_unit_ids = set(str(u) for u in (params.get("unit_ids") or ()))
    node_unit_ids = set(str(u) for u in (getattr(node, "evidence_unit_ids", ()) or ()))
    if ob_unit_ids and (ob_unit_ids & node_unit_ids):
        return True
    ob_ent = str(params.get("entity", ""))
    ob_pred = str(params.get("predicate", ""))
    if ob_pred and label_key(str(getattr(node, "predicate", ""))) != label_key(ob_pred):
        return False
    if ob_ent and getattr(node, "subject", "") and label_key(str(node.subject)) != label_key(ob_ent):
        # Allow entity alias or opposing conflict participant when predicate matches and evidence spans overlap.
        for ref in getattr(node, "evidence", ()) or ():
            r_doc = str(getattr(ref, "document_id", "") or "")
            r_start = getattr(ref, "start", None)
            r_end = getattr(ref, "end", None)
            if r_doc and _obligation_overlaps_doc_span(
                ob, r_doc, (int(r_start), int(r_end)) if r_start is not None and r_end is not None else None
            ):
                return True
        return False
    return bool(ob_pred)


@dataclass(frozen=True)
class CorpusCoverageResult:
    """Result of checking a JustificationGraph + candidate text against CorpusReferenceRegister."""

    total_obligations: int
    covered_ids: tuple[str, ...]
    uncovered_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    all_obligation_ids: tuple[str, ...]
    issues: tuple[dict[str, Any], ...]

    @property
    def covered_count(self) -> int:
        return len(self.covered_ids)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_ids)


def verify_corpus_obligations_coverage(
    register: CorpusReferenceRegister,
    graph: Any,
    candidate_response: str | None,
    *,
    refuted_claim_ids: set[str] | frozenset[str] = frozenset(),
    correspondence_report: Mapping[str, Any] | None = None,
) -> CorpusCoverageResult:
    """Enforce closure rule: every corpus obligation must be a verified graph leaf AND linked in text."""
    if not register.available:
        init_issues: list[dict[str, Any]] = []
        if register.incomplete:
            init_issues.append({
                "code": "REFERENCE_REGISTER_INCOMPLETE",
                "source_layer": "v4",
                "repairable": False,
                "affected_documents": list(register.incomplete_documents),
                "document_statuses": {k: dict(v) for k, v in register.document_statuses.items()},
                "message": (
                    "Corpus reference register is incomplete; relevant document(s) did not complete reading: "
                    f"{list(register.incomplete_documents)}."
                ),
                "remediation": "Ensure all relevant corpus documents reach read_ok before evaluating.",
            })
        else:
            init_issues.append({
                "code": "REFERENCE_REGISTER_MISSING",
                "source_layer": "v4",
                "repairable": False,
                "message": register.reason or "No corpus reference register is available for this document.",
                "remediation": "Configure a reader on EvidenceLayer or provide corpus readings before evaluating.",
            })
        return CorpusCoverageResult(
            total_obligations=0,
            covered_ids=(),
            uncovered_ids=(),
            rejected_ids=(),
            all_obligation_ids=(),
            issues=tuple(init_issues),
        )

    rejected_map: Mapping[str, str] = getattr(graph, "rejected_obligations", {}) or {}
    issues: list[dict[str, Any]] = []
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    rejected_ids: list[str] = []

    if register.incomplete:
        issues.append({
            "code": "REFERENCE_REGISTER_INCOMPLETE",
            "source_layer": "v4",
            "repairable": False,
            "affected_documents": list(register.incomplete_documents),
            "document_statuses": {k: dict(v) for k, v in register.document_statuses.items()},
            "message": (
                "Corpus reference register is incomplete; relevant document(s) not in read_ok: "
                f"{list(register.incomplete_documents)}."
            ),
            "remediation": "Re-run corpus reader on failed/unread documents before acceptance.",
        })

    for gap in register.gap_divergences:
        sec_key = str(gap.get("section_key", "unknown"))
        gap_doc = str(gap.get("document_id") or gap.get("secondary_source_id") or gap.get("primary_source_id") or "")
        gap_offset = gap.get("offset")
        gap_token = str(gap.get("token") or "")
        gap_sent = str(gap.get("sentence") or "")
        gap_kind = str(gap.get("kind") or "quantity")
        gap_uid = digest([sec_key, gap_kind, gap_doc, gap_offset, gap_token, gap_sent])[:12]
        issues.append({
            "code": "REFERENCE_REGISTER_GAP",
            "source_layer": "v4",
            "repairable": False,
            "kind": gap_kind,
            "obligation_id": f"corpus::gap::{gap_uid}",
            "document_id": gap_doc,
            "offset": gap_offset,
            "token": gap_token,
            "sentence": gap_sent,
            "section_key": sec_key,
            "section_label": str(gap.get("section_label", sec_key)),
            "gap_divergence": dict(gap),
            "message": (
                f"Cross-check detected uncovered {gap_kind} divergence token '{gap_token}' "
                f"in '{gap_doc}'@{gap_offset} ({gap_sent[:120]!r}) between "
                f"'{gap.get('primary_source_id')}' and '{gap.get('secondary_source_id')}'."
            ),
            "remediation": "Re-extract corpus obligations covering all divergent quantities and clauses before acceptance.",
        })

    # Collect explicit rejections (even if they name '*' or any corpus obligation)
    for ob_id, reason in sorted(rejected_map.items()):
        if ob_id in register.obligation_ids or ob_id == "*":
            rejected_ids.append(ob_id)
            issues.append({
                "code": "CORPUS_OBLIGATION_REJECTED",
                "source_layer": "v4",
                "repairable": False,
                "obligation_id": ob_id,
                "message": f"Agent explicitly rejected corpus obligation '{ob_id}': {reason}",
                "remediation": "Explicitly rejected obligations require human escalation; cannot ACCEPT.",
            })
        else:
            rejected_ids.append(ob_id)
            issues.append({
                "code": "CORPUS_OBLIGATION_REJECTED",
                "source_layer": "v4",
                "repairable": False,
                "obligation_id": ob_id,
                "message": f"Agent explicitly rejected obligation '{ob_id}': {reason}",
                "remediation": "Explicitly rejected obligations require human escalation; cannot ACCEPT.",
            })

    verified_leaves = [
        node
        for cid, node in sorted(getattr(graph, "claims", {}).items())
        if cid not in refuted_claim_ids
        and getattr(node, "kind", "") != "opinion"
        and getattr(node, "assertion_kind", "") != "opinion"
        and (getattr(node, "evidence", ()) or getattr(node, "evidence_unit_ids", ()))
    ]
    text = candidate_response or ""

    for ob in register.obligations:
        if ob.obligation_id in rejected_map or "*" in rejected_map:
            continue
        params = ob.parameters or {}
        kind = str(params.get("corpus_obligation_kind", "mandatory_predicate"))
        spans = list(params.get("evidence_spans") or ())
        matching_leaves = [n for n in verified_leaves if _claim_matches_obligation(n, ob)]

        in_graph = False
        in_text = True if candidate_response is None else False
        extra_content_issues: list[dict[str, Any]] = []

        if kind in ("mandatory_predicate", "authority_override"):
            if kind == "authority_override":
                winner_id = str(params.get("winner_unit_id", ""))
                winner_val = params.get("winner_value")
                loser_val = params.get("loser_value")
                w_doc = str(params.get("winner_document_id", "") or "")
                l_doc = str(params.get("loser_document_id", "") or "")
                w_rank = int(params.get("winner_authority_rank", 0) or 0)
                l_rank = int(params.get("loser_authority_rank", 0) or 0)
                in_graph = any(
                    (winner_id and winner_id in set(getattr(n, "evidence_unit_ids", ()) or ()))
                    or _value_in_text(winner_val, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                if candidate_response is not None:
                    extra_content_issues = _check_conflict_disclosure_content(
                        text=text,
                        obligation_id=ob.obligation_id,
                        predicate=str(params.get("predicate", "")),
                        winner_doc=w_doc,
                        winner_val=winner_val,
                        winner_rank=w_rank,
                        loser_doc=l_doc,
                        loser_val=loser_val,
                        loser_rank=l_rank,
                        doc_statuses=register.document_statuses,
                    )
                if candidate_response is not None and in_graph:
                    has_winner = _value_in_text(winner_val, text)
                    has_loser = _value_in_text(loser_val, text)
                    has_gov = bool(_GOVERNANCE_DISCLOSURE_RE.search(text))
                    has_unres = bool(_UNRESOLVED_DISCLOSURE_RE.search(text))
                    if w_rank != l_rank:
                        in_text = has_winner and (not has_loser or has_gov) and not extra_content_issues
                    else:
                        in_text = has_winner and (not has_loser or has_gov or has_unres) and not extra_content_issues
            else:
                expected_val = params.get("expected_value")
                in_graph = bool(matching_leaves)
                if candidate_response is not None and in_graph:
                    in_text = any(
                        _value_in_text(getattr(n, "value", None), text)
                        for n in matching_leaves
                    ) or _value_in_text(expected_val, text)

        elif kind == "material_conflict":
            conf_vals = list(params.get("conflicting_values") or ())
            if len(conf_vals) >= 2:
                v1, v2 = conf_vals[0].get("value"), conf_vals[1].get("value")
                u1_id, u2_id = str(conf_vals[0].get("unit_id", "")), str(conf_vals[1].get("unit_id", ""))
                covered_v1 = any(
                    u1_id in set(getattr(n, "evidence_unit_ids", ()) or ())
                    or _value_in_text(v1, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                covered_v2 = any(
                    u2_id in set(getattr(n, "evidence_unit_ids", ()) or ())
                    or _value_in_text(v2, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                in_graph = covered_v1 and covered_v2
                if candidate_response is not None:
                    c0, c1 = conf_vals[0], conf_vals[1]
                    r0, r1 = int(c0.get("authority_rank", 0) or 0), int(c1.get("authority_rank", 0) or 0)
                    win_c, lose_c = (c0, c1) if r0 >= r1 else (c1, c0)
                    extra_content_issues = _check_conflict_disclosure_content(
                        text=text,
                        obligation_id=ob.obligation_id,
                        predicate=str(params.get("predicate", "")),
                        winner_doc=str(win_c.get("document_id", "") or ""),
                        winner_val=win_c.get("value"),
                        winner_rank=int(win_c.get("authority_rank", 0) or 0),
                        loser_doc=str(lose_c.get("document_id", "") or ""),
                        loser_val=lose_c.get("value"),
                        loser_rank=int(lose_c.get("authority_rank", 0) or 0),
                        doc_statuses=register.document_statuses,
                    )
                if candidate_response is not None and in_graph:
                    both_in_text = _value_in_text(v1, text) and _value_in_text(v2, text)
                    if r0 != r1:
                        disclosed = bool(_GOVERNANCE_DISCLOSURE_RE.search(text))
                    else:
                        disclosed = bool(
                            _GOVERNANCE_DISCLOSURE_RE.search(text)
                            or _UNRESOLVED_DISCLOSURE_RE.search(text)
                        )
                    in_text = both_in_text and disclosed and not extra_content_issues
            else:
                in_graph = bool(matching_leaves)
                in_text = candidate_response is None or bool(matching_leaves)

        elif kind == "section_divergence":
            pri_id = str(params.get("primary_source_id", "") or "")
            sec_id = str(params.get("secondary_source_id", "") or "")
            pri_sp = tuple(params["primary_span"]) if params.get("primary_span") else None
            sec_sp = tuple(params["secondary_span"]) if params.get("secondary_span") else None

            def _leaf_overlaps_sec(leaf: Any) -> bool:
                for ev in (getattr(leaf, "evidence", ()) or ()):
                    ev_doc = str(ev.get("document_id", "") if isinstance(ev, Mapping) else getattr(ev, "document_id", ""))
                    ev_s = ev.get("start") if isinstance(ev, Mapping) else getattr(ev, "start", None)
                    ev_e = ev.get("end") if isinstance(ev, Mapping) else getattr(ev, "end", None)
                    ev_rng = (int(ev_s), int(ev_e)) if ev_s is not None and ev_e is not None else None
                    if pri_id and _spans_overlap(ev_doc, ev_rng, pri_id, pri_sp):
                        return True
                    if sec_id and _spans_overlap(ev_doc, ev_rng, sec_id, sec_sp):
                        return True
                return False

            div_leaves = [n for n in verified_leaves if _leaf_overlaps_sec(n)]
            in_graph = bool(div_leaves)
            if candidate_response is not None and in_graph:
                div_leaf_ids = {str(getattr(n, "claim_id", "")) for n in div_leaves}
                linked_by_corr = False
                if correspondence_report:
                    for rel_chk in (correspondence_report.get("relation_checks") or ()):
                        if rel_chk.get("status") == "verified" and str(rel_chk.get("matched_claim_id", "")) in div_leaf_ids:
                            linked_by_corr = True
                            break
                    if not linked_by_corr:
                        for cc in (correspondence_report.get("clause_claims") or ()):
                            if not cc.get("is_grounded"):
                                continue
                            cl_low = str(cc.get("clause_text", "")).lower()
                            if any(
                                str(getattr(n, "subject", "")).strip().lower() in cl_low
                                and (
                                    getattr(n, "value", None) is None
                                    or _value_in_text(getattr(n, "value", None), cl_low)
                                )
                                for n in div_leaves
                                if str(getattr(n, "subject", "")).strip()
                            ):
                                linked_by_corr = True
                                break
                in_text = linked_by_corr or any(
                    _value_in_text(getattr(n, "value", None), text) for n in div_leaves
                    if getattr(n, "value", None) is not None
                )

        if in_graph and in_text:
            covered_ids.append(ob.obligation_id)
        else:
            uncovered_ids.append(ob.obligation_id)
            if extra_content_issues:
                issues.extend(extra_content_issues)
            else:
                span_preview = spans[0] if spans else {}
                if kind == "material_conflict":
                    remed = (
                        f"Cover material conflict for '{params.get('predicate')}' on '{params.get('entity')}': "
                        "include both conflicting values in the graph and text, and state which governs or declare the conflict unresolved."
                    )
                else:
                    remed = (
                        f"Cover corpus obligation '{ob.obligation_id}' ({params.get('predicate')} on '{params.get('entity')}') "
                        f"as a verified graph leaf and linked text statement citing source span {span_preview}."
                    )
                issues.append({
                    "code": "CORPUS_OBLIGATION_UNCOVERED",
                    "source_layer": "v4",
                    "repairable": True,
                    "obligation_id": ob.obligation_id,
                    "corpus_obligation_kind": kind,
                    "evidence_spans": spans,
                    "message": (
                        f"Corpus obligation '{ob.obligation_id}' is not covered in "
                        f"{'graph' if not in_graph else 'response text'}."
                    ),
                    "remediation": remed,
                })

    return CorpusCoverageResult(
        total_obligations=len(register.obligations),
        covered_ids=tuple(covered_ids),
        uncovered_ids=tuple(uncovered_ids),
        rejected_ids=tuple(rejected_ids),
        all_obligation_ids=register.obligation_ids,
        issues=tuple(issues),
    )
