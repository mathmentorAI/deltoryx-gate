"""Extraction-fidelity controls for ClaimLayer v4.

Verifies per-field evidence bindings, clause-boundary completeness (ensuring
limiting provisos/exceptions are not dropped by fragment boundaries), and
independent structural checks on proposed propositions. Also provides
dual-reader agreement and correlated-error diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import EvidenceRef
from ..verifiers.quote_quantity import extract_dates, extract_quantities
from .models import EvidenceUnit
from .vocabulary import STANDARD_UNITS, FieldSpec, UnitRegistry, UnmappedProposal, Vocabulary, label_key, normalize_date

_CONDITIONAL_MARKER_RE = re.compile(
    r"\b(?:unless|except(?:\s+for|\s+that|\s+where)?|provided\s+that|subject\s+to(?!\s+a\s+(?:floor|cap|minimum|maximum)\b)|"
    r"notwithstanding|save\s+that|if\s+(?:and\s+only\s+if)?|when|where|in\s+the\s+event|"
    r"for\s+disputed|during\s+force\s+majeure|excluding|other\s+than|"
    r"salvo|excepto|siempre\s+que|a\s+menos\s+que|sujeto\s+a|en\s+caso\s+de)\b",
    re.IGNORECASE,
)

_ISO_DATE_FRAGMENT_RE = re.compile(
    r"(?<![0-9\-])\d{4}(?:-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)?)?(?![0-9\-])"
)

_CLAUSE_NEGATION_RE = re.compile(
    r"\b(?:shall\s+not|may\s+not|must\s+not|cannot|can\s+not|will\s+not|is\s+not|are\s+not|"
    r"in\s+no\s+event|under\s+no\s+circumstances|no\s+longer|never|prohibited|forbidden|"
    r"no\s+podr[aá]|en\s+ning[uú]n\s+caso|prohibido)\b",
    re.IGNORECASE,
)

_BOUND_MARKER_MAP: dict[str, re.Pattern[str]] = {
    "gt": re.compile(r"\b(?:more\s+than|greater\s+than|exceeding|exceeds|above|over|más\s+de|superior\s+a)\b", re.IGNORECASE),
    "ge": re.compile(r"\b(?:at\s+least|not\s+less\s+than|minimum\s+of|or\s+more|al\s+menos|como\s+m[ií]nimo)\b", re.IGNORECASE),
    "lt": re.compile(r"\b(?:less\s+than|fewer\s+than|below|under|menos\s+de|inferior\s+a)\b", re.IGNORECASE),
    "le": re.compile(r"\b(?:at\s+most|not\s+more\s+than|up\s+to|maximum\s+of|or\s+less|como\s+m[aá]ximo|hasta)\b", re.IGNORECASE),
}

_INCREASE_MARKER_RE = re.compile(r"\b(?:increase[ds]?|growth|grew|rose|risen|higher|upward|aument[oó]|subi[oó])\b", re.IGNORECASE)
_DECREASE_MARKER_RE = re.compile(r"\b(?:decrease[ds]?|decline[ds]?|drop(?:ped)?|fell|fallen|lower|downward|reduc(?:ed|tion)|disminuy[oó]|baj[oó])\b", re.IGNORECASE)


@dataclass(frozen=True)
class FieldEvidence:
    """Explicit span binding for an individual proposition field."""

    field: str
    evidence: EvidenceRef


@dataclass(frozen=True)
class FidelityFinding:
    """Deterministic extraction-fidelity issue."""

    code: str
    message: str
    field: str | None = None
    detail: Any = None


def enclosing_sentence(doc_text: str, start: int, end: int) -> tuple[int, int, str]:
    """Return `(sent_start, sent_end, sentence_text)` enclosing `doc_text[start:end]`."""
    n = len(doc_text)
    s = max(0, min(start, n))
    e = max(s, min(end, n))

    sent_start = 0
    for i in range(s - 1, -1, -1):
        ch = doc_text[i]
        if ch in "\n;":
            sent_start = i + 1
            break
        if ch == ".":
            prev_ch = doc_text[i - 1] if i > 0 else ""
            next_ch = doc_text[i + 1] if i + 1 < n else ""
            if not (prev_ch.isdigit() and next_ch.isdigit()):
                sent_start = i + 1
                break

    sent_end = n
    if e > s and doc_text[e - 1] in ".;\n":
        sent_end = e
    else:
        for i in range(e, n):
            ch = doc_text[i]
            if ch in "\n;":
                sent_end = i + 1
                break
            if ch == ".":
                prev_ch = doc_text[i - 1] if i > 0 else ""
                next_ch = doc_text[i + 1] if i + 1 < n else ""
                if not (prev_ch.isdigit() and next_ch.isdigit()):
                    sent_end = i + 1
                    break

    return sent_start, sent_end, doc_text[sent_start:sent_end].strip()


def check_clause_boundary_completeness(
    doc_text: str,
    ref: EvidenceRef,
    *,
    check_conditional: bool = True,
) -> list[FidelityFinding]:
    """Verify that `ref` matches `doc_text` verbatim and does not drop adjacent restrictive qualifiers."""
    findings: list[FidelityFinding] = []
    if ref.start < 0 or ref.end > len(doc_text) or ref.start >= ref.end:
        findings.append(
            FidelityFinding(
                code="INVALID_EVIDENCE_SPAN",
                message="Evidence span offsets are out of bounds.",
                field="evidence",
                detail={"start": ref.start, "end": ref.end},
            )
        )
        return findings

    if doc_text[ref.start : ref.end] != ref.quote:
        findings.append(
            FidelityFinding(
                code="QUOTE_SPAN_MISMATCH",
                message="Evidence quote does not match document slice at declared offsets.",
                field="evidence",
            )
        )
        return findings

    sent_start, sent_end, _ = enclosing_sentence(doc_text, ref.start, ref.end)
    leading = doc_text[sent_start : ref.start].strip()
    trailing = doc_text[ref.end : sent_end].strip()

    if leading and ((check_conditional and _CONDITIONAL_MARKER_RE.search(leading)) or _CLAUSE_NEGATION_RE.search(leading)):
        findings.append(
            FidelityFinding(
                code="TRUNCATED_CLAUSE_CONTEXT",
                message="Cited span drops leading condition or negation in the same sentence.",
                field="conditions",
                detail={"dropped_prefix": leading},
            )
        )
    if check_conditional and trailing and _CONDITIONAL_MARKER_RE.search(trailing):
        findings.append(
            FidelityFinding(
                code="TRUNCATED_CLAUSE_CONTEXT",
                message="Cited span drops trailing proviso or exception in the same sentence.",
                field="conditions",
                detail={"dropped_suffix": trailing},
            )
        )
    return findings


_UNIT_TO_V4_FACTOR: dict[str, tuple[Decimal, str]] = {
    "DAYS": (Decimal("1"), "calendar_days"),
    "CALENDAR_DAYS": (Decimal("1"), "calendar_days"),
    "WEEKS": (Decimal("7"), "calendar_days"),
    "BUSINESS_DAYS": (Decimal("1"), "business_days"),
    "MONTHS": (Decimal("1"), "months"),
    "YEARS": (Decimal("12"), "months"),
    "HOURS": (Decimal("1"), "hours"),
    "PERCENT": (Decimal("1"), "percent"),
    "BPS": (Decimal("0.01"), "percentage_point"),
    "USD": (Decimal("1"), "USD"),
    "EUR": (Decimal("1"), "EUR"),
    "GBP": (Decimal("1"), "GBP"),
    "MULTIPLIER": (Decimal("1"), "multiple"),
}


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _extract_standalone_decimal_spans_untyped(
    text: str,
    occupied_spans: Sequence[tuple[int, int]],
) -> list[tuple[Decimal, tuple[int, int]]]:
    out: list[tuple[Decimal, tuple[int, int]]] = []
    for m in re.finditer(r"(?<![0-9.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9.])", text):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue
        tok = m.group(0).replace(",", "")
        try:
            out.append((Decimal(tok), span))
        except InvalidOperation:
            continue
    return out


def _extract_standalone_decimals_untyped(
    text: str,
    occupied_spans: Sequence[tuple[int, int]],
) -> list[Decimal]:
    return [n for n, _ in _extract_standalone_decimal_spans_untyped(text, occupied_spans)]


_PCT_PER_ANNUM_SUFFIX_RE = re.compile(r"\s*(?:per\s+annum|p\.a\.|per\s+year|annually)(?![a-z0-9_])", re.IGNORECASE)
_CUR_PER_YEAR_SUFFIX_RE = re.compile(r"\s*(?:per\s+year|annually|per\s+annum)(?![a-z0-9_])", re.IGNORECASE)
_CUR_PER_MONTH_SUFFIX_RE = re.compile(r"\s*(?:per\s+month|monthly)(?![a-z0-9_])", re.IGNORECASE)
_HOURS_PER_WEEK_SUFFIX_RE = re.compile(r"\s*(?:/\s*week|per\s+week|a\s+week)(?![a-z0-9_])", re.IGNORECASE)
_HOURS_PER_DAY_SUFFIX_RE = re.compile(r"\s*(?:/\s*day|per\s+day|a\s+day)(?![a-z0-9_])", re.IGNORECASE)


def _extract_typed_v4_quantity_spans(
    text: str,
    units: UnitRegistry = STANDARD_UNITS,
) -> tuple[list[tuple[Decimal, str, tuple[int, int]]], list[tuple[int, int]]]:
    """Extract `(normalized_Decimal, v4_base_unit, (start, end))` tuples and occupied spans."""
    typed: list[tuple[Decimal, str, tuple[int, int]]] = []
    occupied: list[tuple[int, int]] = []

    for q in extract_quantities(text):
        start, end = q.span
        tail = text[end:]
        mapped: tuple[Decimal, str] | None = None

        if q.unit == "PERCENT":
            m = _PCT_PER_ANNUM_SUFFIX_RE.match(tail)
            if m:
                end += m.end()
                mapped = (q.magnitude, "percent_per_annum")
        elif q.unit in ("USD", "EUR"):
            m_yr = _CUR_PER_YEAR_SUFFIX_RE.match(tail)
            m_mo = _CUR_PER_MONTH_SUFFIX_RE.match(tail) if not m_yr else None
            if m_yr:
                end += m_yr.end()
                mapped = (q.magnitude, f"{q.unit}_per_year")
            elif m_mo:
                end += m_mo.end()
                mapped = (q.magnitude * Decimal("12"), f"{q.unit}_per_year")
        elif q.unit == "HOURS":
            m_wk = _HOURS_PER_WEEK_SUFFIX_RE.match(tail)
            m_dy = _HOURS_PER_DAY_SUFFIX_RE.match(tail) if not m_wk else None
            if m_wk:
                end += m_wk.end()
                mapped = (q.magnitude, "hours_per_week")
            elif m_dy:
                end += m_dy.end()
                mapped = (q.magnitude, "hours_per_day")

        if mapped is None:
            if q.unit in _UNIT_TO_V4_FACTOR:
                factor, base_u = _UNIT_TO_V4_FACTOR[q.unit]
                if label_key(base_u) in units._labels:
                    amt, reg_base = units.to_base(q.magnitude * factor, base_u, "fidelity")
                    mapped = (Decimal(amt), reg_base)
                else:
                    mapped = (q.magnitude * factor, base_u)
            elif label_key(q.unit) in units._labels:
                amt, reg_base = units.to_base(q.magnitude, q.unit, "fidelity")
                mapped = (Decimal(amt), reg_base)

        if mapped is not None or q.unit not in ("NONE", "COUNT", None, ""):
            occupied.append((start, end))
        if mapped is not None:
            typed.append((mapped[0], mapped[1], (start, end)))

    # Scan for additional registry units (including custom Vocabulary units or snake_case names)
    labels_by_len: list[str] = []
    seen_labels: set[str] = set()
    for u in units._units.values():
        for raw_label in (u.name, *u.aliases):
            for candidate in (raw_label, raw_label.replace("_", " ")):
                key = candidate.strip()
                if len(key) >= 2 and key.lower() not in seen_labels:
                    seen_labels.add(key.lower())
                    labels_by_len.append(key)
    labels_by_len.sort(key=lambda s: (-len(s), s.lower()))

    if labels_by_len:
        parts: list[str] = []
        for lbl in labels_by_len:
            esc = re.escape(lbl).replace(r"\ ", r"\s+")
            if lbl[-1].isalnum():
                esc += r"(?![a-z0-9_])"
            parts.append(esc)
        unit_pat = re.compile(
            r"(?<![0-9.])(?P<num>[-+]?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s+|-)?"
            r"(?P<unit>(?:" + "|".join(parts) + r"))",
            re.IGNORECASE,
        )
        for m in unit_pat.finditer(text):
            span = (m.start(), m.end())
            if any(_spans_overlap(span, occ) for occ in occupied):
                continue
            num_tok = m.group("num").replace(",", "")
            lbl_tok = re.sub(r"\s+", " ", m.group("unit").strip())
            # Require whitespace or hyphen before purely alphabetic multi-char labels
            raw_between = text[m.end("num") : m.start("unit")]
            if lbl_tok[0].isalpha() and len(lbl_tok) > 1 and not raw_between:
                continue
            lookup_lbl = lbl_tok if label_key(lbl_tok) in units._labels else lbl_tok.replace(" ", "_")
            if label_key(lookup_lbl) not in units._labels:
                continue
            try:
                amt, base_u = units.to_base(num_tok, lookup_lbl, "fidelity")
            except (UnmappedProposal, ValueError, InvalidOperation):
                continue
            occupied.append(span)
            typed.append((Decimal(amt), base_u, span))

    return typed, occupied


def _extract_typed_v4_quantities(
    text: str,
    units: UnitRegistry = STANDARD_UNITS,
) -> tuple[list[tuple[Decimal, str]], list[tuple[int, int]]]:
    """Extract (normalized_Decimal, v4_base_unit) pairs and occupied spans from prose."""
    typed_spans, occupied = _extract_typed_v4_quantity_spans(text, units)
    return [(mag, base_u) for mag, base_u, _ in typed_spans], occupied


def _numeric_marker_present(
    raw_num: Any,
    text: str,
    *,
    unit_hint: str | None = None,
    units: UnitRegistry = STANDARD_UNITS,
) -> bool:
    if isinstance(raw_num, bool):
        return False
    try:
        target = Decimal(str(raw_num).strip())
    except (InvalidOperation, ValueError):
        return False
    if not target.is_finite():
        return False

    typed_qs, occupied_spans = _extract_typed_v4_quantities(text, units)
    if unit_hint is not None:
        if label_key(unit_hint) not in units._labels:
            return False
        try:
            norm_amt, norm_unit = units.to_base(target, unit_hint, "fidelity")
            target_val = Decimal(norm_amt)
            target_unit = norm_unit
        except (UnmappedProposal, ValueError, InvalidOperation):
            return False
        return any(base_u == target_unit and mag == target_val for mag, base_u in typed_qs)

    return any(n == target for n in _extract_standalone_decimals_untyped(text, occupied_spans))


def _tokens_present_bounded(tokens: Sequence[str], text: str) -> bool:
    if not tokens:
        return True
    low = text.lower()
    return all(
        re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low) is not None
        for t in tokens
    )


def _date_marker_present(value: Any, text: str) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if re.search(r"(?<![0-9\-])" + re.escape(stripped) + r"(?![0-9\-])", text):
        return True
    for m in _ISO_DATE_FRAGMENT_RE.finditer(text):
        try:
            if normalize_date(m.group(0)) == stripped:
                return True
        except ValueError:
            continue
    return stripped in extract_dates(text, allow_slash=True)


def _value_markers_present(
    value: Any,
    text: str,
    *,
    units: UnitRegistry = STANDARD_UNITS,
    spec: FieldSpec | None = None,
) -> bool:
    if spec is not None:
        t = spec.type
        if t in {"quantity", "bound"}:
            if not isinstance(value, dict) or "value" not in value or not value.get("unit"):
                return False
            return _numeric_marker_present(value["value"], text, unit_hint=value["unit"], units=units)
        if t == "number":
            if isinstance(value, (dict, list, tuple, bool)):
                return False
            return _numeric_marker_present(value, text, unit_hint=None, units=units)
        if t == "boolean":
            return isinstance(value, bool)
        if t == "date":
            return _date_marker_present(value, text)
        if t == "record":
            if not isinstance(value, dict):
                return False
            declared = dict(spec.fields)
            return all(
                _value_markers_present(v, text, units=units, spec=declared.get(k))
                for k, v in value.items()
            )
        if t == "set":
            if not isinstance(value, (list, tuple)) or not value:
                return False
            elem_spec = FieldSpec("enum", enum=spec.enum)
            return all(_value_markers_present(item, text, units=units, spec=elem_spec) for item in value)
        if t == "list":
            if not isinstance(value, (list, tuple)) or not value or spec.element is None:
                return False
            return all(_value_markers_present(item, text, units=units, spec=spec.element) for item in value)
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", stripped.lower()) if len(tok) >= 2]
        if not tokens:
            tokens = re.findall(r"[a-z0-9]+", stripped.lower())
        return _tokens_present_bounded(tokens, text)

    if isinstance(value, dict) and "value" in value:
        return _numeric_marker_present(value["value"], text, unit_hint=value.get("unit"), units=units)
    if isinstance(value, dict):
        return all(_value_markers_present(v, text, units=units) for v in value.values())
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_value_markers_present(item, text, units=units) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, Decimal)):
        return _numeric_marker_present(value, text, units=units)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
            return _numeric_marker_present(stripped, text, units=units)
        if re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?)?)?", stripped):
            return _date_marker_present(stripped, text)
        tokens = [t for t in re.findall(r"[a-z0-9]+", stripped.lower()) if len(t) >= 2]
        if not tokens:
            tokens = re.findall(r"[a-z0-9]+", stripped.lower())
        return _tokens_present_bounded(tokens, text)
    return _value_markers_present(str(value), text, units=units)


_GENERIC_PREDICATE_TOKENS = frozenset({
    "period", "days", "day", "hours", "hour", "months", "month", "years", "year",
    "date", "time", "rate", "ratio", "pct", "percent", "percentage", "target",
    "threshold", "limit", "limits", "value", "values", "number", "count", "qty",
    "quantity", "level", "status", "of", "in", "to", "for", "by", "on", "at",
    "is", "are", "the", "a", "an", "multi", "dim", "plain", "standard",
})

_PREDICATE_TOKEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "payment": ("payment", "payments", "pay", "paid", "payable"),
    "delivery": ("delivery", "deliver", "delivered", "deliveries"),
    "governing": ("governing", "governed", "govern", "governs"),
    "law": ("law", "laws"),
    "terms": ("terms", "term"),
    "term": ("term", "terms"),
    "reporting": ("reporting", "report", "reports"),
    "servicer": ("servicer", "servicing"),
    "renewal": ("renewal", "renew", "renews", "renewed"),
    "auto": ("auto", "automatic", "automatically"),
}

_HEADER_NON_SUBJECT_WORDS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "all", "any", "each",
    "every", "such", "standard", "general", "section", "clause", "article",
    "paragraph", "annex", "schedule", "amendment", "supplemental", "indenture",
    "record", "source", "term", "base", "year", "years", "month", "months",
    "day", "days", "rsf", "usd", "eur", "gbp",
})


def _term_in_text(text: str, term: str) -> bool:
    """Case-insensitive token-bounded match allowing space/'_'/'-' equivalence."""
    key = label_key(term)
    if not key:
        return False
    parts = [re.escape(p) for p in key.split()]
    pattern = r"(?<![a-z0-9])" + r"[\s_\-]+".join(parts) + r"(?![a-z0-9])"
    return re.search(pattern, text.casefold()) is not None


def _entity_candidate_labels(entity: str, vocabulary: Vocabulary | None) -> list[str]:
    labels: list[str] = [entity]
    if vocabulary is not None:
        canon = label_key(entity)
        for cname, aliases in vocabulary.entities:
            if label_key(cname) == canon:
                labels.extend(aliases)
                break
    seen: set[str] = set()
    out: list[str] = []
    for lbl in labels:
        k = label_key(lbl)
        if k and k not in seen:
            seen.add(k)
            out.append(lbl)
    return out


def _leading_header_text(doc_text: str) -> str:
    """Extract a short title/header prefix (first line/sentence before section/clause body)."""
    m = re.match(r"^\s*([^\n.;]{1,80}(?:[.;]\s*[^\n.;]{1,60})?)", doc_text)
    if not m:
        return ""
    header = m.group(1)
    # Only treat short title-like fragments without finite verbs as a document header
    if re.search(r"\b(?:is|are|was|were|shall|will|must|may|has|have)\b", header, re.IGNORECASE):
        return ""
    return header


def _has_competing_subject_in_sentences(
    sentences: Sequence[str],
    entity_labels: Sequence[str],
    predicate_name: str,
) -> bool:
    entity_words = {w for lbl in entity_labels for w in label_key(lbl).split()}
    pred_words = set(label_key(predicate_name).split()) | {
        syn for w in label_key(predicate_name).split() for syn in _PREDICATE_TOKEN_SYNONYMS.get(w, ())
    }
    for sent in sentences:
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", sent):
            low = tok.casefold()
            if low in _HEADER_NON_SUBJECT_WORDS or low in entity_words or low in pred_words:
                continue
            return True
    return False


def _entity_grounded_in_evidence(
    entity: str,
    combined_quotes: str,
    enclosing_sents: Sequence[str],
    doc_text: str,
    vocabulary: Vocabulary | None,
) -> bool:
    labels = _entity_candidate_labels(entity, vocabulary)
    local_texts = (combined_quotes, " ".join(enclosing_sents))
    if any(_term_in_text(t, lbl) for t in local_texts for lbl in labels):
        return True
    header = _leading_header_text(doc_text)
    if header and any(_term_in_text(header, lbl) for lbl in labels):
        return True
    return False


def _predicate_grounded_in_evidence(
    predicate_name: str,
    entity: str,
    combined_quotes: str,
    enclosing_sents: Sequence[str],
    doc_text: str,
    vocabulary: Vocabulary | None,
    *,
    predicate_tokens_only: bool = False,
) -> bool:
    """Verify predicate cue presence against quote/enclosing sentences (and declared aliases)."""
    header = _leading_header_text(doc_text) if not predicate_tokens_only else ""
    search_text = " ".join(part for part in (combined_quotes, " ".join(enclosing_sents), header) if part)
    candidates: list[str] = [predicate_name]
    if vocabulary is not None:
        try:
            spec = vocabulary.predicate(predicate_name)
            candidates.extend(spec.aliases)
        except UnmappedProposal:
            pass
    if any(_term_in_text(search_text, cand) for cand in candidates):
        return True

    ent_tokens = set(label_key(entity).split())
    raw_tokens = [t for t in label_key(predicate_name).split() if t not in ent_tokens]
    if not raw_tokens:
        raw_tokens = label_key(predicate_name).split()
    specific_tokens = [t for t in raw_tokens if t not in _GENERIC_PREDICATE_TOKENS]
    required_tokens = specific_tokens if specific_tokens else raw_tokens
    for tok in required_tokens:
        forms = _PREDICATE_TOKEN_SYNONYMS.get(tok, (tok,))
        if not any(_term_in_text(search_text, form) for form in forms):
            return False
    return True


def _scope_grounded_in_evidence(
    scope: Mapping[str, Any],
    combined_quotes: str,
    enclosing_sents: Sequence[str],
) -> list[tuple[str, str]]:
    """Return list of `(scope_key, scope_value)` pairs not grounded in quote/sentence."""
    local_text = f"{combined_quotes} {' '.join(enclosing_sents)}"
    ungrounded: list[tuple[str, str]] = []
    for k, v in scope.items():
        if k in {"event_id", "clause_id"}:
            continue
        if isinstance(v, str) and v.strip():
            if not _term_in_text(local_text, v):
                ungrounded.append((str(k), v))
    return ungrounded


_COMPETING_PRED_HEADS = frozenset({
    "period", "periods", "window", "windows", "terms", "term",
    "cap", "caps", "limit", "limits", "threshold", "thresholds",
    "rate", "rates", "ratio", "ratios", "target", "targets",
    "fee", "fees", "charge", "charges", "penalty", "penalties",
    "deadline", "deadlines", "payout", "payouts",
    "renewal", "law", "jurisdiction", "frequency", "priority",
})

_COMPETING_PRED_HEAD_RE = re.compile(
    r"\b(?P<mod>[A-Za-z][A-Za-z0-9_-]{2,}(?:\s+[A-Za-z][A-Za-z0-9_-]{2,})?)\s+"
    r"(?P<head>" + "|".join(sorted(_COMPETING_PRED_HEADS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_BENIGN_PRED_MODIFIERS = frozenset({
    "the", "a", "an", "this", "that", "these", "those", "all", "any", "each", "every",
    "such", "no", "nor", "not", "and", "or", "but", "if", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "with", "without", "within", "under",
    "upon", "after", "before", "during", "per", "over", "following", "preceding",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "shall", "will", "must", "may", "might", "can", "could", "should", "would",
    "applicable", "agreed", "stated", "specified", "relevant", "contractual", "written",
    "prior", "calendar", "business", "working", "total", "aggregate", "maximum",
    "minimum", "max", "min", "initial", "standard", "base", "regular", "general",
    "overall", "annual", "monthly", "daily", "weekly", "quarterly", "fixed", "flat",
    "normal", "full", "remaining", "additional", "new", "existing", "current", "active",
    "effective", "original", "revised", "amended", "cumulative", "net", "gross", "direct",
    "respective", "corresponding", "mandatory", "required", "permitted", "allowed",
    "extended", "shortened", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "day", "days", "hour", "hours", "month", "months",
    "year", "years", "week", "weeks", "usd", "eur", "gbp", "percent", "percentage",
})

_NON_COMPETING_SUBJECT_WORDS = _HEADER_NON_SUBJECT_WORDS | _BENIGN_PRED_MODIFIERS | _COMPETING_PRED_HEADS | frozenset({
    "if", "when", "where", "unless", "except", "provided", "subject", "notwithstanding",
    "pursuant", "until", "while", "whereas", "however", "moreover", "furthermore",
    "additionally", "neither", "either", "both", "agreement", "contract", "msa", "sla",
    "nda", "dpa", "sow", "exhibit", "table", "true", "false", "c1", "c1a", "c2", "c3",
    "c4", "s1", "s2", "s3", "notice", "payment", "cure", "reporting", "delivery",
    "liability", "indemnity", "confidentiality", "service", "enterprise", "general",
    "governing", "auto", "renewal", "uptime", "credit", "payout", "filing", "late",
    "interest", "retention", "jurisdiction",
})

_CLAUSE_VERB_OR_ASSIGN_RE = re.compile(
    r"(?::|=|\b(?:is|are|was|were|be|shall|will|must|may|can|has|have|had|remains|equals|totals|exceeds|amended|set)\b)",
    re.IGNORECASE,
)

_SOFT_CLAUSE_SEP_RE = re.compile(
    r"(?<!\d),\s*(?:and|or|but|while|whereas)\b|\b(?:and|or|but|while|whereas)\b|(?<!\d),(?!\d)",
    re.IGNORECASE,
)

_PREPOSITION_BEFORE_PRED_RE = re.compile(
    r"(?:^|\b(?:during|after|before|following|upon|under|within|until|for|per|except|excluding|including|"
    r"subject\s+to|prior\s+to))\s+(?:the\s+|a\s+|an\s+|any\s+|such\s+|this\s+)?$",
    re.IGNORECASE,
)


def _term_spans_in_text(text: str, term: str) -> list[tuple[int, int]]:
    """Return exact `(start, end)` character spans in *text* matching *term* cleanly."""
    key = label_key(term)
    if not key:
        return []
    parts = [re.escape(p) for p in key.split()]
    pattern = r"(?<![A-Za-z0-9])" + r"[\s_\-]+".join(parts) + r"(?![A-Za-z0-9])"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text, re.IGNORECASE)]


def _merge_sorted_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    out: list[tuple[int, int]] = []
    for s, e in ordered:
        if out and s < out[-1][1] and e <= out[-1][1]:
            continue
        out.append((s, e))
    return out


def _find_value_spans(
    value: Any,
    text: str,
    *,
    units: UnitRegistry = STANDARD_UNITS,
    spec: FieldSpec | None = None,
) -> list[tuple[int, int]]:
    """Locate exact character spans in *text* corresponding to the proposed value."""
    spans: list[tuple[int, int]] = []
    if spec is not None:
        t = spec.type
        if t in {"quantity", "bound"}:
            if not isinstance(value, dict) or "value" not in value or not value.get("unit"):
                return []
            try:
                norm_amt, norm_unit = units.to_base(Decimal(str(value["value"]).strip()), value["unit"], "fidelity")
                target_val = Decimal(norm_amt)
            except (UnmappedProposal, ValueError, InvalidOperation):
                return []
            typed_spans, _ = _extract_typed_v4_quantity_spans(text, units)
            return [sp for mag, base_u, sp in typed_spans if base_u == norm_unit and mag == target_val]
        if t == "number":
            if isinstance(value, (dict, list, tuple, bool)):
                return []
            try:
                target_val = Decimal(str(value).strip())
            except (InvalidOperation, ValueError):
                return []
            _, occupied = _extract_typed_v4_quantity_spans(text, units)
            return [sp for n, sp in _extract_standalone_decimal_spans_untyped(text, occupied) if n == target_val]
        if t == "boolean":
            return []
        if t == "date":
            if not isinstance(value, str):
                return []
            stripped = value.strip()
            spans.extend(
                (m.start(), m.end())
                for m in re.finditer(r"(?<![0-9\-])" + re.escape(stripped) + r"(?![0-9\-])", text)
            )
            for m in _ISO_DATE_FRAGMENT_RE.finditer(text):
                try:
                    if normalize_date(m.group(0)) == stripped:
                        spans.append((m.start(), m.end()))
                except ValueError:
                    continue
            return _merge_sorted_spans(spans)
        if not isinstance(value, str):
            return []
        exact = _term_spans_in_text(text, value)
        if exact:
            return exact
        toks = [tok for tok in re.findall(r"[A-Za-z0-9]+", value) if len(tok) >= 2]
        for tok in toks:
            spans.extend(_term_spans_in_text(text, tok))
        return _merge_sorted_spans(spans)

    if isinstance(value, dict) and "value" in value:
        unit_hint = value.get("unit")
        if unit_hint:
            try:
                norm_amt, norm_unit = units.to_base(Decimal(str(value["value"]).strip()), unit_hint, "fidelity")
                target_val = Decimal(norm_amt)
            except (UnmappedProposal, ValueError, InvalidOperation):
                return []
            typed_spans, _ = _extract_typed_v4_quantity_spans(text, units)
            return [sp for mag, base_u, sp in typed_spans if base_u == norm_unit and mag == target_val]
        try:
            target_val = Decimal(str(value["value"]).strip())
        except (InvalidOperation, ValueError):
            return []
        _, occupied = _extract_typed_v4_quantity_spans(text, units)
        return [sp for n, sp in _extract_standalone_decimal_spans_untyped(text, occupied) if n == target_val]

    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float, Decimal)):
        try:
            target_val = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            return []
        _, occupied = _extract_typed_v4_quantity_spans(text, units)
        return [sp for n, sp in _extract_standalone_decimal_spans_untyped(text, occupied) if n == target_val]
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
            return _find_value_spans(Decimal(stripped), text, units=units)
        exact = _term_spans_in_text(text, stripped)
        if exact:
            return exact
        toks = [tok for tok in re.findall(r"[A-Za-z0-9]+", stripped) if len(tok) >= 2]
        for tok in toks:
            spans.extend(_term_spans_in_text(text, tok))
        return _merge_sorted_spans(spans)
    return []


def _all_quantity_spans(text: str, *, units: UnitRegistry = STANDARD_UNITS) -> list[tuple[int, int]]:
    typed_spans, occupied = _extract_typed_v4_quantity_spans(text, units)
    spans = [sp for _, _, sp in typed_spans]
    spans.extend(sp for _, sp in _extract_standalone_decimal_spans_untyped(text, occupied))
    return _merge_sorted_spans(spans)


def _find_entity_spans(entity: str, text: str, *, vocabulary: Vocabulary | None = None) -> list[tuple[int, int]]:
    if not isinstance(entity, str) or not entity.strip():
        return []
    spans: list[tuple[int, int]] = []
    for lbl in _entity_candidate_labels(entity, vocabulary):
        spans.extend(_term_spans_in_text(text, lbl))
    return _merge_sorted_spans(spans)


def _find_competing_entity_spans(
    prop_entity: str,
    text: str,
    *,
    vocabulary: Vocabulary | None = None,
    prop_predicate: str = "",
    prop: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """Return `(declared_competing, heuristic_competing)` entity mentions in *text*."""
    prop_ent_spans = _find_entity_spans(prop_entity, text, vocabulary=vocabulary)
    declared: list[tuple[str, int, int]] = []
    prop_ent_key = label_key(prop_entity)
    if vocabulary is not None:
        for cname, aliases in vocabulary.entities:
            if label_key(cname) == prop_ent_key:
                continue
            for lbl in (cname, *aliases):
                for s, e in _term_spans_in_text(text, lbl):
                    if any(_spans_overlap((s, e), ps) for ps in prop_ent_spans):
                        continue
                    declared.append((cname, s, e))

    excluded_words = set(_NON_COMPETING_SUBJECT_WORDS)
    for lbl in _entity_candidate_labels(prop_entity, vocabulary):
        excluded_words.update(label_key(lbl).split())
    for tok in label_key(prop_predicate).split():
        excluded_words.add(tok)
        excluded_words.update(_PREDICATE_TOKEN_SYNONYMS.get(tok, ()))
    if vocabulary is not None:
        try:
            pspec = vocabulary.predicate(prop_predicate)
            for al in pspec.aliases:
                excluded_words.update(label_key(al).split())
        except UnmappedProposal:
            pass
        for p in vocabulary.predicates:
            excluded_words.update(label_key(p.name).split())
            for al in p.aliases:
                excluded_words.update(label_key(al).split())
    if isinstance(prop, Mapping):
        scope_val = prop.get("scope")
        if isinstance(scope_val, Mapping):
            for sv in scope_val.values():
                if isinstance(sv, str):
                    excluded_words.update(label_key(sv).split())
        pv = prop.get("value")
        if isinstance(pv, str):
            excluded_words.update(label_key(pv).split())

    heuristic: list[tuple[str, int, int]] = []
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", text):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, ps) for ps in prop_ent_spans):
            continue
        if any(_spans_overlap(span, (ds, de)) for _, ds, de in declared):
            continue
        words = m.group(0).split()
        filtered = [w for w in words if w.casefold() not in excluded_words]
        if not filtered:
            continue
        candidate = " ".join(filtered)
        # Skip single uppercase abbreviations like EU/UK/US unless explicitly multi-char proper name
        if len(candidate) <= 2 and candidate.isupper():
            continue
        heuristic.append((candidate, m.start(), m.end()))

    return declared, heuristic


def _predicate_label_tokens(predicate_name: str, entity: str, vocabulary: Vocabulary | None) -> list[str]:
    ent_tokens = set(label_key(entity).split())
    raw_tokens = [t for t in label_key(predicate_name).split() if t not in ent_tokens]
    if not raw_tokens:
        raw_tokens = label_key(predicate_name).split()
    specific = [t for t in raw_tokens if t not in _GENERIC_PREDICATE_TOKENS]
    return specific if specific else raw_tokens


def _find_predicate_spans(
    predicate_name: str,
    text: str,
    *,
    vocabulary: Vocabulary | None = None,
    entity: str = "",
) -> list[tuple[int, int]]:
    """Locate spans in *text* matching *predicate_name* or its declared aliases."""
    if not isinstance(predicate_name, str) or not predicate_name.strip():
        return []
    candidates: list[str] = [predicate_name]
    if vocabulary is not None:
        try:
            candidates.extend(vocabulary.predicate(predicate_name).aliases)
        except UnmappedProposal:
            pass
    spans: list[tuple[int, int]] = []
    for cand in candidates:
        spans.extend(_term_spans_in_text(text, cand))
    if spans:
        return _merge_sorted_spans(spans)

    req_tokens = _predicate_label_tokens(predicate_name, entity, vocabulary)
    if not req_tokens:
        return []
    token_hit_lists: list[list[tuple[int, int]]] = []
    for tok in req_tokens:
        hits: list[tuple[int, int]] = []
        for form in _PREDICATE_TOKEN_SYNONYMS.get(tok, (tok,)):
            hits.extend(_term_spans_in_text(text, form))
        if not hits:
            return []
        token_hit_lists.append(_merge_sorted_spans(hits))
    if len(token_hit_lists) == 1:
        return token_hit_lists[0]
    co_spans: list[tuple[int, int]] = []
    for first_s, first_e in token_hit_lists[0]:
        best_s, best_e = first_s, first_e
        ok = True
        for other_hits in token_hit_lists[1:]:
            nearest = min(other_hits, key=lambda sp: abs(sp[0] - first_s))
            lo = min(best_s, nearest[0])
            hi = max(best_e, nearest[1])
            if hi - lo > 60 or re.search(r"[\n;]", text[lo:hi]):
                ok = False
                break
            best_s, best_e = lo, hi
        if ok:
            co_spans.append((best_s, best_e))
    return _merge_sorted_spans(co_spans)


def _find_competing_predicate_spans(
    prop_predicate: str,
    text: str,
    *,
    vocabulary: Vocabulary | None = None,
    entity: str = "",
    prop_pred_spans: Sequence[tuple[int, int]] | None = None,
) -> list[tuple[str, int, int]]:
    """Return competing predicate spans `(label, start, end)` in *text*."""
    p_spans = (
        list(prop_pred_spans)
        if prop_pred_spans is not None
        else _find_predicate_spans(prop_predicate, text, vocabulary=vocabulary, entity=entity)
    )
    prop_key = label_key(prop_predicate)
    prop_tokens = set(_predicate_label_tokens(prop_predicate, entity, vocabulary))
    excluded_mod_words = set(_BENIGN_PRED_MODIFIERS) | set(label_key(entity).split()) | set(label_key(prop_predicate).split())
    for tok in list(prop_tokens):
        excluded_mod_words.update(_PREDICATE_TOKEN_SYNONYMS.get(tok, ()))
    if vocabulary is not None:
        try:
            pspec = vocabulary.predicate(prop_predicate)
            for al in pspec.aliases:
                excluded_mod_words.update(label_key(al).split())
        except UnmappedProposal:
            pass
    for unit_lbl in ("calendar_days", "business_days", "percentage_point"):
        excluded_mod_words.update(unit_lbl.split("_"))

    found: list[tuple[str, int, int]] = []

    def _is_subsumed(span: tuple[int, int]) -> bool:
        for ps, pe in p_spans:
            if _spans_overlap(span, (ps, pe)):
                # Only treat another predicate as competing if it strictly extends prop_predicate (more specific),
                # never when it is equal to or shorter than the matched prop_predicate span.
                if (span[1] - span[0]) <= (pe - ps):
                    return True
        return False

    if vocabulary is not None:
        for other in vocabulary.predicates:
            if label_key(other.name) == prop_key:
                continue
            for lbl in (other.name, *other.aliases):
                for s, e in _term_spans_in_text(text, lbl):
                    if _is_subsumed((s, e)):
                        continue
                    found.append((other.name, s, e))

    for m in _COMPETING_PRED_HEAD_RE.finditer(text):
        mod_tokens_with_pos = [
            (tm.group(0).casefold(), m.start("mod") + tm.start(), m.start("mod") + tm.end())
            for tm in re.finditer(r"[A-Za-z0-9]+", m.group("mod"))
            if not tm.group(0).isdigit()
        ]
        # Strip leading stopwords/prepositions/entities/units while preserving genuine modifier span offset
        remaining_with_pos = [(w, ts, te) for w, ts, te in mod_tokens_with_pos if w not in excluded_mod_words]
        if not remaining_with_pos:
            continue
        eff_span = (remaining_with_pos[0][1], m.end())
        if _is_subsumed(eff_span):
            continue
        if any(_spans_overlap(eff_span, (fs, fe)) for _, fs, fe in found):
            continue
        head_tok = m.group("head").casefold()
        label = " ".join(w for w, _, _ in remaining_with_pos) + " " + head_tok
        found.append((label, eff_span[0], eff_span[1]))

    found.sort(key=lambda item: (item[1], -(item[2] - item[1])))
    return found


def _split_into_atomic_clauses(
    text: str,
    *,
    units: UnitRegistry = STANDARD_UNITS,
) -> list[tuple[str, int, int, bool]]:
    """Split *text* into atomic clauses `(clause_text, start, end, new_sentence)`.

    Splits unconditionally on sentence/line/semicolon boundaries, and splits on
    coordinating conjunctions/commas only when both sides form independent
    clause structures (containing a finite verb/assigner or a numeric/quantity marker).
    """
    n = len(text)
    if not text.strip():
        return []

    hard_bounds: list[tuple[int, int]] = []
    for m in re.finditer(r"[\n;]+|(?<!\d)\.(?!\d)(?=\s|$)", text):
        hard_bounds.append((m.start(), m.end()))

    raw_sentences: list[tuple[int, int]] = []
    cur = 0
    for hs, he in hard_bounds:
        if hs > cur and text[cur:hs].strip():
            raw_sentences.append((cur, hs))
        cur = he
    if cur < n and text[cur:n].strip():
        raw_sentences.append((cur, n))

    def _has_clause_core(seg: str) -> bool:
        if _CLAUSE_VERB_OR_ASSIGN_RE.search(seg):
            return True
        if _all_quantity_spans(seg, units=units):
            return True
        return False

    clauses: list[tuple[str, int, int, bool]] = []
    for s_start, s_end in raw_sentences:
        sent_text = text[s_start:s_end]
        split_points: list[tuple[int, int]] = []
        for m in _SOFT_CLAUSE_SEP_RE.finditer(sent_text):
            left = sent_text[: m.start()]
            right = sent_text[m.end() :]
            if not left.strip() or not right.strip():
                continue
            # Never split inside "Month DD, YYYY" dates
            if m.group(0).startswith(",") and re.search(
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*$",
                left,
                re.IGNORECASE,
            ) and re.match(r"\s*\d{4}\b", right):
                continue
            if _has_clause_core(left) and _has_clause_core(right):
                split_points.append((m.start(), m.end()))

        prev = 0
        first_in_sent = True
        for sp_s, sp_e in split_points:
            piece = sent_text[prev:sp_s]
            if piece.strip():
                l_pad = len(piece) - len(piece.lstrip())
                r_pad = len(piece) - len(piece.rstrip())
                clauses.append(
                    (
                        piece.strip(),
                        s_start + prev + l_pad,
                        s_start + sp_s - r_pad,
                        first_in_sent,
                    )
                )
                first_in_sent = False
            prev = sp_e
        tail = sent_text[prev:]
        if tail.strip():
            l_pad = len(tail) - len(tail.lstrip())
            r_pad = len(tail) - len(tail.rstrip())
            clauses.append(
                (
                    tail.strip(),
                    s_start + prev + l_pad,
                    s_end - r_pad,
                    first_in_sent,
                )
            )
    return clauses


def _parse_pipe_row_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c or "") is not None for c in cells)


_GENERIC_VALUE_COL_HEADERS = frozenset({
    "value", "amount", "terms", "requirement", "setting", "metric", "parameter", "spec", "detail", "details"
})


def _verify_table_relational_association(
    prop_entity: str,
    prop_predicate: str,
    val: Any,
    unit: EvidenceUnit,
    primary_doc_text: str,
    *,
    vocabulary: Vocabulary | None,
    units: UnitRegistry,
    spec: FieldSpec | None,
    prop: Mapping[str, Any],
) -> list[FidelityFinding]:
    """Verify relational row/column alignment when evidence is formatted as a pipe-delimited table."""
    has_pipes = any(line.count("|") >= 2 for r in unit.evidence for line in r.quote.splitlines())
    if not has_pipes:
        return []

    doc_lines: list[tuple[int, int, str]] = []
    pos = 0
    for raw_line in primary_doc_text.splitlines(keepends=True):
        clean = raw_line.rstrip("\r\n")
        doc_lines.append((pos, pos + len(clean), clean))
        pos += len(raw_line)
    if not doc_lines and primary_doc_text:
        doc_lines.append((0, len(primary_doc_text), primary_doc_text))

    header_entity = ""
    hdr_txt = _leading_header_text(primary_doc_text)
    if hdr_txt and prop_entity and _find_entity_spans(prop_entity, hdr_txt, vocabulary=vocabulary):
        header_entity = prop_entity

    # Identify table blocks near cited evidence
    cited_line_indices: list[int] = []
    for ref in unit.evidence:
        if ref.document_id != unit.evidence[0].document_id:
            continue
        for idx, (ls, le, line) in enumerate(doc_lines):
            if le > ref.start and ls < ref.end and line.count("|") >= 2:
                if idx not in cited_line_indices:
                    cited_line_indices.append(idx)

    if not cited_line_indices:
        return []

    # Scan backwards from first cited table line to recover table header row
    block_start = cited_line_indices[0]
    while block_start > 0 and doc_lines[block_start - 1][2].count("|") >= 2:
        block_start -= 1
    block_end = cited_line_indices[-1]
    while block_end + 1 < len(doc_lines) and doc_lines[block_end + 1][2].count("|") >= 2:
        block_end += 1

    table_rows = [
        (idx, _parse_pipe_row_cells(doc_lines[idx][2]), doc_lines[idx][2])
        for idx in range(block_start, block_end + 1)
    ]
    header_cells: list[str] = []
    header_row_idx: int | None = None
    if len(table_rows) >= 2:
        if _is_markdown_separator_row(table_rows[1][1]):
            header_row_idx = table_rows[0][0]
            header_cells = table_rows[0][1]
        elif not any(_all_quantity_spans(c, units=units) for c in table_rows[0][1]):
            header_row_idx = table_rows[0][0]
            header_cells = table_rows[0][1]

    data_rows = [
        (idx, cells, raw)
        for idx, cells, raw in table_rows
        if idx in cited_line_indices
        and idx != header_row_idx
        and not _is_markdown_separator_row(cells)
    ]
    if not data_rows:
        return []

    def _cell_matches_prop_pred(cell: str) -> bool:
        return _predicate_grounded_in_evidence(
            prop_predicate, prop_entity, cell, (cell,), cell, vocabulary, predicate_tokens_only=True
        )

    matched_any_row = False
    mismatch_field = "value"
    mismatch_detail: dict[str, Any] = {}

    for _, cells, raw_row in data_rows:
        row_ent_spans = _find_entity_spans(prop_entity, raw_row, vocabulary=vocabulary) if prop_entity else []
        decl_comp, heur_comp = _find_competing_entity_spans(
            prop_entity, raw_row, vocabulary=vocabulary, prop_predicate=prop_predicate, prop=prop
        )
        row_has_comp_entity = bool(decl_comp) or (not row_ent_spans and bool(heur_comp))
        if prop_entity:
            if row_ent_spans and not decl_comp:
                row_entity_ok = True
            elif not row_ent_spans and not row_has_comp_entity and header_entity:
                row_entity_ok = True
            else:
                row_entity_ok = False
        else:
            row_entity_ok = True

        for col_idx, cell in enumerate(cells):
            if isinstance(val, bool):
                if not _cell_matches_prop_pred(cell):
                    continue
            elif not _value_markers_present(val, cell, units=units, spec=spec):
                continue

            if not row_entity_ok:
                mismatch_field = "entity"
                mismatch_detail = {"row": raw_row, "competing_entities": [c for c, _, _ in (decl_comp or heur_comp)]}
                continue

            col_header = header_cells[col_idx] if header_cells and col_idx < len(header_cells) else ""
            if col_header and label_key(col_header) not in _GENERIC_VALUE_COL_HEADERS:
                hdr_comp_preds = _find_competing_predicate_spans(
                    prop_predicate, col_header, vocabulary=vocabulary, entity=prop_entity
                )
                hdr_matches_prop = _cell_matches_prop_pred(col_header)
                if hdr_comp_preds and not hdr_matches_prop:
                    mismatch_field = "predicate"
                    mismatch_detail = {"row": raw_row, "column_header": col_header}
                    continue
                if hdr_matches_prop and not hdr_comp_preds:
                    matched_any_row = True
                    break

            # Key-value row or unheaded column: inspect non-value cells of the same row for predicate match
            other_cells_text = " ".join(c for j, c in enumerate(cells) if j != col_idx)
            row_comp = _find_competing_predicate_spans(
                prop_predicate, other_cells_text, vocabulary=vocabulary, entity=prop_entity
            )
            row_prop = _cell_matches_prop_pred(other_cells_text) or (
                bool(col_header) and _cell_matches_prop_pred(col_header)
            )
            if row_prop and not row_comp:
                matched_any_row = True
                break
            if row_comp and not row_prop:
                mismatch_field = "predicate"
                mismatch_detail = {"row": raw_row, "competing_predicates": [p for p, _, _ in row_comp]}
                continue
            if not row_comp:
                matched_any_row = True
                break
        if matched_any_row:
            break

    if matched_any_row:
        return []
    return [
        FidelityFinding(
            code="RELATIONAL_ASSOCIATION_MISMATCH",
            message=(
                f"Proposed tuple ({prop_entity!r}, {prop_predicate!r}, {val!r}) does not align with the same "
                "table row and predicate column in cited evidence."
            ),
            field=mismatch_field,
            detail=mismatch_detail or {"quotes": [r.quote for r in unit.evidence]},
        )
    ]


def _check_clause_proximity_binding(
    clause_text: str,
    prop_entity: str,
    prop_predicate: str,
    val: Any,
    *,
    vocabulary: Vocabulary | None,
    units: UnitRegistry,
    spec: FieldSpec | None,
    prop: Mapping[str, Any],
) -> tuple[bool, str, str]:
    """Within a single clause containing *val*, verify *prop_predicate* & *prop_entity* bind *val*."""
    val_spans = _find_value_spans(val, clause_text, units=units, spec=spec)
    if not val_spans:
        return True, "", ""

    p_spans = _find_predicate_spans(prop_predicate, clause_text, vocabulary=vocabulary, entity=prop_entity)
    cp_spans = _find_competing_predicate_spans(
        prop_predicate, clause_text, vocabulary=vocabulary, entity=prop_entity, prop_pred_spans=p_spans
    )
    all_qtys = _all_quantity_spans(clause_text, units=units)
    e_spans = _find_entity_spans(prop_entity, clause_text, vocabulary=vocabulary) if prop_entity else []
    decl_comp_e, heur_comp_e = _find_competing_entity_spans(
        prop_entity, clause_text, vocabulary=vocabulary, prop_predicate=prop_predicate, prop=prop
    )
    comp_e_spans = decl_comp_e if decl_comp_e else (heur_comp_e if not e_spans else [])

    val_unit = val.get("unit") if isinstance(val, Mapping) else None
    val_dim: str | None = None
    if isinstance(val_unit, str) and val_unit.strip():
        try:
            val_dim = units.dimension_of(val_unit)
        except Exception:
            val_dim = None
    if cp_spans and vocabulary is not None and val_dim is not None:
        filtered_cp: list[tuple[str, int, int]] = []
        for cp_name, cs, ce in cp_spans:
            try:
                cp_spec = vocabulary.predicate(cp_name).value
            except Exception:
                cp_spec = None
            if cp_spec is not None and cp_spec.type in {"quantity", "bound"} and cp_spec.dimensions:
                if val_dim not in cp_spec.dimensions:
                    continue
            filtered_cp.append((cp_name, cs, ce))
        cp_spans = filtered_cp

    for vs, ve in val_spans:
        pred_ok_here = True
        if cp_spans:
            if not p_spans:
                pred_ok_here = False
            else:
                left_cp = max(((name, cs, ce) for name, cs, ce in cp_spans if ce <= vs), key=lambda x: x[2], default=None)
                left_p = max(((ps, pe) for ps, pe in p_spans if pe <= vs), key=lambda x: x[1], default=None)
                if left_cp is not None:
                    cp_name, cp_s, cp_e = left_cp
                    prefix_before_cp = clause_text[:cp_s]
                    in_prep_modifier = bool(_PREPOSITION_BEFORE_PRED_RE.search(prefix_before_cp))
                    between_cp_and_v = clause_text[cp_e:vs]
                    direct_link = bool(
                        _CLAUSE_VERB_OR_ASSIGN_RE.search(between_cp_and_v)
                        or re.fullmatch(r"[\s:ofat=isbeupto\-]*", between_cp_and_v.lower())
                    )
                    other_qty_between_p_and_cp = (
                        left_p is not None
                        and any(qs >= left_p[1] and qe <= cp_s and not _spans_overlap((qs, qe), (vs, ve)) for qs, qe in all_qtys)
                    )
                    if left_p is None:
                        if not in_prep_modifier and direct_link:
                            pred_ok_here = False
                    elif cp_e > left_p[1]:
                        if other_qty_between_p_and_cp or (not in_prep_modifier and direct_link):
                            pred_ok_here = False

        ent_ok_here = True
        if comp_e_spans and prop_entity:
            left_ce = max(((name, cs, ce) for name, cs, ce in comp_e_spans if ce <= vs), key=lambda x: x[2], default=None)
            left_e = max(((es, ee) for es, ee in e_spans if ee <= vs), key=lambda x: x[1], default=None)
            if left_ce is not None:
                if left_e is None or left_ce[2] > left_e[1]:
                    # Competing subject sits closer to this predicate/value occurrence than prop_entity
                    if decl_comp_e or (left_e is not None and any(qs >= left_e[1] and qe <= left_ce[1] for qs, qe in all_qtys)):
                        ent_ok_here = False

        if pred_ok_here and ent_ok_here:
            return True, "", ""

    if not pred_ok_here:
        return False, "predicate", "Value is locally bound to a competing predicate rather than the proposed predicate."
    return False, "entity", "Value is locally bound to a competing entity rather than the proposed entity."


def _verify_text_relational_association(
    text: str,
    prop_entity: str,
    prop_predicate: str,
    val: Any,
    *,
    header_entity_ok: bool,
    header_predicate_ok: bool,
    vocabulary: Vocabulary | None,
    units: UnitRegistry,
    spec: FieldSpec | None,
    prop: Mapping[str, Any],
) -> list[FidelityFinding]:
    clauses = _split_into_atomic_clauses(text, units=units)
    if not clauses:
        return []

    is_bool = isinstance(val, bool)
    section_entity: str | None = prop_entity if (header_entity_ok and prop_entity) else None
    active_entity: str | None = section_entity
    has_other_clause_with_prop_pred = False
    for c_text, _, _, _ in clauses:
        if _find_predicate_spans(prop_predicate, c_text, vocabulary=vocabulary, entity=prop_entity):
            has_other_clause_with_prop_pred = True
            break

    candidate_clauses: list[tuple[str, bool, str | None]] = []
    for c_text, _, _, new_sent in clauses:
        if new_sent:
            active_entity = section_entity

        e_spans = _find_entity_spans(prop_entity, c_text, vocabulary=vocabulary) if prop_entity else []
        decl_comp_e, heur_comp_e = _find_competing_entity_spans(
            prop_entity, c_text, vocabulary=vocabulary, prop_predicate=prop_predicate, prop=prop
        )

        # Short non-verbal title/header line sets section subject for subsequent bullets/lines
        is_section_title = (
            len(c_text) <= 80
            and not _all_quantity_spans(c_text, units=units)
            and not re.search(r"\b(?:is|are|was|were|shall|will|must|may|has|have)\b", c_text, re.IGNORECASE)
        )

        # Track sentence-local active subject for legitimate coordinate clause subject inheritance
        if prop_entity:
            if e_spans and not decl_comp_e:
                clause_ent_ok = True
                active_entity = prop_entity
                if is_section_title:
                    section_entity = prop_entity
            elif decl_comp_e and not e_spans:
                clause_ent_ok = False
                active_entity = decl_comp_e[-1][0]
                if is_section_title:
                    section_entity = decl_comp_e[-1][0]
            elif not e_spans and heur_comp_e:
                clause_ent_ok = False
                active_entity = heur_comp_e[-1][0]
                if is_section_title:
                    section_entity = heur_comp_e[-1][0]
            elif e_spans and decl_comp_e:
                # Defer to local proximity check between prop_entity and declared competing entities
                clause_ent_ok = True
            else:
                clause_ent_ok = active_entity == prop_entity
        else:
            clause_ent_ok = True

        if is_bool:
            p_spans = _find_predicate_spans(prop_predicate, c_text, vocabulary=vocabulary, entity=prop_entity)
            if p_spans:
                candidate_clauses.append((c_text, clause_ent_ok, "prop" if not _find_competing_predicate_spans(prop_predicate, c_text, vocabulary=vocabulary, entity=prop_entity, prop_pred_spans=p_spans) else "mixed"))
            continue

        if _value_markers_present(val, c_text, units=units, spec=spec):
            candidate_clauses.append((c_text, clause_ent_ok, None))

    if is_bool:
        if not candidate_clauses:
            return []
        for c_text, clause_ent_ok, pred_tag in candidate_clauses:
            if clause_ent_ok and pred_tag in {"prop", "mixed"}:
                return []
        return [
            FidelityFinding(
                code="RELATIONAL_ASSOCIATION_MISMATCH",
                message=f"Proposed tuple ({prop_entity!r}, {prop_predicate!r}, {val!r}) is not bound in the same clause.",
                field="entity",
                detail={"text": text},
            )
        ]

    if not candidate_clauses:
        return []

    last_mismatch_field = "value"
    last_mismatch_msg = (
        f"Proposed tuple ({prop_entity!r}, {prop_predicate!r}, {val!r}) is not relationally bound "
        "in the same clause of cited evidence."
    )

    for c_text, clause_ent_ok, _ in candidate_clauses:
        if not clause_ent_ok:
            last_mismatch_field = "entity"
            continue

        p_spans = _find_predicate_spans(prop_predicate, c_text, vocabulary=vocabulary, entity=prop_entity)
        cp_spans = _find_competing_predicate_spans(
            prop_predicate, c_text, vocabulary=vocabulary, entity=prop_entity, prop_pred_spans=p_spans
        )

        if p_spans:
            ok_prox, bad_field, bad_msg = _check_clause_proximity_binding(
                c_text,
                prop_entity,
                prop_predicate,
                val,
                vocabulary=vocabulary,
                units=units,
                spec=spec,
                prop=prop,
            )
            if ok_prox:
                return []
            last_mismatch_field = bad_field
            last_mismatch_msg = bad_msg
            continue

        # Clause contains the proposed value but does NOT explicitly name prop_predicate.
        # Reject if the clause explicitly names a competing predicate or if another clause already
        # named prop_predicate with a different value (unless no competing predicate is present).
        if cp_spans:
            last_mismatch_field = "predicate"
            last_mismatch_msg = (
                f"Proposed value {val!r} is bound to competing predicate {cp_spans[0][0]!r} rather than {prop_predicate!r}."
            )
            continue
        if len(clauses) == 1 and header_predicate_ok:
            return []
        if has_other_clause_with_prop_pred and not cp_spans:
            # Elliptical continuation (e.g., "Acme cure period is 30 days and 15 days for IP breaches")
            return []
        if not cp_spans and (header_predicate_ok or clause_ent_ok):
            return []

    return [
        FidelityFinding(
            code="RELATIONAL_ASSOCIATION_MISMATCH",
            message=last_mismatch_msg,
            field=last_mismatch_field,
            detail={"text": text, "entity": prop_entity, "predicate": prop_predicate, "value": val},
        )
    ]


def _verify_relational_association(
    primary_doc_text: str,
    unit: EvidenceUnit,
    full_sentences: Sequence[str],
    *,
    by_field: Mapping[str, FieldEvidence],
    vocabulary: Vocabulary | None,
    units: UnitRegistry,
    spec: FieldSpec | None,
    documents: Mapping[str, Any] | None,
) -> list[FidelityFinding]:
    """Ensure `(entity, predicate, value)` are relationally bound in the same clause or table cell/row."""
    prop = unit.proposition
    prop_entity = str(prop.get("entity") or "")
    prop_predicate = str(prop.get("predicate") or "")
    val = prop.get("value")
    if not prop_predicate:
        return []

    table_findings = _verify_table_relational_association(
        prop_entity,
        prop_predicate,
        val,
        unit,
        primary_doc_text,
        vocabulary=vocabulary,
        units=units,
        spec=spec,
        prop=prop,
    )
    if table_findings:
        return table_findings

    # Check FieldEvidence relational alignment when explicit per-field spans are provided on the same document
    if spec is not None and spec.type == "list":
        # Per-element FieldEvidence spans and ordered document positions are verified in verify_extraction_fidelity
        return []

    if "value" in by_field and ("predicate" in by_field or "entity" in by_field):
        fe_val = by_field["value"].evidence
        fe_pred = by_field.get("predicate")
        fe_ent = by_field.get("entity")
        fe_doc = (
            documents[fe_val.document_id].text
            if (documents and fe_val.document_id in documents)
            else primary_doc_text
        )
        if fe_pred is not None and fe_pred.evidence.document_id == fe_val.document_id:
            lo = min(fe_pred.evidence.start, fe_val.start)
            hi = max(fe_pred.evidence.end, fe_val.end)
            span_between = fe_doc[lo:hi]
            # FieldEvidence spans for predicate and value must belong to the same clause / sentence
            sub_clauses = _split_into_atomic_clauses(span_between, units=units)
            if len(sub_clauses) > 1 or _find_competing_predicate_spans(
                prop_predicate, span_between, vocabulary=vocabulary, entity=prop_entity
            ):
                return [
                    FidelityFinding(
                        code="RELATIONAL_ASSOCIATION_MISMATCH",
                        message="Declared FieldEvidence spans for 'predicate' and 'value' lie in different clauses or across a competing predicate.",
                        field="predicate",
                        detail={"span": span_between},
                    )
                ]
        if fe_ent is not None and fe_ent.evidence.document_id == fe_val.document_id:
            sent_start, sent_end, val_sent = enclosing_sentence(fe_doc, fe_val.start, fe_val.end)
            raw_sent_slice = fe_doc[sent_start:sent_end]
            sent_lpad = len(raw_sent_slice) - len(raw_sent_slice.lstrip())
            val_clauses = _split_into_atomic_clauses(val_sent, units=units)
            for c_text, c_s_rel, c_e_rel, _ in val_clauses:
                abs_s = sent_start + sent_lpad + c_s_rel
                abs_e = sent_start + sent_lpad + c_e_rel
                if abs_s <= fe_val.start and fe_val.end <= abs_e:
                    decl_ce, heur_ce = _find_competing_entity_spans(
                        prop_entity, c_text, vocabulary=vocabulary, prop_predicate=prop_predicate, prop=prop
                    )
                    if not _find_entity_spans(prop_entity, c_text, vocabulary=vocabulary) and (decl_ce or heur_ce):
                        if fe_ent.evidence.start < abs_s or fe_ent.evidence.end > abs_e:
                            return [
                                FidelityFinding(
                                    code="RELATIONAL_ASSOCIATION_MISMATCH",
                                    message="Declared FieldEvidence for 'entity' lies outside the competing subject clause containing 'value'.",
                                    field="entity",
                                    detail={"clause": c_text},
                                )
                            ]

    if any(line.count("|") >= 2 for r in unit.evidence for line in r.quote.splitlines()):
        return []

    hdr = _leading_header_text(primary_doc_text)
    header_ent_ok = bool(hdr and prop_entity and _find_entity_spans(prop_entity, hdr, vocabulary=vocabulary))
    header_pred_ok = bool(
        hdr
        and _predicate_grounded_in_evidence(
            prop_predicate, prop_entity, hdr, (hdr,), hdr, vocabulary, predicate_tokens_only=True
        )
    )

    quote_text = "\n".join(r.quote for r in unit.evidence)
    quote_findings = _verify_text_relational_association(
        quote_text,
        prop_entity,
        prop_predicate,
        val,
        header_entity_ok=header_ent_ok,
        header_predicate_ok=header_pred_ok,
        vocabulary=vocabulary,
        units=units,
        spec=spec,
        prop=prop,
    )
    if quote_findings:
        return quote_findings

    # Also verify enclosing sentences when a partial quote omitted a clause-level competing predicate/subject
    sent_text = "\n".join(full_sentences)
    if sent_text.strip() and sent_text.strip() != quote_text.strip():
        sent_findings = _verify_text_relational_association(
            sent_text,
            prop_entity,
            prop_predicate,
            val,
            header_entity_ok=header_ent_ok,
            header_predicate_ok=header_pred_ok,
            vocabulary=vocabulary,
            units=units,
            spec=spec,
            prop=prop,
        )
        if sent_findings:
            return sent_findings

    return []


def verify_extraction_fidelity(
    doc_text: str,
    unit: EvidenceUnit,
    *,
    field_evidence: Sequence[FieldEvidence] | None = None,
    require_field_evidence: Sequence[str] = (),
    vocabulary: Vocabulary | None = None,
    documents: Mapping[str, Any] | None = None,
) -> list[FidelityFinding]:
    """Run independent deterministic fidelity checks on an admitted EvidenceUnit."""
    findings: list[FidelityFinding] = []
    if not unit.evidence:
        return findings

    prop = unit.proposition
    cond_status = prop.get(
        "condition_status",
        "conditional" if prop.get("conditions") else "unconditional",
    )
    check_cond = cond_status == "unconditional"
    full_sentences: list[str] = []
    for ref in unit.evidence:
        ref_text = documents[ref.document_id].text if (documents and ref.document_id in documents) else doc_text
        findings.extend(check_clause_boundary_completeness(ref_text, ref, check_conditional=check_cond))
        _, _, sent = enclosing_sentence(ref_text, ref.start, ref.end)
        full_sentences.append(sent)

    combined_sentence = " ".join(full_sentences)

    # 1. Uncaptured condition / exception in sentence when proposition claims to be unconditional
    if cond_status == "unconditional" and _CONDITIONAL_MARKER_RE.search(combined_sentence):
        findings.append(
            FidelityFinding(
                code="UNCAPTURED_CONDITION",
                message=(
                    "Source sentence contains a conditional/exception marker, but proposition "
                    "was proposed as 'unconditional'."
                ),
                field="conditions",
                detail={"sentence": combined_sentence},
            )
        )

    # 2. Polarity parity check
    sentence_has_neg = bool(_CLAUSE_NEGATION_RE.search(combined_sentence))
    if sentence_has_neg and prop.get("polarity") == "affirmed":
        findings.append(
            FidelityFinding(
                code="POLARITY_MISMATCH",
                message="Source sentence contains clausal negation, but polarity is 'affirmed'.",
                field="polarity",
                detail={"sentence": combined_sentence},
            )
        )

    # 3. Bound operator consistency check
    val = prop.get("value")
    if isinstance(val, dict) and "operator" in val:
        op = val["operator"]
        for other_op, pat in _BOUND_MARKER_MAP.items():
            if other_op != op and pat.search(combined_sentence) and not _BOUND_MARKER_MAP[op].search(combined_sentence):
                findings.append(
                    FidelityFinding(
                        code="BOUND_OPERATOR_MISMATCH",
                        message=f"Sentence suggests '{other_op}' boundary, but proposal used '{op}'.",
                        field="value",
                        detail={"proposed_operator": op, "sentence": combined_sentence},
                    )
                )
                break

    # 3b. Direction-of-change sign parity for signed numeric values
    if isinstance(val, dict) and "value" in val:
        try:
            num_val = float(val["value"])
        except (TypeError, ValueError):
            num_val = None
        if num_val is not None:
            sentence_for_dir = re.sub(r"\b(?:lower|less)\s+than\b", " ", combined_sentence, flags=re.IGNORECASE)
            has_dec = bool(_DECREASE_MARKER_RE.search(sentence_for_dir))
            has_inc = bool(_INCREASE_MARKER_RE.search(sentence_for_dir))
            if has_dec and not has_inc and num_val > 0 and "decrease" not in prop.get("predicate", "").lower():
                findings.append(
                    FidelityFinding(
                        code="DIRECTION_OF_CHANGE_MISMATCH",
                        message="Sentence states a decrease/reduction, but proposed numeric value is positive.",
                        field="value",
                        detail={"value": val["value"], "sentence": combined_sentence},
                    )
                )
            elif has_inc and not has_dec and num_val < 0:
                findings.append(
                    FidelityFinding(
                        code="DIRECTION_OF_CHANGE_MISMATCH",
                        message="Sentence states an increase/growth, but proposed numeric value is negative.",
                        field="value",
                        detail={"value": val["value"], "sentence": combined_sentence},
                    )
                )

    # 4. Mandatory value grounding against cited evidence quotes (even without field_evidence)
    unit_reg = vocabulary.units if vocabulary is not None else STANDARD_UNITS
    val_spec: FieldSpec | None = None
    if vocabulary is not None and isinstance(prop.get("predicate"), str):
        try:
            val_spec = vocabulary.predicate(prop["predicate"]).value
        except UnmappedProposal:
            val_spec = None
    cited_quotes = " ".join(r.quote for r in unit.evidence)
    val_grounded = _value_markers_present(val, cited_quotes, units=unit_reg, spec=val_spec)
    if not val_grounded:
        findings.append(
            FidelityFinding(
                code="VALUE_NOT_IN_EVIDENCE",
                message=f"Proposed value {val!r} is not grounded in the cited evidence quote.",
                field="value",
                detail={"value": val, "quotes": [r.quote for r in unit.evidence]},
            )
        )

    # 4b. Mandatory entity grounding against cited evidence quotes / enclosing sentence
    primary_doc_text = (
        documents[unit.evidence[0].document_id].text
        if (documents and unit.evidence[0].document_id in documents)
        else doc_text
    )
    prop_entity = prop.get("entity")
    prop_predicate = prop.get("predicate")
    if isinstance(prop_entity, str) and prop_entity.strip():
        labels = _entity_candidate_labels(prop_entity, vocabulary)
        local_match = any(_term_in_text(t, lbl) for t in (cited_quotes, combined_sentence) for lbl in labels)
        if not local_match:
            header = _leading_header_text(primary_doc_text)
            competing = _has_competing_subject_in_sentences(
                full_sentences, labels, str(prop_predicate or "")
            )
            header_ok = bool(header) and not competing and any(_term_in_text(header, lbl) for lbl in labels)
            if not header_ok:
                findings.append(
                    FidelityFinding(
                        code="ENTITY_NOT_IN_EVIDENCE",
                        message=f"Proposed entity {prop_entity!r} is not grounded in the cited evidence.",
                        field="entity",
                        detail={"entity": prop_entity, "quotes": [r.quote for r in unit.evidence]},
                    )
                )

    # 4c. Mandatory predicate grounding against cited evidence quotes / enclosing sentence
    if isinstance(prop_predicate, str) and prop_predicate.strip():
        if not _predicate_grounded_in_evidence(
            prop_predicate,
            str(prop_entity or ""),
            cited_quotes,
            full_sentences,
            primary_doc_text,
            vocabulary,
        ):
            findings.append(
                FidelityFinding(
                    code="PREDICATE_NOT_IN_EVIDENCE",
                    message=f"Proposed predicate {prop_predicate!r} is not grounded in the cited evidence.",
                    field="predicate",
                    detail={"predicate": prop_predicate, "quotes": [r.quote for r in unit.evidence]},
                )
            )

    # 4d. Mandatory semantic scope grounding against cited evidence quotes / enclosing sentence
    prop_scope = prop.get("scope")
    if isinstance(prop_scope, Mapping) and prop_scope:
        ungrounded_scope = _scope_grounded_in_evidence(prop_scope, cited_quotes, full_sentences)
        if ungrounded_scope:
            findings.append(
                FidelityFinding(
                    code="SCOPE_NOT_IN_EVIDENCE",
                    message=f"Proposed scope qualifier(s) {ungrounded_scope!r} are not grounded in the cited evidence.",
                    field="scope",
                    detail={"ungrounded": ungrounded_scope, "quotes": [r.quote for r in unit.evidence]},
                )
            )

    # 5. Per-field evidence binding checks
    by_field = {fe.field: fe for fe in (field_evidence or ())}
    for req_field in require_field_evidence:
        if req_field not in by_field:
            findings.append(
                FidelityFinding(
                    code="MISSING_FIELD_EVIDENCE",
                    message=f"Required field '{req_field}' lacks an explicit FieldEvidence span.",
                    field=req_field,
                )
            )

    _list_item_re = re.compile(r"^value\[(\d+)\]$")
    primary_doc_id = unit.evidence[0].document_id if unit.evidence else None

    for fe in field_evidence or ():
        ref = fe.evidence
        fe_text = documents[ref.document_id].text if (documents and ref.document_id in documents) else doc_text
        if ref.start < 0 or ref.end > len(fe_text) or fe_text[ref.start : ref.end] != ref.quote:
            findings.append(
                FidelityFinding(
                    code="INVALID_FIELD_EVIDENCE",
                    message=f"FieldEvidence for '{fe.field}' does not match source text at declared offsets.",
                    field=fe.field,
                )
            )
            continue
        list_item_match = _list_item_re.match(fe.field)
        if list_item_match and val_spec is not None and val_spec.type == "list":
            idx = int(list_item_match.group(1))
            if not isinstance(val, (list, tuple)) or idx >= len(val):
                findings.append(
                    FidelityFinding(
                        code="INVALID_FIELD_EVIDENCE",
                        message=f"FieldEvidence '{fe.field}' index out of range for list value.",
                        field=fe.field,
                    )
                )
            elif not _value_markers_present(val[idx], ref.quote, units=unit_reg, spec=val_spec.element):
                findings.append(
                    FidelityFinding(
                        code="VALUE_NOT_IN_FIELD_EVIDENCE",
                        message=f"List element '{fe.field}' is not grounded in its declared FieldEvidence span '{ref.quote}'.",
                        field=fe.field,
                    )
                )
            continue
        if fe.field == "entity" and prop.get("entity"):
            ent_labels = _entity_candidate_labels(str(prop["entity"]), vocabulary)
            if not any(_term_in_text(ref.quote, lbl) for lbl in ent_labels):
                findings.append(
                    FidelityFinding(
                        code="ENTITY_NOT_IN_FIELD_EVIDENCE",
                        message=f"Declared entity '{prop['entity']}' is not grounded in its FieldEvidence span.",
                        field="entity",
                    )
                )
        elif fe.field == "predicate" and isinstance(prop_predicate, str) and prop_predicate.strip():
            if not _predicate_grounded_in_evidence(
                prop_predicate,
                str(prop_entity or ""),
                ref.quote,
                (ref.quote,),
                ref.quote,
                vocabulary,
                predicate_tokens_only=True,
            ):
                findings.append(
                    FidelityFinding(
                        code="PREDICATE_NOT_IN_FIELD_EVIDENCE",
                        message=f"Declared predicate '{prop_predicate}' is not grounded in its FieldEvidence span.",
                        field="predicate",
                    )
                )
        elif fe.field == "value" and not _value_markers_present(
            prop.get("value"), ref.quote, units=unit_reg, spec=val_spec
        ):
            findings.append(
                FidelityFinding(
                    code="VALUE_NOT_IN_FIELD_EVIDENCE",
                    message=f"Proposed value is not grounded in its declared FieldEvidence span '{ref.quote}'.",
                    field="value",
                )
            )
        elif fe.field == "conditions" and prop.get("conditions") and not _CONDITIONAL_MARKER_RE.search(ref.quote):
            # Check that at least condition terms or markers appear in the cited span
            cond_tokens = {
                t
                for c in prop["conditions"]
                for t in re.findall(r"[a-z0-9]+", str(c.get("field", "")).lower())
                if len(t) >= 3
            }
            if cond_tokens and not any(t in ref.quote.lower() for t in cond_tokens):
                findings.append(
                    FidelityFinding(
                        code="CONDITION_NOT_IN_FIELD_EVIDENCE",
                        message="Declared condition field is not grounded in its FieldEvidence span.",
                        field="conditions",
                    )
                )

    if val_spec is not None and val_spec.type == "list" and isinstance(val, (list, tuple)):
        prev_start: int | None = None
        for idx in range(len(val)):
            fe_key = f"value[{idx}]"
            fe_item = by_field.get(fe_key)
            if fe_item is None:
                findings.append(
                    FidelityFinding(
                        code="MISSING_FIELD_EVIDENCE",
                        message=f"List element '{fe_key}' lacks an explicit FieldEvidence span.",
                        field=fe_key,
                    )
                )
                continue
            item_ref = fe_item.evidence
            if primary_doc_id and item_ref.document_id != primary_doc_id:
                findings.append(
                    FidelityFinding(
                        code="INVALID_FIELD_EVIDENCE",
                        message=f"List element '{fe_key}' references document '{item_ref.document_id}' instead of '{primary_doc_id}'.",
                        field=fe_key,
                    )
                )
                continue
            if prev_start is not None and item_ref.start < prev_start:
                findings.append(
                    FidelityFinding(
                        code="LIST_ORDER_MISMATCH",
                        message=f"List element '{fe_key}' starts at {item_ref.start} before preceding element at {prev_start}.",
                        field=fe_key,
                    )
                )
            prev_start = item_ref.start

    # 5b. Relational association check: ensure (entity, predicate, value) align in same clause/row/column
    blocking_individual_codes = {
        "INVALID_EVIDENCE_SPAN",
        "QUOTE_SPAN_MISMATCH",
        "VALUE_NOT_IN_EVIDENCE",
        "ENTITY_NOT_IN_EVIDENCE",
        "PREDICATE_NOT_IN_EVIDENCE",
        "VALUE_NOT_IN_FIELD_EVIDENCE",
        "ENTITY_NOT_IN_FIELD_EVIDENCE",
        "PREDICATE_NOT_IN_FIELD_EVIDENCE",
        "MISSING_FIELD_EVIDENCE",
        "INVALID_FIELD_EVIDENCE",
        "LIST_ORDER_MISMATCH",
    }
    if not ({f.code for f in findings} & blocking_individual_codes):
        findings.extend(
            _verify_relational_association(
                primary_doc_text,
                unit,
                full_sentences,
                by_field=by_field,
                vocabulary=vocabulary,
                units=unit_reg,
                spec=val_spec,
                documents=documents,
            )
        )

    return findings


def evaluate_dual_readers(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate two independent readers against gold labels, explicitly measuring shared errors.

    Each case mapping expects keys: `gold`, `reader_a`, `reader_b`.
    Agreement is NOT treated as truth: `shared_error_rate` measures correlated failures
    where both readers agree on an incorrect structure.
    """
    total = len(cases)
    if total == 0:
        return {
            "total": 0,
            "agreement_rate": 0.0,
            "both_correct_rate": 0.0,
            "shared_error_rate": 0.0,
            "disagreement_rate": 0.0,
            "agreement_reliability": 0.0,
        }

    agree = 0
    both_correct = 0
    shared_errors = 0
    disagree = 0

    for c in cases:
        gold = c["gold"]
        ra = c["reader_a"]
        rb = c["reader_b"]
        if ra == rb:
            agree += 1
            if ra == gold:
                both_correct += 1
            else:
                shared_errors += 1
        else:
            disagree += 1

    return {
        "total": total,
        "agreement_rate": round(agree / total, 4),
        "both_correct_rate": round(both_correct / total, 4),
        "shared_error_rate": round(shared_errors / total, 4),
        "disagreement_rate": round(disagree / total, 4),
        # Fraction of agreements that are actually correct (never assume agreement == truth)
        "agreement_reliability": round(both_correct / agree, 4) if agree else 0.0,
    }


def render_canonical_claim(unit: EvidenceUnit) -> str:
    """Render a deterministic human-readable proposition for round-trip fidelity verification."""
    p = unit.proposition
    val = p["value"]
    if isinstance(val, dict) and "operator" in val:
        val_txt = f"{val['operator']} {val['value']} {val['unit']}"
    elif isinstance(val, dict) and "value" in val and "unit" in val:
        val_txt = f"{val['value']} {val['unit']}"
    else:
        val_txt = str(val)
    cstatus = p.get("condition_status", "conditional" if p.get("conditions") else "unconditional")
    cond_txt = ""
    if p.get("conditions"):
        parts = [f"{c['field']} {c['operator']} {c['value']}" for c in p["conditions"]]
        cond_txt = f" IF ({' AND '.join(parts)})"
    scope_txt = f" IN {p['scope']}" if p.get("scope") else ""
    pol_txt = "NOT " if p.get("polarity") == "negated" else ""
    return f"[{p['assertion_kind']} | {cstatus}] {p['entity']}.{p['predicate']} = {pol_txt}{val_txt}{cond_txt}{scope_txt}"
