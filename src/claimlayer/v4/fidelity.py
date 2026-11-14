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


def _extract_standalone_decimals_untyped(
    text: str,
    occupied_spans: Sequence[tuple[int, int]],
) -> list[Decimal]:
    out: list[Decimal] = []
    for m in re.finditer(r"(?<![0-9.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![0-9.])", text):
        span = (m.start(), m.end())
        if any(_spans_overlap(span, occ) for occ in occupied_spans):
            continue
        tok = m.group(0).replace(",", "")
        try:
            out.append(Decimal(tok))
        except InvalidOperation:
            continue
    return out


_PCT_PER_ANNUM_SUFFIX_RE = re.compile(r"\s*(?:per\s+annum|p\.a\.|per\s+year|annually)(?![a-z0-9_])", re.IGNORECASE)
_CUR_PER_YEAR_SUFFIX_RE = re.compile(r"\s*(?:per\s+year|annually|per\s+annum)(?![a-z0-9_])", re.IGNORECASE)
_CUR_PER_MONTH_SUFFIX_RE = re.compile(r"\s*(?:per\s+month|monthly)(?![a-z0-9_])", re.IGNORECASE)
_HOURS_PER_WEEK_SUFFIX_RE = re.compile(r"\s*(?:/\s*week|per\s+week|a\s+week)(?![a-z0-9_])", re.IGNORECASE)
_HOURS_PER_DAY_SUFFIX_RE = re.compile(r"\s*(?:/\s*day|per\s+day|a\s+day)(?![a-z0-9_])", re.IGNORECASE)


def _extract_typed_v4_quantities(
    text: str,
    units: UnitRegistry = STANDARD_UNITS,
) -> tuple[list[tuple[Decimal, str]], list[tuple[int, int]]]:
    """Extract (normalized_Decimal, v4_base_unit) pairs and occupied spans from prose."""
    typed: list[tuple[Decimal, str]] = []
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
            typed.append(mapped)

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
            typed.append((Decimal(amt), base_u))

    return typed, occupied


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
            has_dec = bool(_DECREASE_MARKER_RE.search(combined_sentence))
            has_inc = bool(_INCREASE_MARKER_RE.search(combined_sentence))
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
