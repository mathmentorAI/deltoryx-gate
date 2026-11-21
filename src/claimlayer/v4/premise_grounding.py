"""Deterministic premise-to-evidence correspondence and anti-co-occurrence checks for v4.

Validates that every extracted premise proposed by an LLM planner genuinely corresponds
to an authentic corpus source span (document, revision, verbatim span, clause boundary,
value, unit, sign/direction, entity, event/attribute, and explicit relative anchor)
without relying on mere lexical co-occurrence in a wide quote or trusting self-citation
to the user claim or system clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence
import unicodedata

from ..semantic.models import EvidenceRef, SourceDocument
from .fidelity import (
    FidelityFinding,
    _date_marker_present,
    _extract_standalone_decimal_spans_untyped,
    _extract_typed_v4_quantity_spans,
    check_clause_boundary_completeness,
    enclosing_sentence,
)
from .normalizers import normalize_date_token, normalize_separator_number
from .operations import CANONICAL_WEEKDAYS_EN, CANONICAL_WEEKDAYS_ES, normalize_weekday
from .spans import locate_quote
from .vocabulary import STANDARD_UNITS, UnitRegistry, UnmappedProposal, label_key

_FORBIDDEN_SOURCE_IDS: frozenset[str] = frozenset({
    "claim",
    "afirmacion",
    "afirmación",
    "user_claim",
    "hypothesis",
    "system_clock",
    "clock",
    "today",
    "hoy",
    "now",
    "external",
    "none",
    "null",
})

_FUNCTION_STOPWORDS: frozenset[str] = frozenset({
    # Spanish function words
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    "de", "a", "en", "por", "para", "con", "sin", "sobre", "entre", "hacia", "hasta", "desde",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "nuestros", "nuestras",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
    "le", "les", "se", "me", "te", "nos", "ella", "ellas", "ellos", "yo",
    "y", "e", "o", "u", "pero", "que", "como", "cuando", "donde",
    "es", "son", "fue", "fueron", "era", "eran", "ser", "esta", "estan", "estuvo", "ha", "han",
    # English function words
    "the", "an", "of", "in", "on", "at", "to", "for", "with", "without", "by", "from", "into", "onto",
    "his", "her", "their", "its", "my", "our", "your", "this", "that", "these", "those",
    "she", "he", "they", "it", "we", "you", "me", "him", "them",
    "and", "or", "but", "while", "whereas", "as", "when", "where",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
})

_DEICTIC_OR_META_TOKENS: frozenset[str] = frozenset({
    "anchor", "ancla", "reference", "referencia", "offset", "desplazamiento", "delta",
    "day", "days", "dia", "dias", "weekday", "date", "fecha", "period", "periodo",
    "actual", "current", "present", "presente",
    "speaker", "narrator", "narrador", "yo",
    "enunciado", "statement", "hecho", "fact", "ocurrencia", "occurrence",
    "evento", "event", "value", "valor", "amount", "importe", "cantidad", "quantity",
    "number", "numero", "result", "resultado",
})

_CONTESTABLE_ROLE_TOKENS: frozenset[str] = frozenset({
    "hoy", "today", "cita", "appointment", "meeting", "reunion", "acuerdo", "agreement",
})

_GENERIC_ANCHOR_OR_ROLE_LABELS: frozenset[str] = _DEICTIC_OR_META_TOKENS | _CONTESTABLE_ROLE_TOKENS

_ANAPHORIC_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "eso", "ello", "esto", "aquello", "entonces", "that", "it", "this", "then",
})

_CARDINAL_WORDS: dict[str, int] = {
    # Spanish 0..30 + round numbers (exclude bare indefinite articles 'un'/'una' except in unit phrases)
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "treinta": 30,
    "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
    "noventa": 90, "cien": 100, "ciento": 100, "mil": 1000,
    # English 0..30 + round numbers
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}

_UNIT_ARTICLE_ONE_RE = re.compile(
    r"\b(?:un|una)\s+(?:dia|dias|semana|semanas|mes|meses|ano|anos|hora|horas)\b",
    re.IGNORECASE,
)

_BACKWARD_DIR_RE = re.compile(
    r"\b(?:antes|atras|previo|previa|anterior(?:mente)?|hace|menos|disminuy\w*|reduc\w*|baj\w*|descuento|rest\w*|"
    r"before|earlier|ago|prior|previous(?:ly)?|minus|less|decreas\w*|drop\w*|subtract\w*)\b",
    re.IGNORECASE,
)

_FORWARD_DIR_RE = re.compile(
    r"\b(?:despues|posterior(?:mente)?|siguiente|adelante|luego|mas\s+tarde|aument\w*|subi\w*|increment\w*|sum\w*|"
    r"after|later|following|subsequent(?:ly)?|plus|more|increas\w*|rose|risen|added)\b",
    re.IGNORECASE,
)

_EXPLICIT_ANCHOR_ES_RE = re.compile(
    r"\b(?:antes|despu[eé]s|previo|previa|posterior|a\s+partir)\s+(?:de(?:l)?|a)\s+(?P<anchor>[^.,;!?]+)",
    re.IGNORECASE,
)

_EXPLICIT_ANCHOR_EN_RE = re.compile(
    r"\b(?:before|after|prior\s+to|following|from)\s+(?P<anchor>[^.,;!?]+)",
    re.IGNORECASE,
)

_CLAUSE_NUM_RE = re.compile(
    r"(?<![0-9.,])[-+]?(?:\d{1,3}(?:[.\s]\d{3})+,\d+|\d+(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+|\d+,\d{1,2}|\d+(?:\.\d+)?)(?![0-9.,])"
)

_LEGAL_SUFFIX_SPLIT_RE = re.compile(r"^(?:S\.?\s*A\.?|S\.?\s*L\.?|Inc\.?|LLC|Ltd\.?)\b", re.IGNORECASE)

_WEEKDAY_TOKEN_FIND_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


def _fold_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def _content_tokens(text: str) -> tuple[str, ...]:
    folded = _fold_accents(text or "").casefold()
    raw_tokens = re.findall(r"[a-z0-9]+", folded)
    return tuple(tok for tok in raw_tokens if tok not in _FUNCTION_STOPWORDS and len(tok) >= 2)


def _all_word_tokens(text: str) -> tuple[str, ...]:
    folded = _fold_accents(text or "").casefold()
    return tuple(re.findall(r"[a-z0-9]+", folded))


def _split_clauses(text: str) -> list[str]:
    """Split prose into elementary clauses without breaking inside formatted numbers or legal suffixes."""
    if not text or not text.strip():
        return []
    # First split on sentence boundaries (. ! ? ; \n) while preserving decimal dots and legal abbreviations like S.A./S.L.
    n = len(text)
    segments: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in ";!?\n":
            seg = text[start:i].strip()
            if seg:
                segments.append(seg)
            start = i + 1
        elif ch == ".":
            prev_ch = text[i - 1] if i > 0 else ""
            next_ch = text[i + 1] if i + 1 < n else ""
            if prev_ch.isdigit() and next_ch.isdigit():
                continue
            # Keep short abbreviations such as S.A. / S.L. / p.a. intact when followed without space or inside word
            if next_ch.isalpha():
                continue
            seg = text[start:i].strip()
            if seg:
                segments.append(seg)
            start = i + 1
        elif ch == ",":
            prev_ch = text[i - 1] if i > 0 else ""
            next_ch = text[i + 1] if i + 1 < n else ""
            if prev_ch.isdigit() and next_ch.isdigit():
                continue
            tail = text[i + 1 :].lstrip()
            if _LEGAL_SUFFIX_SPLIT_RE.match(tail):
                continue
            seg = text[start:i].strip()
            if seg:
                segments.append(seg)
            start = i + 1
    tail_seg = text[start:].strip()
    if tail_seg:
        segments.append(tail_seg)

    # Second pass: split coordinating clause conjunctions (y / e / pero / mientras / aunque / and / but / while / whereas)
    clauses: list[str] = []
    conj_pat = re.compile(r"\s+(?:y|e|pero|mientras|aunque|and|but|while|whereas)\s+", re.IGNORECASE)
    for seg in segments:
        parts = [p.strip() for p in conj_pat.split(seg) if p.strip()]
        clauses.extend(parts)
    return clauses


def _locate_span_in_doc(doc_text: str, quote: str) -> tuple[int, int, str] | None:
    q = (quote or "").strip()
    if not q:
        return None
    idx = doc_text.find(q)
    if idx >= 0:
        return idx, idx + len(q), doc_text[idx : idx + len(q)]
    span = locate_quote(doc_text, q, min_chars=min(4, len(q)))
    if span is not None:
        st, en = span
        return st, en, doc_text[st:en]
    tokens = [re.escape(tok) for tok in q.split()]
    if tokens:
        m = re.search(r"\s+".join(tokens), doc_text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end(), m.group(0)
    return None


def _extract_clause_numbers(clause_text: str) -> list[tuple[Decimal, str]]:
    """Extract `(absolute_or_signed_decimal, raw_token)` values from digits and small cardinal words."""
    found: list[tuple[Decimal, str]] = []
    for m in _CLAUSE_NUM_RE.finditer(clause_text):
        raw = m.group(0)
        norm = normalize_separator_number(raw)
        if norm is not None:
            try:
                found.append((Decimal(norm), raw))
            except InvalidOperation:
                pass
        # Also support European thousands dot notation (e.g. '1.250' -> 1250) when no comma is present
        unsigned = raw.lstrip("+-")
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", unsigned):
            sign = -1 if raw.startswith("-") else 1
            try:
                found.append((Decimal(unsigned.replace(".", "")) * sign, raw))
            except InvalidOperation:
                pass
    # Small cardinal words (e.g. 'dos' -> 2, 'two' -> 2, 'un día' -> 1)
    folded = _fold_accents(clause_text).casefold()
    for m in re.finditer(r"\b[a-z]+\b", folded):
        word = m.group(0)
        if word in _CARDINAL_WORDS:
            found.append((Decimal(_CARDINAL_WORDS[word]), word))
    for m in _UNIT_ARTICLE_ONE_RE.finditer(folded):
        found.append((Decimal(1), m.group(0)))
    return found


def _extract_clause_weekdays(clause_text: str) -> list[tuple[int, str, str]]:
    """Return `[(index_0_to_6, canonical_es, matched_token)]` for weekday words in *clause_text*."""
    out: list[tuple[int, str, str]] = []
    for m in _WEEKDAY_TOKEN_FIND_RE.finditer(clause_text):
        tok = m.group(0)
        if len(tok) <= 3:
            # Skip short abbreviations in free prose to avoid 'mar' / 'sat' / 'sun' false positives
            continue
        norm = normalize_weekday(tok)
        if norm is not None:
            out.append((norm[0], norm[1], tok))
    return out


def _extract_explicit_anchor_phrase(clause_text: str) -> dict[str, Any] | None:
    """Extract an explicit relative-anchor phrase (e.g. 'su cumpleaños') from a clause."""
    for pat in (_EXPLICIT_ANCHOR_ES_RE, _EXPLICIT_ANCHOR_EN_RE):
        m = pat.search(clause_text or "")
        if not m:
            continue
        raw_anchor = m.group("anchor").strip(" .,:;\"'")
        tokens = tuple(t for t in _content_tokens(raw_anchor) if t not in _ANAPHORIC_ANCHOR_TOKENS)
        if not tokens:
            return None
        return {"phrase": raw_anchor, "tokens": tokens}
    return None


def _is_material_named_entity(entity: str, all_doc_tokens: set[str]) -> bool:
    raw = (entity or "").strip()
    if not raw:
        return False
    tokens = _content_tokens(raw)
    if not tokens:
        return False
    tok_set = set(tokens)
    if tok_set <= _DEICTIC_OR_META_TOKENS:
        return False
    if tok_set <= _GENERIC_ANCHOR_OR_ROLE_LABELS:
        # Generic role/event words like 'hoy' or 'cita' are material whenever the corpus explicitly names them
        return bool(tok_set & all_doc_tokens)
    return True


def _has_competing_capitalized_entity(
    clause_text: str,
    expected_tokens: set[str],
) -> bool:
    """Return True if *clause_text* names a capitalized entity/proper noun disjoint from *expected_tokens*."""
    tokens_raw = re.findall(r"\b[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_-]{2,}\b", clause_text)
    if not tokens_raw:
        return False
    for idx, tok in enumerate(tokens_raw):
        ftok = _fold_accents(tok).casefold()
        if ftok in expected_tokens or ftok in _FUNCTION_STOPWORDS or ftok in _GENERIC_ANCHOR_OR_ROLE_LABELS:
            continue
        # Ignore common sentence-initial labels/currency codes
        if ftok in {"usd", "eur", "gbp", "hoy", "today", "gasto", "presupuesto", "multa", "proyecto", "budget", "expense"}:
            continue
        if normalize_weekday(ftok) is not None:
            continue
        return True
    return False


def _find_matching_clauses_for_value(
    clauses: Sequence[str],
    value: Any,
    unit: str | None,
    unit_registry: UnitRegistry,
) -> tuple[list[str], str | None]:
    """Locate the elementary clause(s) within *clauses* supporting *value* and check value/unit presence."""
    # 1. Weekday value
    w_norm = normalize_weekday(value) if isinstance(value, str) else None
    if w_norm is not None:
        w_idx = w_norm[0]
        matched = [
            cl for cl in clauses
            if any(cw_idx == w_idx for cw_idx, _, _ in _extract_clause_weekdays(cl))
        ]
        if not matched:
            return [], "VALUE_NOT_IN_EVIDENCE"
        return matched, None

    # 2. ISO date value
    if isinstance(value, str) and re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value.strip()):
        matched = [cl for cl in clauses if _date_marker_present(value, cl)]
        if not matched:
            return [], "VALUE_NOT_IN_EVIDENCE"
        return matched, None

    # 3. Numeric value (with optional unit)
    num_dec: Decimal | None = None
    if not isinstance(value, bool):
        try:
            num_dec = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            num_dec = None

    if num_dec is not None and num_dec.is_finite():
        abs_target = abs(num_dec)
        # Validate unit in UnitRegistry first if supplied
        target_dim: str | None = None
        target_base_val: Decimal | None = None
        target_base_unit: str | None = None
        if unit:
            try:
                u_obj = unit_registry.lookup(unit)
                target_dim = u_obj.dimension
                target_base_val = abs_target * Decimal(u_obj.factor)
                target_base_unit = unit_registry.base(target_dim)
            except UnmappedProposal:
                return [], f"UNKNOWN_UNIT:{unit}"

        matched_clauses: list[str] = []
        unit_mismatch_seen = False
        for cl in clauses:
            cl_nums = _extract_clause_numbers(cl)
            raw_match = any(abs(n) == abs_target for n, _ in cl_nums)
            typed_spans, _ = _extract_typed_v4_quantity_spans(cl, unit_registry)

            if unit is None:
                if raw_match:
                    matched_clauses.append(cl)
                continue

            # Unit declared: require compatible unit or dimension in the clause!
            assert target_dim is not None and target_base_val is not None
            exact_typed_match = any(
                unit_registry.lookup(base_u).dimension == target_dim and abs(mag) == target_base_val
                for mag, base_u, _ in typed_spans
            )
            if exact_typed_match:
                matched_clauses.append(cl)
                continue

            # Also check explicit unit aliases / tokens in clause alongside exact raw_match (e.g. 'días', 'EUR', 'USD')
            cl_folded = _fold_accents(cl).casefold()
            u_obj = unit_registry.lookup(unit)
            unit_terms = {_fold_accents(lbl).casefold() for lbl in (u_obj.name, *u_obj.aliases)}
            has_declared_unit_token = any(
                (re.search(r"(?<![a-z0-9])" + re.escape(ut) + r"(?![a-z0-9])", cl_folded) is not None)
                if ut.replace(" ", "").isalnum()
                else (ut in cl_folded)
                for ut in unit_terms if ut
            )
            if raw_match and has_declared_unit_token:
                matched_clauses.append(cl)
                continue

            # Check if the raw magnitude appears in this clause with a CONFLICTING unit!
            if raw_match and typed_spans:
                if any(unit_registry.lookup(base_u).dimension != target_dim for _, base_u, _ in typed_spans):
                    unit_mismatch_seen = True
                    continue
                if any(
                    unit_registry.lookup(base_u).dimension == target_dim and abs(mag) != target_base_val
                    for mag, base_u, _ in typed_spans
                ):
                    unit_mismatch_seen = True
                    continue
            if raw_match and not typed_spans:
                # Check if clause explicitly names a foreign unit word (e.g. 'semanas' when premise claims 'calendar_days')
                other_unit_hit = False
                for other_u in unit_registry._units.values():
                    if other_u.name == u_obj.name:
                        continue
                    for al in (other_u.name, *other_u.aliases):
                        al_f = _fold_accents(al).casefold()
                        if len(al_f) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(al_f) + r"(?![a-z0-9])", cl_folded):
                            other_unit_hit = True
                            break
                    if other_unit_hit:
                        break
                if other_unit_hit:
                    unit_mismatch_seen = True
                else:
                    matched_clauses.append(cl)

        if matched_clauses:
            return matched_clauses, None
        if unit_mismatch_seen:
            return [], "UNIT_MISMATCH_WITH_EVIDENCE"
        return [], "VALUE_NOT_IN_EVIDENCE"

    # 4. Boolean or generic text value
    if isinstance(value, bool):
        return list(clauses), None
    tokens = _content_tokens(str(value or ""))
    if not tokens:
        return list(clauses), None
    matched = [
        cl for cl in clauses
        if set(tokens) <= set(_all_word_tokens(cl))
    ]
    if not matched:
        return [], "VALUE_NOT_IN_EVIDENCE"
    return matched, None


def _check_direction_and_sign(
    value: Any,
    direction: str | None,
    unit: str | None,
    predicate: str,
    val_clauses: Sequence[str],
) -> str | None:
    """Verify sign/direction consistency against the matched evidence clause(s)."""
    num_dec: Decimal | None = None
    if not isinstance(value, bool) and value is not None:
        try:
            d = Decimal(str(value).strip())
            if d.is_finite():
                num_dec = d
        except (InvalidOperation, ValueError):
            num_dec = None

    dir_norm = _fold_accents(str(direction or "").strip()).casefold()
    neg_dirs = {"before", "antes", "backward", "atras", "minus", "menos", "decrease", "disminucion"}
    pos_dirs = {"after", "despues", "forward", "adelante", "plus", "mas", "increase", "aumento"}

    if num_dec is not None:
        if dir_norm in neg_dirs and num_dec > 0:
            return "DIRECTION_SIGN_MISMATCH"
        if dir_norm in pos_dirs and num_dec < 0:
            return "DIRECTION_SIGN_MISMATCH"

    joined = _fold_accents(" ".join(val_clauses)).casefold()
    has_back = bool(_BACKWARD_DIR_RE.search(joined))
    has_fwd = bool(_FORWARD_DIR_RE.search(joined))
    has_minus_token = bool(re.search(r"(?:^|[\s:(])-\s*\d", joined))

    is_shift_or_signed = bool(
        dir_norm in (neg_dirs | pos_dirs)
        or (unit and label_key(unit) in {"calendar_days", "days", "day", "weeks", "week", "dia", "dias", "día", "días", "semana", "semanas"})
        or any(tok in label_key(predicate) for tok in ("offset", "shift", "delta", "desplazamiento", "diferencia"))
    )

    if has_back and not has_fwd:
        if dir_norm in pos_dirs:
            return "DIRECTION_SIGN_MISMATCH"
        if num_dec is not None and num_dec > 0 and is_shift_or_signed:
            return "DIRECTION_SIGN_MISMATCH"

    if has_fwd and not has_back:
        if dir_norm in neg_dirs:
            return "DIRECTION_SIGN_MISMATCH"
        if num_dec is not None and num_dec < 0:
            return "DIRECTION_SIGN_MISMATCH"

    if num_dec is not None and num_dec < 0 and not has_back and not has_minus_token:
        return "DIRECTION_SIGN_MISMATCH"

    return None


@dataclass(frozen=True)
class PremiseGroundingResult:
    """Deterministic verification result for one extracted premise."""

    claim_id: str
    grounded: bool
    evidence_ref: EvidenceRef | None
    defects: tuple[FidelityFinding, ...] = ()
    checks_passed: tuple[str, ...] = ()
    normalized_value: Any = None
    normalized_unit: str | None = None
    matched_clauses: tuple[str, ...] = ()
    explicit_anchor: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "grounded": self.grounded,
            "evidence": (
                {
                    "document_id": self.evidence_ref.document_id,
                    "revision": self.evidence_ref.revision,
                    "start": self.evidence_ref.start,
                    "end": self.evidence_ref.end,
                    "quote": self.evidence_ref.quote,
                }
                if self.evidence_ref is not None
                else None
            ),
            "defects": [
                {"code": d.code, "message": d.message, "field": d.field, "detail": d.detail}
                for d in self.defects
            ],
            "checks_passed": list(self.checks_passed),
            "normalized_value": self.normalized_value,
            "normalized_unit": self.normalized_unit,
            "matched_clauses": list(self.matched_clauses),
            "explicit_anchor": dict(self.explicit_anchor) if self.explicit_anchor else None,
        }


def verify_premise_grounding(
    raw_premise: Mapping[str, Any],
    documents: Mapping[str, SourceDocument],
    *,
    unit_registry: UnitRegistry = STANDARD_UNITS,
    claim_text: str | None = None,
) -> PremiseGroundingResult:
    """Verify literal span grounding and elementary-clause semantic correspondence for one premise."""
    cid = str(raw_premise.get("claim_id") or "premise")
    doc_id = str(raw_premise.get("document_id") or raw_premise.get("doc_id") or "").strip()
    quote = str(raw_premise.get("quote") or raw_premise.get("verbatim_quote") or "").strip()

    if not doc_id or not quote:
        return PremiseGroundingResult(
            claim_id=cid,
            grounded=False,
            evidence_ref=None,
            defects=(
                FidelityFinding(
                    code="MISSING_SOURCE_REFERENCE",
                    message="Premise must cite a non-empty corpus document_id and verbatim quote.",
                    field="evidence",
                    detail={"document_id": doc_id, "has_quote": bool(quote)},
                ),
            ),
        )

    if _fold_accents(doc_id.casefold()) in _FORBIDDEN_SOURCE_IDS:
        return PremiseGroundingResult(
            claim_id=cid,
            grounded=False,
            evidence_ref=None,
            defects=(
                FidelityFinding(
                    code="FORBIDDEN_NON_CORPUS_SOURCE",
                    message=f"Source '{doc_id}' is not an admissible corpus document (claim/clock cannot serve as premise).",
                    field="document_id",
                    detail={"document_id": doc_id},
                ),
            ),
        )

    doc = documents.get(doc_id)
    if doc is None:
        return PremiseGroundingResult(
            claim_id=cid,
            grounded=False,
            evidence_ref=None,
            defects=(
                FidelityFinding(
                    code="UNKNOWN_SOURCE_DOCUMENT",
                    message=f"Document '{doc_id}' does not exist in the supplied corpus.",
                    field="document_id",
                    detail={"document_id": doc_id, "available_documents": sorted(documents)},
                ),
            ),
        )

    raw_rev = raw_premise.get("revision")
    if raw_rev is not None and str(raw_rev).strip() not in ("", "rev-0") and str(raw_rev).strip() != doc.revision:
        return PremiseGroundingResult(
            claim_id=cid,
            grounded=False,
            evidence_ref=None,
            defects=(
                FidelityFinding(
                    code="STALE_DOCUMENT_REVISION",
                    message=f"Premise cites revision '{raw_rev}', but active revision of '{doc_id}' is '{doc.revision}'.",
                    field="revision",
                    detail={"cited_revision": str(raw_rev), "active_revision": doc.revision},
                ),
            ),
        )

    span = _locate_span_in_doc(doc.text, quote)
    if span is None:
        return PremiseGroundingResult(
            claim_id=cid,
            grounded=False,
            evidence_ref=None,
            defects=(
                FidelityFinding(
                    code="QUOTE_NOT_FOUND_IN_SOURCE",
                    message=f"Cited quote {quote!r} was not found verbatim in document '{doc_id}'.",
                    field="quote",
                    detail={"document_id": doc_id, "quote": quote},
                ),
            ),
        )

    start, end, exact_slice = span
    ref = EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=end,
        quote=exact_slice,
    )

    defects: list[FidelityFinding] = []
    checks_passed: list[str] = ["source_document_and_revision", "verbatim_span_bound"]

    cond_status = str(raw_premise.get("condition_status") or "unconditional")
    boundary_findings = check_clause_boundary_completeness(
        doc.text, ref, check_conditional=(cond_status == "unconditional")
    )
    if boundary_findings:
        defects.extend(boundary_findings)
    else:
        checks_passed.append("clause_boundary_complete")

    # Normalize value and unit if applicable
    raw_val = raw_premise.get("value")
    raw_unit = raw_premise.get("unit")
    norm_val = raw_val
    norm_unit: str | None = str(raw_unit).strip() if isinstance(raw_unit, str) and raw_unit.strip() else None

    if isinstance(raw_val, Mapping) and "value" in raw_val:
        if norm_unit is None and raw_val.get("unit"):
            norm_unit = str(raw_val.get("unit")).strip()
        raw_val = raw_val.get("value")
        norm_val = raw_val

    w_norm = normalize_weekday(raw_val) if isinstance(raw_val, str) else None
    if w_norm is not None:
        norm_val = w_norm[1] if w_norm[2] == "es" else CANONICAL_WEEKDAYS_EN[w_norm[0]]
        norm_unit = None  # Weekday values do not carry a UnitRegistry dimension
    elif isinstance(raw_val, str):
        d_norm = normalize_date_token(raw_val)
        if d_norm is not None and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d_norm):
            norm_val = d_norm
        else:
            sep = normalize_separator_number(raw_val)
            if sep is not None:
                try:
                    d_dec = Decimal(sep)
                    norm_val = int(d_dec) if d_dec == d_dec.to_integral_value() else float(d_dec)
                except (InvalidOperation, ValueError):
                    pass

    if norm_unit is not None:
        try:
            norm_unit = unit_registry.lookup(norm_unit).name
        except UnmappedProposal:
            defects.append(
                FidelityFinding(
                    code=f"UNKNOWN_UNIT:{norm_unit}",
                    message=f"Premise unit '{norm_unit}' is not declared in UnitRegistry.",
                    field="unit",
                    detail={"unit": norm_unit},
                )
            )

    # Split matched quote (or enclosing sentence if quote is a tiny subclause without context) into clauses
    _, _, sent_text = enclosing_sentence(doc.text, start, end)
    quote_clauses = _split_clauses(exact_slice)
    if not quote_clauses:
        quote_clauses = [exact_slice]

    val_clauses, val_err = _find_matching_clauses_for_value(quote_clauses, norm_val, norm_unit, unit_registry)
    if val_err is not None:
        defects.append(
            FidelityFinding(
                code=val_err,
                message=f"Premise value {norm_val!r} (unit={norm_unit!r}) could not be verified in cited quote {exact_slice!r}.",
                field="value",
                detail={"value": norm_val, "unit": norm_unit, "quote": exact_slice},
            )
        )
        val_clauses = list(quote_clauses)
    else:
        checks_passed.append("value_and_unit_correspondence")

    # Sign / direction correspondence against the matched value clause(s)
    direction = raw_premise.get("direction")
    predicate = str(raw_premise.get("predicate") or "")
    dir_err = _check_direction_and_sign(norm_val, direction, norm_unit, predicate, val_clauses)
    if dir_err is not None:
        defects.append(
            FidelityFinding(
                code=dir_err,
                message=(
                    f"Premise value/direction ({norm_val!r}, direction={direction!r}) contradicts directional cues in "
                    f"matched clause {val_clauses!r}."
                ),
                field="direction",
                detail={"value": norm_val, "direction": direction, "matched_clauses": val_clauses},
            )
        )
    else:
        checks_passed.append("direction_sign_correspondence")

    # Build corpus-wide token index to distinguish genuinely absent entities/events from competing-clause swaps
    all_doc_tokens: set[str] = set()
    for d_obj in documents.values():
        all_doc_tokens.update(_content_tokens(d_obj.text))
    doc_clauses_all = _split_clauses(doc.text)
    val_clause_tokens: set[str] = set()
    for vc in val_clauses:
        val_clause_tokens.update(_all_word_tokens(vc))
    quote_all_tokens = set(_all_word_tokens(exact_slice))
    sent_header = sent_text.split(":", 1)[0] if ":" in sent_text else ""
    sent_header_tokens = set(_all_word_tokens(sent_header))

    # Entity correspondence (anti-co-occurrence across elementary clauses)
    entity = str(raw_premise.get("entity") or raw_premise.get("subject") or "").strip()
    if entity and _is_material_named_entity(entity, all_doc_tokens):
        ent_tokens = set(_content_tokens(entity))
        in_val_clause = bool(ent_tokens) and ent_tokens <= val_clause_tokens
        in_sent_header = bool(ent_tokens) and bool(sent_header_tokens) and ent_tokens <= sent_header_tokens
        in_doc_header = bool(ent_tokens) and bool(doc_clauses_all) and ent_tokens <= set(_all_word_tokens(doc_clauses_all[0]))

        if in_val_clause:
            checks_passed.append("entity_same_clause_correspondence")
        elif (in_sent_header or in_doc_header) and not _has_competing_capitalized_entity(" ".join(val_clauses), ent_tokens):
            checks_passed.append("entity_header_correspondence")
        else:
            in_other_quote_clause = bool(ent_tokens) and ent_tokens <= quote_all_tokens
            in_other_doc_clause = bool(ent_tokens) and ent_tokens <= all_doc_tokens
            if in_other_quote_clause or in_other_doc_clause:
                defects.append(
                    FidelityFinding(
                        code="ENTITY_VALUE_CLAUSE_MISMATCH",
                        message=(
                            f"Entity '{entity}' appears in another clause of the corpus, not in the clause stating "
                            f"value {norm_val!r} ({val_clauses!r})."
                        ),
                        field="entity",
                        detail={"entity": entity, "value": norm_val, "matched_clauses": val_clauses},
                    )
                )
            else:
                defects.append(
                    FidelityFinding(
                        code="ENTITY_NOT_IN_EVIDENCE",
                        message=f"Material entity '{entity}' is not grounded in the cited evidence clause.",
                        field="entity",
                        detail={"entity": entity, "matched_clauses": val_clauses},
                    )
                )

    # Event / attribute correspondence (anti-co-occurrence & anchor-event guard)
    event_label = str(raw_premise.get("event") or "").strip()
    if event_label:
        ev_tokens = set(_content_tokens(event_label))
        if ev_tokens:
            ev_in_val = ev_tokens <= val_clause_tokens
            ev_in_corpus = ev_tokens <= all_doc_tokens
            is_pure_meta = ev_tokens <= _DEICTIC_OR_META_TOKENS
            is_generic_role = ev_tokens <= _GENERIC_ANCHOR_OR_ROLE_LABELS
            if ev_in_val:
                checks_passed.append("event_same_clause_correspondence")
            elif not is_pure_meta and ev_in_corpus:
                defects.append(
                    FidelityFinding(
                        code="EVENT_VALUE_MISMATCH",
                        message=(
                            f"Event/attribute '{event_label}' belongs to a different clause in the corpus and does not "
                            f"correspond to matched clause {val_clauses!r}."
                        ),
                        field="event",
                        detail={"event": event_label, "value": norm_val, "matched_clauses": val_clauses},
                    )
                )
            elif w_norm is not None and not is_generic_role:
                # For weekday/date anchors, a concrete event name absent from the clause (e.g. 'cumpleaños' on 'Hoy es viernes')
                defects.append(
                    FidelityFinding(
                        code="EVENT_VALUE_MISMATCH",
                        message=(
                            f"Anchor event '{event_label}' is not evidenced in clause {val_clauses!r} stating '{norm_val}'."
                        ),
                        field="event",
                        detail={"event": event_label, "value": norm_val, "matched_clauses": val_clauses},
                    )
                )
            elif not is_generic_role and any(":" in vc for vc in val_clauses):
                # Explicit key:value clause whose attribute label conflicts with the stated event
                defects.append(
                    FidelityFinding(
                        code="EVENT_VALUE_MISMATCH",
                        message=(
                            f"Event/attribute '{event_label}' does not match labeled clause {val_clauses!r}."
                        ),
                        field="event",
                        detail={"event": event_label, "value": norm_val, "matched_clauses": val_clauses},
                    )
                )

    explicit_anchor = None
    for vc in val_clauses:
        cand_anchor = _extract_explicit_anchor_phrase(vc)
        if cand_anchor is not None:
            explicit_anchor = cand_anchor
            break

    # Validate internal anchor_reference if declared on an offset premise
    anchor_ref_decl = str(raw_premise.get("anchor_reference") or "").strip()
    if explicit_anchor is not None and anchor_ref_decl:
        decl_tokens = set(_content_tokens(anchor_ref_decl)) - _ANAPHORIC_ANCHOR_TOKENS
        exp_tokens = set(explicit_anchor["tokens"])
        if decl_tokens and exp_tokens and not (decl_tokens & exp_tokens):
            defects.append(
                FidelityFinding(
                    code="ANCHOR_MISMATCH",
                    message=(
                        f"Declared anchor_reference '{anchor_ref_decl}' conflicts with explicit anchor "
                        f"'{explicit_anchor['phrase']}' in cited clause."
                    ),
                    field="anchor_reference",
                    detail={"declared_anchor": anchor_ref_decl, "explicit_anchor": explicit_anchor["phrase"]},
                )
            )

    return PremiseGroundingResult(
        claim_id=cid,
        grounded=(len(defects) == 0),
        evidence_ref=ref,
        defects=tuple(defects),
        checks_passed=tuple(dict.fromkeys(checks_passed)),
        normalized_value=norm_val,
        normalized_unit=norm_unit,
        matched_clauses=tuple(val_clauses),
        explicit_anchor=explicit_anchor,
    )


def verify_step_anchor_correspondence(
    step_id: str,
    anchor_grounding: PremiseGroundingResult,
    anchor_raw: Mapping[str, Any],
    offset_grounding: PremiseGroundingResult,
    offset_raw: Mapping[str, Any],
) -> list[FidelityFinding]:
    """Verify that the anchor premise wired into a relative-offset step matches the offset's explicit anchor."""
    findings: list[FidelityFinding] = []
    anchor_texts = " ".join(
        part
        for part in (
            str(anchor_raw.get("entity") or anchor_raw.get("subject") or ""),
            str(anchor_raw.get("event") or ""),
            str(anchor_raw.get("predicate") or ""),
            " ".join(anchor_grounding.matched_clauses),
        )
        if part
    )
    anchor_tokens = set(_content_tokens(anchor_texts))

    exp_anchor = offset_grounding.explicit_anchor
    if exp_anchor is not None:
        exp_tokens = set(exp_anchor["tokens"])
        if exp_tokens and not (exp_tokens & anchor_tokens):
            findings.append(
                FidelityFinding(
                    code="ANCHOR_MISMATCH",
                    message=(
                        f"Step '{step_id}' uses anchor {anchor_grounding.matched_clauses!r} "
                        f"(value={anchor_grounding.normalized_value!r}), but offset clause requires anchor "
                        f"'{exp_anchor['phrase']}'."
                    ),
                    field="anchor",
                    detail={
                        "step_id": step_id,
                        "explicit_anchor_in_quote": exp_anchor["phrase"],
                        "used_anchor_clauses": list(anchor_grounding.matched_clauses),
                        "used_anchor_value": anchor_grounding.normalized_value,
                        "offset_clauses": list(offset_grounding.matched_clauses),
                    },
                )
            )
            return findings

    decl_ref = str(offset_raw.get("anchor_reference") or "").strip()
    if decl_ref:
        decl_tokens = set(_content_tokens(decl_ref)) - _ANAPHORIC_ANCHOR_TOKENS
        if decl_tokens and anchor_tokens and not (decl_tokens & anchor_tokens):
            findings.append(
                FidelityFinding(
                    code="ANCHOR_MISMATCH",
                    message=(
                        f"Step '{step_id}' declares anchor_reference '{decl_ref}', which does not match "
                        f"selected anchor premise {anchor_grounding.matched_clauses!r}."
                    ),
                    field="anchor",
                    detail={
                        "step_id": step_id,
                        "declared_anchor_reference": decl_ref,
                        "used_anchor_clauses": list(anchor_grounding.matched_clauses),
                        "used_anchor_value": anchor_grounding.normalized_value,
                    },
                )
            )
    return findings


def verify_claim_target_correspondence(
    claim_text: str,
    claim_target: Mapping[str, Any],
    *,
    unit_registry: UnitRegistry = STANDARD_UNITS,
) -> list[FidelityFinding]:
    """Ensure the structured `claim_target` faithfully reflects the consumer's `original_claim`."""
    findings: list[FidelityFinding] = []
    if not isinstance(claim_target, Mapping) or "value" not in claim_target:
        findings.append(
            FidelityFinding(
                code="MISSING_CLAIM_TARGET",
                message="Proposal must extract a structured 'claim_target' with 'value'.",
                field="claim_target",
            )
        )
        return findings

    val = claim_target.get("value")
    unit = claim_target.get("unit")
    clauses = [claim_text]

    if isinstance(val, str):
        w_norm = normalize_weekday(val)
        if w_norm is not None:
            idx = w_norm[0]
            if not any(cw_idx == idx for cw_idx, _, _ in _extract_clause_weekdays(claim_text)):
                findings.append(
                    FidelityFinding(
                        code="CLAIM_TARGET_MISMATCH_WITH_CLAIM_TEXT",
                        message=f"claim_target weekday {val!r} does not appear in original claim {claim_text!r}.",
                        field="claim_target.value",
                        detail={"claim_target_value": val, "claim_text": claim_text},
                    )
                )
            return findings

    _, val_err = _find_matching_clauses_for_value(
        clauses, val, str(unit).strip() if isinstance(unit, str) and unit.strip() else None, unit_registry
    )
    if val_err is not None:
        findings.append(
            FidelityFinding(
                code="CLAIM_TARGET_MISMATCH_WITH_CLAIM_TEXT",
                message=f"claim_target ({val!r}, unit={unit!r}) does not match original claim {claim_text!r}.",
                field="claim_target",
                detail={"claim_target": dict(claim_target), "reason": val_err},
            )
        )
    return findings


def diagnose_alternate_anchors(
    documents: Mapping[str, SourceDocument],
    *,
    offset_days: int,
    used_anchor_weekday: str,
    used_anchor_clauses: Sequence[str],
    offset_clauses: Sequence[str],
    claimed_value: Any,
    explicit_anchor_phrase: str | None = None,
) -> list[dict[str, Any]]:
    """Identify other weekday mentions in the corpus that could mislead if chosen as anchor."""
    claimed_norm = normalize_weekday(claimed_value)
    used_norm = normalize_weekday(used_anchor_weekday)
    if claimed_norm is None or used_norm is None:
        return []
    claimed_idx = claimed_norm[0]
    used_clauses_folded = {_fold_accents(c).casefold() for c in used_anchor_clauses}

    diagnostics: list[dict[str, Any]] = []
    for doc_id, doc in sorted(documents.items()):
        for clause in _split_clauses(doc.text):
            if _fold_accents(clause).casefold() in used_clauses_folded:
                continue
            for w_idx, w_es, raw_tok in _extract_clause_weekdays(clause):
                if (w_idx + offset_days) % 7 == claimed_idx and w_idx != used_norm[0]:
                    anchor_hint = (
                        f"'{explicit_anchor_phrase}'"
                        if explicit_anchor_phrase
                        else f"la cláusula {list(used_anchor_clauses)!r}"
                    )
                    diagnostics.append({
                        "document_id": doc_id,
                        "incorrect_anchor_weekday": w_es,
                        "incorrect_anchor_clause": clause,
                        "offset_days": offset_days,
                        "would_yield_weekday": CANONICAL_WEEKDAYS_ES[(w_idx + offset_days) % 7],
                        "correct_anchor_weekday": used_norm[1],
                        "correct_anchor_clauses": list(used_anchor_clauses),
                        "explicit_anchor_phrase": explicit_anchor_phrase,
                        "offset_clauses": list(offset_clauses),
                        "explanation": (
                            f"Aunque '{w_es}' (en «{clause}») con desplazamiento {offset_days:+d} daría "
                            f"'{CANONICAL_WEEKDAYS_ES[claimed_idx]}', es un ancla incorrecta porque el desplazamiento "
                            f"relativo en «{offset_clauses[0] if offset_clauses else ''}» está anclado a {anchor_hint} "
                            f"('{used_norm[1]}'), cuyo cálculo exacto es '{CANONICAL_WEEKDAYS_ES[(used_norm[0] + offset_days) % 7]}'."
                        ),
                    })
    return diagnostics
