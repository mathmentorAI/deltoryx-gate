"""Deterministic span, quantity, unit, and date grounding checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable

from claimlayer.models import Assertion, Issue, Policy, Source, normalize_entity_key, normalize_key

_QUOTE_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2212": "-",
        "\u00a0": " ",
    }
)

_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff]")
_LINEBREAK_HYPHEN_RE = re.compile(
    r"([a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])[\-\u2010\u2011]\s*\r?\n\s*([a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])"
)

_ONES_EN: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_EN: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ONES_ES: dict[str, int] = {
    "cero": 0,
    "un": 1,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciséis": 16,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "veintiún": 21,
    "veintiun": 21,
    "veintiuno": 21,
    "veintidós": 22,
    "veintidos": 22,
    "veintitrés": 23,
    "veintitres": 23,
    "veinticuatro": 24,
    "veinticinco": 25,
    "veintiséis": 26,
    "veintiseis": 26,
    "veintisiete": 27,
    "veintiocho": 28,
    "veintinueve": 29,
}
_TENS_ES: dict[str, int] = {
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
}
_HUNDREDS_ES: dict[str, int] = {
    "cien": 100,
    "ciento": 100,
    "doscientos": 200,
    "trescientos": 300,
    "cuatrocientos": 400,
    "quinientos": 500,
    "seiscientos": 600,
    "setecientos": 700,
    "ochocientos": 800,
    "novecientos": 900,
}

_SCALE_FACTORS: dict[str, Decimal] = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "mil": Decimal("1000"),
    "miles": Decimal("1000"),
    "m": Decimal("1000000"),
    "mm": Decimal("1000000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "millon": Decimal("1000000"),
    "millón": Decimal("1000000"),
    "millones": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
    "mil millones": Decimal("1000000000"),
    "billón": Decimal("1000000000000"),
    "billon": Decimal("1000000000000"),
}

_UNIT_ALIASES: dict[str, str] = {
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
    "dólar": "USD",
    "dólares": "USD",
    "dolar": "USD",
    "dolares": "USD",
    "€": "EUR",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "pound": "GBP",
    "pounds": "GBP",
    "libra": "GBP",
    "libras": "GBP",
    "chf": "CHF",
    "jpy": "JPY",
    "cad": "CAD",
    "aud": "AUD",
    "mxn": "MXN",
    "%": "PERCENT",
    "percent": "PERCENT",
    "pct": "PERCENT",
    "per cent": "PERCENT",
    "per cent.": "PERCENT",
    "por ciento": "PERCENT",
    "bps": "BPS",
    "basis point": "BPS",
    "basis points": "BPS",
    "punto básico": "BPS",
    "puntos básicos": "BPS",
    "punto basico": "BPS",
    "puntos basicos": "BPS",
    "business day": "BUSINESS_DAYS",
    "business days": "BUSINESS_DAYS",
    "working day": "BUSINESS_DAYS",
    "working days": "BUSINESS_DAYS",
    "día hábil": "BUSINESS_DAYS",
    "días hábiles": "BUSINESS_DAYS",
    "dia habil": "BUSINESS_DAYS",
    "dias habiles": "BUSINESS_DAYS",
    "calendar day": "CALENDAR_DAYS",
    "calendar days": "CALENDAR_DAYS",
    "día natural": "CALENDAR_DAYS",
    "días naturales": "CALENDAR_DAYS",
    "dia natural": "CALENDAR_DAYS",
    "dias naturales": "CALENDAR_DAYS",
    "day": "DAYS",
    "days": "DAYS",
    "día": "DAYS",
    "días": "DAYS",
    "dia": "DAYS",
    "dias": "DAYS",
    "week": "WEEKS",
    "weeks": "WEEKS",
    "semana": "WEEKS",
    "semanas": "WEEKS",
    "month": "MONTHS",
    "months": "MONTHS",
    "mes": "MONTHS",
    "meses": "MONTHS",
    "year": "YEARS",
    "years": "YEARS",
    "año": "YEARS",
    "años": "YEARS",
    "ano": "YEARS",
    "anos": "YEARS",
    "hour": "HOURS",
    "hours": "HOURS",
    "hora": "HOURS",
    "horas": "HOURS",
    "hr": "HOURS",
    "hrs": "HOURS",
    "x": "MULTIPLIER",
    "times": "MULTIPLIER",
    "veces": "MULTIPLIER",
}

_CURRENCY_UNITS = frozenset({"USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "MXN"})

_MONTHS_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
    "enero": 1,
    "ene": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "abr": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "ago": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
    "dic": 12,
}

_ISO_DATE_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_MDY_DATE_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_DMY_DATE_RE = re.compile(
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+|de\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\.?"
    r"(?:,?\s+(?:de\s+|del\s+)?(?P<year>\d{4}))\b",
    re.IGNORECASE,
)
_SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")

# Structural identifiers that must NOT be parsed as numeric quantities
_STRUCTURAL_ID_RE = re.compile(
    r"\b(?:Section|Sec\.|Article|Art\.|Clause|Cláusula|Clausula|Artículo|Articulo|"
    r"Paragraph|Page|p\.|pp\.)\s+"
    r"\d+(?:[.\-]\d+)*\b"
    r"|\b(?:Class|Tranche|Series|Tier)\s+[A-Z0-9]+(?:\s+[A-Z0-9]+)?(?:\s+Notes?)?\b"
    r"|\b\d+\s*[MWYD]\s+(?:EURIBOR|LIBOR|SOFR|SONIA|TIIE|ESTR)\b",
    re.IGNORECASE,
)

_UNIT_PATTERN = (
    r"(?:business\s+days?|working\s+days?|d[ií]as?\s+h[aá]biles|"
    r"calendar\s+days?|d[ií]as?\s+naturales|"
    r"per\s+cent\.?|por\s+ciento|percent\b|pct\b|%|"
    r"basis\s+points?\b|puntos?\s+b[aá]sicos?\b|bps\b|"
    r"usd\b|dollars?\b|d[oó]lares?\b|eur\b|euros?\b|gbp\b|pounds?\b|libras?\b|"
    r"chf\b|jpy\b|cad\b|aud\b|mxn\b|"
    r"days?\b|d[ií]as?\b|weeks?\b|semanas?\b|months?\b|mes(?:es)?\b|"
    r"years?\b|a[ñn]os?\b|hours?\b|horas?\b|hrs?\b|hr\b|times\b|veces\b|x\b)"
)

_RANGE_QUANTITY_RE = re.compile(
    r"(?:between\s+|from\s+|entre\s+)?"
    r"(?P<n1>\d+(?:[.,]\d+)?)\s*(?:and|to|or|y|a|\-|–)\s*(?P<n2>\d+(?:[.,]\d+)?)\s+"
    r"(?P<unit>" + _UNIT_PATTERN + r")",
    re.IGNORECASE,
)

_FRACTION_QUANTITY_RE = re.compile(
    r"(?<![\w/])(?:(?P<whole>\d+)\s+)?(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*"
    r"(?P<unit>%|per\s+cent\.?|por\s+ciento|percent\b|pct\b|bps\b|days?\b|months?\b|years?\b)",
    re.IGNORECASE,
)

_NUMERIC_QUANTITY_RE = re.compile(
    r"(?<![a-zA-Z0-9_])"
    r"(?P<sign>[-−])?\s*"
    r"\(?(?:(?P<iso_prefix>\b(?:USD|EUR|GBP|CHF|JPY|CAD|AUD|MXN)\b)\s+|(?P<sym_prefix>US\$|\$|€|£)\s*)?"
    r"(?P<num>"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d+,\d+"
    r"|\d+(?:\.\d+)?"
    r")\)?"
    r"(?P<scale_space>\s*)(?P<scale>mil\s+millones|billions?|bn|millions?|millones|mill[oó]n|thousands?|miles|mil|mm|k|m|b)?\b\)?"
    r"(?:\s*(?:\([a-zA-Z0-9]+\)\s*)?(?:(?:de|of)\s+)?(?P<suffix>" + _UNIT_PATTERN + r"))?",
    re.IGNORECASE,
)

_TABLE_SCALE_HEADER_RE = re.compile(
    r"\b(?:in\s+(?P<en>thousands|millions|billions)|en\s+(?P<es>miles|millones)|"
    r"\$\s*in\s+(?P<dlr>thousands|millions|billions))\b",
    re.IGNORECASE,
)

_NEGATION_PREFIX_RE = re.compile(
    r"\b(?:in\s+no\s+event|under\s+no\s+circumstances|shall\s+not|may\s+not|must\s+not|"
    r"cannot|can\s+not|never|not|no|neither|nor|without|unless|except|subject\s+to|"
    r"provided\s+that|other\s+than|prohibited|forbid(?:den)?|"
    r"en\s+ning[uú]n\s+caso|no\s+podr[aá]|nunca|jam[aá]s|sin|salvo|excepto)\b",
    re.IGNORECASE,
)

_POSITIVE_ASSERTION_WORDS = frozenset(
    {
        "permitted",
        "allowed",
        "authorized",
        "approved",
        "yes",
        "true",
        "valid",
        "permitido",
        "autorizado",
    }
)

_VACUOUS_VALUES = frozenset(
    {
        "reviewed",
        "checked",
        "verified",
        "addressed",
        "completed",
        "done",
        "ok",
        "n/a",
        "na",
        "none",
        "see text",
        "see above",
        "revisado",
        "comprobado",
    }
)

_VALUE_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "be",
        "shall",
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "en",
        "por",
        "para",
        "con",
        "y",
        "o",
    }
)

_BOILERPLATE_STR_PREFIX_RE = re.compile(
    r"^(?:governed\s+by\s+)?(?:the\s+)?(?:laws\s+of\s+)?(?:the\s+)?"
    r"(?:state\s+of\s+|commonwealth\s+of\s+|las\s+leyes\s+del\s+estado\s+de\s+|el\s+estado\s+de\s+)?",
    re.IGNORECASE,
)


def normalize_span(text: str) -> str:
    """Normalize punctuation, ligatures, soft hyphens, and line-break hyphens deterministically."""
    if not text:
        return ""
    dehyphenated = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", text)
    cleaned = _ZERO_WIDTH_RE.sub("", dehyphenated)
    nfkc = unicodedata.normalize("NFKC", cleaned)
    translated = nfkc.translate(_QUOTE_MAP)
    return re.sub(r"\s+", " ", translated).strip()


def canonicalize_unit(raw_unit: str | None) -> str:
    if not raw_unit:
        return "NONE"
    cleaned = re.sub(r"\s+", " ", raw_unit.strip().lower())
    return _UNIT_ALIASES.get(cleaned, _UNIT_ALIASES.get(cleaned.rstrip("."), cleaned.upper()))


def _to_commensurable_base(magnitude: Decimal, unit: str) -> tuple[Decimal, str]:
    """Convert commensurable financial/duration units to a canonical base unit for comparison."""
    if unit == "BPS":
        return (magnitude / Decimal("100")), "PERCENT"
    if unit == "WEEKS":
        return (magnitude * Decimal("7")), "DAYS"
    return magnitude, unit


@dataclass(frozen=True)
class ParsedQuantity:
    magnitude: Decimal
    unit: str  # "NONE" if dimensionless
    is_fraction: bool = False
    span: tuple[int, int] = (0, 0)

    def matches(self, other: ParsedQuantity) -> bool:
        # Strict unit check: NONE only matches NONE (A11)
        if (self.unit == "NONE") != (other.unit == "NONE"):
            return False

        m1, u1 = _to_commensurable_base(self.magnitude, self.unit)
        m2, u2 = _to_commensurable_base(other.magnitude, other.unit)
        if u1 != u2:
            return False

        if m1 == m2:
            return True

        # Vulgar fraction comparison (e.g. 66 2/3% vs 66.67%, B8)
        if self.is_fraction != other.is_fraction:
            frac_mag = m1 if self.is_fraction else m2
            dec_mag = m2 if self.is_fraction else m1
            exp = dec_mag.as_tuple().exponent
            if isinstance(exp, int) and exp <= -2:
                quant = Decimal(10) ** exp
                if frac_mag.quantize(quant) == dec_mag:
                    return True
        return False

    def same_magnitude_wrong_unit(self, other: ParsedQuantity) -> bool:
        if self.magnitude == other.magnitude and self.unit != other.unit:
            return True
        m1, u1 = _to_commensurable_base(self.magnitude, self.unit)
        m2, u2 = _to_commensurable_base(other.magnitude, other.unit)
        return m1 == m2 and u1 != u2


def _parse_decimal_token(raw_num: str) -> Decimal:
    if "," in raw_num and "." in raw_num:
        if raw_num.rfind(",") > raw_num.rfind("."):
            # European format: 1.250.000,50
            return Decimal(raw_num.replace(".", "").replace(",", "."))
        # US/UK format: 1,250,000.50
        return Decimal(raw_num.replace(",", ""))
    if "," in raw_num:
        parts = raw_num.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            return Decimal(raw_num.replace(",", ""))
        # European decimal comma: 5,5 or 12,75
        return Decimal(raw_num.replace(",", "."))
    if "." in raw_num:
        parts = raw_num.split(".")
        # Multiple dots (1.250.000) -> European thousands separator
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            return Decimal(raw_num.replace(".", ""))
    return Decimal(raw_num)


def parse_iso_date(raw: str | None) -> str | None:
    """Validate and normalize a date string to ISO YYYY-MM-DD; return None if invalid."""
    if not raw:
        return None
    cleaned = normalize_span(raw)
    if _SLASH_DATE_RE.fullmatch(cleaned):
        return None
    dates = extract_dates(cleaned)
    if len(dates) == 1:
        return dates[0]
    return None


def extract_dates(text: str) -> list[str]:
    """Extract ISO YYYY-MM-DD and natural-language dates (EN/ES) normalized to YYYY-MM-DD."""
    if not text:
        return []
    norm = normalize_span(text)
    found: list[tuple[int, str]] = []

    for m in _ISO_DATE_RE.finditer(norm):
        try:
            dt = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            found.append((m.start(), dt.isoformat()))
        except ValueError:
            continue

    for m in _MDY_DATE_RE.finditer(norm):
        mon = _MONTHS_MAP.get(m.group("month").lower().rstrip("."))
        if mon:
            try:
                dt = date(int(m.group("year")), mon, int(m.group("day")))
                found.append((m.start(), dt.isoformat()))
            except ValueError:
                continue

    for m in _DMY_DATE_RE.finditer(norm):
        mon = _MONTHS_MAP.get(m.group("month").lower().rstrip("."))
        if mon:
            try:
                dt = date(int(m.group("year")), mon, int(m.group("day")))
                found.append((m.start(), dt.isoformat()))
            except ValueError:
                continue

    found.sort(key=lambda item: item[0])
    deduped: list[str] = []
    for _, iso in found:
        if iso not in deduped:
            deduped.append(iso)
    return deduped


def _parse_word_number_phrase(phrase: str) -> int | None:
    """Parse compound English or Spanish number words ('twenty-five', 'one hundred twenty', 'ciento veinte')."""
    tokens = [
        t
        for t in re.split(r"[\s\-]+", phrase.strip().lower())
        if t and t not in {"and", "y"}
    ]
    if not tokens:
        return None

    total = 0
    current = 0
    matched_any = False

    for tok in tokens:
        if tok in _ONES_EN:
            current += _ONES_EN[tok]
            matched_any = True
        elif tok in _TENS_EN:
            current += _TENS_EN[tok]
            matched_any = True
        elif tok == "hundred":
            current = (current or 1) * 100
            matched_any = True
        elif tok in _ONES_ES:
            current += _ONES_ES[tok]
            matched_any = True
        elif tok in _TENS_ES:
            current += _TENS_ES[tok]
            matched_any = True
        elif tok in _HUNDREDS_ES:
            current += _HUNDREDS_ES[tok]
            matched_any = True
        elif tok in {"thousand", "mil"}:
            total += (current or 1) * 1000
            current = 0
            matched_any = True
        else:
            return None

    return (total + current) if matched_any else None


_ALL_NUMBER_WORDS = sorted(
    set(_ONES_EN)
    | set(_TENS_EN)
    | set(_ONES_ES)
    | set(_TENS_ES)
    | set(_HUNDREDS_ES)
    | {"hundred", "thousand", "mil", "and", "y"},
    key=len,
    reverse=True,
)

_COMPOUND_WORD_QTY_RE = re.compile(
    r"(?<![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\-])"
    r"(?P<phrase>(?:" + "|".join(_ALL_NUMBER_WORDS) + r")(?:[\s\-]+(?:" + "|".join(_ALL_NUMBER_WORDS) + r"))*)"
    r"(?:\s*\((?P<paren_num>\d+)\))?"
    r"\s+(?P<unit>" + _UNIT_PATTERN + r")",
    re.IGNORECASE,
)


def _detect_table_scale(text: str) -> Decimal | None:
    m = _TABLE_SCALE_HEADER_RE.search(text or "")
    if not m:
        return None
    token = (m.group("en") or m.group("es") or m.group("dlr") or "").lower()
    if token in {"thousands", "miles"}:
        return Decimal("1000")
    if token in {"millions", "millones"}:
        return Decimal("1000000")
    if token == "billions":
        return Decimal("1000000000")
    return None


def _mask_preserving_len(match: re.Match[str]) -> str:
    return " " * (match.end() - match.start())


def extract_quantities(
    text: str,
    explicit_unit: str | None = None,
    *,
    apply_table_scale: bool = False,
) -> list[ParsedQuantity]:
    """Extract normalized (Decimal magnitude, unit) quantities from text."""
    if not text:
        return []

    norm = normalize_span(text)
    table_scale = _detect_table_scale(norm) if apply_table_scale else None

    # Mask ISO dates, natural dates, and structural identifiers while preserving character offsets
    scrubbed = _ISO_DATE_RE.sub(_mask_preserving_len, norm)
    scrubbed = _MDY_DATE_RE.sub(_mask_preserving_len, scrubbed)
    scrubbed = _DMY_DATE_RE.sub(_mask_preserving_len, scrubbed)
    scrubbed = _STRUCTURAL_ID_RE.sub(_mask_preserving_len, scrubbed)

    default_unit = canonicalize_unit(explicit_unit)
    results: list[ParsedQuantity] = []
    occupied_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in occupied_spans)

    # 1. Vulgar fractions (e.g. "66 2/3%")
    for m in _FRACTION_QUANTITY_RE.finditer(scrubbed):
        whole = Decimal(m.group("whole") or "0")
        num = Decimal(m.group("num"))
        den = Decimal(m.group("den"))
        if den == 0:
            continue
        mag = whole + (num / den)
        unit = canonicalize_unit(m.group("unit"))
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=mag, unit=unit, is_fraction=True, span=span))

    # 2. Range expressions ("between 30 and 60 days", "30-60 days")
    for m in _RANGE_QUANTITY_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        u = canonicalize_unit(m.group("unit"))
        d1 = _parse_decimal_token(m.group("n1"))
        d2 = _parse_decimal_token(m.group("n2"))
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=d1, unit=u, span=span))
        results.append(ParsedQuantity(magnitude=d2, unit=u, span=span))

    # 3. Compound word quantities ("twenty-five (25) days", "one hundred twenty (120) days")
    for m in _COMPOUND_WORD_QTY_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        word_val = _parse_word_number_phrase(m.group("phrase"))
        if word_val is None:
            continue
        unit = canonicalize_unit(m.group("unit"))
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=Decimal(word_val), unit=unit, span=span))

    # 4. Numeric quantities
    for m in _NUMERIC_QUANTITY_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        raw_num = m.group("num")
        sign = m.group("sign")
        iso_prefix = m.group("iso_prefix")
        sym_prefix = m.group("sym_prefix")
        scale_space = m.group("scale_space") or ""
        scale = m.group("scale")
        suffix = m.group("suffix")

        try:
            magnitude = _parse_decimal_token(raw_num)
        except InvalidOperation:
            continue

        if sign:
            magnitude = -magnitude

        unit = "NONE"
        if iso_prefix:
            unit = canonicalize_unit(iso_prefix)
        elif sym_prefix:
            unit = canonicalize_unit(sym_prefix)
        if suffix:
            suffix_unit = canonicalize_unit(suffix)
            if unit == "NONE":
                unit = suffix_unit
        if unit == "NONE" and default_unit != "NONE":
            unit = default_unit

        if scale:
            scale_key = re.sub(r"\s+", " ", scale.strip().lower())
            # Single-letter scales ('k', 'm', 'b') require either currency context or no space ("$500K", "50m USD", "500k")
            if scale_key in {"k", "m", "b"}:
                has_currency = unit in _CURRENCY_UNITS
                attached = len(scale_space) == 0
                if has_currency or attached:
                    magnitude *= _SCALE_FACTORS[scale_key]
            else:
                magnitude *= _SCALE_FACTORS.get(scale_key, Decimal("1"))
        elif table_scale is not None and unit in _CURRENCY_UNITS and abs(magnitude) < Decimal("100000"):
            magnitude *= table_scale

        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=magnitude.normalize(), unit=unit, span=span))

    return results


def canonicalize_value(value: str, explicit_unit: str | None = None) -> str:
    """Produce a deterministic canonical key for value comparison across sources."""
    iso_date = parse_iso_date(value)
    if iso_date is not None:
        return f"DATE:{iso_date}"

    quantities = extract_quantities(value, explicit_unit=explicit_unit)
    if len(quantities) == 1:
        q = quantities[0]
        base_mag, base_unit = _to_commensurable_base(q.magnitude, q.unit)
        norm_mag = base_mag.normalize()
        mag_str = format(norm_mag, "f")
        if "." in mag_str:
            mag_str = mag_str.rstrip("0").rstrip(".")
        return f"QTY:{mag_str}:{base_unit}"

    cleaned = normalize_span(value).lower().rstrip(".")
    cleaned = _BOILERPLATE_STR_PREFIX_RE.sub("", cleaned).strip()
    return f"STR:{cleaned}"


def _has_truncated_sentence_negation(norm_source: str, norm_quote: str) -> str | None:
    """Return the truncated leading qualifier if *norm_quote* drops a negation from its sentence in *norm_source*."""
    idx = norm_source.lower().find(norm_quote.lower())
    if idx <= 0:
        return None
    # Locate start of enclosing sentence
    prefix = norm_source[:idx]
    sent_start = max(prefix.rfind("."), prefix.rfind(";"), prefix.rfind("\n"))
    leading_clause = prefix[sent_start + 1 :].strip()
    if not leading_clause:
        return None
    m = _NEGATION_PREFIX_RE.search(leading_clause)
    if m:
        return leading_clause
    return None


def _check_multi_quantity_subject_binding(
    assertion: Assertion,
    vq: ParsedQuantity,
    same_unit_quote_quantities: list[ParsedQuantity],
) -> bool:
    """When a quote contains multiple distinct quantities of the same unit (e.g. Class A 5.10%, Class C 6.75%),
    verify that *vq* is not bound to a competing entity when *assertion.subject* appears in the quote."""
    distinct_mags = {qq.magnitude for qq in same_unit_quote_quantities}
    if len(distinct_mags) <= 1:
        return True

    norm_quote = normalize_span(assertion.quote).lower()
    subj_tokens = [
        t
        for t in normalize_entity_key(assertion.subject).split("_")
        if t and t not in {"note", "deal", "waterfall", "tranche", "class", "series"}
    ]
    # Also retain specific designators like "class a", "class b", "class c", "tranche a", "step 13"
    raw_subj_norm = re.sub(r"[_\-]+", " ", assertion.subject.strip().lower())
    designator_match = re.search(
        r"\b((?:class|tranche|series|tier|step)\s+[a-z0-9]+)\b", raw_subj_norm
    )
    target_phrase = designator_match.group(1) if designator_match else None
    if not target_phrase and len(subj_tokens) == 1 and len(subj_tokens[0]) == 1:
        # e.g. subject="Class C" -> subj_tokens=["c"]
        target_phrase = f"class {subj_tokens[0]}"

    subj_pos = -1
    if target_phrase and target_phrase in norm_quote:
        subj_pos = norm_quote.find(target_phrase)
    else:
        for tok in subj_tokens:
            if len(tok) >= 2 and tok in norm_quote:
                subj_pos = norm_quote.find(tok)
                break

    if subj_pos < 0:
        return True

    # Find the quantity of this unit that appears in the same sub-clause or closest AFTER subj_pos
    after_quantities = [qq for qq in same_unit_quote_quantities if qq.span[0] >= subj_pos]
    if after_quantities:
        nearest = min(after_quantities, key=lambda qq: qq.span[0] - subj_pos)
        return vq.matches(nearest)

    nearest_any = min(
        same_unit_quote_quantities, key=lambda qq: abs(qq.span[0] - subj_pos)
    )
    return vq.matches(nearest_any)


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", normalize_span(text).lower())
    return {w for w in words if len(w) >= 2 and w not in _VALUE_STOPWORDS}


def verify_quote_and_quantities(
    assertion: Assertion,
    source: Source | None,
    policy: Policy,
) -> list[Issue]:
    """Verify that the source exists, the quote is grounded in context, and all numbers/dates/units/values match."""
    issues: list[Issue] = []

    if source is None:
        issues.append(
            Issue(
                code="SOURCE_NOT_FOUND",
                disposition="RETRY",
                assertion_id=assertion.id,
                subject=assertion.subject,
                predicate=assertion.predicate,
                message=f"Assertion '{assertion.id}' cites unknown source_id '{assertion.source_id}'.",
                remediation=(
                    f"Cite a valid source_id registered in the ledger instead of '{assertion.source_id}'."
                ),
            )
        )
        return issues

    norm_quote = normalize_span(assertion.quote)
    norm_source = normalize_span(source.text)

    quote_in_source = bool(norm_quote) and (
        norm_quote in norm_source or norm_quote.lower() in norm_source.lower()
    )

    if policy.require_exact_quote:
        if not norm_quote:
            issues.append(
                Issue(
                    code="EMPTY_QUOTE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=f"Assertion '{assertion.id}' provided an empty citation quote.",
                    remediation=f"Provide an exact verbatim quote from source '{source.id}'.",
                )
            )
            return issues

        if not quote_in_source:
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote '{assertion.quote[:80]}' does not appear in source '{source.id}'."
                    ),
                    remediation=(
                        f"Extract an exact verbatim substring from '{source.id}' or remove ungrounded claim."
                    ),
                )
            )
            return issues

        # Reject bare single-token / out-of-context quotes (A1, C4) when source is longer
        quote_words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+", norm_quote)
        if len(norm_source) > len(norm_quote) and (len(norm_quote) < 6 or len(quote_words) < 2):
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote '{assertion.quote}' is too short or lacks clause context from '{source.id}'."
                    ),
                    remediation=(
                        f"Cite the complete clause or sentence from '{source.id}' supporting '{assertion.predicate}'."
                    ),
                )
            )
            return issues

        # Reject quotes that truncate a leading negation or exception qualifier from the sentence (A4)
        truncated_prefix = _has_truncated_sentence_negation(norm_source, norm_quote)
        if truncated_prefix is not None:
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote truncates leading negation/qualifier '{truncated_prefix}' from sentence in '{source.id}'."
                    ),
                    remediation="Include the full sentence with its leading negation or condition.",
                )
            )
            return issues

    # Reference text for quantity/date verification:
    # NEVER trust an unverified quote when require_exact_quote=False!
    reference_text = assertion.quote if quote_in_source else source.text

    # Reject vacuous placeholder values (C4)
    norm_val_low = normalize_span(assertion.value).lower()
    if norm_val_low in _VACUOUS_VALUES and norm_val_low not in norm_quote.lower():
        issues.append(
            Issue(
                code="QUANTITY_MISMATCH",
                disposition="RETRY",
                assertion_id=assertion.id,
                subject=assertion.subject,
                predicate=assertion.predicate,
                message=(
                    f"Proposed value '{assertion.value}' for '{assertion.predicate}' is a vacuous placeholder not grounded in quote."
                ),
                remediation=f"State the specific governing value or obligation from '{source.id}'.",
            )
        )
        return issues

    if policy.require_quantity_match:
        # 1. Check dates in value strictly against the verified quote (NO fallback to full source! A3)
        if _SLASH_DATE_RE.search(assertion.value or ""):
            issues.append(
                Issue(
                    code="DATE_MISMATCH",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=f"Ambiguous slash date in '{assertion.value}'; use ISO YYYY-MM-DD.",
                    remediation="Format dates as ISO YYYY-MM-DD.",
                )
            )
            return issues

        val_dates = extract_dates(assertion.value)
        quote_dates = set(extract_dates(reference_text))
        for d in val_dates:
            if d not in quote_dates:
                issues.append(
                    Issue(
                        code="DATE_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Date '{d}' in proposed value '{assertion.value}' is absent from cited quote."
                        ),
                        remediation=f"Align date in '{assertion.predicate}' with the exact date in the cited quote from '{source.id}'.",
                    )
                )

        # 2. Check numeric quantities & units strictly against the verified quote (NO fallback to full source! A2)
        val_quantities = extract_quantities(assertion.value, explicit_unit=assertion.unit)
        table_scale = _detect_table_scale(reference_text) or _detect_table_scale(source.text)
        quote_quantities = extract_quantities(
            reference_text, apply_table_scale=(table_scale is not None)
        )

        for vq in val_quantities:
            matching_q = [
                qq
                for qq in quote_quantities
                if vq.matches(qq) and not vq.same_magnitude_wrong_unit(qq)
            ]
            if matching_q:
                # Check multi-quantity cross-entity attribution (A8)
                _, base_u = _to_commensurable_base(vq.magnitude, vq.unit)
                same_unit_in_quote = [
                    qq
                    for qq in quote_quantities
                    if _to_commensurable_base(qq.magnitude, qq.unit)[1] == base_u
                ]
                if not _check_multi_quantity_subject_binding(assertion, vq, same_unit_in_quote):
                    issues.append(
                        Issue(
                            code="QUANTITY_MISMATCH",
                            disposition="RETRY",
                            assertion_id=assertion.id,
                            subject=assertion.subject,
                            predicate=assertion.predicate,
                            message=(
                                f"Quantity {vq.magnitude} ({vq.unit}) in '{assertion.value}' belongs to a different "
                                f"entity/clause in the cited quote than '{assertion.subject}'."
                            ),
                            remediation=(
                                f"Use the quantity specifically bound to '{assertion.subject}' in '{source.id}'."
                            ),
                        )
                    )
                continue

            unit_clash = next(
                (qq for qq in quote_quantities if vq.same_magnitude_wrong_unit(qq)), None
            )
            if unit_clash is not None:
                issues.append(
                    Issue(
                        code="UNIT_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Unit mismatch for '{assertion.predicate}': proposed {vq.magnitude} {vq.unit}, "
                            f"but cited quote states {unit_clash.magnitude} {unit_clash.unit}."
                        ),
                        remediation=(
                            f"Correct the unit of '{assertion.predicate}' to '{unit_clash.unit}' as stated in '{source.id}'."
                        ),
                    )
                )
            else:
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Quantity {vq.magnitude} ({vq.unit}) in proposed value '{assertion.value}' "
                            f"is not grounded in cited quote '{assertion.quote[:80]}'."
                        ),
                        remediation=(
                            f"Use the exact figure supported by '{source.id}' without rounding or hallucinating."
                        ),
                    )
                )

        # 3. For non-numeric, non-date categorical claims, verify token grounding and polarity (A4, C4)
        if not val_dates and not val_quantities and quote_in_source:
            val_tokens = _content_tokens(assertion.value)
            quote_tokens = _content_tokens(reference_text)
            if val_tokens and not (val_tokens & quote_tokens):
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Proposed value '{assertion.value}' is not grounded in cited quote '{assertion.quote[:80]}'."
                        ),
                        remediation=f"Ensure '{assertion.predicate}' value reflects terms present in the cited quote.",
                    )
                )
            elif (
                norm_val_low in _POSITIVE_ASSERTION_WORDS
                and _NEGATION_PREFIX_RE.search(reference_text) is not None
            ):
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Proposed positive value '{assertion.value}' contradicts negation in cited quote '{assertion.quote[:80]}'."
                        ),
                        remediation="Reflect the negative/prohibitive condition stated in the quote.",
                    )
                )

    return issues
