"""Verbatim quote grounding, quantity/scale/unit verification, and date/polarity checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from claimlayer.models import Assertion, Issue, Policy, Source, normalize_entity_key

_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2009": " ",
    }
)

_ZERO_WIDTH_RE = re.compile(r"[\u00ad\u200b\u200c\u200d\ufeff]")
_LINEBREAK_HYPHEN_RE = re.compile(
    r"([a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])-\s*\n\s*([a-zA-ZáéíóúñüÁÉÍÓÚÑÜ])"
)

# English and Spanish compound number words
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
    "percent_per_annum": "PERCENT",
    "percent per annum": "PERCENT",
    "% per annum": "PERCENT",
    "% p.a.": "PERCENT",
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
_DURATION_UNITS = frozenset({"BUSINESS_DAYS", "CALENDAR_DAYS", "DAYS", "WEEKS", "MONTHS", "YEARS", "HOURS"})
_RATIO_UNITS = frozenset({"PERCENT", "BPS", "MULTIPLIER"})

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
    r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+|d[ií]a\s+de\s+|of\s+|de\s+)?"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\.?"
    r"(?:,?\s+(?:de\s+|del\s+)?(?P<year>\d{4}))\b",
    re.IGNORECASE,
)
_SLASH_DATE_RE = re.compile(r"\b(?P<p1>\d{1,2})/(?P<p2>\d{1,2})/(?P<y>\d{4})\b")
_DOT_DATE_RE = re.compile(r"\b(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})\b")
_SHORT_SLASH_DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{2}\b")

# Structural document identifiers that must NOT be parsed as numeric quantities:
# - Section / Article / Clause / Schedule / Exhibit / Annex / Part / Rule / Paragraph / Page + number
# - Class / Tranche / Series / Tier + designator (e.g. "Class A", "Class 2 B Notes", without eating following rates/amounts!)
# - Floating-rate tenors ("3M EURIBOR", "6M Term SOFR")
_STRUCTURAL_ID_RE = re.compile(
    r"\b(?:Section|Sec\.|Article|Art\.|Clause|Cláusula|Clausula|Artículo|Articulo|"
    r"Schedule|Exhibit|Annex|Anexo|Appendix|Apéndice|Apendice|Part|Rule|Paragraph|Page|p\.|pp\.|Footnote)\s+"
    r"\d+(?:[.\-]\d+)*[A-Za-z]?\b"
    r"|\b(?:Class|Tranche|Series|Tier)\s+(?:[A-Za-z](?:-\d+)?|\d+(?:\s+[A-Za-z]\b)?)(?![.,\d%])(?:\s+Notes?\b)?"
    r"|\b\d+\s*[MWYD]\s+(?:Term\s+)?(?:EURIBOR|LIBOR|SOFR|SONIA|TIIE|ESTR)\b"
    r"|\b(?:30E?|Actual|Act)\s*/\s*(?:360|365|Actual|Act)\b",
    re.IGNORECASE,
)

_UNIT_PATTERN = (
    r"(?:business[\s-]+days?|working[\s-]+days?|d[ií]as?\s+h[aá]biles|"
    r"calendar[\s-]+days?|d[ií]as?\s+naturales|"
    r"per\s+cent\.?|por\s+ciento|percent\b|pct\b|%|"
    r"basis\s+points?\b|puntos?\s+b[aá]sicos?\b|bps\b|"
    r"us\$|\$|€|£|"
    r"usd\b|dollars?\b|d[oó]lares?\b|eur\b|euros?\b|gbp\b|pounds?\b|libras?\b|"
    r"chf\b|jpy\b|cad\b|aud\b|mxn\b|"
    r"days?\b|d[ií]as?\b|weeks?\b|semanas?\b|months?\b|mes(?:es)?\b|"
    r"years?\b|a[ñn]os?\b|hours?\b|horas?\b|hrs?\b|hr\b|times\b|veces\b|x\b)"
)

# Range expressions: either prefixed by "between/from/entre" with "and/to/y/a", or joined by "or/-/–" without currency prefix on n1
_RANGE_QUANTITY_RE = re.compile(
    r"(?<![$€£\w])"
    r"(?:"
    r"(?:between|from|entre)\s+(?P<n1a>\d+(?:[.,]\d+)?)\s+(?:and|to|or|y|a)\s+(?P<n2a>\d+(?:[.,]\d+)?)"
    r"|(?P<n1b>\d+(?:[.,]\d+)?)\s*(?:or|\-|–)\s*(?P<n2b>\d+(?:[.,]\d+)?)"
    r")"
    r"(?:\s+|\s*-\s*)(?P<unit>" + _UNIT_PATTERN + r")",
    re.IGNORECASE,
)

_FRACTION_QUANTITY_RE = re.compile(
    r"(?<![\w/])(?:(?P<whole>\d+)\s+)?(?P<num>\d+)\s*/\s*(?P<den>\d+)\s*"
    r"(?P<unit>%|per\s+cent\.?|por\s+ciento|percent\b|pct\b|bps\b|days?\b|months?\b|years?\b)",
    re.IGNORECASE,
)

# Financial ratios: "1.25 to 1.00", "1.25 to 1", "1.25:1.00"
_RATIO_COVENANT_RE = re.compile(
    r"\b(?P<num>\d+(?:[.,]\d+)?)\s*(?:to|a|:)\s*1(?:[.,]0+)?\b",
    re.IGNORECASE,
)

_NUMERIC_QUANTITY_RE = re.compile(
    r"(?<![a-zA-Z0-9_])"
    r"(?P<sign>[-−])?(?=US\$|\$|\€|\£|\(|(?:USD|EUR|GBP|CHF|JPY|CAD|AUD|MXN)|\d)"
    r"\(?(?:(?P<iso_prefix>\b(?:USD|EUR|GBP|CHF|JPY|CAD|AUD|MXN))\s*|(?P<sym_prefix>US\$|\$|€|£)\s*)?"
    r"(?P<num>"
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d+,\d+"
    r"|\d+(?:\.\d+)?"
    r")\)?"
    r"(?:(?P<scale_space>\s*)(?P<scale>mil\s+millones|billions?|bn|millions?|millones|mill[oó]n|thousands?|miles|mil|mm|k|m|b)\b\)?)?"
    r"(?:(?:\s*-\s*|\s*)(?:\([a-zA-Z0-9]+\)(?:\s*-\s*|\s*))?(?:(?:de|of)\s+)?(?P<suffix>" + _UNIT_PATTERN + r"))?",
    re.IGNORECASE,
)

# Scale declarations are evaluated in their source block, not a character window.
_TABLE_SCALE_HEADER_RE = re.compile(
    r"\b(?:in|en)\s+(?P<scale>thousands|millions|billions|miles|millones)\b"
    r"(?:\s+(?:of|de)\s+(?:U\.S\.\s+)?(?:dollars?|euros?|pounds?))?"
    r"|\(\s*[$€£](?P<zeros>0{3,9})s?\s*\)", re.IGNORECASE)

_SPANISH_CONTEXT_RE = re.compile(
    r"(?:\b(?:el|la|los|las|del|al|es|será|sera|por\s+ciento|euros?|d[ií]as?|a[ñn]os?|"
    r"meses|cl[áa]usula|primera|segunda|tercera|cuarta|precio|importe|comisi[óo]n|inter[ée]s|pr[ée]stamo)\b|€)",
    re.IGNORECASE,
)

_CLAUSE_NEGATION_RE = re.compile(
    r"\b(?:in\s+no\s+event|under\s+no\s+circumstances|shall\s+not|may\s+not|must\s+not|"
    r"cannot|can\s+not|will\s+not|is\s+not|are\s+not|never|not|neither|nor|none|"
    r"prohibited|forbid(?:den)?|unavailable|impermissible|"
    r"en\s+ning[uú]n\s+caso|no\s+podr[aá]|no\s+se\s+podr[aá]|nunca|jam[aá]s|prohibido)\b",
    re.IGNORECASE,
)

_AFFIRMATIVE_PROPOSITION_RE = re.compile(
    r"\b(?:is|are|was|were|be|may|can|shall|will|must|has|have|"
    r"allow(?:ed|s)?|permit(?:ted|s)?|available|eligible|entitled|valid|authorized|authorised|yes|true|"
    r"podr[aá]|puede|pueden|es|son|ser[aá]|est[aá]|permitid[oa]s?|autorizad[oa]s?|disponibles?)\b",
    re.IGNORECASE,
)

_TRUNCATED_QUALIFIER_RE = re.compile(
    r"\b(?:in\s+no\s+event|under\s+no\s+circumstances|shall\s+not|may\s+not|must\s+not|"
    r"cannot|can\s+not|never|not|no|neither|nor|without|unless|except|subject\s+to|"
    r"provided\s+that|other\s+than|prohibited|forbid(?:den)?|"
    r"en\s+ning[uú]n\s+caso|no\s+podr[aá]|nunca|jam[aá]s|sin|salvo|excepto)\b",
    re.IGNORECASE,
)

_TRAILING_EXCEPTION_RE = re.compile(
    r"(?:^|[,;]\s*|\s+)\b(?:except|unless|subject\s+to|provided\s+that|other\s+than|"
    r"save\s+that|but\s+not|salvo|excepto|a\s+menos\s+que|sin\s+perjuicio)\b",
    re.IGNORECASE,
)

_SECTION_HEADER_ONLY_RE = re.compile(
    r"^\s*(?:section|sec\.?|article|art\.?|clause|cl[áa]usula|art[íi]culo|part|item|schedule|exhibit|annex)\s+"
    r"[a-z0-9\-]+(?:\.[a-z0-9\-]+)*(?:\([a-z0-9]+\))*(?:\s*(?::|—|\.\s+|\s+[-–]\s+)\s*[a-z0-9\s,&()\-]{1,60})?\.?\s*$",
    re.IGNORECASE,
)

_ABBREV_BEFORE_DOT_RE = re.compile(
    r"\b(?:sec|art|no|inc|corp|ltd|llc|co|plc|v|vs|cf|al|jr|sr|dr|mr|mrs|ms|st|p|pp|fig|eq|ch|vol)\.$",
    re.IGNORECASE,
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
        "been",
        "being",
        "was",
        "were",
        "shall",
        "will",
        "would",
        "should",
        "that",
        "this",
        "these",
        "those",
        "from",
        "as",
        "per",
        "annum",
        "governed",
        "laws",
        "law",
        "state",
        "commonwealth",
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "de",
        "del",
        "al",
        "en",
        "por",
        "para",
        "con",
        "y",
        "o",
        "es",
        "son",
        "será",
        "sera",
        "leyes",
        "estado",
        "anual",
    }
)

_BOILERPLATE_STR_PREFIX_RE = re.compile(
    r"^(?:this\s+\w+\s+(?:is|shall\s+be)\s+)?(?:governed\s+by\s+)?(?:the\s+)?(?:laws\s+of\s+)?(?:the\s+)?"
    r"(?:state\s+of\s+|commonwealth\s+of\s+|las\s+leyes\s+del\s+estado\s+de\s+|el\s+estado\s+de\s+|ley\s+aplicable\s*:?\s*)?",
    re.IGNORECASE,
)


def normalize_span(text: str) -> str:
    """Normalize punctuation, ligatures, soft hyphens, and line-break hyphens deterministically."""
    if not text:
        return ""
    dehyphenated = _LINEBREAK_HYPHEN_RE.sub(r"\1\2", text)
    cleaned = _ZERO_WIDTH_RE.sub("", dehyphenated)
    nfkc = unicodedata.normalize("NFKC", cleaned)
    translated = nfkc.translate(_QUOTE_MAP).replace("\\$", "$")
    return re.sub(r"\s+", " ", translated).strip()


def canonicalize_unit(raw_unit: str | None) -> str:
    if not raw_unit:
        return "NONE"
    cleaned = re.sub(r"[\s-]+", " ", raw_unit.strip().lower())
    if cleaned.endswith(" day"):
        cleaned += "s"
    return _UNIT_ALIASES.get(cleaned, _UNIT_ALIASES.get(cleaned.rstrip("."), cleaned.upper()))


def unit_dimension(unit: str) -> str:
    """Return the semantic dimension family of a canonical unit."""
    if unit in _CURRENCY_UNITS:
        return f"CURRENCY:{unit}"
    if unit == "BUSINESS_DAYS":
        return "DURATION:BUSINESS_DAYS"
    if unit == "CALENDAR_DAYS":
        return "DURATION:CALENDAR_DAYS"
    if unit in {"DAYS", "WEEKS"}:
        return "DURATION:DAYS"
    if unit in {"MONTHS", "YEARS"}:
        return "DURATION:MONTHS"
    if unit == "HOURS":
        return "DURATION:HOURS"
    if unit in {"PERCENT", "BPS"}:
        return "RATIO"
    if unit == "MULTIPLIER":
        return "MULTIPLIER"
    return "NONE"


def infer_predicate_dimension(predicate: str) -> str | None:
    """Infer whether a predicate name explicitly requires a specific unit dimension."""
    low = (predicate or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    if "day_count" in low or "daycount" in low or ({"day", "count"} <= tokens):
        return None
    if tokens & {"usd", "dollar", "dollars", "eur", "euro", "euros", "gbp", "currency", "price", "fee", "cost", "salary", "refund", "reimbursement", "settlement"}:
        return "CURRENCY"
    if tokens & {"days", "day", "weeks", "months", "years", "hours", "duration", "deadline", "period"}:
        return "DURATION"
    if tokens & {"rate", "pct", "percent", "percentage", "bps", "coupon", "margin", "yield", "discount"}:
        return "RATIO"
    return None


def _to_commensurable_base(magnitude: Decimal, unit: str) -> tuple[Decimal, str]:
    """Convert commensurable financial/duration units to a canonical base unit for comparison."""
    if unit == "BPS":
        return (magnitude / Decimal("100")), "PERCENT"
    if unit == "YEARS":
        return magnitude * Decimal("12"), "MONTHS"
    if unit == "WEEKS":
        return (magnitude * Decimal("7")), "DAYS"
    return magnitude, unit


@dataclass(frozen=True)
class ParsedQuantity:
    magnitude: Decimal
    unit: str  # "NONE" if dimensionless
    is_fraction: bool = False
    is_range_endpoint: bool = False
    span: tuple[int, int] = (0, 0)

    def matches(self, other: ParsedQuantity) -> bool:
        # Strict unit check: NONE only matches NONE
        if (self.unit == "NONE") != (other.unit == "NONE"):
            return False

        m1, u1 = _to_commensurable_base(self.magnitude, self.unit)
        m2, u2 = _to_commensurable_base(other.magnitude, other.unit)
        if u1 != u2:
            return False

        if m1 == m2:
            return True

        # Vulgar fraction comparison (e.g. 66 2/3% vs 66.67%)
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


def _parse_decimal_token(
    raw_num: str,
    *,
    es_context: bool = False,
    is_percent: bool = False,
) -> Decimal:
    if "," in raw_num and "." in raw_num:
        if raw_num.rfind(",") > raw_num.rfind("."):
            # European format: 1.250.000,50 or 1.250,50
            return Decimal(raw_num.replace(".", "").replace(",", "."))
        # US/UK format: 1,250,000.50
        return Decimal(raw_num.replace(",", ""))
    if "," in raw_num:
        parts = raw_num.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            return Decimal(raw_num.replace(",", ""))
        # European decimal comma: 5,5 or 0,25
        return Decimal(raw_num.replace(",", "."))
    if "." in raw_num:
        parts = raw_num.split(".")
        # Multiple dots (1.250.000) -> European thousands separator
        if len(parts) > 2 and all(len(p) == 3 for p in parts[1:]):
            return Decimal(raw_num.replace(".", ""))
        # Single dot with 3 trailing digits in Spanish/European context ("12.500 euros")
        if (
            es_context
            and not is_percent
            and len(parts) == 2
            and 1 <= len(parts[0]) <= 3
            and len(parts[1]) == 3
        ):
            return Decimal(raw_num.replace(".", ""))
    return Decimal(raw_num)


def parse_iso_date(raw: str | None) -> str | None:
    """Validate and normalize a date string to ISO YYYY-MM-DD; return None if invalid."""
    if not raw:
        return None
    cleaned = normalize_span(raw)
    if _SLASH_DATE_RE.fullmatch(cleaned) or _SHORT_SLASH_DATE_RE.fullmatch(cleaned):
        return None
    dates = extract_dates(cleaned, allow_slash=False)
    if len(dates) == 1:
        return dates[0]
    return None


def extract_dates_with_spans(text: str, *, allow_slash: bool = True) -> list[tuple[tuple[int, int], str]]:
    """Extract (span, ISO YYYY-MM-DD) dates from text (supporting ISO, EN/ES natural language, and DD/MM/YYYY)."""
    if not text:
        return []
    norm = normalize_span(text)
    found: list[tuple[tuple[int, int], str]] = []

    for m in list(_ISO_DATE_RE.finditer(norm)) + list(_DOT_DATE_RE.finditer(norm)):
        try:
            dt = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            found.append(((m.start(), m.end()), dt.isoformat()))
        except ValueError:
            continue

    for m in _MDY_DATE_RE.finditer(norm):
        mon = _MONTHS_MAP.get(m.group("month").lower().rstrip("."))
        if mon:
            try:
                dt = date(int(m.group("year")), mon, int(m.group("day")))
                found.append(((m.start(), m.end()), dt.isoformat()))
            except ValueError:
                continue

    for m in _DMY_DATE_RE.finditer(norm):
        mon = _MONTHS_MAP.get(m.group("month").lower().rstrip("."))
        if mon:
            try:
                dt = date(int(m.group("year")), mon, int(m.group("day")))
                found.append(((m.start(), m.end()), dt.isoformat()))
            except ValueError:
                continue

    if allow_slash:
        for m in _SLASH_DATE_RE.finditer(norm):
            p1, p2, yr = int(m.group("p1")), int(m.group("p2")), int(m.group("y"))
            # DD/MM/YYYY when p1 > 12 or in Spanish/European context
            candidates = [(p2, p1), (p1, p2)] if (p1 > 12 or _SPANISH_CONTEXT_RE.search(norm)) else [(p1, p2), (p2, p1)]
            for mon, day in candidates:
                try:
                    dt = date(yr, mon, day)
                    found.append(((m.start(), m.end()), dt.isoformat()))
                    break
                except ValueError:
                    continue

    found.sort(key=lambda item: item[0][0])
    return found


def extract_dates(text: str, *, allow_slash: bool = True) -> list[str]:
    """Extract ISO YYYY-MM-DD dates from text."""
    deduped: list[str] = []
    for _, iso in extract_dates_with_spans(text, allow_slash=allow_slash):
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
    r"(?:\s+(?P<scale>mil\s+millones|billions?|millions?|millones|mill[oó]n|thousands?|miles|mil))?"
    r"(?:\s+|\s*-\s*)(?:(?:de|of)\s+)?(?P<unit>" + _UNIT_PATTERN + r")",
    re.IGNORECASE,
)


def _extract_scale_value(m: re.Match[str]) -> Decimal | None:
    if m.group("zeros"):
        return Decimal(10) ** len(m.group("zeros"))
    return _SCALE_FACTORS.get(m.group("scale").lower())


def _detect_contextual_table_scale(quote: str, source: str | None = None) -> Decimal | None:
    """Track block-scoped headers; row-local declarations never leak to later lines."""
    source = source or quote
    # Keep line/paragraph boundaries while tolerating whitespace wrapping in a quote.
    normalized_source = "\n".join(normalize_span(line) for line in source.splitlines())
    norm_quote = normalize_span(quote)
    location = re.search(re.escape(norm_quote).replace(r"\ ", r"\s+"), normalized_source, re.IGNORECASE)
    if not location:
        return None
    offset = location.start()
    matches = list(_TABLE_SCALE_HEADER_RE.finditer(normalized_source[:location.end()]))
    if not matches:
        return None
    match = matches[-1]
    between = normalized_source[match.end():offset].lstrip(" .:;)-")
    if re.search(r"[.;](?:\s|$)|\n\s*\n", between):
        return None
    line_end = normalized_source.find("\n", match.end())
    if line_end >= 0 and offset > line_end:
        tail = normalized_source[match.end():line_end]
        if re.search(r"\d", tail):
            return None
    return _extract_scale_value(match)


def _mask_preserving_len(match: re.Match[str]) -> str:
    return " " * (match.end() - match.start())


def extract_quantities(
    text: str,
    explicit_unit: str | None = None,
    *,
    table_scale: Decimal | None = None,
    apply_table_scale: bool = False,
    es_context: bool | None = None,
) -> list[ParsedQuantity]:
    """Extract normalized (Decimal magnitude, unit) quantities from text."""
    if not text:
        return []

    norm = normalize_span(text)
    if apply_table_scale and table_scale is None:
        table_scale = _detect_contextual_table_scale(norm)
    is_es = bool(_SPANISH_CONTEXT_RE.search(norm)) if es_context is None else es_context

    # Mask ISO dates, natural dates, slash dates, and structural identifiers while preserving character offsets
    scrubbed = _DOT_DATE_RE.sub(_mask_preserving_len, norm)
    scrubbed = _ISO_DATE_RE.sub(_mask_preserving_len, scrubbed)
    scrubbed = _MDY_DATE_RE.sub(_mask_preserving_len, scrubbed)
    scrubbed = _DMY_DATE_RE.sub(_mask_preserving_len, scrubbed)
    scrubbed = _SLASH_DATE_RE.sub(_mask_preserving_len, scrubbed)
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

    denominator_words = {"half": 2, "halves": 2, "third": 3, "quarter": 4,
                         "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                         "eighth": 8, "ninth": 9, "tenth": 10}
    fraction_pattern = re.compile(
        r"\b(" + "|".join(_ONES_EN) + r")[ -]+("
        + "|".join(denominator_words) + r")s?\b", re.IGNORECASE)
    fraction_unit_suffix_re = re.compile(
        r"(?:\s+(?:of\s+(?:an?|one|the|1)|de\s+(?:un|una|el|la|1)|an?))?\s*"
        r"(?:(?P<scale>mil\s+millones|billions?|millions?|millones|mill[oó]n|thousands?|miles|mil)\b\s*(?:(?:of|de)\s+)?)?"
        r"(?P<unit>(?!(?:us\$|\$|€|£)\s*\d)" + _UNIT_PATTERN + r")?",
        re.IGNORECASE,
    )
    for m in fraction_pattern.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        ratio = Decimal(_ONES_EN[m[1].lower()]) / Decimal(denominator_words[m[2].lower()])
        tail_m = fraction_unit_suffix_re.match(scrubbed, m.end())
        tail_scale = tail_m.group("scale") if tail_m else None
        tail_unit_raw = tail_m.group("unit") if tail_m else None
        if tail_unit_raw or tail_scale:
            unit = canonicalize_unit(tail_unit_raw) if tail_unit_raw else default_unit
            scale_factor = (
                _SCALE_FACTORS.get(re.sub(r"\s+", " ", tail_scale.strip().lower()), Decimal("1"))
                if tail_scale
                else Decimal("1")
            )
            mag = ratio * scale_factor
            span = (m.start(), tail_m.end())
            if _overlaps(span[0], span[1]):
                continue
            occupied_spans.append(span)
            results.append(ParsedQuantity(mag.normalize(), unit, is_fraction=True, span=span))
        else:
            magnitude = ratio * Decimal("100")
            occupied_spans.append(m.span())
            results.append(ParsedQuantity(magnitude.normalize(), "PERCENT", is_fraction=True, span=m.span()))

    # 2. Financial ratio covenants ("1.25 to 1.00" -> 1.25 MULTIPLIER)
    for m in _RATIO_COVENANT_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        mag = _parse_decimal_token(m.group("num"), es_context=is_es, is_percent=True)
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=mag.normalize(), unit="MULTIPLIER", span=span))

    # 3. Range expressions ("between 30 and 60 days", "30-60 days")
    for m in _RANGE_QUANTITY_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        raw_n1 = m.group("n1a") or m.group("n1b")
        raw_n2 = m.group("n2a") or m.group("n2b")
        if not raw_n1 or not raw_n2:
            continue
        u = canonicalize_unit(m.group("unit"))
        d1 = _parse_decimal_token(raw_n1, es_context=is_es, is_percent=(u == "PERCENT"))
        d2 = _parse_decimal_token(raw_n2, es_context=is_es, is_percent=(u == "PERCENT"))
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=d1.normalize(), unit=u, is_range_endpoint=True, span=span))
        results.append(ParsedQuantity(magnitude=d2.normalize(), unit=u, is_range_endpoint=True, span=span))

    # 4. Compound word quantities ("twenty-five (25) days", "five million dollars", "ten-year")
    for m in _COMPOUND_WORD_QTY_RE.finditer(scrubbed):
        if _overlaps(m.start(), m.end()):
            continue
        prefix_before = scrubbed[: m.start()].rstrip()
        if prefix_before and prefix_before[-1].isdigit():
            continue
        word_val = _parse_word_number_phrase(m.group("phrase"))
        if word_val is None:
            continue
        mag = Decimal(word_val)
        scale_raw = m.group("scale")
        if scale_raw:
            scale_key = re.sub(r"\s+", " ", scale_raw.strip().lower())
            mag *= _SCALE_FACTORS.get(scale_key, Decimal("1"))
        unit = canonicalize_unit(m.group("unit"))
        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=mag.normalize(), unit=unit, span=span))

    # 5. Numeric quantities
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

        try:
            magnitude = _parse_decimal_token(
                raw_num,
                es_context=is_es,
                is_percent=(unit in {"PERCENT", "BPS", "MULTIPLIER"}),
            )
        except InvalidOperation:
            continue

        if sign or (unit == "PERCENT" and re.match(r"\([\d.,]+\)\s*%", m.group(0))):
            magnitude = -magnitude

        if scale:
            scale_key = re.sub(r"\s+", " ", scale.strip().lower())
            if scale_key in {"k", "m", "b"}:
                has_currency = unit in _CURRENCY_UNITS
                attached = len(scale_space) == 0
                if has_currency or attached:
                    magnitude *= _SCALE_FACTORS[scale_key]
            else:
                magnitude *= _SCALE_FACTORS.get(scale_key, Decimal("1"))
        elif table_scale is not None and (unit in _CURRENCY_UNITS or unit == "NONE"):
            magnitude *= table_scale

        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=magnitude.normalize(), unit=unit, span=span))

    results.sort(key=lambda q: q.span[0])
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


def _is_sentence_boundary_dot(text: str, pos: int) -> bool:
    """Return True if `text[pos]` ('.') is a sentence boundary rather than a decimal/section dot or abbreviation."""
    if pos < 0 or pos >= len(text) or text[pos] != ".":
        return False
    # Decimal or section number like "4.01" or "6.75"
    if pos > 0 and pos + 1 < len(text) and text[pos - 1].isdigit() and text[pos + 1].isdigit():
        return False
    # Common legal/financial abbreviation like "Sec.", "Art.", "No."
    window_before = text[max(0, pos - 8) : pos + 1]
    if _ABBREV_BEFORE_DOT_RE.search(window_before):
        return False
    return True


def _enclosing_sentence_bounds(norm_source: str, quote_start: int, quote_end: int) -> tuple[int, int]:
    """Find the start and end character offsets of the sentence enclosing `norm_source[quote_start:quote_end]`."""
    sent_start = 0
    for i in range(quote_start - 1, -1, -1):
        ch = norm_source[i]
        if ch in ";\n" or (ch == "." and _is_sentence_boundary_dot(norm_source, i)):
            sent_start = i + 1
            break

    sent_end = len(norm_source)
    for i in range(quote_end, len(norm_source)):
        ch = norm_source[i]
        if ch in ";\n" or (ch == "." and _is_sentence_boundary_dot(norm_source, i)):
            sent_end = i + 1
            break

    return sent_start, sent_end


def segment_sentences_with_spans(text: str, base_offset: int = 0) -> list[tuple[int, int, str]]:
    """Split *text* into trimmed ``(abs_start, abs_end, sentence)`` tuples preserving offsets.

    Uses ``_is_sentence_boundary_dot`` so decimals (e.g. ``5.80%``), section numbers
    (``Section 4.01``), and legal abbreviations (``Sec.``, ``Art.``, ``No.``) are never
    treated as sentence boundaries. Also avoids splitting inside dotted filenames/emails
    when the dot is immediately followed by a letter (e.g. ``.docx``, ``example.com``).
    """
    if not text:
        return []
    n = len(text)
    raw_bounds: list[tuple[int, int]] = []
    seg_start = 0
    for i, ch in enumerate(text):
        is_break = False
        if ch in "\n;!?":
            is_break = True
        elif ch == ".":
            next_ch = text[i + 1] if i + 1 < n else ""
            prev_ch = text[i - 1] if i > 0 else ""
            if (
                prev_ch != "."
                and not (next_ch and next_ch.isalpha())
                and _is_sentence_boundary_dot(text, i)
            ):
                is_break = True
        if is_break:
            raw_bounds.append((seg_start, i + 1))
            seg_start = i + 1
    if seg_start < n:
        raw_bounds.append((seg_start, n))

    sentences: list[tuple[int, int, str]] = []
    for raw_s, raw_e in raw_bounds:
        chunk = text[raw_s:raw_e]
        ltrim = len(chunk) - len(chunk.lstrip())
        rtrim = len(chunk) - len(chunk.rstrip())
        s = raw_s + ltrim
        e = raw_e - rtrim
        if s < e:
            sentences.append((base_offset + s, base_offset + e, text[s:e]))
    return sentences


def _has_truncated_sentence_qualifier(norm_source: str, norm_quote: str) -> str | None:
    """Return the truncated prefix or suffix if *norm_quote* drops a negation or exception from its sentence."""
    idx = norm_source.lower().find(norm_quote.lower())
    if idx < 0:
        return None
    quote_end = idx + len(norm_quote)
    sent_start, sent_end = _enclosing_sentence_bounds(norm_source, idx, quote_end)

    leading_clause = norm_source[sent_start:idx].strip()
    if leading_clause and _TRUNCATED_QUALIFIER_RE.search(leading_clause):
        return leading_clause

    trailing_clause = norm_source[quote_end:sent_end].strip()
    if trailing_clause and _TRAILING_EXCEPTION_RE.search(trailing_clause):
        return trailing_clause

    return None


def _split_quote_subclauses(norm_quote: str) -> list[tuple[int, int, str]]:
    """Split a quote into `(start, end, text)` sub-clauses along sentence or coordinating clause boundaries."""
    spans: list[tuple[int, int, str]] = []
    # Split on sentence breaks (`. `, `; `) or coordinating conjunctions (` and `, ` whereas `, ` while `)
    pattern = re.compile(
        r"(?:\.(?=\s+[A-ZÁÉÍÓÚÑ])|;\s+|\s+(?:and|whereas|while)\s+)",
        re.IGNORECASE,
    )
    last = 0
    for m in pattern.finditer(norm_quote):
        if not (norm_quote[m.start()] == "." and not _is_sentence_boundary_dot(norm_quote, m.start())):
            seg = norm_quote[last : m.start()].strip()
            if seg:
                spans.append((last, m.start(), seg))
            last = m.end()
    tail = norm_quote[last:].strip()
    if tail:
        spans.append((last, len(norm_quote), tail))
    return spans or [(0, len(norm_quote), norm_quote)]


def _extract_binding_cues(subject: str, predicate: str) -> tuple[list[str], set[str]]:
    """Extract specific multi-word designators ('class c', 'tranche b') and content tokens from subject & predicate."""
    phrases: list[str] = []
    for raw in (subject, predicate):
        cleaned = re.sub(r"[_\-]+", " ", (raw or "").strip().lower())
        for m in re.finditer(r"\b((?:class|tranche|series|tier|step)\s+[a-z0-9]+)\b", cleaned):
            phrases.append(m.group(1))

    generic = {
        "note",
        "notes",
        "deal",
        "waterfall",
        "class",
        "tranche",
        "series",
        "tier",
        "amount",
        "rate",
        "value",
        "term",
        "terms",
        "period",
        "date",
        "days",
        "day",
        "usd",
        "eur",
    }
    tokens: set[str] = set()
    for raw in (subject, predicate):
        for tok in re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", (raw or "").lower()):
            if len(tok) >= 2 and tok not in generic and tok not in _VALUE_STOPWORDS:
                tokens.add(tok)
    return phrases, tokens


def _score_subclause_match(clause_text: str, phrases: list[str], tokens: set[str]) -> int:
    low = clause_text.lower()
    score = 0
    for ph in phrases:
        if re.search(r"\b" + re.escape(ph) + r"\b", low):
            score += 10
        else:
            # If the phrase specifies a designator like "class c" and this clause specifies a DIFFERENT class "class a", penalize!
            prefix = ph.split()[0]
            if re.search(r"\b" + re.escape(prefix) + r"\s+[a-z0-9]+\b", low):
                score -= 10
    clause_words = set(re.findall(r"[a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ]+", low))
    for tok in sorted(tokens):
        if tok in clause_words or any(
            cw.startswith(tok[:4]) and len(tok) >= 4 for cw in sorted(clause_words)
        ):
            score += 3
    return score


def _check_multi_quantity_subject_binding(
    assertion: Assertion,
    vq: ParsedQuantity,
    same_unit_quote_quantities: list[ParsedQuantity],
) -> bool:
    """When a quote contains multiple distinct quantities of the same unit (and not a single range),
    bind *vq* to the sub-clause matching `(assertion.subject, assertion.predicate)`."""
    non_range = [qq for qq in same_unit_quote_quantities if not qq.is_range_endpoint]
    distinct_mags = {qq.magnitude for qq in non_range}
    if len(distinct_mags) <= 1:
        return True

    norm_quote = normalize_span(assertion.quote)
    # Ordered parallel lists have positional binding, not nearest-neighbour binding.
    if re.search(r"\brespectively\b", norm_quote, re.IGNORECASE):
        designators = list(re.finditer(r"\b(?:class|tranche|series|tier)\s+[a-z0-9]+\b",
                                      norm_quote, re.IGNORECASE))
        targets, _ = _extract_binding_cues(assertion.subject, assertion.predicate)
        if (len(designators) == len(non_range) and designators
                and designators[-1].end() < non_range[0].span[0]):
            indexes = [i for i, d in enumerate(designators) if d[0].lower() in targets]
            if len(indexes) == 1:
                return vq.matches(non_range[indexes[0]])
        return False
    subclauses = _split_quote_subclauses(norm_quote)
    phrases, tokens = _extract_binding_cues(assertion.subject, assertion.predicate)

    if len(subclauses) >= 2:
        scored: list[tuple[int, tuple[int, int, str]]] = [
            (_score_subclause_match(sc[2], phrases, tokens), sc) for sc in subclauses
        ]
        max_score = max(s for s, _ in scored)
        if max_score <= 0:
            return False
        best_spans = [(sc[0], sc[1]) for s, sc in scored if s == max_score]
        allowed_q = [
            qq
            for qq in same_unit_quote_quantities
            if any(start <= qq.span[0] < end for start, end in best_spans)
        ]
        # Only accept if the best-scoring subclause(s) uniquely identify a single magnitude!
        if allowed_q and len({qq.magnitude for qq in allowed_q}) == 1:
            return any(vq.matches(qq) for qq in allowed_q)
        return False

    # Fallback within a single unsplit clause: match by nearest character distance to a unique cue
    low_quote = norm_quote.lower()
    target_pos = -1
    for ph in phrases:
        if low_quote.count(ph) == 1:
            target_pos = low_quote.find(ph)
            break
    if target_pos < 0:
        for tok in sorted(tokens):
            if len(re.findall(r"\b" + re.escape(tok) + r"\b", low_quote)) == 1:
                target_pos = low_quote.find(tok)
                break
    if target_pos < 0:
        return False

    nearest = min(same_unit_quote_quantities, key=lambda qq: abs(qq.span[0] - target_pos))
    return vq.matches(nearest)


def _check_multi_date_binding(
    assertion: Assertion,
    proposed_iso_date: str,
    quote_date_spans: list[tuple[tuple[int, int], str]],
) -> bool:
    """When a quote contains 2+ distinct dates (e.g. Maturity Date vs Closing Date),
    bind *proposed_iso_date* to the sub-clause matching `(assertion.subject, assertion.predicate)`."""
    distinct_dates = {iso for _, iso in quote_date_spans}
    if len(distinct_dates) <= 1:
        return True

    norm_quote = normalize_span(assertion.quote)
    subclauses = []
    previous_end = 0
    for (start, end), _ in sorted(quote_date_spans):
        subclauses.append((previous_end, end, norm_quote[previous_end:end]))
        previous_end = end
    phrases, tokens = _extract_binding_cues(assertion.subject, assertion.predicate)

    if len(subclauses) >= 2:
        scored = [(_score_subclause_match(sc[2], phrases, tokens), sc) for sc in subclauses]
        max_score = max(s for s, _ in scored)
        if max_score <= 0:
            return False
        best_spans = [(sc[0], sc[1]) for s, sc in scored if s == max_score]
        allowed_dates = {
            iso
            for (d_start, _), iso in quote_date_spans
            if any(start <= d_start < end for start, end in best_spans)
        }
        if allowed_dates:
            return proposed_iso_date in allowed_dates

    return True


def _content_tokens(text: str) -> set[str]:
    expanded = (text or "").replace("+", " plus ").replace("&", " and ")
    words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+", normalize_span(expanded).lower())
    out: set[str] = set()
    for w in words:
        if len(w) < 2 or w in _VALUE_STOPWORDS:
            continue
        if (
            w in _UNIT_ALIASES
            or w in _SCALE_FACTORS
            or w in _ONES_EN
            or w in _TENS_EN
            or w in _ONES_ES
            or w in _TENS_ES
            or w in _HUNDREDS_ES
            or w in {"hundred", "thousand"}
        ):
            continue
        out.add(w)
    return out


def _tokens_grounded_in_quote(val_tokens: set[str], quote_tokens: set[str]) -> bool:
    """Return True if every substantive content word in *val_tokens* matches a word/stem in *quote_tokens*."""
    if not val_tokens:
        return False
    for vt in val_tokens:
        if vt in quote_tokens:
            continue
        # Allow basic singular/plural or morphological stem match (first 5 chars for words >= 5 chars)
        if len(vt) >= 5 and any(qt.startswith(vt[:5]) or vt.startswith(qt[:5]) for qt in quote_tokens if len(qt) >= 5):
            continue
        return False
    return True


def _has_polarity_inversion(quote_text: str, value_text: str) -> bool:
    """Return True if *quote_text* has clausal negation that is inverted/omitted in a propositional *value_text*."""
    q_low = normalize_span(quote_text).lower()
    v_low = normalize_span(value_text).lower()
    # Remove shared "without <noun>" phrases so "without cause" in both does not mask "In no event" in quote
    for shared in re.findall(r"\b(?:without|sin)\s+[a-záéíóúñü]+\b", q_low):
        if shared in v_low:
            q_low = q_low.replace(shared, " ")
            v_low = v_low.replace(shared, " ")
    q_neg = bool(_CLAUSE_NEGATION_RE.search(q_low))
    v_neg = bool(_CLAUSE_NEGATION_RE.search(v_low))
    if q_neg == v_neg:
        return False
    # If the value is a bare noun phrase (not a negative or affirmative proposition), it is a slot extraction (e.g. carveout/excluded item), not a polarity inversion
    if not v_neg and not _AFFIRMATIVE_PROPOSITION_RE.search(v_low):
        return False
    return True


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
                        f"Extract an exact verbatim substring from '{source.id}'."
                    ),
                )
            )
            return issues

        # Reject bare single-token / out-of-context quotes or section-header-only quotes
        quote_words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+", norm_quote)
        if len(norm_source) > len(norm_quote) and (
            len(norm_quote) < 6
            or len(quote_words) < 2
            or _SECTION_HEADER_ONLY_RE.match(norm_quote) is not None
        ):
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote '{assertion.quote}' only cites a section heading or lacks substantive clause context from '{source.id}'."
                    ),
                    remediation=(
                        f"Cite the substantive clause body from '{source.id}' supporting '{assertion.predicate}'."
                    ),
                )
            )
            return issues

        # Reject quotes that truncate a leading negation or trailing exception from the sentence
        truncated_qual = _has_truncated_sentence_qualifier(norm_source, norm_quote)
        if truncated_qual is not None:
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote truncates material negation or exception '{truncated_qual}' from its sentence in '{source.id}'."
                    ),
                    remediation="Include the complete sentence with its governing negation or exception clause.",
                )
            )
            return issues

    # Reference text for quantity/date verification:
    # NEVER trust an unverified quote when require_exact_quote=False!
    reference_text = assertion.quote if quote_in_source else source.text
    is_es = bool(_SPANISH_CONTEXT_RE.search(norm_source))

    if policy.require_quantity_match:
        # 1. Check dates in value strictly against the verified quote
        if _SLASH_DATE_RE.search(assertion.value or "") or _SHORT_SLASH_DATE_RE.search(assertion.value or ""):
            issues.append(
                Issue(
                    code="DATE_MISMATCH",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=f"Slash date in proposed value '{assertion.value}'; use canonical ISO YYYY-MM-DD.",
                    remediation="Format dates in proposed values as ISO YYYY-MM-DD.",
                )
            )
            return issues

        val_dates = extract_dates(assertion.value, allow_slash=False)
        quote_date_spans = extract_dates_with_spans(reference_text, allow_slash=True)
        quote_dates = {iso for _, iso in quote_date_spans}
        for d in val_dates:
            if d not in quote_dates or not _check_multi_date_binding(assertion, d, quote_date_spans):
                issues.append(
                    Issue(
                        code="DATE_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Date '{d}' in proposed value '{assertion.value}' is not grounded for '{assertion.predicate}' in cited quote."
                        ),
                        remediation=f"Align date in '{assertion.predicate}' with the exact date in the cited quote from '{source.id}'.",
                    )
                )

        # 2. Check numeric quantities & units strictly against the verified quote
        table_scale = _detect_contextual_table_scale(assertion.quote if quote_in_source else source.text, source.text)
        def rate_period(text: str) -> str | None:
            match = re.search(r"(?:%|percent|bps)\s+(?:per\s+|/)([a-z]+)", text, re.IGNORECASE)
            if not match:
                return None
            token = match[1].lower()
            return "YEARS" if token == "annum" else canonicalize_unit(token)

        proposed_period, cited_period = rate_period(assertion.value), rate_period(reference_text)
        if proposed_period and cited_period and proposed_period != cited_period:
            issues.append(Issue(code="UNIT_MISMATCH", disposition="RETRY", assertion_id=assertion.id,
                                subject=assertion.subject, predicate=assertion.predicate,
                                message="The proposed rate uses a different time denominator from the citation.",
                                remediation="Preserve the rate period; annualization requires an explicit conversion policy."))
        val_quantities = extract_quantities(assertion.value, explicit_unit=assertion.unit, es_context=is_es)
        quote_quantities = extract_quantities(
            reference_text,
            table_scale=table_scale,
            es_context=is_es,
        )

        # If predicate explicitly requires a dimension (e.g. cure_period_days) and value is dimensionless (unit="NONE"), reject
        req_dim = infer_predicate_dimension(assertion.predicate)
        for vq in val_quantities:
            if vq.unit == "NONE" and req_dim in {"DURATION", "CURRENCY"}:
                issues.append(
                    Issue(
                        code="UNIT_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Predicate '{assertion.predicate}' requires an explicit unit, but proposed value '{assertion.value}' is dimensionless."
                        ),
                        remediation=f"Include the governing duration unit from '{source.id}'.",
                    )
                )
                continue

            matching_q = [
                qq
                for qq in quote_quantities
                if vq.matches(qq) and not vq.same_magnitude_wrong_unit(qq)
            ]
            if matching_q:
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
                                f"entity or clause in the cited quote than '{assertion.subject}.{assertion.predicate}'."
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

        # 3. Verify substantive content-token grounding and polarity parity between quote and value
        if quote_in_source and not issues:
            val_tokens = _content_tokens(assertion.value)
            quote_tokens = _content_tokens(reference_text)
            if not val_dates and not val_quantities:
                norm_val = normalize_span(assertion.value).lower()
                norm_ref = normalize_span(reference_text).lower()
                literal_grounded = bool(norm_val) and not val_tokens and bool(
                    re.search(r"(?<![a-z0-9])" + re.escape(norm_val) + r"(?![a-z0-9])", norm_ref)
                )
                if not literal_grounded and not _tokens_grounded_in_quote(val_tokens, quote_tokens):
                    issues.append(
                        Issue(
                            code="QUANTITY_MISMATCH",
                            disposition="RETRY",
                            assertion_id=assertion.id,
                            subject=assertion.subject,
                            predicate=assertion.predicate,
                            message=(
                                f"Proposed value '{assertion.value}' contains ungrounded terms not present in cited quote '{assertion.quote[:80]}'."
                            ),
                            remediation=f"Ensure '{assertion.predicate}' value uses only terms grounded in the cited quote.",
                        )
                    )
                elif _has_polarity_inversion(reference_text, assertion.value):
                    issues.append(
                        Issue(
                            code="QUANTITY_MISMATCH",
                            disposition="RETRY",
                            assertion_id=assertion.id,
                            subject=assertion.subject,
                            predicate=assertion.predicate,
                            message=(
                                f"Proposed value '{assertion.value}' inverts the negation/polarity of cited quote '{assertion.quote[:80]}'."
                            ),
                            remediation="Preserve the negative/prohibitive polarity stated in the cited quote.",
                        )
                    )
            elif val_tokens and not _tokens_grounded_in_quote(val_tokens, quote_tokens):
                # Even when a value includes a number (e.g. "Step 13 senior..."), any accompanying substantive words must be grounded in the quote
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Proposed value '{assertion.value}' contains ungrounded qualifier tokens not present in cited quote."
                        ),
                        remediation=f"Ensure all qualifiers in '{assertion.value}' match '{source.id}'.",
                    )
                )

    return issues
