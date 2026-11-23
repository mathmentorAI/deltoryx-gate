"""Internal quantity, date and source-layout extraction primitives."""

from __future__ import annotations


import re


import unicodedata


from dataclasses import dataclass


from datetime import date


from decimal import Decimal, InvalidOperation


from ._source_models import normalize_entity_key


_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
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
    r"([a-zA-ZÀ-ÖØ-öø-ÿ0-9])-\s*\n\s*([a-zA-ZÀ-ÖØ-öø-ÿ0-9])"
)


_BULLET_BLOCKLIST_PREV_RE = re.compile(
    r"(?:section|sec|article|art|clause|part|item|step|rule|paragraph|schedule|exhibit|annex|"
    r"appendix|class|series|tranche|tier|no|vol|fig|pp?|cl[aá]usula|art[ií]culo|apartado|punto|"
    r"page|january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec|u\.s|dr|mr|mrs|ms|jr|sr|co|inc|corp|ltd|st|v)$",
    re.IGNORECASE,
)


_BULLET_MARKER_RE = re.compile(
    r"(?P<tok>(?:x{1,2}(?:i{1,3}|iv|vi{0,3})?|i{1,3}|iv|vi{0,3}|ix|[a-zA-Z]|\d{1,2}))(?P<delim>[.)])(?=[ \t]+[A-Za-z0-9($€£\"'\[\*])"
)


def _canonicalize_punctuation_and_numbers(text: str) -> str:
    """Normalize punctuation/number/currency spacing without altering values or dates."""
    if not text:
        return ""
    s = text
    # Number internal comma/decimal spacing ("5 , 000" -> "5,000", "$5, 000" -> "$5,000", "5 . 25" -> "5.25")
    s = re.sub(r"(?<=\d)[ \t]+,[ \t]*(?=\d{3}(?!\d))", ",", s)
    s = re.sub(r"(?<=[$€£])[ \t]*(\d{1,3}),[ \t]+(?=\d{3}(?!\d))", r"\1,", s)
    s = re.sub(r"(?<=\d)[ \t]+\.[ \t]*(?=\d)", ".", s)
    # Currency symbol spacing before digits ("$ 5,000" -> "$5,000")
    s = re.sub(r"(US\$|\$|€|£)[ \t]+(?=\d)", r"\1", s)
    # Spaces around delimiters: remove horizontal space before , ; : ) ] % and after ( [
    s = re.sub(r"[ \t]+([,;:\)%\]])", r"\1", s)
    s = re.sub(r"([(\[])[ \t]+", r"\1", s)
    return s


_LOWER_ROMAN_MARKERS = frozenset(
    {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"}
)


def _canonicalize_bullets(text: str) -> str:
    """Canonicalize enumeration markers '(a)' / 'a)' / 'a.', '(i)' / 'i)' / 'i.', '(1)' / '1)' / '1.' -> '(tok)'."""
    if not text:
        return ""
    out: list[str] = []
    last = 0
    for m in _BULLET_MARKER_RE.finditer(text):
        start, end = m.start(), m.end()
        tok = m.group("tok")
        delim = m.group("delim")
        if start > 0 and (text[start - 1].isalnum() or text[start - 1] in ("(", ".", ",", "$", "€", "£", "-", "_")):
            continue
        prefix = text[:start]
        stripped_prefix = prefix.rstrip(" \t\r\n")
        if delim == ")":
            # Ensure ')' does not close an earlier unmatched '(' on the same clause/line
            clause_tail = prefix[max(prefix.rfind("\n"), prefix.rfind(";")) + 1 :]
            if clause_tail.count("(") > clause_tail.count(")"):
                continue
        else:
            # For `tok.`, reject decimals/dotted section ids
            if start + len(tok) + 1 < len(text) and text[start + len(tok) + 1].isdigit():
                continue
            is_lower_roman = tok in _LOWER_ROMAN_MARKERS
            # Require valid list boundary unless lowercase roman numeral bullet ("means i. the period")
            if not is_lower_roman and stripped_prefix and stripped_prefix[-1] not in ("\n", ";", ":", "."):
                if not re.search(r"(?:^|[\s,;:])(?:and|or)$", stripped_prefix, re.IGNORECASE):
                    continue
            if stripped_prefix and not is_lower_roman:
                prev_word_m = re.search(r"([A-Za-z.]+)\.?$", stripped_prefix)
                if prev_word_m and _BULLET_BLOCKLIST_PREV_RE.search(prev_word_m.group(1).rstrip(".")):
                    continue
                # Avoid converting uppercase initial after a personal/entity word ("John A. Smith")
                if len(tok) == 1 and tok.isupper() and stripped_prefix[-1] not in ("\n", ";", ":"):
                    continue
        out.append(text[last:start])
        out.append(f"({tok})")
        last = end
    out.append(text[last:])
    return "".join(out)


def _canonicalize_layout_text_with_provenance(text: str) -> tuple[str, list[dict[str, str]]]:
    """Canonicalize typography, spacing, bullets, and equivalent quantity notations with provenance."""
    if not text:
        return "", []
    s = re.sub(r"\u00ad(?=[ \t]*\r?\n)", "-", text)
    s = re.sub(r"(?<=[A-Za-z0-9])\u2014(?=[A-Za-z0-9])", " - ", s)
    s = s.translate(_QUOTE_MAP)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s).replace("\\$", "$")
    s = re.sub(r"\s+", " ", s).strip()
    s = _canonicalize_punctuation_and_numbers(s)
    s = _canonicalize_bullets(s)
    s, prov = canonicalize_quantity_notations_with_provenance(s)
    return re.sub(r"\s+", " ", s).strip(), prov


def _canonicalize_layout_text(text: str) -> str:
    """Canonicalize typography, whitespace, punctuation/number spaces, bullets, and quantity notations."""
    canon, _ = _canonicalize_layout_text_with_provenance(text)
    return canon


def _build_line_tolerant_search_regex(norm_quote: str) -> str:
    """Build a regex matching canonical *norm_quote* against line-wrapped/separated-$ text."""
    parts: list[str] = []
    i = 0
    n = len(norm_quote)
    while i < n:
        ch = norm_quote[i]
        if ch == " ":
            parts.append(r"\s+")
            i += 1
            continue
        if norm_quote.startswith("US$", i) and i + 3 < n and norm_quote[i + 3].isdigit():
            parts.append(r"US\$\s*")
            i += 3
            continue
        if ch in ("$", "€", "£") and i + 1 < n and norm_quote[i + 1].isdigit():
            parts.append(re.escape(ch) + r"\s*")
            i += 1
            continue
        if ch in (",", ";", ":", ")", "]", "%"):
            parts.append(r"[ \t]*" + re.escape(ch))
            if ch == "," and i > 0 and norm_quote[i - 1].isdigit() and i + 1 < n and norm_quote[i + 1].isdigit():
                parts.append(r"[ \t]*")
            i += 1
            continue
        if ch in ("(", "["):
            parts.append(re.escape(ch) + r"[ \t]*")
            i += 1
            continue
        if ch == "." and i > 0 and i + 1 < n and norm_quote[i - 1].isdigit() and norm_quote[i + 1].isdigit():
            parts.append(r"[ \t]*\.[ \t]*")
            i += 1
            continue
        parts.append(re.escape(ch))
        i += 1
    return "".join(parts)


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
    "banking day": "BUSINESS_DAYS",
    "banking days": "BUSINESS_DAYS",
    "trading day": "BUSINESS_DAYS",
    "trading days": "BUSINESS_DAYS",
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
    "calendar month": "CALENDAR_MONTHS",
    "calendar months": "CALENDAR_MONTHS",
    "mes natural": "CALENDAR_MONTHS",
    "meses naturales": "CALENDAR_MONTHS",
    "calendar year": "CALENDAR_YEARS",
    "calendar years": "CALENDAR_YEARS",
    "año natural": "CALENDAR_YEARS",
    "años naturales": "CALENDAR_YEARS",
    "fiscal year": "FISCAL_YEARS",
    "fiscal years": "FISCAL_YEARS",
    "año fiscal": "FISCAL_YEARS",
    "años fiscales": "FISCAL_YEARS",
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
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:\s*,\s*|\s+)(?P<year>\d{4})\b",
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
    r"(?:business[\s-]+days?|working[\s-]+days?|banking[\s-]+days?|trading[\s-]+days?|d[ií]as?\s+h[aá]biles|"
    r"calendar[\s-]+days?|d[ií]as?\s+naturales|"
    r"calendar[\s-]+months?|mes(?:es)?\s+natural(?:es)?|"
    r"calendar[\s-]+years?|a[ñn]os?\s+natural(?:es)?|"
    r"fiscal[\s-]+years?|a[ñn]os?\s+fiscales?|"
    r"per\s+cent\.?|por\s+ciento|percent\b|pct\b|%|"
    r"basis\s+points?\b|puntos?\s+b[aá]sicos?\b|bps\b|"
    r"us\$|\$|€|£|"
    r"usd\b|dollars?\b|d[oó]lares?\b|eur\b|euros?\b|gbp\b|pounds?\b|libras?\b|"
    r"chf\b|jpy\b|cad\b|aud\b|mxn\b|"
    r"days?\b|d[ií]as?\b|weeks?\b|semanas?\b|months?\b|mes(?:es)?\b|"
    r"years?\b|a[ñn]os?\b|hours?\b|horas?\b|hrs?\b|hr\b|times\b|veces\b|x\b)"
)


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
    r"|[1-9]\d{0,2}(?:\.\d{3})+(?!\d)(?:,\d+)?"
    r"|\d+,\d+"
    r"|\d+(?:\.\d+)?"
    r")\)?"
    r"(?:(?P<scale_space>\s*)(?P<scale>mil\s+millones|billions?|bn|millions?|millones|mill[oó]n|thousands?|miles|mil|mm|k|m|b)\b\)?)?"
    r"(?:(?:\s*-\s*|\s*)(?:\([a-zA-Z0-9]+\)(?:\s*-\s*|\s*))?(?:(?:de|of)\s+)?(?!(?:us\$|\$|€|£)\s*\d)(?P<suffix>" + _UNIT_PATTERN + r"))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _TableScaleHeaderSpec:
    scale: Decimal
    currency: str | None
    hdr_start: int
    hdr_end: int
    span_end: int
    is_parenthetical: bool
    exclude_per_share: bool = False
    exclude_par_value: bool = False
    exclude_percentages: bool = False
    exclude_shares: bool = False
    inherited_row_label: str = ""


_SQUASHED_SCALE_WORDS: dict[str, Decimal] = {
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "miles": Decimal("1000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "millones": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
}


_PAREN_PREFIX_SQUASHED_RE = re.compile(
    r"^(?:unaudited|audited|condensed|consolidated)*"
    r"(?P<pre>(?:amounts?|dollars?|usdollars?|usd|eur|gbp|euros?|pounds?|[$€£])+)?"
    r"(?P<prep>in|en)?"
    r"(?P<sym>[$€£])?"
    r"(?:(?P<scale_w>thousands?|millions?|billions?|miles|millones)|(?P<zeros>0{3,9})s?)"
    r"(?P<post>(?:of|de)(?:u\.?s\.?)?(?:dollars?|euros?|pounds?))?$",
    re.IGNORECASE,
)


_BANNER_SCALE_DECL_RE = re.compile(
    r"(?:(?:amounts?|dollars?|u\.?s\.?\s*dollars?|usd|eur|gbp|euros?|pounds?|[$€£])\s+(?:in|en)\s+[$€£]?(?P<scale_b>thousands|millions|billions|miles|millones)\b"
    r"|\b(?:in|en)\s+[$€£]?(?P<scale_i>thousands|millions|billions|miles|millones)\b"
    r"(?:\s+(?:of|de)\s+(?:U\.S\.\s+)?(?:dollars?|euros?|pounds?))?)"
    r"(?:\s*[,;]\s*except[^\n.;]{1,140}(?:\n[^\n.;]{1,80})?)?",
    re.IGNORECASE,
)


_PAREN_BLOCK_RE = re.compile(r"\(([^()]{2,220})\)")


_ROW_PER_SHARE_OR_PAR_RE = re.compile(
    r"\b(?:per\b(?:\s+[a-z0-9\-]+){0,5}\s+(?:shares?|units?|ads|adss)|"
    r"per[-\s]+(?:share|unit|ads)|"
    r"(?:par|stated)\s+value|"
    r"dollars?\s+per\s+share|in\s+dollars\s+per\s+share|"
    r"(?:basic|diluted)\s+(?:eps|net\s+(?:income|loss|earnings)\s+per\s+share)|"
    r"eps)\b",
    re.IGNORECASE,
)


_ROW_PERCENTAGE_RE = re.compile(
    r"(?:%|\b(?:percentage\s+of|percent|pct|bps|basis\s+points?)\b)",
    re.IGNORECASE,
)


_ROW_SHARES_RE = re.compile(
    r"\b(?:number\s+of\s+shares|weighted[- ]average(?:\s+[a-z0-9\-]+){0,4}\s+shares?|"
    r"common\s+shares?|preferred\s+shares?|shares?\s+(?:outstanding|authorized|issued|used))\b",
    re.IGNORECASE,
)


_ROW_RATIO_RE = re.compile(r"\bratios?\b", re.IGNORECASE)


_EXCEPT_SPLIT_RE = re.compile(r"[,;\s]*\bexcept\b|(?<=,)except", re.IGNORECASE)


def _currency_from_squashed(token: str) -> str | None:
    if "$" in token or "usd" in token or "dollar" in token:
        return "USD"
    if "€" in token or "eur" in token:
        return "EUR"
    if "£" in token or "gbp" in token or "pound" in token:
        return "GBP"
    return None


def _parse_paren_inner_scale(
    inner: str,
) -> tuple[Decimal, str | None, bool, bool, bool, bool, bool] | None:
    """Parse parenthetical scale declaration using normalized-token grammar (OCR-space tolerant).

    Returns `(scale, currency, has_prefix_or_prep, exclude_per_share, exclude_par_value, exclude_percentages, exclude_shares)`.
    """
    parts = _EXCEPT_SPLIT_RE.split(inner, maxsplit=1)
    prefix_raw = parts[0].strip()
    exc_raw = parts[1].strip() if len(parts) > 1 else ""
    if not prefix_raw:
        return None
    sq = re.sub(r"[^a-z0-9$€£]", "", prefix_raw.lower())
    m = _PAREN_PREFIX_SQUASHED_RE.match(sq)
    if not m:
        return None
    zeros = m.group("zeros")
    scale_w = m.group("scale_w")
    scale = (Decimal(10) ** len(zeros)) if zeros else _SQUASHED_SCALE_WORDS.get((scale_w or "").lower())
    if scale is None:
        return None
    pre = m.group("pre") or ""
    prep = m.group("prep") or ""
    sym = m.group("sym") or ""
    post = m.group("post") or ""
    # Reject bare `(thousand)` singular without context
    if scale_w and scale_w.lower() in {"thousand", "million", "billion"} and not (pre or prep or sym or zeros or post):
        return None
    currency = _currency_from_squashed(f"{pre}{sym}{post}")
    has_ctx = bool(pre or prep or sym or zeros or post or exc_raw)
    exc_sq = re.sub(r"[^a-z%]", "", exc_raw.lower())
    exc_per_share = ("pershare" in exc_sq) or ("perunit" in exc_sq) or ("perads" in exc_sq) or ("eps" in exc_sq)
    exc_par_value = "parvalue" in exc_sq
    exc_pct = ("percent" in exc_sq) or ("%" in exc_sq) or ("ratio" in exc_sq) or ("bps" in exc_sq)
    exc_shares = False
    if "share" in exc_sq:
        without_pershare = exc_sq.replace("pershare", "")
        if "share" in without_pershare:
            exc_shares = True
    return scale, currency, has_ctx, exc_per_share, exc_par_value, exc_pct, exc_shares


def _find_table_scale_headers(text: str) -> list[_TableScaleHeaderSpec]:
    """Find parenthetical and banner table scale declarations in offset order."""
    specs: list[_TableScaleHeaderSpec] = []
    paren_spans: list[tuple[int, int]] = []

    for m in _PAREN_BLOCK_RE.finditer(text):
        inner_raw = m.group(1)
        if "\n\n" in inner_raw or re.search(r"\n[ \t]*\n", inner_raw):
            continue
        parsed = _parse_paren_inner_scale(re.sub(r"\s+", " ", inner_raw))
        if parsed is None:
            continue
        scale, cur, has_ctx, exc_ps, exc_pv, exc_pct, exc_sh = parsed
        # If bare `(millions)` / `(thousands)` with no `in`, currency, or `except`:
        # require table-like header line context (not running prose sentence like "grew by (millions)").
        if not has_ctx:
            ls = text.rfind("\n", 0, m.start()) + 1
            le = text.find("\n", m.end())
            if le < 0:
                le = len(text)
            before_on_line = text[ls : m.start()].strip()
            if before_on_line and before_on_line.lower().endswith(("of", "by", "over", "into", "around", "about")):
                continue
        paren_spans.append((m.start(), m.end()))
        specs.append(
            _TableScaleHeaderSpec(
                scale=scale,
                currency=cur,
                hdr_start=m.start(),
                hdr_end=m.end(),
                span_end=m.end(),
                is_parenthetical=True,
                exclude_per_share=exc_ps,
                exclude_par_value=exc_pv,
                exclude_percentages=exc_pct,
                exclude_shares=exc_sh,
            )
        )

    for m in _BANNER_SCALE_DECL_RE.finditer(text):
        if any(not (m.end() <= ps or m.start() >= pe) for ps, pe in paren_spans):
            continue
        # Ignore non-monetary unit declarations such as "shares in Thousands"
        before_prefix = text[max(0, m.start() - 24) : m.start()]
        if re.search(r"\bshares?\s+$", before_prefix, re.IGNORECASE):
            continue
        raw_scale = (m.group("scale_b") or m.group("scale_i") or "").lower()
        scale = _SCALE_FACTORS.get(raw_scale)
        if scale is None:
            continue
        seg = m.group(0)
        cur = _currency_from_squashed(re.sub(r"[^a-z0-9$€£]", "", seg.lower()))
        exc_ps = exc_pv = exc_pct = exc_sh = False
        if "except" in seg.lower():
            parts = _EXCEPT_SPLIT_RE.split(seg, maxsplit=1)
            exc_sq = re.sub(r"[^a-z%]", "", (parts[1] if len(parts) > 1 else "").lower())
            exc_ps = ("pershare" in exc_sq) or ("perunit" in exc_sq)
            exc_pv = "parvalue" in exc_sq
            exc_pct = ("percent" in exc_sq) or ("%" in exc_sq)
            exc_sh = "share" in exc_sq.replace("pershare", "")
        specs.append(
            _TableScaleHeaderSpec(
                scale=scale,
                currency=cur,
                hdr_start=m.start(),
                hdr_end=m.end(),
                span_end=m.end(),
                is_parenthetical=False,
                exclude_per_share=exc_ps,
                exclude_par_value=exc_pv,
                exclude_percentages=exc_pct,
                exclude_shares=exc_sh,
            )
        )

    specs.sort(key=lambda sp: sp.hdr_start)
    return specs


_SPANISH_CONTEXT_RE = re.compile(
    r"(?:\b(?:el|la|los|las|del|al|es|será|sera|por\s+ciento|euros?|d[ií]as?|a[ñn]os?|"
    r"meses|cl[áa]usula|primera|segunda|tercera|cuarta|precio|importe|comisi[óo]n|inter[ée]s|pr[ée]stamo)\b|€)",
    re.IGNORECASE,
)


_SECTION_HEADER_ONLY_RE = re.compile(
    r"^\s*(?:section|sec\.?|article|art\.?|clause|cl[áa]usula|art[íi]culo|part|item|schedule|exhibit|annex)\s+"
    r"[a-z0-9\-]+(?:\.[a-z0-9\-]+)*(?:\([a-z0-9]+\))*(?:\s*(?::|—|\.\s+|\s+[-–]\s+)\s*[a-z0-9\s,&()\-]{1,60})?\.?\s*$",
    re.IGNORECASE,
)


_ABBREV_BEFORE_DOT_RE = re.compile(
    r"(?:(?:[a-z]\.){2,}|\b(?:sec|art|no|inc|corp|ltd|llc|co|plc|v|vs|cf|al|jr|sr|dr|mr|mrs|ms|st|p|pp|fig|eq|ch|vol|approx|est|misc|ref|jan|feb|mar|apr|aug|sept?|oct|nov|dec)\.$)",
    re.IGNORECASE,
)


_LEXICAL_HYPHEN_LEFT_PREFIXES = frozenset(
    {
        "non",
        "self",
        "co",
        "semi",
        "multi",
        "cross",
        "third",
        "first",
        "second",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "twelve",
        "fifteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "royalty",
        "arm",
        "day",
        "year",
        "month",
        "full",
        "half",
        "long",
        "short",
        "high",
        "low",
        "well",
        "pre",
        "post",
        "up",
        "all",
    }
)


_LEXICAL_HYPHEN_RIGHT_HEADS = frozenset(
    {
        "exclusive",
        "disclosure",
        "compete",
        "refundable",
        "transferable",
        "payment",
        "free",
        "party",
        "parties",
        "year",
        "years",
        "month",
        "months",
        "day",
        "days",
        "term",
        "based",
        "rate",
        "owning",
        "ownership",
        "executing",
        "contained",
        "existing",
        "termination",
        "border",
        "length",
        "time",
        "price",
        "scale",
        "risk",
        "tier",
        "level",
        "stage",
        "hand",
        "front",
        "end",
        "wise",
        "wide",
        "in",
        "to",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    }
)


_EXPLICIT_PAGE_MARKER_RE = re.compile(
    r"^(?:"
    r"(?:page|p\.?|pp\.?|p[aá]g(?:ina)?\.?)\s+(?:\d{1,4}|[ivxlcdm]{1,6})"
    r"(?:\s*(?:of|de|/)\s*(?:\d{1,4}|[ivxlcdm]{1,6}))?\.?"
    r"|[-–—]\s*\d{1,4}\s*[-–—]"
    r"|\d{1,4}\s+(?:of|de|/)\s+\d{1,4}"
    r")$",
    re.IGNORECASE,
)


_BARE_PAGE_NUMBER_RE = re.compile(r"^(\d{1,4})$")


_OUTLINE_ITEM_PREFIX_RE = re.compile(
    r"^(?:\(\s*[a-z0-9]{1,4}\s*\)|[a-z]\)|[0-9]+(?:\.[0-9]+)*\.?\s+)",
    re.IGNORECASE,
)


def _collect_hyphen_lexical_sets(texts: list[str] | tuple[str, ...]) -> tuple[frozenset[tuple[str, str]], frozenset[str]]:
    """Collect inline hyphenated pairs and single joined words across one or more texts."""
    hyphenated: set[tuple[str, str]] = set()
    joined: set[str] = set()
    for raw in texts:
        if not raw:
            continue
        cleaned = raw.translate(_QUOTE_MAP)
        # Skip line-break hyphens when discovering genuine inline lexical hyphens (include accented letters & digits).
        for m in re.finditer(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ0-9])([A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,})-([A-Za-zÀ-ÖØ-öø-ÿ0-9]{2,})(?![A-Za-zÀ-ÖØ-öø-ÿ0-9])", cleaned):
            between = cleaned[m.end(1) : m.start(2)]
            if "\n" not in between and "\r" not in between:
                hyphenated.add((m.group(1).lower(), m.group(2).lower()))
        # Standalone inline unhyphenated words (exclude sides adjacent to a hyphen).
        for m in re.finditer(r"(?<![A-Za-zÀ-ÖØ-öø-ÿ-])([A-Za-zÀ-ÖØ-öø-ÿ]{4,})(?![A-Za-zÀ-ÖØ-öø-ÿ-])", cleaned):
            joined.add(m.group(1).lower())
    return frozenset(hyphenated), frozenset(joined)


def _should_preserve_linebreak_hyphen(
    left_word: str,
    right_word: str,
    known_hyphenated: frozenset[tuple[str, str]],
    known_joined: frozenset[str],
) -> bool:
    """Return True if a hyphen across a line break represents a genuine compound hyphen."""
    ll = left_word.lower()
    rl = right_word.lower()
    if (ll, rl) in known_hyphenated and (ll + rl) not in known_joined:
        return True
    if (ll + rl) in known_joined and (ll, rl) not in known_hyphenated:
        return False
    if (ll, rl) in known_hyphenated:
        return True
    if ll in {"twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"} and rl in {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    }:
        return True
    if ll in _LEXICAL_HYPHEN_LEFT_PREFIXES and rl in _LEXICAL_HYPHEN_RIGHT_HEADS:
        return True
    return False


def normalize_span(
    text: str,
    *,
    lexical_texts: tuple[str, ...] | None = None,
) -> str:
    """Normalize punctuation, ligatures, soft hyphens, spacing, and line-break hyphens deterministically."""
    if not text:
        return ""
    pre_text = re.sub(r"\u00ad(?=[ \t]*\r?\n)", "-", text)
    pre_text = pre_text.replace("\r\n", "\n").replace("\r", "\n").translate(_QUOTE_MAP)
    vocab_inputs = (
        tuple(re.sub(r"\u00ad(?=[ \t]*\r?\n)", "-", t).translate(_QUOTE_MAP) for t in lexical_texts)
        if lexical_texts is not None
        else (pre_text,)
    )
    known_hyphenated, known_joined = _collect_hyphen_lexical_sets(vocab_inputs)

    def _replace_lb(m: re.Match[str]) -> str:
        # Preserve hyphen directly when either side is a digit ("30-\nday", "2024-\n2025").
        if not (m.group(1).isalpha() and m.group(2).isalpha()):
            return f"{m.group(1)}-{m.group(2)}"
        # Expand to full adjacent ASCII word tokens around the hyphen for lexical check.
        start = m.start()
        end = m.end()
        while start > 0 and pre_text[start - 1].isalpha():
            start -= 1
        while end < len(pre_text) and pre_text[end].isalpha():
            end += 1
        seg = pre_text[start:end]
        lm = re.match(r"([A-Za-zÀ-ÖØ-öø-ÿ]+)", seg)
        rm = re.search(r"([A-Za-zÀ-ÖØ-öø-ÿ]+)$", seg)
        if lm and rm and _should_preserve_linebreak_hyphen(lm.group(1), rm.group(1), known_hyphenated, known_joined):
            return f"{m.group(1)}-{m.group(2)}"
        return f"{m.group(1)}{m.group(2)}"

    dehyphenated = _LINEBREAK_HYPHEN_RE.sub(_replace_lb, pre_text)
    cleaned = _ZERO_WIDTH_RE.sub("", dehyphenated)
    nfkc = unicodedata.normalize("NFKC", cleaned)
    translated = nfkc.translate(_QUOTE_MAP).replace("\\$", "$")
    collapsed = re.sub(r"\s+", " ", translated).strip()
    return _canonicalize_punctuation_and_numbers(collapsed)


def _split_raw_furniture_lines(text: str) -> list[dict[str, object]]:
    """Split *text* into line records over CRLF, LF, and form-feed preserving exact byte/char offsets."""
    lines: list[dict[str, object]] = []
    n = len(text)
    pos = 0
    while pos < n:
        line_start = pos
        while pos < n and text[pos] not in ("\n", "\r", "\f"):
            pos += 1
        content_end = pos
        delim = ""
        if pos < n:
            if text[pos] == "\r" and pos + 1 < n and text[pos + 1] == "\n":
                delim = "\r\n"
                pos += 2
            else:
                delim = text[pos]
                pos += 1
        raw_content = text[line_start:content_end]
        stripped = raw_content.strip()
        lines.append(
            {
                "start": line_start,
                "content_end": content_end,
                "end": pos,
                "content": raw_content,
                "stripped": stripped,
                "delim": delim,
                "is_blank": stripped == "",
            }
        )
    return lines


def _detect_candidate_furniture_indices(lines: list[dict[str, object]]) -> dict[int, str]:
    """Identify candidate structural PDF running headers/footers and page numbers per line index."""
    num_lines = len(lines)
    if num_lines == 0:
        return {}

    def _is_boundary_gap(idx: int, direction: int) -> tuple[bool, bool]:
        """Return `(has_blank_or_doc_edge, adjacent_to_ff)`."""
        adj_ff = (lines[idx]["delim"] == "\f") if direction > 0 else (idx > 0 and lines[idx - 1]["delim"] == "\f")
        if direction < 0:
            if idx == 0:
                return True, adj_ff
            return bool(lines[idx - 1]["is_blank"]) or lines[idx - 1]["delim"] == "\f", adj_ff
        if idx == num_lines - 1:
            return True, adj_ff
        return bool(lines[idx + 1]["is_blank"]) or lines[idx]["delim"] == "\f", adj_ff

    banner_groups: dict[str, list[int]] = {}
    bare_num_candidates: list[tuple[int, int]] = []
    strip_reasons: dict[int, str] = {}

    for idx, rec in enumerate(lines):
        stripped = str(rec["stripped"])
        if not stripped:
            continue
        prev_gap, prev_ff = _is_boundary_gap(idx, -1)
        next_gap, next_ff = _is_boundary_gap(idx, +1)
        isolated = (prev_gap and next_gap) or prev_ff or next_ff

        # 1) Explicit structural page markers (`Page 2 of 6`, `- 3 -`, `3 / 6`, `p. 4`) on their own line.
        if len(stripped) <= 36 and _EXPLICIT_PAGE_MARKER_RE.match(stripped):
            strip_reasons[idx] = "page_marker"
            continue

        # 2) Bare standalone integers: collect for ascending sequence check across page breaks.
        bare_m = _BARE_PAGE_NUMBER_RE.match(stripped)
        if bare_m:
            val = int(bare_m.group(1))
            if 1 <= val <= 999:
                bare_num_candidates.append((idx, val))
            continue

        # 3) Repeating running banners/footers.
        if len(stripped) > 120 or len(stripped.split()) > 14:
            continue
        if _SECTION_HEADER_ONLY_RE.match(stripped) or _OUTLINE_ITEM_PREFIX_RE.match(stripped):
            continue
        if re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]-$", stripped):
            continue
        # Do not treat a prose wrap continuation whose previous line ended mid-word ("word-") or mid-sentence as PDF furniture unless bounded by \f.
        if idx > 0 and not prev_ff and not next_ff:
            prev_s = str(lines[idx - 1]["stripped"])
            if prev_s and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]-$", prev_s):
                continue
        if stripped.startswith("|") and re.search(r"\|\s*[$€£]?\d", stripped):
            continue
        words = stripped.split()
        has_explicit_banner_syntax = bool(
            "|" in stripped
            or " — " in stripped
            or " - " in stripped
            or " -- " in stripped
            or "©" in stripped
            or stripped.isupper()
            or re.search(r"\bpage\b", stripped, re.IGNORECASE)
        )
        # Short repeated lines without explicit banner syntax must be visually isolated (blank lines / form-feed)
        # and free of inline currency figures so repeating clause fragments at narrow wrap widths are never stripped.
        has_currency_figure = bool(re.search(r"[$€£]\s*\d", stripped))
        short_isolated_header = bool(
            isolated and len(words) <= 6 and not stripped.endswith(".") and not has_currency_figure
        )
        if not (has_explicit_banner_syntax or short_isolated_header or (isolated and not has_currency_figure)):
            continue
        skeleton = re.sub(r"\s+", " ", re.sub(r"\d+", "#", stripped.lower())).strip()
        if len(re.sub(r"[^a-z#]", "", skeleton)) < 3:
            continue
        banner_groups.setdefault(skeleton, []).append(idx)

    # Mark repeating running headers/footers spaced across the document first so adjacent page numbers are visible.
    for skeleton, idxs in banner_groups.items():
        if len(idxs) < 2:
            continue
        spaced_idxs: list[int] = []
        for k, idx in enumerate(idxs):
            prev_ok = False
            next_ok = False
            if k > 0:
                prev_idx = idxs[k - 1]
                dist = int(lines[idx]["start"]) - int(lines[prev_idx]["end"])
                has_ff = any(lines[m]["delim"] == "\f" for m in range(prev_idx, idx))
                has_para = any(lines[m]["is_blank"] for m in range(prev_idx + 1, idx))
                prev_ok = dist >= 140 or has_ff or (dist >= 70 and has_para)
            if k + 1 < len(idxs):
                nxt_idx = idxs[k + 1]
                dist = int(lines[nxt_idx]["start"]) - int(lines[idx]["end"])
                has_ff = any(lines[m]["delim"] == "\f" for m in range(idx, nxt_idx))
                has_para = any(lines[m]["is_blank"] for m in range(idx + 1, nxt_idx))
                next_ok = dist >= 140 or has_ff or (dist >= 70 and has_para)
            if prev_ok or next_ok:
                spaced_idxs.append(idx)
        if len(spaced_idxs) < 2:
            continue
        sample_stripped = str(lines[spaced_idxs[0]]["stripped"])
        has_marker_or_sep = bool(
            "#" in skeleton or "|" in sample_stripped or "—" in sample_stripped or "©" in sample_stripped
        )
        words_cnt = len(sample_stripped.split())
        if (
            sample_stripped.endswith(".")
            and not has_marker_or_sep
            and words_cnt >= 6
            and len(spaced_idxs) < 3
        ):
            continue
        for idx in spaced_idxs:
            strip_reasons[idx] = "repeating_banner"

    # Mark ascending standalone page number series across page-scale spacing (isolated by gap, \f, or adjacent banner).
    edge_bare_cands: list[tuple[int, int]] = []
    for idx, val in bare_num_candidates:
        prev_gap, prev_ff = _is_boundary_gap(idx, -1)
        next_gap, next_ff = _is_boundary_gap(idx, +1)
        adj_stripped = (idx - 1 in strip_reasons) or (idx + 1 in strip_reasons)
        if (prev_gap and next_gap) or prev_ff or next_ff or adj_stripped:
            edge_bare_cands.append((idx, val))
    if len(edge_bare_cands) >= 2:
        in_seq: set[int] = set()
        for k in range(1, len(edge_bare_cands)):
            i1, v1 = edge_bare_cands[k - 1]
            i2, v2 = edge_bare_cands[k]
            char_dist = int(lines[i2]["start"]) - int(lines[i1]["end"])
            has_ff_between = any(lines[m]["delim"] == "\f" for m in range(i1, i2))
            has_para = any(lines[m]["is_blank"] for m in range(i1 + 1, i2))
            if 1 <= (v2 - v1) <= 3 and (char_dist >= 120 or has_ff_between or (char_dist >= 60 and has_para)):
                in_seq.add(i1)
                in_seq.add(i2)
        for idx in in_seq:
            strip_reasons[idx] = "sequential_page_number"

    return strip_reasons


def _render_filtered_lines_with_map(
    text: str,
    lines: list[dict[str, object]],
    strip_reasons: dict[int, str],
    *,
    base_offset: int = 0,
) -> tuple[str, list[int], list[int], tuple[dict[str, object], ...]]:
    n = len(text)
    num_lines = len(lines)
    if not strip_reasons:
        if "\f" not in text:
            starts = [base_offset + k for k in range(n)]
            ends = [base_offset + k + 1 for k in range(n)]
            return text, starts, ends, ()

    out_chars: list[str] = []
    out_starts: list[int] = []
    out_ends: list[int] = []
    excluded: list[dict[str, object]] = []

    idx = 0
    while idx < num_lines:
        if idx in strip_reasons:
            rec = lines[idx]
            s_off = base_offset + int(rec["start"])
            e_off = base_offset + int(rec["content_end"])
            excluded.append(
                {
                    "text": str(rec["stripped"]),
                    "span": [s_off, e_off],
                    "kind": strip_reasons[idx],
                }
            )
            while out_chars and out_chars[-1] == "\n":
                if len(out_chars) >= 2 and out_chars[-2] == "\n":
                    out_chars.pop()
                    out_starts.pop()
                    out_ends.pop()
                else:
                    break
            idx += 1
            while idx < num_lines and lines[idx]["is_blank"]:
                idx += 1
            continue

        rec = lines[idx]
        l_start = int(rec["start"])
        l_cend = int(rec["content_end"])
        for off in range(l_start, l_cend):
            out_chars.append(text[off])
            out_starts.append(base_offset + off)
            out_ends.append(base_offset + off + 1)
        delim = str(rec["delim"])
        if delim:
            mapped_delim = "\n" if delim in ("\n", "\r", "\f", "\r\n") else delim
            d_start = base_offset + l_cend
            d_end = base_offset + int(rec["end"])
            if not (mapped_delim == "\n" and excluded and len(out_chars) >= 2 and out_chars[-1] == "\n" and out_chars[-2] == "\n"):
                out_chars.append(mapped_delim)
                out_starts.append(d_start)
                out_ends.append(d_end)
        idx += 1

    return "".join(out_chars), out_starts, out_ends, tuple(excluded)


def _strip_pdf_page_furniture_with_map(
    text: str,
    *,
    base_offset: int = 0,
) -> tuple[str, list[int], list[int], tuple[dict[str, object], ...]]:
    """Structurally detect and exclude running PDF headers/footers and sequential pagination.

    Never drops an isolated single-occurrence short clause or standalone figure (negative control).
    Returns `(cleaned_text, clean_char_orig_starts, clean_char_orig_ends, excluded_records)`.
    """
    if not text:
        return "", [], [], ()
    lines = _split_raw_furniture_lines(text)
    strip_reasons = _detect_candidate_furniture_indices(lines)
    return _render_filtered_lines_with_map(text, lines, strip_reasons, base_offset=base_offset)


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
    "\u2010": "-",
    "\u2011": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
}


_OCR_STRIP_CHARS = frozenset({"\u00ad", "\u200b", "\u200c", "\u200d", "\ufeff"})


def _stage2_canonicalize_punct_with_map(
    stage2: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """Normalize punctuation/number/currency spacing on char-mapped *stage2* preserving exact source spans."""
    if not stage2:
        return []
    out: list[tuple[str, int, int]] = []
    n = len(stage2)
    i = 0
    while i < n:
        ch, os, oe = stage2[i]
        if ch in (",", ";", ":", ")", "]", "%"):
            while out and out[-1][0] in (" ", "\t"):
                out.pop()
            # Drop spaces after ',' inside comma thousands ("5, 000" -> "5,000" when preceding digits & currency or space before comma)
            out.append((ch, os, oe))
            if ch == "," and len(out) >= 2 and out[-2][0].isdigit():
                # Check if previous char before comma was digit and next digits are exact 3-digit thousands group with space after ','
                k = i + 1
                sp_cnt = 0
                while k < n and stage2[k][0] in (" ", "\t"):
                    sp_cnt += 1
                    k += 1
                if sp_cnt > 0 and k + 2 < n and stage2[k][0].isdigit() and stage2[k + 1][0].isdigit() and stage2[k + 2][0].isdigit():
                    if k + 3 >= n or not stage2[k + 3][0].isdigit():
                        # Only collapse if there was a space BEFORE ',' in original or a currency symbol precedes the number
                        prev_was_space = (i > 0 and stage2[i - 1][0] in (" ", "\t"))
                        j_back = len(out) - 2
                        while j_back >= 0 and (out[j_back][0].isdigit() or out[j_back][0] in (",", " ", "\t")):
                            j_back -= 1
                        has_cur_prefix = j_back >= 0 and out[j_back][0] in ("$", "€", "£")
                        if prev_was_space or has_cur_prefix:
                            i = k
                            continue
            if ch == "," and i + 1 < n and stage2[i + 1][0] in (" ", "\t"):
                pass
            i += 1
            continue

        if ch in ("(", "["):
            out.append((ch, os, oe))
            i += 1
            while i < n and stage2[i][0] in (" ", "\t"):
                i += 1
            continue

        if ch in ("$", "€", "£"):
            out.append((ch, os, oe))
            k = i + 1
            while k < n and stage2[k][0] in (" ", "\t"):
                k += 1
            if k > i + 1 and k < n and stage2[k][0].isdigit():
                i = k
                continue
            i += 1
            continue

        if ch == "." and len(stage2[i - 1 : i]) == 1 and i > 0 and stage2[i - 1][0] in (" ", "\t"):
            # Collapse "5 . 25" -> "5.25" when digit precedes whitespace before dot AND digit follows optional space
            j_prev = len(out) - 1
            while j_prev >= 0 and out[j_prev][0] in (" ", "\t"):
                j_prev -= 1
            k_next = i + 1
            while k_next < n and stage2[k_next][0] in (" ", "\t"):
                k_next += 1
            if j_prev >= 0 and out[j_prev][0].isdigit() and k_next < n and stage2[k_next][0].isdigit():
                while out and out[-1][0] in (" ", "\t"):
                    out.pop()
                out.append((".", os, oe))
                i = k_next
                continue

        out.append((ch, os, oe))
        i += 1
    return out


def _normalize_pdf_ocr_with_map(
    text: str,
    *,
    base_offset: int = 0,
    apply_furniture_filter: bool = True,
    known_hyphenated: frozenset[tuple[str, str]] | None = None,
    known_joined: frozenset[str] | None = None,
    furniture_out: list[dict[str, object]] | None = None,
    excluded_line_override: dict[int, str] | None = None,
) -> tuple[str, list[int], list[int]]:
    """Return `(normalized_text, norm_char_to_orig_start, norm_char_to_orig_end)`."""
    if not text:
        return "", [], []

    if known_hyphenated is None or known_joined is None:
        self_hyph, self_joined = _collect_hyphen_lexical_sets((text,))
        eff_hyphenated = self_hyph if known_hyphenated is None else known_hyphenated
        eff_joined = self_joined if known_joined is None else known_joined
    else:
        eff_hyphenated = known_hyphenated
        eff_joined = known_joined

    if apply_furniture_filter:
        if excluded_line_override is not None:
            raw_lines = _split_raw_furniture_lines(text)
            cleaned_text, clean_starts, clean_ends, excluded_items = _render_filtered_lines_with_map(
                text, raw_lines, excluded_line_override, base_offset=base_offset
            )
        else:
            cleaned_text, clean_starts, clean_ends, excluded_items = _strip_pdf_page_furniture_with_map(
                text, base_offset=base_offset
            )
        if furniture_out is not None and excluded_items:
            furniture_out.extend(excluded_items)
    else:
        cleaned_text = text
        clean_starts = [base_offset + k for k in range(len(text))]
        clean_ends = [base_offset + k + 1 for k in range(len(text))]

    if not cleaned_text:
        return "", [], []

    # Step 1: per-character ligatures, typographic punctuation, soft hyphens, CRLF -> LF
    stage1: list[tuple[str, int, int]] = []
    n = len(cleaned_text)
    i = 0
    while i < n:
        ch = cleaned_text[i]
        orig_s = clean_starts[i]
        if ch == "\r":
            has_lf = i + 1 < n and cleaned_text[i + 1] == "\n"
            orig_e = clean_ends[i + 1] if has_lf else clean_ends[i]
            stage1.append(("\n", orig_s, orig_e))
            i += 2 if has_lf else 1
            continue
        orig_e = clean_ends[i]
        i += 1
        if ch == "\u00ad":
            # Soft hyphen before a line break represents a hyphenated line wrap and must pass
            # through line-break hyphen reconstruction rather than collapsing into a space.
            k_lb = i
            while k_lb < n and cleaned_text[k_lb] in (" ", "\t"):
                k_lb += 1
            if k_lb < n and cleaned_text[k_lb] in ("\r", "\n"):
                stage1.append(("-", orig_s, orig_e))
            continue
        if ch in _OCR_STRIP_CHARS:
            continue
        if ch in _OCR_LIGATURE_MAP:
            for rep_ch in _OCR_LIGATURE_MAP[ch]:
                stage1.append((rep_ch, orig_s, orig_e))
            continue
        if ch == "\u2014":
            prev_al = bool(stage1 and stage1[-1][0].isalnum())
            next_al = bool(i < n and cleaned_text[i].isalnum())
            if prev_al and next_al:
                for rep_ch in (" ", "-", " "):
                    stage1.append((rep_ch, orig_s, orig_e))
            else:
                stage1.append(("-", orig_s, orig_e))
            continue
        if ch in _OCR_PUNCT_MAP:
            stage1.append((_OCR_PUNCT_MAP[ch], orig_s, orig_e))
            continue
        nfkc = unicodedata.normalize("NFKC", ch)
        for rep_ch in nfkc:
            stage1.append((rep_ch, orig_s, orig_e))

    # Step 2: reconstruct line-break hyphens (preserving lexical compounds) and unwrap paragraphs
    stage2: list[tuple[str, int, int]] = []
    m = len(stage1)
    j = 0
    while j < m:
        ch, orig_s, orig_e = stage1[j]
        if ch == "-" and stage2 and stage2[-1][0].isalnum():
            k = j + 1
            while k < m and stage1[k][0] in (" ", "\t"):
                k += 1
            if k < m and stage1[k][0] == "\n":
                k += 1
                while k < m and stage1[k][0] in (" ", "\t"):
                    k += 1
                if k < m and stage1[k][0].isalnum():
                    left_prev_alpha = stage2[-1][0].isalpha()
                    next_ch = stage1[k][0]
                    right_next_alpha = next_ch.isalpha()
                    # Numeric or alphanumeric line-break hyphens ("30-\nday", "2024-\n2025", "77aaa-\n77bbbb")
                    # always preserve the hyphen directly with no inserted space.
                    if not (left_prev_alpha and right_next_alpha):
                        stage2.append(("-", orig_s, orig_e))
                        j = k
                        continue
                    l_idx = len(stage2) - 1
                    l_chars: list[str] = []
                    while l_idx >= 0 and stage2[l_idx][0].isalpha():
                        l_chars.append(stage2[l_idx][0])
                        l_idx -= 1
                    left_word = "".join(reversed(l_chars))
                    r_idx = k
                    r_chars: list[str] = []
                    while r_idx < m and stage1[r_idx][0].isalpha():
                        r_chars.append(stage1[r_idx][0])
                        r_idx += 1
                    right_word = "".join(r_chars)
                    all_caps_wrap = (
                        next_ch.isupper()
                        and len(stage2) >= 2
                        and stage2[-1][0].isupper()
                        and stage2[-2][0].isupper()
                    )
                    title_compound_wrap = (
                        next_ch.isupper()
                        and bool(left_word)
                        and (
                            (left_word.lower(), right_word.lower()) in eff_hyphenated
                            or left_word.lower() in _LEXICAL_HYPHEN_LEFT_PREFIXES
                            or (
                                left_word[0].isupper()
                                and any(c.islower() for c in left_word[1:])
                                and (left_word + right_word).lower() not in eff_joined
                            )
                        )
                    )
                    if next_ch.islower() or all_caps_wrap or title_compound_wrap:
                        if _should_preserve_linebreak_hyphen(left_word, right_word, eff_hyphenated, eff_joined):
                            stage2.append(("-", orig_s, orig_e))
                        elif title_compound_wrap and (left_word + right_word).lower() not in eff_joined:
                            stage2.append(("-", orig_s, orig_e))
                        j = k
                        continue

        if ch == "\n" and stage2:
            k = j + 1
            nl_count = 1
            while k < m and stage1[k][0] in (" ", "\t", "\n"):
                if stage1[k][0] == "\n":
                    nl_count += 1
                k += 1
            prev_line_chars: list[str] = []
            idx_p = len(stage2) - 1
            while idx_p >= 0 and stage2[idx_p][0] != "\n":
                prev_line_chars.append(stage2[idx_p][0])
                idx_p -= 1
            prev_line = "".join(reversed(prev_line_chars)).strip()
            ends_sentence = bool(prev_line) and (
                prev_line[-1] in ":;!?"
                or (prev_line[-1] == "." and _is_sentence_boundary_dot(prev_line, len(prev_line) - 1))
            )
            if (
                prev_line
                and not ends_sentence
                and "|" not in prev_line
                and k < m
            ):
                next_line_chars: list[str] = []
                idx_n = k
                while idx_n < m and stage1[idx_n][0] != "\n" and len(next_line_chars) < 24:
                    next_line_chars.append(stage1[idx_n][0])
                    idx_n += 1
                next_prefix = "".join(next_line_chars).lstrip()
                is_structural_next = bool(
                    (
                        next_prefix.startswith(("|", "- ", "* ", "•"))
                        or re.match(r"^(?:\(\d+\)|\(\w\)|\d+\.\s)", next_prefix)
                    )
                    and not re.search(
                        r"\b(?:section|sec\.?|article|art\.?|clause|part|item|schedule|exhibit|annex|cl[aá]usula|art[ií]culo|no\.?)$",
                        prev_line,
                        re.IGNORECASE,
                    )
                )
                is_email_header_prev = bool(
                    prev_line.endswith(">")
                    or re.match(r"^(?:From|To|Cc|Bcc|Subject|Date|Sent|Reply-To)\s*:", prev_line, re.IGNORECASE)
                )
                header_only_prev = _SECTION_HEADER_ONLY_RE.match(prev_line) is not None and not (
                    next_prefix and next_prefix[0].islower()
                )
                allow_blank_unwrap = nl_count == 1 or (nl_count <= 2 and bool(next_prefix) and next_prefix[0].islower())
                if (
                    allow_blank_unwrap
                    and not is_email_header_prev
                    and not header_only_prev
                    and not is_structural_next
                    and next_prefix
                    and (next_prefix[0].islower() or re.match(r"^(?:[A-Za-z0-9$\"'(])", next_prefix))
                ):
                    if stage2[-1][0] != " ":
                        stage2.append((" ", orig_s, orig_e))
                    j = k
                    continue

        stage2.append((ch, orig_s, orig_e))
        j += 1

    stage3 = _stage2_canonicalize_punct_with_map(stage2)
    norm_text = "".join(c for c, _, _ in stage3)
    orig_starts = [s for _, s, _ in stage3]
    orig_ends = [e for _, _, e in stage3]
    return norm_text, orig_starts, orig_ends


def _count_phrase_in_norm_body(key_text: str, body_text: str) -> int:
    if not key_text or not body_text:
        return 0
    return len(re.findall(r"(?<![a-z0-9])" + re.escape(key_text) + r"(?![a-z0-9])", body_text, re.IGNORECASE))


_HEADING_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "y", "de", "del", "la", "el", "los", "las"}
)


_HEADING_MARKER_PREFIX_RE = re.compile(
    r"^(?P<lead>[ \t]*(?:#{1,6}[ \t]*)?)(?P<marker>(?:\d+(?:\.\d+)*\.?[ \t]+)?[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ \t&/\-]{2,48}?)(?P<colon>[ \t]*:)(?=[ \t]+\S|[ \t]*$)"
)


def _is_typographic_heading_text(text: str) -> bool:
    """True iff *text* is set as a heading: ALL CAPS, or Title Case on every significant word."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    if all(c.isupper() for c in letters):
        return True
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]*", text)
    significant = [w for w in words if w.lower() not in _HEADING_SMALL_WORDS]
    return bool(significant) and all(w[0].isupper() for w in significant)


def _heading_candidates(text: str) -> list[tuple[int, int, str, str]]:
    """Return `(start, end, case_insensitive_key, raw_form)` for heading lines / heading-marker prefixes.

    Only typographic headings qualify (short line or `MARKER:` prefix, starting with a letter/digit, set
    in ALL CAPS or Title Case). Body prose lines never qualify, so body casing is never folded.
    """
    out: list[tuple[int, int, str, str]] = []
    pos = 0
    for raw_ln in text.splitlines(keepends=True):
        line = raw_ln.rstrip("\r\n")
        ln_start = pos
        pos += len(raw_ln)
        stripped = line.strip()
        if not stripped:
            continue
        lead = len(line) - len(line.lstrip())
        body = stripped.lstrip("#").strip()
        first = body[:1]
        if not first or not (first.isalpha() or first.isdigit() or first == "§"):
            continue
        whole_ok = (
            len(stripped) <= 80
            and len(stripped.split()) <= 12
            and not stripped.endswith((".", ",", ";"))
            and _is_typographic_heading_text(re.sub(r"^[\d.\s§]+", "", body))
        )
        if whole_ok:
            s = ln_start + lead
            e = ln_start + lead + len(stripped)
            key = re.sub(r"\s+", " ", stripped.lower()).rstrip(" :")
            out.append((s, e, key, stripped))
            continue
        m = _HEADING_MARKER_PREFIX_RE.match(line)
        if m:
            marker = m.group("marker").strip()
            if len(marker.split()) <= 5 and _is_typographic_heading_text(re.sub(r"^[\d.\s]+", "", marker)):
                s = ln_start + m.start("marker")
                e = ln_start + m.end("marker")
                key = re.sub(r"\s+", " ", marker.lower())
                out.append((s, e, key, marker))
    return out


def _upper_same_length(text: str) -> str:
    return "".join(c.upper() if len(c.upper()) == 1 else c for c in text)


def joint_fold_heading_case(primary_text: str, secondary_text: str) -> tuple[str, str, list[dict[str, str]]]:
    """Decide heading casing jointly for both versions (length-preserving, offsets unchanged).

    Rule: a heading line or `MARKER:` prefix whose text matches a heading in the other version
    case-insensitively but with a different casing (e.g. ``WITNESSETH:`` vs ``Witnesseth:``,
    ``1. COVENANTS`` vs ``1. Covenants``) is rendered in upper case in BOTH versions, so heading
    recognition and section alignment are identical on both sides and a heading-only case change is
    not a divergence. Body prose is never folded: a defined-term case change inside a clause
    (``"Affiliate"`` vs ``"affiliate"``) remains a detected textual divergence.
    """
    if not primary_text or not secondary_text:
        return primary_text, secondary_text, []
    p_c = _heading_candidates(primary_text)
    s_c = _heading_candidates(secondary_text)
    forms: dict[str, set[str]] = {}
    p_keys = {k for _, _, k, _ in p_c}
    s_keys = {k for _, _, k, _ in s_c}
    for _, _, k, raw in (*p_c, *s_c):
        forms.setdefault(k, set()).add(re.sub(r"\s+", " ", raw).rstrip(" :"))
    differing = {k for k in (p_keys & s_keys) if len(forms.get(k, ())) > 1}
    if not differing:
        return primary_text, secondary_text, []

    def _apply(text: str, cands: list[tuple[int, int, str, str]]) -> str:
        chars = list(text)
        for s, e, k, _ in cands:
            if k in differing:
                seg = _upper_same_length(text[s:e])
                chars[s:e] = list(seg)
        return "".join(chars)

    prov = [{"rule": "header_case_folding", "from": " | ".join(sorted(forms[k])), "to": k.upper()} for k in sorted(differing)]
    return _apply(primary_text, p_c), _apply(secondary_text, s_c), prov


def _normalize_pair_with_maps(
    primary_text: str,
    secondary_text: str,
    *,
    primary_base_offset: int = 0,
    secondary_base_offset: int = 0,
    apply_furniture_filter: bool = True,
) -> tuple[
    tuple[str, list[int], list[int], tuple[dict[str, object], ...]],
    tuple[str, list[int], list[int], tuple[dict[str, object], ...]],
]:
    """Normalize a primary/secondary text pair with symmetric furniture exclusion and shared hyphen vocab."""
    # Joint, length-preserving heading-case decision (offsets into the raw texts remain valid).
    primary_text, secondary_text, _ = joint_fold_heading_case(primary_text, secondary_text)
    known_hyph, known_joined = _collect_hyphen_lexical_sets((primary_text, secondary_text))
    p_override: dict[int, str] | None = None
    s_override: dict[int, str] | None = None

    if apply_furniture_filter and primary_text and secondary_text:
        p_lines = _split_raw_furniture_lines(primary_text)
        s_lines = _split_raw_furniture_lines(secondary_text)
        p_strip = _detect_candidate_furniture_indices(p_lines)
        s_strip = _detect_candidate_furniture_indices(s_lines)

        if p_strip or s_strip:
            for _ in range(3):
                changed = False
                p_ret = _canonicalize_layout_text(
                    " ".join(str(r["content"]) for i, r in enumerate(p_lines) if i not in p_strip)
                )
                s_ret = _canonicalize_layout_text(
                    " ".join(str(r["content"]) for i, r in enumerate(s_lines) if i not in s_strip)
                )
                for idx in sorted(list(p_strip)):
                    cand = _canonicalize_layout_text(str(p_lines[idx]["stripped"]))
                    if not cand:
                        continue
                    c_s = _count_phrase_in_norm_body(cand, s_ret)
                    c_p = _count_phrase_in_norm_body(cand, p_ret)
                    if c_s > c_p:
                        del p_strip[idx]
                        changed = True
                for idx in sorted(list(s_strip)):
                    cand = _canonicalize_layout_text(str(s_lines[idx]["stripped"]))
                    if not cand:
                        continue
                    c_p = _count_phrase_in_norm_body(cand, p_ret)
                    c_s = _count_phrase_in_norm_body(cand, s_ret)
                    if c_p > c_s:
                        del s_strip[idx]
                        changed = True
                if not changed:
                    break
            p_override = p_strip
            s_override = s_strip

    p_furn: list[dict[str, object]] = []
    s_furn: list[dict[str, object]] = []
    p_norm, p_s, p_e = _normalize_pdf_ocr_with_map(
        primary_text,
        base_offset=primary_base_offset,
        apply_furniture_filter=apply_furniture_filter,
        known_hyphenated=known_hyph,
        known_joined=known_joined,
        furniture_out=p_furn,
        excluded_line_override=p_override,
    )
    s_norm, s_s, s_e = _normalize_pdf_ocr_with_map(
        secondary_text,
        base_offset=secondary_base_offset,
        apply_furniture_filter=apply_furniture_filter,
        known_hyphenated=known_hyph,
        known_joined=known_joined,
        furniture_out=s_furn,
        excluded_line_override=s_override,
    )
    return (p_norm, p_s, p_e, tuple(p_furn)), (s_norm, s_s, s_e, tuple(s_furn))


def normalize_pdf_ocr_noise(text: str) -> str:
    """Normalize PDF/OCR extraction noise (ligatures, line-break hyphens, wraps, smart punctuation)."""
    norm_text, _, _ = _normalize_pdf_ocr_with_map(text or "", base_offset=0)
    return norm_text


def canonicalize_unit(raw_unit: str | None) -> str:
    if not raw_unit:
        return "NONE"
    cleaned = re.sub(r"[\s-]+", " ", raw_unit.strip().lower())
    if cleaned.endswith((" day", " month", " year")):
        cleaned += "s"
    return _UNIT_ALIASES.get(cleaned, _UNIT_ALIASES.get(cleaned.rstrip("."), cleaned.upper()))


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
    | {"hundred", "thousand", "mil"},
    key=len,
    reverse=True,
)


_ALL_NUMBER_TAIL_WORDS = sorted(set(_ALL_NUMBER_WORDS) | {"and", "y"}, key=len, reverse=True)


_COMPOUND_WORD_QTY_RE = re.compile(
    r"(?<![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\-])"
    r"(?P<phrase>(?:" + "|".join(_ALL_NUMBER_WORDS) + r")(?:[\s\-]+(?:" + "|".join(_ALL_NUMBER_TAIL_WORDS) + r"))*)"
    r"(?:\s*\((?P<paren_num>\d+)\))?"
    r"(?:\s+(?P<scale>mil\s+millones|billions?|millions?|millones|mill[oó]n|thousands?|miles|mil))?"
    r"(?:\s+|\s*-\s*)(?:(?:de|of)\s+)?(?P<unit>" + _UNIT_PATTERN + r")",
    re.IGNORECASE,
)


_PAREN_NUM_WORD_RE = re.compile(
    r"(?<![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\-])"
    r"(?P<phrase>(?:" + "|".join(_ALL_NUMBER_WORDS) + r")(?:[\s\-]+(?:" + "|".join(_ALL_NUMBER_TAIL_WORDS) + r"))*)"
    r"\s*\((?P<paren_num>\d+)\)",
    re.IGNORECASE,
)


_STANDALONE_WORD_QTY_CANON_RE = re.compile(
    r"(?<![a-zA-ZáéíóúñüÁÉÍÓÚÑÜ0-9\-])"
    r"(?P<phrase>(?:" + "|".join(_ALL_NUMBER_WORDS) + r")(?:[\s\-]+(?:" + "|".join(_ALL_NUMBER_TAIL_WORDS) + r"))*)"
    r"(?P<sep>\s+)(?=(?:"
    r"(?:mil\s+millones|billions?|millions?|millones|mill[oó]n|thousands?|miles)\b|"
    r"(?:business|working|banking|trading|calendar|fiscal)?\s*(?:days?|weeks?|months?|years?|hours?|d[ií]as?|semanas?|mes(?:es)?|a[ñn]os?|horas?)\b|"
    r"per\s+cent\.?|percent\b|por\s+ciento\b|%|basis\s+points?|bps\b|dollars?|d[oó]lares?|euros?|pounds?|libras?"
    r"))",
    re.IGNORECASE,
)


def detect_internal_quantity_conflicts(text: str) -> list[dict[str, object]]:
    """Detect internal written-vs-parenthetical numeral discrepancies such as 'thirty (40) days'."""
    if not text:
        return []
    conflicts: list[dict[str, object]] = []
    for m in _PAREN_NUM_WORD_RE.finditer(text):
        word_val = _parse_word_number_phrase(m.group("phrase"))
        if word_val is None:
            continue
        paren_val = int(m.group("paren_num"))
        if word_val != paren_val:
            conflicts.append(
                {
                    "kind": "word_paren_mismatch",
                    "raw": m.group(0),
                    "word_value": word_val,
                    "paren_value": paren_val,
                    "span": [m.start(), m.end()],
                }
            )
    return conflicts


def canonicalize_quantity_notations_with_provenance(
    text: str,
) -> tuple[str, list[dict[str, str]]]:
    """Symmetrically normalize equivalent quantity notations while recording rule provenance.

    Rules normalized symmetrically:
    - ``thousands_separator``: comma thousands present vs absent (``1,000,000`` -> ``1000000``), preserving decimals.
    - ``percent_notation``: ``5 percent`` / ``5 per cent`` / ``5 %`` -> ``5%``.
    - ``word_paren_number``: matching ``thirty (30)`` -> ``30`` and ``thirty days`` -> ``30 days``;
      contradictory forms such as ``thirty (40)`` are NEVER collapsed so they always remain flagged.
    """
    if not text:
        return "", []
    prov: list[dict[str, str]] = []
    s = text

    # 1. Equivalent written + parenthetical numeral "thirty (30)" -> "30" (skip contradictions like "thirty (40)")
    def _rep_wp(m: re.Match[str]) -> str:
        word_val = _parse_word_number_phrase(m.group("phrase"))
        paren_val = int(m.group("paren_num"))
        if word_val is not None and word_val == paren_val:
            canon = str(paren_val)
            prov.append({"rule": "word_paren_number", "from": m.group(0), "to": canon})
            return canon
        return m.group(0)

    s = _PAREN_NUM_WORD_RE.sub(_rep_wp, s)

    # 2. Standalone word number directly quantifying a typed unit or scale ("thirty days" -> "30 days")
    def _rep_word_qty(m: re.Match[str]) -> str:
        phrase = m.group("phrase")
        word_val = _parse_word_number_phrase(phrase)
        if word_val is None:
            return m.group(0)
        # Keep non-trivial written numerals ("one" only when followed by a unit, avoid "a"/"one of")
        low_p = phrase.strip().lower()
        if low_p in {"one", "un", "uno", "una"} and not re.search(r"\d", text):
            # Only canonicalize "one"/"un" if unambiguous unit check; still safe before days/months/years/percent/dollars
            pass
        canon = str(word_val)
        prov.append({"rule": "word_number_unit", "from": phrase, "to": canon})
        return f"{canon}{m.group('sep')}"

    s = _STANDALONE_WORD_QTY_CANON_RE.sub(_rep_word_qty, s)

    # 3. Percent notation: "5 percent" / "5 per cent" / "5 %" -> "5%"
    def _rep_pct(m: re.Match[str]) -> str:
        canon_pct = f"{m.group('num')}%"
        if m.group(0).strip() != canon_pct:
            prov.append({"rule": "percent_notation", "from": m.group(0), "to": canon_pct})
        return canon_pct

    s = re.sub(
        r"(?P<num>\b\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:\.\d+)?)\s*(?:per\s+cent\.?(?!\w)|percent\b|%)",
        _rep_pct,
        s,
        flags=re.IGNORECASE,
    )

    # 4. Thousands separator presence/absence ("1,000,000" -> "1000000", "$1,250.50" -> "$1250.50")
    has_euro_decimal_pattern = bool(re.search(r"\b\d+\.\d{3}(?:,\d{1,2})?\b", s))

    def _rep_thousands(m: re.Match[str]) -> str:
        raw_tok = m.group(0)
        # Do not alter single-comma European decimal like "1,250 euros" in dot-grouped EU text
        if has_euro_decimal_pattern and raw_tok.count(",") == 1 and "." not in raw_tok:
            return raw_tok
        stripped = raw_tok.replace(",", "")
        prov.append({"rule": "thousands_separator", "from": raw_tok, "to": stripped})
        return stripped

    s = re.sub(r"(?<![0-9.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![0-9,])", _rep_thousands, s)
    return s, prov


def _has_blocking_table_boundary(between: str) -> bool:
    """Return True if *between* separates a contextual table scale header from a later row."""
    if not between:
        return False
    # Allow 1-2 blank lines common in PDF table banners, but stop after >=3 blank lines.
    if re.search(r"\n\s*\n\s*\n\s*\n", between):
        return True
    # Ignore semicolons/periods inside parenthetical equity/footnote descriptions (e.g. "(authorized ...; issued ...)").
    scrubbed = re.sub(r"\([^()\n]{0,240}\)", " ", between)
    if ";" in scrubbed:
        return True
    for i, ch in enumerate(scrubbed):
        if ch == "." and _is_sentence_boundary_dot(scrubbed, i):
            nxt = scrubbed[i + 1 : i + 2]
            if not nxt or nxt.isspace():
                return True
    return False


_PERIOD_COL_HEADER_RE = re.compile(
    r"\b(?:three|six|nine|twelve|\d+)\s+(?:months?|weeks?|quarters?|years?)\s+ended\b|"
    r"\b(?:fiscal|fy|q[1-4])\s*(?:19|20)?\d{2}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z.]*\s+\d{1,2}(?:\s*,?\s*(?:19|20)\d{2})?\b",
    re.IGNORECASE,
)


def _scrub_header_dates_and_periods(tail: str) -> str:
    """Strip date and period column headers before testing for row-local numbers on a header line."""
    s = _ISO_DATE_RE.sub(" ", tail)
    s = _MDY_DATE_RE.sub(" ", s)
    s = _DMY_DATE_RE.sub(" ", s)
    s = _SLASH_DATE_RE.sub(" ", s)
    s = _DOT_DATE_RE.sub(" ", s)
    s = _PERIOD_COL_HEADER_RE.sub(" ", s)
    return s.strip()


def _detect_contextual_table_spec(
    quote: str,
    source: str | None = None,
) -> _TableScaleHeaderSpec | None:
    """Detect governing table-scale header spec (scale, currency, exception rules) for *quote* inside *source*."""
    source = source or quote
    normalized_source = "\n".join(normalize_span(line) for line in source.splitlines())
    norm_quote = normalize_span(quote)
    if not norm_quote:
        return None
    tolerant_pat = _build_line_tolerant_search_regex(norm_quote)
    location = re.search(tolerant_pat, normalized_source, re.IGNORECASE)
    if not location:
        location = re.search(re.escape(norm_quote).replace(r"\ ", r"\s+"), normalized_source, re.IGNORECASE)
    if not location:
        return None
    offset = location.start()
    specs = [
        sp
        for sp in _find_table_scale_headers(normalized_source[: location.end()])
        if sp.hdr_start <= offset or (offset <= sp.span_end <= location.end())
    ]
    if not specs:
        return None
    spec = specs[-1]
    tail_start = spec.span_end
    between = normalized_source[tail_start:offset].lstrip(" .:;)-")
    if _has_blocking_table_boundary(between):
        return None
    hdr_raw_snippet = normalized_source[spec.hdr_start : spec.hdr_end].strip()
    is_table_banner = (
        spec.is_parenthetical
        or spec.exclude_per_share
        or spec.exclude_par_value
        or spec.exclude_percentages
        or spec.exclude_shares
        or bool(
            re.search(
                r"\b(?:amounts?|dollars?|usd|eur|gbp|[$€£]|in\s+(?:thousands|millions|billions)|"
                r"(?:thousands|millions|billions)\s*(?:,|\bexcept))\b",
                hdr_raw_snippet,
                re.IGNORECASE,
            )
        )
    )
    looks_like_prose_sentence = (
        norm_quote.endswith((".", "!", "?"))
        and "|" not in quote
        and not re.search(r"\s{2,}(?:[$€£]?\d)", quote)
    )
    if re.search(r"\n\s*\n", between) and (not is_table_banner or looks_like_prose_sentence):
        return None
    line_end = normalized_source.find("\n", tail_start)
    if line_end >= 0 and offset > line_end:
        tail = normalized_source[tail_start:line_end].strip()
        tail_scrubbed = _scrub_header_dates_and_periods(tail) if tail else ""
        # Reject row-local quantities on the header line (e.g. "Revenue (in thousands): $125"),
        # while allowing column date headers ("August 1, 2026", "3 Months Ended", "2023  2022").
        if tail_scrubbed and re.search(
            r"(?:[$€£]|:\s*[-−]?\d|\d+(?:[.,]\d+)?\s*(?:%|usd|eur|gbp|millions?|thousands?|billions?)|"
            r"(?<![A-Za-z0-9])(?!19\d\d\b|20\d\d\b)\d+(?:,\d{3})*(?:\.\d+)?(?![A-Za-z0-9]))",
            tail_scrubbed,
            re.IGNORECASE,
        ):
            return None

    currency = spec.currency
    if currency is None:
        table_window = normalized_source[max(0, spec.hdr_start - 120) : min(len(normalized_source), location.end() + 250)]
        has_usd = bool(re.search(r"(?:\$|\busd\b|\bdollars?\b)", table_window, re.IGNORECASE))
        has_eur = bool(re.search(r"(?:€|\beur\b|\beuros?\b)", table_window, re.IGNORECASE))
        has_gbp = bool(re.search(r"(?:£|\bgbp\b|\bpounds?\b|\bsterling\b)", table_window, re.IGNORECASE))
        if has_usd and not has_eur and not has_gbp:
            currency = "USD"
        elif has_eur and not has_usd and not has_gbp:
            currency = "EUR"
        elif has_gbp and not has_usd and not has_eur:
            currency = "GBP"
        elif spec.exclude_per_share or re.search(r"\bper[- ]share\b", normalized_source[spec.hdr_start : spec.hdr_end], re.IGNORECASE):
            if not re.search(r"(?:€|£|\beur\b|\beuros?\b|\bgbp\b|\bpounds?\b|\bsterling\b)", normalized_source, re.IGNORECASE):
                currency = "USD"

    inherited_row_label = ""
    if source and "\n" in source:
        src_line_ctx = _build_norm_line_context(source, normalized_source.replace("\n", " "))
        for s_off, e_off, ctx_lbl in src_line_ctx:
            if s_off <= offset <= e_off:
                inherited_row_label = ctx_lbl
                break
        if not inherited_row_label:
            # Walk raw lines directly up to `offset` inside `normalized_source`
            raw_prefix_lines = [normalize_span(ln) for ln in normalized_source[:offset].splitlines()]
            _, inherited_row_label = _scan_row_group_state(raw_prefix_lines)

    return _TableScaleHeaderSpec(
        scale=spec.scale,
        currency=currency,
        hdr_start=spec.hdr_start,
        hdr_end=spec.hdr_end,
        span_end=spec.span_end,
        is_parenthetical=spec.is_parenthetical,
        exclude_per_share=spec.exclude_per_share,
        exclude_par_value=spec.exclude_par_value,
        exclude_percentages=spec.exclude_percentages,
        exclude_shares=spec.exclude_shares,
        inherited_row_label=inherited_row_label,
    )


def _mask_preserving_len(match: re.Match[str]) -> str:
    return " " * (match.end() - match.start())


_SUBROW_CONTINUATION_RE = re.compile(
    r"^\s*(?:basic|diluted|class\s+[a-z0-9]+|common|preferred|total\s+(?:basic|diluted))\b",
    re.IGNORECASE,
)


_DATA_NUMBER_IN_LINE_RE = re.compile(
    r"(?:[$€£]\s*[-−(]?\d|\b\d+(?:,\d{3})*(?:\.\d+)?\b)"
)


def _line_has_table_data_number(norm_line: str) -> bool:
    """Return True if *norm_line* contains actual numeric cell values (ignoring footnote markers '(1)', '(a)', '(Note 4)')."""
    cleaned = re.sub(r"\(\s*(?:\d{1,2}|[a-zA-Z]|note\s+\d+)\s*\)", " ", norm_line, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", " ", cleaned)
    return bool(_DATA_NUMBER_IN_LINE_RE.search(cleaned))


def _scale_header_line_indices(norm_lines: list[str]) -> set[int]:
    """Indices of lines belonging to a (possibly multi-line) table-scale header declaration."""
    joined = "\n".join(norm_lines)
    specs = _find_table_scale_headers(joined)
    if not specs:
        return set()
    starts: list[int] = []
    pos = 0
    for ln in norm_lines:
        starts.append(pos)
        pos += len(ln) + 1
    out: set[int] = set()
    for i, st in enumerate(starts):
        en = st + len(norm_lines[i])
        if any(sp.hdr_start < en and st < sp.span_end for sp in specs):
            out.add(i)
    return out


def _scan_row_group_state(lines: list[str]) -> tuple[str, str]:
    """Return `(active_group_label, combined_group_and_pending_label)` after walking *lines*."""
    group_label = ""
    pending_labels: list[str] = []
    blank_run = 0
    hdr_idx = _scale_header_line_indices([x.strip() for x in lines])
    for li, ln in enumerate(lines):
        if li in hdr_idx:
            pending_labels.clear()
            group_label = ""
            continue
        s = ln.strip()
        if not s:
            blank_run += 1
            if blank_run >= 2:
                pending_labels.clear()
                group_label = ""
            continue
        blank_run = 0
        if s in {"$", "€", "£"}:
            continue
        if _find_table_scale_headers(s):
            pending_labels.clear()
            group_label = ""
            continue
        if not _line_has_table_data_number(s):
            stripped_trail = s.rstrip()
            if stripped_trail.endswith((":", "—", "-")) or _ROW_PER_SHARE_OR_PAR_RE.search(s) or _ROW_SHARES_RE.search(s):
                if len(pending_labels) == 0 and stripped_trail.endswith((":", "—", "-")):
                    group_label = s
                elif _SUBROW_CONTINUATION_RE.match(s):
                    pending_labels.append(s)
                else:
                    if group_label and not stripped_trail.endswith((":", "—", "-")):
                        pending_labels.append(s)
                    else:
                        group_label = s
                        pending_labels.clear()
            else:
                if group_label and not _SUBROW_CONTINUATION_RE.match(s) and len(s.split()) >= 2:
                    group_label = ""
                pending_labels.append(s)
        else:
            if group_label and not pending_labels and not _SUBROW_CONTINUATION_RE.match(s) and not re.match(r"^\s*[$€£]?\s*[\d.,()-]+\s*$", s):
                if not (_ROW_PER_SHARE_OR_PAR_RE.search(s) or _ROW_SHARES_RE.search(s)):
                    group_label = ""
            pending_labels.clear()
    combined = " ".join(x for x in [group_label, *pending_labels] if x).strip()
    return group_label, combined


def _build_norm_line_context(raw_text: str, norm_text: str) -> list[tuple[int, int, str]]:
    """Return `[(norm_start, norm_end, inherited_group_or_wrapped_label), ...]` for lines in *raw_text*."""
    if "\n" not in raw_text:
        return [(0, len(norm_text), "")]
    out: list[tuple[int, int, str]] = []
    cursor = 0
    group_label = ""
    pending_labels: list[str] = []
    blank_run = 0

    raw_lines_all = raw_text.splitlines()
    hdr_idx = _scale_header_line_indices([normalize_span(x) for x in raw_lines_all])
    for li, raw_ln in enumerate(raw_lines_all):
        nln = normalize_span(raw_ln)
        if li in hdr_idx and nln:
            found_h = norm_text.find(nln, cursor)
            h_start = found_h if found_h >= 0 else min(cursor, len(norm_text))
            h_end = min(len(norm_text), h_start + len(nln))
            cursor = h_end
            out.append((h_start, h_end, ""))
            pending_labels.clear()
            group_label = ""
            continue
        if not nln:
            blank_run += 1
            if blank_run >= 2:
                pending_labels.clear()
                group_label = ""
            continue
        blank_run = 0
        found = norm_text.find(nln, cursor)
        if found >= 0:
            start = found
            end = found + len(nln)
            cursor = end
        else:
            start = min(cursor, len(norm_text))
            end = min(len(norm_text), start + len(nln))
            cursor = end

        if nln.strip() in {"$", "€", "£"}:
            eff_ctx = " ".join(x for x in [group_label, *pending_labels] if x).strip()
            out.append((start, end, eff_ctx))
            continue

        if _find_table_scale_headers(nln):
            out.append((start, end, ""))
            pending_labels.clear()
            group_label = ""
            continue

        is_data = _line_has_table_data_number(nln)
        if not is_data:
            stripped_trail = nln.strip()
            eff_ctx = " ".join(x for x in [group_label, *pending_labels] if x).strip()
            out.append((start, end, eff_ctx))
            if stripped_trail.endswith((":", "—", "-")) and not pending_labels:
                group_label = nln
            elif _SUBROW_CONTINUATION_RE.match(nln):
                pending_labels.append(nln)
            elif len(nln.split()) <= 5 and (_ROW_PER_SHARE_OR_PAR_RE.search(nln) or _ROW_SHARES_RE.search(nln)):
                group_label = nln
                pending_labels.clear()
            else:
                if group_label and not _SUBROW_CONTINUATION_RE.match(nln) and len(pending_labels) == 0:
                    # Wait: only reset group_label if this standalone label is clearly a new top-level section
                    if stripped_trail.endswith(":"):
                        group_label = nln
                pending_labels.append(nln)
        else:
            if (
                group_label
                and not pending_labels
                and not _SUBROW_CONTINUATION_RE.match(nln)
                and not re.match(r"^\s*[$€£]?\s*[-−(]?[\d.,]+(?:\s*[$€£]?\s*[-−(]?[\d.,]+)*\s*$", nln)
                and not (_ROW_PER_SHARE_OR_PAR_RE.search(nln) or _ROW_SHARES_RE.search(nln))
            ):
                group_label = ""
            eff_ctx = " ".join(x for x in [group_label, *pending_labels] if x).strip()
            out.append((start, end, eff_ctx))
            pending_labels.clear()

    return out or [(0, len(norm_text), "")]


def _line_context_for_pos(norm_text: str, line_ctx: list[tuple[int, int, str]], pos: int) -> tuple[int, int, str]:
    for s, e, ctx in line_ctx:
        if s <= pos <= e:
            return s, e, ctx
    return 0, len(norm_text), ""


def _is_scale_excluded_for_match(
    norm_text: str,
    line_bounds: list[tuple[int, int, str]] | list[tuple[int, int]],
    m_start: int,
    m_end: int,
    raw_num: str,
    unit: str,
    table_scale_spec: _TableScaleHeaderSpec | None,
) -> bool:
    """Return True if *m_start..m_end* falls under an excluded row type (per share/unit, par value, %, shares, ratio)."""
    if unit not in _CURRENCY_UNITS and unit != "NONE":
        return True
    if unit == "NONE" and re.fullmatch(r"(?:19|20)\d{2}", raw_num.strip()):
        return True

    # Support both legacy 2-tuple bounds and rich 3-tuple (start, end, inherited_ctx) entries
    if line_bounds and len(line_bounds[0]) == 3:  # type: ignore[index]
        ls, le, row_group_ctx = _line_context_for_pos(norm_text, line_bounds, m_start)  # type: ignore[arg-type]
    else:
        ls, le, row_group_ctx = 0, len(norm_text), ""
        for sp in line_bounds:
            if sp[0] <= m_start <= sp[1]:
                ls, le = sp[0], sp[1]
                break

    seg_s = max(ls, norm_text.rfind(";", ls, m_start) + 1)
    seg_s = max(seg_s, norm_text.rfind("|", ls, m_start) + 1)
    before = norm_text[seg_s:m_start]
    full_seg = norm_text[seg_s:le]

    inherited = (table_scale_spec.inherited_row_label if table_scale_spec is not None else "") or ""
    combined_before = " ".join(x for x in (inherited, row_group_ctx, before) if x).strip()
    combined_seg = " ".join(x for x in (inherited, row_group_ctx, full_seg) if x).strip()

    # Handle balance-sheet capital lines where par value appears inside label BEFORE an equity column total:
    # e.g. "Common stock — par value $1.00 per share (authorized 4.3B shares; issued 3.1B shares)  3,120"
    post_share_paren_equity = bool(re.search(r"\([^)]*\bshares?\b[^)]*\)\s*(?:[$€£]\s*)?$", before, re.IGNORECASE))

    # Row-type exclusion: always enforced by row label/group type regardless of header exception wording
    if not post_share_paren_equity:
        if _ROW_PER_SHARE_OR_PAR_RE.search(combined_before):
            return True
        if re.search(r"\(\s*(?:in\s+)?dollars?\s+per\s+share\s*\)", combined_seg, re.IGNORECASE):
            return True

    if unit == "NONE":
        if _ROW_PERCENTAGE_RE.search(combined_seg):
            return True
        if _ROW_SHARES_RE.search(combined_before):
            return True
        if _ROW_RATIO_RE.search(combined_before):
            return True

    return False


def extract_quantities(
    text: str,
    explicit_unit: str | None = None,
    *,
    table_scale: Decimal | None = None,
    table_scale_spec: _TableScaleHeaderSpec | None = None,
    apply_table_scale: bool = False,
    es_context: bool | None = None,
) -> list[ParsedQuantity]:
    """Extract normalized (Decimal magnitude, unit) quantities from text."""
    if not text:
        return []

    norm = normalize_span(text)
    if table_scale_spec is not None and table_scale is None:
        table_scale = table_scale_spec.scale
    elif apply_table_scale and table_scale is None:
        table_scale_spec = _detect_contextual_table_spec(norm)
        table_scale = table_scale_spec.scale if table_scale_spec is not None else None
    is_es = bool(_SPANISH_CONTEXT_RE.search(norm)) if es_context is None else es_context
    line_bounds = _build_norm_line_context(text, norm) if table_scale is not None else [(0, len(norm), "")]

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
        scale_raw = m.group("scale")
        scale_mult = _SCALE_FACTORS.get(re.sub(r"\s+", " ", scale_raw.strip().lower()), Decimal("1")) if scale_raw else Decimal("1")
        unit = canonicalize_unit(m.group("unit"))
        span = (m.start(), m.end())
        mag = Decimal(word_val) * scale_mult
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=mag.normalize(), unit=unit, span=span))
        # Negative control: never hide a contradicting parenthetical numeral such as "thirty (40) days".
        paren_raw = m.group("paren_num")
        if paren_raw is not None:
            paren_val = int(paren_raw)
            if paren_val != word_val:
                paren_mag = Decimal(paren_val) * scale_mult
                results.append(ParsedQuantity(magnitude=paren_mag.normalize(), unit=unit, span=span))

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
            if not _is_scale_excluded_for_match(
                norm,
                line_bounds,
                m.start(),
                m.end(),
                raw_num,
                unit,
                table_scale_spec,
            ):
                magnitude *= table_scale

        span = (m.start(), m.end())
        occupied_spans.append(span)
        results.append(ParsedQuantity(magnitude=magnitude.normalize(), unit=unit, span=span))

    results.sort(key=lambda q: q.span[0])
    return results


def _is_sentence_boundary_dot(text: str, pos: int) -> bool:
    """Return True if `text[pos]` ('.') is a sentence boundary rather than a decimal/section dot, abbreviation, or list bullet."""
    if pos < 0 or pos >= len(text) or text[pos] != ".":
        return False
    # Decimal or section number like "4.01" or "6.75"
    if pos > 0 and pos + 1 < len(text) and text[pos - 1].isdigit() and text[pos + 1].isdigit():
        return False
    # Common legal/financial abbreviation like "Sec.", "Art.", "No."
    window_before = text[max(0, pos - 8) : pos + 1]
    if _ABBREV_BEFORE_DOT_RE.search(window_before):
        return False
    # List enumeration marker ("a. Item", "i. Item", "1. Item") at clause/line start or inline roman marker
    win_bul = text[max(0, pos - 16) : pos + 1]
    m_bul = re.search(r"(?:^|[\n;:]\s*|(?<=\.\s))\s*([a-z]|x{1,2}(?:i{1,3}|iv|vi{0,3})?|i{1,3}|iv|vi{0,3}|ix|\d{1,2})\.$", win_bul)
    if not m_bul:
        m_bul = re.search(r"(?:^|\s)(x{1,2}(?:i{1,3}|iv|vi{0,3})?|i{1,3}|iv|vi{0,3}|ix)\.$", win_bul)
    if m_bul:
        tail_after = text[pos + 1 : pos + 12]
        if re.match(r"^[ \t]+[A-Za-z0-9($€£\"'\[\*]", tail_after):
            return False
    return True


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

