"""Deterministic, typed, versioned normalizers for v4 proposition & evidence alignment.

Every normalizer applies identically to structured proposition values and to
evidence spans without ever altering the quoted source text in ``EvidenceRef``.
Every invocation records ``NORMALIZER_VERSION`` so downstream certificates and
audits can reproduce the exact normalization semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any

NORMALIZER_VERSION = "v4-norm-1"

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "january": 1, "ene": 1, "enero": 1,
    "feb": 2, "february": 2, "febrero": 2,
    "mar": 3, "march": 3, "marzo": 3,
    "apr": 4, "april": 4, "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "june": 6, "junio": 6,
    "jul": 7, "july": 7, "julio": 7,
    "aug": 8, "august": 8, "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "september": 9, "septiembre": 9, "setiembre": 9,
    "oct": 10, "october": 10, "octubre": 10,
    "nov": 11, "november": 11, "noviembre": 11,
    "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
}

def _endpoint_re(suffix: str) -> str:
    return (
        rf"(?P<h{suffix}>\d{{1,2}})"
        rf"(?:\s*:\s*(?P<m{suffix}>\d{{1,2}})(?:\s*:\s*(?P<s{suffix}>\d{{2}}))?)?"
        rf"(?:\s*(?P<ap{suffix}>[ap]\s*\.?\s*m\.?))?"
    )


_TIME_SEP = r"(?:\s*(?:[-–—~]|to|until|through|thru|a|hasta)\s*)"
_TIME_INTERVAL_EXACT_RE = re.compile(
    r"^\s*" + _endpoint_re("1") + _TIME_SEP + _endpoint_re("2") + r"\s*$",
    re.IGNORECASE,
)
_TIME_INTERVAL_FIND_RE = re.compile(
    r"(?<![A-Za-z0-9:.])"
    + _endpoint_re("1")
    + _TIME_SEP
    + _endpoint_re("2")
    + r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_EN_NAMED_DATE_RE = re.compile(
    r"^\s*([A-Za-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\s*$",
    re.IGNORECASE,
)
_DMY_NAMED_DATE_RE = re.compile(
    r"^\s*(\d{1,2})(?:st|nd|rd|th)?\s+(?:de\s+)?([A-Za-z]+)\.?\s+(?:(?:de|of)\s+)?(\d{4})\s*$",
    re.IGNORECASE,
)
_YMD_SLASH_RE = re.compile(r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s*$")
_DMY_OR_MDY_SLASH_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})\s*$")

_PCT_EXACT_RE = re.compile(
    r"^\s*(?P<num>[-+]?\d[\d.,\s]*\d|[-+]?\d)\s*(?P<unit>%|pct\b|per\s*cent\b|percent(?:age)?\b)\s*$",
    re.IGNORECASE,
)
_PCT_FIND_RE = re.compile(
    r"(?<![0-9A-Za-z_])(?P<num>[-+]?\d[\d.,]*\d|[-+]?\d)\s*(?P<unit>%|pct\b|per\s*cent\b|percent\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NormalizationTrace:
    """Provenance record for a single deterministic normalization."""

    kind: str
    raw: Any
    canonical: Any
    version: str = NORMALIZER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalizer_version": self.version,
            "kind": self.kind,
            "raw": self.raw,
            "canonical": self.canonical,
        }


def _parse_clock_endpoint(
    hour_s: str,
    min_s: str | None,
    sec_s: str | None,
    ampm_s: str | None,
    *,
    inherited_ampm: str | None = None,
) -> tuple[str, str | None] | None:
    """Parse one time endpoint into `('HH:MM[:SS]', detected_ampm)` or None."""
    if min_s is None and not (ampm_s or inherited_ampm):
        return None
    hour = int(hour_s)
    if min_s is None:
        minute = 0
    elif len(min_s) == 1:
        # Handle OCR trailing-zero loss like '21:0' -> 21:00; reject ambiguous '21:5'
        if min_s != "0":
            return None
        minute = 0
    else:
        minute = int(min_s)
    second = int(sec_s) if sec_s is not None else 0
    if not (0 <= minute <= 59 and 0 <= second <= 59):
        return None

    ap_clean = re.sub(r"[\s.]", "", ampm_s.lower()) if ampm_s else None
    eff_ap = ap_clean or inherited_ampm
    if eff_ap is not None:
        if not (1 <= hour <= 12):
            return None
        if eff_ap == "am":
            hour = 0 if hour == 12 else hour
        elif eff_ap == "pm":
            hour = 12 if hour == 12 else hour + 12
        else:
            return None
    else:
        if not (0 <= hour <= 23):
            return None

    base = f"{hour:02d}:{minute:02d}"
    if sec_s is not None and second != 0:
        base += f":{second:02d}"
    return base, ap_clean


def normalize_time_interval(raw: Any) -> str | None:
    """Normalize a clock time interval into canonical 24h `'HH:MM-HH:MM'`, or None."""
    if not isinstance(raw, str):
        return None
    m = _TIME_INTERVAL_EXACT_RE.match(raw)
    if m is None:
        return None
    ap1 = re.sub(r"[\s.]", "", m.group("ap1").lower()) if m.group("ap1") else None
    ap2 = re.sub(r"[\s.]", "", m.group("ap2").lower()) if m.group("ap2") else None
    # Inherit trailing 'PM'/'AM' only onto bare '9' or when neither side is > 12 (e.g. '4:30-9 PM')
    inh1 = ap2 if (ap1 is None and ap2 is not None and int(m.group("h1")) <= 12) else None
    p1 = _parse_clock_endpoint(m.group("h1"), m.group("m1"), m.group("s1"), m.group("ap1"), inherited_ampm=inh1)
    p2 = _parse_clock_endpoint(m.group("h2"), m.group("m2"), m.group("s2"), m.group("ap2"))
    if p1 is None or p2 is None:
        return None
    return f"{p1[0]}-{p2[0]}"


def extract_time_intervals(text: str) -> list[tuple[str, int, int, str]]:
    """Find candidate time-interval spans `(canonical, start, end, raw_slice)` in prose."""
    out: list[tuple[str, int, int, str]] = []
    for m in _TIME_INTERVAL_FIND_RE.finditer(text or ""):
        # Require at least one ':' or AM/PM marker so plain ranges like '10-20' are not treated as times
        has_time_cue = any(
            m.group(g) is not None for g in ("m1", "m2", "ap1", "ap2")
        )
        if not has_time_cue:
            continue
        canon = normalize_time_interval(m.group(0))
        if canon is not None:
            out.append((canon, m.start(), m.end(), m.group(0)))
    return out


def normalize_separator_number(raw: Any) -> str | None:
    """Normalize US (`1,250.50`) or EU (`1.250,50`, `5,80`) formatted numbers to decimal text.

    Returns canonical decimal string without scientific notation, or ``None`` when
    the token is not a valid finite formatted number.
    """
    if type(raw) is bool:
        return None
    if isinstance(raw, (int, Decimal)):
        d = Decimal(raw) if isinstance(raw, int) else raw
        if not d.is_finite():
            return None
        norm = d.normalize()
        if norm == norm.to_integral_value():
            norm = norm.quantize(Decimal(1))
        return format(norm, "f")
    if isinstance(raw, float):
        if not Decimal(str(raw)).is_finite():
            return None
        d = Decimal(str(raw)).normalize()
        if d == d.to_integral_value():
            d = d.quantize(Decimal(1))
        return format(d, "f")
    if not isinstance(raw, str):
        return None

    s = raw.strip().replace("\u00a0", " ")
    if not s:
        return None
    sign = ""
    if s[0] in "+-":
        sign = "-" if s[0] == "-" else ""
        s = s[1:].strip()
    if not s or not re.fullmatch(r"\d[\d\s.,]*", s):
        return None

    # Remove thousands spaces like '1 250.50' or '1 250,50' ONLY if grouped properly
    if " " in s:
        parts = s.split(" ")
        body_tail = parts[1:]
        if not (parts[0].isdigit() and all(re.fullmatch(r"\d{3}(?:[.,]\d+)?", p) for p in body_tail)):
            return None
        s = "".join(parts)

    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_dot > last_comma:
            # US format: 1,250.50
            int_part, frac_part = s[:last_dot], s[last_dot + 1 :]
            if "." in int_part or not frac_part.isdigit():
                return None
            groups = int_part.split(",")
            if not (groups[0].isdigit() and all(len(g) == 3 and g.isdigit() for g in groups[1:])):
                return None
            cleaned = "".join(groups) + "." + frac_part
        else:
            # EU/Spanish format: 1.250,50
            int_part, frac_part = s[:last_comma], s[last_comma + 1 :]
            if "," in int_part or not frac_part.isdigit():
                return None
            groups = int_part.split(".")
            if not (groups[0].isdigit() and all(len(g) == 3 and g.isdigit() for g in groups[1:])):
                return None
            cleaned = "".join(groups) + "." + frac_part
    elif has_comma:
        groups = s.split(",")
        if len(groups) > 2:
            if not (groups[0].isdigit() and all(len(g) == 3 and g.isdigit() for g in groups[1:])):
                return None
            cleaned = "".join(groups)
        else:
            left, right = groups
            if not left.isdigit() or not right.isdigit():
                return None
            if len(right) == 3 and 1 <= len(left) <= 3:
                # English thousands grouping '1,250'
                cleaned = left + right
            else:
                # Decimal comma '5,80' or '1000,5'
                cleaned = left + "." + right
    elif has_dot:
        groups = s.split(".")
        if len(groups) > 2:
            if not (groups[0].isdigit() and all(len(g) == 3 and g.isdigit() for g in groups[1:])):
                return None
            cleaned = "".join(groups)
        else:
            left, right = groups
            if not left.isdigit() or not right.isdigit():
                return None
            cleaned = left + "." + right
    else:
        if not s.isdigit():
            return None
        cleaned = s

    try:
        dec = Decimal(sign + cleaned)
    except InvalidOperation:
        return None
    if not dec.is_finite():
        return None
    norm = dec.normalize()
    if norm == norm.to_integral_value():
        norm = norm.quantize(Decimal(1))
    return format(norm, "f")


def normalize_percentage(raw: Any) -> str | None:
    """Normalize percentage text (`'5,80 %'`, `'5.80%'`, `'5.8 percent'`) into `'<num>%'`."""
    if not isinstance(raw, str):
        return None
    m = _PCT_EXACT_RE.match(raw)
    if m is None:
        return None
    num = normalize_separator_number(m.group("num"))
    if num is None:
        return None
    return f"{num}%"


def normalize_date_token(raw: Any) -> str | None:
    """Normalize unambiguous date tokens (ISO 8601, YYYY/MM/DD, named months, unambiguous DD/MM/YYYY)."""
    if not isinstance(raw, str):
        return None
    s = re.sub(r"\s+", " ", raw.strip())
    if not s:
        return None

    # ISO forms
    if re.fullmatch(r"\d{4}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}", s):
        try:
            date(int(s[:4]), int(s[5:7]), 1)
            return s
        except ValueError:
            return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            return date.fromisoformat(s).isoformat()
        except ValueError:
            return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?", s):
        try:
            moment = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
        except ValueError:
            return None
        if moment.tzinfo is None:
            return moment.replace(microsecond=0).isoformat()
        return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    m_ymd = _YMD_SLASH_RE.match(s)
    if m_ymd:
        try:
            return date(int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))).isoformat()
        except ValueError:
            return None

    m_en = _EN_NAMED_DATE_RE.match(s)
    if m_en:
        mon = _MONTH_MAP.get(m_en.group(1).lower())
        if mon is not None:
            try:
                return date(int(m_en.group(3)), mon, int(m_en.group(2))).isoformat()
            except ValueError:
                return None

    m_dmy_named = _DMY_NAMED_DATE_RE.match(s)
    if m_dmy_named:
        mon = _MONTH_MAP.get(m_dmy_named.group(2).lower())
        if mon is not None:
            try:
                return date(int(m_dmy_named.group(3)), mon, int(m_dmy_named.group(1))).isoformat()
            except ValueError:
                return None

    m_slash = _DMY_OR_MDY_SLASH_RE.match(s)
    if m_slash:
        p1, p2, yr = int(m_slash.group(1)), int(m_slash.group(2)), int(m_slash.group(3))
        if p1 > 12 and 1 <= p2 <= 12:
            try:
                return date(yr, p2, p1).isoformat()
            except ValueError:
                return None
        if p2 > 12 and 1 <= p1 <= 12:
            try:
                return date(yr, p1, p2).isoformat()
            except ValueError:
                return None
        if p1 == p2 and 1 <= p1 <= 12:
            try:
                return date(yr, p1, p2).isoformat()
            except ValueError:
                return None
    return None


def extract_normalized_dates(text: str) -> set[str]:
    """Extract all unambiguously normalizable ISO calendar dates from prose."""
    out: set[str] = set()
    if not text:
        return out
    patterns = (
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
        r"\b\d{4}/\d{1,2}/\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
        r"\b[A-Za-z]{3,10}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:de\s+)?[A-Za-z]{3,12}\.?\s+(?:(?:de|of)\s+)?\d{4}\b",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            norm = normalize_date_token(m.group(0))
            if norm is not None:
                out.add(norm)
    return out


def normalize_value_with_provenance(kind: str, raw: Any) -> tuple[Any, NormalizationTrace]:
    """Apply a typed versioned normalizer symmetrically and return `(canonical, NormalizationTrace)`."""
    fn_map = {
        "time_interval": normalize_time_interval,
        "date": normalize_date_token,
        "number": normalize_separator_number,
        "percentage": normalize_percentage,
    }
    fn = fn_map.get(kind)
    if fn is None:
        raise ValueError(f"Unsupported normalizer kind: {kind!r}")
    norm = fn(raw)
    if norm is None:
        raise ValueError(f"Value {raw!r} cannot be normalized as {kind!r}")
    return norm, NormalizationTrace(kind=kind, raw=raw, canonical=norm, version=NORMALIZER_VERSION)
