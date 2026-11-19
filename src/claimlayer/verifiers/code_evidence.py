"""Strict evidence verification profile for source code, diffs, and config files."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

from claimlayer.models import Assertion, Issue, Policy, Source
from claimlayer.verifiers.quote_quantity import (
    ParsedQuantity,
    canonicalize_unit,
    extract_dates,
    extract_quantities,
    unit_dimension,
)

_CODE_UNIT_CANON: dict[str, tuple[str, str]] = {
    "s": ("SECONDS", "DURATION"),
    "sec": ("SECONDS", "DURATION"),
    "secs": ("SECONDS", "DURATION"),
    "second": ("SECONDS", "DURATION"),
    "seconds": ("SECONDS", "DURATION"),
    "ms": ("MILLISECONDS", "DURATION"),
    "msec": ("MILLISECONDS", "DURATION"),
    "msecs": ("MILLISECONDS", "DURATION"),
    "millisecond": ("MILLISECONDS", "DURATION"),
    "milliseconds": ("MILLISECONDS", "DURATION"),
    "us": ("MICROSECONDS", "DURATION"),
    "usec": ("MICROSECONDS", "DURATION"),
    "usecs": ("MICROSECONDS", "DURATION"),
    "microsecond": ("MICROSECONDS", "DURATION"),
    "microseconds": ("MICROSECONDS", "DURATION"),
    "ns": ("NANOSECONDS", "DURATION"),
    "nsec": ("NANOSECONDS", "DURATION"),
    "nsecs": ("NANOSECONDS", "DURATION"),
    "nanosecond": ("NANOSECONDS", "DURATION"),
    "nanoseconds": ("NANOSECONDS", "DURATION"),
    "min": ("MINUTES", "DURATION"),
    "mins": ("MINUTES", "DURATION"),
    "minute": ("MINUTES", "DURATION"),
    "minutes": ("MINUTES", "DURATION"),
    "b": ("BYTES", "SIZE"),
    "byte": ("BYTES", "SIZE"),
    "bytes": ("BYTES", "SIZE"),
    "kb": ("KB", "SIZE"),
    "kib": ("KB", "SIZE"),
    "kilobyte": ("KB", "SIZE"),
    "kilobytes": ("KB", "SIZE"),
    "mb": ("MB", "SIZE"),
    "mib": ("MB", "SIZE"),
    "megabyte": ("MB", "SIZE"),
    "megabytes": ("MB", "SIZE"),
    "gb": ("GB", "SIZE"),
    "gib": ("GB", "SIZE"),
    "gigabyte": ("GB", "SIZE"),
    "gigabytes": ("GB", "SIZE"),
    "tb": ("TB", "SIZE"),
    "tib": ("TB", "SIZE"),
    "terabyte": ("TB", "SIZE"),
    "terabytes": ("TB", "SIZE"),
}

_CODE_DIM_QTY_RE = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<num>-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>milliseconds?|msecs?|ms|microseconds?|usecs?|us|"
    r"nanoseconds?|nsecs?|ns|seconds?|secs?|sec|s|minutes?|mins?|min|"
    r"kilobytes?|kib|kb|megabytes?|mib|mb|gigabytes?|gib|gb|"
    r"terabytes?|tib|tb|bytes?|b)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

_CODE_TOKEN_RE = re.compile(
    r"0[xX][0-9a-fA-F]+|(?<![A-Za-z0-9_.])-\d+(?:\.\d+)*|\d+(?:\.\d+)*|[A-Za-z_][A-Za-z0-9_]*"
)
_CALL_SUFFIX_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\(\)")
_PATH_COMPOUND_RE = re.compile(r"[A-Za-z0-9_@.-]+(?:/[A-Za-z0-9_@.-]+)+")
_QUALIFIED_COMPOUND_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.|::|->)[A-Za-z_][A-Za-z0-9_]*)+"
)
_STANDALONE_SLASH_DATE_RE = re.compile(
    r"(?<![A-Za-z0-9_/.-])\d{1,2}/\d{1,2}/\d{2,4}(?![A-Za-z0-9_/.-])"
)
_COMP_OP_RE = re.compile(
    r"===|!==|(?<![=!])!=(?!=)|(?<![=!<>])==(?!=)|<=|>=|(?<![-<=>])<(?![<=])|(?<![-<=>])>(?![>=])"
)


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _has_boundary_aligned_match(haystack: str, needle: str) -> bool:
    """Return True if *needle* occurs in *haystack* with matching identifier/numeric boundaries."""
    if not needle:
        return False
    check_left = _is_word_char(needle[0])
    check_right = _is_word_char(needle[-1])
    start = 0
    n_len = len(needle)
    h_len = len(haystack)
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return False
        left_ok = not check_left or idx == 0 or not _is_word_char(haystack[idx - 1])
        right_ok = (
            not check_right
            or idx + n_len == h_len
            or not _is_word_char(haystack[idx + n_len])
        )
        if left_ok and right_ok:
            return True
        start = idx + 1


def _normalize_code_whitespace(text: str) -> str:
    """Collapse horizontal whitespace per line while preserving line breaks and exact case."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(re.sub(r"[ \t]+", " ", ln.strip()) for ln in lines)


def _quote_has_code_substance(raw_q: str) -> bool:
    if len(raw_q) < 2:
        return False
    tokens = _CODE_TOKEN_RE.findall(raw_q)
    return any(len(tok) >= 2 for tok in tokens) or len(tokens) >= 2


def _canon_code_or_std_unit(raw_unit: str | None) -> str | None:
    if not raw_unit or not raw_unit.strip():
        return None
    cleaned = re.sub(r"[\s-]+", " ", raw_unit.strip().lower())
    if cleaned in _CODE_UNIT_CANON:
        return _CODE_UNIT_CANON[cleaned][0]
    std = canonicalize_unit(raw_unit)
    return std if std != "NONE" else None


def _extract_code_quantities(
    text: str,
    explicit_unit: str | None = None,
) -> list[ParsedQuantity]:
    """Extract standard quantities plus code/config time and byte-size units."""
    out: list[ParsedQuantity] = []
    occupied: list[tuple[int, int]] = []

    for m in _CODE_DIM_QTY_RE.finditer(text):
        try:
            mag = Decimal(m.group("num"))
        except InvalidOperation:
            continue
        canon = _CODE_UNIT_CANON[m.group("unit").lower()][0]
        span = (m.start(), m.end())
        occupied.append(span)
        out.append(ParsedQuantity(magnitude=mag, unit=canon, span=span))

    # Avoid overriding explicit code-unit spans (e.g. "30s" / "30 seconds") with dimensionless numbers
    for pq in extract_quantities(text, explicit_unit=explicit_unit):
        if any(not (pq.span[1] <= s or pq.span[0] >= e) for s, e in occupied):
            continue
        # If explicit_unit is a code-specific unit (e.g. "seconds"), prefer our canonical code name
        eff_unit = pq.unit
        if explicit_unit is not None:
            code_u = _canon_code_or_std_unit(explicit_unit)
            if code_u is not None and pq.unit == canonicalize_unit(explicit_unit):
                eff_unit = code_u
        out.append(
            ParsedQuantity(
                magnitude=pq.magnitude,
                unit=eff_unit,
                is_fraction=pq.is_fraction,
                is_range_endpoint=pq.is_range_endpoint,
                span=pq.span,
            )
        )
    return out


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def _unsigned_number_in_text(tok: str, text: str) -> bool:
    pat = re.compile(
        r"(?<![A-Za-z0-9_])(?<!\d\.)" + re.escape(tok) + r"(?![A-Za-z0-9_])(?!\.\d)"
    )
    for m in pat.finditer(text):
        pos = m.start()
        if pos > 0 and text[pos - 1] == "-":
            if pos == 1 or not (text[pos - 2].isalnum() or text[pos - 2] in "_."):
                continue
        return True
    return False


def _token_in_code_text(tok: str, text: str) -> bool:
    if tok.startswith(("0x", "0X")):
        return _has_boundary_aligned_match(text, tok)
    if tok.startswith("-"):
        pat = re.compile(
            r"(?<![A-Za-z0-9_.])" + re.escape(tok) + r"(?![A-Za-z0-9_])(?!\.\d)"
        )
        return bool(pat.search(text))
    if tok[0].isdigit():
        return _unsigned_number_in_text(tok, text)
    return _has_boundary_aligned_match(text, tok)


def _operator_in_code_text(op: str, text: str) -> bool:
    if op == "==":
        return bool(re.search(r"(?<![=!<>])==(?!=)", text))
    if op == "!=":
        return bool(re.search(r"(?<![=!])!=(?!=)", text))
    if op == "<":
        return bool(re.search(r"(?<![-<=>])<(?![<=])", text))
    if op == ">":
        return bool(re.search(r"(?<![-<=>])>(?![>=])", text))
    return op in text


def verify_code_config_evidence(
    assertion: Assertion,
    source: Source | None,
    policy: Policy,
) -> list[Issue]:
    """Verify code, diff, or config assertions without legal-prose structural heuristics."""
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

    raw_q = (assertion.quote or "").strip()
    raw_src = source.text or ""
    quote_in_source = bool(raw_q) and _has_boundary_aligned_match(raw_src, raw_q)
    if not quote_in_source and raw_q:
        ws_q = _normalize_code_whitespace(raw_q)
        ws_src = _normalize_code_whitespace(raw_src)
        quote_in_source = bool(ws_q) and _has_boundary_aligned_match(ws_src, ws_q)

    if policy.require_exact_quote:
        if not raw_q:
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

        if not quote_in_source or not _quote_has_code_substance(raw_q):
            issues.append(
                Issue(
                    code="QUOTE_NOT_IN_SOURCE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Quote '{assertion.quote[:80]}' does not appear verbatim at token boundaries "
                        f"or lacks substantive code/config context in '{source.id}'."
                    ),
                    remediation=f"Extract an exact verbatim code/config substring from '{source.id}'.",
                )
            )
            return issues

    if not policy.require_quantity_match:
        return issues

    reference_text = assertion.quote if quote_in_source else raw_src

    # 1. Strict ISO date grounding (without rejecting filesystem paths like api/1/2/25/items)
    val_text = assertion.value or ""
    if _STANDALONE_SLASH_DATE_RE.search(val_text):
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

    val_dates = extract_dates(val_text, allow_slash=False)
    if val_dates:
        quote_dates = extract_dates(reference_text, allow_slash=False)
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
                            f"Date '{d}' in proposed value '{assertion.value}' is not grounded in '{source.id}'."
                        ),
                        remediation=f"Align date in '{assertion.predicate}' with '{source.id}'.",
                    )
                )
                return issues

    # 2. Explicit rate denominator & dimensioned quantity/unit matching
    def rate_period(text: str) -> str | None:
        m = re.search(r"(?:%|percent|bps)\s+(?:per\s+|/)([a-z]+)", text, re.IGNORECASE)
        if not m:
            return None
        tok = m.group(1).lower()
        return "YEARS" if tok == "annum" else canonicalize_unit(tok)

    proposed_period, cited_period = rate_period(val_text), rate_period(reference_text)
    if proposed_period and cited_period and proposed_period != cited_period:
        issues.append(
            Issue(
                code="UNIT_MISMATCH",
                disposition="RETRY",
                assertion_id=assertion.id,
                subject=assertion.subject,
                predicate=assertion.predicate,
                message="The proposed rate uses a different time denominator from the citation.",
                remediation="Preserve the rate period stated in the citation.",
            )
        )
        return issues

    val_quantities = _extract_code_quantities(val_text, explicit_unit=assertion.unit)
    ref_quantities = _extract_code_quantities(reference_text)
    verified_dim_spans: list[tuple[int, int]] = []

    for vq in val_quantities:
        has_dim = vq.unit != "NONE" or (
            assertion.unit is not None and unit_dimension(canonicalize_unit(assertion.unit)) != "NONE"
        )
        if not has_dim:
            continue
        matched = [
            rq
            for rq in ref_quantities
            if vq.matches(rq) and not vq.same_magnitude_wrong_unit(rq)
        ]
        if matched:
            verified_dim_spans.append(vq.span)
            continue
        unit_clash = next((rq for rq in ref_quantities if vq.same_magnitude_wrong_unit(rq)), None)
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
                    remediation=f"Use the exact unit '{unit_clash.unit}' from '{source.id}'.",
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
                    remediation=f"Use the exact value and unit stated in '{source.id}'.",
                )
            )
        return issues

    # 3. Strict case-sensitive identifier, number/semver, path, and operator grounding
    cleaned_val = _CALL_SUFFIX_RE.sub(r"\1", val_text)
    ws_ref = _normalize_code_whitespace(reference_text)

    for pat in (_PATH_COMPOUND_RE, _QUALIFIED_COMPOUND_RE):
        for m in pat.finditer(cleaned_val):
            comp = m.group(0)
            if not (
                _has_boundary_aligned_match(reference_text, comp)
                or _has_boundary_aligned_match(ws_ref, comp)
            ):
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Code/config path or qualified symbol '{comp}' in '{assertion.value}' "
                            f"is not grounded in cited quote '{assertion.quote[:80]}'."
                        ),
                        remediation=f"Ensure '{assertion.value}' appears verbatim in '{source.id}'.",
                    )
                )
                return issues

    token_matches = list(_CODE_TOKEN_RE.finditer(cleaned_val))
    if not token_matches:
        stripped_val = val_text.strip()
        if not stripped_val or stripped_val not in reference_text:
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
                    remediation=f"Cite a quote from '{source.id}' containing '{assertion.value}'.",
                )
            )
            return issues
    else:
        for tm in token_matches:
            if any(_spans_overlap(tm.span(), dspan) for dspan in verified_dim_spans):
                continue
            tok = tm.group(0)
            if not _token_in_code_text(tok, reference_text):
                issues.append(
                    Issue(
                        code="QUANTITY_MISMATCH",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Token '{tok}' in proposed value '{assertion.value}' does not match "
                            f"exact case/boundaries in cited quote '{assertion.quote[:80]}'."
                        ),
                        remediation=f"Use exact code/config identifiers and values from '{source.id}'.",
                    )
                )
                return issues

    for op_m in _COMP_OP_RE.finditer(val_text):
        op = op_m.group(0)
        if not _operator_in_code_text(op, reference_text):
            issues.append(
                Issue(
                    code="QUANTITY_MISMATCH",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Operator '{op}' in proposed value '{assertion.value}' is not grounded "
                        f"in cited quote '{assertion.quote[:80]}'."
                    ),
                    remediation=f"Preserve the exact operator stated in '{source.id}'.",
                )
            )
            return issues

    return issues


def is_code_config_assertion_self_consistent(assertion: Assertion) -> bool:
    """Verify intrinsic quote/value consistency for a code_config PASS verdict receipt."""
    if not assertion.quote or not assertion.quote.strip():
        return False
    synthetic = Source(
        id=assertion.source_id,
        text=assertion.quote,
        evidence_profile="code_config",
    )
    return not verify_code_config_evidence(assertion, synthetic, Policy())
