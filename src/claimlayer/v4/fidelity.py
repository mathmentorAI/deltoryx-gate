"""Extraction-fidelity controls for ClaimLayer v4.

Verifies per-field evidence bindings, clause-boundary completeness (ensuring
limiting provisos/exceptions are not dropped by fragment boundaries), and
independent structural checks on proposed propositions. Also provides
dual-reader agreement and correlated-error diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import EvidenceRef
from .models import EvidenceUnit

_CONDITIONAL_MARKER_RE = re.compile(
    r"\b(?:unless|except(?:\s+for|\s+that|\s+where)?|provided\s+that|subject\s+to|"
    r"notwithstanding|save\s+that|if\s+(?:and\s+only\s+if)?|when|where|in\s+the\s+event|"
    r"for\s+disputed|during\s+force\s+majeure|excluding|other\s+than|"
    r"salvo|excepto|siempre\s+que|a\s+menos\s+que|sujeto\s+a|en\s+caso\s+de)\b",
    re.IGNORECASE,
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


def check_clause_boundary_completeness(doc_text: str, ref: EvidenceRef) -> list[FidelityFinding]:
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

    if leading and (_CONDITIONAL_MARKER_RE.search(leading) or _CLAUSE_NEGATION_RE.search(leading)):
        findings.append(
            FidelityFinding(
                code="TRUNCATED_CLAUSE_CONTEXT",
                message="Cited span drops leading condition or negation in the same sentence.",
                field="conditions",
                detail={"dropped_prefix": leading},
            )
        )
    if trailing and _CONDITIONAL_MARKER_RE.search(trailing):
        findings.append(
            FidelityFinding(
                code="TRUNCATED_CLAUSE_CONTEXT",
                message="Cited span drops trailing proviso or exception in the same sentence.",
                field="conditions",
                detail={"dropped_suffix": trailing},
            )
        )
    return findings


def _value_markers_present(value: Any, text: str) -> bool:
    low = text.lower()
    if isinstance(value, dict) and "value" in value:
        num = str(value["value"])
        if num in text:
            return True
        if "." in num and num.rstrip("0").rstrip(".") in text:
            return True
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        tokens = [t for t in re.findall(r"[a-z0-9]+", value.lower()) if len(t) >= 3]
        return not tokens or all(t in low for t in tokens)
    return str(value).lower() in low


def verify_extraction_fidelity(
    doc_text: str,
    unit: EvidenceUnit,
    *,
    field_evidence: Sequence[FieldEvidence] | None = None,
    require_field_evidence: Sequence[str] = (),
) -> list[FidelityFinding]:
    """Run independent deterministic fidelity checks on an admitted EvidenceUnit."""
    findings: list[FidelityFinding] = []
    if not unit.evidence:
        return findings

    prop = unit.proposition
    full_sentences: list[str] = []
    for ref in unit.evidence:
        findings.extend(check_clause_boundary_completeness(doc_text, ref))
        _, _, sent = enclosing_sentence(doc_text, ref.start, ref.end)
        full_sentences.append(sent)

    combined_sentence = " ".join(full_sentences)
    cond_status = prop.get(
        "condition_status",
        "conditional" if prop.get("conditions") else "unconditional",
    )

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

    # 4. Per-field evidence binding checks
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
        if ref.start < 0 or ref.end > len(doc_text) or doc_text[ref.start : ref.end] != ref.quote:
            findings.append(
                FidelityFinding(
                    code="INVALID_FIELD_EVIDENCE",
                    message=f"FieldEvidence for '{fe.field}' does not match source text at declared offsets.",
                    field=fe.field,
                )
            )
            continue
        if fe.field == "value" and not _value_markers_present(prop.get("value"), ref.quote):
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
