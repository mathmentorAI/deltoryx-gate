"""Response-to-Graph Correspondence & Clause-Level Semantic Repair Verification.

Verifies whether a natural-language candidate response matches an explicit
JustificationGraph without relying on isolated keyword checks:
- Inspects target vs. non-target clauses to ensure qualifications (e.g.
  approximations, intervals, conditions, source attribution) attach to the
  concluding claim and remain consistent across the entire response.
- Detects self-contradictory hedging ("aproximadamente X ... exactamente X"),
  subject swaps, calculation/magnitude mutations, unit changes, dropped
  conditions, hidden material conflicts, unsupported factual additions, and
  question evasion vs. legitimate pruning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import SourceDocument, digest
from .justification import JustificationGraph
from .vocabulary import UnitRegistry, label_key

_UNICODE_MINUS_RE = re.compile(r"[−–—]")

_APPROX_MODIFIER_RE = re.compile(
    r"\b(aproximadamente|aprox\.?|alrededor de|cerca de|aproximaci[oó]n|"
    r"tendr[ií]a|podr[ií]a tener|estar[ií]a|habr[ií]a nacido|"
    r"approximately|approx\.?|roughly|around|about|estimated|roughly speaking)\b",
    re.IGNORECASE,
)

_RANGE_PATTERN_RE = re.compile(
    r"(?:\b(?:entre|between|de|from)\s+(-?\d+(?:[.,]\d+)?)\s+(?:y|a|and|to)\s+(-?\d+(?:[.,]\d+)?)\b|"
    r"\[\s*(-?\d+(?:[.,]\d+)?)\s*[-–,]\s*(-?\d+(?:[.,]\d+)?)\s*\]|"
    r"\b(-?\d+(?:[.,]\d+)?)\s*[–]\s*(-?\d+(?:[.,]\d+)?)\b|"
    r"(-?\d+(?:[.,]\d+)?)\s*(?:±|\+/-)\s*(\d+(?:[.,]\d+)?))",
    re.IGNORECASE,
)

_EXACT_ASSERTION_RE = re.compile(
    r"\b(exactamente|exacta|exacto|exact|exactly|precisamente|precisely|"
    r"con exactitud|de forma exacta|sabemos que tiene exactamente|sin lugar a dudas|"
    r"inequ[ií]vocamente|indiscutiblemente|definitivamente|total completo|total general)\b",
    re.IGNORECASE,
)

_NEGATED_LIMIT_CONTEXT_RE = re.compile(
    r"\b(no se puede|no es posible|imposible|no hay|no existen?|no se conocen?|no se sabe|"
    r"no basta|no alcanza|impide|sin|faltan?|falta de|carece de|no permite|no puede afirmarse|"
    r"no es exact[ao]|no puede determinarse|cannot|can't|not possible|unable to|without|"
    r"insufficient|not verified as|not exact|cannot determine|impossible to determine)\b",
    re.IGNORECASE,
)

_MISSING_PRECISION_DISCLOSURE_RE = re.compile(
    r"\b(sin fechas completas|fechas completas|sin fecha exacta|falta.*fecha|"
    r"no se conoce la fecha|granularidad|fecha parcial|solo se conoce el a[nñ]o|"
    r"without full dates|without exact dates|missing exact date|coarse date|partial date|"
    r"poblaci[oó]n incompleta|sin constancia de completitud|incomplete population|"
    r"no representan necesariamente el total)\b",
    re.IGNORECASE,
)

_SOURCE_ATTRIBUTION_RE = re.compile(
    r"\b(seg[uú]n el texto|seg[uú]n los textos|seg[uú]n los datos|seg[uú]n el documento|"
    r"seg[uú]n los documentos|de acuerdo con el texto|de acuerdo con los textos|"
    r"de acuerdo con el documento|de acuerdo con los documentos|seg[uú]n la fuente|"
    r"seg[uú]n las fuentes|seg[uú]n la evidencia|seg[uú]n las evidencias|"
    r"seg[uú]n las facturas|seg[uú]n lo aportado|con base en el texto|"
    r"according to the text|according to the texts|according to the document|"
    r"according to the documents|based on the provided text|based on the provided texts|"
    r"from the supplied evidence|from the supplied texts|stated in the text|"
    r"no se ha comprobado.*externamente)\b",
    re.IGNORECASE,
)

_CONTRADICTORY_EXTERNAL_TRUTH_RE = re.compile(
    r"\b(hecho hist[oó]rico verificado|comprobado externamente|contrastado externamente|"
    r"verdad emp[ií]rica demostrada|independently verified historical fact|"
    r"externally confirmed beyond doubt|proven externally)\b",
    re.IGNORECASE,
)

_SUBTOTAL_QUALIFICATION_RE = re.compile(
    r"\b(subtotal|suma parcial|parcial|de los elementos aportados|de las facturas listadas|"
    r"partial sum|listed items only|subtotal only)\b",
    re.IGNORECASE,
)

_UNQUALIFIED_TOTAL_RE = re.compile(
    r"\b(el total es|total general|total completo|monto total definitivo|total amount is|full total|grand total)\b",
    re.IGNORECASE,
)

_CONFLICT_DISCLOSURE_RE = re.compile(
    r"\b(conflicto|contradicci[oó]n|contradictorio|discrepancia|fuentes divergentes|"
    r"sin resolver|conflict|contradict|discrepancy|disagree|conflicting sources|"
    r"divergent|inconclusive)\b",
    re.IGNORECASE,
)

_CONDITIONAL_MARKER_RE = re.compile(
    r"\b(si|salvo|siempre que|a menos que|condicionad[oa]|sujet[oa] a|excepto|"
    r"if|unless|subject to|provided that|conditional upon|contingent on|except)\b",
    re.IGNORECASE,
)

_EVASION_NO_ANSWER_RE = re.compile(
    r"\b(no puedo responder|no es posible responder|prefiero no responder|"
    r"no dar[eé] una respuesta|no se aborda|omito responder|"
    r"cannot answer the question|will not answer|unable to provide any estimate)\b",
    re.IGNORECASE,
)

_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_.-])(-?\d+(?:[.,]\d+)?)(?![A-Za-z0-9_.-])")
_YEAR_OR_DATE_TOKEN_RE = re.compile(r"\b(\d{4}-\d{2}(?:-\d{2})?|(?:1[5-9]\d{2}|20\d{2}|21\d{2}))\b")

_COMMON_STOPWORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "en", "con", "por", "para",
    "que", "se", "su", "sus", "al", "es", "son", "tiene", "tendria", "tendría", "esta", "está",
    "estamos", "y", "o", "si", "no", "sin", "sobre", "entre", "desde", "hasta", "como", "segun",
    "según", "texto", "documento", "calculo", "cálculo", "aproximadamente", "edad", "anos", "años",
    "ano", "año", "fecha", "fechas", "completas", "exacta", "exacto", "cuando", "era", "fue",
    "subtotal", "total", "condicion", "condición", "salvo", "pero", "aunque", "debido", "dado",
    "ademas", "además", "embargo", "entonces", "luego", "ambas", "fuentes", "conflicto",
    "the", "and", "for", "with", "that", "this", "from", "into", "when", "was", "were", "has",
    "have", "would", "could", "should", "about", "approx", "approximately", "according", "text",
    "document", "year", "years", "age", "old", "exact", "dates", "full", "what", "how", "many",
    "given", "based", "however", "therefore", "thus", "note", "calculation", "without", "partial",
})

_UNIT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "years": ("año", "años", "year", "years", "yr", "yrs"),
    "months": ("mes", "meses", "month", "months"),
    "calendar_days": ("día", "días", "day", "days", "calendar days", "días naturales"),
    "business_days": ("días hábiles", "días laborables", "business day", "business days", "working days"),
    "USD": ("usd", "dólar", "dólares", "dollar", "dollars", "$"),
    "EUR": ("eur", "euro", "euros", "€"),
    "GBP": ("gbp", "libra", "libras", "pound", "pounds", "£"),
    "percent": ("%", "por ciento", "percent", "pct"),
    "hours": ("hora", "horas", "hour", "hours", "hr", "hrs"),
    "minutes": ("minuto", "minutos", "minute", "minutes", "min"),
}


def compute_response_digest(
    candidate_response: str | None,
    question: str | Mapping[str, Any] | None = None,
) -> str | None:
    """Compute canonical digest binding exact response text and question."""
    if candidate_response is None:
        return None
    q_norm = (
        question
        if isinstance(question, str)
        else (str(question.get("text") or question.get("question") or "") if isinstance(question, Mapping) else "")
    )
    return digest({"question": q_norm, "candidate_response": candidate_response})


def _to_dec(raw: Any) -> Decimal | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, Decimal):
        return raw if raw.is_finite() else None
    try:
        s = _UNICODE_MINUS_RE.sub("-", str(raw).strip()).replace(",", ".")
        d = Decimal(s)
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _split_clauses(text: str) -> list[str]:
    norm = _UNICODE_MINUS_RE.sub("-", text or "")
    parts = re.split(r"(?:[.;!?\n]+|,\s*(?=(?:por tanto|por lo tanto|as[ií] que|therefore|hence|thus|so)\b))", norm)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_clause_numbers(clause: str) -> list[Decimal]:
    out: list[Decimal] = []
    for m in _NUMBER_TOKEN_RE.finditer(clause):
        d = _to_dec(m.group(1))
        if d is not None:
            out.append(d)
    return out


def _extract_ranges(text: str) -> list[tuple[Decimal, Decimal]]:
    ranges: list[tuple[Decimal, Decimal]] = []
    for m in _RANGE_PATTERN_RE.finditer(text):
        g = m.groups()
        if g[0] is not None and g[1] is not None:
            a, b = _to_dec(g[0]), _to_dec(g[1])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[2] is not None and g[3] is not None:
            a, b = _to_dec(g[2]), _to_dec(g[3])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[4] is not None and g[5] is not None:
            a, b = _to_dec(g[4]), _to_dec(g[5])
            if a is not None and b is not None and a <= b:
                ranges.append((a, b))
        elif g[6] is not None and g[7] is not None:
            center, tol = _to_dec(g[6]), _to_dec(g[7])
            if center is not None and tol is not None and tol >= 0:
                ranges.append((center - tol, center + tol))
    return ranges


def _clause_asserts_exactness(clause: str) -> bool:
    for m in _EXACT_ASSERTION_RE.finditer(clause):
        prefix = clause[: m.start()]
        # Check whether the exactness phrase appears inside a disclaimer explaining why
        # exact determination is NOT possible (e.g. "Sin fechas completas no se puede determinar la edad exacta")
        if _NEGATED_LIMIT_CONTEXT_RE.search(prefix):
            continue
        return True
    return False


def _clause_is_explanatory_formula_or_disclaimer(clause: str) -> bool:
    low = clause.lower()
    if _NEGATED_LIMIT_CONTEXT_RE.search(low) or _MISSING_PRECISION_DISCLOSURE_RE.search(low):
        return True
    if re.search(r"(c[aá]lculo|formula|f[oó]rmula|calculation|compute)\s+(?:es|is)\s+\d+\s*[+\-*/]", low):
        return True
    if "=" in clause and any(op in clause for op in ("+", "-", "*", "/")):
        return True
    return False


def verify_response_graph_correspondence(
    candidate_response: str | None,
    graph: JustificationGraph,
    *,
    question: str | Mapping[str, Any] | None = None,
    documents: Mapping[str, SourceDocument] | None = None,
    unit_registry: UnitRegistry | None = None,
    expected_subject: str | None = None,
    has_material_conflict: bool = False,
) -> dict[str, Any]:
    """Analyze clause-level semantic & structural alignment between text and graph."""
    if candidate_response is None:
        return {
            "response_checked": False,
            "correspondence_ok": False,
            "issues": (),
        }

    text = _UNICODE_MINUS_RE.sub("-", candidate_response).strip()
    q_text = (
        _UNICODE_MINUS_RE.sub("-", question)
        if isinstance(question, str)
        else (
            _UNICODE_MINUS_RE.sub("-", str(question.get("text") or question.get("question") or ""))
            if isinstance(question, Mapping)
            else ""
        )
    )
    conc = graph.claims[graph.conclusion_id]
    anc_ids = graph.ancestor_claims(graph.conclusion_id)
    anc_nodes = [graph.claims[aid] for aid in sorted(anc_ids) if aid in graph.claims]
    clauses = _split_clauses(text)
    issues: list[dict[str, Any]] = []

    if not text:
        issues.append({
            "code": "EMPTY_CANDIDATE_RESPONSE",
            "reason": "Candidate response text is empty.",
            "severity": "error",
        })
        return {
            "response_checked": True,
            "correspondence_ok": False,
            "issues": tuple(issues),
        }

    # Gather expected target numbers / interval from conclusion
    conc_val = _to_dec(conc.value)
    conc_interval: tuple[Decimal, Decimal] | None = None
    if isinstance(conc.value, Mapping) and "min" in conc.value and "max" in conc.value:
        lo_d, hi_d = _to_dec(conc.value["min"]), _to_dec(conc.value["max"])
        if lo_d is not None and hi_d is not None:
            conc_interval = (lo_d, hi_d)

    all_nums_in_text = _extract_clause_numbers(text)
    ranges_in_text = _extract_ranges(text)

    # Known safe numbers present in documents, question, premise operands, or conclusion itself
    known_doc_nums: set[Decimal] = set()
    for doc in (documents or {}).values():
        known_doc_nums.update(_extract_clause_numbers(doc.text))
    if q_text:
        known_doc_nums.update(_extract_clause_numbers(q_text))
    for node in anc_nodes + [conc]:
        nv = _to_dec(node.value)
        if nv is not None:
            known_doc_nums.add(nv)
        elif isinstance(node.value, Mapping):
            for k in ("min", "max", "low", "high", "value"):
                d = _to_dec(node.value.get(k))
                if d is not None:
                    known_doc_nums.add(d)
    # Include derived differences & percentage fractions for transparent step explanations
    doc_years = {int(n) for n in known_doc_nums if 1500 <= n <= 2150 and n == int(n)}
    for y1 in doc_years:
        for y2 in doc_years:
            if y1 > y2:
                known_doc_nums.add(Decimal(y1 - y2))
    for n in list(known_doc_nums):
        if abs(n) <= Decimal(1000) and n != 0:
            known_doc_nums.add(n / Decimal(100))

    # 1. Precision / approximation consistency across ALL clauses
    approx_clauses: list[str] = []
    target_approx_clauses: list[str] = []
    exact_clauses: list[str] = []
    unqualified_target_clauses: list[str] = []

    target_vals_to_match: list[Decimal] = []
    if conc_val is not None:
        target_vals_to_match.append(conc_val)
    elif conc_interval is not None:
        target_vals_to_match.extend([conc_interval[0], conc_interval[1]])

    for cl in clauses:
        has_approx = bool(_APPROX_MODIFIER_RE.search(cl) or _extract_ranges(cl))
        has_exact = _clause_asserts_exactness(cl)
        cl_nums = _extract_clause_numbers(cl)
        is_disclaimer_or_formula = _clause_is_explanatory_formula_or_disclaimer(cl)
        mentions_target_val = bool(
            target_vals_to_match
            and any(any(abs(cn - tv) <= Decimal("0.0001") for tv in target_vals_to_match) for cn in cl_nums)
        )
        mentions_target_unit = bool(
            conc.unit
            and any(syn in cl.lower() for syn in _UNIT_SYNONYMS.get(conc.unit, (conc.unit.lower(),)))
        )
        is_target_assertion_clause = (mentions_target_val or (ranges_in_text and _extract_ranges(cl))) and not is_disclaimer_or_formula

        if has_approx:
            approx_clauses.append(cl)
            if is_target_assertion_clause or mentions_target_unit:
                target_approx_clauses.append(cl)
        if has_exact:
            exact_clauses.append(cl)
        if is_target_assertion_clause and not has_approx:
            # Bare assertion like "Tiene 106 años" without approximation/interval
            unqualified_target_clauses.append(cl)

    contradictory_precision = bool(approx_clauses and exact_clauses)
    if exact_clauses and any(_extract_ranges(c) for c in clauses):
        contradictory_precision = True

    if contradictory_precision:
        issues.append({
            "code": "CONTRADICTORY_PRECISION_QUALIFICATION",
            "reason": "Response asserts both an approximation/interval and an exact claim in the same answer.",
            "severity": "error",
            "clauses": {"approx": approx_clauses, "exact": exact_clauses},
        })

    # Response is properly approximated/ranged ONLY if target claim clause carries qualification AND no clause asserts exactness
    response_approximated_or_ranged = bool(target_approx_clauses and not contradictory_precision and not exact_clauses)
    response_claims_exact_unqualified = bool(exact_clauses or (unqualified_target_clauses and not target_approx_clauses))

    # 2. Source attribution check across clauses
    has_source_attr = bool(_SOURCE_ATTRIBUTION_RE.search(text))
    has_contradictory_ext_truth = bool(_CONTRADICTORY_EXTERNAL_TRUTH_RE.search(text))
    source_qualified = bool(has_source_attr and not has_contradictory_ext_truth)
    if has_source_attr and has_contradictory_ext_truth:
        issues.append({
            "code": "CONTRADICTORY_SOURCE_ATTRIBUTION",
            "reason": "Response cites the supplied text while also claiming independently verified external truth.",
            "severity": "error",
        })

    # 3. Subject preservation check
    subj = expected_subject or conc.subject
    if not subj:
        for n in anc_nodes:
            if n.subject:
                subj = n.subject
                break
    subject_changed = False
    if subj:
        subj_tokens = [t for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", subj.lower()) if len(t) >= 3 and t not in _COMMON_STOPWORDS]
        if subj_tokens:
            low_text = text.lower()
            subj_in_text = any(tok in low_text for tok in subj_tokens)
            # Check for ungrounded entities OR entity swap to another subject in documents when expected subject is missing
            allowed_cap_words = set(subj_tokens)
            other_doc_entities: set[str] = set()
            unit_words_low = {s.lower() for syns in _UNIT_SYNONYMS.values() for s in syns}
            for doc in (documents or {}).values():
                doc_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", doc.text)
                allowed_cap_words.update(w.lower() for w in doc_words)
                for m_d in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})\b", doc.text):
                    w_low = m_d.group(1).lower()
                    if w_low not in _COMMON_STOPWORDS and w_low not in subj_tokens and w_low not in unit_words_low:
                        other_doc_entities.add(w_low)
            if q_text:
                allowed_cap_words.update(w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", q_text))
            allowed_cap_words.update(_COMMON_STOPWORDS)

            # Collect capitalized tokens appearing in response (supporting sentence-boundary check cleanly)
            for m in re.finditer(r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})*)\b", text):
                cand_words = [w.lower() for w in m.group(1).split()]
                prev_char = text[m.start() - 1] if m.start() > 0 else ""
                prev_two = text[max(0, m.start() - 2):m.start()].strip()
                at_sentence_start = m.start() == 0 or prev_char == "\n" or prev_two in {".", "!", "?", ":"}
                if len(cand_words) >= 2 and any(w not in allowed_cap_words for w in cand_words):
                    subject_changed = True
                    break
                if len(cand_words) == 1 and not at_sentence_start:
                    if cand_words[0] not in allowed_cap_words:
                        subject_changed = True
                        break
            if not subj_in_text and other_doc_entities:
                # If expected subject vanished and response names a DIFFERENT document entity, flag subject swap
                for other_ent in other_doc_entities:
                    if re.search(rf"\b{re.escape(other_ent)}\b", low_text):
                        subject_changed = True
                        break

            if subject_changed:
                issues.append({
                    "code": "SUBJECT_CHANGED_IN_REPAIR",
                    "reason": f"Response altered subject '{subj}' or introduced an ungrounded entity.",
                    "severity": "error",
                })

    # 4. Calculation / magnitude fidelity & evasion detection
    evaded = bool(_EVASION_NO_ANSWER_RE.search(text))
    calc_mismatch = False
    if evaded and not has_material_conflict:
        issues.append({
            "code": "QUESTION_LEFT_UNANSWERED_EVASION",
            "reason": "Response omits or refuses to answer the core questioned claim instead of qualifying limitations.",
            "severity": "error",
        })
    elif target_vals_to_match and not evaded:
        # Check if the concluding value (or declared range endpoints) appears in target assertion clauses
        asserted_nums: list[Decimal] = []
        for cl in clauses:
            if not _clause_is_explanatory_formula_or_disclaimer(cl):
                asserted_nums.extend(_extract_clause_numbers(cl))
        if not asserted_nums:
            asserted_nums = all_nums_in_text

        matched_graph_val = False
        for tv in target_vals_to_match:
            if any(abs(an - tv) <= Decimal("0.01") for an in asserted_nums):
                matched_graph_val = True
                break
        if conc_interval is not None and ranges_in_text:
            for r_lo, r_hi in ranges_in_text:
                if abs(r_lo - conc_interval[0]) <= Decimal("0.01") and abs(r_hi - conc_interval[1]) <= Decimal("0.01"):
                    matched_graph_val = True
                    break

        if not matched_graph_val:
            if asserted_nums:
                calc_mismatch = True
                issues.append({
                    "code": "CALCULATION_ALTERED_OR_INCONSISTENT",
                    "reason": f"Response numbers { [str(x) for x in asserted_nums] } do not match verified conclusion value {conc.value!r}.",
                    "severity": "error",
                })
            else:
                issues.append({
                    "code": "QUESTION_LEFT_UNANSWERED_EVASION",
                    "reason": "Response removed the concluding value without addressing the question.",
                    "severity": "error",
                })

    # 5. Unit preservation check (detect unit dimension swap e.g. USD -> EUR or years -> days)
    if conc.unit:
        low_text = text.lower()
        target_unit_syns = _UNIT_SYNONYMS.get(conc.unit, (conc.unit.lower(),))

        def _unit_tok_re(tok: str) -> str:
            return rf"(?<!\w){re.escape(tok)}(?!\w)" if tok.isalpha() else re.escape(tok)

        target_unit_present = any(re.search(_unit_tok_re(s), low_text) for s in target_unit_syns)
        wrong_units_detected: list[str] = []
        for uname, syns in _UNIT_SYNONYMS.items():
            if uname == conc.unit:
                continue
            # Only flag another unit if it's in a different dimension AND wasn't present in premise/docs/question
            if unit_registry is not None:
                try:
                    u_c = unit_registry.lookup(conc.unit)
                    u_o = unit_registry.lookup(uname)
                    if u_c.dimension == u_o.dimension:
                        continue
                except Exception:
                    pass
            bound_to_number = any(
                re.search(
                    rf"(?:\d+(?:[.,]\d+)?\s*(?:(?:a|y|-|to)\s*\d+(?:[.,]\d+)?\s*)?{_unit_tok_re(s)}|{_unit_tok_re(s)}\s*\d+)",
                    low_text,
                )
                for s in syns
            )
            bare_present = any(re.search(_unit_tok_re(s), low_text) for s in syns)
            in_resp = bound_to_number or (bare_present and not target_unit_present)
            in_sources = any(
                any(s in d.text.lower() for s in syns) for d in (documents or {}).values()
            ) or (q_text and any(s in q_text.lower() for s in syns))
            if in_resp and not in_sources:
                wrong_units_detected.append(uname)
        if wrong_units_detected:
            issues.append({
                "code": "UNIT_ALTERED_IN_REPAIR",
                "reason": f"Response uses unexpected unit(s) {wrong_units_detected} instead of '{conc.unit}'.",
                "severity": "error",
            })

    # 6. Condition preservation check
    req_conds = list(conc.conditions)
    for n in anc_nodes:
        req_conds.extend(n.conditions)
    condition_dropped = False
    if req_conds:
        has_cond_marker = bool(_CONDITIONAL_MARKER_RE.search(text))
        cond_terms_found = False
        for cond in req_conds:
            for k in ("field", "value"):
                cv = cond.get(k)
                if isinstance(cv, str) and len(cv) >= 3:
                    parts = [p for p in re.split(r"[_\s]+", cv.lower()) if len(p) >= 3]
                    if any(p in text.lower() for p in parts):
                        cond_terms_found = True
        if not (has_cond_marker or cond_terms_found):
            condition_dropped = True
            issues.append({
                "code": "CONDITION_DROPPED_IN_RESPONSE",
                "reason": "Response omits mandatory conditions or exceptions carried by the evidence.",
                "severity": "error",
            })

    # 7. Subtotal vs unqualified total check
    qualified_as_subtotal = bool(_SUBTOTAL_QUALIFICATION_RE.search(text))
    unqualified_total_claim = False
    for cl in clauses:
        if _UNQUALIFIED_TOTAL_RE.search(cl) and not _NEGATED_LIMIT_CONTEXT_RE.search(cl):
            unqualified_total_claim = True
            break
    presents_subtotal_as_total = bool(unqualified_total_claim and not qualified_as_subtotal)

    # 8. Material conflict disclosure check
    conflict_disclosed = bool(_CONFLICT_DISCLOSURE_RE.search(text))
    if has_material_conflict and not conflict_disclosed:
        issues.append({
            "code": "CONFLICT_HIDDEN_IN_RESPONSE",
            "reason": "Response fails to disclose unresolved contradicting evidence.",
            "severity": "error",
        })

    # 9. Unsupported factual addition check (ungrounded dates / unexpected quantities)
    doc_and_q_text = " ".join([d.text for d in (documents or {}).values()] + ([q_text] if q_text else []))
    doc_dates = set(_YEAR_OR_DATE_TOKEN_RE.findall(doc_and_q_text))
    resp_dates = set(_YEAR_OR_DATE_TOKEN_RE.findall(text))
    invented_dates = sorted(
        d for d in resp_dates
        if d not in doc_dates and _to_dec(d) != conc_val
    )
    if invented_dates:
        issues.append({
            "code": "UNSUPPORTED_FACTUAL_ADDITION",
            "reason": f"Response introduces ungrounded dates/years not present in evidence: {invented_dates}.",
            "severity": "error",
        })

    unexpected_nums = [
        str(n) for n in all_nums_in_text
        if not any(abs(n - kn) <= Decimal("0.0001") for kn in known_doc_nums)
        and not (ranges_in_text and any(lo <= n <= hi for lo, hi in ranges_in_text))
    ]
    if unexpected_nums and not calc_mismatch:
        issues.append({
            "code": "UNSUPPORTED_FACTUAL_ADDITION",
            "reason": f"Response introduces ungrounded numeric values not derivable from evidence: {unexpected_nums}.",
            "severity": "error",
        })

    correspondence_ok = len(issues) == 0
    return {
        "response_checked": True,
        "correspondence_ok": correspondence_ok,
        "response_approximated_or_ranged": response_approximated_or_ranged,
        "response_claims_exact_unqualified": response_claims_exact_unqualified,
        "contradictory_precision_qualification": contradictory_precision,
        "source_qualified": source_qualified,
        "contradictory_source_attribution": has_source_attr and has_contradictory_ext_truth,
        "condition_dropped_in_response": condition_dropped,
        "qualified_as_subtotal": qualified_as_subtotal,
        "presents_subtotal_as_total": presents_subtotal_as_total,
        "conflict_disclosed": conflict_disclosed,
        "has_missing_precision_disclosure": bool(_MISSING_PRECISION_DISCLOSURE_RE.search(text)),
        "issues": tuple(issues),
    }
