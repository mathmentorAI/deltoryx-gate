"""Corpus-derived verification obligations and reference register (v4).

Computes deterministic verification obligations directly from admitted
``EvidenceLayer`` units and the domain ``Vocabulary`` before evaluating any
agent-supplied graph, assertions, or metadata flags.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..semantic.models import canonical, digest
from .engine import EvidenceLayer
from .justification import VerificationObligation
from .vocabulary import Vocabulary, label_key, value_relation


@dataclass(frozen=True)
class CorpusReferenceRegister:
    """Reference register computed solely from admitted corpus evidence."""

    available: bool
    entity: str | None = None
    required_predicates: tuple[str, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    document_ids: tuple[str, ...] = ()
    unit_count: int = 0
    has_material_conflicts: bool = False
    has_authority_overrides: bool = False
    reason: str = ""
    incomplete: bool = False
    incomplete_documents: tuple[str, ...] = ()
    document_statuses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    gap_divergences: tuple[Mapping[str, Any], ...] = ()
    section_divergences: tuple[Mapping[str, Any], ...] = ()
    admitted_unit_refs: tuple[Mapping[str, Any], ...] = ()

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(o.obligation_id for o in self.obligations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "entity": self.entity,
            "required_predicates": list(self.required_predicates),
            "obligation_ids": list(self.obligation_ids),
            "document_ids": list(self.document_ids),
            "unit_count": self.unit_count,
            "has_material_conflicts": self.has_material_conflicts,
            "has_authority_overrides": self.has_authority_overrides,
            "reason": self.reason,
            "incomplete": self.incomplete,
            "incomplete_documents": list(self.incomplete_documents),
            "document_statuses": {k: dict(v) for k, v in self.document_statuses.items()},
            "gap_divergences": [dict(g) for g in self.gap_divergences],
            "section_divergences": [dict(s) for s in self.section_divergences],
        }


def _unit_authority_rank(layer: EvidenceLayer, unit: Any) -> int:
    ranks: list[int] = []
    for ref in getattr(unit, "evidence", ()) or ():
        try:
            doc = layer._document(ref.document_id, ref.revision, current=False)
            ranks.append(int(doc.authority_rank))
        except Exception:
            continue
    return max(ranks) if ranks else 0


def _unit_spans(unit: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for ref in getattr(unit, "evidence", ()) or ():
        spans.append({
            "document_id": str(ref.document_id),
            "revision": str(ref.revision),
            "start": int(ref.start),
            "end": int(ref.end),
            "quote": str(ref.quote),
        })
    return spans


def _units_incompatible(vocabulary: Vocabulary, spec: Any, u1: Any, u2: Any) -> bool:
    p1 = u1.proposition
    p2 = u2.proposition
    if u1.assertion_key == u2.assertion_key:
        return False
    u1_unit = p1.get("unit")
    u2_unit = p2.get("unit")
    if u1_unit is not None and u2_unit is not None:
        if vocabulary.units.dimension_of(u1_unit) != vocabulary.units.dimension_of(u2_unit):
            # Incompatible counting conventions on same predicate (e.g. calendar_days vs business_days)
            if p1.get("predicate") == p2.get("predicate") and u1_unit != u2_unit:
                return True
            return False
    rel = value_relation(spec.value, p1.get("value"), p2.get("value"))
    if rel == "equal":
        return p1.get("polarity") != p2.get("polarity")
    return rel == "incompatible"


def _spans_overlap(
    doc_a: str, span_a: tuple[int, int] | None, doc_b: str, span_b: tuple[int, int] | None
) -> bool:
    if not doc_a or doc_a != doc_b:
        return False
    if span_a is None or span_b is None:
        return True
    s1, e1 = int(span_a[0]), int(span_a[1])
    s2, e2 = int(span_b[0]), int(span_b[1])
    return s1 < e2 and s2 < e1


def _obligation_overlaps_doc_span(
    ob: VerificationObligation, doc_id: str, span: tuple[int, int] | None
) -> bool:
    spans = (ob.parameters or {}).get("evidence_spans") or ()
    for sp in spans:
        sp_doc = str(sp.get("document_id", ""))
        sp_range = (
            (int(sp["start"]), int(sp["end"]))
            if sp.get("start") is not None and sp.get("end") is not None
            else None
        )
        if _spans_overlap(sp_doc, sp_range, doc_id, span):
            return True
    return False


def _divergence_covered_by_reader_obligations(
    div: Any, reader_obligations: Sequence[VerificationObligation]
) -> bool:
    pri_id = str(getattr(div, "primary_source_id", "") or "")
    sec_id = str(getattr(div, "secondary_source_id", "") or "")
    pri_span = getattr(div, "primary_span", None)
    sec_span = getattr(div, "secondary_span", None)
    pri_quants = tuple(getattr(div, "primary_quantities", ()) or ())
    sec_quants = tuple(getattr(div, "secondary_quantities", ()) or ())

    pri_matches = (
        [ob for ob in reader_obligations if _obligation_overlaps_doc_span(ob, pri_id, pri_span)]
        if pri_id and pri_span is not None
        else []
    )
    sec_matches = (
        [ob for ob in reader_obligations if _obligation_overlaps_doc_span(ob, sec_id, sec_span)]
        if sec_id and sec_span is not None
        else []
    )

    if pri_span is not None and sec_span is not None:
        if not pri_matches or not sec_matches:
            return False
        # If both sections have distinct quantitative tokens, require that the reader
        # extracted at least one overlapping conflict or override obligation (not just an unrelated shared number).
        if pri_quants and sec_quants and set(pri_quants) != set(sec_quants):
            has_conflict_ob = any(
                (ob.parameters or {}).get("corpus_obligation_kind")
                in ("material_conflict", "authority_override")
                for ob in (pri_matches + sec_matches)
            )
            return has_conflict_ob
        return True
    if pri_span is not None:
        return bool(pri_matches)
    if sec_span is not None:
        return bool(sec_matches)
    return False


def build_corpus_reference_register(
    evidence: EvidenceLayer | None,
    vocabulary: Vocabulary,
    *,
    question: str | Mapping[str, Any] | None = None,
    entity: str | None = None,
    required_predicates: Sequence[str] | None = None,
    comparison_divergences: Sequence[Any] = (),
    min_relevant_authority_rank: int = 1,
    expected_documents: Mapping[str, Any] | Sequence[str] | None = None,
) -> CorpusReferenceRegister:
    """Build a deterministic reference register and ``origin='corpus'`` obligations.

    Does not inspect the candidate response, justification graph, assertions, or
    any agent-supplied metadata flags.
    """
    if not isinstance(evidence, EvidenceLayer):
        return CorpusReferenceRegister(
            available=False,
            entity=entity,
            reason="REFERENCE_REGISTER_MISSING: no EvidenceLayer available.",
        )

    doc_statuses = {k: dict(v) for k, v in (getattr(evidence, "document_read_statuses", {}) or {}).items()}
    if expected_documents is not None:
        if isinstance(expected_documents, Mapping):
            for doc_id, sdoc in expected_documents.items():
                did = str(doc_id)
                if did not in doc_statuses:
                    rank = int(getattr(sdoc, "authority_rank", min_relevant_authority_rank))
                    rev = str(getattr(sdoc, "revision", "") or "")
                    doc_statuses[did] = {
                        "document_id": did,
                        "revision": rev,
                        "authority_rank": rank,
                        "status": "not_read",
                        "errors": [],
                    }
        else:
            for doc_id in expected_documents:
                did = str(doc_id)
                if did not in doc_statuses:
                    doc_statuses[did] = {
                        "document_id": did,
                        "revision": "",
                        "authority_rank": int(min_relevant_authority_rank),
                        "status": "not_read",
                        "errors": [],
                    }
    relevant_incomplete = tuple(
        sorted(
            doc_id
            for doc_id, st in doc_statuses.items()
            if int(st.get("authority_rank", 0)) >= int(min_relevant_authority_rank)
            and st.get("status") != "read_ok"
        )
    )
    has_read_failures = any(
        st.get("status") == "read_failed"
        for doc_id, st in doc_statuses.items()
        if int(st.get("authority_rank", 0)) >= int(min_relevant_authority_rank)
    )
    has_reader_incomplete = bool(
        has_read_failures or (relevant_incomplete and getattr(evidence, "reader", None) is not None)
    )

    admitted = sorted(evidence.units, key=lambda u: str(u.unit_id))
    if not admitted:
        return CorpusReferenceRegister(
            available=False,
            entity=entity,
            reason="REFERENCE_REGISTER_MISSING: EvidenceLayer has no admitted units.",
            incomplete=has_reader_incomplete,
            incomplete_documents=relevant_incomplete if has_reader_incomplete else (),
            document_statuses=doc_statuses,
        )

    target_entity = entity
    if target_entity is None and isinstance(question, Mapping):
        raw_ent = question.get("entity")
        if isinstance(raw_ent, str) and raw_ent.strip():
            target_entity = raw_ent.strip()

    # Determine mandatory predicates from explicit override or Vocabulary.required flags
    if required_predicates is not None:
        req_preds = tuple(dict.fromkeys(str(p) for p in required_predicates))
    else:
        vocab_req = tuple(p.name for p in vocabulary.predicates if getattr(p, "required", False))
        req_preds = vocab_req if vocab_req else tuple(
            sorted({str(u.proposition.get("predicate", "")) for u in admitted if u.proposition.get("predicate")})
        )

    doc_ids = tuple(sorted({
        str(ref.document_id)
        for u in admitted
        for ref in (getattr(u, "evidence", ()) or ())
    }))

    # Filter units for entity if target_entity is specified and matches at least one unit
    relevant_units = admitted
    if target_entity is not None:
        ent_units = [u for u in admitted if str(u.proposition.get("entity", "")) == target_entity]
        if ent_units:
            relevant_units = ent_units

    obligations: list[VerificationObligation] = []
    seen_ids: set[str] = set()

    # (i) Mandatory predicates that have at least one admissible unit
    by_predicate: dict[str, list[Any]] = {}
    for u in relevant_units:
        pred = str(u.proposition.get("predicate", ""))
        if pred:
            by_predicate.setdefault(pred, []).append(u)

    for pred in req_preds:
        pred_units = by_predicate.get(pred, ())
        if not pred_units:
            continue
        # Pick highest-authority unit(s) as primary source span anchor
        ranked = sorted(
            pred_units,
            key=lambda u: (-_unit_authority_rank(evidence, u), str(u.unit_id)),
        )
        top_unit = ranked[0]
        ctx_key = str(top_unit.context_key)
        ob_id = f"corpus::predicate::{pred}::{digest(ctx_key)[:12]}"
        if ob_id in seen_ids:
            continue
        seen_ids.add(ob_id)
        obligations.append(
            VerificationObligation(
                obligation_id=ob_id,
                family="document_authority",
                description=(
                    f"Cover admitted corpus predicate '{pred}' for entity "
                    f"'{top_unit.proposition.get('entity', '')}'."
                ),
                critical=True,
                origin="corpus",
                parameters={
                    "corpus_obligation_kind": "mandatory_predicate",
                    "entity": str(top_unit.proposition.get("entity", "")),
                    "predicate": pred,
                    "context_key": ctx_key,
                    "expected_value": top_unit.proposition.get("value"),
                    "expected_unit": top_unit.proposition.get("unit"),
                    "unit_ids": [str(u.unit_id) for u in ranked],
                    "evidence_spans": _unit_spans(top_unit),
                },
            )
        )

    # Group by predicate to detect same-context and compatible-scope cross-document conflicts/overrides
    by_pred: dict[str, list[Any]] = {}
    for u in relevant_units:
        pred = str(u.proposition.get("predicate", ""))
        by_pred.setdefault(pred, []).append(u)

    def _unit_doc_id(u: Any) -> str:
        evs = getattr(u, "evidence", ()) or ()
        return str(evs[0].document_id) if evs else ""

    def _unit_scope(u: Any) -> Mapping[str, Any]:
        sc = u.proposition.get("scope")
        return sc if isinstance(sc, Mapping) else {}

    _SUB_ENTITY_QUAL_RE = re.compile(r"\b(class|tranche|series|tier|facility)\s+([a-z0-9-]+)\b", re.IGNORECASE)

    def _norm(val: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (val or "").strip().lower()).strip("_")

    def _entity_qualifier(ent_txt: str) -> tuple[str, str] | None:
        m = _SUB_ENTITY_QUAL_RE.search(ent_txt or "")
        return (m.group(1).lower(), m.group(2).lower()) if m else None

    def _scopes_compatible(s1: Mapping[str, Any], s2: Mapping[str, Any]) -> bool:
        for k in set(s1) & set(s2):
            if _norm(str(s1[k])) != _norm(str(s2[k])):
                return False
        return True

    has_conflicts = False
    has_overrides = False

    for pred in sorted(by_pred):
        pred_units = by_pred[pred]
        if len(pred_units) < 2:
            continue
        try:
            spec = vocabulary.predicate(pred)
        except Exception:
            continue

        # Track whether each document has a single normalized entity for this predicate
        doc_entities: dict[str, set[str]] = {}
        for u in pred_units:
            doc_entities.setdefault(_unit_doc_id(u), set()).add(_norm(str(u.proposition.get("entity", ""))))

        n = len(pred_units)
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = pred_units[i], pred_units[j]
                d1, d2 = _unit_doc_id(u1), _unit_doc_id(u2)
                if d1 and d2 and d1 == d2:
                    # Skip two alternative unit-readings from overlapping character spans in the
                    # same document (e.g. parenthetical alternatives in the same clause).
                    evs1 = getattr(u1, "evidence", ()) or ()
                    evs2 = getattr(u2, "evidence", ()) or ()
                    if evs1 and evs2:
                        s1 = getattr(evs1[0], "start", None)
                        e1_pos = getattr(evs1[0], "end", None)
                        s2 = getattr(evs2[0], "start", None)
                        e2_pos = getattr(evs2[0], "end", None)
                        if (
                            s1 is not None
                            and e1_pos is not None
                            and s2 is not None
                            and e2_pos is not None
                            and max(int(s1), int(s2)) < min(int(e1_pos), int(e2_pos))
                        ):
                            continue
                same_ctx = str(u1.context_key) == str(u2.context_key)
                if not same_ctx:
                    if not d1 or not d2 or d1 == d2:
                        continue
                    s1, s2 = _unit_scope(u1), _unit_scope(u2)
                    if not _scopes_compatible(s1, s2):
                        continue
                    e1, e2 = str(u1.proposition.get("entity", "")), str(u2.proposition.get("entity", ""))
                    q1, q2 = _entity_qualifier(e1), _entity_qualifier(e2)
                    if q1 and q2 and q1[0] == q2[0] and q1[1] != q2[1]:
                        continue
                    same_ent = _norm(e1) == _norm(e2)
                    singleton_per_doc = len(doc_entities.get(d1, set())) == 1 and len(doc_entities.get(d2, set())) == 1
                    if not (same_ent or singleton_per_doc):
                        continue
                ctx_key = str(u1.context_key) if same_ctx else f"{pred}::{_norm(str(u1.proposition.get('entity', '')))}"
                if not _units_incompatible(vocabulary, spec, u1, u2):
                    continue
                r1 = _unit_authority_rank(evidence, u1)
                r2 = _unit_authority_rank(evidence, u2)
                # (ii) Material conflict for every incompatible pair on the same context_key
                has_conflicts = True
                pair_sorted = sorted([u1, u2], key=lambda u: str(u.unit_id))
                a_u, b_u = pair_sorted[0], pair_sorted[1]
                ob_id = f"corpus::conflict::{pred}::{digest([a_u.unit_id, b_u.unit_id])[:12]}"
                if ob_id not in seen_ids:
                    seen_ids.add(ob_id)
                    obligations.append(
                        VerificationObligation(
                            obligation_id=ob_id,
                            family="document_authority",
                            description=(
                                f"Disclose material conflict for '{pred}' between incompatible "
                                "admitted corpus values (present both values and governance/unresolved status)."
                            ),
                            critical=True,
                            origin="corpus",
                            parameters={
                                "corpus_obligation_kind": "material_conflict",
                                "entity": str(a_u.proposition.get("entity", "")),
                                "predicate": pred,
                                "context_key": ctx_key,
                                "unit_ids": [str(a_u.unit_id), str(b_u.unit_id)],
                                "conflicting_values": [
                                    {
                                        "unit_id": str(a_u.unit_id),
                                        "document_id": str(a_u.evidence[0].document_id) if getattr(a_u, "evidence", ()) else "",
                                        "value": a_u.proposition.get("value"),
                                        "unit": a_u.proposition.get("unit"),
                                        "authority_rank": _unit_authority_rank(evidence, a_u),
                                    },
                                    {
                                        "unit_id": str(b_u.unit_id),
                                        "document_id": str(b_u.evidence[0].document_id) if getattr(b_u, "evidence", ()) else "",
                                        "value": b_u.proposition.get("value"),
                                        "unit": b_u.proposition.get("unit"),
                                        "authority_rank": _unit_authority_rank(evidence, b_u),
                                    },
                                ],
                                "evidence_spans": _unit_spans(a_u) + _unit_spans(b_u),
                            },
                        )
                    )
                if r1 != r2:
                    # (iii) Higher authority_rank overrides lower authority_rank
                    has_overrides = True
                    winner, loser = (u1, u2) if r1 > r2 else (u2, u1)
                    w_rank, l_rank = max(r1, r2), min(r1, r2)
                    ob_id = f"corpus::override::{pred}::{digest([winner.unit_id, loser.unit_id])[:12]}"
                    if ob_id in seen_ids:
                        continue
                    seen_ids.add(ob_id)
                    obligations.append(
                        VerificationObligation(
                            obligation_id=ob_id,
                            family="document_authority",
                            description=(
                                f"Cover authority override for '{pred}': higher authority_rank ({w_rank}) "
                                f"supersedes lower authority_rank ({l_rank}) while disclosing the discrepancy."
                            ),
                            critical=True,
                            origin="corpus",
                            parameters={
                                "corpus_obligation_kind": "authority_override",
                                "entity": str(winner.proposition.get("entity", "")),
                                "predicate": pred,
                                "context_key": ctx_key,
                                "unit_ids": [str(winner.unit_id), str(loser.unit_id)],
                                "winner_unit_id": str(winner.unit_id),
                                "loser_unit_id": str(loser.unit_id),
                                "winner_document_id": str(winner.evidence[0].document_id) if getattr(winner, "evidence", ()) else "",
                                "loser_document_id": str(loser.evidence[0].document_id) if getattr(loser, "evidence", ()) else "",
                                "winner_value": winner.proposition.get("value"),
                                "winner_unit": winner.proposition.get("unit"),
                                "winner_authority_rank": w_rank,
                                "loser_value": loser.proposition.get("value"),
                                "loser_unit": loser.proposition.get("unit"),
                                "loser_authority_rank": l_rank,
                                "evidence_spans": _unit_spans(winner) + _unit_spans(loser),
                            },
                        )
                    )

    # Cross-check reader-derived obligations against deterministic v3 section divergences
    reader_obligations = list(obligations)
    gap_divergences: list[dict[str, Any]] = []
    sec_div_records: list[dict[str, Any]] = []
    for div in comparison_divergences or ():
        pri_id = str(getattr(div, "primary_source_id", "") or "")
        sec_id = str(getattr(div, "secondary_source_id", "") or "")
        sec_key = str(getattr(div, "section_key", "") or "")
        sec_label = str(getattr(div, "section_label", "") or "")
        pri_span = getattr(div, "primary_span", None)
        sec_span = getattr(div, "secondary_span", None)
        pri_quants = list(getattr(div, "primary_quantities", ()) or ())
        sec_quants = list(getattr(div, "secondary_quantities", ()) or ())
        div_rec = {
            "section_key": sec_key,
            "section_label": sec_label,
            "primary_source_id": pri_id,
            "secondary_source_id": sec_id,
            "primary_span": list(pri_span) if pri_span is not None else None,
            "secondary_span": list(sec_span) if sec_span is not None else None,
            "primary_quantities": pri_quants,
            "secondary_quantities": sec_quants,
            "reason": str(getattr(div, "reason", "") or ""),
        }
        sec_div_records.append(div_rec)
        if not _divergence_covered_by_reader_obligations(div, reader_obligations):
            gap_divergences.append(div_rec)

        # Register every v3 section divergence as an origin='corpus' obligation (family='section_divergence')
        div_ob_id = f"corpus::section_divergence::{sec_key}::{digest([pri_id, sec_id, sec_key])[:12]}"
        if div_ob_id not in seen_ids:
            seen_ids.add(div_ob_id)
            div_spans: list[dict[str, Any]] = []
            if pri_id and pri_span is not None:
                div_spans.append({
                    "document_id": pri_id,
                    "start": int(pri_span[0]),
                    "end": int(pri_span[1]),
                    "quote": sec_label,
                })
            if sec_id and sec_span is not None:
                div_spans.append({
                    "document_id": sec_id,
                    "start": int(sec_span[0]),
                    "end": int(sec_span[1]),
                    "quote": sec_label,
                })
            overlapping_uids = sorted({
                uid
                for rob in reader_obligations
                if (
                    (pri_id and pri_span is not None and _obligation_overlaps_doc_span(rob, pri_id, pri_span))
                    or (sec_id and sec_span is not None and _obligation_overlaps_doc_span(rob, sec_id, sec_span))
                )
                for uid in ((rob.parameters or {}).get("unit_ids") or ())
            })
            obligations.append(
                VerificationObligation(
                    obligation_id=div_ob_id,
                    family="section_divergence",
                    description=(
                        f"Cover divergent section '{sec_label}' between '{pri_id}' and '{sec_id}' "
                        "in both verified graph leaves and linked response text."
                    ),
                    critical=True,
                    origin="corpus",
                    parameters={
                        "corpus_obligation_kind": "section_divergence",
                        "section_key": sec_key,
                        "section_label": sec_label,
                        "primary_source_id": pri_id,
                        "secondary_source_id": sec_id,
                        "primary_span": list(pri_span) if pri_span is not None else None,
                        "secondary_span": list(sec_span) if sec_span is not None else None,
                        "primary_quantities": pri_quants,
                        "secondary_quantities": sec_quants,
                        "unit_ids": overlapping_uids,
                        "evidence_spans": div_spans,
                    },
                )
            )

    admitted_refs = tuple(
        {
            "unit_id": str(u.unit_id),
            "document_id": str(sp.get("document_id", "")),
            "start": sp.get("start"),
            "end": sp.get("end"),
            "entity": str(u.proposition.get("entity", "")),
            "predicate": str(u.proposition.get("predicate", "")),
            "value": u.proposition.get("value"),
            "unit": u.proposition.get("unit"),
            "condition_status": str(u.proposition.get("condition_status", "unconditional")),
        }
        for u in admitted
        for sp in _unit_spans(u)
    )

    return CorpusReferenceRegister(
        available=True,
        entity=target_entity,
        required_predicates=req_preds,
        obligations=tuple(obligations),
        document_ids=doc_ids,
        unit_count=len(admitted),
        has_material_conflicts=has_conflicts,
        has_authority_overrides=has_overrides,
        reason="Corpus reference register built deterministically from EvidenceLayer.",
        incomplete=bool(relevant_incomplete),
        incomplete_documents=relevant_incomplete,
        document_statuses=doc_statuses,
        gap_divergences=tuple(gap_divergences),
        section_divergences=tuple(sec_div_records),
        admitted_unit_refs=admitted_refs,
    )


_GOVERNANCE_DISCLOSURE_RE = re.compile(
    r"\b(?:governs?|governed\s+by|governing|controls?|controlled\s+by|controlling|"
    r"supersedes?|superseded\s+by|overridden\s+by|prevails?|prevailing|"
    r"higher\s+authority|priority\s+over|takes?\s+precedence|according\s+to\s+[^.;,]{1,60}\s+over)\b",
    re.IGNORECASE,
)
_GOVERNANCE_ACTIVE_RE = re.compile(
    r"\b(?:governs?|governing|controls?|controlling|supersedes?|superseding|prevails?|prevailing|takes?\s+precedence|priority)\b",
    re.IGNORECASE,
)
_GOVERNANCE_PASSIVE_RE = re.compile(
    r"\b(?:governed|controlled|superseded|overridden|prevailed)\s+by\b",
    re.IGNORECASE,
)
_UNRESOLVED_DISCLOSURE_RE = re.compile(
    r"\b(?:unresolved|conflict(?:ing)?|discrepanc(?:y|ies)|contradict(?:s|ory|ion)?|"
    r"inconsistent|cannot\s+be\s+resolved|not\s+resolved|unclear\s+which)\b",
    re.IGNORECASE,
)
_STRICT_UNRESOLVED_RE = re.compile(
    r"\b(?:unresolved|unresolvable|cannot\s+be\s+resolved|not\s+resolved|unclear\s+which|"
    r"neither\s+(?:source|document)\s+governs|no\s+governing\s+(?:source|document))\b",
    re.IGNORECASE,
)
_DOC_EXT_RE = re.compile(r"\.(?:docx|pdf|xlsx|eml|txt|md|json)$", re.IGNORECASE)
_GENERIC_DOC_WORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "document", "source", "text", "notes", "file", "sheet",
})


def _doc_stem_words(doc_id: str, title: str = "") -> list[str]:
    stem = _DOC_EXT_RE.sub("", str(doc_id or ""))
    raw = f"{stem} {title}".strip()
    return [w.lower() for w in re.findall(r"[A-Za-z]+", raw) if w]


def _pair_doc_aliases(
    doc_a: str,
    doc_b: str,
    all_doc_statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Build pair-disjoint surface aliases for two documents purely from structural metadata."""
    statuses = all_doc_statuses or {}
    # Count word frequency across all registered documents so shared deal/prefix words drop automatically
    word_df: dict[str, int] = {}
    for did, meta in statuses.items():
        title = str((meta or {}).get("title", "") or "")
        for w in set(_doc_stem_words(did, title)):
            word_df[w] = word_df.get(w, 0) + 1

    def _aliases_for(did: str, other_did: str) -> tuple[str, ...]:
        meta = statuses.get(did, {}) or {}
        other_meta = statuses.get(other_did, {}) or {}
        title = str(meta.get("title", "") or "")
        other_title = str(other_meta.get("title", "") or "")
        words = _doc_stem_words(did, title)
        other_words = set(_doc_stem_words(other_did, other_title))
        stem_norm = " ".join(words)
        other_stem_norm = " ".join(_doc_stem_words(other_did, other_title))
        cands: list[str] = [did.lower()]
        if stem_norm and stem_norm != other_stem_norm:
            cands.append(stem_norm)
        if title.strip():
            cands.append(title.strip().lower())
        unique_words = [
            w for w in words
            if w not in other_words and w not in _GENERIC_DOC_WORDS and not w.isdigit()
        ]
        # Phrases of consecutive unique words (e.g. "offering memorandum", "draft oc", "master agreement")
        if len(unique_words) >= 2:
            cands.append(" ".join(unique_words))
            acronym = "".join(w[0] for w in unique_words if w.isalpha())
            if len(acronym) == 2 and acronym not in other_words:
                cands.append(acronym)
        for idx, w in enumerate(unique_words):
            if len(w) >= 4 and word_df.get(w, 0) <= 1 and w not in other_words:
                cands.append(w)
            if idx + 1 < len(unique_words):
                bi = f"{w} {unique_words[idx + 1]}"
                cands.append(bi)
        # Keep only candidates that do not match inside the other document's stem words
        filtered: list[str] = []
        seen_c: set[str] = set()
        for c in sorted(set(cands), key=len, reverse=True):
            c = c.strip()
            if not c or c in seen_c:
                continue
            if re.search(rf"\b{re.escape(c)}\b", other_stem_norm):
                continue
            seen_c.add(c)
            filtered.append(c)
        return tuple(filtered)

    return _aliases_for(doc_a, doc_b), _aliases_for(doc_b, doc_a)


def _text_matches_alias(text: str, aliases: Sequence[str]) -> bool:
    if not text or not aliases:
        return False
    low = text.lower()
    return any(re.search(rf"\b{re.escape(a)}\b", low) for a in aliases)


def _value_in_segment_with_unit(val: Any, seg: str) -> bool:
    """Check whether *val* (with its unit suffix if present) appears in *seg* without cross-unit false hits."""
    if val is None or not seg:
        return False
    unit_str = ""
    raw_val = val
    if isinstance(val, Mapping) and "value" in val:
        raw_val = val.get("value")
        unit_str = str(val.get("unit") or "").lower()
    if not _value_in_text(raw_val, seg):
        return False
    if not unit_str:
        return True
    try:
        dec = Decimal(str(raw_val).strip().replace(",", ""))
    except (InvalidOperation, ValueError):
        return True
    # Check at least one numeric match in seg has compatible trailing/leading unit context
    for m in re.finditer(r"([$€£]?\s*[-+]?\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|per\s+annum\b|x\b|bps?\b|business\s+days?|calendar\s+days?|days?|usd\b)?", seg, re.IGNORECASE):
        num_raw = m.group(1).lstrip("$€£ ").replace(",", "")
        try:
            if Decimal(num_raw) != dec:
                continue
        except (InvalidOperation, ValueError):
            continue
        has_dollar = m.group(1).lstrip().startswith(("$", "€", "£"))
        sfx = (m.group(2) or "").strip().lower()
        if "percent" in unit_str:
            if sfx == "x" or has_dollar or sfx == "usd" or "day" in sfx or sfx.startswith("bp"):
                continue
            return True
        if unit_str == "multiple":
            if sfx in ("%", "percent", "per annum") or has_dollar or sfx == "usd" or "day" in sfx:
                continue
            return True
        if "day" in unit_str:
            if sfx in ("%", "percent", "per annum", "x") or has_dollar or sfx == "usd":
                continue
            return True
        if unit_str in ("usd", "usd_per_year"):
            if sfx in ("%", "percent", "x") or "day" in sfx:
                continue
            return True
        return True
    return False


def _check_conflict_disclosure_content(
    *,
    text: str,
    obligation_id: str,
    predicate: str,
    winner_doc: str,
    winner_val: Any,
    winner_rank: int,
    loser_doc: str,
    loser_val: Any,
    loser_rank: int,
    doc_statuses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Verify directional governance and value-to-document attribution in conflict prose."""
    if not text or not winner_doc or not loser_doc or winner_doc == loser_doc:
        return []
    win_aliases, lose_aliases = _pair_doc_aliases(winner_doc, loser_doc, doc_statuses)
    content_issues: list[dict[str, Any]] = []
    unequal_ranks = winner_rank != loser_rank

    if unequal_ranks:
        # 1a. Declaring "unresolved" when authority_rank decides a winner is invalid
        if _STRICT_UNRESOLVED_RE.search(text):
            content_issues.append({
                "code": "GOVERNING_SOURCE_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "expected_governing_document": winner_doc,
                "winner_authority_rank": winner_rank,
                "loser_authority_rank": loser_rank,
                "message": (
                    f"Text declares '{predicate}' conflict unresolved, but '{winner_doc}' "
                    f"(authority_rank={winner_rank}) supersedes '{loser_doc}' (authority_rank={loser_rank})."
                ),
                "remediation": (
                    f"State that '{winner_doc}' (authority_rank={winner_rank}) governs over "
                    f"'{loser_doc}' (authority_rank={loser_rank}); do not claim it is unresolved."
                ),
            })

        # 1b. Check grammatical governance claims in sentences/clauses
        clauses = [c.strip() for c in re.split(r"[.;\n]+", text) if c.strip()]
        gov_mismatch_found = False
        for cl in clauses:
            for mp in _GOVERNANCE_PASSIVE_RE.finditer(cl):
                after = re.split(r"[,;:]|\bover\b", cl[mp.end() :], maxsplit=1)[0]
                if _text_matches_alias(after, lose_aliases) and not _text_matches_alias(after, win_aliases):
                    gov_mismatch_found = True
                    break
            if gov_mismatch_found:
                break
            for ma in _GOVERNANCE_ACTIVE_RE.finditer(cl):
                prefix = cl[: ma.start()]
                if prefix.rstrip().endswith(("is", "are", "was", "were", "be", "been")):
                    continue
                lhs = re.split(r"[:,;]|(?:\b(?:whereas|while|however|although|but)\b)", prefix)[-1]
                # If comma separates independent subject before active governance, keep last comma-segment containing a doc alias
                comma_parts = [p.strip() for p in lhs.split(",") if p.strip()]
                subj_span = lhs
                for part in reversed(comma_parts):
                    if _text_matches_alias(part, win_aliases) or _text_matches_alias(part, lose_aliases):
                        subj_span = part
                        break
                if _text_matches_alias(subj_span, lose_aliases) and not _text_matches_alias(subj_span, win_aliases):
                    gov_mismatch_found = True
                    break
            if gov_mismatch_found:
                break
        if gov_mismatch_found:
            content_issues.append({
                "code": "GOVERNING_SOURCE_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "expected_governing_document": winner_doc,
                "claimed_subordinate_document": loser_doc,
                "message": (
                    f"Response claims subordinate document '{loser_doc}' (rank={loser_rank}) governs "
                    f"instead of '{winner_doc}' (rank={winner_rank}) for '{predicate}'."
                ),
                "remediation": (
                    f"Correct governing attribution: state that '{winner_doc}' governs over '{loser_doc}'."
                ),
            })

    # 2. Check value-to-document attribution across local segments & parenthetical attachments
    if not _value_in_text(winner_val, str(loser_val or "")) or not _value_in_text(loser_val, str(winner_val or "")):
        segments = [
            s.strip()
            for s in re.split(
                r"[;\n()]+|(?<!\d)[.,](?!\d)|\.(?=\s|$)|\b(?:whereas|while|however|although|but|vs\.?|versus)\b|"
                r"\band\b(?=\s*(?:in|per|under|according|the\s+[a-z0-9_-]+\s+(?:states?|provides?|proposes?|specifies?|carries?|shows?|gives?)))",
                text,
                flags=re.IGNORECASE,
            )
            if s.strip()
        ]
        # Also include tight `<doc> (<val>)` pairs where parenthetical carries one value without naming the other doc
        for pm in re.finditer(r"([^.;\n()]{3,80})\(([^()]{1,60})\)", text):
            before, inside = pm.group(1), pm.group(2)
            tail = re.split(r"[,:;]|\b(?:over|vs\.?|versus|than|and)\b", before, flags=re.IGNORECASE)[-1]
            if (
                (_text_matches_alias(tail, win_aliases) or _text_matches_alias(tail, lose_aliases))
                and not (_text_matches_alias(inside, win_aliases) or _text_matches_alias(inside, lose_aliases))
            ):
                segments.append(f"{tail} {inside}")

        val_mismatch_found = False
        for seg in segments:
            seg_w = _text_matches_alias(seg, win_aliases)
            seg_l = _text_matches_alias(seg, lose_aliases)
            has_w_val = _value_in_segment_with_unit(winner_val, seg)
            has_l_val = _value_in_segment_with_unit(loser_val, seg)
            if seg_w and not seg_l and has_l_val and not has_w_val:
                val_mismatch_found = True
                break
            if seg_l and not seg_w and has_w_val and not has_l_val:
                val_mismatch_found = True
                break
        if val_mismatch_found:
            content_issues.append({
                "code": "VALUE_ATTRIBUTION_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "obligation_id": obligation_id,
                "predicate": predicate,
                "winner_document_id": winner_doc,
                "winner_value": winner_val,
                "loser_document_id": loser_doc,
                "loser_value": loser_val,
                "message": (
                    f"Response swaps or misattributes conflicting '{predicate}' values across "
                    f"'{winner_doc}' ({winner_val}) and '{loser_doc}' ({loser_val})."
                ),
                "remediation": (
                    f"Attribute '{winner_val}' strictly to '{winner_doc}' and '{loser_val}' strictly to '{loser_doc}'."
                ),
            })

    return content_issues


def _value_in_text(val: Any, text: str) -> bool:
    if val is None or not text:
        return False
    if isinstance(val, Mapping) and "value" in val:
        val = val["value"]
    raw = str(val).strip()
    if not raw:
        return False
    try:
        dec = Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return raw.lower() in text.lower()
    # Numeric values must match an exact parsed numeric token (never substring inside "13" or "50")
    for m in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", text):
        tok = m.group(0).replace(",", "")
        try:
            if Decimal(tok) == dec:
                return True
        except (InvalidOperation, ValueError):
            continue
    return False


def _claim_matches_obligation(node: Any, ob: VerificationObligation) -> bool:
    if getattr(node, "kind", "") == "opinion" or getattr(node, "assertion_kind", "") == "opinion":
        return False
    params = ob.parameters or {}
    ob_unit_ids = set(str(u) for u in (params.get("unit_ids") or ()))
    node_unit_ids = set(str(u) for u in (getattr(node, "evidence_unit_ids", ()) or ()))
    if ob_unit_ids and (ob_unit_ids & node_unit_ids):
        return True
    ob_ent = str(params.get("entity", ""))
    ob_pred = str(params.get("predicate", ""))
    if ob_pred and label_key(str(getattr(node, "predicate", ""))) != label_key(ob_pred):
        return False
    if ob_ent and getattr(node, "subject", "") and label_key(str(node.subject)) != label_key(ob_ent):
        # Allow entity alias or opposing conflict participant when predicate matches and evidence spans overlap.
        for ref in getattr(node, "evidence", ()) or ():
            r_doc = str(getattr(ref, "document_id", "") or "")
            r_start = getattr(ref, "start", None)
            r_end = getattr(ref, "end", None)
            if r_doc and _obligation_overlaps_doc_span(
                ob, r_doc, (int(r_start), int(r_end)) if r_start is not None and r_end is not None else None
            ):
                return True
        return False
    return bool(ob_pred)


@dataclass(frozen=True)
class CorpusCoverageResult:
    """Result of checking a JustificationGraph + candidate text against CorpusReferenceRegister."""

    total_obligations: int
    covered_ids: tuple[str, ...]
    uncovered_ids: tuple[str, ...]
    rejected_ids: tuple[str, ...]
    all_obligation_ids: tuple[str, ...]
    issues: tuple[dict[str, Any], ...]

    @property
    def covered_count(self) -> int:
        return len(self.covered_ids)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_ids)


def verify_corpus_obligations_coverage(
    register: CorpusReferenceRegister,
    graph: Any,
    candidate_response: str | None,
    *,
    refuted_claim_ids: set[str] | frozenset[str] = frozenset(),
    correspondence_report: Mapping[str, Any] | None = None,
) -> CorpusCoverageResult:
    """Enforce closure rule: every corpus obligation must be a verified graph leaf AND linked in text."""
    if not register.available:
        init_issues: list[dict[str, Any]] = []
        if register.incomplete:
            init_issues.append({
                "code": "REFERENCE_REGISTER_INCOMPLETE",
                "source_layer": "v4",
                "repairable": False,
                "affected_documents": list(register.incomplete_documents),
                "document_statuses": {k: dict(v) for k, v in register.document_statuses.items()},
                "message": (
                    "Corpus reference register is incomplete; relevant document(s) did not complete reading: "
                    f"{list(register.incomplete_documents)}."
                ),
                "remediation": "Ensure all relevant corpus documents reach read_ok before evaluating.",
            })
        else:
            init_issues.append({
                "code": "REFERENCE_REGISTER_MISSING",
                "source_layer": "v4",
                "repairable": False,
                "message": register.reason or "No corpus reference register is available for this document.",
                "remediation": "Configure a reader on EvidenceLayer or provide corpus readings before evaluating.",
            })
        return CorpusCoverageResult(
            total_obligations=0,
            covered_ids=(),
            uncovered_ids=(),
            rejected_ids=(),
            all_obligation_ids=(),
            issues=tuple(init_issues),
        )

    rejected_map: Mapping[str, str] = getattr(graph, "rejected_obligations", {}) or {}
    issues: list[dict[str, Any]] = []
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    rejected_ids: list[str] = []

    if register.incomplete:
        issues.append({
            "code": "REFERENCE_REGISTER_INCOMPLETE",
            "source_layer": "v4",
            "repairable": False,
            "affected_documents": list(register.incomplete_documents),
            "document_statuses": {k: dict(v) for k, v in register.document_statuses.items()},
            "message": (
                "Corpus reference register is incomplete; relevant document(s) not in read_ok: "
                f"{list(register.incomplete_documents)}."
            ),
            "remediation": "Re-run corpus reader on failed/unread documents before acceptance.",
        })

    for gap in register.gap_divergences:
        sec_key = str(gap.get("section_key", "unknown"))
        issues.append({
            "code": "REFERENCE_REGISTER_GAP",
            "source_layer": "v4",
            "repairable": False,
            "obligation_id": f"corpus::gap::{sec_key}",
            "section_key": sec_key,
            "section_label": str(gap.get("section_label", sec_key)),
            "gap_divergence": dict(gap),
            "message": (
                f"Cross-check detected divergent section '{gap.get('section_label', sec_key)}' "
                f"between '{gap.get('primary_source_id')}' and '{gap.get('secondary_source_id')}' "
                "with no associated reader obligation in the reference register."
            ),
            "remediation": "Re-extract corpus obligations for all divergent sections before acceptance.",
        })

    # Collect explicit rejections (even if they name '*' or any corpus obligation)
    for ob_id, reason in sorted(rejected_map.items()):
        if ob_id in register.obligation_ids or ob_id == "*":
            rejected_ids.append(ob_id)
            issues.append({
                "code": "CORPUS_OBLIGATION_REJECTED",
                "source_layer": "v4",
                "repairable": False,
                "obligation_id": ob_id,
                "message": f"Agent explicitly rejected corpus obligation '{ob_id}': {reason}",
                "remediation": "Explicitly rejected obligations require human escalation; cannot ACCEPT.",
            })
        else:
            rejected_ids.append(ob_id)
            issues.append({
                "code": "CORPUS_OBLIGATION_REJECTED",
                "source_layer": "v4",
                "repairable": False,
                "obligation_id": ob_id,
                "message": f"Agent explicitly rejected obligation '{ob_id}': {reason}",
                "remediation": "Explicitly rejected obligations require human escalation; cannot ACCEPT.",
            })

    verified_leaves = [
        node
        for cid, node in sorted(getattr(graph, "claims", {}).items())
        if cid not in refuted_claim_ids
        and getattr(node, "kind", "") != "opinion"
        and getattr(node, "assertion_kind", "") != "opinion"
        and (getattr(node, "evidence", ()) or getattr(node, "evidence_unit_ids", ()))
    ]
    text = candidate_response or ""

    for ob in register.obligations:
        if ob.obligation_id in rejected_map or "*" in rejected_map:
            continue
        params = ob.parameters or {}
        kind = str(params.get("corpus_obligation_kind", "mandatory_predicate"))
        spans = list(params.get("evidence_spans") or ())
        matching_leaves = [n for n in verified_leaves if _claim_matches_obligation(n, ob)]

        in_graph = False
        in_text = True if candidate_response is None else False
        extra_content_issues: list[dict[str, Any]] = []

        if kind in ("mandatory_predicate", "authority_override"):
            if kind == "authority_override":
                winner_id = str(params.get("winner_unit_id", ""))
                winner_val = params.get("winner_value")
                loser_val = params.get("loser_value")
                w_doc = str(params.get("winner_document_id", "") or "")
                l_doc = str(params.get("loser_document_id", "") or "")
                w_rank = int(params.get("winner_authority_rank", 0) or 0)
                l_rank = int(params.get("loser_authority_rank", 0) or 0)
                in_graph = any(
                    (winner_id and winner_id in set(getattr(n, "evidence_unit_ids", ()) or ()))
                    or _value_in_text(winner_val, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                if candidate_response is not None:
                    extra_content_issues = _check_conflict_disclosure_content(
                        text=text,
                        obligation_id=ob.obligation_id,
                        predicate=str(params.get("predicate", "")),
                        winner_doc=w_doc,
                        winner_val=winner_val,
                        winner_rank=w_rank,
                        loser_doc=l_doc,
                        loser_val=loser_val,
                        loser_rank=l_rank,
                        doc_statuses=register.document_statuses,
                    )
                if candidate_response is not None and in_graph:
                    has_winner = _value_in_text(winner_val, text)
                    has_loser = _value_in_text(loser_val, text)
                    has_gov = bool(_GOVERNANCE_DISCLOSURE_RE.search(text))
                    has_unres = bool(_UNRESOLVED_DISCLOSURE_RE.search(text))
                    if w_rank != l_rank:
                        in_text = has_winner and (not has_loser or has_gov) and not extra_content_issues
                    else:
                        in_text = has_winner and (not has_loser or has_gov or has_unres) and not extra_content_issues
            else:
                expected_val = params.get("expected_value")
                in_graph = bool(matching_leaves)
                if candidate_response is not None and in_graph:
                    in_text = any(
                        _value_in_text(getattr(n, "value", None), text)
                        for n in matching_leaves
                    ) or _value_in_text(expected_val, text)

        elif kind == "material_conflict":
            conf_vals = list(params.get("conflicting_values") or ())
            if len(conf_vals) >= 2:
                v1, v2 = conf_vals[0].get("value"), conf_vals[1].get("value")
                u1_id, u2_id = str(conf_vals[0].get("unit_id", "")), str(conf_vals[1].get("unit_id", ""))
                covered_v1 = any(
                    u1_id in set(getattr(n, "evidence_unit_ids", ()) or ())
                    or _value_in_text(v1, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                covered_v2 = any(
                    u2_id in set(getattr(n, "evidence_unit_ids", ()) or ())
                    or _value_in_text(v2, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                in_graph = covered_v1 and covered_v2
                if candidate_response is not None:
                    c0, c1 = conf_vals[0], conf_vals[1]
                    r0, r1 = int(c0.get("authority_rank", 0) or 0), int(c1.get("authority_rank", 0) or 0)
                    win_c, lose_c = (c0, c1) if r0 >= r1 else (c1, c0)
                    extra_content_issues = _check_conflict_disclosure_content(
                        text=text,
                        obligation_id=ob.obligation_id,
                        predicate=str(params.get("predicate", "")),
                        winner_doc=str(win_c.get("document_id", "") or ""),
                        winner_val=win_c.get("value"),
                        winner_rank=int(win_c.get("authority_rank", 0) or 0),
                        loser_doc=str(lose_c.get("document_id", "") or ""),
                        loser_val=lose_c.get("value"),
                        loser_rank=int(lose_c.get("authority_rank", 0) or 0),
                        doc_statuses=register.document_statuses,
                    )
                if candidate_response is not None and in_graph:
                    both_in_text = _value_in_text(v1, text) and _value_in_text(v2, text)
                    if r0 != r1:
                        disclosed = bool(_GOVERNANCE_DISCLOSURE_RE.search(text))
                    else:
                        disclosed = bool(
                            _GOVERNANCE_DISCLOSURE_RE.search(text)
                            or _UNRESOLVED_DISCLOSURE_RE.search(text)
                        )
                    in_text = both_in_text and disclosed and not extra_content_issues
            else:
                in_graph = bool(matching_leaves)
                in_text = candidate_response is None or bool(matching_leaves)

        elif kind == "section_divergence":
            pri_id = str(params.get("primary_source_id", "") or "")
            sec_id = str(params.get("secondary_source_id", "") or "")
            pri_sp = tuple(params["primary_span"]) if params.get("primary_span") else None
            sec_sp = tuple(params["secondary_span"]) if params.get("secondary_span") else None

            def _leaf_overlaps_sec(leaf: Any) -> bool:
                for ev in (getattr(leaf, "evidence", ()) or ()):
                    ev_doc = str(ev.get("document_id", "") if isinstance(ev, Mapping) else getattr(ev, "document_id", ""))
                    ev_s = ev.get("start") if isinstance(ev, Mapping) else getattr(ev, "start", None)
                    ev_e = ev.get("end") if isinstance(ev, Mapping) else getattr(ev, "end", None)
                    ev_rng = (int(ev_s), int(ev_e)) if ev_s is not None and ev_e is not None else None
                    if pri_id and _spans_overlap(ev_doc, ev_rng, pri_id, pri_sp):
                        return True
                    if sec_id and _spans_overlap(ev_doc, ev_rng, sec_id, sec_sp):
                        return True
                return False

            div_leaves = [n for n in verified_leaves if _leaf_overlaps_sec(n)]
            in_graph = bool(div_leaves)
            if candidate_response is not None and in_graph:
                div_leaf_ids = {str(getattr(n, "claim_id", "")) for n in div_leaves}
                linked_by_corr = False
                if correspondence_report:
                    for rel_chk in (correspondence_report.get("relation_checks") or ()):
                        if rel_chk.get("status") == "verified" and str(rel_chk.get("matched_claim_id", "")) in div_leaf_ids:
                            linked_by_corr = True
                            break
                    if not linked_by_corr:
                        for cc in (correspondence_report.get("clause_claims") or ()):
                            if not cc.get("is_grounded"):
                                continue
                            cl_low = str(cc.get("clause_text", "")).lower()
                            if any(
                                str(getattr(n, "subject", "")).strip().lower() in cl_low
                                and (
                                    getattr(n, "value", None) is None
                                    or _value_in_text(getattr(n, "value", None), cl_low)
                                )
                                for n in div_leaves
                                if str(getattr(n, "subject", "")).strip()
                            ):
                                linked_by_corr = True
                                break
                in_text = linked_by_corr or any(
                    _value_in_text(getattr(n, "value", None), text) for n in div_leaves
                    if getattr(n, "value", None) is not None
                )

        if in_graph and in_text:
            covered_ids.append(ob.obligation_id)
        else:
            uncovered_ids.append(ob.obligation_id)
            if extra_content_issues:
                issues.extend(extra_content_issues)
            else:
                span_preview = spans[0] if spans else {}
                if kind == "material_conflict":
                    remed = (
                        f"Cover material conflict for '{params.get('predicate')}' on '{params.get('entity')}': "
                        "include both conflicting values in the graph and text, and state which governs or declare the conflict unresolved."
                    )
                else:
                    remed = (
                        f"Cover corpus obligation '{ob.obligation_id}' ({params.get('predicate')} on '{params.get('entity')}') "
                        f"as a verified graph leaf and linked text statement citing source span {span_preview}."
                    )
                issues.append({
                    "code": "CORPUS_OBLIGATION_UNCOVERED",
                    "source_layer": "v4",
                    "repairable": True,
                    "obligation_id": ob.obligation_id,
                    "corpus_obligation_kind": kind,
                    "evidence_spans": spans,
                    "message": (
                        f"Corpus obligation '{ob.obligation_id}' is not covered in "
                        f"{'graph' if not in_graph else 'response text'}."
                    ),
                    "remediation": remed,
                })

    return CorpusCoverageResult(
        total_obligations=len(register.obligations),
        covered_ids=tuple(covered_ids),
        uncovered_ids=tuple(uncovered_ids),
        rejected_ids=tuple(rejected_ids),
        all_obligation_ids=register.obligation_ids,
        issues=tuple(issues),
    )
