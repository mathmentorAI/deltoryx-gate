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
            return False
    rel = value_relation(spec.value, p1.get("value"), p2.get("value"))
    if rel == "equal":
        return p1.get("polarity") != p2.get("polarity")
    return rel == "incompatible"


def build_corpus_reference_register(
    evidence: EvidenceLayer | None,
    vocabulary: Vocabulary,
    *,
    question: str | Mapping[str, Any] | None = None,
    entity: str | None = None,
    required_predicates: Sequence[str] | None = None,
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

    admitted = sorted(evidence.units, key=lambda u: str(u.unit_id))
    if not admitted:
        return CorpusReferenceRegister(
            available=False,
            entity=entity,
            reason="REFERENCE_REGISTER_MISSING: EvidenceLayer has no admitted units.",
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

    # Group by context_key for (ii) material conflicts and (iii) authority overrides
    by_context: dict[str, list[Any]] = {}
    for u in relevant_units:
        by_context.setdefault(str(u.context_key), []).append(u)

    has_conflicts = False
    has_overrides = False

    for ctx_key in sorted(by_context):
        ctx_units = by_context[ctx_key]
        if len(ctx_units) < 2:
            continue
        pred = str(ctx_units[0].proposition.get("predicate", ""))
        try:
            spec = vocabulary.predicate(pred)
        except Exception:
            continue

        # Pairwise check across distinct assertion_keys
        n = len(ctx_units)
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = ctx_units[i], ctx_units[j]
                if not _units_incompatible(vocabulary, spec, u1, u2):
                    continue
                r1 = _unit_authority_rank(evidence, u1)
                r2 = _unit_authority_rank(evidence, u2)
                if r1 == r2:
                    # (ii) Material conflict at equal authority rank
                    has_conflicts = True
                    pair_sorted = sorted([u1, u2], key=lambda u: str(u.unit_id))
                    a_u, b_u = pair_sorted[0], pair_sorted[1]
                    ob_id = f"corpus::conflict::{pred}::{digest([a_u.unit_id, b_u.unit_id])[:12]}"
                    if ob_id in seen_ids:
                        continue
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
                                        "value": a_u.proposition.get("value"),
                                        "unit": a_u.proposition.get("unit"),
                                        "authority_rank": _unit_authority_rank(evidence, a_u),
                                    },
                                    {
                                        "unit_id": str(b_u.unit_id),
                                        "value": b_u.proposition.get("value"),
                                        "unit": b_u.proposition.get("unit"),
                                        "authority_rank": _unit_authority_rank(evidence, b_u),
                                    },
                                ],
                                "evidence_spans": _unit_spans(a_u) + _unit_spans(b_u),
                            },
                        )
                    )
                else:
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
    )


_GOVERNANCE_DISCLOSURE_RE = re.compile(
    r"\b(?:governs?|governed\s+by|controls?|controlling|supersedes?|superseded\s+by|"
    r"prevails?|prevailing|higher\s+authority|priority\s+over|according\s+to\s+[^.;,]{1,60}\s+over)\b",
    re.IGNORECASE,
)
_UNRESOLVED_DISCLOSURE_RE = re.compile(
    r"\b(?:unresolved|conflict(?:ing)?|discrepanc(?:y|ies)|contradict(?:s|ory|ion)?|"
    r"inconsistent|cannot\s+be\s+resolved|not\s+resolved|unclear\s+which)\b",
    re.IGNORECASE,
)


def _value_in_text(val: Any, text: str) -> bool:
    if val is None or not text:
        return False
    if isinstance(val, Mapping) and "value" in val:
        val = val["value"]
    raw = str(val).strip()
    if not raw:
        return False
    if raw.lower() in text.lower():
        return True
    try:
        dec = Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return False
    norm = format(dec.normalize(), "f")
    if "." in norm:
        norm = norm.rstrip("0").rstrip(".")
    # Match numeric token with optional trailing zeros / commas
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
        return CorpusCoverageResult(
            total_obligations=0,
            covered_ids=(),
            uncovered_ids=(),
            rejected_ids=(),
            all_obligation_ids=(),
            issues=(
                {
                    "code": "REFERENCE_REGISTER_MISSING",
                    "source_layer": "v4",
                    "repairable": False,
                    "message": register.reason or "No corpus reference register is available for this document.",
                    "remediation": "Ingest units into EvidenceLayer or configure a corpus reader before evaluating.",
                },
            ),
        )

    rejected_map: Mapping[str, str] = getattr(graph, "rejected_obligations", {}) or {}
    issues: list[dict[str, Any]] = []
    covered_ids: list[str] = []
    uncovered_ids: list[str] = []
    rejected_ids: list[str] = []

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

        if kind in ("mandatory_predicate", "authority_override"):
            if kind == "authority_override":
                winner_id = str(params.get("winner_unit_id", ""))
                winner_val = params.get("winner_value")
                loser_val = params.get("loser_value")
                in_graph = any(
                    (winner_id and winner_id in set(getattr(n, "evidence_unit_ids", ()) or ()))
                    or _value_in_text(winner_val, str(getattr(n, "value", "")))
                    for n in matching_leaves
                )
                if candidate_response is not None and in_graph:
                    has_winner = _value_in_text(winner_val, text)
                    has_loser = _value_in_text(loser_val, text)
                    has_gov = bool(_GOVERNANCE_DISCLOSURE_RE.search(text) or _UNRESOLVED_DISCLOSURE_RE.search(text))
                    in_text = has_winner and (not has_loser or has_gov)
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
                if candidate_response is not None and in_graph:
                    both_in_text = _value_in_text(v1, text) and _value_in_text(v2, text)
                    disclosed = bool(
                        _GOVERNANCE_DISCLOSURE_RE.search(text)
                        or _UNRESOLVED_DISCLOSURE_RE.search(text)
                    )
                    in_text = both_in_text and disclosed
            else:
                in_graph = bool(matching_leaves)
                in_text = candidate_response is None or bool(matching_leaves)

        if in_graph and in_text:
            covered_ids.append(ob.obligation_id)
        else:
            uncovered_ids.append(ob.obligation_id)
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
