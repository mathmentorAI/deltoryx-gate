"""Corpus-derived verification obligations and reference register (v4).

Computes deterministic verification obligations directly from admitted
``EvidenceLayer`` units and the domain ``Vocabulary`` before evaluating any
agent-supplied graph, assertions, or metadata flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..semantic.models import canonical, digest
from .engine import EvidenceLayer
from .justification import VerificationObligation
from .vocabulary import Vocabulary, value_relation


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
                family="grounding",
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
