"""Scope, exception & clause-precedence resolution + two-stage authorization gate for v4.

Stage 1 (Epistemic integrity): read errors, unread documents, fidelity failures,
unknown/unrepresentable conditions, pending conditions, or ambiguous proposals
for required predicates MUST block authorization (RETRY or ESCALATE).

Stage 2 (Authorization policy): grounded, unambiguous propositions must also
satisfy AcceptancePolicy (Q_g >= tau, S_g - O_g >= delta, allowed kinds) and
operational numeric bounds before PASS is emitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import math
import os
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import EvidenceRef, SourceDocument, canonical, digest, instant
from .engine import EvidenceLayer, _noisy_or, _q_index
from .epistemics import DEFAULT_PROFILE, EpistemicProfile
from .fidelity import FieldEvidence, FidelityFinding, verify_extraction_fidelity
from .models import AcceptancePolicy, EvidenceUnit
from .vocabulary import UnmappedProposal, Vocabulary, label_key, value_relation


@dataclass(frozen=True)
class ClauseRelation:
    """Explicit intra- or cross-document relation between clauses."""

    kind: str  # 'amends_clause' | 'exception_to' | 'supersedes_clause'
    source_clause_id: str
    target_clause_id: str
    valid_from: str | None = None
    valid_until: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"amends_clause", "exception_to", "supersedes_clause"}:
            raise ValueError(f"Unsupported ClauseRelation kind: {self.kind!r}")


@dataclass(frozen=True)
class AuthorizationIssue:
    """Structured reason explaining why an operation was bounced or escalated."""

    code: str
    stage: str  # 'epistemic_integrity' | 'authorization_policy'
    disposition: str  # 'RETRY' | 'ESCALATE'
    message: str
    predicate: str | None = None
    detail: Any = None


@dataclass(frozen=True)
class AuthorizationVerdict:
    """Operational authorization result with explicit abstention semantics."""

    status: str  # 'PASS' | 'RETRY' | 'ESCALATE'
    allowed: bool
    requires_human: bool
    resolved_values: dict[str, Any]
    issues: tuple[AuthorizationIssue, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    operation_id: str | None = None
    payload_digest: str | None = None
    snapshot_digest: str | None = None
    decision_snapshot: Mapping[str, Any] = field(default_factory=dict)
    signature: str | None = None


def _is_active_at(valid_from: str | None, valid_until: str | None, as_of: str | None) -> bool:
    if as_of is None:
        return True
    at = instant(as_of)
    start = instant(valid_from)
    end = instant(valid_until)
    if start is not None and at < start:
        return False
    if end is not None and at > end:
        return False
    return True


def _scope_matches(unit_scope: Mapping[str, str], query_scope: Mapping[str, str]) -> bool:
    """Return True if *unit_scope* applies to *query_scope* (general scope {} applies to any)."""
    for k, v in unit_scope.items():
        if k == "clause_id":
            continue
        if k not in query_scope or label_key(query_scope[k]) != label_key(v):
            return False
    return True


def _scope_specificity(unit_scope: Mapping[str, str]) -> int:
    return sum(1 for k in unit_scope if k != "clause_id")


def _filter_admissible_units(
    units: Sequence[EvidenceUnit],
    *,
    target_entity: str,
    predicates: set[str] | frozenset[str],
    policy: AcceptancePolicy,
    profile: EpistemicProfile = DEFAULT_PROFILE,
    as_of: str | None = None,
    query_scope: Mapping[str, str] | None = None,
    documents: Mapping[str, SourceDocument] | None = None,
    excluded_unit_ids: set[str] | frozenset[str] = frozenset(),
    vocabulary: Vocabulary | None = None,
    field_evidences: Mapping[str, Sequence[FieldEvidence]] | None = None,
    stored_field_evidences: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    require_field_evidence: Sequence[str] = (),
    check_evidence_refs: bool = True,
    skip_fidelity: bool = False,
) -> list[EvidenceUnit]:
    """Single shared admissibility filter used by both fidelity checks and scope resolution."""
    eligible: list[EvidenceUnit] = []
    scope_map = query_scope or {}
    for u in units:
        if u.unit_id in excluded_unit_ids:
            continue
        p = u.proposition
        if label_key(p["entity"]) != target_entity:
            continue
        if p["predicate"] not in predicates:
            continue
        if p.get("assertion_kind") not in policy.allowed_assertion_kinds:
            continue
        if not _is_active_at(p.get("valid_from"), p.get("valid_until"), as_of):
            continue
        if not _scope_matches(p.get("scope", {}), scope_map):
            continue
        if not profile.admissible(u.basis):
            continue
        if not u.evidence:
            if u.origin != "rule_derived":
                continue
        elif documents:
            if any(
                ref.document_id in documents
                and documents[ref.document_id].attribution not in policy.allowed_source_attributions
                for ref in u.evidence
            ):
                continue
            if check_evidence_refs:
                refs_ok = True
                for ref in u.evidence:
                    doc = documents.get(ref.document_id)
                    if (
                        doc is None
                        or doc.revision != ref.revision
                        or ref.start < 0
                        or ref.end > len(doc.text)
                        or doc.text[ref.start : ref.end] != ref.quote
                    ):
                        refs_ok = False
                        break
                if not refs_ok:
                    continue
                if not skip_fidelity:
                    fe_by_field: dict[str, FieldEvidence] = {}
                    raw_fe = stored_field_evidences.get(u.unit_id) if stored_field_evidences else None
                    if raw_fe:
                        try:
                            for fname, ref_dict in raw_fe.items():
                                fe_by_field[fname] = FieldEvidence(fname, EvidenceRef(**ref_dict))
                        except (TypeError, ValueError):
                            continue
                    if field_evidences and u.unit_id in field_evidences:
                        for caller_fe in field_evidences[u.unit_id]:
                            fe_by_field[caller_fe.field] = caller_fe
                    merged_fe = list(fe_by_field.values()) if fe_by_field else None
                    primary_doc = documents[u.evidence[0].document_id]
                    if verify_extraction_fidelity(
                        primary_doc.text,
                        u,
                        field_evidence=merged_fe,
                        require_field_evidence=require_field_evidence,
                        vocabulary=vocabulary,
                        documents=documents,
                    ):
                        continue
        eligible.append(u)
    return eligible


def resolve_scope_and_precedence(
    units: Sequence[EvidenceUnit],
    *,
    entity: str,
    predicate: str,
    vocabulary: Vocabulary,
    as_of: str | None = None,
    query_scope: Mapping[str, str] | None = None,
    query_conditions: Mapping[str, Any] | None = None,
    clause_relations: Sequence[ClauseRelation] = (),
    source_tiers: Mapping[str, int] | None = None,
    source_weights: Mapping[str, float] | None = None,
    documents: Mapping[str, SourceDocument] | Sequence[SourceDocument] | None = None,
    policy: AcceptancePolicy | None = None,
    profile: EpistemicProfile = DEFAULT_PROFILE,
    field_evidences: Mapping[str, Sequence[FieldEvidence]] | None = None,
    stored_field_evidences: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    require_field_evidence: Sequence[str] = (),
    check_evidence_refs: bool = True,
    skip_fidelity: bool = False,
) -> dict[str, Any]:
    """Resolve governing value for `(entity, predicate)` with scope/condition overlap & clause precedence."""
    q_scope = dict(query_scope or {})
    q_conds = dict(query_conditions or {})
    pol = policy or AcceptancePolicy()
    target_entity = label_key(vocabulary.entity(entity))
    if isinstance(documents, Mapping):
        doc_map = dict(documents)
    elif documents is not None:
        doc_map = {d.document_id: d for d in documents}
    else:
        doc_map = {}

    matching = _filter_admissible_units(
        units,
        target_entity=target_entity,
        predicates={predicate},
        policy=pol,
        profile=profile,
        as_of=as_of,
        query_scope=q_scope,
        documents=doc_map,
        vocabulary=vocabulary,
        field_evidences=field_evidences,
        stored_field_evidences=stored_field_evidences,
        require_field_evidence=require_field_evidence,
        check_evidence_refs=check_evidence_refs,
        skip_fidelity=skip_fidelity,
    )

    if not matching:
        return {"status": "NO_EVIDENCE", "value": None, "winning_units": (), "issues": ()}

    issues: list[AuthorizationIssue] = []

    # 1. Check for unknown / unrepresentable conditions on any matching unit in scope
    for u in matching:
        cstatus = u.proposition.get("condition_status")
        if cstatus == "unknown_condition":
            issues.append(
                AuthorizationIssue(
                    code="UNKNOWN_CONDITION",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    predicate=predicate,
                    message=f"Unit {u.unit_id} has 'unknown_condition' for {entity}.{predicate}; cannot authorize.",
                    detail={"unit_id": u.unit_id},
                )
            )
        elif cstatus == "unrepresentable_condition":
            issues.append(
                AuthorizationIssue(
                    code="UNREPRESENTABLE_CONDITION",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    predicate=predicate,
                    message=f"Unit {u.unit_id} has 'unrepresentable_condition' for {entity}.{predicate}; cannot authorize.",
                    detail={"unit_id": u.unit_id},
                )
            )

    if issues:
        return {"status": "BLOCKED_GAP", "value": None, "winning_units": (), "issues": tuple(issues)}

    # 2. Evaluate conditions against query_conditions using Vocabulary.evaluate_condition
    spec = vocabulary.predicate(predicate)
    applicable: list[tuple[EvidenceUnit, int]] = []
    for u in matching:
        conds = u.proposition.get("conditions") or ()
        if not conds:
            applicable.append((u, 0))
            continue
        results = [vocabulary.evaluate_condition(spec, c, q_conds) for c in conds]
        if "invalid" in results:
            issues.append(
                AuthorizationIssue(
                    code="INVALID_CONDITION_VALUE",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    predicate=predicate,
                    message=f"Condition for {entity}.{predicate} received an invalid or incommensurable runtime value.",
                    detail={"unit_id": u.unit_id, "conditions": conds},
                )
            )
            continue
        if "false" in results:
            continue
        if "pending" in results:
            issues.append(
                AuthorizationIssue(
                    code="PENDING_CONDITION",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    predicate=predicate,
                    message=f"Condition for {entity}.{predicate} cannot be evaluated because operational context is missing.",
                    detail={"unit_id": u.unit_id, "conditions": conds},
                )
            )
            continue
        applicable.append((u, len(conds)))

    if issues:
        status_code = (
            "INVALID_CONDITION"
            if any(i.code == "INVALID_CONDITION_VALUE" for i in issues)
            else "PENDING_CONDITION"
        )
        return {"status": status_code, "value": None, "winning_units": (), "issues": tuple(issues)}

    if not applicable:
        return {"status": "NO_APPLICABLE_RULE", "value": None, "winning_units": (), "issues": ()}

    # 3. Apply explicit ClauseRelation overrides (amends_clause, supersedes_clause, exception_to)
    applicable_clauses = {
        u.proposition.get("scope", {}).get("clause_id")
        for u, _ in applicable
        if u.proposition.get("scope", {}).get("clause_id")
    }
    suppressed_clauses: set[str] = set()
    for rel in clause_relations:
        if not _is_active_at(rel.valid_from, rel.valid_until, as_of):
            continue
        if rel.source_clause_id in applicable_clauses and rel.target_clause_id in applicable_clauses:
            suppressed_clauses.add(rel.target_clause_id)

    surviving = [
        (u, cond_spec)
        for u, cond_spec in applicable
        if u.proposition.get("scope", {}).get("clause_id") not in suppressed_clauses
    ]
    if not surviving:
        return {"status": "NO_APPLICABLE_RULE", "value": None, "winning_units": (), "issues": ()}

    # 4. Precedence by tier -> condition specificity -> scope specificity
    def _doc_tier(doc_id: str) -> int:
        if source_tiers and doc_id in source_tiers:
            return source_tiers[doc_id]
        if doc_id in doc_map:
            return doc_map[doc_id].authority_rank
        return 0

    def _rank_key(item: tuple[EvidenceUnit, int]) -> tuple[int, int, int]:
        u, cond_spec = item
        tier = min(_doc_tier(ref.document_id) for ref in u.evidence) if u.evidence else 0
        scope_spec = _scope_specificity(u.proposition.get("scope", {}))
        return (tier, cond_spec, scope_spec)

    best_rank = max(_rank_key(item) for item in surviving)
    top_units = [u for u, cspec in surviving if _rank_key((u, cspec)) == best_rank]

    # 5. Group by assertion_key and compute dependence-aware support/opposition (Q_g, S_g - O_g)
    spec = vocabulary.predicate(predicate)
    groups: dict[str, list[EvidenceUnit]] = {}
    for u in top_units:
        groups.setdefault(u.assertion_key, []).append(u)

    def _unit_weight(u: EvidenceUnit) -> float | None:
        # Unassessed structural reliability (None) MUST remain unassessed, never 1.0
        if u.structural_reliability is None:
            return None
        if u.evidence:
            doc_w = min(
                source_weights[ref.document_id]
                if (source_weights and ref.document_id in source_weights)
                else (doc_map[ref.document_id].weight if ref.document_id in doc_map else 1.0)
                for ref in u.evidence
            )
        else:
            doc_w = 1.0
        basis_w = profile.weight(u.basis)
        return max(0.0, min(1.0, min(doc_w, float(u.structural_reliability), basis_w)))

    def _unit_dependence(u: EvidenceUnit) -> frozenset[str]:
        if not u.evidence:
            return frozenset({f"unit:{u.unit_id}"})
        return frozenset(
            (doc_map[r.document_id].dependence_group
             if r.document_id in doc_map and doc_map[r.document_id].dependence_group
             else f"document:{r.document_id}")
            for r in u.evidence
        )

    def _group_support(g_units: Sequence[EvidenceUnit]) -> float | None:
        assessed = sorted(
            ((u, w) for u in g_units if (w := _unit_weight(u)) is not None),
            key=lambda pair: (-pair[1], pair[0].unit_id),
        )
        if not assessed:
            return None
        used_deps: set[str] = set()
        selected_weights: list[float] = []
        for u, w in assessed:
            deps = _unit_dependence(u)
            if not (deps & used_deps):
                used_deps |= deps
                selected_weights.append(w)
        return _noisy_or(selected_weights)

    def _opposition_over(competing_keys: Sequence[str]) -> float:
        by_dep: dict[tuple[str, ...], float] = {}
        for k in competing_keys:
            for u in groups[k]:
                w = _unit_weight(u)
                if w is None:
                    continue
                dkey = tuple(sorted(_unit_dependence(u)))
                by_dep[dkey] = max(by_dep.get(dkey, 0.0), w)
        merged: dict[tuple[str, ...], float] = {}
        for dkey, val in by_dep.items():
            clash = [mk for mk in merged if set(mk) & set(dkey)]
            for mk in clash:
                val = max(val, merged.pop(mk))
                dkey = tuple(sorted(set(dkey) | set(mk)))
            merged[dkey] = val
        return _noisy_or(list(merged.values()))

    group_scores: dict[str, float | None] = {
        akey: _group_support(g_units)
        for akey, g_units in groups.items()
    }
    assessed_scores: dict[str, float] = {k: v for k, v in group_scores.items() if v is not None}

    if not assessed_scores:
        issues.append(
            AuthorizationIssue(
                code="UNASSESSED_READER_RELIABILITY",
                stage="epistemic_integrity",
                disposition="ESCALATE",
                predicate=predicate,
                message=f"Units for {entity}.{predicate} have unassessed structural_reliability=None and cannot authorize operations.",
                detail={"unit_ids": [u.unit_id for u in top_units]},
            )
        )
        return {"status": "UNASSESSED", "value": None, "winning_units": tuple(top_units), "issues": tuple(issues)}

    winner_key = max(assessed_scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    winner_units = tuple(groups[winner_key])
    s_g = assessed_scores[winner_key]

    incompatible_keys: list[str] = []
    unassessed_conflict = False
    p_win = groups[winner_key][0].proposition
    for k in groups:
        if k == winner_key:
            continue
        p_other = groups[k][0].proposition
        if p_win.get("polarity") != p_other.get("polarity"):
            rel_kind = "incompatible"
        elif (
            p_win.get("unit") is not None
            and p_other.get("unit") is not None
            and vocabulary.units.dimension_of(p_win["unit"]) != vocabulary.units.dimension_of(p_other["unit"])
        ):
            rel_kind = "unknown"
        else:
            rel_kind = value_relation(spec.value, p_win["value"], p_other["value"])
        if rel_kind != "compatible":
            if group_scores[k] is None or rel_kind == "unknown":
                unassessed_conflict = True
            else:
                incompatible_keys.append(k)

    if unassessed_conflict:
        issues.append(
            AuthorizationIssue(
                code="UNRESOLVED_COMPARISON",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=predicate,
                message=f"Competing proposal for {entity}.{predicate} cannot be resolved deterministically (unassessed or incommensurable unit).",
                detail={"group_scores": group_scores},
            )
        )
        return {"status": "CONFLICT", "value": None, "winning_units": tuple(top_units), "issues": tuple(issues)}

    o_g = _opposition_over(incompatible_keys)
    q_g = _q_index(s_g, o_g)

    if o_g > 0 and (q_g < pol.tau or (s_g - o_g) < pol.delta):
        issues.append(
            AuthorizationIssue(
                code="UNRESOLVED_CONFLICT",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=predicate,
                message=(
                    f"Overlapping clauses for {entity}.{predicate} conflict at equal precedence "
                    f"(Q={q_g:.3f}, margin={s_g - o_g:.3f})."
                ),
                detail={"group_scores": group_scores, "opposition": o_g},
            )
        )
        return {"status": "CONFLICT", "value": None, "winning_units": tuple(top_units), "issues": tuple(issues)}

    if q_g < pol.tau or (s_g - o_g) < pol.delta:
        issues.append(
            AuthorizationIssue(
                code="INSUFFICIENT_SUPPORT",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=predicate,
                message=f"Support for {entity}.{predicate} below AcceptancePolicy (Q={q_g:.3f}, margin={s_g - o_g:.3f}).",
            )
        )
        return {"status": "LOW_SUPPORT", "value": None, "winning_units": winner_units, "issues": tuple(issues)}

    return {
        "status": "RESOLVED",
        "value": winner_units[0].proposition["value"],
        "unit": winner_units[0].proposition.get("unit"),
        "polarity": winner_units[0].proposition.get("polarity"),
        "winning_units": winner_units,
        "q_g": q_g,
        "margin": round(s_g - o_g, 6),
        "issues": (),
    }


def _finite_decimal(raw: Any, label: str) -> Decimal:
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a finite number, got bool")
    if isinstance(raw, (int, float)) and not math.isfinite(float(raw)):
        raise ValueError(f"{label} must be finite, got {raw!r}")
    try:
        dec = Decimal(str(raw))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number, got {raw!r}") from exc
    if not dec.is_finite():
        raise ValueError(f"{label} must be finite, got {raw!r}")
    return dec


def _resolve_bound_limits(
    numeric_bounds: Mapping[str, Any] | None,
    pred: str,
    vocabulary: Vocabulary,
) -> tuple[Decimal | None, Decimal | None, str | None, str | None]:
    """Resolve finite Decimal (low, high, target_unit, error_msg) bounds via Vocabulary."""
    if not numeric_bounds:
        return None, None, None, None
    bound_entry = None
    explicit_unit: str | None = None
    if pred in numeric_bounds:
        bound_entry = numeric_bounds[pred]
    else:
        for k, entry in numeric_bounds.items():
            if ":" in k:
                b_pred, b_unit = k.split(":", 1)
                if b_pred == pred:
                    bound_entry = entry
                    explicit_unit = b_unit
                    break
    if bound_entry is None:
        return None, None, None, None
    try:
        lo, hi, target_unit = vocabulary.normalize_bound_limits(
            pred, bound_entry, explicit_unit=explicit_unit
        )
    except UnmappedProposal as exc:
        if exc.reason in {
            "unit_required",
            "unknown_unit",
            "unit_dimension_mismatch",
            "unit_not_applicable",
            "numeric_bounds_unsupported_type",
        }:
            return None, None, None, str(exc)
        raise
    return lo, hi, target_unit, None


def _canonical_bound_entry(entry: Any) -> Any:
    if isinstance(entry, Mapping):
        return {str(k): entry[k] for k in sorted(entry)}
    if isinstance(entry, (tuple, list)):
        return list(entry)
    return entry


def authorize_operation(
    layer: EvidenceLayer,
    *,
    entity: str,
    required_predicates: Sequence[str],
    as_of: str | None = None,
    query_scope: Mapping[str, str] | None = None,
    query_conditions: Mapping[str, Any] | None = None,
    clause_relations: Sequence[ClauseRelation] = (),
    source_tiers: Mapping[str, int] | None = None,
    source_weights: Mapping[str, float] | None = None,
    ingest_summary: Mapping[str, Any] | None = None,
    field_evidences: Mapping[str, Sequence[FieldEvidence]] | None = None,
    require_field_evidence: Sequence[str] = (),
    numeric_bounds: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
    payload: Any | None = None,
    payload_digest: str | None = None,
    signing_key: str | bytes | None = None,
) -> AuthorizationVerdict:
    """Run the two-stage operational authorization gate over an atomic v4 EvidenceLayer snapshot."""
    epistemic_issues: list[AuthorizationIssue] = []
    policy_issues: list[AuthorizationIssue] = []

    effective_payload_digest: str | None = None
    if payload is not None:
        computed_payload_digest = digest(payload)
        effective_payload_digest = computed_payload_digest
        if payload_digest is not None and (
            not isinstance(payload_digest, str) or payload_digest != computed_payload_digest
        ):
            epistemic_issues.append(
                AuthorizationIssue(
                    code="PAYLOAD_DIGEST_MISMATCH",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    message=(
                        f"Declared payload_digest {payload_digest!r} does not match "
                        f"computed payload digest {computed_payload_digest!r}."
                    ),
                    detail={"declared": payload_digest, "computed": computed_payload_digest},
                )
            )
    elif payload_digest is not None:
        if isinstance(payload_digest, str) and re.fullmatch(r"[0-9a-f]{64}", payload_digest):
            effective_payload_digest = payload_digest
        else:
            epistemic_issues.append(
                AuthorizationIssue(
                    code="INVALID_PAYLOAD_DIGEST",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    message=f"Declared payload_digest {payload_digest!r} is not a valid 64-char SHA-256 hex digest.",
                    detail={"declared": payload_digest},
                )
            )

    # Atomic single-transaction state read
    state = layer.snapshot()

    # ---- Stage 1: Epistemic Integrity ----
    combined_read_errors = list(state["read_errors"])
    if ingest_summary and ingest_summary.get("read_errors"):
        combined_read_errors.extend(ingest_summary["read_errors"])
    if combined_read_errors:
        epistemic_issues.append(
            AuthorizationIssue(
                code="READ_ERRORS_PRESENT",
                stage="epistemic_integrity",
                disposition="RETRY",
                message="Document reader reported read/contract errors; cannot safely authorize.",
                detail=combined_read_errors,
            )
        )

    combined_unread = sorted(
        set(state["unread_documents"])
        | set(ingest_summary.get("unread_documents", ()) if ingest_summary else ())
    )
    if combined_unread:
        epistemic_issues.append(
            AuthorizationIssue(
                code="UNREAD_DOCUMENTS",
                stage="epistemic_integrity",
                disposition="RETRY",
                message="One or more documents were not read.",
                detail=combined_unread,
            )
        )

    # Check rejections on required predicates (unmapped / ambiguous translations must block authorization)
    target_entity = label_key(layer.vocabulary.entity(entity))
    for rej in state["rejections"]:
        hint_pred = rej.get("hint", {}).get("predicate")
        if hint_pred in required_predicates:
            epistemic_issues.append(
                AuthorizationIssue(
                    code="AMBIGUOUS_OR_UNMAPPED_PROPOSAL",
                    stage="epistemic_integrity",
                    disposition="RETRY",
                    predicate=hint_pred,
                    message=f"Rejected proposal on required predicate '{hint_pred}': {rej.get('error', {}).get('reason')}",
                    detail=rej.get("error"),
                )
            )

    # Single shared admissible unit candidate pool for both fidelity verification and resolution
    all_units: list[EvidenceUnit] = state["units"]
    stored_fe_map: dict[str, dict[str, dict[str, Any]]] = state["field_evidences"]
    layer_docs: dict[str, SourceDocument] = state["documents"]
    doc_revs: dict[tuple[str, str], SourceDocument] = state["document_revisions"]

    candidate_units = _filter_admissible_units(
        all_units,
        target_entity=target_entity,
        predicates=set(required_predicates),
        policy=layer.policy,
        profile=layer.profile,
        as_of=as_of,
        query_scope=query_scope,
        documents=layer_docs,
        vocabulary=layer.vocabulary,
        check_evidence_refs=False,
        skip_fidelity=True,
    )
    ineligible_unit_ids: set[str] = set()

    for u in candidate_units:
        if not u.evidence:
            if u.origin != "rule_derived":
                ineligible_unit_ids.add(u.unit_id)
                epistemic_issues.append(
                    AuthorizationIssue(
                        code="MISSING_EVIDENCE_DOCUMENT",
                        stage="epistemic_integrity",
                        disposition="RETRY",
                        predicate=u.proposition["predicate"],
                        message=f"Unit {u.unit_id} has no evidence references.",
                        detail={"unit_id": u.unit_id},
                    )
                )
            continue

        cited_doc: SourceDocument | None = None
        cited_docs_map: dict[str, SourceDocument] = {}
        for ref in u.evidence:
            cache_key = (ref.document_id, ref.revision)
            if cache_key not in doc_revs:
                ineligible_unit_ids.add(u.unit_id)
                epistemic_issues.append(
                    AuthorizationIssue(
                        code="MISSING_EVIDENCE_DOCUMENT",
                        stage="epistemic_integrity",
                        disposition="RETRY",
                        predicate=u.proposition["predicate"],
                        message=f"Evidence document '{ref.document_id}' revision '{ref.revision}' unavailable.",
                        detail={"unit_id": u.unit_id, "document_id": ref.document_id, "revision": ref.revision},
                    )
                )
                continue
            cited_docs_map[ref.document_id] = doc_revs[cache_key]
            if cited_doc is None:
                cited_doc = doc_revs[cache_key]
            current_doc = layer_docs.get(ref.document_id)
            if current_doc is None or current_doc.revision != ref.revision:
                ineligible_unit_ids.add(u.unit_id)
                epistemic_issues.append(
                    AuthorizationIssue(
                        code="STALE_EVIDENCE_REVISION",
                        stage="epistemic_integrity",
                        disposition="RETRY",
                        predicate=u.proposition["predicate"],
                        message=(
                            f"Unit {u.unit_id} cites stale revision '{ref.revision}' of '{ref.document_id}'."
                        ),
                        detail={
                            "unit_id": u.unit_id,
                            "document_id": ref.document_id,
                            "cited_revision": ref.revision,
                            "current_revision": current_doc.revision if current_doc else None,
                        },
                    )
                )

        if cited_doc is None or u.unit_id in ineligible_unit_ids:
            continue

        # Merge stored FieldEvidence from layer with caller-provided overrides
        fe_by_field: dict[str, FieldEvidence] = {}
        raw_stored_fe = stored_fe_map.get(u.unit_id)
        if raw_stored_fe:
            for fname, ref_dict in raw_stored_fe.items():
                try:
                    fe_by_field[fname] = FieldEvidence(fname, EvidenceRef(**ref_dict))
                except (TypeError, ValueError) as exc:
                    ineligible_unit_ids.add(u.unit_id)
                    epistemic_issues.append(
                        AuthorizationIssue(
                            code="INVALID_FIELD_EVIDENCE",
                            stage="epistemic_integrity",
                            disposition="RETRY",
                            predicate=u.proposition["predicate"],
                            message=f"Invalid stored FieldEvidence for '{fname}': {exc}",
                            detail={"unit_id": u.unit_id, "field": fname},
                        )
                    )
        if field_evidences and u.unit_id in field_evidences:
            for caller_fe in field_evidences[u.unit_id]:
                fe_by_field[caller_fe.field] = caller_fe
        merged_fe = list(fe_by_field.values()) if fe_by_field else None

        f_findings: list[FidelityFinding] = verify_extraction_fidelity(
            cited_doc.text,
            u,
            field_evidence=merged_fe,
            require_field_evidence=require_field_evidence,
            vocabulary=layer.vocabulary,
            documents=cited_docs_map,
        )
        if f_findings:
            ineligible_unit_ids.add(u.unit_id)
            for ff in f_findings:
                epistemic_issues.append(
                    AuthorizationIssue(
                        code=ff.code,
                        stage="epistemic_integrity",
                        disposition="RETRY",
                        predicate=u.proposition["predicate"],
                        message=ff.message,
                        detail={"unit_id": u.unit_id, "field": ff.field, "detail": ff.detail},
                    )
                )

    verified_eligible_units = [u for u in candidate_units if u.unit_id not in ineligible_unit_ids]

    # ---- Stage 2: Scope/Precedence Resolution & Policy Gate ----
    resolved_values: dict[str, Any] = {}
    winning_unit_ids_by_pred: dict[str, list[str]] = {}
    for pred in required_predicates:
        res = resolve_scope_and_precedence(
            verified_eligible_units,
            entity=entity,
            predicate=pred,
            vocabulary=layer.vocabulary,
            as_of=as_of,
            query_scope=query_scope,
            query_conditions=query_conditions,
            clause_relations=clause_relations,
            source_tiers=source_tiers,
            source_weights=source_weights,
            documents=layer_docs,
            policy=layer.policy,
            profile=layer.profile,
            check_evidence_refs=False,
            skip_fidelity=True,
        )
        winning_unit_ids_by_pred[pred] = [u.unit_id for u in res.get("winning_units", ())]
        for iss in res.get("issues", ()):
            if iss.stage == "epistemic_integrity":
                epistemic_issues.append(iss)
            else:
                policy_issues.append(iss)

        if res["status"] == "NO_EVIDENCE" or res["status"] == "NO_APPLICABLE_RULE":
            policy_issues.append(
                AuthorizationIssue(
                    code="MISSING_REQUIRED_PREDICATE",
                    stage="authorization_policy",
                    disposition="RETRY",
                    predicate=pred,
                    message=f"No applicable grounded rule for required predicate '{pred}'.",
                )
            )
            continue

        if res["status"] != "RESOLVED":
            continue

        val = res["value"]
        resolved_values[pred] = val

        # Check operational numeric bounds via Vocabulary (unit-normalized Decimal comparison)
        low, high, target_unit, bound_error = _resolve_bound_limits(numeric_bounds, pred, layer.vocabulary)
        if bound_error is not None:
            policy_issues.append(
                AuthorizationIssue(
                    code="POLICY_BOUND_UNIT_MISMATCH",
                    stage="authorization_policy",
                    disposition="ESCALATE",
                    predicate=pred,
                    message=bound_error,
                )
            )
            continue
        if low is not None or high is not None:
            if isinstance(val, dict) and "unit" in val and target_unit and val["unit"] != target_unit:
                policy_issues.append(
                    AuthorizationIssue(
                        code="POLICY_BOUND_UNIT_MISMATCH",
                        stage="authorization_policy",
                        disposition="ESCALATE",
                        predicate=pred,
                        message=f"Unit '{val['unit']}' for {pred} is not commensurable with bound unit '{target_unit}'.",
                    )
                )
                continue
            raw_num = val["value"] if isinstance(val, dict) and "value" in val else val
            try:
                num_val = _finite_decimal(raw_num, f"resolved_values[{pred}]")
            except ValueError:
                num_val = None
            if num_val is not None:
                if low is not None and num_val < low:
                    policy_issues.append(
                        AuthorizationIssue(
                            code="POLICY_BOUND_BELOW_MIN",
                            stage="authorization_policy",
                            disposition="ESCALATE",
                            predicate=pred,
                            message=f"{pred}={num_val} is below operational floor {low}.",
                        )
                    )
                if high is not None and num_val > high:
                    policy_issues.append(
                        AuthorizationIssue(
                            code="POLICY_BOUND_ABOVE_MAX",
                            stage="authorization_policy",
                            disposition="ESCALATE",
                            predicate=pred,
                            message=f"{pred}={num_val} exceeds operational cap {high}.",
                        )
                    )

    all_issues = tuple(epistemic_issues + policy_issues)
    if any(i.disposition == "RETRY" for i in all_issues):
        status = "RETRY"
    elif any(i.disposition == "ESCALATE" for i in all_issues):
        status = "ESCALATE"
    else:
        status = "PASS"

    final_resolved_values = dict(resolved_values) if status == "PASS" else {}

    decision_snapshot: dict[str, Any] = {
        "configuration": state["configuration"],
        "entity": target_entity,
        "required_predicates": list(required_predicates),
        "as_of": as_of,
        "query_scope": {str(k): query_scope[k] for k in sorted(query_scope)} if query_scope else {},
        "query_conditions": (
            {str(k): query_conditions[k] for k in sorted(query_conditions)} if query_conditions else {}
        ),
        "clause_relations": [asdict(r) for r in clause_relations],
        "require_field_evidence": list(require_field_evidence),
        "numeric_bounds": (
            {str(k): _canonical_bound_entry(numeric_bounds[k]) for k in sorted(numeric_bounds)}
            if numeric_bounds
            else {}
        ),
        "source_tiers_override": {str(k): int(source_tiers[k]) for k in sorted(source_tiers)} if source_tiers else {},
        "source_weights_override": (
            {str(k): float(source_weights[k]) for k in sorted(source_weights)} if source_weights else {}
        ),
        "documents": [
            {
                "document_id": d.document_id,
                "revision": d.revision,
                "authority_rank": d.authority_rank,
                "weight": d.weight,
                "dependence_group": d.dependence_group,
                "attribution": d.attribution,
            }
            for _, d in sorted(layer_docs.items())
        ],
        "eligible_unit_ids": [u.unit_id for u in verified_eligible_units],
        "winning_unit_ids": {pred: winning_unit_ids_by_pred[pred] for pred in sorted(winning_unit_ids_by_pred)},
        "read_state": {
            "unread_documents": combined_unread,
            "read_errors": combined_read_errors,
        },
        "resolved_values": final_resolved_values,
        "status": status,
        "issues": [asdict(i) for i in all_issues],
        "operation_id": operation_id,
        "payload_digest": effective_payload_digest,
    }
    snapshot_digest = digest(decision_snapshot)

    active_key = signing_key if signing_key is not None else os.environ.get("CLAIMLAYER_SIGNING_KEY")
    signature: str | None = None
    if active_key is not None and active_key != "" and active_key != b"":
        key_bytes = active_key.encode("utf-8") if isinstance(active_key, str) else active_key
        signature = hmac.new(key_bytes, snapshot_digest.encode("utf-8"), hashlib.sha256).hexdigest()

    auth_record = {
        "snapshot": decision_snapshot,
        "snapshot_digest": snapshot_digest,
        "signature": signature,
    }
    layer.store.put(
        "authorization",
        snapshot_digest,
        auth_record,
        head=operation_id or snapshot_digest,
    )

    return AuthorizationVerdict(
        status=status,
        allowed=(status == "PASS"),
        requires_human=(status == "ESCALATE"),
        resolved_values=final_resolved_values,
        issues=all_issues,
        metrics={
            "epistemic_issues": len(epistemic_issues),
            "policy_issues": len(policy_issues),
            "predicates_resolved": len(final_resolved_values),
            "predicates_required": len(required_predicates),
        },
        operation_id=operation_id,
        payload_digest=effective_payload_digest,
        snapshot_digest=snapshot_digest,
        decision_snapshot=decision_snapshot,
        signature=signature,
    )


def verify_authorization_verdict(
    verdict: AuthorizationVerdict,
    *,
    layer: EvidenceLayer | None = None,
    operation_id: str | None = None,
    payload: Any | None = None,
    payload_digest: str | None = None,
    signing_key: str | bytes | None = None,
    allow_unkeyed: bool = False,
) -> bool:
    """Verify authenticity, snapshot integrity, and operation/payload binding for an executor."""
    if not isinstance(verdict, AuthorizationVerdict):
        return False
    if verdict.status != "PASS" or verdict.allowed is not True or verdict.requires_human is not False:
        return False
    if not verdict.decision_snapshot or not verdict.snapshot_digest:
        return False

    snap_dict = dict(verdict.decision_snapshot)
    if snap_dict.get("status") != verdict.status:
        return False
    if snap_dict.get("operation_id") != verdict.operation_id:
        return False
    if snap_dict.get("payload_digest") != verdict.payload_digest:
        return False
    if canonical(snap_dict.get("resolved_values")) != canonical(verdict.resolved_values):
        return False
    try:
        if digest(snap_dict) != verdict.snapshot_digest:
            return False
    except (TypeError, ValueError):
        return False

    if operation_id is not None and verdict.operation_id != operation_id:
        return False

    if payload is not None:
        try:
            computed = digest(payload)
        except (TypeError, ValueError):
            return False
        if verdict.payload_digest != computed:
            return False
        if payload_digest is not None and payload_digest != computed:
            return False
    elif payload_digest is not None:
        if verdict.payload_digest != payload_digest:
            return False

    if layer is not None:
        try:
            current_state = layer.snapshot()
        except (ValueError, TypeError, KeyError):
            return False
        if snap_dict.get("configuration") != current_state["configuration"]:
            return False
        expected_docs = [
            {
                "document_id": d.document_id,
                "revision": d.revision,
                "authority_rank": d.authority_rank,
                "weight": d.weight,
                "dependence_group": d.dependence_group,
                "attribution": d.attribution,
            }
            for _, d in sorted(current_state["documents"].items())
        ]
        if snap_dict.get("documents") != expected_docs:
            return False
        current_unit_ids = {u.unit_id for u in current_state["units"]}
        if not set(snap_dict.get("eligible_unit_ids", ())) <= current_unit_ids:
            return False
        for unit_ids in snap_dict.get("winning_unit_ids", {}).values():
            if not set(unit_ids) <= current_unit_ids:
                return False
        if snap_dict.get("read_state") != {
            "unread_documents": list(current_state["unread_documents"]),
            "read_errors": list(current_state["read_errors"]),
        }:
            return False
        try:
            stored = layer.store.get("authorization", verdict.snapshot_digest)
        except (ValueError, TypeError, KeyError):
            return False
        if not isinstance(stored, Mapping):
            return False
        if stored.get("snapshot_digest") != verdict.snapshot_digest:
            return False
        if canonical(stored.get("snapshot")) != canonical(snap_dict):
            return False
        if stored.get("signature") != verdict.signature:
            return False

    active_key = signing_key if signing_key is not None else os.environ.get("CLAIMLAYER_SIGNING_KEY")
    if active_key is not None and active_key != "" and active_key != b"":
        if not isinstance(verdict.signature, str):
            return False
        key_bytes = active_key.encode("utf-8") if isinstance(active_key, str) else active_key
        expected_sig = hmac.new(key_bytes, verdict.snapshot_digest.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(verdict.signature, expected_sig)

    if not allow_unkeyed:
        return False
    return verdict.signature is None
