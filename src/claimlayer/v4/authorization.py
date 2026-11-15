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

from ..semantic.models import SourceDocument, canonical, digest, instant
from .domain_profile import DomainProfile, PolicyConfiguration
from .engine import (
    EvidenceLayer,
    UnitEvaluation,
    _noisy_or,
    _q_index,
    evaluate_unit_admissibility,
)
from .epistemics import DEFAULT_PROFILE, EpistemicProfile
from .evaluation import ResponseAssessment
from .fidelity import FieldEvidence
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


def _scope_specificity(unit_scope: Mapping[str, str]) -> int:
    return sum(1 for k in unit_scope if k != "clause_id")


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
    document_revisions: Mapping[tuple[str, str], SourceDocument] | None = None,
    units_by_id: Mapping[str, EvidenceUnit] | None = None,
    rules: Mapping[str, tuple[str, Any]] | None = None,
    evaluations: Sequence[UnitEvaluation] | None = None,
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

    if evaluations is None:
        if isinstance(documents, Mapping):
            doc_map = dict(documents) if documents else None
        elif documents is not None:
            doc_map = {d.document_id: d for d in documents} or None
        else:
            doc_map = None
        u_map = dict(units_by_id) if units_by_id is not None else {u.unit_id: u for u in units}
        evals_all = [
            evaluate_unit_admissibility(
                u,
                vocabulary=vocabulary,
                policy=pol,
                profile=profile,
                units_by_id=u_map,
                rules=rules,
                documents=doc_map,
                document_revisions=document_revisions,
                as_of=as_of,
                query_scope=q_scope,
                query_conditions=q_conds,
                condition_mode="discharge",
                field_evidences=field_evidences,
                stored_field_evidences=stored_field_evidences,
                require_field_evidence=require_field_evidence,
                source_tiers=source_tiers,
                source_weights=source_weights,
                check_evidence_refs=check_evidence_refs,
                skip_fidelity=skip_fidelity,
            )
            for u in units
            if label_key(u.proposition["entity"]) == target_entity
            and u.proposition["predicate"] == predicate
        ]
    else:
        evals_all = [
            ev
            for ev in evaluations
            if label_key(ev.unit.proposition["entity"]) == target_entity
            and ev.unit.proposition["predicate"] == predicate
        ]

    # 1. Check unknown / unrepresentable conditions
    gap_issues = [
        AuthorizationIssue(
            code=ev.issue_code,
            stage=ev.issue_stage or "epistemic_integrity",
            disposition=ev.issue_disposition or "ESCALATE",
            predicate=predicate,
            message=ev.issue_message or f"Condition gap on {entity}.{predicate}; cannot authorize.",
            detail=ev.issue_detail,
        )
        for ev in evals_all
        if not ev.admissible and ev.issue_code in {"UNKNOWN_CONDITION", "UNREPRESENTABLE_CONDITION"}
    ]
    if gap_issues:
        return {"status": "BLOCKED_GAP", "value": None, "winning_units": (), "issues": tuple(gap_issues)}

    # 2. Check invalid or pending conditions
    cond_issues = [
        AuthorizationIssue(
            code=ev.issue_code,
            stage=ev.issue_stage or "epistemic_integrity",
            disposition=ev.issue_disposition or "ESCALATE",
            predicate=predicate,
            message=ev.issue_message or f"Condition check failed on {entity}.{predicate}.",
            detail=ev.issue_detail,
        )
        for ev in evals_all
        if not ev.admissible and ev.issue_code in {"INVALID_CONDITION_VALUE", "PENDING_CONDITION"}
    ]
    if cond_issues:
        status_code = (
            "INVALID_CONDITION"
            if any(i.code == "INVALID_CONDITION_VALUE" for i in cond_issues)
            else "PENDING_CONDITION"
        )
        return {"status": status_code, "value": None, "winning_units": (), "issues": tuple(cond_issues)}

    admissible_evals = [ev for ev in evals_all if ev.admissible]
    if not admissible_evals:
        unassessed_evals = [
            ev
            for ev in evals_all
            if not ev.admissible and ev.issue_code == "UNASSESSED_READER_RELIABILITY"
        ]
        if unassessed_evals:
            unassessed_issues = tuple(
                AuthorizationIssue(
                    code=ev.issue_code,
                    stage=ev.issue_stage or "epistemic_integrity",
                    disposition=ev.issue_disposition or "ESCALATE",
                    predicate=predicate,
                    message=ev.issue_message or f"Unassessed reliability on {entity}.{predicate}; cannot authorize.",
                    detail=ev.issue_detail,
                )
                for ev in unassessed_evals
            )
            return {
                "status": "UNASSESSED",
                "value": None,
                "winning_units": tuple(ev.unit for ev in unassessed_evals),
                "issues": unassessed_issues,
            }
        saw_false_condition = any(not ev.admissible and ev.reason == "condition_not_met" for ev in evals_all)
        return {
            "status": "NO_APPLICABLE_RULE" if saw_false_condition else "NO_EVIDENCE",
            "value": None,
            "winning_units": (),
            "issues": (),
        }

    # 3. Apply explicit ClauseRelation overrides (amends_clause, supersedes_clause, exception_to)
    applicable_clauses = {
        ev.unit.proposition.get("scope", {}).get("clause_id")
        for ev in admissible_evals
        if ev.unit.proposition.get("scope", {}).get("clause_id")
    }
    suppressed_clauses: set[str] = set()
    for rel in clause_relations:
        if not _is_active_at(rel.valid_from, rel.valid_until, as_of):
            continue
        if rel.source_clause_id in applicable_clauses and rel.target_clause_id in applicable_clauses:
            suppressed_clauses.add(rel.target_clause_id)

    surviving = [
        ev
        for ev in admissible_evals
        if ev.unit.proposition.get("scope", {}).get("clause_id") not in suppressed_clauses
    ]
    if not surviving:
        return {"status": "NO_APPLICABLE_RULE", "value": None, "winning_units": (), "issues": ()}

    # 4. Precedence by rank (document tier / weakest-premise tier) -> condition specificity -> scope specificity
    def _rank_key(ev: UnitEvaluation) -> tuple[int, int, int]:
        cond_spec = len(ev.unit.proposition.get("conditions") or ())
        scope_spec = _scope_specificity(ev.unit.proposition.get("scope", {}))
        return (ev.rank, cond_spec, scope_spec)

    best_rank = max(_rank_key(ev) for ev in surviving)
    top_evals = [ev for ev in surviving if _rank_key(ev) == best_rank]
    top_units = [ev.unit for ev in top_evals]

    # 5. Group by assertion_key and compute dependence-aware support/opposition (Q_g, S_g - O_g)
    spec = vocabulary.predicate(predicate)
    groups: dict[str, list[UnitEvaluation]] = {}
    for ev in top_evals:
        groups.setdefault(ev.unit.assertion_key, []).append(ev)

    def _group_support(g_evals: Sequence[UnitEvaluation]) -> float | None:
        assessed = sorted(
            (ev for ev in g_evals if ev.support is not None),
            key=lambda ev: (-ev.support, ev.unit.unit_id),
        )
        if not assessed:
            return None
        used_deps: set[str] = set()
        selected_weights: list[float] = []
        for ev in assessed:
            if not (ev.dependence & used_deps):
                used_deps |= ev.dependence
                selected_weights.append(ev.support)
        return _noisy_or(selected_weights)

    def _opposition_over(competing_keys: Sequence[str]) -> float:
        by_dep: dict[tuple[str, ...], float] = {}
        for k in competing_keys:
            for ev in groups[k]:
                if ev.support is None:
                    continue
                dkey = tuple(sorted(ev.dependence))
                by_dep[dkey] = max(by_dep.get(dkey, 0.0), ev.support)
        merged: dict[tuple[str, ...], float] = {}
        for dkey, val in by_dep.items():
            clash = [mk for mk in merged if set(mk) & set(dkey)]
            for mk in clash:
                val = max(val, merged.pop(mk))
                dkey = tuple(sorted(set(dkey) | set(mk)))
            merged[dkey] = val
        return _noisy_or(list(merged.values()))

    group_scores: dict[str, float | None] = {
        akey: _group_support(g_evals)
        for akey, g_evals in groups.items()
    }
    assessed_scores: dict[str, float] = {k: v for k, v in group_scores.items() if v is not None}

    if not assessed_scores:
        unassessed_issue = AuthorizationIssue(
            code="UNASSESSED_READER_RELIABILITY",
            stage="epistemic_integrity",
            disposition="ESCALATE",
            predicate=predicate,
            message=f"Units for {entity}.{predicate} have unassessed structural_reliability=None and cannot authorize operations.",
            detail={"unit_ids": [u.unit_id for u in top_units]},
        )
        return {"status": "UNASSESSED", "value": None, "winning_units": tuple(top_units), "issues": (unassessed_issue,)}

    winner_key = max(assessed_scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    winner_evals = groups[winner_key]
    winner_units = tuple(ev.unit for ev in winner_evals)
    s_g = assessed_scores[winner_key]

    incompatible_keys: list[str] = []
    unassessed_conflict = False
    p_win = winner_units[0].proposition
    for k in groups:
        if k == winner_key:
            continue
        p_other = groups[k][0].unit.proposition
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
        comp_issue = AuthorizationIssue(
            code="UNRESOLVED_COMPARISON",
            stage="authorization_policy",
            disposition="ESCALATE",
            predicate=predicate,
            message=f"Competing proposal for {entity}.{predicate} cannot be resolved deterministically (unassessed or incommensurable unit).",
            detail={"group_scores": group_scores},
        )
        return {"status": "CONFLICT", "value": None, "winning_units": tuple(top_units), "issues": (comp_issue,)}

    o_g = _opposition_over(incompatible_keys)
    q_g = _q_index(s_g, o_g)

    if o_g > 0 and (q_g < pol.tau or (s_g - o_g) < pol.delta):
        conflict_issue = AuthorizationIssue(
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
        return {"status": "CONFLICT", "value": None, "winning_units": tuple(top_units), "issues": (conflict_issue,)}

    if q_g < pol.tau or (s_g - o_g) < pol.delta:
        support_issue = AuthorizationIssue(
            code="INSUFFICIENT_SUPPORT",
            stage="authorization_policy",
            disposition="ESCALATE",
            predicate=predicate,
            message=f"Support for {entity}.{predicate} below AcceptancePolicy (Q={q_g:.3f}, margin={s_g - o_g:.3f}).",
        )
        return {"status": "LOW_SUPPORT", "value": None, "winning_units": winner_units, "issues": (support_issue,)}

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

    # Single unified admissibility evaluation for every candidate unit matching (entity, required_predicates)
    all_units: list[EvidenceUnit] = state["units"]
    units_by_id = {u.unit_id: u for u in all_units}
    stored_fe_map: dict[str, dict[str, dict[str, Any]]] = state["field_evidences"]
    layer_docs: dict[str, SourceDocument] = state["documents"]
    doc_revs: dict[tuple[str, str], SourceDocument] = state["document_revisions"]
    required_pred_set = set(required_predicates)
    condition_codes = {
        "UNKNOWN_CONDITION",
        "UNREPRESENTABLE_CONDITION",
        "INVALID_CONDITION_VALUE",
        "PENDING_CONDITION",
        "UNASSESSED_READER_RELIABILITY",
    }

    evals_by_pred: dict[str, list[UnitEvaluation]] = {pred: [] for pred in required_predicates}
    verified_eligible_units: list[EvidenceUnit] = []
    for u in all_units:
        p = u.proposition
        if label_key(p["entity"]) != target_entity:
            continue
        pred = p["predicate"]
        if pred not in required_pred_set:
            continue
        ev = evaluate_unit_admissibility(
            u,
            vocabulary=layer.vocabulary,
            policy=layer.policy,
            profile=layer.profile,
            units_by_id=units_by_id,
            rules=layer._rules,
            documents=layer_docs,
            document_revisions=doc_revs,
            as_of=as_of,
            query_scope=query_scope,
            query_conditions=query_conditions,
            condition_mode="discharge",
            field_evidences=field_evidences,
            stored_field_evidences=stored_fe_map,
            require_field_evidence=require_field_evidence,
            source_tiers=source_tiers,
            source_weights=source_weights,
            check_evidence_refs=True,
            skip_fidelity=False,
        )
        if ev.admissible:
            verified_eligible_units.append(u)
            evals_by_pred[pred].append(ev)
            continue
        if ev.issue_code in condition_codes or ev.reason == "condition_not_met":
            evals_by_pred[pred].append(ev)
            continue
        if ev.fidelity_findings:
            for ff in ev.fidelity_findings:
                epistemic_issues.append(
                    AuthorizationIssue(
                        code=ff.code,
                        stage="epistemic_integrity",
                        disposition="RETRY",
                        predicate=pred,
                        message=ff.message,
                        detail={"unit_id": u.unit_id, "field": ff.field, "detail": ff.detail},
                    )
                )
            continue
        if ev.issue_code:
            epistemic_issues.append(
                AuthorizationIssue(
                    code=ev.issue_code,
                    stage=ev.issue_stage or "epistemic_integrity",
                    disposition=ev.issue_disposition or "RETRY",
                    predicate=pred,
                    message=ev.issue_message or (ev.reason or ev.issue_code),
                    detail=ev.issue_detail,
                )
            )

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
            document_revisions=doc_revs,
            units_by_id=units_by_id,
            rules=layer._rules,
            evaluations=evals_by_pred[pred],
            policy=layer.policy,
            profile=layer.profile,
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
    auth_head = (
        str(operation_id)
        if operation_id is not None
        else f"__scope__:{target_entity}:{','.join(sorted(required_predicates))}"
    )

    decision_snapshot: dict[str, Any] = {
        "configuration": state["configuration"],
        "ledger_state_digest": state["ledger_state_digest"],
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
        "all_unit_ids": sorted(u.unit_id for u in all_units),
        "eligible_unit_ids": [u.unit_id for u in verified_eligible_units],
        "winning_unit_ids": {pred: winning_unit_ids_by_pred[pred] for pred in sorted(winning_unit_ids_by_pred)},
        "authorization_head": auth_head,
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
        head=auth_head,
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


def authorize(
    assessment: ResponseAssessment,
    operation: str | Mapping[str, Any] = "default_operation",
    policy: AcceptancePolicy | PolicyConfiguration | DomainProfile | None = None,
    *,
    layer: EvidenceLayer | None = None,
    payload: Mapping[str, Any] | None = None,
    signing_key: str | bytes | None = None,
) -> AuthorizationVerdict:
    """Decoupled authorization gate consuming a ResponseAssessment under a policy.

    Evaluations (`evaluate_response`) always run independently of policy; this gate
    decides whether an assessment justifies executing a concrete operational action
    (`PASS` / `RETRY` / `ESCALATE`). High weights can never bypass refuted checks,
    non-factual epistemic classes (`hypothesis`, `prediction`, `recommendation`),
    probabilistic-only critical steps, or incomplete verification coverage.
    """
    if not isinstance(assessment, ResponseAssessment):
        raise TypeError("assessment must be a ResponseAssessment.")

    if policy is None:
        pol_cfg = PolicyConfiguration()
    elif isinstance(policy, DomainProfile):
        pol_cfg = policy.policy
    elif isinstance(policy, PolicyConfiguration):
        pol_cfg = policy
    elif isinstance(policy, AcceptancePolicy):
        pol_cfg = PolicyConfiguration(tau=policy.tau, delta=policy.delta)
    else:
        raise TypeError("policy must be None, AcceptancePolicy, PolicyConfiguration, or DomainProfile.")

    op_id = (
        str(operation)
        if isinstance(operation, str)
        else str(operation.get("operation_id") or operation.get("name") or "default_operation")
    )

    issues: list[AuthorizationIssue] = []

    # 1. Hard formal refutation gate (never overridable by weights)
    refuted_checks = [c for c in assessment.completed_checks if c.status == "refuted"]
    if assessment.assessment == "refuted" or refuted_checks:
        issues.append(
            AuthorizationIssue(
                code="FORMAL_CHECK_REFUTED",
                stage="epistemic_integrity",
                disposition="RETRY",
                predicate=assessment.conclusion.predicate or None,
                message="One or more formal checks or premise extractions were refuted.",
                detail={
                    "reasons": [r for c in refuted_checks for r in c.reasons],
                    "assessment": assessment.assessment,
                },
            )
        )

    # 2. Epistemic claim class check (hypotheses/predictions/recommendations cannot authorize)
    if assessment.effective_epistemic_kind not in pol_cfg.allowed_claim_kinds_for_authorization:
        issues.append(
            AuthorizationIssue(
                code="NON_FACTUAL_EPISTEMIC_KIND_FOR_OPERATION",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    f"Effective epistemic kind '{assessment.effective_epistemic_kind}' is not "
                    f"permitted for operational authorization ({pol_cfg.allowed_claim_kinds_for_authorization})."
                ),
                detail={"effective_epistemic_kind": assessment.effective_epistemic_kind},
            )
        )

    # 3. Critical unverified steps / pending obligations
    if pol_cfg.block_critical_unverified_in_authorization and assessment.critical_unverified_steps:
        issues.append(
            AuthorizationIssue(
                code="CRITICAL_UNVERIFIED_INFERENCE_STEPS",
                stage="epistemic_integrity",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    f"Critical inference steps remain unverified: {list(assessment.critical_unverified_steps)}."
                ),
                detail={"critical_unverified_steps": list(assessment.critical_unverified_steps)},
            )
        )

    # 4. Critical steps relying solely on open_semantics probabilistic models
    if pol_cfg.require_deterministic_critical_steps:
        checks_by_step: dict[str, list[Any]] = {}
        for chk in assessment.completed_checks:
            checks_by_step.setdefault(chk.step_id, []).append(chk)
        prob_only_steps: list[str] = []
        for st_info in assessment.inference_steps:
            sid = str(st_info.get("step_id", ""))
            st_checks = checks_by_step.get(sid, [])
            verified_checks = [c for c in st_checks if c.status == "verified"]
            if verified_checks and all(c.guarantee_kind == "probabilistic_model" for c in verified_checks):
                prob_only_steps.append(sid)
        if prob_only_steps:
            issues.append(
                AuthorizationIssue(
                    code="PROBABILISTIC_ONLY_CRITICAL_STEP",
                    stage="epistemic_integrity",
                    disposition="ESCALATE",
                    predicate=assessment.conclusion.predicate or None,
                    message=(
                        f"Inference steps {prob_only_steps} rely solely on probabilistic_model checks "
                        "without deterministic verification."
                    ),
                    detail={"probabilistic_only_steps": prob_only_steps},
                )
            )

    # 5. Coverage requirement
    if assessment.verification_coverage < pol_cfg.min_verification_coverage:
        issues.append(
            AuthorizationIssue(
                code="INSUFFICIENT_VERIFICATION_COVERAGE",
                stage="epistemic_integrity",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    f"Verification coverage {assessment.verification_coverage:.3f} is below "
                    f"required {pol_cfg.min_verification_coverage:.3f}."
                ),
                detail={
                    "verification_coverage": assessment.verification_coverage,
                    "required_coverage": pol_cfg.min_verification_coverage,
                },
            )
        )

    # 6. Conflict and support thresholds
    if assessment.assessment == "contested" or assessment.conflict_index > pol_cfg.max_conflict_index:
        issues.append(
            AuthorizationIssue(
                code="HIGH_EVIDENCE_CONFLICT",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    f"Conflict index {assessment.conflict_index:.3f} exceeds max allowed "
                    f"{pol_cfg.max_conflict_index:.3f}."
                ),
                detail={"conflict_index": assessment.conflict_index, "max_conflict_index": pol_cfg.max_conflict_index},
            )
        )

    margin = assessment.evidence_support_index - assessment.conflict_index
    if (
        assessment.evidence_support_index < pol_cfg.tau
        or assessment.uncertainty_index > pol_cfg.max_uncertainty_index
        or margin < pol_cfg.delta
    ):
        issues.append(
            AuthorizationIssue(
                code="INSUFFICIENT_EVIDENCE_SUPPORT",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    f"Support={assessment.evidence_support_index:.3f} (tau={pol_cfg.tau}), "
                    f"margin={margin:.3f} (delta={pol_cfg.delta}), "
                    f"uncertainty={assessment.uncertainty_index:.3f} (max={pol_cfg.max_uncertainty_index})."
                ),
                detail={
                    "evidence_support_index": assessment.evidence_support_index,
                    "conflict_index": assessment.conflict_index,
                    "uncertainty_index": assessment.uncertainty_index,
                    "margin": round(margin, 6),
                },
            )
        )

    # 6b. Exploratory / approximate / source-conditional / subtotal gate
    exploratory_reasons = {
        r
        for chk in assessment.completed_checks
        for r in chk.reasons
        if r in {
            "TEMPORAL_APPROXIMATION_WITHIN_DERIVED_BOUNDS",
            "AGGREGATION_SUBTOTAL_VERIFIED",
            "SOURCE_CONDITIONAL_ATTRIBUTION_VERIFIED",
        }
    }
    if assessment.conclusion.approximate or exploratory_reasons:
        issues.append(
            AuthorizationIssue(
                code="EXPLORATORY_QUALIFIED_CONCLUSION_NOT_OPERATIONAL",
                stage="authorization_policy",
                disposition="ESCALATE",
                predicate=assessment.conclusion.predicate or None,
                message=(
                    "Conclusion is verified only as an approximate, partial-subtotal, or source-conditional "
                    "narrative explanation; autonomous operational execution requires exact corroborated evidence."
                ),
                detail={
                    "approximate": bool(assessment.conclusion.approximate),
                    "exploratory_reasons": sorted(exploratory_reasons),
                },
            )
        )

    # 7. Ledger unread/read_error state check if bound to an EvidenceLayer
    state: dict[str, Any] | None = None
    layer_docs: dict[str, SourceDocument] = {}
    all_units: tuple[EvidenceUnit, ...] = ()
    if layer is not None:
        state = layer.snapshot()
        layer_docs = state["documents"]
        all_units = state["units"]
        if state["unread_documents"]:
            issues.append(
                AuthorizationIssue(
                    code="UNREAD_DOCUMENTS_IN_LEDGER",
                    stage="epistemic_integrity",
                    disposition="RETRY",
                    message=f"Ledger contains unread documents: {state['unread_documents']}.",
                    detail={"unread_documents": list(state["unread_documents"])},
                )
            )
        if state["read_errors"]:
            issues.append(
                AuthorizationIssue(
                    code="READ_ERRORS_IN_LEDGER",
                    stage="epistemic_integrity",
                    disposition="RETRY",
                    message="Ledger contains unresolved reader extraction errors.",
                    detail={"read_errors": list(state["read_errors"])},
                )
            )

    all_issues = tuple(issues)
    if any(i.disposition == "RETRY" for i in all_issues):
        status = "RETRY"
    elif any(i.disposition == "ESCALATE" for i in all_issues):
        status = "ESCALATE"
    else:
        status = "PASS"

    conc = assessment.conclusion
    pred_key = conc.predicate or conc.claim_id
    resolved_values = (
        {
            pred_key: {
                "claim_id": conc.claim_id,
                "subject": conc.subject,
                "predicate": conc.predicate,
                "value": conc.value,
                "unit": conc.unit,
                "epistemic_kind": assessment.effective_epistemic_kind,
            }
        }
        if status == "PASS"
        else {}
    )

    eff_payload_digest = (
        digest(payload)
        if payload is not None
        else digest({"operation_id": op_id, "conclusion_id": conc.claim_id, "assessment_digest": assessment.assessment_digest})
    )

    cov_snap = getattr(assessment, "corpus_coverage", None) or {}
    decision_snapshot: dict[str, Any] = {
        "operation_id": op_id,
        "payload_digest": eff_payload_digest,
        "assessment_digest": assessment.assessment_digest,
        "domain_profile_fingerprint": assessment.domain_profile_fingerprint,
        "conclusion_id": assessment.conclusion_id,
        "effective_epistemic_kind": assessment.effective_epistemic_kind,
        "verification_coverage": assessment.verification_coverage,
        "evidence_support_index": assessment.evidence_support_index,
        "conflict_index": assessment.conflict_index,
        "uncertainty_index": assessment.uncertainty_index,
        "resolved_values": resolved_values,
        "status": status,
        "issues": [asdict(i) for i in all_issues],
        "authorization_head": op_id,
        "corpus_obligations_total": int(cov_snap.get("total_obligations", 0)),
        "corpus_obligations_covered": int(cov_snap.get("covered_count", 0)),
        "corpus_obligations_rejected": int(cov_snap.get("rejected_count", 0)),
        "corpus_obligation_ids": list(cov_snap.get("obligation_ids") or ()),
    }

    if state is not None:
        decision_snapshot["configuration"] = state["configuration"]
        decision_snapshot["ledger_state_digest"] = state["ledger_state_digest"]
        decision_snapshot["documents"] = [
            {
                "document_id": d.document_id,
                "revision": d.revision,
                "authority_rank": d.authority_rank,
                "weight": d.weight,
                "dependence_group": d.dependence_group,
                "attribution": d.attribution,
            }
            for _, d in sorted(layer_docs.items())
        ]
        decision_snapshot["all_unit_ids"] = sorted(u.unit_id for u in all_units)
        decision_snapshot["eligible_unit_ids"] = sorted(conc.evidence_unit_ids)
        decision_snapshot["winning_unit_ids"] = {pred_key: sorted(conc.evidence_unit_ids)}
        decision_snapshot["read_state"] = {
            "unread_documents": list(state["unread_documents"]),
            "read_errors": list(state["read_errors"]),
        }

    snapshot_digest = digest(decision_snapshot)
    active_key = signing_key if signing_key is not None else os.environ.get("CLAIMLAYER_SIGNING_KEY")
    signature: str | None = None
    if active_key is not None and active_key != "" and active_key != b"":
        key_bytes = active_key.encode("utf-8") if isinstance(active_key, str) else active_key
        signature = hmac.new(key_bytes, snapshot_digest.encode("utf-8"), hashlib.sha256).hexdigest()

    if layer is not None:
        auth_record = {
            "snapshot": decision_snapshot,
            "snapshot_digest": snapshot_digest,
            "signature": signature,
        }
        layer.store.put("authorization", snapshot_digest, auth_record, head=op_id)

    return AuthorizationVerdict(
        status=status,
        allowed=(status == "PASS"),
        requires_human=(status == "ESCALATE"),
        resolved_values=resolved_values,
        issues=all_issues,
        metrics={
            "evidence_support_index": assessment.evidence_support_index,
            "conflict_index": assessment.conflict_index,
            "uncertainty_index": assessment.uncertainty_index,
            "verification_coverage": assessment.verification_coverage,
            "critical_unverified_steps": len(assessment.critical_unverified_steps),
        },
        operation_id=op_id,
        payload_digest=eff_payload_digest,
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
    allow_consumed: bool = False,
) -> bool:
    """Verify authenticity, snapshot integrity, ledger-state freshness, and operation/payload binding."""
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
        if snap_dict.get("ledger_state_digest") != current_state.get("ledger_state_digest"):
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
        if set(snap_dict.get("all_unit_ids", ())) != current_unit_ids:
            return False
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
        expected_head = (
            str(verdict.operation_id)
            if verdict.operation_id is not None
            else (snap_dict.get("authorization_head") or verdict.snapshot_digest)
        )
        try:
            active_head_id, _ = layer.store.head("authorization", expected_head)
        except (ValueError, TypeError, KeyError):
            return False
        if active_head_id != verdict.snapshot_digest:
            return False
        try:
            exec_state = get_execution_state(layer, verdict.snapshot_digest)
        except (ValueError, TypeError, KeyError):
            return False
        if isinstance(exec_state, Mapping):
            state_val = exec_state.get("state")
            if state_val in {"UNCERTAIN", "ABORTED", "IN_FLIGHT"}:
                return False
            if not allow_consumed and exec_state.get("consumed"):
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


EXECUTION_STATES = frozenset({"IN_FLIGHT", "COMPLETED", "UNCERTAIN", "ABORTED", "ABORTED_RETRYABLE"})
_RECONCILIATION_RESOLUTIONS = frozenset({"CONFIRM_COMPLETED", "CONFIRM_ABORTED_ALLOW_RETRY", "MARK_ABORTED"})


def compute_execution_idempotency_key(verdict: AuthorizationVerdict) -> str:
    """Return a deterministic idempotency key bound to verdict snapshot, operation, and payload."""
    return digest(
        {
            "snapshot_digest": verdict.snapshot_digest,
            "operation_id": verdict.operation_id,
            "payload_digest": verdict.payload_digest,
        }
    )


def get_execution_state(
    layer: EvidenceLayer,
    verdict_or_digest: AuthorizationVerdict | str,
) -> dict[str, Any] | None:
    """Return the latest persisted execution record for an AuthorizationVerdict, or None."""
    snap_digest = (
        verdict_or_digest.snapshot_digest
        if isinstance(verdict_or_digest, AuthorizationVerdict)
        else str(verdict_or_digest)
    )
    if not snap_digest:
        return None
    _, head_record = layer.store.head("execution", snap_digest)
    if isinstance(head_record, Mapping):
        return dict(head_record)
    legacy_record = layer.store.get("execution", snap_digest)
    if isinstance(legacy_record, Mapping):
        return dict(legacy_record)
    return None


def reconcile_authorized_execution(
    layer: EvidenceLayer,
    verdict_or_digest: AuthorizationVerdict | str,
    *,
    resolution: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Explicitly resolve an 'UNCERTAIN' or 'IN_FLIGHT' execution record after external verification."""
    if resolution not in _RECONCILIATION_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution {resolution!r}; expected one of {sorted(_RECONCILIATION_RESOLUTIONS)}."
        )
    snap_digest = (
        verdict_or_digest.snapshot_digest
        if isinstance(verdict_or_digest, AuthorizationVerdict)
        else str(verdict_or_digest)
    )
    with layer.store.transaction(mode="IMMEDIATE"):
        existing = get_execution_state(layer, snap_digest)
        if not isinstance(existing, Mapping):
            raise ValueError(f"No execution state recorded for verdict {snap_digest!r}.")
        cur_state = existing.get("state")
        if cur_state not in {"UNCERTAIN", "IN_FLIGHT"}:
            raise ValueError(
                f"Execution state {cur_state!r} for verdict {snap_digest!r} cannot be reconciled; "
                "expected 'UNCERTAIN' or 'IN_FLIGHT'."
            )
        next_seq = int(existing.get("transition_seq", 0)) + 1
        if resolution == "CONFIRM_COMPLETED":
            next_state, next_consumed = "COMPLETED", True
        elif resolution == "MARK_ABORTED":
            next_state, next_consumed = "ABORTED", True
        else:  # CONFIRM_ABORTED_ALLOW_RETRY
            next_state, next_consumed = "ABORTED_RETRYABLE", False

        reconciled = {
            "snapshot_digest": snap_digest,
            "operation_id": existing.get("operation_id"),
            "payload_digest": existing.get("payload_digest"),
            "ledger_state_digest": existing.get("ledger_state_digest"),
            "idempotency_key": existing.get("idempotency_key"),
            "one_shot": bool(existing.get("one_shot", True)),
            "attempt_count": int(existing.get("attempt_count", 1)),
            "transition_seq": next_seq,
            "state": next_state,
            "consumed": next_consumed,
            "reconciled_from": cur_state,
            "reconciled_resolution": resolution,
            "notes": notes,
        }
        layer.store.put("execution", digest(reconciled), reconciled, head=snap_digest)
        return dict(reconciled)


def execute_authorized_operation(
    layer: EvidenceLayer,
    verdict: AuthorizationVerdict,
    action: Any,
    *,
    operation_id: str | None = None,
    payload: Any | None = None,
    payload_digest: str | None = None,
    signing_key: str | bytes | None = None,
    allow_unkeyed: bool = False,
    one_shot: bool = True,
) -> Any:
    """Execute *action(verdict)* using a 3-phase persisted intent/state protocol.

    Phase 1 (SQLite IMMEDIATE tx): Verify verdict & existing state, then commit
    an `'IN_FLIGHT'` (`consumed=True`) record with a deterministic `idempotency_key`
    BEFORE invoking the external callback.
    Phase 2 (outside SQLite tx): Invoke `action(verdict)`.
    Phase 3 (new SQLite IMMEDIATE tx): Persist `'COMPLETED'` on success or
    `'UNCERTAIN'` (`consumed=True`) on callback exception so SQLite rollback
    never un-consumes a potentially executed external side effect.
    """
    if not callable(action):
        raise TypeError("action must be callable")

    # Phase 1: Verify and commit execution intent before any external callback runs
    with layer.store.transaction(mode="IMMEDIATE"):
        snap_digest = getattr(verdict, "snapshot_digest", None)
        existing = get_execution_state(layer, snap_digest) if isinstance(snap_digest, str) and snap_digest else None
        if isinstance(existing, Mapping):
            cur_state = existing.get("state")
            if cur_state == "UNCERTAIN":
                raise PermissionError(
                    f"AuthorizationVerdict {snap_digest!r} is in UNCERTAIN state after a prior failure; "
                    "reconcile execution state before retrying."
                )
            if cur_state == "IN_FLIGHT":
                raise PermissionError(f"AuthorizationVerdict {snap_digest!r} is currently IN_FLIGHT.")
            if cur_state == "ABORTED":
                raise PermissionError(f"AuthorizationVerdict {snap_digest!r} was permanently ABORTED.")
            if one_shot and existing.get("consumed"):
                raise PermissionError(f"AuthorizationVerdict {snap_digest!r} was already consumed.")

        if not verify_authorization_verdict(
            verdict,
            layer=layer,
            operation_id=operation_id,
            payload=payload,
            payload_digest=payload_digest,
            signing_key=signing_key,
            allow_unkeyed=allow_unkeyed,
            allow_consumed=not one_shot or (isinstance(existing, Mapping) and existing.get("state") == "ABORTED_RETRYABLE"),
        ):
            raise PermissionError(
                f"AuthorizationVerdict {snap_digest!r} failed verification or was superseded/consumed."
            )

        prev_attempts = int(existing.get("attempt_count", 0)) if isinstance(existing, Mapping) else 0
        prev_seq = int(existing.get("transition_seq", 0)) if isinstance(existing, Mapping) else 0
        attempt_count = prev_attempts + 1
        seq_in_flight = prev_seq + 1
        idempotency_key = compute_execution_idempotency_key(verdict)
        ledger_state_digest = verdict.decision_snapshot.get("ledger_state_digest")

        in_flight_record = {
            "snapshot_digest": verdict.snapshot_digest,
            "operation_id": verdict.operation_id,
            "payload_digest": verdict.payload_digest,
            "ledger_state_digest": ledger_state_digest,
            "idempotency_key": idempotency_key,
            "one_shot": bool(one_shot),
            "attempt_count": attempt_count,
            "transition_seq": seq_in_flight,
            "state": "IN_FLIGHT",
            "consumed": True,
        }
        layer.store.put("execution", digest(in_flight_record), in_flight_record, head=verdict.snapshot_digest)

    # Phase 2: Run external callback outside the SQLite transaction so failures cannot roll back consumption
    try:
        result = action(verdict)
    except Exception as exc:
        # Phase 3a: Record UNCERTAIN state (remains consumed and blocks blind retries)
        with layer.store.transaction(mode="IMMEDIATE"):
            uncertain_record = {
                "snapshot_digest": verdict.snapshot_digest,
                "operation_id": verdict.operation_id,
                "payload_digest": verdict.payload_digest,
                "ledger_state_digest": ledger_state_digest,
                "idempotency_key": idempotency_key,
                "one_shot": bool(one_shot),
                "attempt_count": attempt_count,
                "transition_seq": seq_in_flight + 1,
                "state": "UNCERTAIN",
                "consumed": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            layer.store.put("execution", digest(uncertain_record), uncertain_record, head=verdict.snapshot_digest)
        raise

    # Phase 3b: Record COMPLETED state
    with layer.store.transaction(mode="IMMEDIATE"):
        completed_record = {
            "snapshot_digest": verdict.snapshot_digest,
            "operation_id": verdict.operation_id,
            "payload_digest": verdict.payload_digest,
            "ledger_state_digest": ledger_state_digest,
            "idempotency_key": idempotency_key,
            "one_shot": bool(one_shot),
            "attempt_count": attempt_count,
            "transition_seq": seq_in_flight + 1,
            "state": "COMPLETED",
            "consumed": bool(one_shot),
        }
        layer.store.put("execution", digest(completed_record), completed_record, head=verdict.snapshot_digest)
    return result
