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
import math
from typing import Any, Mapping, Sequence

from ..semantic.models import EvidenceRef, SourceDocument, canonical, digest, instant
from .engine import EvidenceLayer, _noisy_or, _q_index
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


def _is_active_at(valid_from: str | None, valid_until: str | None, as_of: str | None) -> bool:
    if as_of is None:
        return True
    ref = instant(as_of)
    if valid_from is not None and instant(valid_from) > ref:
        return False
    if valid_until is not None and instant(valid_until) < ref:
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


def _eval_condition(
    cond: Mapping[str, Any],
    query_conditions: Mapping[str, Any],
    vocabulary: Vocabulary,
) -> str:
    """Evaluate one declared condition against *query_conditions*.

    Returns 'true', 'false', or 'pending' (when query_conditions lacks the field).
    """
    field_name = cond["field"]
    if field_name not in query_conditions or query_conditions[field_name] is None:
        return "pending"

    op = cond["operator"]
    expected = cond["value"]
    actual = query_conditions[field_name]

    if isinstance(expected, dict) and "unit" in expected:
        if isinstance(actual, dict) and "value" in actual and "unit" in actual:
            amt, base = vocabulary.units.to_base(actual["value"], actual["unit"], f"query_conditions/{field_name}")
            if base != expected["unit"]:
                return "pending"
            lhs = Decimal(amt)
        else:
            lhs = Decimal(str(actual))
        rhs = Decimal(str(expected["value"]))
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if op == "eq":
            return "true" if bool(actual) is bool(expected) else "false"
        if op == "ne":
            return "true" if bool(actual) is not bool(expected) else "false"
        return "pending"
    elif op in {"lt", "le", "gt", "ge"}:
        lhs = Decimal(str(actual))
        rhs = Decimal(str(expected))
    else:
        eq = label_key(str(actual)) == label_key(str(expected))
        if op == "eq":
            return "true" if eq else "false"
        if op == "ne":
            return "true" if not eq else "false"
        return "pending"

    if op == "eq":
        return "true" if lhs == rhs else "false"
    if op == "ne":
        return "true" if lhs != rhs else "false"
    if op == "lt":
        return "true" if lhs < rhs else "false"
    if op == "le":
        return "true" if lhs <= rhs else "false"
    if op == "gt":
        return "true" if lhs > rhs else "false"
    if op == "ge":
        return "true" if lhs >= rhs else "false"
    return "pending"


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

    matching = [
        u
        for u in units
        if label_key(u.proposition["entity"]) == target_entity
        and u.proposition["predicate"] == predicate
        and _is_active_at(u.proposition.get("valid_from"), u.proposition.get("valid_until"), as_of)
        and _scope_matches(u.proposition.get("scope", {}), q_scope)
    ]

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

    # 2. Evaluate conditions against query_conditions
    applicable: list[tuple[EvidenceUnit, int]] = []
    for u in matching:
        conds = u.proposition.get("conditions") or ()
        if not conds:
            applicable.append((u, 0))
            continue
        results = [_eval_condition(c, q_conds, vocabulary) for c in conds]
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
        return {"status": "PENDING_CONDITION", "value": None, "winning_units": (), "issues": tuple(issues)}

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
        return max(0.0, min(1.0, doc_w * float(u.structural_reliability)))

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
    val: Any,
    vocabulary: Vocabulary,
) -> tuple[Decimal | None, Decimal | None, str | None]:
    """Resolve finite Decimal (low, high, target_unit) bounds with vocabulary unit normalization."""
    if not numeric_bounds:
        return None, None, None
    val_unit = val.get("unit") if isinstance(val, dict) else None
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
        return None, None, None

    if isinstance(bound_entry, Mapping):
        raw_lo = bound_entry.get("min")
        raw_hi = bound_entry.get("max")
        explicit_unit = bound_entry.get("unit", explicit_unit)
    elif isinstance(bound_entry, (tuple, list)) and len(bound_entry) == 3:
        raw_lo, raw_hi, explicit_unit = bound_entry
    elif isinstance(bound_entry, (tuple, list)) and len(bound_entry) == 2:
        raw_lo, raw_hi = bound_entry
    else:
        raise ValueError(f"Invalid numeric_bounds entry for '{pred}': {bound_entry!r}")

    lo = _finite_decimal(raw_lo, f"numeric_bounds[{pred}][0]") if raw_lo is not None else None
    hi = _finite_decimal(raw_hi, f"numeric_bounds[{pred}][1]") if raw_hi is not None else None

    target_unit = val_unit
    if explicit_unit is not None:
        if lo is not None:
            amt_lo, base_lo = vocabulary.units.to_base(lo, explicit_unit, f"numeric_bounds/{pred}/min")
            lo = Decimal(amt_lo)
            target_unit = base_lo
        if hi is not None:
            amt_hi, base_hi = vocabulary.units.to_base(hi, explicit_unit, f"numeric_bounds/{pred}/max")
            hi = Decimal(amt_hi)
            target_unit = base_hi
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"numeric_bounds[{pred}] lower bound {lo} exceeds upper bound {hi}")
    return lo, hi, target_unit


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
) -> AuthorizationVerdict:
    """Run the two-stage operational authorization gate over a v4 EvidenceLayer."""
    epistemic_issues: list[AuthorizationIssue] = []
    policy_issues: list[AuthorizationIssue] = []

    if payload is not None and payload_digest is None:
        payload_digest = digest(payload)

    # ---- Stage 1: Epistemic Integrity ----
    combined_read_errors = list(layer.read_errors)
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
        set(layer.unread_documents)
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
    for rej in layer.rejections():
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

    # Fidelity checks and stale-revision checks on all admitted units matching entity & required predicates
    all_units = layer.units
    stored_fe_map = layer.field_evidences
    doc_cache: dict[tuple[str, str], SourceDocument] = {}
    layer_docs = {d.document_id: d for d in layer.documents}
    stale_unit_ids: set[str] = set()

    for u in all_units:
        if label_key(u.proposition["entity"]) != target_entity:
            continue
        if u.proposition["predicate"] not in required_predicates:
            continue
        if u.proposition.get("assertion_kind") not in layer.policy.allowed_assertion_kinds:
            continue
        if not u.evidence:
            continue

        cited_doc: SourceDocument | None = None
        for ref in u.evidence:
            cache_key = (ref.document_id, ref.revision)
            if cache_key not in doc_cache:
                try:
                    doc_cache[cache_key] = layer._document(ref.document_id, ref.revision, current=False)
                except ValueError as exc:
                    stale_unit_ids.add(u.unit_id)
                    epistemic_issues.append(
                        AuthorizationIssue(
                            code="MISSING_EVIDENCE_DOCUMENT",
                            stage="epistemic_integrity",
                            disposition="RETRY",
                            predicate=u.proposition["predicate"],
                            message=f"Evidence document '{ref.document_id}' revision '{ref.revision}' unavailable: {exc}",
                            detail={"unit_id": u.unit_id, "document_id": ref.document_id, "revision": ref.revision},
                        )
                    )
                    continue
            if cited_doc is None:
                cited_doc = doc_cache[cache_key]
            current_doc = layer_docs.get(ref.document_id)
            if current_doc is None or current_doc.revision != ref.revision:
                stale_unit_ids.add(u.unit_id)
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

        if cited_doc is None or u.unit_id in stale_unit_ids:
            continue

        # Merge stored FieldEvidence from layer with caller-provided overrides
        fe_by_field: dict[str, FieldEvidence] = {}
        raw_stored_fe = stored_fe_map.get(u.unit_id)
        if raw_stored_fe:
            for fname, ref_dict in raw_stored_fe.items():
                try:
                    fe_by_field[fname] = FieldEvidence(fname, EvidenceRef(**ref_dict))
                except (TypeError, ValueError) as exc:
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
        )
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

    # ---- Stage 2: Scope/Precedence Resolution & Policy Gate ----
    resolved_values: dict[str, Any] = {}
    for pred in required_predicates:
        res = resolve_scope_and_precedence(
            all_units,
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
        )
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

        # Check operational numeric bounds (unit-normalized Decimal comparison)
        low, high, target_unit = _resolve_bound_limits(numeric_bounds, pred, val, layer.vocabulary)
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

    snapshot_digest = digest(
        {
            "configuration": layer.configuration,
            "entity": target_entity,
            "required_predicates": list(required_predicates),
            "query_scope": dict(query_scope or {}),
            "query_conditions": dict(query_conditions or {}),
            "resolved_values": resolved_values,
            "status": status,
            "operation_id": operation_id,
            "payload_digest": payload_digest,
        }
    )

    return AuthorizationVerdict(
        status=status,
        allowed=(status == "PASS"),
        requires_human=(status == "ESCALATE"),
        resolved_values=resolved_values,
        issues=all_issues,
        metrics={
            "epistemic_issues": len(epistemic_issues),
            "policy_issues": len(policy_issues),
            "predicates_resolved": len(resolved_values),
            "predicates_required": len(required_predicates),
        },
        operation_id=operation_id,
        payload_digest=payload_digest,
        snapshot_digest=snapshot_digest,
    )
