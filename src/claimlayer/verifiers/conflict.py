"""Deterministic peer-conflict resolution, Noisy-OR support, and threshold gates."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import TYPE_CHECKING, Iterable

from claimlayer.models import (
    Assertion,
    ConflictDiff,
    Issue,
    Policy,
    Source,
    normalize_entity_key,
    normalize_key,
    normalize_predicate_key,
)
from claimlayer.verifiers.diff import _split_into_sections
from claimlayer.verifiers.precedence import (
    _clauses_address_same_topic,
    _extract_conflicting_clause_value,
    _subjects_and_designators_compatible,
    is_source_active,
)
from claimlayer.verifiers.quote_quantity import (
    _to_commensurable_base,
    canonicalize_value,
    extract_quantities,
    infer_predicate_dimension,
    normalize_span,
    unit_dimension,
)

if TYPE_CHECKING:
    from claimlayer.ledger import Ledger

_CONDITIONAL_CLAUSE_RE = re.compile(
    r"\b(?:if|when|where|in\s+the\s+case\s+of|in\s+case\s+of|with\s+respect\s+to|"
    r"for\s+interest|for\s+principal|for\s+covenant|for\s+payment|for\s+non-payment|"
    r"prior\s+to|after|during|si\b|cuando\b|en\s+caso\s+de|respecto\s+a)\b",
    re.IGNORECASE,
)

_MAIN_VERB_SPLIT_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|shall|will|may|must|bears?|accrues?|es|son|tiene|ser[aá])\b",
    re.IGNORECASE,
)

_GENERIC_BOUND_SUFFIXES = frozenset(
    {
        "amount",
        "total",
        "value",
        "sum",
        "limit",
        "cap",
        "maximum",
        "max",
        "minimum",
        "min",
        "threshold",
        "floor",
        "ceiling",
        "rate",
        "fee",
        "cost",
        "price",
        "payout",
        "payment",
        "term",
        "period",
        "usd",
        "eur",
        "gbp",
        "days",
        "day",
        "pct",
        "percent",
    }
)


def noisy_or(scores: Iterable[float]) -> float:
    """Aggregate independent evidence probabilities via Noisy-OR."""
    prod_complement = 1.0
    has_any = False
    for s in scores:
        fval = float(s)
        if math.isnan(fval) or math.isinf(fval) or fval < 0.0 or fval > 1.0:
            raise ValueError(f"Invalid probability score in noisy_or: {s!r}")
        prod_complement *= 1.0 - fval
        has_any = True
    return round(1.0 - prod_complement, 6) if has_any else 0.0


def _effective_assertion_support(source: Source | None) -> float:
    """Determine assertion support from the operator-configured source reliability (never from self-declared agent confidence, C6)."""
    if source is not None and "reliability" in source.metadata:
        return max(0.0, min(1.0, float(source.metadata["reliability"])))
    return 1.0


def _effective_fact_support(fact_conf: float, source: Source | None) -> float:
    """Determine ledger FactRecord support capped by source reliability."""
    base = float(fact_conf)
    if source is not None and "reliability" in source.metadata:
        src_rel = float(source.metadata["reliability"])
        return max(0.0, min(base, src_rel))
    return max(0.0, min(1.0, base))


def _grammatical_subject_key(sentence: str) -> str:
    m = _MAIN_VERB_SPLIT_RE.search(sentence)
    if not m or m.start() == 0:
        return ""
    return normalize_entity_key(sentence[: m.start()].strip())


def _are_distinct_conditional_clauses(a1: Assertion, a2: Assertion) -> bool:
    """Return True if two assertions on the same source describe distinct conditional branches or sub-cases (B11 vs C3v)."""
    if a1.condition and a2.condition and normalize_key(a1.condition) != normalize_key(a2.condition):
        return True
    q1 = normalize_span(a1.quote)
    q2 = normalize_span(a2.quote)
    if q1.lower() == q2.lower():
        return False
    if _CONDITIONAL_CLAUSE_RE.search(q1) or _CONDITIONAL_CLAUSE_RE.search(q2):
        return True
    gs1 = _grammatical_subject_key(q1)
    gs2 = _grammatical_subject_key(q2)
    if gs1 and gs2 and gs1 != gs2:
        return True
    return False


def _dimension_compatible_with_bound(bound_key: str, q_unit: str) -> bool:
    b_dim = infer_predicate_dimension(bound_key)
    q_dim = unit_dimension(q_unit)
    if b_dim is None or q_dim == "NONE":
        return True
    if b_dim == "RATIO":
        return q_dim == "RATIO"
    if b_dim == "CURRENCY":
        return q_dim.startswith("CURRENCY")
    if b_dim == "DURATION":
        return q_dim.startswith("DURATION")
    return True


def _match_numeric_bound(
    assertion: Assertion,
    q_unit: str,
    policy: Policy,
) -> tuple[float | None, float | None] | None:
    """Match an assertion & quantity against policy.numeric_bounds with dimension and quote-topic awareness (C11b, C11v1, C11v2, C11fp)."""
    if not policy.numeric_bounds:
        return None
    predicate = assertion.predicate
    norm_pred = normalize_predicate_key(predicate, policy.predicate_aliases)
    raw_pred = normalize_key(predicate)

    for k, bounds in policy.numeric_bounds.items():
        if not _dimension_compatible_with_bound(k, q_unit):
            continue
        if (
            k == predicate
            or normalize_key(k) == raw_pred
            or normalize_predicate_key(k, policy.predicate_aliases) == norm_pred
        ):
            return bounds

    pred_stems = {t for t in norm_pred.split("_") if t and t not in _GENERIC_BOUND_SUFFIXES}
    quote_stems = {
        t
        for t in normalize_predicate_key(assertion.quote).split("_")
        if len(t) >= 3 and t not in _GENERIC_BOUND_SUFFIXES
    }

    for k, bounds in policy.numeric_bounds.items():
        if not _dimension_compatible_with_bound(k, q_unit):
            continue
        k_norm = normalize_predicate_key(k, policy.predicate_aliases)
        k_stems = {t for t in k_norm.split("_") if t and t not in _GENERIC_BOUND_SUFFIXES}
        if not k_stems:
            continue
        if pred_stems == k_stems or k_stems.issubset(pred_stems) or k_stems.issubset(quote_stems):
            return bounds

    return None


def _find_indexed_peer_clause_conflict(
    assertion: Assertion,
    sec_body: str | None,
    clauses: list[str],
) -> tuple[str, str] | None:
    """Check pre-indexed section/clauses of a peer source for a conflicting value (C2, C2v1-v3, C2fp)."""
    if sec_body is not None and _subjects_and_designators_compatible(assertion, None, sec_body):
        comp_val = _extract_conflicting_clause_value(assertion, sec_body)
        if comp_val is not None:
            return comp_val, sec_body[:160]

    for sent_clean in clauses:
        if not _subjects_and_designators_compatible(assertion, None, sent_clean):
            continue
        if not _clauses_address_same_topic(assertion, sent_clean):
            continue
        comp_val = _extract_conflicting_clause_value(assertion, sent_clean)
        if comp_val is not None:
            return comp_val, sent_clean[:160]

    return None


def evaluate_conflicts_and_thresholds(
    assertion: Assertion,
    source: Source,
    all_assertions: tuple[Assertion, ...],
    ledger: Ledger,
    policy: Policy,
    as_of: str | None = None,
    *,
    assertion_index: int = 0,
) -> tuple[float, list[Issue], list[ConflictDiff]]:
    """Compute content-deduplicated Noisy-OR support and detect peer conflicts or policy bounds."""
    issues: list[Issue] = []
    conflicts: list[ConflictDiff] = []
    effective_as_of = as_of or policy.as_of

    subj_key = normalize_entity_key(assertion.subject)
    pred_key = normalize_predicate_key(assertion.predicate, policy.predicate_aliases)
    raw_pred_key = normalize_key(assertion.predicate)
    proposed_canon = canonicalize_value(assertion.value, assertion.unit)

    superseded_ids = ledger.superseded_source_ids(
        as_of=effective_as_of,
        subject=assertion.subject,
        predicate=assertion.predicate,
    )

    # 1. Human resolution check (unless superseded by a newer active amendment, C8)
    resolution = ledger.get_resolution(assertion.subject, assertion.predicate)
    if resolution is not None:
        resolved_canon = canonicalize_value(resolution.selected_value)
        # Check if the current source is an active superseding amendment that updated the resolved value
        is_newer_amendment = bool(source.supersedes) and source.id not in superseded_ids
        if proposed_canon != resolved_canon and not is_newer_amendment:
            issues.append(
                Issue(
                    code="CONTRADICTS_HUMAN_RESOLUTION",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Human reviewer '{resolution.resolver}' resolved {assertion.subject}.{assertion.predicate} "
                        f"to '{resolution.selected_value}', but agent proposed '{assertion.value}'."
                    ),
                    remediation=(
                        f"Adopt human-approved value '{resolution.selected_value}' for '{assertion.predicate}'."
                    ),
                )
            )
            return 0.0, issues, conflicts
        net_confidence = 1.0
    else:
        # 2. Gather peer-tier evidence deduplicated by content hash (source.sha256, N1)
        # canon_val -> dedup_key (sha256) -> (confidence, raw_value, quote, tier, source_id)
        by_canon_source: dict[str, dict[str, tuple[float, str, str, str, str]]] = {}

        def _record_candidate(
            canon_val: str,
            src_obj: Source | None,
            src_id: str,
            conf: float,
            raw_val: str,
            quote_txt: str,
            tier_name: str,
        ) -> None:
            dedup_key = src_obj.content_sha256 if src_obj is not None else f"id:{src_id}"
            bucket = by_canon_source.setdefault(canon_val, {})
            prev = bucket.get(dedup_key)
            if prev is None or conf > prev[0]:
                bucket[dedup_key] = (conf, raw_val, quote_txt, tier_name, src_id)

        own_eff_conf = _effective_assertion_support(source)
        _record_candidate(
            proposed_canon,
            source,
            source.id,
            own_eff_conf,
            assertion.value,
            assertion.quote,
            source.tier,
        )

        # Check sibling assertions in the same proposal batch (by index, closing C3b duplicate-id bypass!)
        for other_idx, other in enumerate(all_assertions):
            if other_idx == assertion_index:
                continue
            other_subj = normalize_entity_key(other.subject)
            other_pred = normalize_predicate_key(other.predicate, policy.predicate_aliases)
            if other_subj == subj_key and (
                other_pred == pred_key or normalize_key(other.predicate) == raw_pred_key
            ):
                other_src = ledger.get_source(other.source_id)
                if other_src is not None and other_src.id == source.id:
                    if _are_distinct_conditional_clauses(assertion, other):
                        continue
                other_tier = other_src.tier if other_src else "standard"
                other_rank = other_src.rank if other_src else source.rank
                if other_rank >= source.rank:
                    _record_candidate(
                        canonicalize_value(other.value, other.unit),
                        other_src,
                        other.source_id,
                        _effective_assertion_support(other_src),
                        other.value,
                        other.quote,
                        other_tier,
                    )

        # Check active peer-rank facts in ledger (excluding superseded sources)
        indexed_peer_ids: set[str] = set()
        for fact in ledger.get_facts(
            assertion.subject,
            assertion.predicate,
            aliases=policy.predicate_aliases,
            allow_section_predicate_match=True,
        ):
            if fact.source_id in superseded_ids:
                continue
            fact_src = ledger.get_source(fact.source_id)
            if fact_src is None:
                continue
            active_src, _, _ = is_source_active(fact_src, effective_as_of)
            if not active_src:
                continue
            if fact_src.rank == source.rank:
                indexed_peer_ids.add(fact.source_id)
                _record_candidate(
                    canonicalize_value(fact.value, fact.unit),
                    fact_src,
                    fact.source_id,
                    _effective_fact_support(fact.confidence, fact_src),
                    fact.value,
                    fact.quote or fact_src.text[:160],
                    fact_src.tier,
                )

        # Query Ledger's pre-computed clause index for active peer-rank conflicts (closing C2 omission bypass in O(1)!)
        for peer_src, sec_body, clauses in ledger.get_indexed_source_clauses(
            assertion, raw_pred_key, exclude_source_id=source.id
        ):
            if peer_src.id in superseded_ids or peer_src.id in indexed_peer_ids:
                continue
            if peer_src.rank != source.rank:
                continue
            active_peer, _, _ = is_source_active(peer_src, effective_as_of)
            if not active_peer:
                continue
            unindexed = _find_indexed_peer_clause_conflict(assertion, sec_body, clauses)
            if unindexed is not None:
                comp_val_str, comp_quote_str = unindexed
                _record_candidate(
                    canonicalize_value(comp_val_str, assertion.unit),
                    peer_src,
                    peer_src.id,
                    _effective_assertion_support(peer_src),
                    comp_val_str,
                    comp_quote_str,
                    peer_src.tier,
                )

        # Compute Noisy-OR per canonical value across independent content hashes
        group_support: dict[str, float] = {
            cval: noisy_or(entry[0] for entry in src_map.values())
            for cval, src_map in by_canon_source.items()
        }

        own_support = group_support.get(proposed_canon, 0.0)
        competing_groups = {k: v for k, v in group_support.items() if k != proposed_canon}
        total_support = sum(group_support.values())
        share_penalty = (own_support / total_support) if total_support > 0 else 0.0
        net_confidence = round(own_support * share_penalty, 6)

        if competing_groups:
            top_comp_canon = max(competing_groups.items(), key=lambda kv: (kv[1], kv[0]))[0]
            top_comp_support = competing_groups[top_comp_canon]
            comp_sources = by_canon_source[top_comp_canon]
            _, (_, comp_val, comp_quote, comp_tier, best_comp_src_id) = max(
                comp_sources.items(), key=lambda kv: kv[1][0]
            )

            # Self-contradiction inside the same source -> RETRY
            if best_comp_src_id == source.id:
                issues.append(
                    Issue(
                        code="INTERNAL_SELF_CONTRADICTION",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Proposal contains conflicting values ('{assertion.value}' vs '{comp_val}') "
                            f"for {assertion.subject}.{assertion.predicate} within source '{source.id}'."
                        ),
                        remediation="Reconcile internal contradiction before submitting proposal.",
                    )
                )
                return net_confidence, issues, conflicts

            if top_comp_support >= policy.conflict_escalation_threshold:
                diff = ConflictDiff(
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    proposed_value=assertion.value,
                    proposed_source_id=source.id,
                    proposed_quote=assertion.quote,
                    proposed_tier=source.tier,
                    competing_value=comp_val,
                    competing_source_id=best_comp_src_id,
                    competing_quote=comp_quote,
                    competing_tier=comp_tier,
                    proposed_support=own_support,
                    competing_support=top_comp_support,
                )
                conflicts.append(diff)
                issues.append(
                    Issue(
                        code="UNRESOLVED_PEER_CONFLICT",
                        disposition="ESCALATE",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Peer-authority conflict on {assertion.subject}.{assertion.predicate}: "
                            f"'{assertion.value}' ({source.id}) vs '{comp_val}' ({best_comp_src_id})."
                        ),
                        remediation=(
                            "Human reviewer must select governing value between equally ranked sources."
                        ),
                        conflict_diff=diff,
                    )
                )

    # 3. Numeric policy threshold gate (with dimension, predicate-stem, and quote-topic matching for C11b, C11v1, C11v2, C11fp)
    if policy.numeric_bounds:
        quantities = extract_quantities(assertion.value, explicit_unit=assertion.unit)
        for q in quantities:
            bounds = _match_numeric_bound(assertion, q.unit, policy)
            if bounds is None:
                continue
            min_bound, max_bound = bounds
            if min_bound is not None and q.magnitude < Decimal(str(min_bound)):
                issues.append(
                    Issue(
                        code="POLICY_THRESHOLD_EXCEEDED",
                        disposition="ESCALATE",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"{assertion.subject}.{assertion.predicate} = {q.magnitude} is below "
                            f"policy auto-approval floor ({min_bound:g})."
                        ),
                        remediation="Requires human sign-off for below-floor exception.",
                    )
                )
            if max_bound is not None and q.magnitude > Decimal(str(max_bound)):
                issues.append(
                    Issue(
                        code="POLICY_THRESHOLD_EXCEEDED",
                        disposition="ESCALATE",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"{assertion.subject}.{assertion.predicate} = {q.magnitude} exceeds "
                            f"policy auto-approval cap ({max_bound:g})."
                        ),
                        remediation="Requires human sign-off for above-cap commitment.",
                    )
                )

    # 4. Minimum net confidence gate (only if not already flagged by conflict)
    if not issues and net_confidence < policy.min_support:
        issues.append(
            Issue(
                code="LOW_EVIDENCE_SUPPORT",
                disposition="ESCALATE",
                assertion_id=assertion.id,
                subject=assertion.subject,
                predicate=assertion.predicate,
                message=(
                    f"Net evidence support ({net_confidence:.2f}) for {assertion.subject}.{assertion.predicate} "
                    f"is below policy min_support ({policy.min_support:.2f})."
                ),
                remediation="Provide corroborating independent source or escalate for human review.",
            )
        )

    return net_confidence, issues, conflicts
