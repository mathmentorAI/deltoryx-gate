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
from claimlayer.verifiers.precedence import is_source_active
from claimlayer.verifiers.quote_quantity import (
    _to_commensurable_base,
    canonicalize_value,
    extract_quantities,
    normalize_span,
)

if TYPE_CHECKING:
    from claimlayer.ledger import Ledger

_CONDITIONAL_CLAUSE_RE = re.compile(
    r"\b(?:if|when|where|upon|in\s+the\s+case\s+of|in\s+case\s+of|with\s+respect\s+to|"
    r"for\s+interest|for\s+principal|for\s+covenant|for\s+payment|for\s+non-payment|"
    r"prior\s+to|after|during|si\b|cuando\b|en\s+caso\s+de|respecto\s+a|para\b)\b",
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


def _effective_support(assertion_conf: float, source: Source | None) -> float:
    """Cap agent-declared confidence by the source's operator-configured reliability."""
    base = float(assertion_conf)
    if source is not None and "reliability" in source.metadata:
        src_rel = float(source.metadata["reliability"])
        return max(0.0, min(base, src_rel))
    return max(0.0, min(1.0, base))


def _are_distinct_conditional_clauses(a1: Assertion, a2: Assertion) -> bool:
    """Return True if two assertions on the same source describe distinct conditional branches (B11)."""
    if a1.condition and a2.condition and normalize_key(a1.condition) != normalize_key(a2.condition):
        return True
    q1 = normalize_span(a1.quote)
    q2 = normalize_span(a2.quote)
    if q1.lower() == q2.lower():
        return False
    if _CONDITIONAL_CLAUSE_RE.search(q1) or _CONDITIONAL_CLAUSE_RE.search(q2):
        return True
    return False


def _match_numeric_bound(
    predicate: str,
    policy: Policy,
) -> tuple[float | None, float | None] | None:
    """Match a predicate against policy.numeric_bounds, catching predicate renaming/aliasing (C11b)."""
    if not policy.numeric_bounds:
        return None
    norm_pred = normalize_predicate_key(predicate, policy.predicate_aliases)
    raw_pred = normalize_key(predicate)

    for k, bounds in policy.numeric_bounds.items():
        if k == predicate or normalize_key(k) == raw_pred or normalize_predicate_key(k, policy.predicate_aliases) == norm_pred:
            return bounds

    # Stem matching: strip generic measure suffixes ('refund_total' <-> 'refund_amount')
    pred_stems = [t for t in norm_pred.split("_") if t and t not in _GENERIC_BOUND_SUFFIXES]
    if not pred_stems:
        return None

    for k, bounds in policy.numeric_bounds.items():
        k_norm = normalize_predicate_key(k, policy.predicate_aliases)
        k_stems = [t for t in k_norm.split("_") if t and t not in _GENERIC_BOUND_SUFFIXES]
        if k_stems and pred_stems == k_stems:
            return bounds

    return None


def _scan_unindexed_peer_source_for_conflict(
    assertion: Assertion,
    peer_src: Source,
    raw_pred_key: str,
) -> tuple[str, str] | None:
    """Scan *peer_src.text* for an unindexed sentence/section contradicting *assertion* (C2)."""
    val_quants = extract_quantities(assertion.value, explicit_unit=assertion.unit)
    if not val_quants:
        return None

    sections = _split_into_sections(peer_src.text)
    candidate_sentences: list[str] = []
    if raw_pred_key in sections:
        candidate_sentences.append(sections[raw_pred_key].body)
    else:
        subj_words = [
            w for w in normalize_entity_key(assertion.subject).split("_") if len(w) >= 2
        ]
        pred_words = [
            w
            for w in normalize_predicate_key(assertion.predicate).split("_")
            if len(w) >= 3 and w not in _GENERIC_BOUND_SUFFIXES
        ]
        for sent in re.split(r"(?<=[.;\n])\s+", peer_src.text):
            sent_norm = normalize_key(sent)
            if subj_words and all(w in sent_norm for w in subj_words):
                if not pred_words or any(pw in sent_norm for pw in pred_words):
                    candidate_sentences.append(sent.strip())

    for sent in candidate_sentences:
        peer_quants = extract_quantities(sent)
        for vq in val_quants:
            if any(vq.matches(pq) for pq in peer_quants):
                continue
            _, base_u = _to_commensurable_base(vq.magnitude, vq.unit)
            same_unit_pq = [
                pq
                for pq in peer_quants
                if _to_commensurable_base(pq.magnitude, pq.unit)[1] == base_u
            ]
            if same_unit_pq:
                comp_q = same_unit_pq[0]
                unit_suffix = "%" if comp_q.unit == "PERCENT" else f" {comp_q.unit}"
                return f"{comp_q.magnitude}{unit_suffix}", sent[:160]

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
            dedup_key = src_obj.sha256 if src_obj is not None else f"id:{src_id}"
            bucket = by_canon_source.setdefault(canon_val, {})
            prev = bucket.get(dedup_key)
            if prev is None or conf > prev[0]:
                bucket[dedup_key] = (conf, raw_val, quote_txt, tier_name, src_id)

        own_eff_conf = _effective_support(assertion.confidence, source)
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
                        _effective_support(other.confidence, other_src),
                        other.value,
                        other.quote,
                        other_tier,
                    )

        # Check active peer-rank facts in ledger (excluding superseded sources)
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
                _record_candidate(
                    canonicalize_value(fact.value, fact.unit),
                    fact_src,
                    fact.source_id,
                    _effective_support(fact.confidence, fact_src),
                    fact.value,
                    fact.quote or fact_src.text[:160],
                    fact_src.tier,
                )

        # Also scan active peer-rank sources in ledger for unindexed conflicts (closing C2 omission bypass!)
        for peer_src in ledger.list_sources():
            if peer_src.id == source.id or peer_src.id in superseded_ids:
                continue
            if peer_src.rank != source.rank:
                continue
            active_peer, _, _ = is_source_active(peer_src, effective_as_of)
            if not active_peer:
                continue
            unindexed = _scan_unindexed_peer_source_for_conflict(
                assertion, peer_src, raw_pred_key
            )
            if unindexed is not None:
                comp_val_str, comp_quote_str = unindexed
                _record_candidate(
                    canonicalize_value(comp_val_str, assertion.unit),
                    peer_src,
                    peer_src.id,
                    _effective_support(1.0, peer_src),
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

    # 3. Numeric policy threshold gate (with predicate alias/stem matching for C11b)
    bounds = _match_numeric_bound(assertion.predicate, policy)
    if bounds is not None:
        min_bound, max_bound = bounds
        quantities = extract_quantities(assertion.value, explicit_unit=assertion.unit)
        for q in quantities:
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
