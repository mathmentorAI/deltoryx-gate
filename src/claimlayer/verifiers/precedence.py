"""Deterministic authority hierarchy, supersession, and temporal validity checks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from claimlayer.models import (
    Assertion,
    Issue,
    Policy,
    Source,
    normalize_entity_key,
    normalize_key,
    normalize_predicate_key,
)
from claimlayer.verifiers.diff import _split_into_sections
from claimlayer.verifiers.quote_quantity import (
    _to_commensurable_base,
    canonicalize_value,
    extract_quantities,
    parse_iso_date,
)

if TYPE_CHECKING:
    from claimlayer.ledger import Ledger


def is_source_active(source: Source, as_of: str | None) -> tuple[bool, str, str]:
    """Check whether *source* is temporally valid at *as_of* (ISO YYYY-MM-DD).

    Returns ``(is_active, issue_code, reason)``.
    """
    if source.metadata.get("expired") is True:
        return False, "SOURCE_EXPIRED", f"Source '{source.id}' is explicitly marked expired."

    if not as_of:
        return True, "", ""

    iso_as_of = parse_iso_date(as_of)
    if iso_as_of is None:
        return (
            False,
            "SOURCE_EXPIRED",
            f"Invalid or non-ISO evaluation date as_of='{as_of}'; expected YYYY-MM-DD.",
        )

    if source.valid_from:
        iso_from = parse_iso_date(source.valid_from)
        if iso_from is None or iso_as_of < iso_from:
            return (
                False,
                "SOURCE_NOT_YET_EFFECTIVE",
                f"Source '{source.id}' is not yet effective as of {iso_as_of} (valid_from={source.valid_from}).",
            )
    if source.valid_until:
        iso_until = parse_iso_date(source.valid_until)
        if iso_until is None or iso_as_of > iso_until:
            return (
                False,
                "SOURCE_EXPIRED",
                f"Source '{source.id}' expired before {iso_as_of} (valid_until={source.valid_until}).",
            )
    return True, "", ""


def _find_governing_text_override(
    assertion: Assertion,
    higher_src: Source,
    pred_key: str,
) -> str | None:
    """Detect if *higher_src.text* directly overrides *assertion.value* on the same section or clause,
    even when FactRecords were not manually preloaded (C7b, C7c)."""
    val_quants = extract_quantities(assertion.value, explicit_unit=assertion.unit)
    if not val_quants:
        return None

    # 1. Check matching section in higher_src (e.g., predicate='section_2_04')
    sections = _split_into_sections(higher_src.text)
    target_text = ""
    if pred_key in sections:
        target_text = sections[pred_key].body
    else:
        # 2. Check sentence in higher_src mentioning both subject and predicate/quote tokens
        subj_key = normalize_entity_key(assertion.subject)
        subj_words = [w for w in subj_key.split("_") if len(w) >= 2]
        for sent in re.split(r"(?<=[.;\n])\s+", higher_src.text):
            sent_norm = normalize_key(sent)
            if subj_words and all(w in sent_norm for w in subj_words):
                target_text = sent
                break

    if not target_text:
        return None

    higher_quants = extract_quantities(target_text)
    if not higher_quants:
        return None

    for vq in val_quants:
        if any(vq.matches(hq) for hq in higher_quants):
            continue
        _, base_u = _to_commensurable_base(vq.magnitude, vq.unit)
        same_unit_hq = [
            hq for hq in higher_quants if _to_commensurable_base(hq.magnitude, hq.unit)[1] == base_u
        ]
        if same_unit_hq:
            gov = same_unit_hq[-1]
            return f"{gov.magnitude} {gov.unit}"
    return None


def verify_precedence_and_validity(
    assertion: Assertion,
    source: Source,
    ledger: Ledger,
    policy: Policy,
    as_of: str | None = None,
) -> list[Issue]:
    """Verify source authority rank, temporal validity, and supersession/override rules."""
    issues: list[Issue] = []
    effective_as_of = as_of or policy.as_of

    # 1. Temporal validity gate ALWAYS runs first (even if a prior human resolution existed, C8b)
    if policy.block_expired:
        active, code, reason = is_source_active(source, effective_as_of)
        if not active:
            issues.append(
                Issue(
                    code=code or "SOURCE_EXPIRED",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=reason,
                    remediation=f"Replace '{source.id}' with a currently active source document.",
                )
            )
            return issues

    # 2. Minimum authority tier gate
    if source.rank < policy.min_tier_rank:
        issues.append(
            Issue(
                code="INSUFFICIENT_AUTHORITY_TIER",
                disposition="RETRY",
                assertion_id=assertion.id,
                subject=assertion.subject,
                predicate=assertion.predicate,
                message=(
                    f"Source '{source.id}' has tier '{source.tier}' (rank={source.rank}), "
                    f"below required min_tier_rank={policy.min_tier_rank}."
                ),
                remediation=(
                    f"Ground '{assertion.predicate}' in a source with authority rank >= {policy.min_tier_rank}."
                ),
            )
        )
        return issues

    if not policy.block_superseded:
        return issues

    subj_key = normalize_entity_key(assertion.subject)
    pred_key = normalize_predicate_key(assertion.predicate, policy.predicate_aliases)
    raw_pred_key = normalize_key(assertion.predicate)
    proposed_canon = canonicalize_value(assertion.value, assertion.unit)

    # 3. Explicit document supersession check (Source.supersedes)
    # Runs BEFORE human resolution short-circuit so post-resolution amendments supersede old resolutions (C8)
    for candidate_src in ledger.list_sources():
        if candidate_src.id == source.id:
            continue
        active_cand, _, _ = is_source_active(candidate_src, effective_as_of)
        if not active_cand:
            continue
        if source.id in candidate_src.supersedes:
            cand_facts = ledger.get_facts_by_source(candidate_src.id)
            if not cand_facts:
                issues.append(
                    Issue(
                        code="SOURCE_SUPERSEDED",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=f"Source '{source.id}' has been superseded by '{candidate_src.id}'.",
                        remediation=f"Re-verify '{assertion.predicate}' against superseding source '{candidate_src.id}'.",
                    )
                )
                return issues

            # Partial amendment: match using canonical entity & predicate keys (catching C1b, C1c)
            for cf in cand_facts:
                cf_subj = normalize_entity_key(cf.subject)
                cf_pred = normalize_predicate_key(cf.predicate, policy.predicate_aliases)
                cf_raw_pred = normalize_key(cf.predicate)
                pred_matches = (cf_pred == pred_key) or (cf_raw_pred == raw_pred_key)
                subj_matches = (cf_subj == subj_key) or (
                    pred_matches and len(cand_facts) == 1
                )
                if (
                    subj_matches
                    and pred_matches
                    and canonicalize_value(cf.value, cf.unit) != proposed_canon
                ):
                    issues.append(
                        Issue(
                            code="SOURCE_SUPERSEDED",
                            disposition="RETRY",
                            assertion_id=assertion.id,
                            subject=assertion.subject,
                            predicate=assertion.predicate,
                            message=(
                                f"Clause '{assertion.predicate}' for '{assertion.subject}' in '{source.id}' "
                                f"was amended by '{candidate_src.id}' to '{cf.value}'."
                            ),
                            remediation=(
                                f"Update '{assertion.predicate}' citing superseding source '{candidate_src.id}'."
                            ),
                        )
                    )
                    return issues

    # If a human already resolved this (subject, predicate) and no newer amendment superseded it, defer to conflict.py
    if ledger.get_resolution(assertion.subject, assertion.predicate) is not None:
        return issues

    # 4. Higher authority tier override check across active ledger facts (catching C7c)
    matching_facts = ledger.get_facts(
        assertion.subject,
        assertion.predicate,
        aliases=policy.predicate_aliases,
        allow_section_predicate_match=True,
    )
    for fact in matching_facts:
        fact_src = ledger.get_source(fact.source_id)
        if fact_src is None or fact_src.id == source.id:
            continue
        active_fact_src, _, _ = is_source_active(fact_src, effective_as_of)
        if not active_fact_src:
            continue
        if fact_src.rank > source.rank:
            fact_canon = canonicalize_value(fact.value, fact.unit)
            if fact_canon != proposed_canon:
                issues.append(
                    Issue(
                        code="HIGHER_TIER_OVERRIDE",
                        disposition="RETRY",
                        assertion_id=assertion.id,
                        subject=assertion.subject,
                        predicate=assertion.predicate,
                        message=(
                            f"Higher-authority source '{fact_src.id}' (tier='{fact_src.tier}', rank={fact_src.rank}) "
                            f"overrides '{source.id}' (tier='{source.tier}', rank={source.rank}) on "
                            f"{assertion.subject}.{assertion.predicate}."
                        ),
                        remediation=(
                            f"Ground '{assertion.predicate}' in higher-authority governing source '{fact_src.id}'."
                        ),
                    )
                )
                return issues

    # 5. Higher authority tier override directly from unindexed Source text (C7b)
    for candidate_src in ledger.list_sources():
        if candidate_src.id == source.id or candidate_src.rank <= source.rank:
            continue
        active_cand, _, _ = is_source_active(candidate_src, effective_as_of)
        if not active_cand:
            continue
        override_desc = _find_governing_text_override(assertion, candidate_src, raw_pred_key)
        if override_desc is not None:
            issues.append(
                Issue(
                    code="HIGHER_TIER_OVERRIDE",
                    disposition="RETRY",
                    assertion_id=assertion.id,
                    subject=assertion.subject,
                    predicate=assertion.predicate,
                    message=(
                        f"Higher-authority source '{candidate_src.id}' (tier='{candidate_src.tier}', rank={candidate_src.rank}) "
                        f"overrides '{source.id}' (tier='{source.tier}', rank={source.rank}) on '{assertion.predicate}'."
                    ),
                    remediation=(
                        f"Ground '{assertion.predicate}' in higher-authority governing source '{candidate_src.id}'."
                    ),
                )
            )
            return issues

    return issues
