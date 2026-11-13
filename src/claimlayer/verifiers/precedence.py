"""Deterministic authority hierarchy, supersession, and temporal validity checks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from claimlayer.models import (
    Assertion,
    Issue,
    Policy,
    Source,
    _singularize_token,
    normalize_entity_key,
    normalize_key,
    normalize_predicate_key,
)
from claimlayer.verifiers.diff import _split_into_sections
from claimlayer.verifiers.quote_quantity import (
    _to_commensurable_base,
    canonicalize_value,
    extract_quantities,
    normalize_span,
    parse_iso_date,
)

if TYPE_CHECKING:
    from claimlayer.ledger import Ledger

_GENERIC_SUBJECT_KEYS = frozenset(
    {
        "deal",
        "waterfall",
        "deal_waterfall",
        "notes",
        "note",
        "agreement",
        "contract",
        "indenture",
        "issuer",
        "borrower",
        "servicer",
        "party",
        "customer",
        "loan",
        "facility",
        "x",
        "s",
        "o",
    }
)

_DESIGNATOR_RE = re.compile(
    r"\b((?:class|tranche|series|tier)\s+[a-z0-9]+)\b",
    re.IGNORECASE,
)

_CLAUSE_TOPIC_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "shall",
        "will",
        "may",
        "must",
        "are",
        "was",
        "were",
        "been",
        "being",
        "from",
        "that",
        "this",
        "these",
        "those",
        "each",
        "all",
        "any",
        "within",
        "upon",
        "per",
        "annum",
        "under",
        "into",
        "such",
        "days",
        "day",
        "business",
        "calendar",
        "working",
        "weeks",
        "week",
        "months",
        "month",
        "years",
        "year",
        "hours",
        "hour",
        "percent",
        "pct",
        "bps",
        "basis",
        "points",
        "usd",
        "eur",
        "gbp",
        "dollars",
        "dollar",
        "euros",
        "euro",
        "million",
        "millions",
        "billion",
        "billions",
        "thousand",
        "thousands",
        "step",
        "steps",
        "section",
        "article",
        "clause",
        "clausula",
        "schedule",
        "exhibit",
        "annex",
        "notes",
        "note",
        "class",
        "tranche",
        "series",
        "tier",
        "deal",
        "company",
        "corp",
        "inc",
        "llc",
        "ltd",
        "agreement",
        "indenture",
        "contract",
        "party",
        "issuer",
        "borrower",
        "servicer",
        "senior",
        "junior",
        "ahead",
        "new",
        "york",
        "delaware",
        "california",
        "amended",
    }
)


def _extract_designators(text: str) -> set[str]:
    cleaned = re.sub(r"[_\-]+", " ", (text or "").lower())
    return {re.sub(r"\s+", " ", m.group(1).strip()) for m in _DESIGNATOR_RE.finditer(cleaned)}


def _extract_clause_topic_tokens(text: str, subject: str = "") -> set[str]:
    subj_tokens = set(normalize_entity_key(subject).split("_")) if subject else set()
    words = re.findall(r"[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,}", normalize_span(text).lower())
    out: set[str] = set()
    for w in words:
        if w in _CLAUSE_TOPIC_STOPWORDS or w in subj_tokens:
            continue
        stem = _singularize_token(w)
        if stem not in _CLAUSE_TOPIC_STOPWORDS and stem not in subj_tokens:
            out.add(stem)
    return out


def _subjects_and_designators_compatible(
    assertion: Assertion,
    cand_subj: str | None,
    cand_text: str,
) -> bool:
    """Return True if *assertion* and a candidate fact/clause refer to compatible entities and note classes."""
    a_des = _extract_designators(f"{assertion.subject} {assertion.quote}")
    c_des = _extract_designators(f"{cand_subj or ''} {cand_text}")
    if a_des and c_des and not (a_des & c_des):
        return False

    a_sk = normalize_entity_key(assertion.subject)
    c_sk = normalize_entity_key(cand_subj) if cand_subj else ""
    if a_sk and c_sk and a_sk != c_sk:
        if a_sk not in _GENERIC_SUBJECT_KEYS and c_sk not in _GENERIC_SUBJECT_KEYS:
            return False

    if a_sk and a_sk not in _GENERIC_SUBJECT_KEYS:
        quote_words = set(normalize_key(assertion.quote).split("_"))
        cand_words = set(normalize_key(cand_text).split("_"))
        # If the assertion quote explicitly names the specific party (e.g. "Beta Inc"), the candidate clause must too (C1fp)
        if a_sk in quote_words and a_sk not in cand_words and c_sk != a_sk:
            return False

    return True


def _clauses_address_same_topic(
    assertion: Assertion,
    cand_text: str,
    cand_pred: str | None = None,
) -> bool:
    """Return True if *assertion* (quote/predicate) and *cand_text* address the same substantive provision."""
    a_quote_topics = _extract_clause_topic_tokens(assertion.quote, assertion.subject)
    a_pred_topics = _extract_clause_topic_tokens(assertion.predicate.replace("_", " "), assertion.subject)
    a_topics = a_quote_topics | a_pred_topics

    c_quote_topics = _extract_clause_topic_tokens(cand_text, assertion.subject)
    c_pred_topics = _extract_clause_topic_tokens((cand_pred or "").replace("_", " "), assertion.subject)
    c_topics = c_quote_topics | c_pred_topics

    shared = a_topics & c_topics
    if not shared:
        return False
    if len(a_quote_topics) <= 2 or len(c_quote_topics) <= 2:
        return len(shared) >= 1
    return len(shared) >= 2


def _extract_conflicting_clause_value(
    assertion: Assertion,
    cand_text: str,
) -> str | None:
    """If *cand_text* states a conflicting quantity or textual value for *assertion*, return the competing value string."""
    val_quants = extract_quantities(assertion.value, explicit_unit=assertion.unit)
    if val_quants:
        cand_quants = extract_quantities(cand_text)
        if not cand_quants:
            return None
        for vq in val_quants:
            if any(vq.matches(cq) for cq in cand_quants):
                continue
            _, base_u = _to_commensurable_base(vq.magnitude, vq.unit)
            same_unit_cq = [
                cq
                for cq in cand_quants
                if _to_commensurable_base(cq.magnitude, cq.unit)[1] == base_u
            ]
            if same_unit_cq:
                gov = same_unit_cq[-1]
                unit_suffix = "%" if gov.unit == "PERCENT" else f" {gov.unit}"
                return f"{gov.magnitude}{unit_suffix}"
        return None

    prop_canon = canonicalize_value(assertion.value, assertion.unit)
    if prop_canon.startswith("STR:"):
        val_core = prop_canon[4:].strip()
        cand_norm = normalize_span(cand_text).lower()
        if val_core and val_core not in cand_norm:
            return normalize_span(cand_text).rstrip(".")
    return None


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


def _find_governing_clause_override(
    assertion: Assertion,
    sec_body: str | None,
    clauses: list[str],
) -> str | None:
    """Detect if a higher-authority source's pre-indexed section or clauses override *assertion.value* (C7b, C7c, C7v1, C7fp)."""
    if sec_body is not None and _subjects_and_designators_compatible(assertion, None, sec_body):
        conflict_val = _extract_conflicting_clause_value(assertion, sec_body)
        if conflict_val is not None:
            return conflict_val

    for sent_clean in clauses:
        if not _subjects_and_designators_compatible(assertion, None, sent_clean):
            continue
        if not _clauses_address_same_topic(assertion, sent_clean):
            continue
        conflict_val = _extract_conflicting_clause_value(assertion, sent_clean)
        if conflict_val is not None:
            return conflict_val
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

            # Partial amendment: match using canonical entity & predicate keys or clause topic (catching C1b, C1c, C1v1-v3, while avoiding C1fp)
            for cf in cand_facts:
                cf_pred = normalize_predicate_key(cf.predicate, policy.predicate_aliases)
                cf_raw_pred = normalize_key(cf.predicate)
                cf_text = cf.quote or candidate_src.text
                pred_matches = (
                    (cf_pred == pred_key)
                    or (cf_raw_pred == raw_pred_key)
                    or _clauses_address_same_topic(assertion, cf_text, cf.predicate)
                )
                subj_matches = _subjects_and_designators_compatible(assertion, cf.subject, cf_text)
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

    # 5. Higher authority tier override from Ledger's pre-computed clause index (C7b, C7v1)
    for candidate_src, sec_body, clauses in ledger.get_indexed_source_clauses(
        assertion, raw_pred_key, exclude_source_id=source.id
    ):
        if candidate_src.rank <= source.rank:
            continue
        active_cand, _, _ = is_source_active(candidate_src, effective_as_of)
        if not active_cand:
            continue
        override_desc = _find_governing_clause_override(assertion, sec_body, clauses)
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
