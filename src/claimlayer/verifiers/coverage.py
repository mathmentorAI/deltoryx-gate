"""Deterministic completeness and mandatory checklist coverage checks."""

from __future__ import annotations

from claimlayer.models import Assertion, Issue, Policy, normalize_key, normalize_predicate_key


def verify_coverage(assertions: tuple[Assertion, ...], policy: Policy) -> list[Issue]:
    """Verify that the agent addressed all required predicates and mandatory sources."""
    issues: list[Issue] = []

    if not assertions:
        issues.append(
            Issue(
                code="EMPTY_PROPOSAL",
                disposition="RETRY",
                message="Agent submitted zero assertions for verification.",
                remediation="Include at least one structured Assertion backed by a source quote.",
            )
        )
        return issues

    present_raw = {normalize_key(a.predicate) for a in assertions if a.value and a.value.strip()}
    present_canon = {
        normalize_predicate_key(a.predicate, policy.predicate_aliases)
        for a in assertions
        if a.value and a.value.strip()
    }
    for req_pred in policy.required_predicates:
        req_raw = normalize_key(req_pred)
        req_canon = normalize_predicate_key(req_pred, policy.predicate_aliases)
        if req_raw not in present_raw and req_canon not in present_canon:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_PREDICATE",
                    disposition="RETRY",
                    predicate=req_pred,
                    message=f"Required predicate '{req_pred}' was omitted from the agent proposal.",
                    remediation=f"Extract and cite '{req_pred}' from the governing source documents.",
                )
            )

    # The operator's comparison policy records changed values, not just section labels.
    from claimlayer.verifiers.diff import _quantity_tokens
    for predicate, required in policy.required_value_tokens.items():
        provided = set()
        for assertion in assertions:
            if normalize_predicate_key(assertion.predicate, policy.predicate_aliases) == normalize_predicate_key(predicate, policy.predicate_aliases):
                provided.update(_quantity_tokens(assertion.value))
        missing = set(required) - provided
        if missing:
            issues.append(Issue(code="MISSING_REQUIRED_VALUE", disposition="RETRY", predicate=predicate,
                                message=f"Comparison omits governing values: {sorted(missing)}",
                                remediation="Report the changed governing values, not only the section topic."))

    cited_sources = {a.source_id for a in assertions}
    for req_src in policy.required_source_ids:
        if req_src not in cited_sources:
            issues.append(
                Issue(
                    code="MISSING_REQUIRED_SOURCE",
                    disposition="RETRY",
                    message=f"Mandatory source '{req_src}' was not addressed or cited in the proposal.",
                    remediation=f"Inspect mandatory source '{req_src}' and include relevant assertions.",
                )
            )

    return issues
