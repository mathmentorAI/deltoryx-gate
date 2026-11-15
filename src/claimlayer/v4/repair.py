"""ClaimLayer v4 bounded evaluation, structured repair feedback, and reverification loop.

Separates exploratory explanatory evaluation (``ReviewResult.disposition`` in
``{"ACCEPT", "RETRY", "ESCALATE", "ABSTAIN"}``) from operational execution
authorization (``authorize(...)`` returning ``PASS / RETRY / ESCALATE``), so
an appropriately hedged narrative explanation can be accepted for informational
use while still remaining ineligible for autonomous side-effect execution.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..semantic.adapters import OpenAICompatibleModel
from ..semantic.models import SourceDocument, canonical, digest
from .authorization import AuthorizationVerdict, authorize
from .corpus_obligations import CorpusReferenceRegister
from .correspondence import compute_response_digest
from .domain_profile import DomainProfile, PolicyConfiguration
from .engine import EvidenceLayer
from .evaluation import (
    ResponseAssessment,
    VerificationBudget,
    VerificationCache,
    evaluate_response,
)
from .justification import JustificationGraph


# ---------------------------------------------------------------------------
# Structured review diagnostics and actionable repair instructions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewIssue:
    """Single deterministic or epistemic defect found during response review."""

    code: str
    severity: str  # "error" | "warning" | "info"
    family: str
    obligation_id: str
    step_id: str
    message: str
    target_claim_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "family": self.family,
            "obligation_id": self.obligation_id,
            "step_id": self.step_id,
            "message": self.message,
            "target_claim_id": self.target_claim_id,
            "evidence": canonical(dict(self.evidence)),
        }


@dataclass(frozen=True)
class RequiredRepair:
    """Actionable, machine- and LLM-consumable instruction to repair an issue."""

    repair_id: str
    obligation_id: str
    family: str
    target_claim_id: str
    issue_code: str
    problem: str
    instruction: str
    allowed_repairs: tuple[str, ...]
    forbidden_shortcuts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repair_id": self.repair_id,
            "obligation_id": self.obligation_id,
            "family": self.family,
            "target_claim_id": self.target_claim_id,
            "issue_code": self.issue_code,
            "problem": self.problem,
            "instruction": self.instruction,
            "allowed_repairs": list(self.allowed_repairs),
            "forbidden_shortcuts": list(self.forbidden_shortcuts),
        }


@dataclass(frozen=True)
class ReviewResult:
    """Graded response review bound by digest to exact question + response text."""

    disposition: str  # "ACCEPT" | "RETRY" | "ESCALATE" | "ABSTAIN"
    question: str | None
    candidate_response: str | None
    response_digest: str | None
    assessment_digest: str
    representation_mode: str
    verified_elements: tuple[dict[str, Any], ...]
    issues: tuple[ReviewIssue, ...]
    required_repairs: tuple[RequiredRepair, ...]
    unresolved_obligations: tuple[dict[str, Any], ...]
    assessment: ResponseAssessment
    operational_authorization: AuthorizationVerdict | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "question": self.question,
            "candidate_response": self.candidate_response,
            "response_digest": self.response_digest,
            "assessment_digest": self.assessment_digest,
            "representation_mode": self.representation_mode,
            "verified_elements": list(self.verified_elements),
            "issues": [i.to_dict() for i in self.issues],
            "required_repairs": [r.to_dict() for r in self.required_repairs],
            "unresolved_obligations": list(self.unresolved_obligations),
            "assessment": self.assessment.to_summary_dict(),
            "operational_authorization": (
                self.operational_authorization.to_dict()
                if self.operational_authorization is not None
                else None
            ),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Parameterized repair guidance table (no entity- or benchmark-specific rules)
# ---------------------------------------------------------------------------

_CALC_ALTERED_SPEC: dict[str, Any] = {
    "repairable": True,
    "problem": "Numeric value in response diverges from the verified derived value/interval.",
    "instruction": "Restore the mathematically derived value or interval from the evidence.",
    "allowed": ("restore_verified_value_or_interval",),
    "forbidden": ("alter_verified_calculation", "adding_approximation_to_wrong_number"),
}

_UNIT_CHANGED_SPEC: dict[str, Any] = {
    "repairable": True,
    "problem": "Unit expressed in response conflicts with the unit derived in the graph.",
    "instruction": "Use the verified unit (or an explicitly equivalent converted unit matching the value).",
    "allowed": ("restore_verified_unit",),
    "forbidden": ("swap_units_without_conversion",),
}

_QUESTION_EVADED_SPEC: dict[str, Any] = {
    "repairable": True,
    "problem": "Response fails to address the question being asked.",
    "instruction": (
        "Answer the question directly using only the verified derivation/bounds, preserving required "
        "hedging and attribution."
    ),
    "allowed": ("answer_question_with_verified_conclusion_and_qualifiers",),
    "forbidden": ("evade_question_with_unrelated_text",),
}

_REPAIR_SPEC_TABLE: dict[str, dict[str, Any]] = {
    "UNJUSTIFIED_TEMPORAL_PRECISION": {
        "repairable": True,
        "problem": "Exact age/duration asserted without complete birth/start and reference dates.",
        "instruction": (
            "Qualify the temporal figure as approximate or state the derived bounds, and "
            "explicitly note that incomplete dates prevent determining an exact value."
        ),
        "allowed": (
            "express_as_approximate_with_uncertainty_note",
            "state_derived_interval_bounds",
            "state_missing_full_dates_limitation",
        ),
        "forbidden": (
            "inventing_missing_dates",
            "retaining_unqualified_exact_claim",
            "contradictory_hedging_plus_exact_claim",
        ),
    },
    "TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE": {
        "repairable": True,
        "problem": "Temporal evidence lacks full date granularity to certify an exact point figure.",
        "instruction": (
            "Qualify the temporal figure as approximate or state the derived bounds, and "
            "explicitly state the missing full-date limitation."
        ),
        "allowed": (
            "express_as_approximate_with_uncertainty_note",
            "state_derived_interval_bounds",
            "state_missing_full_dates_limitation",
        ),
        "forbidden": (
            "inventing_missing_dates",
            "retaining_unqualified_exact_claim",
            "contradictory_hedging_plus_exact_claim",
        ),
    },
    "UNQUALIFIED_EXACT_VALUE_FOR_APPROX_CLAIM": {
        "repairable": True,
        "problem": "Natural-language response states an unqualified exact figure for an approximate or interval conclusion.",
        "instruction": (
            "Add an explicit approximation or interval qualifier matching the verified uncertainty bounds, "
            "without adding any contradictory exactness assertion."
        ),
        "allowed": ("qualify_with_approximate_marker", "state_verified_interval"),
        "forbidden": ("unqualified_exact_assertion", "contradictory_hedging_plus_exact_claim"),
    },
    "APPROXIMATION_HIDES_WIDE_UNCERTAINTY": {
        "repairable": True,
        "problem": "Simple point approximation hides an uncertainty interval wider than allowed tolerance.",
        "instruction": (
            "State the explicit minimum-to-maximum range supported by the evidence rather than "
            "a single hedged point number."
        ),
        "allowed": ("state_explicit_min_max_range", "abstain_from_point_estimate"),
        "forbidden": ("single_point_approximation_on_wide_span", "inventing_missing_dates"),
    },
    "TEMPORAL_DERIVATION_CONTRADICTED": {
        "repairable": True,
        "problem": "Stated temporal value contradicts the bounds mathematically derived from evidence anchors.",
        "instruction": (
            "Replace the inconsistent value with the calculation/interval derived from the evidence anchors "
            "and qualify precision according to date granularity."
        ),
        "allowed": ("recompute_from_evidence_anchors", "state_verified_interval", "abstain"),
        "forbidden": ("adding_approximation_to_wrong_number", "inventing_missing_dates"),
    },
    "CONTRADICTORY_PRECISION_QUALIFICATION": {
        "repairable": True,
        "problem": "Response simultaneously hedges an answer as approximate and asserts exact certainty.",
        "instruction": (
            "Remove contradictory exactness claims (e.g. 'exactamente', 'con certeza', 'sin duda') "
            "and keep consistent approximation and missing-date disclosure."
        ),
        "allowed": ("consistent_approximate_or_ranged_wording",),
        "forbidden": ("contradictory_hedging_plus_exact_claim",),
    },
    "SOURCE_CONDITIONAL_CONCLUSION": {
        "repairable": True,
        "problem": "Response presents a narrative document claim as unattributed external fact.",
        "instruction": (
            "Explicitly attribute the conclusion to the provided text/source (e.g. 'Según el texto...') "
            "and avoid claiming independent historical/external verification."
        ),
        "allowed": ("add_source_conditional_attribution", "note_unverified_external_truth"),
        "forbidden": ("assert_external_fact_without_corroboration",),
    },
    "CONTRADICTORY_SOURCE_ATTRIBUTION": {
        "repairable": True,
        "problem": "Response attributes to source text while also claiming independent factual verification.",
        "instruction": (
            "Remove claims of independent historical/external verification and retain strict "
            "source-conditional attribution."
        ),
        "allowed": ("restrict_to_source_conditional_attribution",),
        "forbidden": ("claim_independent_verification_of_narrative_text",),
    },
    "DROPPED_REQUIRED_CONDITION": {
        "repairable": True,
        "problem": "Inference graph omits mandatory qualifying conditions present on supporting premises.",
        "instruction": (
            "Propagate all required conditions and exceptions into the conclusion and explicitly state them."
        ),
        "allowed": ("state_required_conditions_and_exceptions",),
        "forbidden": ("drop_qualifying_condition",),
    },
    "CONDITION_DROPPED_IN_RESPONSE": {
        "repairable": True,
        "problem": "Response states an unconditional conclusion while evidence carries mandatory conditions/exceptions.",
        "instruction": (
            "Explicitly include the qualifying condition or exception (e.g. 'salvo...', 'sujeto a...') "
            "carried by the evidence."
        ),
        "allowed": ("state_required_conditions_and_exceptions",),
        "forbidden": ("unconditional_statement_ignoring_exception",),
    },
    "SUBTOTAL_PRESENTED_AS_TOTAL": {
        "repairable": True,
        "problem": "Partial aggregation is presented as a complete total without verified population completeness.",
        "instruction": (
            "Qualify the figure explicitly as a partial subtotal of the referenced items and note that "
            "completeness of the full population has not been verified."
        ),
        "allowed": ("reframe_as_partial_subtotal", "state_population_incompleteness"),
        "forbidden": ("claim_complete_total_without_population_verification",),
    },
    "UNQUALIFIED_SUBTOTAL_CLAIM": {
        "repairable": True,
        "problem": "Response describes a partial aggregation as an unqualified total.",
        "instruction": (
            "Replace unqualified 'total' wording with explicit partial/subtotal framing specifying "
            "only the listed items."
        ),
        "allowed": ("reframe_as_partial_subtotal",),
        "forbidden": ("claim_complete_total_without_population_verification",),
    },
    "UNVERIFIED_POPULATION_COMPLETENESS": {
        "repairable": True,
        "problem": "Aggregation lacks verified population completeness over expected items.",
        "instruction": (
            "Qualify the aggregate as a partial subtotal of the observed items and note missing population verification."
        ),
        "allowed": ("reframe_as_partial_subtotal", "supply_verified_population_ids"),
        "forbidden": ("claim_complete_total_without_population_verification",),
    },
    "UNDISCLOSED_MATERIAL_CONFLICT": {
        "repairable": True,
        "problem": "Unresolved material conflict exists across supporting and contradicting evidence.",
        "instruction": (
            "Explicitly disclose the conflicting figures/claims from the evidence and do not present "
            "a single resolved conclusion."
        ),
        "allowed": ("disclose_material_conflict_both_sides", "abstain_from_single_value"),
        "forbidden": ("suppress_contradicting_evidence", "declare_resolved_total"),
    },
    "CONFLICT_HIDDEN_IN_RESPONSE": {
        "repairable": True,
        "problem": "Response presents one side while suppressing contradicting evidence in the record.",
        "instruction": (
            "Explicitly mention the contradiction/discrepancy across sources and avoid presenting one "
            "value as settled."
        ),
        "allowed": ("disclose_material_conflict_both_sides",),
        "forbidden": ("suppress_contradicting_evidence",),
    },
    "REGISTER_STATUS_CONTRADICTED": {
        "repairable": True,
        "problem": "Response denies material discrepancies while the corpus reference register records unresolved conflicts or overrides.",
        "instruction": (
            "Remove false 'no material discrepancy' assurances and disclose all corpus conflicts and governing sources."
        ),
        "allowed": ("disclose_material_conflict_both_sides", "align_with_corpus_register_status"),
        "forbidden": ("deny_recorded_corpus_discrepancy",),
    },
    "OPINION_CARRIES_FACT": {
        "repairable": True,
        "problem": "Opinion-marked clause asserts concrete factual values belonging to a Vocabulary predicate.",
        "instruction": (
            "Move concrete factual figures into grounded extraction/derivation leaves or confine the opinion "
            "sentence strictly to interpretive assessment."
        ),
        "allowed": ("confine_opinion_to_interpretive_judgment", "promote_fact_to_extraction_leaf"),
        "forbidden": ("smuggle_predicate_fact_into_opinion",),
    },
    "GOVERNING_VALUE_MISMATCH": {
        "repairable": True,
        "problem": "Response or opinion clause asserts a subordinate conflicting value instead of the governing value.",
        "instruction": (
            "State both values accurately and identify the higher-ranked governing source value rather than "
            "asserting the subordinate value as applicable."
        ),
        "allowed": ("state_governing_value_and_hierarchy",),
        "forbidden": ("assert_subordinate_value_as_governing",),
    },
    "OPINION_PREMISE_UNRELATED": {
        "repairable": True,
        "problem": "An opinion step cites premise claim(s) whose subject or lineage is unrelated to the opinion claim.",
        "instruction": (
            "Bind every opinion claim only to verified premises whose subject and evidence lineage directly support "
            "the opinion's subject."
        ),
        "allowed": ("attach_subject_aligned_verified_premises",),
        "forbidden": ("attach_unrelated_premise_to_opinion",),
    },
    "AMBIGUOUS_OR_GENERIC_SEMANTIC_TYPE": {
        "repairable": True,
        "problem": "Claim uses an untyped generic predicate label without a declared Vocabulary FieldSpec.",
        "instruction": (
            "Declare an explicit typed predicate / FieldSpec in the domain Vocabulary or clarify the "
            "exact semantic type and unit explicitly in the response and graph."
        ),
        "allowed": ("declare_typed_predicate_field_spec", "disclose_semantic_uncertainty"),
        "forbidden": ("rely_on_generic_number_label_alone",),
    },
    "UNSPECIFIED_QUANTITY_ROUNDING_OR_TOLERANCE": {
        "repairable": True,
        "problem": "Quantity uses approximate precision without explicit rounding decimals or tolerance.",
        "instruction": "Qualify precision clearly and specify rounding or tolerance bounds.",
        "allowed": ("specify_rounding_or_tolerance",),
        "forbidden": ("unqualified_exact_assertion",),
    },
    "SUBJECT_CHANGED_IN_REPAIR": {
        "repairable": True,
        "problem": "Response alters or replaces the subject entity relative to the verified graph/evidence.",
        "instruction": (
            "Restore the verified subject entity from the evidence without substituting a different entity."
        ),
        "allowed": ("restore_verified_subject",),
        "forbidden": ("substitute_subject_entity",),
    },
    "CALCULATION_ALTERED_IN_RESPONSE": _CALC_ALTERED_SPEC,
    "CALCULATION_ALTERED_OR_INCONSISTENT": _CALC_ALTERED_SPEC,
    "UNIT_CHANGED_IN_RESPONSE": _UNIT_CHANGED_SPEC,
    "UNIT_ALTERED_IN_REPAIR": _UNIT_CHANGED_SPEC,
    "UNSUPPORTED_FACTUAL_ADDITION": {
        "repairable": True,
        "problem": "Response introduces dates, numbers, or facts not present in or derivable from evidence.",
        "instruction": (
            "Remove all invented dates/facts not present in the provided evidence; keep only grounded values."
        ),
        "allowed": ("remove_ungrounded_factual_additions",),
        "forbidden": ("invent_dates_or_background_facts",),
    },
    "QUESTION_EVADED": _QUESTION_EVADED_SPEC,
    "QUESTION_LEFT_UNANSWERED_EVASION": _QUESTION_EVADED_SPEC,
    "EMPTY_CANDIDATE_RESPONSE": {
        "repairable": True,
        "problem": "Candidate response text is empty.",
        "instruction": "Provide a non-empty answer grounded strictly in the verified graph and evidence.",
        "allowed": ("generate_grounded_response",),
        "forbidden": ("empty_output",),
    },
    "FALSE_EQUATION_IN_RESPONSE": {
        "repairable": True,
        "problem": "Response states an arithmetic equation whose sides do not evaluate equally.",
        "instruction": (
            "Correct the arithmetic equation so both sides equal the mathematically verified value, "
            "or remove the incorrect equation."
        ),
        "allowed": ("correct_equation_to_match_verified_derivation", "remove_false_equation"),
        "forbidden": ("retain_false_mathematical_equality",),
    },
    "UNGROUNDED_EQUATION_IN_RESPONSE": {
        "repairable": True,
        "problem": "Response states an equation whose operands or result do not correspond to verified graph steps.",
        "instruction": (
            "Remove the ungrounded equation or replace it strictly with the calculation derived from "
            "verified premise operands."
        ),
        "allowed": ("align_equation_with_verified_steps", "remove_ungrounded_equation"),
        "forbidden": ("invent_arithmetic_operands",),
    },
    "UNVERIFIED_EXPRESSION_IN_RESPONSE": {
        "repairable": True,
        "problem": "Response includes a symbolic expression that cannot be deterministically verified.",
        "instruction": "Rewrite the expression in verified numeric terms or remove the unparseable formula.",
        "allowed": ("replace_with_verified_arithmetic_expression", "remove_unparseable_expression"),
        "forbidden": ("claim_unverified_symbolic_formula",),
    },
    "CONDITION_POLARITY_INVERTED": {
        "repairable": True,
        "problem": (
            "Response inverts the logical polarity of a mandatory condition/exception "
            "(e.g. turning an excluding exception into a concessive override or positive condition)."
        ),
        "instruction": (
            "Restore the exact condition polarity from the evidence (e.g. 'salvo en caso de ...' or "
            "'a menos que ...', not 'incluso en caso de ...')."
        ),
        "allowed": ("restore_verified_condition_polarity",),
        "forbidden": ("invert_exception_with_concessive_marker", "claim_unconditional"),
    },
    "UNSUPPORTED_CONDITION_SUBSTITUTION": {
        "repairable": True,
        "problem": "Response substitutes an unrelated or ungrounded condition instead of the evidence condition.",
        "instruction": "Remove the invented condition and state the exact required condition from the evidence.",
        "allowed": ("restore_verified_condition_from_evidence",),
        "forbidden": ("substitute_irrelevant_condition",),
    },
    "CONDITION_CORRESPONDENCE_UNRESOLVED": {
        "repairable": True,
        "problem": "Condition equivalence in the response could not be confirmed unambiguously.",
        "instruction": "State the mandatory condition/exception explicitly using unmistakable conditional wording.",
        "allowed": ("state_condition_with_explicit_operator",),
        "forbidden": ("leave_condition_ambiguous",),
    },
    "CONCLUSION_POLARITY_INVERTED": {
        "repairable": True,
        "problem": "Response negates or reverses the polarity of the verified conclusion.",
        "instruction": "Restore the affirmative verified conclusion polarity supported by the evidence and graph.",
        "allowed": ("restore_verified_conclusion_polarity",),
        "forbidden": ("negate_verified_conclusion",),
    },
    "RESPONSE_CORRESPONDENCE_PENDING_SEMANTIC": {
        "repairable": True,
        "problem": "Open-language terms in the response could not be grounded without semantic verification.",
        "instruction": "Rephrase the claim strictly using terms grounded in the evidence and graph.",
        "allowed": ("rephrase_using_grounded_terms",),
        "forbidden": ("add_ungrounded_factual_claims",),
    },
    "RELATIONAL_VALUE_MISMATCH": {
        "repairable": True,
        "problem": "The response assigns a value, unit, counted noun, or temporal anchor to the wrong entity, role, or slot relative to the evidence graph.",
        "instruction": "Bind each number, unit, entity, counted object, and temporal anchor strictly to the same role and entity supported by the evidence and derivation graph.",
        "allowed": ("restate_exact_entity_role_value_relations", "attribute_each_value_to_its_documented_entity_and_date"),
        "forbidden": ("swap_entity_values", "swap_value_and_year_roles", "reassign_counted_nouns_across_slots"),
    },
    "RELATIONAL_CORRESPONDENCE_UNRESOLVED": {
        "repairable": True,
        "problem": "A relational claim pairing values or entities in the response could not be syntactically bound to a verified graph relation.",
        "instruction": "Express the relationship between entities, quantities, and dates explicitly using the exact predicates, units, and roles present in the evidence.",
        "allowed": ("clarify_entity_predicate_value_binding", "rephrase_using_grounded_terms"),
        "forbidden": ("leave_numeric_entity_pairing_ambiguous", "add_ungrounded_factual_claims"),
    },
    "ARITHMETIC_MISMATCH": {
        "repairable": True,
        "problem": "Derived arithmetic value does not match premise operands.",
        "instruction": "Recompute the conclusion strictly from the verified premise values.",
        "allowed": ("recompute_arithmetic_from_premises",),
        "forbidden": ("adding_approximation_to_wrong_number",),
    },
    "INVALID_JUSTIFICATION_GRAPH_STRUCTURE": {
        "repairable": False,
        "problem": "JustificationGraph has cyclic or invalid structural references.",
        "instruction": "Re-extract a valid acyclic JustificationGraph before attempting response repair.",
        "allowed": (),
        "forbidden": ("accept_invalid_graph",),
    },
    "PROBABILISTIC_ONLY_CRITICAL_STEP": {
        "repairable": False,
        "problem": "Critical inference relies exclusively on probabilistic semantic evaluation.",
        "instruction": "Supply a deterministic formal derivation or escalate for human review.",
        "allowed": (),
        "forbidden": ("bypass_probabilistic_gate",),
    },
    "PENDING_BUDGET": {
        "repairable": False,
        "problem": "Verification stopped early due to latency, step, or semantic-call budget.",
        "instruction": "Increase verification budget or escalate.",
        "allowed": (),
        "forbidden": ("accept_unverified_checks",),
    },
    "OMITTED_DERIVATION_ASSUMPTIONS": {
        "repairable": True,
        "problem": "Quantitative rate/spread impact derivation omits required structural assumptions (e.g. constant principal, simple annual calculation).",
        "instruction": (
            "State explicit derivation assumptions (constant principal, simple annual calculation, "
            "no amortization or day-count adjustments) and qualify the figure as an illustrative annual calculation."
        ),
        "allowed": ("state_explicit_derivation_assumptions", "qualify_as_illustrative_annual_calculation"),
        "forbidden": ("omit_principal_or_day_count_assumptions", "assert_guaranteed_realized_cashflow"),
    },
    "UNQUALIFIED_DAMAGES_OR_CASHFLOW_CLAIM": {
        "repairable": True,
        "problem": "Response presents an illustrative rate calculation as exact legal damages or guaranteed realized cashflow.",
        "instruction": (
            "Remove unqualified legal damages or realized cashflow loss language and frame the figure as an "
            "illustrative annual estimate conditioned on constant principal and simple annual accrual."
        ),
        "allowed": ("reframe_as_illustrative_annual_difference", "remove_unqualified_damages_claim"),
        "forbidden": ("claim_exact_legal_damages", "claim_guaranteed_realized_loss"),
    },
    "PERCENT_VS_PERCENTAGE_POINT_CONFUSION": {
        "repairable": True,
        "problem": "Response confuses absolute percentage points / basis points with relative percent difference.",
        "instruction": (
            "Express the coupon/rate spread explicitly in percentage points (pp) and/or basis points (bps), "
            "not as a relative percentage change or unqualified '%' difference."
        ),
        "allowed": ("state_spread_in_percentage_points_and_basis_points",),
        "forbidden": ("conflate_relative_percent_with_percentage_points",),
    },
    "WATERFALL_ORDER_CYCLE": {
        "repairable": False,
        "problem": "Payment waterfall ordering contains an inconsistent cyclic precedence relation.",
        "instruction": "Resolve the cyclic payment ordering conflict against the governing Indenture waterfall.",
        "allowed": (),
        "forbidden": ("retain_cyclic_waterfall_order",),
    },
    "WATERFALL_PRIORITY_INVERSION": {
        "repairable": True,
        "problem": "Report fails to report a cross-document payment priority inversion accurately or inverts governing vs subordinate waterfall precedence.",
        "instruction": (
            "Correct the payment priority comparison so each step's beneficiary and relative precedence match "
            "the governing vs subordinate documents accurately (explicitly disclosing priority inversions)."
        ),
        "allowed": ("restore_verified_waterfall_priority_comparison", "disclose_cross_document_inversion"),
        "forbidden": ("swap_waterfall_priority_direction", "hide_material_waterfall_inversion"),
    },
    "WATERFALL_BENEFICIARY_SWAP": {
        "repairable": True,
        "problem": "Response assigns a waterfall step or seniority slot to the wrong beneficiary/tranche.",
        "instruction": "Restore the documented beneficiary/tranche for each waterfall step without swapping classes.",
        "allowed": ("restore_verified_waterfall_beneficiary",),
        "forbidden": ("swap_beneficiaries_across_waterfall_steps",),
    },
    "WATERFALL_PAYMENT_TYPE_CONFUSION": {
        "repairable": True,
        "problem": "Response confuses payment type (e.g. interest vs principal vs reserve vs fees) at a waterfall step.",
        "instruction": "Correct the payment type (interest vs principal vs reserve vs fees) to match the documented waterfall step.",
        "allowed": ("restore_verified_waterfall_payment_type",),
        "forbidden": ("conflate_interest_and_principal_steps",),
    },
    "WATERFALL_INCOMPARABLE_REGIME": {
        "repairable": True,
        "problem": "Response compares waterfall steps across different regimes/conditions (e.g. pre-acceleration vs post-acceleration) without regime qualification.",
        "instruction": "Explicitly distinguish the governing waterfall regime and triggering conditions before comparing payment order.",
        "allowed": ("qualify_waterfall_regime_and_conditions",),
        "forbidden": ("equate_steps_across_incompatible_regimes",),
    },
    "AUTHORITY_INVERSION": {
        "repairable": True,
        "problem": "Response treats a lower-authority document (e.g. Offering Memorandum or associate email) as governing over the executed Indenture.",
        "instruction": "Enforce governing-document precedence explicitly so the executed Indenture governs over the Offering Memorandum and internal correspondence.",
        "allowed": ("restore_governing_document_precedence",),
        "forbidden": ("treat_subordinate_document_or_email_as_governing",),
    },
}


def _normalize_question_str(question: str | Mapping[str, Any] | None) -> str | None:
    if question is None:
        return None
    if isinstance(question, Mapping):
        val = str(question.get("text") or question.get("question") or "").strip()
        return val or None
    val = str(question).strip()
    return val or None


def _resolve_graph_input(
    graph_or_adapter: Any,
    *,
    question: str | Mapping[str, Any] | None,
    candidate_response: str | None,
    evidence: Any,
    domain_profile: DomainProfile,
    representation_mode: str | None,
) -> tuple[JustificationGraph | Mapping[str, Any], str]:
    """Resolve explicit graph or invoke graph extraction adapter cleanly."""
    if isinstance(graph_or_adapter, (JustificationGraph, Mapping)):
        rep_mode = representation_mode or (
            "graph_only" if candidate_response is None else "supplied_graph_with_response"
        )
        return graph_or_adapter, rep_mode

    if hasattr(graph_or_adapter, "extract_graph") and callable(graph_or_adapter.extract_graph):
        graph = graph_or_adapter.extract_graph(
            question=question,
            candidate_response=candidate_response,
            evidence=evidence,
            domain_profile=domain_profile,
        )
        return graph, representation_mode or "adapter_extracted_graph"

    if hasattr(graph_or_adapter, "propose_graph") and callable(graph_or_adapter.propose_graph):
        graph = graph_or_adapter.propose_graph(
            question=question,
            candidate_response=candidate_response,
            evidence=evidence,
            domain_profile=domain_profile,
        )
        return graph, representation_mode or "adapter_extracted_graph"

    if callable(graph_or_adapter):
        graph = graph_or_adapter(
            question=question,
            candidate_response=candidate_response,
            evidence=evidence,
            domain_profile=domain_profile,
        )
        return graph, representation_mode or "adapter_extracted_graph"

    raise TypeError(
        "graph_or_adapter must be a JustificationGraph, Mapping, callable, or adapter exposing extract_graph/propose_graph."
    )


def _extract_interval_from_evidence(check_evidence: Mapping[str, Any]) -> tuple[Any, Any, float | None]:
    """Extract interval bounds [lo, hi] and numeric span from verifier evidence cleanly."""
    lo = check_evidence.get("derived_min")
    hi = check_evidence.get("derived_max")
    if (lo is None or hi is None) and isinstance(check_evidence.get("computed_interval"), Sequence):
        ci = check_evidence["computed_interval"]
        if len(ci) == 2:
            lo, hi = ci[0], ci[1]
    span_raw = check_evidence.get("derived_span") if check_evidence.get("derived_span") is not None else check_evidence.get("span")
    span: float | None = None
    if span_raw is not None:
        try:
            span = float(span_raw)
        except (TypeError, ValueError):
            span = None
    elif lo is not None and hi is not None:
        try:
            span = float(hi) - float(lo)
        except (TypeError, ValueError):
            span = None
    return lo, hi, span


def _contextualize_instruction(
    code: str,
    base_instruction: str,
    assessment: ResponseAssessment,
    check_evidence: Mapping[str, Any],
) -> str:
    """Enrich generic repair instructions dynamically with check evidence (no hardcoded entities)."""
    details: list[str] = [base_instruction]
    conc = assessment.conclusion

    if code in ("UNJUSTIFIED_TEMPORAL_PRECISION", "TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE", "APPROXIMATION_HIDES_WIDE_UNCERTAINTY", "TEMPORAL_DERIVATION_CONTRADICTED"):
        lo, hi, span = _extract_interval_from_evidence(check_evidence)
        if lo is not None and hi is not None:
            details.append(f"Verified derived interval from evidence: [{lo}, {hi}] (span={span}).")
        elif conc.value is not None:
            unit_s = f" {conc.unit}" if conc.unit else ""
            details.append(f"Derived target value: ~{conc.value}{unit_s}.")

    elif code in ("CONDITION_DROPPED_IN_RESPONSE", "DROPPED_REQUIRED_CONDITION"):
        req_conds = []
        if conc.conditions:
            req_conds.extend(conc.conditions)
        for p in assessment.premises:
            cid = p.get("claim_id")
            if cid:
                req_conds.append({"claim_id": cid})
        if check_evidence.get("missing_conditions"):
            req_conds = list(check_evidence["missing_conditions"])
        if req_conds:
            cond_strs = [
                f"{c.get('field', 'condition')}={c.get('value')}"
                if isinstance(c, Mapping) and "value" in c
                else str(c)
                for c in req_conds[:4]
            ]
            details.append(f"Missing required condition(s): {', '.join(cond_strs)}.")

    elif code in ("UNDISCLOSED_MATERIAL_CONFLICT", "CONFLICT_HIDDEN_IN_RESPONSE"):
        if assessment.contradicting_evidence:
            conflicts = [
                f"{c.get('subject')}.{c.get('predicate')}={c.get('value')} {c.get('unit') or ''}".strip()
                for c in assessment.contradicting_evidence[:3]
            ]
            details.append(f"Disclose contradicting evidence: {'; '.join(conflicts)}.")

    elif code == "SUBJECT_CHANGED_IN_REPAIR" and conc.subject:
        details.append(f"Expected subject: {conc.subject!r}.")

    elif code in ("CALCULATION_ALTERED_IN_RESPONSE", "CALCULATION_ALTERED_OR_INCONSISTENT") and conc.value is not None:
        details.append(f"Expected derived value: {conc.value!r}.")

    elif code in ("UNIT_CHANGED_IN_RESPONSE", "UNIT_ALTERED_IN_REPAIR") and conc.unit:
        details.append(f"Expected unit: {conc.unit!r}.")

    return " ".join(details)


def _extract_issues_and_repairs(
    assessment: ResponseAssessment,
) -> tuple[tuple[ReviewIssue, ...], tuple[RequiredRepair, ...], bool]:
    """Extract deduplicated ReviewIssues and RequiredRepairs from a ResponseAssessment."""
    issues: list[ReviewIssue] = []
    repairs: list[RequiredRepair] = []
    seen_codes: set[str] = set()
    all_repairable = True

    present_codes = {
        r
        for chk in assessment.completed_checks
        if chk.status in ("refuted", "unknown")
        for r in chk.reasons
    }
    suppress_insufficient_when_overclaimed = "UNJUSTIFIED_TEMPORAL_PRECISION" in present_codes

    for chk in assessment.completed_checks:
        if chk.status not in ("refuted", "unknown"):
            continue
        severity = "error" if chk.status == "refuted" else "warning"
        codes = tuple(chk.reasons) if chk.reasons else (
            "VERIFICATION_REFUTED" if chk.status == "refuted" else "OBLIGATION_UNRESOLVED"
        )
        for idx, code in enumerate(codes):
            if suppress_insufficient_when_overclaimed and code == "TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE":
                continue
            dedup_key = (
                f"{code}::{chk.obligation_id}"
                if code
                in (
                    "CORPUS_OBLIGATION_UNCOVERED",
                    "CORPUS_OBLIGATION_REJECTED",
                    "GOVERNING_SOURCE_MISMATCH",
                    "VALUE_ATTRIBUTION_MISMATCH",
                )
                else code
            )
            if dedup_key in seen_codes:
                continue
            seen_codes.add(dedup_key)

            spec = _REPAIR_SPEC_TABLE.get(code)
            if code == "CORPUS_OBLIGATION_UNCOVERED":
                is_rep = True
                problem = str(chk.evidence.get("message") or f"Corpus obligation '{chk.obligation_id}' is uncovered.")
                instruction = str(chk.evidence.get("remediation") or "Cover the corpus obligation in graph and text.")
                allowed = ("add_verified_corpus_leaf_and_text_coverage",)
                forbidden = ("omit_corpus_obligation", "prune_prior_obligation")
            elif code in ("GOVERNING_SOURCE_MISMATCH", "VALUE_ATTRIBUTION_MISMATCH"):
                is_rep = True
                problem = str(chk.evidence.get("message") or f"Invalid conflict disclosure content ({code}).")
                instruction = str(chk.evidence.get("remediation") or "Fix governing source and value attribution.")
                allowed = ("fix_conflict_governance_and_value_attribution",)
                forbidden = ("invert_governing_source", "swap_conflicting_values", "claim_unequal_rank_unresolved")
            elif code in (
                "REFERENCE_REGISTER_MISSING",
                "REFERENCE_REGISTER_INCOMPLETE",
                "REFERENCE_REGISTER_GAP",
                "CORPUS_OBLIGATION_REJECTED",
            ):
                is_rep = False
                problem = str(chk.evidence.get("message") or f"Non-repairable corpus closure failure ({code}).")
                instruction = str(chk.evidence.get("remediation") or "Escalate to human review.")
                allowed = ("escalate_to_human",)
                forbidden = ("accept_without_reference_register", "bypass_rejected_obligation")
            elif spec is not None:
                is_rep = bool(spec["repairable"])
                problem = str(spec["problem"])
                instruction = _contextualize_instruction(code, str(spec["instruction"]), assessment, chk.evidence)
                allowed = tuple(spec["allowed"])
                forbidden = tuple(spec["forbidden"])
            else:
                is_rep = chk.status == "refuted" and chk.family in (
                    "temporal",
                    "quantities",
                    "aggregation",
                    "explicit_logic",
                    "document_authority",
                    "arithmetic",
                )
                problem = f"Check '{chk.obligation_id}' ({chk.family}) ended with status='{chk.status}' ({code})."
                instruction = (
                    f"Revise the response to satisfy obligation '{chk.obligation_id}' ({code}) without violating evidence grounding."
                )
                allowed = ("qualify_response_to_match_verified_evidence",)
                forbidden = ("bypass_mandatory_obligation",)

            if not is_rep:
                all_repairable = False

            target_claim = assessment.conclusion_id
            issues.append(
                ReviewIssue(
                    code=code,
                    severity=severity,
                    family=chk.family,
                    obligation_id=chk.obligation_id,
                    step_id=chk.step_id,
                    message=problem,
                    target_claim_id=target_claim,
                    evidence=dict(chk.evidence),
                )
            )
            if is_rep:
                repairs.append(
                    RequiredRepair(
                        repair_id=f"repair::{chk.obligation_id}::{code.lower()}::{idx}",
                        obligation_id=chk.obligation_id,
                        family=chk.family,
                        target_claim_id=target_claim,
                        issue_code=code,
                        problem=problem,
                        instruction=instruction,
                        allowed_repairs=allowed,
                        forbidden_shortcuts=forbidden,
                    )
                )

    return tuple(issues), tuple(repairs), all_repairable


def review_response(
    question: str | Mapping[str, Any] | None,
    candidate_response: str | None,
    evidence: EvidenceLayer | Mapping[str, SourceDocument] | Sequence[SourceDocument] | Mapping[str, str] | None,
    graph_or_adapter: Any,
    domain_profile: DomainProfile | None = None,
    *,
    cache: VerificationCache | None = None,
    budget: VerificationBudget | None = None,
    as_of: str | None = None,
    previous_attempt: Mapping[str, Any] | ReviewResult | None = None,
    operation_policy: DomainProfile | PolicyConfiguration | None = None,
    authorize_operation_id: str | None = None,
    representation_mode: str | None = None,
    corpus_register: CorpusReferenceRegister | None = None,
    require_reference_register: bool = False,
) -> ReviewResult:
    """Evaluate a candidate response + justification against evidence and emit ACCEPT/RETRY/ESCALATE/ABSTAIN."""
    profile = domain_profile or DomainProfile(profile_id="default")
    q_norm = _normalize_question_str(question)

    resolved_graph, rep_mode = _resolve_graph_input(
        graph_or_adapter,
        question=question,
        candidate_response=candidate_response,
        evidence=evidence,
        domain_profile=profile,
        representation_mode=representation_mode,
    )

    assessment = evaluate_response(
        resolved_graph,
        evidence=evidence,
        domain_profile=profile,
        cache=cache,
        budget=budget,
        as_of=as_of,
        question=question,
        candidate_response=candidate_response,
        representation_mode=rep_mode,
        corpus_register=corpus_register,
        require_reference_register=require_reference_register,
    )

    op_auth: AuthorizationVerdict | None = None
    if authorize_operation_id is not None or operation_policy is not None:
        layer = evidence if isinstance(evidence, EvidenceLayer) else None
        op_auth = authorize(
            assessment,
            operation=authorize_operation_id or "default_operation",
            policy=operation_policy or profile,
            layer=layer,
        )

    issues, repairs, all_repairable = _extract_issues_and_repairs(assessment)
    verified_elements = tuple(
        {
            "obligation_id": chk.obligation_id,
            "step_id": chk.step_id,
            "family": chk.family,
            "reasons": list(chk.reasons),
            "guarantee_kind": chk.guarantee_kind,
        }
        for chk in assessment.completed_checks
        if chk.status == "verified"
    )

    # Detect formal structural errors immediately
    has_structural_error = any(i.code == "INVALID_JUSTIFICATION_GRAPH_STRUCTURE" for i in issues)
    has_non_repairable_corpus = any(
        i.code in (
            "REFERENCE_REGISTER_MISSING",
            "REFERENCE_REGISTER_INCOMPLETE",
            "REFERENCE_REGISTER_GAP",
            "CORPUS_OBLIGATION_REJECTED",
        )
        for i in issues
    )
    has_budget_deferral = any(i.code == "PENDING_BUDGET" for i in issues)

    if has_structural_error:
        disposition = "ESCALATE"
        summary = "Graph structure is invalid or cyclic; manual/structural re-extraction required."
    elif has_non_repairable_corpus:
        disposition = "ESCALATE"
        summary = "Corpus reference register missing or obligation explicitly rejected; escalation required."
    elif issues:
        if any(i.code in ("PROBABILISTIC_ONLY_CRITICAL_STEP", "LEAF_EVIDENCE_MISSING") for i in issues) and not repairs:
            disposition = "ESCALATE"
            summary = "Verification failed on non-repairable grounding or probabilistic-only step."
        elif repairs and all_repairable and not has_budget_deferral:
            disposition = "RETRY"
            summary = f"Response overclaims or omits {len(repairs)} required qualifier(s); RETRY with specific repairs."
        elif repairs and assessment.assessment == "refuted":
            disposition = "RETRY"
            summary = f"Response violates {len(repairs)} check(s); RETRY with required repairs."
        elif has_budget_deferral:
            disposition = "ABSTAIN"
            summary = "Verification interrupted by budget constraints; abstaining from acceptance."
        else:
            disposition = "ESCALATE"
            summary = "Unresolved non-repairable obligations require escalation."
    elif assessment.assessment == "supported" and (
        candidate_response is None or assessment.response_correspondence_verified
    ):
        disposition = "ACCEPT"
        summary = "Response is fully supported by verified evidence and satisfies all mandatory obligations."
    elif (
        assessment.assessment == "contested"
        and candidate_response is not None
        and assessment.response_correspondence_verified
        and bool(assessment.correspondence_report.get("conflict_disclosed"))
    ):
        disposition = "ACCEPT"
        summary = (
            "Response faithfully discloses unresolved material conflict without taking an unsupported side "
            "(operational execution remains blocked)."
        )
    elif (
        assessment.assessment == "partially_supported"
        and assessment.verification_coverage == 1.0
        and not assessment.pending_obligations
        and (candidate_response is None or assessment.response_correspondence_verified)
        and assessment.evidence_support_index > 0.0
    ):
        disposition = "ACCEPT"
        summary = (
            "Response passes all deterministic and correspondence checks as a qualified explanation "
            "(graduated support remains below autonomous operational threshold)."
        )
    elif assessment.assessment == "contested":
        disposition = "ESCALATE"
        summary = "Unresolved conflict in evidence without explicit conflict-disclosing response."
    elif assessment.assessment in ("inconclusive", "unsupported"):
        disposition = "ABSTAIN"
        summary = f"Insufficient verified support ({assessment.assessment}); abstaining from acceptance."
    else:
        disposition = "ESCALATE"
        summary = f"Assessment '{assessment.assessment}' cannot be accepted."

    # Track previous attempt regression note without inventing issues
    if previous_attempt is not None:
        prev_resp = (
            previous_attempt.candidate_response
            if isinstance(previous_attempt, ReviewResult)
            else previous_attempt.get("candidate_response")
        )
        if prev_resp is not None and prev_resp == candidate_response and disposition == "ACCEPT":
            # Same exact text would only happen if prior was already acceptable
            pass

    return ReviewResult(
        disposition=disposition,
        question=q_norm,
        candidate_response=candidate_response,
        response_digest=assessment.response_digest,
        assessment_digest=assessment.assessment_digest,
        representation_mode=assessment.representation_mode,
        verified_elements=verified_elements,
        issues=issues,
        required_repairs=repairs,
        unresolved_obligations=assessment.pending_obligations,
        assessment=assessment,
        operational_authorization=op_auth,
        summary=summary,
    )


def verify_review_freshness(
    review_result: ReviewResult,
    candidate_response: str | None,
    question: str | Mapping[str, Any] | None = None,
) -> bool:
    """Cryptographically bind an ACCEPT review verdict to exact question & candidate response text.

    Any post-approval mutation to ``candidate_response`` or ``question`` immediately
    invalidates prior acceptance and returns ``False``.
    """
    if review_result.disposition != "ACCEPT":
        return False
    if review_result.response_digest is None:
        return False
    expected_q = question if question is not None else review_result.question
    current_digest = compute_response_digest(candidate_response, expected_q)
    return bool(
        current_digest is not None
        and current_digest == review_result.response_digest
        and review_result.candidate_response == candidate_response
    )


# ---------------------------------------------------------------------------
# Bounded repair orchestrator & doubles / real LLM adapters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairBudget:
    """Hard bounds preventing endless repair loops or uncontrolled LLM costs."""

    max_iterations: int = 3
    max_latency_ms: float | None = 30000.0
    max_semantic_calls: int = 4
    stop_on_duplicate_response: bool = True
    stop_on_stagnant_issues: bool = True

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("RepairBudget.max_iterations must be >= 1.")


@dataclass(frozen=True)
class RepairAttemptRecord:
    """Audit record for one evaluate -> feedback iteration."""

    iteration: int
    candidate_response: str | None
    response_digest: str | None
    review: ReviewResult
    cache_metrics: dict[str, int]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "candidate_response": self.candidate_response,
            "response_digest": self.response_digest,
            "disposition": self.review.disposition,
            "issues": [i.to_dict() for i in self.review.issues],
            "required_repairs": [r.to_dict() for r in self.review.required_repairs],
            "cache_metrics": dict(self.cache_metrics),
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass(frozen=True)
class RepairCycleResult:
    """Complete audit trace of a bounded evaluation -> repair -> reverification run."""

    final_disposition: str  # "ACCEPT" | "ESCALATE" | "ABSTAIN"
    accepted: bool
    stop_reason: str
    question: str | None
    initial_response: str | None
    final_response: str | None
    attempts: tuple[RepairAttemptRecord, ...]
    final_review: ReviewResult
    operational_authorization: AuthorizationVerdict | None = None
    iterations_used: int = 0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_disposition": self.final_disposition,
            "accepted": self.accepted,
            "stop_reason": self.stop_reason,
            "question": self.question,
            "initial_response": self.initial_response,
            "final_response": self.final_response,
            "attempts": [a.to_dict() for a in self.attempts],
            "final_review": self.final_review.to_dict(),
            "operational_authorization": (
                self.operational_authorization.to_dict()
                if self.operational_authorization is not None
                else None
            ),
            "iterations_used": self.iterations_used,
            "total_latency_ms": round(self.total_latency_ms, 3),
        }


class ScriptedRepairAgent:
    """Deterministic scripted test double returning predefined candidate repairs."""

    def __init__(self, responses: Sequence[str] | Callable[[Mapping[str, Any]], str]) -> None:
        self._responses = responses
        self._index = 0

    def repair(
        self,
        *,
        question: str | None,
        candidate_response: str | None,
        evidence: Any,
        graph: JustificationGraph,
        review: ReviewResult,
    ) -> str:
        if callable(self._responses):
            return self._responses({
                "question": question,
                "candidate_response": candidate_response,
                "review": review,
                "graph": graph,
                "iteration": self._index + 1,
            })
        if self._index < len(self._responses):
            out = self._responses[self._index]
            self._index += 1
            return out
        return self._responses[-1] if self._responses else (candidate_response or "")


class RuleGuidedRepairAdapter:
    """Deterministic, non-entity-specific repair generator driven solely by ReviewResult repairs."""

    def repair(
        self,
        *,
        question: str | None,
        candidate_response: str | None,
        evidence: Any,
        graph: JustificationGraph,
        review: ReviewResult,
    ) -> str:
        conc = graph.claims[graph.conclusion_id]
        issue_codes = {i.code for i in review.issues}

        val = conc.value
        unit = f" {conc.unit}" if conc.unit else ""
        if unit.strip().lower() in ("years", "year"):
            unit = " años"
        elif unit.strip().lower() in ("months", "month"):
            unit = " meses"

        parts: list[str] = []
        notes: list[str] = []

        # Source attribution qualification
        if "SOURCE_CONDITIONAL_CONCLUSION" in issue_codes or "CONTRADICTORY_SOURCE_ATTRIBUTION" in issue_codes or any(
            c.obligation_id == "mandatory::documental_attribution" and c.status == "verified"
            for c in review.assessment.completed_checks
        ):
            prefix = "Según el texto"
            notes.append(
                "Esta conclusión depende únicamente del documento aportado y no ha sido verificada como hecho externo."
            )
        else:
            prefix = "Según la evidencia"

        # Temporal / interval bounds & missing date limitations
        derived_lo = None
        derived_hi = None
        derived_span = None
        for chk in review.assessment.completed_checks:
            lo, hi, sp = _extract_interval_from_evidence(chk.evidence)
            if lo is not None and hi is not None:
                derived_lo, derived_hi, derived_span = lo, hi, sp
                break

        # Subject phrase (preserve expected subject if altered or temporal anchor)
        subj_phrase = f" sobre {conc.subject}" if conc.subject else ""

        if "UNDISCLOSED_MATERIAL_CONFLICT" in issue_codes or "CONFLICT_HIDDEN_IN_RESPONSE" in issue_codes:
            alt_vals = [
                f"{c.get('value')}{(' ' + str(c.get('unit'))) if c.get('unit') else ''}"
                for c in review.assessment.contradicting_evidence
            ]
            alt_desc = ", ".join(alt_vals) if alt_vals else "otra cifra alternativa"
            parts.append(
                f"{prefix}{subj_phrase}, existe un conflicto sin resolver entre fuentes: "
                f"una fuente indica {val}{unit} mientras otra señala {alt_desc}, por lo que no puede darse una cifra única."
            )
        elif "APPROXIMATION_HIDES_WIDE_UNCERTAINTY" in issue_codes or (
            derived_lo is not None and derived_hi is not None and (derived_span or 0) > 2
        ):
            parts.append(
                f"{prefix}{subj_phrase}, el valor se sitúa entre {derived_lo} y {derived_hi}{unit}."
            )
            notes.append("No hay fechas completas para determinar una cifra exacta.")
        elif (
            "UNJUSTIFIED_TEMPORAL_PRECISION" in issue_codes
            or "UNQUALIFIED_EXACT_VALUE_FOR_APPROX_CLAIM" in issue_codes
            or "CONTRADICTORY_PRECISION_QUALIFICATION" in issue_codes
            or "TEMPORAL_DERIVATION_CONTRADICTED" in issue_codes
            or "UNSPECIFIED_QUANTITY_ROUNDING_OR_TOLERANCE" in issue_codes
        ):
            target_num = val
            if "TEMPORAL_DERIVATION_CONTRADICTED" in issue_codes and derived_lo is not None and derived_hi is not None:
                target_num = f"entre {derived_lo} y {derived_hi}"
                parts.append(f"{prefix}{subj_phrase}, tendría aproximadamente {target_num}{unit}.")
            else:
                parts.append(f"{prefix}{subj_phrase}, tendría aproximadamente {target_num}{unit}.")
            if derived_lo is not None or "UNJUSTIFIED_TEMPORAL_PRECISION" in issue_codes or "CONTRADICTORY_PRECISION_QUALIFICATION" in issue_codes:
                notes.append("Sin fechas completas de nacimiento no puede determinarse una edad exacta.")
        elif "SUBTOTAL_PRESENTED_AS_TOTAL" in issue_codes or "UNQUALIFIED_SUBTOTAL_CLAIM" in issue_codes or "UNVERIFIED_POPULATION_COMPLETENESS" in issue_codes:
            parts.append(
                f"{prefix}{subj_phrase}, el subtotal parcial de las partidas incluidas es de {val}{unit}, "
                "sin verificar que represente la población total completa."
            )
        else:
            parts.append(f"{prefix}{subj_phrase}, el resultado sería aproximadamente {val}{unit}.")

        # Preserve mandatory conditions
        cond_items: list[str] = list(conc.conditions)
        for p_id in sorted(graph.ancestor_claims(graph.conclusion_id)):
            node = graph.claims[p_id]
            cond_items.extend(node.conditions)
        if cond_items and ("DROPPED_REQUIRED_CONDITION" in issue_codes or "CONDITION_DROPPED_IN_RESPONSE" in issue_codes or cond_items):
            cond_strs = []
            for c in cond_items:
                if isinstance(c, Mapping):
                    f = str(c.get("field") or c.get("type") or "condición")
                    v = c.get("value")
                    if v is not None and str(v).strip().lower().startswith(("salvo", "excepto", "a menos que", "unless", "except")):
                        cond_strs.append(str(v).strip())
                    else:
                        cond_strs.append(f"{f}: {v}" if v is not None else f)
                else:
                    cond_strs.append(str(c))
            dedup_conds = list(dict.fromkeys(cond_strs))
            if dedup_conds:
                notes.append(f"Sujeto a la condición / excepción: {'; '.join(dedup_conds)}.")

        # Mention question year/scope if present so correspondence knows question wasn't evaded
        if question:
            for tok in question.split():
                clean = tok.strip("¿?.,;:")
                if clean.isdigit() and len(clean) == 4 and clean not in parts[0]:
                    parts[0] = parts[0].rstrip(".") + f" en {clean}."
                    break

        return " ".join(parts + notes).strip()


class LLMGraphProposalAdapter:
    """Adapter using OpenAICompatibleModel to propose a structured JustificationGraph."""

    _SYSTEM_PROMPT = (
        "You extract a structured JSON JustificationGraph from a question, candidate response, "
        "and source documents. Output strictly JSON with keys: claims (list), steps (list), conclusion_id (str). "
        "Each claim has claim_id, kind ('extraction'|'derivation'|'hypothesis'|'recommendation'|'prediction'), "
        "subject, predicate, value, unit, approximate (bool), evidence (list of {document_id, quote}), conditions (list). "
        "Each step has step_id, operation, premises (list of claim_id), conclusion_id, parameters (object)."
    )

    def __init__(self, model: OpenAICompatibleModel) -> None:
        self.model = model

    def extract_graph(
        self,
        *,
        question: str | Mapping[str, Any] | None,
        candidate_response: str | None,
        evidence: Any,
        domain_profile: DomainProfile | None = None,
    ) -> JustificationGraph:
        doc_payload = []
        if isinstance(evidence, Mapping):
            for did, doc in evidence.items():
                text = doc.text if hasattr(doc, "text") else str(doc)
                doc_payload.append({"document_id": str(did), "text": text})
        elif isinstance(evidence, Sequence):
            for idx, doc in enumerate(evidence):
                did = getattr(doc, "document_id", f"doc_{idx}")
                text = doc.text if hasattr(doc, "text") else str(doc)
                doc_payload.append({"document_id": did, "text": text})

        res = self.model.complete(
            self._SYSTEM_PROMPT,
            {
                "question": _normalize_question_str(question),
                "candidate_response": candidate_response,
                "documents": doc_payload,
            },
        )
        out = res.get("output")
        if not isinstance(out, Mapping):
            raise ValueError("LLMGraphProposalAdapter received non-dict JSON output.")
        vocab = domain_profile.vocabulary if domain_profile is not None else None
        return JustificationGraph.from_response(out, vocabulary=vocab)


class LLMRepairAgent:
    """Repair agent backed by claimlayer.semantic.adapters.OpenAICompatibleModel."""

    _SYSTEM_PROMPT = (
        "Eres un asistente de reparación de respuestas auditado por ClaimLayer v4.\n"
        "Debes corregir 'candidate_response' siguiendo EXACTAMENTE 'required_repairs' y evitando 'forbidden_shortcuts'.\n"
        "Reglas estrictas:\n"
        "1. No inventes fechas ni hechos ausentes de los documentos.\n"
        "2. No cambies el sujeto, los valores ni las unidades verificadas.\n"
        "3. Si falta información temporal completa, califica la cifra como aproximada/intervalo y explica esa limitación, "
        "sin añadir afirmaciones contradictorias de exactitud.\n"
        "4. Si el texto es narrativo sin verificación externa, atribúyelo explícitamente a la fuente/texto.\n"
        "5. Conserva todas las condiciones/excepciones y revela cualquier conflicto material o carácter de subtotal parcial.\n"
        "Devuelve SOLO un objeto JSON con clave obligatoria \"repaired_response\" (string)."
    )

    def __init__(self, model: OpenAICompatibleModel) -> None:
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def repair(
        self,
        *,
        question: str | None,
        candidate_response: str | None,
        evidence: Any,
        graph: JustificationGraph,
        review: ReviewResult,
    ) -> str:
        ev_snippets: list[dict[str, str]] = []
        if isinstance(evidence, Mapping):
            for did, doc in evidence.items():
                ev_snippets.append({
                    "document_id": str(did),
                    "text": doc.text if hasattr(doc, "text") else str(doc),
                })
        elif isinstance(evidence, Sequence):
            for idx, doc in enumerate(evidence):
                ev_snippets.append({
                    "document_id": getattr(doc, "document_id", f"doc_{idx}"),
                    "text": doc.text if hasattr(doc, "text") else str(doc),
                })

        conc = graph.claims[graph.conclusion_id]
        payload = {
            "question": question,
            "previous_response": candidate_response,
            "evidence_excerpts": ev_snippets,
            "verified_conclusion": {
                "subject": conc.subject,
                "predicate": conc.predicate,
                "value": conc.value,
                "unit": conc.unit,
                "approximate": conc.approximate,
                "conditions": list(conc.conditions),
            },
            "issues": [i.to_dict() for i in review.issues],
            "required_repairs": [r.to_dict() for r in review.required_repairs],
        }
        res = self.model.complete(self._SYSTEM_PROMPT, payload)
        self.calls.append({"payload": payload, "response": res})
        output = res.get("output")
        if isinstance(output, Mapping):
            for key in ("repaired_response", "response", "answer", "text"):
                val = output.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        if isinstance(output, str) and output.strip():
            return output.strip()
        raw_text = res.get("raw_text") or json.dumps(output, ensure_ascii=False)
        return str(raw_text).strip()


class RepairOrchestrator:
    """Executes bounded evaluate -> feedback -> repair -> reverify cycles."""

    def __init__(
        self,
        *,
        budget: RepairBudget | None = None,
        cache: VerificationCache | None = None,
        domain_profile: DomainProfile | None = None,
        operation_policy: DomainProfile | PolicyConfiguration | None = None,
        authorize_operation_id: str | None = None,
    ) -> None:
        self.budget = budget or RepairBudget()
        self.cache = cache if cache is not None else VerificationCache()
        self.domain_profile = domain_profile or DomainProfile(profile_id="default")
        self.operation_policy = operation_policy
        self.authorize_operation_id = authorize_operation_id

    def run(
        self,
        *,
        question: str | Mapping[str, Any] | None,
        initial_response: str | None,
        evidence: EvidenceLayer | Mapping[str, SourceDocument] | Sequence[SourceDocument] | Mapping[str, str] | None,
        graph_or_adapter: Any,
        repair_agent: Any,
        as_of: str | None = None,
        representation_mode: str | None = None,
        corpus_register: CorpusReferenceRegister | None = None,
        require_reference_register: bool = False,
    ) -> RepairCycleResult:
        t_cycle_start = time.monotonic()
        ver_budget = VerificationBudget(
            max_latency_ms=self.budget.max_latency_ms,
            max_semantic_calls=self.budget.max_semantic_calls,
        )
        q_norm = _normalize_question_str(question)

        current_response = initial_response
        current_graph_source = graph_or_adapter
        attempts: list[RepairAttemptRecord] = []
        seen_digests: set[str] = set()
        seen_covered_corpus_ids: set[str] = set()
        prev_issue_sig: tuple[tuple[str, str], ...] | None = None
        stop_reason = "max_iterations_reached"
        previous_review: ReviewResult | None = None

        for iteration in range(1, self.budget.max_iterations + 1):
            t_iter = time.monotonic()
            review = review_response(
                question,
                current_response,
                evidence,
                current_graph_source,
                self.domain_profile,
                cache=self.cache,
                budget=ver_budget,
                as_of=as_of,
                previous_attempt=previous_review,
                operation_policy=self.operation_policy,
                authorize_operation_id=self.authorize_operation_id,
                representation_mode=representation_mode,
                corpus_register=corpus_register,
                require_reference_register=require_reference_register,
            )
            curr_covered_corpus = set((review.assessment.corpus_coverage or {}).get("covered_ids") or ())
            dropped_corpus = sorted(seen_covered_corpus_ids - curr_covered_corpus)
            seen_covered_corpus_ids |= curr_covered_corpus
            if dropped_corpus:
                dropped_issue = ReviewIssue(
                    code="CORPUS_OBLIGATION_DROPPED",
                    severity="error",
                    family="document_authority",
                    obligation_id=dropped_corpus[0],
                    step_id="corpus::coverage",
                    message=f"Previously covered corpus obligation(s) dropped in attempt {iteration}: {dropped_corpus}.",
                    target_claim_id=review.assessment.conclusion_id,
                    evidence={"dropped_obligation_ids": dropped_corpus},
                )
                dropped_repair = RequiredRepair(
                    repair_id=f"repair::corpus_dropped::{iteration}",
                    obligation_id=dropped_corpus[0],
                    family="document_authority",
                    target_claim_id=review.assessment.conclusion_id,
                    issue_code="CORPUS_OBLIGATION_DROPPED",
                    problem=dropped_issue.message,
                    instruction="Restore all previously covered corpus obligations in graph and text.",
                    allowed_repairs=("restore_dropped_corpus_obligations",),
                    forbidden_shortcuts=("prune_covered_corpus_obligation",),
                )
                new_disp = "RETRY" if iteration < self.budget.max_iterations else "ESCALATE"
                review = ReviewResult(
                    disposition=new_disp,
                    summary=f"Corpus obligation(s) dropped in repair attempt {iteration}: {dropped_corpus}.",
                    assessment=review.assessment,
                    issues=tuple(review.issues) + (dropped_issue,),
                    required_repairs=tuple(review.required_repairs) + (dropped_repair,),
                    verified_elements=review.verified_elements,
                    operational_authorization=review.operational_authorization,
                    response_digest=review.response_digest,
                    assessment_digest=review.assessment_digest,
                    representation_mode=review.representation_mode,
                )
            iter_ms = (time.monotonic() - t_iter) * 1000.0
            attempts.append(
                RepairAttemptRecord(
                    iteration=iteration,
                    candidate_response=current_response,
                    response_digest=review.response_digest,
                    review=review,
                    cache_metrics=dict(review.assessment.cache_metrics),
                    latency_ms=iter_ms,
                )
            )
            previous_review = review

            if review.response_digest is not None:
                seen_digests.add(review.response_digest)

            curr_issue_sig = tuple(sorted((i.code, i.obligation_id) for i in review.issues))

            if review.disposition == "ACCEPT":
                stop_reason = "accepted"
                break

            if review.disposition in ("ESCALATE", "ABSTAIN"):
                stop_reason = "non_repairable_disposition"
                break

            # Stagnation check on subsequent RETRY attempts
            if (
                iteration >= 2
                and self.budget.stop_on_stagnant_issues
                and prev_issue_sig is not None
                and curr_issue_sig == prev_issue_sig
            ):
                stop_reason = "stagnant_issues"
                break
            prev_issue_sig = curr_issue_sig

            if iteration >= self.budget.max_iterations:
                stop_reason = "max_iterations_reached"
                break

            elapsed_ms = (time.monotonic() - t_cycle_start) * 1000.0
            if self.budget.max_latency_ms is not None and elapsed_ms > self.budget.max_latency_ms:
                stop_reason = "latency_budget_exceeded"
                break

            # Resolve concrete JustificationGraph to pass into repair agent
            if isinstance(current_graph_source, JustificationGraph):
                concrete_graph = current_graph_source
            elif isinstance(current_graph_source, Mapping):
                concrete_graph = JustificationGraph.from_response(
                    current_graph_source, vocabulary=self.domain_profile.vocabulary
                )
            else:
                concrete_graph = JustificationGraph(
                    claims={review.assessment.conclusion_id: review.assessment.conclusion},
                    steps=(),
                    conclusion_id=review.assessment.conclusion_id,
                )

            if hasattr(repair_agent, "repair") and callable(repair_agent.repair):
                repaired = repair_agent.repair(
                    question=q_norm,
                    candidate_response=current_response,
                    evidence=evidence,
                    graph=concrete_graph,
                    review=review,
                )
            elif callable(repair_agent):
                repaired = repair_agent(
                    question=q_norm,
                    candidate_response=current_response,
                    evidence=evidence,
                    graph=concrete_graph,
                    review=review,
                )
            else:
                raise TypeError("repair_agent must be callable or expose a .repair(...) method.")

            if isinstance(repaired, tuple) and len(repaired) == 2:
                next_response, next_graph = repaired
                current_graph_source = next_graph
            else:
                next_response = str(repaired)

            next_digest = compute_response_digest(next_response, question)
            if (
                self.budget.stop_on_duplicate_response
                and next_digest is not None
                and next_digest in seen_digests
            ):
                stop_reason = "duplicate_response"
                break

            current_response = next_response

        last_attempt = attempts[-1]
        final_review = last_attempt.review
        accepted = stop_reason == "accepted" and final_review.disposition == "ACCEPT"
        if accepted:
            final_disposition = "ACCEPT"
        elif final_review.disposition == "ABSTAIN":
            final_disposition = "ABSTAIN"
        else:
            # Never convert exhausted/stagnant/duplicate retries into ACCEPT
            final_disposition = "ESCALATE"

        total_ms = (time.monotonic() - t_cycle_start) * 1000.0
        return RepairCycleResult(
            final_disposition=final_disposition,
            accepted=accepted,
            stop_reason=stop_reason,
            question=q_norm,
            initial_response=initial_response,
            final_response=last_attempt.candidate_response,
            attempts=tuple(attempts),
            final_review=final_review,
            operational_authorization=final_review.operational_authorization,
            iterations_used=len(attempts),
            total_latency_ms=total_ms,
        )
