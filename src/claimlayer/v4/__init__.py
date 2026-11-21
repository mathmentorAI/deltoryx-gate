"""ClaimLayer v4 (experimental, 4.0.0a1): the model proposes, deterministic policy decides.

    from claimlayer.v4 import EvidenceLayer, Vocabulary, PredicateSpec, FieldSpec

v4 reuses the v3 source/reference/store contracts and the v2 epistemic table,
and replaces model verdicts with deterministic comparison over a declared
vocabulary. v3 (claimlayer.ClaimLayer) and v2 (claimlayer.legacy) are unchanged.
"""
from .authorization import (
    AuthorizationIssue,
    AuthorizationVerdict,
    ClauseRelation,
    authorize,
    authorize_operation,
    compute_execution_idempotency_key,
    execute_authorized_operation,
    get_execution_state,
    reconcile_authorized_execution,
    resolve_scope_and_precedence,
    verify_authorization_verdict,
)
from .correspondence import (
    EquationCheckResult,
    GraphRelationTuple,
    ResponseClauseClaim,
    ResponseRelationTuple,
    StructuredCondition,
    build_supported_graph_relations,
    compute_response_digest,
    verify_response_equations,
    verify_response_graph_correspondence,
)
from .corpus_obligations import (
    CorpusReferenceRegister,
    ReferenceDivergenceCollection,
    TargetedRereadConfig,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from .domain_profile import (
    DeclaredRule,
    DomainProfile,
    EmpiricalCalibration,
    PolicyConfiguration,
    RelationCompositionRule,
    SemanticEvaluatorConfig,
)
from .domains import (
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_VOCABULARY,
)
from .engine import (
    ENGINE_VERSION,
    TIER_ANNEX,
    TIER_EXTERNAL,
    TIER_PRIMARY,
    EvidenceLayer,
    UnitEvaluation,
    evaluate_unit_admissibility,
)
from .epistemics import DEFAULT_PROFILE, EXPLICIT, EpistemicProfile
from .evaluation import (
    EVALUATION_ASSESSMENTS,
    ResponseAssessment,
    VerificationBudget,
    VerificationCache,
    evaluate_response,
)
from .fidelity import (
    FieldEvidence,
    FidelityFinding,
    check_clause_boundary_completeness,
    evaluate_dual_readers,
    render_canonical_claim,
    verify_extraction_fidelity,
)
from .justification import (
    CLAIM_KINDS,
    EPISTEMIC_KIND_RANK,
    MANDATORY_OBLIGATION_FAMILIES,
    OBLIGATION_GENERATOR_VERSION,
    VERIFIER_FAMILIES,
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    VerificationObligation,
    effective_epistemic_kind,
    generate_mandatory_obligations,
    infer_conclusion_semantic_domain,
)
from .models import AcceptancePolicy, EvidenceUnit
from .operations import (
    CANONICAL_WEEKDAYS_EN,
    CANONICAL_WEEKDAYS_ES,
    DEFAULT_OPERATION_CATALOG,
    OperationCatalog,
    OperationOutput,
    OperationSpec,
    build_default_operation_catalog,
    compute_date_offset,
    compute_weekday_shift,
    normalize_day_offset,
    normalize_weekday,
)
from .paths import TransitiveRule
from .planner import (
    DERIVATION_PLANNER_PROMPT,
    DERIVATION_PLANNER_PROMPT_VERSION,
    CertaintyReport,
    ClaimDerivationPlanner,
    ClaimVerificationResult,
    ClaimVerifier,
    DerivationPlanner,
    OperationalRouting,
    PlannerCallResult,
    PlannerTelemetry,
    verify_claim,
    UNIT_PROFILE_CHOICES,
    DEFAULT_PLANNER_BASE_URL,
    DEFAULT_PLANNER_MODEL,
    build_derivation_planner_model,
    detect_text_language,
    resolve_unit_profile,
    verify_plan,
)
from .premise_grounding import (
    PremiseGroundingResult,
    diagnose_alternate_anchors,
    verify_claim_target_correspondence,
    verify_premise_grounding,
    verify_step_anchor_correspondence,
)
from .reader import (
    LLMProposalReader,
    ProposalReader,
    READER_PROMPT,
    REREAD_PROMPT,
    REREAD_PROMPT_VERSION,
    RecordedProposalReader,
    STRICT_PROMPT_VERSION,
    STRICT_READER_PROMPT,
)
from .repair import (
    LLMGraphProposalAdapter,
    LLMRepairAgent,
    RepairAttemptRecord,
    RepairBudget,
    RepairCycleResult,
    RepairOrchestrator,
    RequiredRepair,
    ReviewIssue,
    ReviewResult,
    RuleGuidedRepairAdapter,
    ScriptedRepairAgent,
    review_response,
    verify_review_freshness,
)
from .verifiers import (
    DEFAULT_VERIFIER_REGISTRY,
    VERIFIER_GUARANTEE_KINDS,
    VERIFIER_STATUSES,
    AggregationVerifier,
    ArithmeticVerifier,
    ComparisonVerifier,
    DocumentAuthorityVerifier,
    ExplicitLogicVerifier,
    OpenSemanticsVerifier,
    QuantityVerifier,
    RelationVerifier,
    TemporalVerifier,
    VerifierContext,
    VerifierRegistry,
    VerifierResult,
)
from .vocabulary import CONDITION_STATUSES, SPANISH_UNITS, SPANISH_UNIT_ALIASES, STANDARD_UNITS, UNIT_PROFILES, FieldSpec, PredicateSpec, Unit, UnitRegistry, UnmappedProposal, Vocabulary
from .autovocab import build_vocabulary, propose_vocabulary, with_entities
from .triage import LANES, draft_vocabulary, segment_text, triage, verify_draft
from ..semantic.models import EvidenceRef, SourceDocument, StructuredQuery
from ..semantic.adapters import OpenAICompatibleModel

__all__ = ['ENGINE_VERSION', 'EvidenceLayer', 'UnitEvaluation', 'evaluate_unit_admissibility',
           'Vocabulary', 'PredicateSpec', 'FieldSpec', 'Unit', 'UnitRegistry',
           'STANDARD_UNITS', 'SPANISH_UNITS', 'SPANISH_UNIT_ALIASES', 'UNIT_PROFILES', 'CONDITION_STATUSES', 'UnmappedProposal', 'EpistemicProfile', 'DEFAULT_PROFILE', 'EXPLICIT',
           'AcceptancePolicy', 'EvidenceUnit', 'TransitiveRule', 'LLMProposalReader', 'ProposalReader', 'RecordedProposalReader',
           'READER_PROMPT', 'STRICT_READER_PROMPT', 'STRICT_PROMPT_VERSION', 'REREAD_PROMPT', 'REREAD_PROMPT_VERSION', 'SourceDocument', 'EvidenceRef', 'StructuredQuery', 'OpenAICompatibleModel',
           'TIER_PRIMARY', 'TIER_ANNEX', 'TIER_EXTERNAL', 'propose_vocabulary', 'build_vocabulary', 'with_entities',
           'verify_draft', 'triage', 'segment_text', 'LANES', 'draft_vocabulary',
           'AuthorizationIssue', 'AuthorizationVerdict', 'ClauseRelation', 'authorize', 'authorize_operation',
           'compute_execution_idempotency_key', 'execute_authorized_operation', 'get_execution_state',
           'reconcile_authorized_execution', 'resolve_scope_and_precedence', 'verify_authorization_verdict',
           'DeclaredRule', 'DomainProfile', 'EmpiricalCalibration', 'PolicyConfiguration',
           'RelationCompositionRule', 'SemanticEvaluatorConfig',
           'EVALUATION_ASSESSMENTS', 'ResponseAssessment', 'VerificationBudget', 'VerificationCache', 'evaluate_response',
           'CLAIM_KINDS', 'EPISTEMIC_KIND_RANK', 'VERIFIER_FAMILIES', 'ClaimNode', 'InferenceStep',
           'JustificationGraph', 'VerificationObligation', 'effective_epistemic_kind',
           'DEFAULT_VERIFIER_REGISTRY', 'VERIFIER_GUARANTEE_KINDS', 'VERIFIER_STATUSES',
           'AggregationVerifier', 'ArithmeticVerifier', 'ComparisonVerifier', 'DocumentAuthorityVerifier',
           'ExplicitLogicVerifier', 'OpenSemanticsVerifier', 'QuantityVerifier', 'RelationVerifier',
           'TemporalVerifier', 'VerifierContext', 'VerifierRegistry', 'VerifierResult',
           'FieldEvidence', 'FidelityFinding', 'check_clause_boundary_completeness',
           'evaluate_dual_readers', 'render_canonical_claim', 'verify_extraction_fidelity', 'B2B_CONTRACTS_VOCABULARY',
           'B2B_DEFAULT_ACCEPTANCE_POLICY', 'SECURITIZATION_VOCABULARY', 'SECURITIZATION_ACCEPTANCE_POLICY',
           'EquationCheckResult', 'GraphRelationTuple', 'ResponseClauseClaim',
           'ResponseRelationTuple', 'StructuredCondition', 'build_supported_graph_relations',
           'compute_response_digest', 'verify_response_equations', 'verify_response_graph_correspondence',
           'MANDATORY_OBLIGATION_FAMILIES', 'OBLIGATION_GENERATOR_VERSION',
           'generate_mandatory_obligations', 'infer_conclusion_semantic_domain',
           'LLMGraphProposalAdapter', 'LLMRepairAgent', 'RepairAttemptRecord', 'RepairBudget',
           'RepairCycleResult', 'RepairOrchestrator', 'RequiredRepair', 'ReviewIssue',
           'ReviewResult', 'RuleGuidedRepairAdapter', 'ScriptedRepairAgent',
           'review_response', 'verify_review_freshness',
           'CorpusReferenceRegister', 'ReferenceDivergenceCollection', 'TargetedRereadConfig',
           'build_corpus_reference_register', 'collect_reference_divergences',
           'CANONICAL_WEEKDAYS_EN', 'CANONICAL_WEEKDAYS_ES', 'DEFAULT_OPERATION_CATALOG',
           'OperationCatalog', 'OperationOutput', 'OperationSpec', 'build_default_operation_catalog',
           'compute_date_offset', 'compute_weekday_shift', 'normalize_day_offset', 'normalize_weekday',
           'DERIVATION_PLANNER_PROMPT', 'DERIVATION_PLANNER_PROMPT_VERSION', 'CertaintyReport',
           'ClaimDerivationPlanner', 'ClaimVerificationResult', 'ClaimVerifier', 'DerivationPlanner',
           'OperationalRouting', 'PlannerCallResult', 'PlannerTelemetry', 'verify_claim', 'verify_plan', 'UNIT_PROFILE_CHOICES', 'DEFAULT_PLANNER_BASE_URL', 'DEFAULT_PLANNER_MODEL', 'build_derivation_planner_model', 'detect_text_language', 'resolve_unit_profile',
           'PremiseGroundingResult', 'diagnose_alternate_anchors', 'verify_claim_target_correspondence',
           'verify_premise_grounding', 'verify_step_anchor_correspondence']
