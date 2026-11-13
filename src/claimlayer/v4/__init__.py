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
    authorize_operation,
    resolve_scope_and_precedence,
)
from .domains import B2B_CONTRACTS_VOCABULARY, B2B_DEFAULT_ACCEPTANCE_POLICY
from .engine import ENGINE_VERSION, TIER_ANNEX, TIER_EXTERNAL, TIER_PRIMARY, EvidenceLayer
from .epistemics import DEFAULT_PROFILE, EXPLICIT, EpistemicProfile
from .fidelity import (
    FieldEvidence,
    FidelityFinding,
    check_clause_boundary_completeness,
    evaluate_dual_readers,
    verify_extraction_fidelity,
)
from .models import AcceptancePolicy, EvidenceUnit
from .paths import TransitiveRule
from .reader import LLMProposalReader, ProposalReader, READER_PROMPT
from .vocabulary import CONDITION_STATUSES, STANDARD_UNITS, FieldSpec, PredicateSpec, Unit, UnitRegistry, UnmappedProposal, Vocabulary
from .autovocab import build_vocabulary, propose_vocabulary, with_entities
from .triage import LANES, draft_vocabulary, segment_text, triage, verify_draft
from ..semantic.models import EvidenceRef, SourceDocument, StructuredQuery
from ..semantic.adapters import OpenAICompatibleModel

__all__ = ['ENGINE_VERSION', 'EvidenceLayer', 'Vocabulary', 'PredicateSpec', 'FieldSpec', 'Unit', 'UnitRegistry',
           'STANDARD_UNITS', 'CONDITION_STATUSES', 'UnmappedProposal', 'EpistemicProfile', 'DEFAULT_PROFILE', 'EXPLICIT',
           'AcceptancePolicy', 'EvidenceUnit', 'TransitiveRule', 'LLMProposalReader', 'ProposalReader',
           'READER_PROMPT', 'SourceDocument', 'EvidenceRef', 'StructuredQuery', 'OpenAICompatibleModel',
           'TIER_PRIMARY', 'TIER_ANNEX', 'TIER_EXTERNAL', 'propose_vocabulary', 'build_vocabulary', 'with_entities',
           'verify_draft', 'triage', 'segment_text', 'LANES', 'draft_vocabulary',
           'AuthorizationIssue', 'AuthorizationVerdict', 'ClauseRelation', 'authorize_operation',
           'resolve_scope_and_precedence', 'FieldEvidence', 'FidelityFinding', 'check_clause_boundary_completeness',
           'evaluate_dual_readers', 'verify_extraction_fidelity', 'B2B_CONTRACTS_VOCABULARY',
           'B2B_DEFAULT_ACCEPTANCE_POLICY']
