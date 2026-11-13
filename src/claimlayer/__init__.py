"""ClaimLayer v3: Deterministic computable verification for AI agent outputs."""

from claimlayer.ledger import Ledger
from claimlayer.mcp_server import MCPVerificationServer
from claimlayer.models import (
    Assertion,
    ConflictDiff,
    FactRecord,
    Issue,
    Policy,
    ResolutionRecord,
    Source,
    Verdict,
    VerificationReceipt,
)
from claimlayer.receipt import RECEIPT_SCHEME, verify_receipt
from claimlayer import semantic, v4
from claimlayer.v4 import (
    AcceptancePolicy,
    EvidenceLayer,
    EvidenceUnit,
    FieldSpec,
    PredicateSpec,
    UnmappedProposal,
    Vocabulary,
    Unit,
    UnitRegistry,
    STANDARD_UNITS,
)
from claimlayer.verifiers.diff import (
    SectionDivergence,
    build_comparison_policy,
    extract_divergent_sections,
)
from claimlayer.verify import verify

__version__ = "3.2.1"

__all__ = [
    "AcceptancePolicy",
    "Assertion",
    "ConflictDiff",
    "EvidenceLayer",
    "EvidenceUnit",
    "FactRecord",
    "FieldSpec",
    "Issue",
    "Ledger",
    "MCPVerificationServer",
    "Policy",
    "PredicateSpec",
    "RECEIPT_SCHEME",
    "ResolutionRecord",
    "SectionDivergence",
    "Source",
    "UnmappedProposal",
    "Verdict",
    "VerificationReceipt",
    "Vocabulary",
    "Unit",
    "UnitRegistry",
    "STANDARD_UNITS",
    "__version__",
    "build_comparison_policy",
    "extract_divergent_sections",
    "semantic",
    "v4",
    "verify",
    "verify_receipt",
]
