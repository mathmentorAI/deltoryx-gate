"""Public facade for live verification through the six Gate responsibility layers."""
from .plan_synthesis import (
    ClaimDerivationPlanner, DerivationPlanner, PlannerCallResult,
    DERIVATION_PLANNER_PROMPT, DERIVATION_PLANNER_PROMPT_VERSION,
    DEFAULT_PLANNER_BASE_URL, DEFAULT_PLANNER_MODEL, build_derivation_planner_model,
)
from .orchestration import ClaimVerifier, verify_claim
from .contracts import GateResult
from ._formal_utils import UNIT_PROFILE_CHOICES, detect_text_language, resolve_unit_profile

__all__ = ["ClaimDerivationPlanner", "DerivationPlanner", "PlannerCallResult", "ClaimVerifier",
           "verify_claim", "GateResult", "DERIVATION_PLANNER_PROMPT", "DERIVATION_PLANNER_PROMPT_VERSION",
           "DEFAULT_PLANNER_BASE_URL", "DEFAULT_PLANNER_MODEL", "build_derivation_planner_model",
           "UNIT_PROFILE_CHOICES", "detect_text_language", "resolve_unit_profile"]
