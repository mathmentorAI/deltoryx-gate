"""Versioned DomainProfile separating formal rules, policy parameters, and calibration (Phase 2).

Enforces strict separation between:
1. Formal domain specifications (vocabulary, unit registry, declared inference rules,
   authorized relation compositions).
2. Organizational policy parameters (allowed claim classes per context, inference
   discounts, authority ranks, known source dependencies, coverage/margin thresholds).
3. Empirical calibration (observed extractor / semantic evaluator reliabilities and
   optional empirical calibration bins). By default `calibration_bins=None`, so
   `calibrated_correctness_probability` MUST remain `None`.

Hard constraint: NO weight, authority rank, or reliability score in a DomainProfile
can ever override a refuted arithmetic check, incompatible unit, tampered span,
or violated formal rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Callable, Mapping

from ..semantic.models import canonical, digest
from .epistemics import DEFAULT_PROFILE, EpistemicProfile
from .justification import CLAIM_KINDS
from .vocabulary import STANDARD_UNITS, UnitRegistry, Vocabulary


@dataclass(frozen=True)
class DeclaredRule:
    """Versioned formal logic / derivation rule declared in the DomainProfile."""

    rule_id: str
    version: str = "1.0"
    description: str = ""
    kind: str = "implication"  # 'implication' | 'conjunction' | 'disjunction' | 'threshold' | 'exclusion'
    required_premises: tuple[dict[str, Any], ...] = ()
    conclusion_template: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string.")
        allowed_kinds = {"implication", "conjunction", "disjunction", "threshold", "exclusion"}
        if self.kind not in allowed_kinds:
            raise ValueError(f"Unsupported DeclaredRule kind {self.kind!r}; expected one of {sorted(allowed_kinds)}.")
        req_prem = tuple(dict(p) for p in (self.required_premises or ()))
        conc_tpl = dict(self.conclusion_template or {})
        params = dict(self.parameters or {})
        canonical({"req": req_prem, "conc": conc_tpl, "params": params})
        object.__setattr__(self, "required_premises", req_prem)
        object.__setattr__(self, "conclusion_template", conc_tpl)
        object.__setattr__(self, "parameters", params)

    def describe(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "version": self.version,
            "description": self.description,
            "kind": self.kind,
            "required_premises": list(self.required_premises),
            "conclusion_template": self.conclusion_template,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class RelationCompositionRule:
    """Authorized relation composition (left_predicate o right_predicate -> composed_predicate)."""

    rule_id: str
    left_predicate: str
    right_predicate: str
    composed_predicate: str
    version: str = "1.0"
    transitive: bool = False
    require_temporal_overlap: bool = True
    require_exact_bridge: bool = True

    def __post_init__(self) -> None:
        for attr in ("rule_id", "left_predicate", "right_predicate", "composed_predicate", "version"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{attr} must be a non-empty string.")

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def _default_inference_discounts() -> dict[str, float]:
    return {
        "deterministic_formal": 1.0,
        "relation_composition": 0.92,
        "probabilistic_model": 0.65,
    }


def _default_authority_ranks() -> dict[str, int]:
    return {
        "amendment": 100,
        "controlling_instrument": 95,
        "audited_record": 90,
        "master_agreement": 85,
        "schedule": 75,
        "policy": 70,
        "source_statement": 60,
        "application_record": 55,
        "memo": 40,
        "allegation": 10,
        "party_allegation": 10,
    }


@dataclass(frozen=True)
class PolicyConfiguration:
    """Organizational policy thresholds and evidence-handling preferences (NOT empirical truth)."""

    allowed_claim_kinds_for_evaluation: tuple[str, ...] = CLAIM_KINDS
    allowed_claim_kinds_for_authorization: tuple[str, ...] = ("extraction", "derivation")
    inference_discounts: dict[str, float] = field(default_factory=_default_inference_discounts)
    authority_ranks: dict[str, int] = field(default_factory=_default_authority_ranks)
    unverified_attestor_roles: tuple[str, ...] = (
        "party_allegation",
        "plaintiff",
        "defendant",
        "witness",
        "applicant_claim",
        "unaudited_projection",
        "unverified_user",
    )
    known_source_dependencies: dict[str, str] = field(default_factory=dict)
    excluded_comparison_roles: dict[str, str] = field(default_factory=dict)
    min_verification_coverage: float = 1.0
    block_critical_unverified_in_authorization: bool = True
    require_deterministic_critical_steps: bool = True
    tau: float = 0.70
    delta: float = 0.15
    max_conflict_index: float = 0.25
    max_uncertainty_index: float = 0.35
    # Safety lock: formally refuted obligations NEVER get waived by weights or policy.
    allow_weight_override_of_formal_error: bool = False

    def __post_init__(self) -> None:
        if self.allow_weight_override_of_formal_error:
            raise ValueError(
                "Invariant violation: no policy may allow weights to override formal errors "
                "(arithmetic, unit, span, or rule failures)."
            )
        eval_kinds = tuple(self.allowed_claim_kinds_for_evaluation)
        auth_kinds = tuple(self.allowed_claim_kinds_for_authorization)
        for kind in eval_kinds + auth_kinds:
            if kind not in CLAIM_KINDS:
                raise ValueError(f"Unknown claim kind {kind!r} in policy configuration.")

        discounts = {str(k): float(v) for k, v in (self.inference_discounts or _default_inference_discounts()).items()}
        for k, v in discounts.items():
            if not math.isfinite(v) or not 0.0 < v <= 1.0:
                raise ValueError(f"Inference discount for {k!r} must be in (0, 1], got {v}.")

        ranks = {str(k): int(v) for k, v in (self.authority_ranks or _default_authority_ranks()).items()}
        deps = {str(k): str(v) for k, v in (self.known_source_dependencies or {}).items()}
        excluded_roles: dict[str, str] = {}
        for k, v in (self.excluded_comparison_roles or {}).items():
            role_key = str(k).strip()
            reason = str(v).strip()
            if not role_key:
                raise ValueError("excluded_comparison_roles keys must be non-empty role names.")
            if not reason:
                raise ValueError(f"excluded_comparison_roles[{role_key!r}] must specify a non-empty rule/rationale.")
            excluded_roles[role_key] = reason
        unverified = tuple(str(r) for r in self.unverified_attestor_roles)

        for name, lo, hi in (
            ("min_verification_coverage", 0.0, 1.0),
            ("tau", 0.0, 1.0),
            ("delta", 0.0, 1.0),
            ("max_conflict_index", 0.0, 1.0),
            ("max_uncertainty_index", 0.0, 1.0),
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f"Policy field {name} must be finite in [{lo}, {hi}], got {val}.")

        object.__setattr__(self, "allowed_claim_kinds_for_evaluation", eval_kinds)
        object.__setattr__(self, "allowed_claim_kinds_for_authorization", auth_kinds)
        object.__setattr__(self, "inference_discounts", discounts)
        object.__setattr__(self, "authority_ranks", ranks)
        object.__setattr__(self, "unverified_attestor_roles", unverified)
        object.__setattr__(self, "known_source_dependencies", deps)
        object.__setattr__(self, "excluded_comparison_roles", excluded_roles)

    def describe(self) -> dict[str, Any]:
        return {
            "allowed_claim_kinds_for_evaluation": list(self.allowed_claim_kinds_for_evaluation),
            "allowed_claim_kinds_for_authorization": list(self.allowed_claim_kinds_for_authorization),
            "inference_discounts": dict(sorted(self.inference_discounts.items())),
            "authority_ranks": dict(sorted(self.authority_ranks.items())),
            "unverified_attestor_roles": list(self.unverified_attestor_roles),
            "known_source_dependencies": dict(sorted(self.known_source_dependencies.items())),
            "excluded_comparison_roles": dict(sorted(self.excluded_comparison_roles.items())),
            "min_verification_coverage": self.min_verification_coverage,
            "block_critical_unverified_in_authorization": self.block_critical_unverified_in_authorization,
            "require_deterministic_critical_steps": self.require_deterministic_critical_steps,
            "tau": self.tau,
            "delta": self.delta,
            "max_conflict_index": self.max_conflict_index,
            "max_uncertainty_index": self.max_uncertainty_index,
        }


@dataclass(frozen=True)
class EmpiricalCalibration:
    """Empirical reliability parameters separated from organizational policy.

    Unless an empirical calibrator (`calibration_bins`) calibrated on held-out data
    is supplied, `calibrate_correctness()` MUST return `None`.
    """

    extractor_reliabilities: dict[str, float] = field(default_factory=dict)
    verifier_reliabilities: dict[str, float] = field(default_factory=dict)
    default_extractor_reliability: float | None = None
    calibration_bins: tuple[tuple[float, float, float], ...] | None = None
    calibration_dataset_id: str | None = None

    def __post_init__(self) -> None:
        ext = {str(k): float(v) for k, v in (self.extractor_reliabilities or {}).items()}
        ver = {str(k): float(v) for k, v in (self.verifier_reliabilities or {}).items()}
        for label, d in (("extractor_reliabilities", ext), ("verifier_reliabilities", ver)):
            for k, v in d.items():
                if not math.isfinite(v) or not 0.0 <= v <= 1.0:
                    raise ValueError(f"{label}[{k!r}] must be in [0, 1], got {v}.")
        if self.default_extractor_reliability is not None:
            dv = float(self.default_extractor_reliability)
            if not math.isfinite(dv) or not 0.0 <= dv <= 1.0:
                raise ValueError("default_extractor_reliability must be in [0, 1] or None.")
        bins_norm: tuple[tuple[float, float, float], ...] | None = None
        if self.calibration_bins is not None:
            cleaned: list[tuple[float, float, float]] = []
            for lo, hi, prob in self.calibration_bins:
                flo, fhi, fprob = float(lo), float(hi), float(prob)
                if not (math.isfinite(flo) and math.isfinite(fhi) and math.isfinite(fprob)):
                    raise ValueError("Calibration bin values must be finite.")
                if not (0.0 <= flo <= fhi <= 1.0 and 0.0 <= fprob <= 1.0):
                    raise ValueError("Calibration bin bounds and probabilities must lie in [0, 1].")
                cleaned.append((flo, fhi, fprob))
            bins_norm = tuple(cleaned)
        object.__setattr__(self, "extractor_reliabilities", ext)
        object.__setattr__(self, "verifier_reliabilities", ver)
        object.__setattr__(self, "calibration_bins", bins_norm)

    def calibrate_correctness(self, support_index: float) -> float | None:
        """Return calibrated correctness probability ONLY if an empirical calibrator exists."""
        if not self.calibration_bins:
            return None
        for lo, hi, prob in self.calibration_bins:
            if lo <= support_index <= hi:
                return round(float(prob), 6)
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "extractor_reliabilities": dict(sorted(self.extractor_reliabilities.items())),
            "verifier_reliabilities": dict(sorted(self.verifier_reliabilities.items())),
            "default_extractor_reliability": self.default_extractor_reliability,
            "calibration_bins": [list(b) for b in self.calibration_bins] if self.calibration_bins else None,
            "calibration_dataset_id": self.calibration_dataset_id,
        }


@dataclass(frozen=True)
class SemanticEvaluatorConfig:
    """Configuration for optional open-semantics evaluator (always probabilistic_model)."""

    evaluator_id: str = "none"
    version: str = "0.0"
    enabled: bool = False
    callable_evaluator: Callable[..., Mapping[str, Any]] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "evaluator_id": self.evaluator_id,
            "version": self.version,
            "enabled": bool(self.enabled and self.callable_evaluator is not None),
        }


@dataclass(frozen=True)
class DomainProfile:
    """Versioned contract bundling schema, formal rules, policy, and empirical calibration."""

    profile_id: str
    version: str = "1.0.0"
    vocabulary: Vocabulary | None = None
    unit_registry: UnitRegistry | None = None
    epistemic_profile: EpistemicProfile = DEFAULT_PROFILE
    policy: PolicyConfiguration = field(default_factory=PolicyConfiguration)
    calibration: EmpiricalCalibration = field(default_factory=EmpiricalCalibration)
    declared_rules: dict[str, DeclaredRule] = field(default_factory=dict)
    relation_rules: dict[str, RelationCompositionRule] = field(default_factory=dict)
    semantic_evaluator: SemanticEvaluatorConfig = field(default_factory=SemanticEvaluatorConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must be a non-empty string.")

        rules_map = {str(k): v for k, v in (self.declared_rules or {}).items()}
        rel_map = {str(k): v for k, v in (self.relation_rules or {}).items()}
        for k, r in rules_map.items():
            if not isinstance(r, DeclaredRule):
                raise ValueError(f"declared_rules[{k!r}] must be a DeclaredRule.")
        for k, r in rel_map.items():
            if not isinstance(r, RelationCompositionRule):
                raise ValueError(f"relation_rules[{k!r}] must be a RelationCompositionRule.")

        object.__setattr__(self, "declared_rules", rules_map)
        object.__setattr__(self, "relation_rules", rel_map)
        self.validate_profile()

    def effective_unit_registry(self) -> UnitRegistry:
        if self.unit_registry is not None:
            return self.unit_registry
        if self.vocabulary is not None:
            return self.vocabulary.units
        return STANDARD_UNITS

    def validate_profile(self) -> None:
        if self.policy.allow_weight_override_of_formal_error:
            raise ValueError("Weights cannot override formal verifier errors.")

    def describe(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "vocabulary_fingerprint": self.vocabulary.fingerprint if self.vocabulary is not None else None,
            "units": self.effective_unit_registry().describe(),
            "epistemic_profile": self.epistemic_profile.fingerprint,
            "policy": self.policy.describe(),
            "calibration": self.calibration.describe(),
            "declared_rules": {k: self.declared_rules[k].describe() for k in sorted(self.declared_rules)},
            "relation_rules": {k: self.relation_rules[k].describe() for k in sorted(self.relation_rules)},
            "semantic_evaluator": self.semantic_evaluator.describe(),
        }

    def fingerprint(self) -> str:
        return digest(self.describe())
