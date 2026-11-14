"""Extensible Registry of 9 Reusable Inference-Verifier Families (Phase 2).

Every verifier implements a uniform contract returning a `VerifierResult` with:
- `status`: `'verified' | 'refuted' | 'unknown' | 'not_applicable'`
- `guarantee_kind`: `'deterministic_formal' | 'probabilistic_model'`
- `evidence`, `assumptions`, `reasons`, `verifier_id`, `verifier_version`

Important invariants:
1. Formal failures (`refuted` arithmetic, incompatible units, violated rules,
   contradicted dates, invalid document spans) can NEVER be waived by weights.
2. `open_semantics` evaluators MUST always carry `guarantee_kind='probabilistic_model'`
   and NEVER share the `'deterministic_formal'` guarantee label.
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..semantic.models import SourceDocument, canonical
from .domain_profile import DomainProfile
from .engine import EvidenceLayer, evaluate_unit_admissibility
from .justification import ClaimNode, InferenceStep, VERIFIER_FAMILIES, VerificationObligation
from .models import AcceptancePolicy
from .vocabulary import UnitRegistry, UnmappedProposal, label_key, normalize_date

VERIFIER_STATUSES: tuple[str, ...] = ("verified", "refuted", "unknown", "not_applicable")
VERIFIER_GUARANTEE_KINDS: tuple[str, ...] = ("deterministic_formal", "probabilistic_model")


@dataclass(frozen=True)
class VerifierResult:
    """Uniform structured result returned by all verifier families."""

    obligation_id: str
    step_id: str
    family: str
    status: str  # 'verified' | 'refuted' | 'unknown' | 'not_applicable'
    guarantee_kind: str  # 'deterministic_formal' | 'probabilistic_model'
    evidence: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    verifier_id: str = ""
    verifier_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.family not in VERIFIER_FAMILIES:
            raise ValueError(f"Unknown verifier family {self.family!r}.")
        if self.status not in VERIFIER_STATUSES:
            raise ValueError(f"Unknown verifier status {self.status!r}; expected {VERIFIER_STATUSES}.")
        if self.guarantee_kind not in VERIFIER_GUARANTEE_KINDS:
            raise ValueError(
                f"Unknown guarantee_kind {self.guarantee_kind!r}; expected {VERIFIER_GUARANTEE_KINDS}."
            )
        if self.family == "open_semantics" and self.guarantee_kind != "probabilistic_model":
            raise ValueError("open_semantics verifier must always carry guarantee_kind='probabilistic_model'.")
        clean_ev = dict(self.evidence or {})
        canonical(clean_ev)
        object.__setattr__(self, "evidence", clean_ev)
        object.__setattr__(self, "assumptions", tuple(str(a) for a in (self.assumptions or ())))
        object.__setattr__(self, "reasons", tuple(str(r) for r in (self.reasons or ())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "step_id": self.step_id,
            "family": self.family,
            "status": self.status,
            "guarantee_kind": self.guarantee_kind,
            "evidence": self.evidence,
            "assumptions": list(self.assumptions),
            "reasons": list(self.reasons),
            "verifier_id": self.verifier_id,
            "verifier_version": self.verifier_version,
        }

    @classmethod
    def from_dict(cls, data: VerifierResult | Mapping[str, Any]) -> "VerifierResult":
        if isinstance(data, cls):
            return data
        return cls(
            obligation_id=str(data["obligation_id"]),
            step_id=str(data["step_id"]),
            family=str(data["family"]),
            status=str(data["status"]),
            guarantee_kind=str(data["guarantee_kind"]),
            evidence=dict(data.get("evidence") or {}),
            assumptions=tuple(data.get("assumptions") or ()),
            reasons=tuple(data.get("reasons") or ()),
            verifier_id=str(data.get("verifier_id", "")),
            verifier_version=str(data.get("verifier_version", "1.0.0")),
        )


@dataclass(frozen=True)
class VerifierContext:
    """Execution context supplied to a verifier when checking one obligation."""

    step: InferenceStep
    conclusion: ClaimNode
    premises: tuple[ClaimNode, ...]
    domain_profile: DomainProfile
    documents: Mapping[str, SourceDocument] = field(default_factory=dict)
    layer: EvidenceLayer | None = None
    clause_relations: tuple[Any, ...] = ()
    as_of: str | None = None
    allow_semantic_call: bool = True
    question: str | Mapping[str, Any] | None = None
    candidate_response: str | None = None
    correspondence_facts: Mapping[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    family: str
    verifier_id: str
    version: str
    guarantee_kind: str

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        ...


# ---------------------------------------------------------------------------
# Shared extraction / quantity / date helpers
# ---------------------------------------------------------------------------

def _to_decimal(val: Any) -> Decimal | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, Decimal):
        return val if val.is_finite() else None
    if isinstance(val, (int, float, str)):
        try:
            d = Decimal(str(val).strip())
            return d if d.is_finite() else None
        except (InvalidOperation, ValueError):
            return None
    return None


def _extract_quantity_pair(node_or_val: ClaimNode | Mapping[str, Any] | Any, default_unit: str | None = None) -> tuple[Decimal | None, str | None]:
    if isinstance(node_or_val, ClaimNode):
        val = node_or_val.value
        unit = node_or_val.unit
    elif isinstance(node_or_val, Mapping):
        if "value" in node_or_val and ("unit" in node_or_val or default_unit is not None):
            val = node_or_val.get("value")
            unit = node_or_val.get("unit", default_unit)
        else:
            return None, default_unit
    else:
        return _to_decimal(node_or_val), default_unit

    if isinstance(val, Mapping) and "value" in val:
        return _to_decimal(val.get("value")), val.get("unit") or unit
    return _to_decimal(val), unit


def _convert_to_base_decimal(
    value: Decimal,
    unit_label: str | None,
    registry: UnitRegistry,
) -> tuple[Decimal, str | None, str | None]:
    """Return `(value_in_base, dimension, base_unit_name)` using exact UnitRegistry factors."""
    if not unit_label:
        return value, None, None
    unit_obj = registry.lookup(unit_label)
    base_val = value * Decimal(unit_obj.factor)
    return base_val, unit_obj.dimension, registry.base(unit_obj.dimension)


def _extract_interval(val: Any) -> tuple[Decimal, Decimal] | None:
    """Return closed interval `(lo, hi)` if value represents a number or `[min, max]` range."""
    num = _to_decimal(val)
    if num is not None:
        return num, num
    if isinstance(val, Mapping):
        if "min" in val and "max" in val:
            lo = _to_decimal(val["min"])
            hi = _to_decimal(val["max"])
            if lo is not None and hi is not None and lo <= hi:
                return lo, hi
        if "low" in val and "high" in val:
            lo = _to_decimal(val["low"])
            hi = _to_decimal(val["high"])
            if lo is not None and hi is not None and lo <= hi:
                return lo, hi
        if "value" in val:
            return _extract_interval(val["value"])
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo = _to_decimal(val[0])
        hi = _to_decimal(val[1])
        if lo is not None and hi is not None and lo <= hi:
            return lo, hi
    return None


def _parse_date_bounds(raw: Any) -> tuple[date, date, str] | None:
    """Return `(earliest_date, latest_date, precision)` for YYYY, YYYY-MM, YYYY-MM-DD or ISO datetime."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        iso = normalize_date(raw)
    except ValueError:
        return None
    if re.fullmatch(r"\d{4}", iso):
        y = int(iso)
        return date(y, 1, 1), date(y, 12, 31), "year"
    if re.fullmatch(r"\d{4}-\d{2}", iso):
        y, m = int(iso[:4]), int(iso[5:7])
        last_d = monthrange(y, m)[1]
        return date(y, m, 1), date(y, m, last_d), "month"
    d_part = date.fromisoformat(iso[:10])
    return d_part, d_part, "day"


def _completed_years(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def _completed_months(start: date, end: date) -> int:
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return months


# ---------------------------------------------------------------------------
# Family 1: arithmetic
# ---------------------------------------------------------------------------

class ArithmeticVerifier:
    """Exact Decimal arithmetic verifier with unit-aligned operands and explicit rounding/tolerance."""

    family = "arithmetic"
    verifier_id = "deterministic_decimal_arithmetic"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    _SUPPORTED_OPS = {
        "add",
        "sum",
        "sub",
        "diff",
        "mul",
        "prod",
        "div",
        "ratio",
        "pct",
        "percentage_of",
        "round",
        "linear_combination",
        "arithmetic",
    }

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        op = str(params.get("arithmetic_op") or params.get("op") or context.step.operation).lower().strip()
        if op not in self._SUPPORTED_OPS:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="not_applicable",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNSUPPORTED_ARITHMETIC_OP:{op}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        if op == "arithmetic":
            op = str(params.get("sub_op", "add")).lower().strip()

        registry = context.domain_profile.effective_unit_registry()
        premises = context.premises
        if not premises:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("MISSING_ARITHMETIC_PREMISES",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        pairs = [_extract_quantity_pair(p) for p in premises]
        if any(val is None for val, _ in pairs):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("NON_NUMERIC_ARITHMETIC_OPERAND",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        conc_val, conc_unit = _extract_quantity_pair(context.conclusion)
        if conc_val is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("NON_NUMERIC_CONCLUSION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Convert premise operands via UnitRegistry where dimensional conversion applies.
        converted_vals: list[Decimal] = []
        dims: list[str | None] = []
        for raw_v, raw_u in pairs:
            assert raw_v is not None
            try:
                base_v, dim, _ = _convert_to_base_decimal(raw_v, raw_u, registry)
            except UnmappedProposal as exc:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=(f"UNKNOWN_UNIT:{exc.detail}",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            converted_vals.append(base_v)
            dims.append(dim)

        assumptions: list[str] = list(context.step.assumptions)
        expected_base: Decimal
        expected_dim: str | None = None

        if op in {"add", "sum", "sub", "diff", "linear_combination"}:
            non_null_dims = {d for d in dims if d is not None}
            if len(non_null_dims) > 1 or (len(non_null_dims) == 1 and any(d is None for d in dims)):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"premise_dimensions": dims},
                    reasons=("ARITHMETIC_INCOMPATIBLE_UNIT_DIMENSIONS",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            expected_dim = next(iter(non_null_dims), None)
            if op in {"add", "sum"}:
                expected_base = sum(converted_vals, Decimal(0))
            elif op in {"sub", "diff"}:
                if len(converted_vals) < 2:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="unknown",
                        guarantee_kind=self.guarantee_kind,
                        reasons=("SUBTRACTION_REQUIRES_AT_LEAST_TWO_OPERANDS",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
                expected_base = converted_vals[0] - sum(converted_vals[1:], Decimal(0))
            else:
                coeffs = params.get("coefficients")
                if not isinstance(coeffs, Sequence) or len(coeffs) != len(converted_vals):
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="unknown",
                        guarantee_kind=self.guarantee_kind,
                        reasons=("INVALID_LINEAR_COMBINATION_COEFFICIENTS",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
                expected_base = sum(
                    (_to_decimal(c) or Decimal(0)) * v for c, v in zip(coeffs, converted_vals)
                )
        elif op in {"mul", "prod"}:
            expected_base = Decimal(1)
            for v, _ in pairs:
                assert v is not None
                expected_base *= v
            raw_dim_set = {d for d in dims if d is not None and d != "ratio"}
            if len(raw_dim_set) == 1 and any(d == "ratio" for d in dims):
                expected_dim = next(iter(raw_dim_set))
                # If multiplying quantity by percent/fraction, apply unit factors properly.
                expected_base = Decimal(1)
                for (raw_v, raw_u), dim in zip(pairs, dims):
                    assert raw_v is not None
                    if dim == "ratio" and raw_u:
                        u_obj = registry.lookup(raw_u)
                        if u_obj.name == "percent":
                            expected_base *= raw_v / Decimal(100)
                        else:
                            expected_base *= raw_v * Decimal(u_obj.factor) / Decimal(100)
                    else:
                        conv_v, _, _ = _convert_to_base_decimal(raw_v, raw_u, registry)
                        expected_base *= conv_v
        elif op in {"div", "ratio"}:
            if len(pairs) != 2:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("DIVISION_REQUIRES_TWO_OPERANDS",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            num_raw, num_u = pairs[0]
            den_raw, den_u = pairs[1]
            assert num_raw is not None and den_raw is not None
            if den_raw == 0:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("DIVISION_BY_ZERO",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            if dims[0] is not None and dims[0] == dims[1]:
                # Same dimension ratio yields dimensionless fraction -> convert to percent if conc is percent.
                expected_base = converted_vals[0] / converted_vals[1]
                expected_dim = "ratio" if conc_unit else None
                if conc_unit:
                    u_obj = registry.lookup(conc_unit)
                    if u_obj.dimension == "ratio":
                        # To express fraction in UnitRegistry 'ratio' base (percent), scale by 100
                        expected_base = expected_base * Decimal(100)
            else:
                expected_base = num_raw / den_raw
        elif op in {"pct", "percentage_of"}:
            if len(pairs) != 2:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("PERCENTAGE_REQUIRES_BASE_AND_RATE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            base_v, base_u = pairs[0]
            rate_v, rate_u = pairs[1]
            assert base_v is not None and rate_v is not None
            conv_base, base_dim, _ = _convert_to_base_decimal(base_v, base_u, registry)
            expected_dim = base_dim
            if rate_u:
                r_unit = registry.lookup(rate_u)
                if r_unit.name == "fraction":
                    rate_factor = rate_v
                elif r_unit.dimension == "ratio":
                    rate_factor = rate_v / Decimal(100)
                elif r_unit.dimension == "ratio_difference":
                    rate_factor = (rate_v * Decimal(r_unit.factor)) / Decimal(100)
                else:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="refuted",
                        guarantee_kind=self.guarantee_kind,
                        reasons=("PERCENTAGE_RATE_INVALID_UNIT",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
            else:
                rate_factor = rate_v if params.get("rate_is_fraction") else rate_v / Decimal(100)
            expected_base = conv_base * rate_factor
        else:  # round
            expected_base = converted_vals[0]
            expected_dim = dims[0]

        # Convert conclusion to same comparison scale
        try:
            conc_base, conc_dim, _ = _convert_to_base_decimal(conc_val, conc_unit, registry)
        except UnmappedProposal as exc:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNKNOWN_CONCLUSION_UNIT:{exc.detail}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if expected_dim is not None and conc_dim is not None and expected_dim != conc_dim:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"expected_dimension": expected_dim, "conclusion_dimension": conc_dim},
                reasons=("ARITHMETIC_CONCLUSION_UNIT_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        compare_conc = conc_base if (expected_dim is not None and conc_dim is not None) else conc_val
        if "decimals" in params:
            decimals = int(params["decimals"])
            quant = Decimal(10) ** (-decimals)
            expected_cmp = expected_base.quantize(quant, rounding=ROUND_HALF_UP)
            assumptions.append(f"rounding_half_up_decimals:{decimals}")
        else:
            expected_cmp = expected_base

        tol = _to_decimal(params.get("rounding_tolerance", params.get("tolerance", "0")))
        if tol is None or tol < 0:
            tol = Decimal(0)

        diff = abs(expected_cmp - compare_conc)
        if diff <= tol:
            if diff > 0:
                assumptions.append(f"within_declared_tolerance:{str(tol)}")
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence={
                    "expected": str(expected_cmp),
                    "proposed": str(compare_conc),
                    "difference": str(diff),
                    "tolerance": str(tol),
                },
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("ARITHMETIC_VERIFIED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="refuted",
            guarantee_kind=self.guarantee_kind,
            evidence={
                "expected": str(expected_cmp),
                "proposed": str(compare_conc),
                "difference": str(diff),
                "tolerance": str(tol),
            },
            assumptions=tuple(dict.fromkeys(assumptions)),
            reasons=("ARITHMETIC_MISMATCH",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 2: quantities
# ---------------------------------------------------------------------------

class QuantityVerifier:
    """Dimensional compatibility and exact UnitRegistry conversion verifier."""

    family = "quantities"
    verifier_id = "deterministic_unit_registry"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        registry = context.domain_profile.effective_unit_registry()
        premises = context.premises
        conc = context.conclusion
        params = {**context.step.parameters, **obligation.parameters}

        conc_val, conc_unit = _extract_quantity_pair(conc)
        premise_pairs = [_extract_quantity_pair(p) for p in premises]

        # Validate all declared units exist in UnitRegistry
        resolved_premises: list[tuple[Decimal | None, str | None, str | None]] = []
        for idx, (pval, punit) in enumerate(premise_pairs):
            if punit is not None:
                try:
                    p_uobj = registry.lookup(punit)
                    resolved_premises.append((pval, p_uobj.name, p_uobj.dimension))
                except UnmappedProposal:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="unknown",
                        guarantee_kind=self.guarantee_kind,
                        evidence={"premise_index": idx, "unit": punit},
                        reasons=(f"UNKNOWN_PREMISE_UNIT:{punit}",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
            else:
                resolved_premises.append((pval, None, None))

        conc_dim: str | None = None
        conc_uname: str | None = None
        if conc_unit is not None:
            try:
                c_uobj = registry.lookup(conc_unit)
                conc_uname = c_uobj.name
                conc_dim = c_uobj.dimension
            except UnmappedProposal:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"unit": conc_unit},
                    reasons=(f"UNKNOWN_CONCLUSION_UNIT:{conc_unit}",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        # For explicit conversion or same-dimension comparison/addition/aggregation checks
        op = context.step.operation.lower()
        is_mandatory_qty = params.get("mandatory_check") == "quantity_precision_and_units"
        temporal_age_mode = bool(params.get("temporal_age_mode", False))
        additive_or_conversion = {
            "unit_conversion",
            "quantities",
            "add",
            "sum",
            "sub",
            "diff",
            "comparison",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "between",
            "within_bounds",
            "exceeds_cap",
            "below_floor",
            "aggregation",
            "agg_sum",
            "agg_average",
            "agg_mean",
            "agg_min",
            "agg_max",
        }
        if op not in additive_or_conversion or (is_mandatory_qty and temporal_age_mode):
            # Multiplicative/ratio ops and date-based age computations still validate that declared units are valid.
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence={
                    "premise_dimensions": [d for _, _, d in resolved_premises],
                    "conclusion_dimension": conc_dim,
                },
                reasons=("UNITS_VALID_IN_REGISTRY",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        p_dims = {d for _, _, d in resolved_premises if d is not None}
        if len(p_dims) > 1 or (len(p_dims) == 1 and any(d is None for _, _, d in resolved_premises)):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"premise_dimensions": [d for _, _, d in resolved_premises]},
                reasons=("UNIT_DIMENSION_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        prem_dim = next(iter(p_dims), None)
        if prem_dim is not None and conc_dim is not None and prem_dim != conc_dim:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"premise_dimension": prem_dim, "conclusion_dimension": conc_dim},
                reasons=("UNIT_DIMENSION_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if is_mandatory_qty and conc.approximate and params.get("rounding_tolerance") is None and params.get("tolerance") is None and conc.approx_tolerance is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={"dimension": prem_dim or conc_dim, "unit": conc_uname},
                reasons=("UNSPECIFIED_QUANTITY_ROUNDING_OR_TOLERANCE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if op in {"unit_conversion", "quantities"} and resolved_premises and resolved_premises[0][0] is not None and conc_val is not None:
            p_val, p_uname, _ = resolved_premises[0]
            assert p_val is not None
            p_factor = Decimal(registry.lookup(p_uname).factor) if p_uname else Decimal(1)
            c_factor = Decimal(registry.lookup(conc_uname).factor) if conc_uname else Decimal(1)
            converted_base_p = p_val * p_factor
            converted_base_c = conc_val * c_factor
            tol = _to_decimal(params.get("tolerance", "0")) or Decimal(0)
            if abs(converted_base_p - converted_base_c) > tol:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={
                        "premise_base_value": str(converted_base_p),
                        "conclusion_base_value": str(converted_base_c),
                        "dimension": prem_dim or conc_dim,
                    },
                    reasons=("UNIT_CONVERSION_VALUE_MISMATCH",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            evidence={"dimension": prem_dim or conc_dim},
            reasons=("UNIT_COMPATIBILITY_VERIFIED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 3: temporal
# ---------------------------------------------------------------------------

def _safe_date_shift_year(d: date, years_delta: int) -> date:
    """Shift calendar date by years_delta safely handling Feb 29."""
    target_year = d.year + years_delta
    try:
        return d.replace(year=target_year)
    except ValueError:
        return date(target_year, d.month, 28)


class TemporalVerifier:
    """Dates, Allen interval relations, derived ages and durations with granularity uncertainty."""

    family = "temporal"
    verifier_id = "deterministic_temporal_interval"
    version = "1.1.0"
    guarantee_kind = "deterministic_formal"

    def _extract_start_end_bounds(
        self, obligation: VerificationObligation, context: VerifierContext
    ) -> tuple[tuple[date, date, str] | None, tuple[date, date, str] | None, int | None, tuple[date, date, str] | None]:
        params = {**context.step.parameters, **obligation.parameters}
        anchor_age_raw = params.get("anchor_age")
        anchor_age: int | None = None
        if anchor_age_raw is not None:
            dec_a = _to_decimal(anchor_age_raw)
            if dec_a is not None and dec_a >= 0 and dec_a == int(dec_a):
                anchor_age = int(dec_a)

        anchor_date_raw = params.get("anchor_date")
        start_raw = params.get("start_date") or params.get("birth_date")
        end_raw = params.get("end_date") or params.get("reference_date") or context.as_of

        # Attempt to discover anchor_age + anchor_date from premises if explicit params omitted
        if anchor_age is None and start_raw is None:
            for p in context.premises:
                pk = label_key(p.predicate).replace(" ", "_") if p.predicate else ""
                if pk in {"age", "age_at_anchor", "anchor_age", "edad"} or (
                    p.unit and label_key(p.unit) in {"year", "years", "año", "años"} and p.valid_from
                ):
                    dec_p = _to_decimal(p.value)
                    if dec_p is not None and dec_p >= 0 and dec_p == int(dec_p):
                        anchor_age = int(dec_p)
                        if anchor_date_raw is None:
                            anchor_date_raw = p.valid_from or p.valid_until
                        break

        if anchor_age is not None and anchor_date_raw is None:
            for p in context.premises:
                b = _parse_date_bounds(p.value if isinstance(p.value, str) else str(p.value) if isinstance(p.value, int) and 1500 <= p.value <= 2150 else p.valid_from)
                if b is not None and str(p.value) != str(end_raw):
                    anchor_date_raw = p.value if isinstance(p.value, str) else str(p.value) if isinstance(p.value, int) else p.valid_from
                    break

        if anchor_age is not None:
            anchor_b = _parse_date_bounds(anchor_date_raw)
            if end_raw is None and len(context.premises) >= 2:
                for p in reversed(context.premises):
                    if p.predicate not in {"age", "age_at_anchor", "anchor_age", "edad"}:
                        cand_e = p.value if isinstance(p.value, str) else str(p.value) if isinstance(p.value, int) and 1500 <= p.value <= 2150 else p.valid_from
                        if cand_e and str(cand_e) != str(anchor_date_raw):
                            end_raw = cand_e
                            break
            end_b = _parse_date_bounds(str(end_raw) if isinstance(end_raw, int) else end_raw)
            if anchor_b is not None:
                a0_min, a0_max, a_prec = anchor_b
                # On a0_min, completed age == anchor_age requires birth after (a0_min - (anchor_age+1) years)
                birth_min = _safe_date_shift_year(a0_min, -(anchor_age + 1)) + timedelta(days=1)
                # On a0_max, completed age >= anchor_age allows birth up to (a0_max - anchor_age years)
                birth_max = _safe_date_shift_year(a0_max, -anchor_age)
                return (birth_min, birth_max, f"anchor_{a_prec}"), end_b, anchor_age, anchor_b

        if start_raw is None and len(context.premises) >= 1:
            p0 = context.premises[0]
            start_raw = p0.value if isinstance(p0.value, str) else p0.valid_from
        if end_raw is None and len(context.premises) >= 2:
            p1 = context.premises[1]
            end_raw = p1.value if isinstance(p1.value, str) else p1.valid_from

        return _parse_date_bounds(start_raw), _parse_date_bounds(end_raw), None, None

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        op = str(params.get("temporal_op") or params.get("op") or context.step.operation).lower().strip()

        if op in {"precedes", "meets", "overlaps", "during", "contains", "temporal_order"}:
            return self._verify_allen_relation(obligation, context, op, params)

        # Derived age or duration (including mandatory::age_and_duration checks even when step uses op="add")
        if op in {"age", "temporal_age", "duration", "temporal_duration", "temporal"} or params.get("mandatory_check") == "temporal_precision":
            return self._verify_age_or_duration(obligation, context, op, params)

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="not_applicable",
            guarantee_kind=self.guarantee_kind,
            reasons=(f"UNSUPPORTED_TEMPORAL_OP:{op}",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_age_or_duration(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
        op: str,
        params: Mapping[str, Any],
    ) -> VerifierResult:
        start_b, end_b, anchor_age, anchor_b = self._extract_start_end_bounds(obligation, context)
        if start_b is None or end_b is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("MISSING_OR_UNPARSEABLE_DATES",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        s_min, s_max, s_prec = start_b
        e_min, e_max, e_prec = end_b
        if anchor_b is not None and anchor_b[0] > e_max:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"anchor_bounds": [anchor_b[0].isoformat(), anchor_b[1].isoformat()], "end_bounds": [e_min.isoformat(), e_max.isoformat()]},
                reasons=("ANCHOR_DATE_AFTER_REFERENCE_DATE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        if s_min > e_max:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"start_bounds": [s_min.isoformat(), s_max.isoformat()], "end_bounds": [e_min.isoformat(), e_max.isoformat()]},
                reasons=("START_DATE_AFTER_REFERENCE_DATE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        unit_hint = str(params.get("duration_unit") or params.get("metric") or context.conclusion.unit or "years").lower()
        assumptions = list(context.step.assumptions)
        if s_prec != "day" or e_prec != "day":
            assumptions.append(f"coarse_date_granularity:{s_prec}->{e_prec}")

        if unit_hint in {"month", "months", "mes", "meses"} and op not in {"age", "temporal_age"}:
            lo_val = Decimal(_completed_months(s_max, e_min))
            hi_val = Decimal(_completed_months(s_min, e_max))
            metric = "months"
        elif unit_hint in {"day", "days", "día", "días", "dia", "dias"} and op not in {"age", "temporal_age"}:
            lo_days = (e_min - s_max).days
            hi_days = (e_max - s_min).days
            lo_val = Decimal(max(0, lo_days))
            hi_val = Decimal(max(0, hi_days))
            metric = "days"
        elif op in {"age", "temporal_age"} or params.get("mandatory_check") == "temporal_precision" or unit_hint in {"year", "years", "año", "años"}:
            lo_val = Decimal(_completed_years(s_max, e_min))
            hi_val = Decimal(_completed_years(s_min, e_max))
            metric = "years"
        else:
            lo_days = (e_min - s_max).days
            hi_days = (e_max - s_min).days
            lo_val = Decimal(max(0, lo_days))
            hi_val = Decimal(max(0, hi_days))
            metric = "days"

        conc = context.conclusion
        conc_interval = _extract_interval(conc.value)
        if conc_interval is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("CONCLUSION_NOT_NUMERIC_OR_INTERVAL",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        raw_lo, raw_hi = conc_interval
        approx_tol = _to_decimal(
            params.get("approx_tolerance")
            if params.get("approx_tolerance") is not None
            else conc.approx_tolerance
        )

        # Incorporate natural-language response precision state when correspondence check runs
        corr = context.correspondence_facts or {}
        resp_approx_contradicted = bool(corr.get("contradictory_precision_qualification", False))
        resp_approx_present = corr.get("response_approximated_or_ranged")
        is_approx = bool(conc.approximate or approx_tol is not None or params.get("approximate", False))
        if corr.get("response_checked") and not resp_approx_present:
            # Even if graph claimed `approximate=True`, an unqualified exact statement in text fails exactness check
            is_approx = False
        elif corr.get("response_checked") and resp_approx_present and not resp_approx_contradicted:
            is_approx = True

        # Check explicit numeric/interval discrepancy BEFORE applying any approximation tolerance:
        # "Aproximadamente 110" NEVER passes when the true derived interval is [105, 107]!
        ev_base = {
            "computed_interval": [str(lo_val), str(hi_val)],
            "proposed_interval": [str(raw_lo), str(raw_hi)],
            "metric": metric,
            "start_precision": s_prec,
            "end_precision": e_prec,
        }
        if anchor_age is not None and anchor_b is not None:
            ev_base["anchor_age"] = anchor_age
            ev_base["anchor_bounds"] = [anchor_b[0].isoformat(), anchor_b[1].isoformat()]

        if resp_approx_contradicted:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence=ev_base,
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("CONTRADICTORY_PRECISION_QUALIFICATION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if raw_hi < lo_val or raw_lo > hi_val:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence=ev_base,
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("TEMPORAL_DERIVATION_CONTRADICTED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Effective interval when explicit approx_tolerance is declared vs derived granularity interval
        if approx_tol is not None and raw_lo == raw_hi:
            p_lo = raw_lo - approx_tol
            p_hi = raw_hi + approx_tol
            assumptions.append(f"declared_approx_tolerance:{str(approx_tol)}")
            ev_data = {**ev_base, "effective_interval": [str(p_lo), str(p_hi)]}
            if p_lo <= lo_val and hi_val <= p_hi:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="verified",
                    guarantee_kind=self.guarantee_kind,
                    evidence=ev_data,
                    assumptions=tuple(dict.fromkeys(assumptions)),
                    reasons=("TEMPORAL_DERIVATION_VERIFIED",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence=ev_data,
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("APPROXIMATION_HIDES_WIDE_UNCERTAINTY",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Case 1: Exact dates (lo_val == hi_val) or explicit interval [raw_lo, raw_hi] covering [lo_val, hi_val]
        if raw_lo <= lo_val and hi_val <= raw_hi:
            if corr.get("response_checked") and corr.get("response_claims_exact_unqualified") and lo_val < hi_val:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence=ev_base,
                    assumptions=tuple(dict.fromkeys(assumptions)),
                    reasons=("TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE", "UNJUSTIFIED_TEMPORAL_PRECISION"),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence=ev_base,
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("TEMPORAL_DERIVATION_VERIFIED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Case 2: Qualified approximation around nominal point inside derived bounds [lo_val, hi_val]
        # Bounded by the uncertainty derived from the source date granularities (no blanket universal +/-1!).
        derived_span = hi_val - lo_val
        coarse_endpoints = (1 if s_prec != "day" else 0) + (1 if e_prec != "day" else 0)
        max_granularity_span = Decimal(max(1, coarse_endpoints)) if (anchor_b is None) else Decimal(max(2, coarse_endpoints + 1))
        if is_approx and raw_lo == raw_hi and lo_val <= raw_lo <= hi_val:
            if derived_span <= max_granularity_span:
                assumptions.append(f"derived_temporal_interval:[{lo_val},{hi_val}]")
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="verified",
                    guarantee_kind=self.guarantee_kind,
                    evidence={**ev_base, "derived_span": str(derived_span), "max_granularity_span": str(max_granularity_span)},
                    assumptions=tuple(dict.fromkeys(assumptions)),
                    reasons=("TEMPORAL_APPROXIMATION_WITHIN_DERIVED_BOUNDS",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={**ev_base, "derived_span": str(derived_span), "max_granularity_span": str(max_granularity_span)},
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("APPROXIMATION_HIDES_WIDE_UNCERTAINTY",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Case 3: Unqualified point estimate with insufficient date precision (e.g. "Tiene 106 años" without full dates)
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="unknown",
            guarantee_kind=self.guarantee_kind,
            evidence=ev_base,
            assumptions=tuple(dict.fromkeys(assumptions)),
            reasons=("TEMPORAL_PRECISION_INSUFFICIENT_FOR_EXACT_VALUE", "UNJUSTIFIED_TEMPORAL_PRECISION"),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_allen_relation(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
        op: str,
        params: Mapping[str, Any],
    ) -> VerifierResult:
        if len(context.premises) < 2:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("ALLEN_RELATION_REQUIRES_TWO_INTERVALS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        p0, p1 = context.premises[0], context.premises[1]
        a_start = _parse_date_bounds(p0.valid_from or (p0.value.get("start") if isinstance(p0.value, Mapping) else None))
        a_end = _parse_date_bounds(p0.valid_until or (p0.value.get("end") if isinstance(p0.value, Mapping) else None))
        b_start = _parse_date_bounds(p1.valid_from or (p1.value.get("start") if isinstance(p1.value, Mapping) else None))
        b_end = _parse_date_bounds(p1.valid_until or (p1.value.get("end") if isinstance(p1.value, Mapping) else None))
        if not (a_start and a_end and b_start and b_end):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("INCOMPLETE_TEMPORAL_INTERVALS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        rel = str(params.get("relation") or op).lower()
        holds: bool | None = None
        if rel in {"precedes", "temporal_order"}:
            if a_end[1] < b_start[0]:
                holds = True
            elif a_end[0] >= b_start[1]:
                holds = False
        elif rel == "overlaps":
            if a_end[0] >= b_start[1] and b_end[0] >= a_start[1]:
                holds = True
            elif a_end[1] < b_start[0] or b_end[1] < a_start[0]:
                holds = False

        if holds is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("ALLEN_RELATION_AMBIGUOUS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        expected_bool = context.conclusion.value if isinstance(context.conclusion.value, bool) else True
        status = "verified" if holds == expected_bool else "refuted"
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status=status,
            guarantee_kind=self.guarantee_kind,
            evidence={"relation": rel, "holds": holds, "expected": expected_bool},
            reasons=("TEMPORAL_INTERVAL_VERIFIED" if status == "verified" else "TEMPORAL_INTERVAL_CONTRADICTED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 4: comparison
# ---------------------------------------------------------------------------

class ComparisonVerifier:
    """Typed equality, inequality, bound, and overlapping-interval comparison verifier."""

    family = "comparison"
    verifier_id = "deterministic_typed_comparison"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        op = str(params.get("comparison_op") or params.get("op") or context.step.operation).lower().strip()
        if op == "comparison":
            op = str(params.get("sub_op", "le")).lower().strip()

        registry = context.domain_profile.effective_unit_registry()
        premises = context.premises
        if not premises:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("MISSING_COMPARISON_PREMISES",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Left side: premise 0 (number or interval)
        lhs_node = premises[0]
        lhs_iv = _extract_interval(lhs_node.value)
        lhs_unit = lhs_node.unit or (lhs_node.value.get("unit") if isinstance(lhs_node.value, Mapping) else None)

        # Right side: premise 1, or declared bound/threshold in params/conclusion
        rhs_unit: str | None = None
        rhs_iv: tuple[Decimal, Decimal] | None = None
        if op in {"between", "within_bounds"}:
            lo_b = _to_decimal(params.get("min"))
            hi_b = _to_decimal(params.get("max"))
            rhs_unit = params.get("unit")
            if lo_b is None or hi_b is None:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("MISSING_BETWEEN_BOUNDS",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            rhs_iv = (lo_b, hi_b)
        elif len(premises) >= 2:
            rhs_node = premises[1]
            rhs_iv = _extract_interval(rhs_node.value)
            rhs_unit = rhs_node.unit or (rhs_node.value.get("unit") if isinstance(rhs_node.value, Mapping) else None)
        elif "threshold" in params:
            t_iv = _extract_interval(params["threshold"])
            rhs_iv = t_iv
            rhs_unit = params.get("threshold_unit") or lhs_unit
        elif isinstance(context.conclusion.value, Mapping) and "operator" in context.conclusion.value:
            op = str(context.conclusion.value["operator"]).lower()
            rhs_iv = _extract_interval(context.conclusion.value.get("value"))
            rhs_unit = context.conclusion.value.get("unit") or context.conclusion.unit

        if lhs_iv is None or rhs_iv is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("NON_NUMERIC_COMPARISON_OPERANDS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Align units via UnitRegistry
        try:
            l_lo, l_dim, _ = _convert_to_base_decimal(lhs_iv[0], lhs_unit, registry)
            l_hi, _, _ = _convert_to_base_decimal(lhs_iv[1], lhs_unit, registry)
            r_lo, r_dim, _ = _convert_to_base_decimal(rhs_iv[0], rhs_unit, registry)
            r_hi, _, _ = _convert_to_base_decimal(rhs_iv[1], rhs_unit, registry)
        except UnmappedProposal as exc:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNKNOWN_COMPARISON_UNIT:{exc.detail}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if l_dim is not None and r_dim is not None and l_dim != r_dim:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"lhs_dimension": l_dim, "rhs_dimension": r_dim},
                reasons=("COMPARISON_UNIT_DIMENSION_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Three-valued interval evaluation: guaranteed True, guaranteed False, or undecidable
        holds: bool | None = None
        if op == "eq":
            if l_lo == l_hi == r_lo == r_hi:
                holds = True
            elif l_hi < r_lo or r_hi < l_lo:
                holds = False
        elif op == "ne":
            if l_hi < r_lo or r_hi < l_lo:
                holds = True
            elif l_lo == l_hi == r_lo == r_hi:
                holds = False
        elif op in {"lt", "below_floor"}:
            if l_hi < r_lo:
                holds = True
            elif l_lo >= r_hi:
                holds = False
        elif op == "le":
            if l_hi <= r_lo:
                holds = True
            elif l_lo > r_hi:
                holds = False
        elif op in {"gt", "exceeds_cap"}:
            if l_lo > r_hi:
                holds = True
            elif l_hi <= r_lo:
                holds = False
        elif op == "ge":
            if l_lo >= r_hi:
                holds = True
            elif l_hi < r_lo:
                holds = False
        elif op in {"between", "within_bounds"}:
            if r_lo <= l_lo and l_hi <= r_hi:
                holds = True
            elif l_hi < r_lo or l_lo > r_hi:
                holds = False
        else:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="not_applicable",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNSUPPORTED_COMPARISON_OP:{op}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if holds is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={"lhs": [str(l_lo), str(l_hi)], "rhs": [str(r_lo), str(r_hi)], "op": op},
                reasons=("COMPARISON_OVERLAPPING_INTERVALS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        expected_bool = context.conclusion.value if isinstance(context.conclusion.value, bool) else True
        status = "verified" if holds == expected_bool else "refuted"
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status=status,
            guarantee_kind=self.guarantee_kind,
            evidence={"lhs": [str(l_lo), str(l_hi)], "rhs": [str(r_lo), str(r_hi)], "op": op, "holds": holds},
            reasons=("COMPARISON_VERIFIED" if status == "verified" else "COMPARISON_CONTRADICTED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 5: aggregation
# ---------------------------------------------------------------------------

class AggregationVerifier:
    """Hardened aggregation verifier checking population completeness, unit alignment, and duplicates."""

    family = "aggregation"
    verifier_id = "deterministic_population_aggregation"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    _AGG_OPS = {"sum", "agg_sum", "count", "agg_count", "average", "agg_average", "mean", "agg_mean", "min", "agg_min", "max", "agg_max", "aggregation"}

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        op = str(params.get("aggregation_op") or params.get("op") or context.step.operation).lower().strip()
        if op == "aggregation":
            op = str(params.get("sub_op", "sum")).lower().strip()
        op_short = op.replace("agg_", "")
        if op_short == "mean":
            op_short = "average"

        if op not in self._AGG_OPS:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="not_applicable",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNSUPPORTED_AGGREGATION_OP:{op}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        premises = context.premises
        if not premises:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("NO_PREMISES_FOR_AGGREGATION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # 1. Deduplication & event identity check
        dedup_nodes: list[ClaimNode] = []
        seen_events: dict[str, ClaimNode] = {}
        seen_signatures: dict[str, ClaimNode] = {}
        distinct_confirmed = bool(params.get("distinct_events_confirmed", False))
        assumptions = list(context.step.assumptions)

        for node in premises:
            if node.event_id:
                if node.event_id in seen_events:
                    prev = seen_events[node.event_id]
                    if (
                        label_key(prev.subject) != label_key(node.subject)
                        or prev.predicate != node.predicate
                        or canonical(prev.value) != canonical(node.value)
                        or prev.unit != node.unit
                        or prev.polarity != node.polarity
                    ):
                        return VerifierResult(
                            obligation_id=obligation.obligation_id,
                            step_id=context.step.step_id,
                            family=self.family,
                            status="refuted",
                            guarantee_kind=self.guarantee_kind,
                            evidence={"conflicting_event_id": node.event_id},
                            reasons=("CONFLICTING_EVENT_VALUES",),
                            verifier_id=self.verifier_id,
                            verifier_version=self.version,
                        )
                    assumptions.append(f"deduplicated_event:{node.event_id}")
                    continue
                seen_events[node.event_id] = node
                dedup_nodes.append(node)
            else:
                sig = canonical({
                    "subject": label_key(node.subject),
                    "predicate": node.predicate,
                    "value": node.value,
                    "unit": node.unit,
                    "polarity": node.polarity,
                })
                if sig in seen_signatures and not distinct_confirmed:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="unknown",
                        guarantee_kind=self.guarantee_kind,
                        evidence={"duplicate_claim_ids": [seen_signatures[sig].claim_id, node.claim_id]},
                        reasons=("AMBIGUOUS_EVENT_DUPLICATION",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
                seen_signatures[sig] = node
                dedup_nodes.append(node)

        # 2. Population completeness check (subtotal vs total handling for mandatory checks)
        is_mandatory_agg = params.get("mandatory_check") == "aggregation_completeness"
        corr = context.correspondence_facts or {}
        qualified_subtotal = bool(
            (isinstance(context.conclusion.scope, Mapping) and context.conclusion.scope.get("is_subtotal"))
            or corr.get("qualified_as_subtotal")
        )
        presents_as_unqualified_total = bool(
            corr.get("response_checked") and corr.get("presents_subtotal_as_total")
        )
        allow_subtotal = (
            (is_mandatory_agg or bool(corr.get("response_checked")))
            and qualified_subtotal
            and not presents_as_unqualified_total
        )
        subtotal_mode = False
        expected_ids = params.get("expected_population_ids")
        pop_complete = bool(params.get("population_complete", False))
        if expected_ids is not None:
            expected_set = {label_key(str(x)) for x in expected_ids}
            observed_set = {
                label_key(str(n.event_id or n.scope.get("member_id") or n.subject))
                for n in dedup_nodes
            }
            if not expected_set <= observed_set:
                missing = sorted(expected_set - observed_set)
                if allow_subtotal:
                    subtotal_mode = True
                    assumptions.append("partial_population_subtotal_only")
                else:
                    reasons = (
                        ("UNVERIFIED_POPULATION_COMPLETENESS", "SUBTOTAL_PRESENTED_AS_TOTAL")
                        if is_mandatory_agg
                        else ("UNVERIFIED_POPULATION_COMPLETENESS",)
                    )
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="unknown",
                        guarantee_kind=self.guarantee_kind,
                        evidence={"missing_population_members": missing, "is_subtotal": qualified_subtotal},
                        reasons=reasons,
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )
        elif not pop_complete:
            if allow_subtotal:
                subtotal_mode = True
                assumptions.append("partial_population_subtotal_only")
            else:
                reasons = (
                    ("UNVERIFIED_POPULATION_COMPLETENESS", "SUBTOTAL_PRESENTED_AS_TOTAL")
                    if is_mandatory_agg
                    else ("UNVERIFIED_POPULATION_COMPLETENESS",)
                )
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"is_subtotal": qualified_subtotal},
                    reasons=reasons,
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
        elif presents_as_unqualified_total and not pop_complete:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("SUBTOTAL_PRESENTED_AS_TOTAL",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        conc_val, conc_unit = _extract_quantity_pair(context.conclusion)
        if conc_val is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("NON_NUMERIC_AGGREGATION_CONCLUSION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if op_short == "count":
            computed = Decimal(len(dedup_nodes))
            if computed != conc_val:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"expected_count": str(computed), "proposed_count": str(conc_val)},
                    reasons=("AGGREGATION_COUNT_MISMATCH",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence={"count": str(computed)},
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("AGGREGATION_COUNT_VERIFIED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # 3. Unit commensurability for sum / average / min / max
        registry = context.domain_profile.effective_unit_registry()
        base_values: list[Decimal] = []
        dims: set[str | None] = set()
        for node in dedup_nodes:
            v, u = _extract_quantity_pair(node)
            if v is None:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=(f"NON_NUMERIC_AGGREGATION_MEMBER:{node.claim_id}",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            try:
                bv, dim, _ = _convert_to_base_decimal(v, u, registry)
            except UnmappedProposal as exc:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    reasons=(f"UNKNOWN_AGGREGATION_UNIT:{exc.detail}",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            dims.add(dim)
            base_values.append(bv)

        if len(dims) > 1:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"member_dimensions": sorted(str(d) for d in dims)},
                reasons=("AGGREGATION_INCOMPATIBLE_UNITS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        member_dim = next(iter(dims))
        try:
            conc_base, conc_dim, _ = _convert_to_base_decimal(conc_val, conc_unit, registry)
        except UnmappedProposal as exc:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=(f"UNKNOWN_CONCLUSION_UNIT:{exc.detail}",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if member_dim != conc_dim and (member_dim is not None or conc_dim is not None):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"member_dimension": member_dim, "conclusion_dimension": conc_dim},
                reasons=("AGGREGATION_INCOMPATIBLE_UNITS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if op_short == "sum":
            expected = sum(base_values, Decimal(0))
        elif op_short == "average":
            expected = sum(base_values, Decimal(0)) / Decimal(len(base_values))
        elif op_short == "min":
            expected = min(base_values)
        else:  # max
            expected = max(base_values)

        tol = _to_decimal(params.get("rounding_tolerance", params.get("tolerance", "0"))) or Decimal(0)
        if abs(expected - conc_base) <= tol:
            ev_ok: dict[str, Any] = {
                "op": op_short,
                "expected": str(expected),
                "proposed": str(conc_base),
                "dedup_count": len(dedup_nodes),
            }
            if subtotal_mode:
                ev_ok["subtotal_only"] = True
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence=ev_ok,
                assumptions=tuple(dict.fromkeys(assumptions)),
                reasons=("AGGREGATION_SUBTOTAL_VERIFIED" if subtotal_mode else "AGGREGATION_VERIFIED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="refuted",
            guarantee_kind=self.guarantee_kind,
            evidence={
                "op": op_short,
                "expected": str(expected),
                "proposed": str(conc_base),
                "dedup_count": len(dedup_nodes),
            },
            reasons=("AGGREGATION_VALUE_MISMATCH",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 6: explicit_logic
# ---------------------------------------------------------------------------

def _matches_pattern(node: ClaimNode, pattern: Mapping[str, Any]) -> bool:
    for field_name in ("subject", "predicate", "unit", "condition_status"):
        if field_name in pattern and pattern[field_name] is not None:
            actual = getattr(node, field_name)
            expected = pattern[field_name]
            if field_name == "subject":
                if label_key(str(actual)) != label_key(str(expected)):
                    return False
            elif actual != expected:
                return False
    if "polarity" in pattern and pattern["polarity"] is not None:
        if node.polarity != int(pattern["polarity"]):
            return False
    if "value" in pattern:
        if canonical(node.value) != canonical(pattern["value"]):
            return False
    return True


class ExplicitLogicVerifier:
    """Verifies rule application against versioned DeclaredRules in DomainProfile."""

    family = "explicit_logic"
    verifier_id = "deterministic_explicit_logic"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        if params.get("mandatory_check") == "condition_preservation":
            return self._verify_condition_preservation(obligation, context, params)
        rule_id = context.step.rule_id or params.get("rule_id")
        rule_version = context.step.rule_version or params.get("rule_version")
        if not rule_id:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("MISSING_RULE_ID",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        rule = context.domain_profile.declared_rules.get(str(rule_id))
        if rule is None or (rule_version is not None and rule.version != str(rule_version)):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={"rule_id": rule_id, "requested_version": rule_version},
                reasons=("UNDECLARED_OR_MISMATCHED_RULE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        premises = context.premises
        # Block if any premise carries unresolved or negative condition_status
        for p in premises:
            if p.condition_status in {"unknown_condition", "unrepresentable_condition"}:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"premise_id": p.claim_id, "condition_status": p.condition_status},
                    reasons=("PREMISE_HAS_UNRESOLVED_CONDITION",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        req_patterns = rule.required_premises
        matched_count = 0
        for pat in req_patterns:
            matched = any(_matches_pattern(p, pat) for p in premises)
            if matched:
                matched_count += 1

        if rule.kind in {"implication", "conjunction"}:
            if len(req_patterns) > 0 and matched_count < len(req_patterns):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"matched_premises": matched_count, "required_premises": len(req_patterns)},
                    reasons=("RULE_PRECONDITIONS_NOT_SATISFIED",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
        elif rule.kind == "disjunction":
            if len(req_patterns) > 0 and matched_count == 0:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("DISJUNCTION_NO_PREMISE_SATISFIED",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
        elif rule.kind == "threshold":
            min_matches = int(rule.parameters.get("min_matches", 1))
            if matched_count < min_matches:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"matched": matched_count, "min_matches": min_matches},
                    reasons=("THRESHOLD_RULE_NOT_MET",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
        elif rule.kind == "exclusion":
            if len(req_patterns) > 0 and matched_count == len(req_patterns):
                # Trigger for exclusion rule holds -> positive conclusion claiming unblocked is refuted
                if context.conclusion.polarity == 1 and rule.conclusion_template.get("polarity") == -1:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="refuted",
                        guarantee_kind=self.guarantee_kind,
                        reasons=("EXCLUSION_RULE_TRIGGERED",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )

        if rule.conclusion_template and not _matches_pattern(context.conclusion, rule.conclusion_template):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"expected_template": rule.conclusion_template},
                reasons=("CONCLUSION_DOES_NOT_MATCH_RULE_TEMPLATE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            evidence={"rule_id": rule.rule_id, "rule_version": rule.version},
            reasons=("EXPLICIT_RULE_VERIFIED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_condition_preservation(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
        params: Mapping[str, Any],
    ) -> VerifierResult:
        req_conds = list(params.get("required_conditions") or ())
        if not req_conds:
            for p in context.premises:
                for cond in p.conditions:
                    req_conds.append(dict(cond))
                if not p.conditions and p.condition_status != "unconditional":
                    req_conds.append({"claim_id": p.claim_id, "condition_status": p.condition_status})

        if not req_conds:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="not_applicable",
                guarantee_kind=self.guarantee_kind,
                reasons=("NO_REQUIRED_CONDITIONS",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        conc = context.conclusion
        conc_cond_canon = {canonical(dict(c)) for c in conc.conditions}
        missing: list[dict[str, Any]] = []
        for rc in req_conds:
            if canonical(rc) not in conc_cond_canon:
                # Also match by field/predicate if premise stored structured condition and conclusion kept same condition field
                rc_field = rc.get("field")
                if not (rc_field and any(c.get("field") == rc_field for c in conc.conditions)):
                    missing.append(rc)

        if conc.condition_status != "conditional" or missing:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"missing_conditions": missing, "conclusion_condition_status": conc.condition_status},
                reasons=("DROPPED_REQUIRED_CONDITION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        corr = context.correspondence_facts or {}
        if corr.get("response_checked") and corr.get("condition_dropped_in_response"):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"missing_conditions": req_conds, "in_response": True},
                reasons=("CONDITION_DROPPED_IN_RESPONSE", "DROPPED_REQUIRED_CONDITION"),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            evidence={"preserved_conditions": len(req_conds)},
            assumptions=("conclusion_subject_to_preserved_conditions",),
            reasons=("CONDITIONS_PRESERVED_VERIFIED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 7: relations
# ---------------------------------------------------------------------------

class RelationVerifier:
    """Verifies multi-hop relation compositions declared in DomainProfile."""

    family = "relations"
    verifier_id = "deterministic_relation_composition"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        premises = context.premises
        if len(premises) < 2:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("RELATION_COMPOSITION_REQUIRES_AT_LEAST_TWO_EDGES",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if any(p.polarity != 1 for p in premises):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                reasons=("NEGATED_EDGE_IN_RELATION_PATH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Verify chain connectivity: A -[p0]-> B, B -[p1]-> C ...
        visited_entities = [label_key(premises[0].subject)]
        for idx in range(len(premises) - 1):
            curr = premises[idx]
            nxt = premises[idx + 1]
            bridge_out = label_key(str(curr.value))
            bridge_in = label_key(nxt.subject)
            if not bridge_out or bridge_out != bridge_in:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"edge_index": idx, "out": str(curr.value), "in": nxt.subject},
                    reasons=("DISCONNECTED_RELATION_BRIDGE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            if bridge_in in visited_entities:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("CYCLIC_RELATION_PATH",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            visited_entities.append(bridge_in)

        start_sub = label_key(premises[0].subject)
        end_val = label_key(str(premises[-1].value))
        if label_key(context.conclusion.subject) != start_sub or label_key(str(context.conclusion.value)) != end_val:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                reasons=("RELATION_ENDPOINT_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Authorize relation composition pair by pair via DomainProfile.relation_rules
        rules = list(context.domain_profile.relation_rules.values())
        curr_pred = premises[0].predicate
        require_overlap = False
        for edge in premises[1:]:
            matched_rule = None
            for r in rules:
                if (
                    r.left_predicate == curr_pred
                    and r.right_predicate == edge.predicate
                ) or (
                    r.transitive and curr_pred == edge.predicate == r.left_predicate
                ):
                    matched_rule = r
                    break
            if matched_rule is None:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"left": curr_pred, "right": edge.predicate},
                    reasons=("UNAUTHORIZED_RELATION_COMPOSITION",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            curr_pred = matched_rule.composed_predicate
            require_overlap = require_overlap or matched_rule.require_temporal_overlap

        if context.conclusion.predicate and curr_pred != context.conclusion.predicate:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="refuted",
                guarantee_kind=self.guarantee_kind,
                evidence={"composed_predicate": curr_pred, "conclusion_predicate": context.conclusion.predicate},
                reasons=("COMPOSED_PREDICATE_MISMATCH",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if require_overlap:
            starts: list[date] = []
            ends: list[date] = []
            for p in premises:
                if p.valid_from:
                    sb = _parse_date_bounds(p.valid_from)
                    if sb:
                        starts.append(sb[0])
                if p.valid_until:
                    eb = _parse_date_bounds(p.valid_until)
                    if eb:
                        ends.append(eb[1])
            if starts and ends and max(starts) > min(ends):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("DISJOINT_RELATION_TEMPORAL_WINDOWS",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            evidence={"hops": len(premises), "composed_predicate": curr_pred},
            reasons=("RELATION_COMPOSITION_VERIFIED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 8: document_authority
# ---------------------------------------------------------------------------

class DocumentAuthorityVerifier:
    """Checks document validity, revisions, ClauseRelations, and attestor roles."""

    family = "document_authority"
    verifier_id = "deterministic_document_authority"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        mandatory_kind = params.get("mandatory_check")
        if mandatory_kind == "source_conditional_attribution":
            return self._verify_source_attribution(obligation, context)
        if mandatory_kind == "material_conflict_disclosure":
            return self._verify_conflict_disclosure(obligation, context)
        if mandatory_kind == "semantic_type_uncertainty":
            return self._verify_semantic_type_uncertainty(obligation, context, params)

        nodes_to_check = list(context.premises) or [context.conclusion]
        policy = context.domain_profile.policy
        as_of_b = _parse_date_bounds(context.as_of) if context.as_of else None

        # Check suppressed clause ids if explicit ClauseRelations exist
        suppressed_clauses: set[str] = set()
        present_clauses = {
            str(n.scope["clause_id"])
            for n in nodes_to_check
            if isinstance(n.scope, Mapping) and n.scope.get("clause_id")
        }
        for rel in context.clause_relations or ():
            src = getattr(rel, "source_clause_id", None)
            tgt = getattr(rel, "target_clause_id", None)
            if src in present_clauses and tgt in present_clauses:
                suppressed_clauses.add(str(tgt))

        # Check EvidenceLayer admissibility if layer and evidence_unit_ids are provided
        if context.layer is not None and context.domain_profile.vocabulary is not None:
            layer_snap = context.layer.snapshot()
            units_by_id = {u.unit_id: u for u in layer_snap["units"]}
            layer_docs_map = layer_snap["documents"]
            acc_policy = AcceptancePolicy(
                tau=policy.tau,
                delta=policy.delta,
            )
            for node in nodes_to_check:
                for uid in node.evidence_unit_ids:
                    unit = units_by_id.get(uid)
                    if unit is None:
                        return VerifierResult(
                            obligation_id=obligation.obligation_id,
                            step_id=context.step.step_id,
                            family=self.family,
                            status="refuted",
                            guarantee_kind=self.guarantee_kind,
                            evidence={"missing_unit_id": uid},
                            reasons=("MISSING_LEDGER_EVIDENCE_UNIT",),
                            verifier_id=self.verifier_id,
                            verifier_version=self.version,
                        )
                    ev = evaluate_unit_admissibility(
                        unit,
                        vocabulary=context.domain_profile.vocabulary,
                        policy=acc_policy,
                        profile=context.domain_profile.epistemic_profile,
                        units_by_id=units_by_id,
                        documents=layer_docs_map,
                        stored_field_evidences=layer_snap.get("field_evidences"),
                        as_of=context.as_of,
                    )
                    if not ev.admissible:
                        return VerifierResult(
                            obligation_id=obligation.obligation_id,
                            step_id=context.step.step_id,
                            family=self.family,
                            status="refuted",
                            guarantee_kind=self.guarantee_kind,
                            evidence={"unit_id": uid, "reason": ev.reason, "issue_code": ev.issue_code},
                            reasons=(str(ev.issue_code or ev.reason or "UNIT_INADMISSIBLE"),),
                            verifier_id=self.verifier_id,
                            verifier_version=self.version,
                        )

        unverified_roles = {label_key(r) for r in policy.unverified_attestor_roles}
        for node in nodes_to_check:
            cid_clause = node.scope.get("clause_id") if isinstance(node.scope, Mapping) else None
            if cid_clause and str(cid_clause) in suppressed_clauses:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"superseded_clause_id": str(cid_clause), "claim_id": node.claim_id},
                    reasons=("SUPERSEDED_OR_AMENDED_CLAUSE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

            if node.metadata.get("superseded") or node.metadata.get("superseded_by"):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"claim_id": node.claim_id, "superseded_by": node.metadata.get("superseded_by")},
                    reasons=("SUPERSEDED_OR_INACTIVE_DOCUMENT",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

            if as_of_b is not None:
                vf = _parse_date_bounds(node.valid_from)
                vu = _parse_date_bounds(node.valid_until)
                if (vf and as_of_b[1] < vf[0]) or (vu and as_of_b[0] > vu[1]):
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="refuted",
                        guarantee_kind=self.guarantee_kind,
                        evidence={"claim_id": node.claim_id, "as_of": context.as_of},
                        reasons=("OUTSIDE_TEMPORAL_VALIDITY",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )

            for ref in node.evidence:
                doc = context.documents.get(ref.document_id) if context.documents else None
                if doc is not None and ref.revision != "rev-0" and ref.revision != doc.revision:
                    return VerifierResult(
                        obligation_id=obligation.obligation_id,
                        step_id=context.step.step_id,
                        family=self.family,
                        status="refuted",
                        guarantee_kind=self.guarantee_kind,
                        evidence={"document_id": ref.document_id, "ref_revision": ref.revision, "doc_revision": doc.revision},
                        reasons=("STALE_OR_MISMATCHED_DOCUMENT_REVISION",),
                        verifier_id=self.verifier_id,
                        verifier_version=self.version,
                    )

            if node.attestor and label_key(node.attestor) in unverified_roles:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"claim_id": node.claim_id, "attestor": node.attestor},
                    reasons=("UNVERIFIED_ATTESTOR_ROLE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            reasons=("DOCUMENT_AUTHORITY_VERIFIED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_source_attribution(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
    ) -> VerifierResult:
        nodes = list(context.premises) or [context.conclusion]
        doc_ids: set[str] = set()
        for n in nodes:
            doc_ids.update(n.doc_ids())
        cited_docs = [context.documents[did] for did in sorted(doc_ids) if did in context.documents]
        external_verified = bool(
            isinstance(context.conclusion.scope, Mapping) and context.conclusion.scope.get("externally_verified")
        ) or (
            bool(cited_docs)
            and all(d.attribution in {"audited_record", "official_registry"} for d in cited_docs)
        )
        if external_verified:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                reasons=("EXTERNAL_TRUTH_CORROBORATED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        corr = context.correspondence_facts or {}
        if corr.get("response_checked"):
            if corr.get("contradictory_source_attribution"):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("CONTRADICTORY_SOURCE_ATTRIBUTION",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            if not corr.get("source_qualified"):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="unknown",
                    guarantee_kind=self.guarantee_kind,
                    evidence={"cited_document_ids": sorted(doc_ids)},
                    reasons=("SOURCE_CONDITIONAL_CONCLUSION",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="verified",
                guarantee_kind=self.guarantee_kind,
                evidence={"cited_document_ids": sorted(doc_ids)},
                assumptions=("conclusion_conditional_on_cited_sources_unverified_externally",),
                reasons=("SOURCE_CONDITIONAL_ATTRIBUTION_VERIFIED",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # Pure graph_only inspection: if conclusion explicitly asserts external truth uncorroborated, block
        if isinstance(context.conclusion.scope, Mapping) and context.conclusion.scope.get("asserts_external_truth"):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={"cited_document_ids": sorted(doc_ids)},
                reasons=("SOURCE_CONDITIONAL_CONCLUSION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified",
            guarantee_kind=self.guarantee_kind,
            evidence={"cited_document_ids": sorted(doc_ids)},
            assumptions=("conclusion_conditional_on_cited_sources_unverified_externally",),
            reasons=("DOCUMENT_LINKAGE_ATTRIBUTED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_conflict_disclosure(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
    ) -> VerifierResult:
        corr = context.correspondence_facts or {}
        if corr.get("response_checked") and not corr.get("conflict_disclosed"):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("UNDISCLOSED_MATERIAL_CONFLICT",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="unknown" if not corr.get("conflict_disclosed") else "verified",
            guarantee_kind=self.guarantee_kind,
            reasons=("MATERIAL_CONFLICT_UNRESOLVED",) if not corr.get("conflict_disclosed") else ("MATERIAL_CONFLICT_DISCLOSED",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )

    def _verify_semantic_type_uncertainty(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
        params: Mapping[str, Any],
    ) -> VerifierResult:
        discovered = list(params.get("discovered_domains") or ())
        evidence = {"discovered_domains": discovered} if discovered else {}
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="unknown",
            guarantee_kind=self.guarantee_kind,
            evidence=evidence,
            reasons=("AMBIGUOUS_OR_GENERIC_SEMANTIC_TYPE",),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


# ---------------------------------------------------------------------------
# Family 9: open_semantics
# ---------------------------------------------------------------------------

class OpenSemanticsVerifier:
    """Model-backed semantic correspondence evaluator (strictly probabilistic_model)."""

    family = "open_semantics"
    verifier_id = "open_semantics_bridge"
    version = "1.0.0"
    guarantee_kind = "probabilistic_model"

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        cfg = context.domain_profile.semantic_evaluator
        if not context.allow_semantic_call:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind="probabilistic_model",
                reasons=("PENDING_BUDGET",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        if not cfg.enabled or cfg.callable_evaluator is None:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind="probabilistic_model",
                reasons=("OPEN_SEMANTICS_NOT_EXECUTED",),
                verifier_id=cfg.evaluator_id or self.verifier_id,
                verifier_version=cfg.version or self.version,
            )

        try:
            raw = cfg.callable_evaluator(
                obligation=obligation,
                step=context.step,
                conclusion=context.conclusion,
                premises=context.premises,
                domain_profile=context.domain_profile,
                documents=context.documents,
            )
            status = str(raw.get("status", "unknown"))
            if status not in {"verified", "refuted", "unknown", "not_applicable"}:
                status = "unknown"
            ev = dict(raw.get("evidence") or {})
            if "score" in raw:
                ev["model_score"] = float(raw["score"])
            assump = tuple(str(a) for a in (raw.get("assumptions") or ()))
            reasons = tuple(str(r) for r in (raw.get("reasons") or (f"OPEN_SEMANTICS_{status.upper()}",)))
        except Exception as exc:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind="probabilistic_model",
                evidence={"error": type(exc).__name__},
                reasons=("OPEN_SEMANTICS_EVALUATOR_ERROR",),
                verifier_id=cfg.evaluator_id or self.verifier_id,
                verifier_version=cfg.version or self.version,
            )

        # Invariant lock: NEVER emit 'deterministic_formal' from open_semantics
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status=status,
            guarantee_kind="probabilistic_model",
            evidence=ev,
            assumptions=assump,
            reasons=reasons,
            verifier_id=cfg.evaluator_id or self.verifier_id,
            verifier_version=cfg.version or self.version,
        )


class VerifierRegistry:
    """Extensible registry mapping the 9 verifier families to concrete verifier instances."""

    def __init__(self, verifiers: Sequence[Verifier] = ()) -> None:
        self._verifiers: dict[str, Verifier] = {}
        for v in verifiers:
            self.register(v)

    def register(self, verifier: Verifier) -> None:
        if verifier.family not in VERIFIER_FAMILIES:
            raise ValueError(f"Unsupported verifier family {verifier.family!r}.")
        self._verifiers[verifier.family] = verifier

    def get(self, family: str) -> Verifier:
        if family not in self._verifiers:
            raise KeyError(f"No verifier registered for family {family!r}.")
        return self._verifiers[family]

    def versions_snapshot(self) -> dict[str, str]:
        return {
            fam: f"{v.verifier_id}:{v.version}:{v.guarantee_kind}"
            for fam, v in sorted(self._verifiers.items())
        }

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))

    def verify_obligation(
        self,
        obligation: VerificationObligation,
        context: VerifierContext,
    ) -> VerifierResult:
        verifier = self.get(obligation.family)
        res = verifier.verify(obligation, context)
        if obligation.family == "open_semantics" and res.guarantee_kind != "probabilistic_model":
            raise RuntimeError("Protocol violation: open_semantics emitted non-probabilistic guarantee.")
        return res


DEFAULT_VERIFIER_REGISTRY = VerifierRegistry([
    ArithmeticVerifier(),
    QuantityVerifier(),
    TemporalVerifier(),
    ComparisonVerifier(),
    AggregationVerifier(),
    ExplicitLogicVerifier(),
    RelationVerifier(),
    DocumentAuthorityVerifier(),
    OpenSemanticsVerifier(),
])
