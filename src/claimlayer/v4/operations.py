"""Controlled, versioned OperationCatalog and calendar/arithmetic executors for v4.

All executable derivation logic lives here (or inside declared v4 verifiers).
Models propose structured plans referencing catalog operation names; Deltoryx
controls operation semantics, validates inputs, and recomputes conclusions
deterministically without executing model-generated code or reading the system clock.
"""
from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Callable, Mapping, Sequence
import unicodedata

from ..semantic.models import canonical, digest
from .domain_profile import DomainProfile
from .justification import ClaimNode, VERIFIER_FAMILIES
from .vocabulary import STANDARD_UNITS, UnitRegistry, UnmappedProposal, label_key, normalize_date

CANONICAL_WEEKDAYS_ES: tuple[str, ...] = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)

CANONICAL_WEEKDAYS_EN: tuple[str, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

_WEEKDAY_ALIASES: dict[str, tuple[int, str]] = {
    # Spanish (canonical + unaccented + common abbreviations)
    "lunes": (0, "es"),
    "lun": (0, "es"),
    "martes": (1, "es"),
    "mar": (1, "es"),
    "miercoles": (2, "es"),
    "mie": (2, "es"),
    "jueves": (3, "es"),
    "jue": (3, "es"),
    "viernes": (4, "es"),
    "vie": (4, "es"),
    "sabado": (5, "es"),
    "sab": (5, "es"),
    "domingo": (6, "es"),
    "dom": (6, "es"),
    # English (canonical + common abbreviations)
    "monday": (0, "en"),
    "mon": (0, "en"),
    "tuesday": (1, "en"),
    "tue": (1, "en"),
    "tues": (1, "en"),
    "wednesday": (2, "en"),
    "wed": (2, "en"),
    "thursday": (3, "en"),
    "thu": (3, "en"),
    "thur": (3, "en"),
    "thurs": (3, "en"),
    "friday": (4, "en"),
    "fri": (4, "en"),
    "saturday": (5, "en"),
    "sat": (5, "en"),
    "sunday": (6, "en"),
    "sun": (6, "en"),
}


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def normalize_weekday(value: Any) -> tuple[int, str, str] | None:
    """Normalize a Spanish or English weekday string to `(index_0_to_6, canonical_es, lang)`."""
    if not isinstance(value, str):
        return None
    cleaned = _strip_accents(value.strip()).casefold().strip(".,;:()[]{}\"' ")
    if not cleaned:
        return None
    hit = _WEEKDAY_ALIASES.get(cleaned)
    if hit is None:
        return None
    idx, lang = hit
    return idx, CANONICAL_WEEKDAYS_ES[idx], lang


def weekday_name(index: int, lang: str = "es") -> str:
    """Return the canonical weekday name for index `0..6` in `'es'` or `'en'`."""
    idx = index % 7
    return CANONICAL_WEEKDAYS_EN[idx] if lang == "en" else CANONICAL_WEEKDAYS_ES[idx]


def compute_weekday_shift(anchor_weekday: Any, offset_days: int) -> dict[str, Any]:
    """Deterministically shift a weekday by signed `offset_days` across week boundaries."""
    if isinstance(offset_days, bool) or not isinstance(offset_days, int):
        raise ValueError(f"offset_days must be an integer, got {offset_days!r}.")
    norm = normalize_weekday(anchor_weekday)
    if norm is None:
        raise ValueError(f"Unrecognized weekday anchor {anchor_weekday!r}.")
    anchor_idx, anchor_es, lang = norm
    raw_sum = anchor_idx + offset_days
    target_idx = raw_sum % 7
    crossed_week_boundary = raw_sum < 0 or raw_sum >= 7
    return {
        "anchor_weekday": weekday_name(anchor_idx, lang),
        "anchor_weekday_es": anchor_es,
        "anchor_index": anchor_idx,
        "offset_days": offset_days,
        "computed_weekday": weekday_name(target_idx, lang),
        "computed_weekday_es": CANONICAL_WEEKDAYS_ES[target_idx],
        "computed_weekday_en": CANONICAL_WEEKDAYS_EN[target_idx],
        "target_index": target_idx,
        "language": lang,
        "crossed_week_boundary": crossed_week_boundary,
    }


def compute_date_offset(anchor_date: str, offset_days: int) -> dict[str, Any]:
    """Deterministically shift a full ISO `YYYY-MM-DD` date by signed `offset_days`."""
    if isinstance(offset_days, bool) or not isinstance(offset_days, int):
        raise ValueError(f"offset_days must be an integer, got {offset_days!r}.")
    if not isinstance(anchor_date, str) or not anchor_date.strip():
        raise ValueError("anchor_date must be a non-empty ISO date string.")
    iso = normalize_date(anchor_date.strip())
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso):
        raise ValueError(f"date_offset requires YYYY-MM-DD granularity, got {anchor_date!r}.")
    d = date.fromisoformat(iso)
    target = d + timedelta(days=offset_days)
    return {
        "anchor_date": d.isoformat(),
        "offset_days": offset_days,
        "computed_date": target.isoformat(),
        "computed_weekday": CANONICAL_WEEKDAYS_ES[target.weekday()],
        "computed_weekday_en": CANONICAL_WEEKDAYS_EN[target.weekday()],
    }


# ---------------------------------------------------------------------------
# Shared numeric, quantity, and date helpers
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


def _decimal_to_json(val: Decimal) -> int | float | str:
    norm = val.normalize()
    if norm == norm.to_integral_value():
        return int(norm)
    as_float = float(norm)
    try:
        if Decimal(str(as_float)) == norm:
            return as_float
    except (InvalidOperation, ValueError):
        pass
    return format(norm, "f")


def _extract_quantity_pair(
    node_or_val: ClaimNode | Mapping[str, Any] | Any,
    default_unit: str | None = None,
) -> tuple[Decimal | None, str | None]:
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
    if not unit_label:
        return value, None, None
    unit_obj = registry.lookup(unit_label)
    base_val = value * Decimal(unit_obj.factor)
    return base_val, unit_obj.dimension, registry.base(unit_obj.dimension)


def normalize_day_offset(
    node_or_val: ClaimNode | Mapping[str, Any] | Any,
    unit_hint: str | None = None,
    registry: UnitRegistry | None = None,
) -> tuple[int | None, str | None]:
    """Extract an exact integer day offset from a value or ClaimNode.

    Accepts calendar days or weeks (converted x7 via `UnitRegistry`), and rejects
    incommensurable units such as `business_days`, `months`, `years`, or `USD`.
    """
    reg = registry or STANDARD_UNITS
    val, unit = _extract_quantity_pair(node_or_val, default_unit=unit_hint)
    if val is None:
        return None, "NON_NUMERIC_DAY_OFFSET"
    if unit:
        try:
            u_obj = reg.lookup(unit)
        except UnmappedProposal:
            return None, f"INCOMPATIBLE_OFFSET_UNIT:{unit}"
        if u_obj.dimension != "duration_calendar_days":
            return None, f"INCOMPATIBLE_OFFSET_UNIT:{unit}"
        scaled = val * Decimal(u_obj.factor)
    else:
        scaled = val
    if scaled != scaled.to_integral_value():
        return None, "NON_INTEGER_DAY_OFFSET"
    return int(scaled), None


def _parse_date_bounds(raw: Any) -> tuple[date, date, str] | None:
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
# Controlled Operation Catalog contracts
# ---------------------------------------------------------------------------

ERROR_KINDS: tuple[str, ...] = ("missing_input", "formal_error", "undeclared_rule")


@dataclass(frozen=True)
class OperationOutput:
    """Result emitted by a controlled operation executor."""

    ok: bool
    value: Any = None
    unit: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    error_code: str | None = None
    error_kind: str | None = None  # 'missing_input' | 'formal_error' | 'undeclared_rule' | None

    def __post_init__(self) -> None:
        if self.error_kind is not None and self.error_kind not in ERROR_KINDS:
            raise ValueError(f"Unsupported error_kind {self.error_kind!r}; expected one of {ERROR_KINDS}.")
        clean_details = dict(self.details or {})
        canonical({"value": self.value, "details": clean_details})
        object.__setattr__(self, "details", clean_details)
        object.__setattr__(self, "assumptions", tuple(str(a) for a in (self.assumptions or ())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "unit": self.unit,
            "details": dict(self.details),
            "assumptions": list(self.assumptions),
            "error_code": self.error_code,
            "error_kind": self.error_kind,
        }


OperationExecutor = Callable[[Sequence[ClaimNode], Mapping[str, Any], DomainProfile], OperationOutput]


@dataclass(frozen=True)
class OperationSpec:
    """Application-declared executable operation specification."""

    name: str
    version: str
    family: str
    description: str
    inputs: tuple[dict[str, Any], ...]
    output_type: str
    executor: OperationExecutor = field(repr=False, compare=False)
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Operation name must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Operation version must be a non-empty string.")
        if self.family not in VERIFIER_FAMILIES or self.family == "open_semantics":
            raise ValueError(f"Controlled operations must belong to a deterministic verifier family, got {self.family!r}.")
        if not callable(self.executor):
            raise TypeError("OperationSpec.executor must be callable.")
        clean_inputs = tuple(dict(i) for i in (self.inputs or ()))
        canonical(clean_inputs)
        object.__setattr__(self, "inputs", clean_inputs)
        object.__setattr__(self, "aliases", tuple(str(a) for a in (self.aliases or ())))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "description": self.description,
            "inputs": list(self.inputs),
            "output_type": self.output_type,
            "aliases": list(self.aliases),
        }


class OperationCatalog:
    """Immutable-by-copy registry of controlled deterministic derivation operations."""

    def __init__(
        self,
        specs: Sequence[OperationSpec] = (),
        *,
        catalog_id: str = "default",
        version: str = "1.0.0",
    ) -> None:
        self.catalog_id = catalog_id
        self.version = version
        self._specs: dict[str, OperationSpec] = {}
        self._aliases: dict[str, str] = {}
        for spec in specs:
            self.register(spec)

    @staticmethod
    def _norm_op_key(name: str) -> str:
        return label_key(name).replace(" ", "_")

    def register(self, spec: OperationSpec) -> None:
        canon_key = self._norm_op_key(spec.name)
        if canon_key in self._specs:
            raise ValueError(f"Duplicate operation {spec.name!r} in OperationCatalog.")
        self._specs[canon_key] = spec
        self._aliases[canon_key] = canon_key
        for alias in spec.aliases:
            ak = self._norm_op_key(alias)
            if ak in self._aliases and self._aliases[ak] != canon_key:
                raise ValueError(f"Ambiguous operation alias {alias!r}.")
            self._aliases[ak] = canon_key

    def extend(
        self,
        specs: Sequence[OperationSpec],
        *,
        catalog_id: str | None = None,
        version: str | None = None,
    ) -> "OperationCatalog":
        new_cat = OperationCatalog(
            list(self._specs.values()),
            catalog_id=catalog_id or self.catalog_id,
            version=version or self.version,
        )
        for spec in specs:
            new_cat.register(spec)
        return new_cat

    def get(self, name: str) -> OperationSpec | None:
        if not isinstance(name, str) or not name.strip():
            return None
        key = self._aliases.get(self._norm_op_key(name))
        return self._specs.get(key) if key is not None else None

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(spec.name for spec in self._specs.values()))

    def describe_for_planner(self) -> list[dict[str, Any]]:
        return [self._specs[k].describe() for k in sorted(self._specs)]

    def fingerprint(self) -> str:
        return digest({
            "catalog_id": self.catalog_id,
            "version": self.version,
            "operations": self.describe_for_planner(),
        })

    def execute(
        self,
        operation: str,
        premises: Sequence[ClaimNode],
        parameters: Mapping[str, Any] | None = None,
        domain_profile: DomainProfile | None = None,
    ) -> OperationOutput:
        spec = self.get(operation)
        if spec is None:
            return OperationOutput(
                ok=False,
                error_code=f"UNKNOWN_OPERATION:{operation}",
                error_kind="undeclared_rule",
                details={"operation": operation, "catalog_id": self.catalog_id, "catalog_version": self.version},
            )
        profile = domain_profile or DomainProfile(profile_id="default")
        return spec.executor(tuple(premises), dict(parameters or {}), profile)


# ---------------------------------------------------------------------------
# Controlled Executors
# ---------------------------------------------------------------------------

def _exec_weekday_shift(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) == 0:
        return OperationOutput(
            ok=False,
            error_code="MISSING_WEEKDAY_ANCHOR",
            error_kind="missing_input",
        )
    if len(premises) == 1:
        if normalize_weekday(premises[0].value) is not None:
            return OperationOutput(ok=False, error_code="MISSING_WEEKDAY_OFFSET", error_kind="missing_input")
        return OperationOutput(ok=False, error_code="MISSING_WEEKDAY_ANCHOR", error_kind="missing_input")
    if len(premises) > 2:
        return OperationOutput(
            ok=False,
            error_code="TOO_MANY_PREMISES_FOR_WEEKDAY_SHIFT",
            error_kind="formal_error",
            details={"premise_count": len(premises)},
        )

    reg = profile.effective_unit_registry()
    p0, p1 = premises[0], premises[1]
    w0 = normalize_weekday(p0.value)
    w1 = normalize_weekday(p1.value)

    if w0 is not None and w1 is None:
        anchor_node, offset_node = p0, p1
        reordered = False
    elif w1 is not None and w0 is None:
        anchor_node, offset_node = p1, p0
        reordered = True
    elif w0 is None and w1 is None:
        return OperationOutput(
            ok=False,
            error_code="INVALID_OR_MISSING_WEEKDAY_ANCHOR",
            error_kind="missing_input",
            details={"premise_values": [p0.value, p1.value]},
        )
    else:
        return OperationOutput(
            ok=False,
            error_code="MISSING_WEEKDAY_OFFSET",
            error_kind="missing_input",
            details={"premise_values": [p0.value, p1.value]},
        )

    offset_days, off_err = normalize_day_offset(offset_node, registry=reg)
    if offset_days is None:
        err_kind = "formal_error" if (off_err or "").startswith(("INCOMPATIBLE_OFFSET_UNIT", "NON_INTEGER")) else "missing_input"
        return OperationOutput(
            ok=False,
            error_code=off_err or "INVALID_WEEKDAY_OFFSET",
            error_kind=err_kind,
            details={"offset_value": offset_node.value, "offset_unit": offset_node.unit},
        )

    res = compute_weekday_shift(anchor_node.value, offset_days)
    res["anchor_claim_id"] = anchor_node.claim_id
    res["offset_claim_id"] = offset_node.claim_id
    res["value_type"] = "weekday"
    if reordered:
        res["inputs_reordered_by_type"] = True
    return OperationOutput(ok=True, value=res["computed_weekday"], unit=None, details=res)


def _exec_date_offset(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) == 0:
        return OperationOutput(ok=False, error_code="MISSING_ANCHOR_DATE", error_kind="missing_input")
    if len(premises) == 1:
        b = _parse_date_bounds(str(premises[0].value or ""))
        if b is not None:
            return OperationOutput(ok=False, error_code="MISSING_DATE_OFFSET", error_kind="missing_input")
        return OperationOutput(ok=False, error_code="MISSING_ANCHOR_DATE", error_kind="missing_input")
    if len(premises) > 2:
        return OperationOutput(ok=False, error_code="TOO_MANY_PREMISES_FOR_DATE_OFFSET", error_kind="formal_error")

    reg = profile.effective_unit_registry()
    p0, p1 = premises[0], premises[1]
    b0 = _parse_date_bounds(p0.value if isinstance(p0.value, str) else "")
    b1 = _parse_date_bounds(p1.value if isinstance(p1.value, str) else "")
    if b0 is not None and b0[2] == "day" and b1 is None:
        anchor_node, offset_node = p0, p1
    elif b1 is not None and b1[2] == "day" and b0 is None:
        anchor_node, offset_node = p1, p0
    else:
        return OperationOutput(
            ok=False,
            error_code="INVALID_OR_MISSING_ANCHOR_DATE",
            error_kind="missing_input" if (b0 is None and b1 is None) else "formal_error",
        )

    offset_days, off_err = normalize_day_offset(offset_node, registry=reg)
    if offset_days is None:
        return OperationOutput(
            ok=False,
            error_code=off_err or "INVALID_DATE_OFFSET",
            error_kind="formal_error",
        )
    res = compute_date_offset(str(anchor_node.value), offset_days)
    res["anchor_claim_id"] = anchor_node.claim_id
    res["offset_claim_id"] = offset_node.claim_id
    res["value_type"] = "date"
    return OperationOutput(ok=True, value=res["computed_date"], unit=None, details=res)


def _exec_temporal_age(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if not premises:
        return OperationOutput(ok=False, error_code="MISSING_BIRTH_DATE", error_kind="missing_input")
    p0 = premises[0]
    start_raw = parameters.get("birth_date") or parameters.get("start_date") or p0.value or p0.valid_from
    end_raw = parameters.get("reference_date") or parameters.get("end_date")
    if end_raw is None and len(premises) >= 2:
        p1 = premises[1]
        end_raw = p1.value if isinstance(p1.value, str) else p1.valid_from
    if end_raw is None:
        return OperationOutput(ok=False, error_code="MISSING_REFERENCE_DATE", error_kind="missing_input")

    sb = _parse_date_bounds(start_raw)
    eb = _parse_date_bounds(end_raw)
    if sb is None or eb is None:
        return OperationOutput(ok=False, error_code="MISSING_OR_UNPARSEABLE_DATES", error_kind="missing_input")
    if sb[0] > eb[1]:
        return OperationOutput(ok=False, error_code="START_DATE_AFTER_REFERENCE_DATE", error_kind="formal_error")
    lo = _completed_years(sb[1], eb[0])
    hi = _completed_years(sb[0], eb[1])
    if lo == hi:
        return OperationOutput(ok=True, value=lo, unit="years", details={"birth_date": start_raw, "reference_date": end_raw})
    return OperationOutput(
        ok=True,
        value={"min": lo, "max": hi},
        unit="years",
        assumptions=(f"coarse_date_granularity:{sb[2]}->{eb[2]}",),
        details={"computed_interval": [lo, hi]},
    )


def _exec_temporal_duration(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) < 2:
        return OperationOutput(ok=False, error_code="MISSING_DURATION_DATES", error_kind="missing_input")
    s_raw = parameters.get("start_date") or (premises[0].value if isinstance(premises[0].value, str) else premises[0].valid_from)
    e_raw = parameters.get("end_date") or (premises[1].value if isinstance(premises[1].value, str) else premises[1].valid_from)
    sb = _parse_date_bounds(s_raw)
    eb = _parse_date_bounds(e_raw)
    if sb is None or eb is None:
        return OperationOutput(ok=False, error_code="MISSING_OR_UNPARSEABLE_DATES", error_kind="missing_input")
    if sb[0] > eb[1]:
        return OperationOutput(ok=False, error_code="START_DATE_AFTER_REFERENCE_DATE", error_kind="formal_error")

    unit_hint = str(parameters.get("duration_unit") or parameters.get("unit") or "days").lower().strip()
    if unit_hint in {"month", "months", "mes", "meses"}:
        lo, hi = _completed_months(sb[1], eb[0]), _completed_months(sb[0], eb[1])
        unit_out = "months"
    elif unit_hint in {"year", "years", "año", "años"}:
        lo, hi = _completed_years(sb[1], eb[0]), _completed_years(sb[0], eb[1])
        unit_out = "years"
    else:
        lo, hi = max(0, (eb[0] - sb[1]).days), max(0, (eb[1] - sb[0]).days)
        unit_out = "calendar_days"

    if lo == hi:
        return OperationOutput(ok=True, value=lo, unit=unit_out, details={"start_date": s_raw, "end_date": e_raw})
    return OperationOutput(
        ok=True,
        value={"min": lo, "max": hi},
        unit=unit_out,
        assumptions=(f"coarse_date_granularity:{sb[2]}->{eb[2]}",),
        details={"computed_interval": [lo, hi]},
    )


def _exec_additive(
    op: str,
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) < 2:
        return OperationOutput(
            ok=False,
            error_code="SUBTRACTION_REQUIRES_AT_LEAST_TWO_OPERANDS" if op == "sub" else "ADDITION_REQUIRES_AT_LEAST_TWO_OPERANDS",
            error_kind="missing_input",
        )
    reg = profile.effective_unit_registry()
    converted: list[Decimal] = []
    dims: list[str | None] = []
    raw_pairs: list[tuple[Decimal, str | None]] = []
    for p in premises:
        val, unit = _extract_quantity_pair(p)
        if val is None:
            return OperationOutput(
                ok=False,
                error_code="NON_NUMERIC_ARITHMETIC_OPERAND",
                error_kind="missing_input",
                details={"claim_id": p.claim_id, "value": p.value},
            )
        try:
            base_v, dim, _ = _convert_to_base_decimal(val, unit, reg)
        except UnmappedProposal as exc:
            return OperationOutput(
                ok=False,
                error_code=f"UNKNOWN_UNIT:{exc.detail}",
                error_kind="formal_error",
            )
        converted.append(base_v)
        dims.append(dim)
        raw_pairs.append((val, unit))

    non_null_dims = {d for d in dims if d is not None}
    if len(non_null_dims) > 1 or (len(non_null_dims) == 1 and any(d is None for d in dims)):
        return OperationOutput(
            ok=False,
            error_code="ARITHMETIC_INCOMPATIBLE_UNIT_DIMENSIONS",
            error_kind="formal_error",
            details={"premise_dimensions": dims},
        )

    expected_dim = next(iter(non_null_dims), None)
    if op == "add":
        res = sum(converted, Decimal(0))
    else:
        res = converted[0] - sum(converted[1:], Decimal(0))

    if expected_dim is not None:
        if op == "sub" and expected_dim == "ratio":
            out_unit: str | None = "percentage_point"
        elif op == "sub" and expected_dim == "annual_rate":
            out_unit = "percentage_points_per_annum"
        else:
            out_unit = reg.base(expected_dim)
    else:
        out_unit = None

    return OperationOutput(
        ok=True,
        value=_decimal_to_json(res),
        unit=out_unit,
        details={
            "operation": op,
            "operands_base": [str(v) for v in converted],
            "dimension": expected_dim,
            "computed_exact": str(res),
        },
    )


def _exec_mul(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) < 2:
        return OperationOutput(ok=False, error_code="MULTIPLICATION_REQUIRES_AT_LEAST_TWO_OPERANDS", error_kind="missing_input")
    reg = profile.effective_unit_registry()
    pairs = [_extract_quantity_pair(p) for p in premises]
    if any(v is None for v, _ in pairs):
        return OperationOutput(ok=False, error_code="NON_NUMERIC_ARITHMETIC_OPERAND", error_kind="missing_input")

    dims: list[str | None] = []
    for _, u in pairs:
        if u:
            try:
                dims.append(reg.lookup(u).dimension)
            except UnmappedProposal as exc:
                return OperationOutput(ok=False, error_code=f"UNKNOWN_UNIT:{exc.detail}", error_kind="formal_error")
        else:
            dims.append(None)

    non_rate_dims = {d for d in dims if d is not None and not reg.is_rate_or_ratio_dimension(d)}
    has_rate_dim = any(reg.is_rate_or_ratio_dimension(d) for d in dims)
    out_unit: str | None = None
    res = Decimal(1)
    if len(non_rate_dims) == 1 and has_rate_dim:
        base_dim = next(iter(non_rate_dims))
        out_unit = reg.base(base_dim)
        for (raw_v, raw_u), dim in zip(pairs, dims):
            assert raw_v is not None
            if reg.is_rate_or_ratio_dimension(dim) and raw_u:
                factor = reg.rate_factor_to_unitless(raw_u)
                assert factor is not None
                res *= raw_v * factor
            else:
                bv, _, _ = _convert_to_base_decimal(raw_v, raw_u, reg)
                res *= bv
    elif not any(d is not None for d in dims):
        for raw_v, _ in pairs:
            assert raw_v is not None
            res *= raw_v
    else:
        for (raw_v, raw_u), dim in zip(pairs, dims):
            assert raw_v is not None
            if reg.is_rate_or_ratio_dimension(dim) and raw_u:
                factor = reg.rate_factor_to_unitless(raw_u)
                assert factor is not None
                res *= raw_v * factor
            else:
                res *= raw_v
        if len(non_rate_dims) == 1:
            out_unit = reg.base(next(iter(non_rate_dims)))
    return OperationOutput(
        ok=True,
        value=_decimal_to_json(res),
        unit=out_unit,
        details={"computed_exact": str(res), "dimensions": dims},
    )


def _exec_div(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) != 2:
        return OperationOutput(ok=False, error_code="DIVISION_REQUIRES_TWO_OPERANDS", error_kind="missing_input")
    reg = profile.effective_unit_registry()
    (n_val, n_unit), (d_val, d_unit) = _extract_quantity_pair(premises[0]), _extract_quantity_pair(premises[1])
    if n_val is None or d_val is None:
        return OperationOutput(ok=False, error_code="NON_NUMERIC_ARITHMETIC_OPERAND", error_kind="missing_input")
    if d_val == 0:
        return OperationOutput(ok=False, error_code="DIVISION_BY_ZERO", error_kind="formal_error")
    try:
        nb, ndim, _ = _convert_to_base_decimal(n_val, n_unit, reg)
        db, ddim, _ = _convert_to_base_decimal(d_val, d_unit, reg)
    except UnmappedProposal as exc:
        return OperationOutput(ok=False, error_code=f"UNKNOWN_UNIT:{exc.detail}", error_kind="formal_error")
    if ndim is not None and ndim == ddim:
        res = nb / db
        target_u = parameters.get("target_unit")
        if target_u and reg.lookup(str(target_u)).dimension == "ratio":
            res = res * Decimal(100)
            return OperationOutput(ok=True, value=_decimal_to_json(res), unit="percent", details={"computed_exact": str(res)})
        return OperationOutput(ok=True, value=_decimal_to_json(res), unit=None, details={"computed_exact": str(res)})
    if ndim is not None and ddim is None:
        res = nb / db
        return OperationOutput(ok=True, value=_decimal_to_json(res), unit=reg.base(ndim), details={"computed_exact": str(res)})
    res = n_val / d_val
    return OperationOutput(ok=True, value=_decimal_to_json(res), unit=None, details={"computed_exact": str(res)})


def _exec_pct(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) != 2:
        return OperationOutput(ok=False, error_code="PERCENTAGE_REQUIRES_BASE_AND_RATE", error_kind="missing_input")
    reg = profile.effective_unit_registry()
    (base_v, base_u), (rate_v, rate_u) = _extract_quantity_pair(premises[0]), _extract_quantity_pair(premises[1])
    if base_v is None or rate_v is None:
        return OperationOutput(ok=False, error_code="NON_NUMERIC_ARITHMETIC_OPERAND", error_kind="missing_input")
    try:
        conv_base, base_dim, base_unit = _convert_to_base_decimal(base_v, base_u, reg)
        if rate_u:
            factor = reg.rate_factor_to_unitless(rate_u)
            if factor is None:
                return OperationOutput(ok=False, error_code="PERCENTAGE_RATE_INVALID_UNIT", error_kind="formal_error")
            rate_factor = rate_v * factor
        else:
            rate_factor = rate_v if parameters.get("rate_is_fraction") else rate_v / Decimal(100)
    except UnmappedProposal as exc:
        return OperationOutput(ok=False, error_code=f"UNKNOWN_UNIT:{exc.detail}", error_kind="formal_error")
    res = conv_base * rate_factor
    return OperationOutput(ok=True, value=_decimal_to_json(res), unit=base_unit, details={"computed_exact": str(res)})


def _exec_comparison(
    op: str,
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if not premises:
        return OperationOutput(ok=False, error_code="MISSING_COMPARISON_PREMISES", error_kind="missing_input")
    reg = profile.effective_unit_registry()
    l_val, l_unit = _extract_quantity_pair(premises[0])
    if len(premises) >= 2:
        r_val, r_unit = _extract_quantity_pair(premises[1])
    elif "threshold" in parameters:
        r_val, r_unit = _to_decimal(parameters["threshold"]), parameters.get("threshold_unit") or l_unit
    else:
        return OperationOutput(ok=False, error_code="MISSING_COMPARISON_RHS", error_kind="missing_input")
    if l_val is None or r_val is None:
        return OperationOutput(ok=False, error_code="NON_NUMERIC_COMPARISON_OPERANDS", error_kind="formal_error")
    try:
        lb, ldim, _ = _convert_to_base_decimal(l_val, l_unit, reg)
        rb, rdim, _ = _convert_to_base_decimal(r_val, r_unit, reg)
    except UnmappedProposal as exc:
        return OperationOutput(ok=False, error_code=f"UNKNOWN_COMPARISON_UNIT:{exc.detail}", error_kind="formal_error")
    if ldim is not None and rdim is not None and ldim != rdim:
        return OperationOutput(ok=False, error_code="COMPARISON_UNIT_DIMENSION_MISMATCH", error_kind="formal_error")
    holds = {
        "eq": lb == rb,
        "ne": lb != rb,
        "lt": lb < rb,
        "le": lb <= rb,
        "gt": lb > rb,
        "ge": lb >= rb,
    }[op]
    return OperationOutput(
        ok=True,
        value=holds,
        unit=None,
        details={"lhs_base": str(lb), "rhs_base": str(rb), "op": op, "holds": holds},
    )


def _exec_unit_conversion(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) != 1:
        return OperationOutput(ok=False, error_code="UNIT_CONVERSION_REQUIRES_ONE_PREMISE", error_kind="missing_input")
    target_unit = parameters.get("target_unit")
    if not target_unit:
        return OperationOutput(ok=False, error_code="MISSING_TARGET_UNIT", error_kind="missing_input")
    reg = profile.effective_unit_registry()
    val, src_unit = _extract_quantity_pair(premises[0])
    if val is None or not src_unit:
        return OperationOutput(ok=False, error_code="UNIT_CONVERSION_MISSING_SOURCE_QUANTITY", error_kind="missing_input")
    try:
        src_u = reg.lookup(src_unit)
        tgt_u = reg.lookup(str(target_unit))
    except UnmappedProposal as exc:
        return OperationOutput(ok=False, error_code=f"UNKNOWN_UNIT:{exc.detail}", error_kind="formal_error")
    if src_u.dimension != tgt_u.dimension:
        return OperationOutput(
            ok=False,
            error_code="UNIT_DIMENSION_MISMATCH",
            error_kind="formal_error",
            details={"source_dimension": src_u.dimension, "target_dimension": tgt_u.dimension},
        )
    converted = (val * Decimal(src_u.factor)) / Decimal(tgt_u.factor)
    return OperationOutput(
        ok=True,
        value=_decimal_to_json(converted),
        unit=tgt_u.name,
        details={"source_value": str(val), "source_unit": src_u.name, "target_unit": tgt_u.name},
    )


def _exec_identity(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    if len(premises) != 1:
        return OperationOutput(ok=False, error_code="IDENTITY_REQUIRES_ONE_PREMISE", error_kind="missing_input")
    p = premises[0]
    return OperationOutput(
        ok=True,
        value=p.value,
        unit=p.unit,
        details={"source_claim_id": p.claim_id},
    )


def _exec_rule_application(
    premises: Sequence[ClaimNode],
    parameters: Mapping[str, Any],
    profile: DomainProfile,
) -> OperationOutput:
    rule_id = parameters.get("rule_id")
    if not rule_id:
        return OperationOutput(ok=False, error_code="MISSING_RULE_ID", error_kind="undeclared_rule")
    rule = profile.declared_rules.get(str(rule_id))
    if rule is None:
        return OperationOutput(
            ok=False,
            error_code=f"UNDECLARED_RULE:{rule_id}",
            error_kind="undeclared_rule",
            details={"rule_id": str(rule_id)},
        )
    req_ver = parameters.get("rule_version")
    if req_ver is not None and str(req_ver) != rule.version:
        return OperationOutput(
            ok=False,
            error_code=f"RULE_VERSION_MISMATCH:{rule_id}@{req_ver}",
            error_kind="undeclared_rule",
        )
    val = rule.conclusion_template.get("value")
    unit = rule.conclusion_template.get("unit")
    return OperationOutput(
        ok=True,
        value=val,
        unit=unit,
        details={"rule_id": rule.rule_id, "rule_version": rule.version, "rule_kind": rule.kind},
    )


def build_default_operation_catalog() -> OperationCatalog:
    """Build the default controlled operation catalog aligned with v4 verifier families."""
    return OperationCatalog(
        [
            OperationSpec(
                name="weekday_shift",
                version="1.0.0",
                family="temporal",
                description=(
                    "Shift a canonical weekday anchor by a signed integer day offset (negative for before/earlier, "
                    "positive for after/later), wrapping across week boundaries modulo 7."
                ),
                inputs=(
                    {"role": "anchor_weekday", "type": "weekday", "required": True},
                    {"role": "offset_days", "type": "integer_quantity", "dimensions": ["duration_calendar_days"], "required": True},
                ),
                output_type="weekday",
                aliases=("day_of_week_shift",),
                executor=_exec_weekday_shift,
            ),
            OperationSpec(
                name="date_offset",
                version="1.0.0",
                family="temporal",
                description="Shift an ISO YYYY-MM-DD anchor date by a signed integer number of calendar days or weeks.",
                inputs=(
                    {"role": "anchor_date", "type": "date", "required": True},
                    {"role": "offset_days", "type": "integer_quantity", "dimensions": ["duration_calendar_days"], "required": True},
                ),
                output_type="date",
                aliases=("calendar_date_shift",),
                executor=_exec_date_offset,
            ),
            OperationSpec(
                name="temporal_age",
                version="1.0.0",
                family="temporal",
                description="Compute completed years between a birth/start date premise and an explicit reference date premise.",
                inputs=(
                    {"role": "birth_date", "type": "date", "required": True},
                    {"role": "reference_date", "type": "date", "required": True},
                ),
                output_type="quantity",
                aliases=("age",),
                executor=_exec_temporal_age,
            ),
            OperationSpec(
                name="temporal_duration",
                version="1.0.0",
                family="temporal",
                description="Compute exact duration (days, months, or years) between two explicit date premises.",
                inputs=(
                    {"role": "start_date", "type": "date", "required": True},
                    {"role": "end_date", "type": "date", "required": True},
                ),
                output_type="quantity",
                aliases=("duration",),
                executor=_exec_temporal_duration,
            ),
            OperationSpec(
                name="sub",
                version="1.0.0",
                family="arithmetic",
                description="Subtract premise[1..] from premise[0] using exact Decimal arithmetic and compatible UnitRegistry dimensions.",
                inputs=(
                    {"role": "minuend", "type": "number_or_quantity", "required": True},
                    {"role": "subtrahend", "type": "number_or_quantity", "required": True},
                ),
                output_type="number_or_quantity",
                aliases=("subtract", "diff"),
                executor=lambda prems, params, prof: _exec_additive("sub", prems, params, prof),
            ),
            OperationSpec(
                name="add",
                version="1.0.0",
                family="arithmetic",
                description="Add two or more numeric or unit-compatible premises using exact Decimal arithmetic.",
                inputs=(
                    {"role": "term_0", "type": "number_or_quantity", "required": True},
                    {"role": "term_1", "type": "number_or_quantity", "required": True},
                ),
                output_type="number_or_quantity",
                aliases=("sum",),
                executor=lambda prems, params, prof: _exec_additive("add", prems, params, prof),
            ),
            OperationSpec(
                name="mul",
                version="1.0.0",
                family="arithmetic",
                description="Multiply two or more numeric/rate premises using exact Decimal arithmetic.",
                inputs=(
                    {"role": "factor_0", "type": "number_or_quantity", "required": True},
                    {"role": "factor_1", "type": "number_or_quantity", "required": True},
                ),
                output_type="number_or_quantity",
                aliases=("multiply", "prod"),
                executor=_exec_mul,
            ),
            OperationSpec(
                name="div",
                version="1.0.0",
                family="arithmetic",
                description="Divide premise[0] by non-zero premise[1] using exact Decimal arithmetic.",
                inputs=(
                    {"role": "numerator", "type": "number_or_quantity", "required": True},
                    {"role": "denominator", "type": "number_or_quantity", "required": True},
                ),
                output_type="number_or_quantity",
                aliases=("divide", "ratio"),
                executor=_exec_div,
            ),
            OperationSpec(
                name="pct",
                version="1.0.0",
                family="arithmetic",
                description="Apply percentage/rate premise[1] to base quantity premise[0].",
                inputs=(
                    {"role": "base", "type": "number_or_quantity", "required": True},
                    {"role": "rate", "type": "rate_or_number", "required": True},
                ),
                output_type="number_or_quantity",
                aliases=("percentage_of",),
                executor=_exec_pct,
            ),
            OperationSpec(
                name="eq",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned equality between two premises (or premise and threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("eq", prems, params, prof),
            ),
            OperationSpec(
                name="ne",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned inequality between two premises (or premise and threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("ne", prems, params, prof),
            ),
            OperationSpec(
                name="lt",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned strict less-than between premise[0] and premise[1] (or threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("lt", prems, params, prof),
            ),
            OperationSpec(
                name="le",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned less-than-or-equal between premise[0] and premise[1] (or threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("le", prems, params, prof),
            ),
            OperationSpec(
                name="gt",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned strict greater-than between premise[0] and premise[1] (or threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("gt", prems, params, prof),
            ),
            OperationSpec(
                name="ge",
                version="1.0.0",
                family="comparison",
                description="Check unit-aligned greater-than-or-equal between premise[0] and premise[1] (or threshold).",
                inputs=(
                    {"role": "lhs", "type": "number_or_quantity", "required": True},
                    {"role": "rhs", "type": "number_or_quantity", "required": True},
                ),
                output_type="boolean",
                executor=lambda prems, params, prof: _exec_comparison("ge", prems, params, prof),
            ),
            OperationSpec(
                name="unit_conversion",
                version="1.0.0",
                family="quantities",
                description="Convert a single premise quantity into parameters['target_unit'] within the same UnitRegistry dimension.",
                inputs=({"role": "quantity", "type": "quantity", "required": True},),
                output_type="quantity",
                executor=_exec_unit_conversion,
            ),
            OperationSpec(
                name="identity",
                version="1.0.0",
                family="explicit_logic",
                description="Restate a single grounded premise value unchanged.",
                inputs=({"role": "premise", "type": "any", "required": True},),
                output_type="any",
                executor=_exec_identity,
            ),
            OperationSpec(
                name="rule_application",
                version="1.0.0",
                family="explicit_logic",
                description="Apply a declared versioned rule registered in DomainProfile.declared_rules.",
                inputs=({"role": "premises", "type": "any", "required": True},),
                output_type="any",
                executor=_exec_rule_application,
            ),
        ],
        catalog_id="deltoryx_v4_core",
        version="1.0.0",
    )


DEFAULT_OPERATION_CATALOG = build_default_operation_catalog()
