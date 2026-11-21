"""Application-declared domain vocabulary: the shared contract between reader and engine.

Free-form keys and ad-hoc unit strings made v3 comparisons depend on a model
reproducing hidden labels verbatim. v4 requires every admitted proposition to
conform to a versioned vocabulary declared by the application. Normalization is
deterministic and limited to declared labels: no similarity matching, no
natural-language parsing and no unit conversion that the registry does not
explicitly define.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping

from .normalizers import (
    NORMALIZER_VERSION,
    NormalizationTrace,
    normalize_date_token,
    normalize_percentage,
    normalize_separator_number,
    normalize_time_interval,
    normalize_value_with_provenance,
)
from ..semantic.models import KINDS, canonical, digest, instant, require_string

VALUE_TYPES = frozenset({'quantity', 'number', 'bound', 'boolean', 'enum', 'set', 'list', 'string', 'entity', 'date', 'record'})
BOUND_OPERATORS = frozenset({'lt', 'le', 'gt', 'ge'})
CONDITION_OPERATORS = frozenset({'eq', 'ne', 'lt', 'le', 'gt', 'ge'})
CONDITION_STATUSES = frozenset({'unconditional', 'conditional', 'unknown_condition', 'unrepresentable_condition'})


class UnmappedProposal(ValueError):
    """A proposal cannot be represented in the declared vocabulary."""

    def __init__(self, reason: str, path: str = '', detail: Any = None):
        self.reason, self.path, self.detail = reason, path, detail
        super().__init__(f'{reason} at {path or "/"}' + ('' if detail is None else f': {detail}'))

    def as_dict(self):
        return {'reason': self.reason, 'path': self.path, 'detail': self.detail}


def label_key(text: str) -> str:
    """Canonical key for a declared label: case, spacing, '_' and '-' are not meaningful."""
    return re.sub(r'[\s_\-]+', ' ', text.strip().casefold())


def decimal_text(value: Any, path: str) -> str:
    if type(value) is bool or type(value) not in (int, float, str, Decimal):
        raise UnmappedProposal('number_required', path, value)
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        norm = normalize_separator_number(value) if isinstance(value, str) else None
        if norm is None:
            raise UnmappedProposal('number_required', path, value) from None
        return norm
    if not number.is_finite():
        raise UnmappedProposal('finite_number_required', path, value)
    normalized = number.normalize()
    if normalized == normalized.to_integral_value():
        normalized = normalized.quantize(Decimal(1))
    return format(normalized, 'f')


# ------------------------------------------------------------------- dates
# A date value keeps the precision its source states. Comparing values of different precision, or a local
# time with an instant, is not a contradiction unless no reading of the coarser value can match the finer one.
_LOCAL_SLACK = (timedelta(hours=14), timedelta(hours=12))  # UTC offsets range from -12:00 to +14:00


def normalize_date(value: Any) -> str:
    """Canonical ISO text: YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS (local) or ...Z (instant)."""
    if not isinstance(value, str):
        raise ValueError('date text required')
    text = value.strip()
    if re.fullmatch(r'\d{4}', text):
        return text
    if re.fullmatch(r'\d{4}-\d{2}', text):
        date(int(text[:4]), int(text[5:7]), 1)
        return text
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', text):
        return date.fromisoformat(text).isoformat()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?', text):
        raise ValueError('not an ISO 8601 date')
    moment = datetime.fromisoformat(text.replace('Z', '+00:00').replace(' ', 'T', 1))
    if moment.tzinfo is None:  # local time without offset: kept as stated, never shifted
        return moment.replace(microsecond=0).isoformat()
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')


def date_precision(text: str) -> str:
    if len(text) == 4:
        return 'year'
    if len(text) == 7:
        return 'month'
    if len(text) == 10:
        return 'day'
    return 'instant' if text.endswith('Z') else 'local_time'


def _date_span(text: str):
    """(earliest, latest) naive-UTC instants the value can denote, allowing any UTC offset for local values."""
    precision = date_precision(text)
    if precision == 'instant':
        moment = datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ')
        return moment, moment
    if precision == 'local_time':
        moment = datetime.fromisoformat(text)
        return moment - _LOCAL_SLACK[0], moment + _LOCAL_SLACK[1]
    year = int(text[:4])
    if precision == 'year':
        start, end = datetime(year, 1, 1), datetime(year + 1, 1, 1)
    elif precision == 'month':
        month = int(text[5:7])
        start = datetime(year, month, 1)
        end = datetime(year + (month == 12), month % 12 + 1, 1)
    else:
        start = datetime.fromisoformat(text)
        end = start + timedelta(days=1)
    return start - _LOCAL_SLACK[0], end + _LOCAL_SLACK[1]


def date_relation(a: str, b: str) -> str:
    """'equal' | 'compatible' (one refines the other) | 'unknown' (time zone undecidable) | 'incompatible'."""
    if a == b:
        return 'equal'
    pa, pb = date_precision(a), date_precision(b)
    calendar = {'year', 'month', 'day', 'local_time'}
    if pa in calendar and pb in calendar:
        if pa == pb:
            return 'incompatible'
        coarse, fine = (a, b) if len(a) < len(b) else (b, a)
        return 'compatible' if fine.startswith(coarse) else 'incompatible'
    if pa == pb == 'instant':
        return 'incompatible'
    local, moment = (a, b) if pb == 'instant' else (b, a)
    start, end = _date_span(local)
    instant_value = datetime.strptime(moment, '%Y-%m-%dT%H:%M:%SZ')
    if not start <= instant_value <= end:
        return 'incompatible'
    if date_precision(local) != 'local_time' and moment.startswith(local):
        return 'compatible'  # the instant falls inside the stated calendar period in UTC as well
    return 'unknown'


def _text_relation(a: str, b: str) -> str:
    ta, tb = normalize_time_interval(a), normalize_time_interval(b)
    if ta is not None and tb is not None:
        return 'equal' if ta == tb else 'incompatible'
    ka, kb = label_key(a), label_key(b)
    if ka == kb:
        return 'equal'
    short, long_ = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if short and re.search(r'(?<!\w)' + re.escape(short) + r'(?!\w)', long_):
        return 'compatible'  # one names the other more fully ('Ridgeline' / 'Ridgeline Capital Advisors LLC')
    return 'incompatible'


def value_relation(spec: 'FieldSpec', a: Any, b: Any) -> str:
    """Relation between two canonical values of one schema; used only where single values would conflict.

    'equal', 'compatible' (one value states the other less precisely or less fully), 'unknown' (cannot be
    decided without information the values do not carry) or 'incompatible'."""
    if canonical(a) == canonical(b):
        return 'equal'
    if spec.type == 'date' and isinstance(a, str) and isinstance(b, str):
        return date_relation(a, b)
    if spec.type in {'string', 'entity'} and isinstance(a, str) and isinstance(b, str):
        return _text_relation(a, b)
    if spec.type == 'set' and isinstance(a, list) and isinstance(b, list):
        return 'compatible' if set(a) <= set(b) or set(b) <= set(a) else 'incompatible'
    if spec.type == 'list' and isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 'incompatible'
        if not a:
            return 'equal'
        if spec.element is not None:
            relations = [value_relation(spec.element, ai, bi) for ai, bi in zip(a, b)]
        else:
            relations = ['equal' if canonical(ai) == canonical(bi) else 'incompatible' for ai, bi in zip(a, b)]
        for verdict in ('incompatible', 'unknown', 'compatible'):
            if verdict in relations:
                return verdict
        return 'equal'
    if spec.type == 'record' and isinstance(a, dict) and isinstance(b, dict):
        relations = []
        for name, child in spec.fields:
            if name in a and name in b:
                relations.append(value_relation(child, a[name], b[name]))
        for verdict in ('incompatible', 'unknown', 'compatible'):
            if verdict in relations:
                return verdict
        return 'equal' if relations else 'unknown'
    return 'incompatible'


@dataclass(frozen=True)
class Unit:
    """A declared unit. value_in_base = value * factor within one dimension."""
    name: str
    dimension: str
    factor: str = '1'
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        require_string(self.name, 'unit name')
        require_string(self.dimension, 'unit dimension')
        if Decimal(self.factor) <= 0:
            raise ValueError('Unit factors must be positive')


class UnitRegistry:
    def __init__(self, units=()):
        self._units: dict[str, Unit] = {}
        self._labels: dict[str, str] = {}
        self._base: dict[str, str] = {}
        for unit in units:
            self.add(unit)

    def add(self, unit: Unit):
        if unit.name in self._units:
            raise ValueError(f'Duplicate unit {unit.name}')
        for label in (unit.name, *unit.aliases):
            key = label_key(label)
            if key in self._labels and self._labels[key] != unit.name:
                raise ValueError(f'Ambiguous unit label {label!r}')
            self._labels[key] = unit.name
        self._units[unit.name] = unit
        if Decimal(unit.factor) == 1:
            if unit.dimension in self._base and self._base[unit.dimension] != unit.name:
                raise ValueError(f'Two base units for dimension {unit.dimension}')
            self._base[unit.dimension] = unit.name

    def extend(self, units):
        registry = UnitRegistry(self._units.values())
        for unit in units:
            registry.add(unit)
        return registry

    def with_aliases(self, aliases: Mapping[str, tuple[str, ...]]) -> 'UnitRegistry':
        """Return a NEW registry whose units carry extra labels. Names, dimensions and
        factors are unchanged, and the receiver (and its fingerprint) is not modified."""
        unknown = sorted(set(aliases) - set(self._units))
        if unknown:
            raise ValueError(f'Aliases given for undeclared units: {unknown}')
        units = []
        for name, unit in self._units.items():
            extra = tuple(a for a in aliases.get(name, ()) if a not in unit.aliases)
            units.append(replace(unit, aliases=unit.aliases + extra) if extra else unit)
        return UnitRegistry(units)

    def lookup(self, label: str | None, path: str = 'unit') -> Unit:
        if not isinstance(label, str) or not label.strip():
            raise UnmappedProposal('unit_required', path, label)
        name = self._labels.get(label_key(label))
        if name is None:
            raise UnmappedProposal('unknown_unit', path, label)
        return self._units[name]

    def base(self, dimension: str) -> str:
        if dimension not in self._base:
            raise ValueError(f'Dimension {dimension} has no declared base unit')
        return self._base[dimension]

    def to_base(self, value: Any, label: str, path: str) -> tuple[str, str]:
        unit = self.lookup(label, path)
        number = Decimal(decimal_text(value, path)) * Decimal(unit.factor)
        return decimal_text(number, path), self.base(unit.dimension)

    def dimension_of(self, unit_name: str) -> str:
        return self._units[unit_name].dimension

    @staticmethod
    def is_rate_or_ratio_dimension(dimension: str | None) -> bool:
        return dimension in {
            'ratio',
            'ratio_difference',
            'annual_rate',
            'annual_rate_difference',
        }

    def rate_factor_to_unitless(self, label: str | None, path: str = 'unit') -> Decimal | None:
        """Convert a rate/ratio/spread unit into a unitless multiplicative factor (e.g. 25 bps -> 0.0025)."""
        if not label:
            return None
        unit = self.lookup(label, path)
        if not self.is_rate_or_ratio_dimension(unit.dimension):
            return None
        return Decimal(unit.factor) / Decimal(100)

    def describe(self):
        return [asdict(u) for u in sorted(self._units.values(), key=lambda u: u.name)]


# Conversions are declared only where they are exact. Calendar days, business days and
# months are different dimensions: converting between them requires a calendar policy.
STANDARD_UNITS = UnitRegistry([
    Unit('percent', 'ratio', '1', ('%', 'pct', 'per cent')),
    Unit('fraction', 'ratio', '100', ()),
    Unit('percentage_point', 'ratio_difference', '1', ('pp', 'percentage points', 'percentage point')),
    Unit('basis_point', 'ratio_difference', '0.01', ('bp', 'bps', 'basis points')),
    Unit('percent_per_annum', 'annual_rate', '1', ('% per annum', '% p.a.', 'percent per annum', 'percent per year', 'percent annually')),
    Unit('percentage_points_per_annum', 'annual_rate_difference', '1', ('percentage points per annum', 'pp per annum')),
    Unit('basis_points_per_annum', 'annual_rate_difference', '0.01', ('bps per annum', 'bp/yr', 'bps/yr', 'basis points per annum')),
    Unit('multiple', 'multiplier', '1', ('times', 'x', 'multiplier')),
    Unit('calendar_days', 'duration_calendar_days', '1', ('days', 'day', 'calendar day')),
    Unit('weeks', 'duration_calendar_days', '7', ('week',)),
    Unit('business_days', 'duration_business_days', '1', ('business day', 'working days', 'working day')),
    Unit('months', 'duration_months', '1', ('month',)),
    Unit('years', 'duration_months', '12', ('year',)),
    Unit('hours', 'duration_hours', '1', ('hour',)),
    Unit('hours_per_week', 'hours_per_week', '1', ('hours/week', 'hours a week')),
    Unit('hours_per_day', 'hours_per_day', '1', ('hours/day', 'hours a day')),
    Unit('count', 'count', '1', ('items', 'number')),
    Unit('USD', 'currency:USD', '1', ('US dollars', 'dollars', '$')),
    Unit('EUR', 'currency:EUR', '1', ('euros', 'euro', '€')),
    Unit('GBP', 'currency:GBP', '1', ('pounds sterling', '£')),
    Unit('USD_per_year', 'currency_rate:USD/year', '1', ('USD per year', 'USD annually', 'USD per annum')),
    Unit('USD_per_month', 'currency_rate:USD/year', '12', ('USD per month', 'USD monthly')),
    Unit('EUR_per_year', 'currency_rate:EUR/year', '1', ('EUR per year', 'EUR per annum')),
    Unit('EUR_per_month', 'currency_rate:EUR/year', '12', ('EUR per month',)),
])


# Spanish-language unit labels, kept OUT of STANDARD_UNITS on purpose: STANDARD_UNITS is
# fingerprinted by frozen vocabularies and preregistrations (e.g. PR-MCI phase 3), so any
# new label there is a compatibility break. This profile only adds labels; unit names,
# dimensions and conversion factors are identical to STANDARD_UNITS.
SPANISH_UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    'percent': ('por ciento', 'porcentaje'),
    'calendar_days': ('día', 'días', 'dia', 'dias', 'día natural', 'días naturales'),
    'weeks': ('semana', 'semanas'),
    'business_days': ('día hábil', 'días hábiles', 'dia habil', 'dias habiles', 'día laborable', 'días laborables'),
    'months': ('mes', 'meses'),
    'years': ('año', 'años', 'ano', 'anos'),
    'hours': ('hora', 'horas'),
    'USD': ('dólares', 'dolares', 'dólar', 'dolar'),
    'GBP': ('libras', 'libra'),
}
SPANISH_UNITS = STANDARD_UNITS.with_aliases(SPANISH_UNIT_ALIASES)

# Named, explicit unit profiles. Callers select one by name; nothing mutates STANDARD_UNITS.
UNIT_PROFILES: dict[str, UnitRegistry] = {
    'standard': STANDARD_UNITS,
    'es': SPANISH_UNITS,
}


@dataclass(frozen=True)
class FieldSpec:
    """Typed value schema. Quantities carry a unit restricted to declared dimensions."""
    type: str
    dimensions: tuple[str, ...] = ()
    enum: tuple[str, ...] = ()
    fields: tuple[tuple[str, 'FieldSpec'], ...] = ()
    required: tuple[str, ...] | None = None
    element: 'FieldSpec | None' = None
    min_items: int = 0
    description: str = ''

    def __post_init__(self):
        if self.type not in VALUE_TYPES:
            raise ValueError(f'Unknown value type {self.type}')
        if self.type in {'quantity', 'bound'} and not self.dimensions:
            raise ValueError('Quantities and bounds need declared dimensions')
        if self.type in {'enum', 'set'} and not self.enum:
            raise ValueError('Enum and set values need declared members')
        if self.type == 'record' and not self.fields:
            raise ValueError('Records need declared fields')
        if self.type == 'list':
            if not isinstance(self.element, FieldSpec):
                raise ValueError('List values need an element FieldSpec')
            if self.min_items < 0:
                raise ValueError('List min_items cannot be negative')
        elif self.element is not None or self.min_items:
            raise ValueError('element and min_items apply only to list values')

    @classmethod
    def record(cls, description='', required=None, **fields):
        return cls('record', fields=tuple(sorted(fields.items())), required=None if required is None else tuple(sorted(required)), description=description)

    @classmethod
    def list(cls, element: 'FieldSpec', *, min_items: int = 1, description: str = ''):
        """Construct an ordered list FieldSpec with typed elements."""
        return cls('list', element=element, min_items=min_items, description=description)

    SHAPES = {
        'quantity': 'number in "value" with the unit name in "unit"; inside records or conditions use {"value": number, "unit": unit_name}',
        'bound': '{"operator": "lt"|"le"|"gt"|"ge", "value": number, "unit": unit_name}',
        'number': 'a JSON number, no unit',
        'boolean': 'true or false',
        'enum': 'exactly one of the listed enum members (string)',
        'set': 'JSON array of listed enum members',
        'list': 'ordered JSON array whose elements each conform to "element"',
        'string': 'text, compared verbatim after whitespace normalization',
        'entity': 'the name of an entity (string)',
        'date': 'ISO 8601 string at the precision the text states: YYYY, YYYY-MM, YYYY-MM-DD, or a date-time '
                'such as 2023-03-08T02:14:00; add the offset (2023-03-08T02:14:00-07:00) only when the text states '
                'the time zone. Never invent a day or month the text does not give.',
        'record': 'JSON object with exactly the listed fields',
    }

    def describe(self):
        out = {'type': self.type, 'json_shape': self.SHAPES[self.type]}
        if self.dimensions:
            out['dimensions'] = list(self.dimensions)
        if self.enum:
            out['enum'] = list(self.enum)
        if self.fields:
            out['fields'] = {k: v.describe() for k, v in self.fields}
            out['required'] = list(self.required) if self.required is not None else [k for k, _ in self.fields]
        if self.element is not None:
            out['element'] = self.element.describe()
        if self.min_items:
            out['min_items'] = self.min_items
        if self.description:
            out['description'] = self.description
        return out


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    value: FieldSpec
    cardinality: str = 'single'
    conditions: tuple[tuple[str, FieldSpec], ...] = ()
    assertion_kinds: tuple[str, ...] = tuple(sorted(KINDS))
    description: str = ''
    aliases: tuple[str, ...] = ()
    required: bool = False

    def __post_init__(self):
        require_string(self.name, 'predicate name')
        if self.cardinality not in {'single', 'multi'}:
            raise ValueError('cardinality must be single or multi')
        if not set(self.assertion_kinds) <= KINDS:
            raise ValueError('Unknown assertion kind in predicate spec')
        for name, spec in self.conditions:
            require_string(name, 'condition field')
            if spec.type in {'record', 'set', 'string', 'bound'}:
                raise ValueError('Condition values must be boolean, number, quantity, enum or entity')

    def describe(self):
        out = {'name': self.name, 'value': self.value.describe(), 'cardinality': self.cardinality,
               'conditions': {k: v.describe() for k, v in self.conditions},
               'assertion_kinds': list(self.assertion_kinds), 'description': self.description}
        if self.aliases:
            out['aliases'] = list(self.aliases)
        if self.required:
            out['required'] = True
        return out


@dataclass(frozen=True)
class Vocabulary:
    """Versioned application contract for admitted propositions."""
    name: str
    version: str
    predicates: tuple[PredicateSpec, ...]
    units: UnitRegistry = field(default=STANDARD_UNITS, compare=False)
    entities: tuple[tuple[str, tuple[str, ...]], ...] = ()
    closed_entities: bool = False
    scope_keys: tuple[str, ...] = ('event_id',)
    require_explicit_condition_status: bool = False
    register_status_terms: tuple[str, ...] = ()

    def __post_init__(self):
        require_string(self.name, 'vocabulary name')
        require_string(self.version, 'vocabulary version')
        names = [p.name for p in self.predicates]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate predicate')
        labels = {}
        for canonical_name, aliases in self.entities:
            for label in (canonical_name, *aliases):
                key = label_key(label)
                if labels.get(key, canonical_name) != canonical_name:
                    raise ValueError(f'Entity alias {label!r} maps to two entities')
                labels[key] = canonical_name
        object.__setattr__(self, '_entity_labels', labels)
        object.__setattr__(self, '_predicates', {p.name: p for p in self.predicates})
        # Validate declared dimensions exist with a base unit.
        for predicate in self.predicates:
            for spec in self._specs(predicate.value):
                for dimension in spec.dimensions:
                    self.units.base(dimension)

    def _specs(self, spec):
        yield spec
        for _, child in spec.fields:
            yield from self._specs(child)
        if spec.element is not None:
            yield from self._specs(spec.element)

    @property
    def fingerprint(self) -> str:
        return digest(self.describe())

    def describe(self):
        return {'name': self.name, 'version': self.version,
                'predicates': [p.describe() for p in self.predicates],
                'units': self.units.describe(),
                'entities': {k: list(v) for k, v in self.entities}, 'closed_entities': self.closed_entities,
                'scope_keys': list(self.scope_keys), 'assertion_kinds': sorted(KINDS),
                'register_status_terms': list(self.register_status_terms)}

    def predicate(self, name: str) -> PredicateSpec:
        if not isinstance(name, str) or name not in self._predicates:
            raise UnmappedProposal('unknown_predicate', 'predicate', name)
        return self._predicates[name]

    def entity(self, label: Any) -> str:
        if not isinstance(label, str) or not label.strip():
            raise UnmappedProposal('entity_required', 'entity', label)
        canonical_name = self._entity_labels.get(label_key(label))
        if canonical_name is not None:
            return canonical_name
        if self.closed_entities:
            raise UnmappedProposal('unknown_entity', 'entity', label)
        return ' '.join(label.split())

    # ----------------------------------------------------------------- values
    def value(self, spec: FieldSpec, value: Any, unit: str | None, path: str):
        """Return (canonical_value, canonical_unit_or_None)."""
        t = spec.type
        if t == 'quantity':
            if type(value) is dict:
                if set(value) != {'value', 'unit'}:
                    raise UnmappedProposal('quantity_shape', path, sorted(value))
                value, unit = value['value'], value['unit']
            amount, base = self.units.to_base(value, unit, path + '/unit')
            if self.units.dimension_of(base) not in spec.dimensions:
                raise UnmappedProposal('unit_dimension_mismatch', path,
                                       {'unit': unit, 'allowed': list(spec.dimensions)})
            return {'value': amount, 'unit': base}, base
        if t == 'bound':
            if type(value) is not dict or not {'operator', 'value'} <= set(value) <= {'operator', 'value', 'unit'}:
                raise UnmappedProposal('bound_shape', path, value)
            if value['operator'] not in BOUND_OPERATORS:
                raise UnmappedProposal('bound_operator', path, value['operator'])
            inner_unit = value.get('unit', unit)
            amount, base = self.units.to_base(value['value'], inner_unit, path + '/unit')
            if self.units.dimension_of(base) not in spec.dimensions:
                raise UnmappedProposal('unit_dimension_mismatch', path,
                                       {'unit': inner_unit, 'allowed': list(spec.dimensions)})
            return {'operator': value['operator'], 'value': amount, 'unit': base}, base
        if unit is not None and path == 'value':
            raise UnmappedProposal('unit_not_applicable', 'unit', unit)
        if t == 'number':
            return decimal_text(value, path), None
        if t == 'boolean':
            if type(value) is not bool:
                raise UnmappedProposal('boolean_required', path, value)
            return value, None
        if t == 'enum':
            return self._member(spec, value, path), None
        if t == 'set':
            if type(value) not in (list, tuple) or not value:
                raise UnmappedProposal('set_required', path, value)
            members = sorted({self._member(spec, v, path) for v in value})
            return members, None
        if t == 'list':
            min_req = max(1, spec.min_items)
            if type(value) not in (list, tuple) or len(value) < min_req:
                raise UnmappedProposal('list_required', path, value)
            items = []
            for idx, elem in enumerate(value):
                norm_elem, _ = self.value(spec.element, elem, None, f'{path}[{idx}]')
                items.append(norm_elem)
            return items, None
        if t == 'string':
            if not isinstance(value, str) or not value.strip():
                raise UnmappedProposal('string_required', path, value)
            return ' '.join(value.split()), None
        if t == 'entity':
            return self.entity(value), None
        if t == 'date':
            try:
                return normalize_date(value), None
            except ValueError:
                raise UnmappedProposal('iso_date_required', path, value) from None
        # record
        if type(value) is not dict:
            raise UnmappedProposal('record_required', path, value)
        declared = dict(spec.fields)
        extra = set(value) - set(declared)
        if extra:
            raise UnmappedProposal('undeclared_field', path, sorted(extra))
        required = set(spec.required) if spec.required is not None else set(declared)
        missing = required - set(value)
        if missing:
            raise UnmappedProposal('missing_field', path, sorted(missing))
        out = {}
        for key in sorted(value):
            out[key], _ = self.value(declared[key], value[key], None, f'{path}/{key}')
        return out, None

    def _member(self, spec, value, path):
        if not isinstance(value, str):
            raise UnmappedProposal('enum_member_required', path, value)
        for member in spec.enum:
            if label_key(member) == label_key(value):
                return member
        raise UnmappedProposal('unknown_enum_member', path, value)

    def conditions(self, predicate: PredicateSpec, conditions: Any):
        if conditions in (None, ()):
            return ()
        if type(conditions) not in (list, tuple):
            raise UnmappedProposal('conditions_list_required', 'conditions', conditions)
        declared = dict(predicate.conditions)
        out = []
        for i, condition in enumerate(conditions):
            path = f'conditions[{i}]'
            if type(condition) is not dict or set(condition) != {'field', 'operator', 'value'}:
                raise UnmappedProposal('condition_shape', path, condition)
            name = condition['field']
            if name not in declared:
                raise UnmappedProposal('undeclared_condition', path, name)
            if condition['operator'] not in CONDITION_OPERATORS:
                raise UnmappedProposal('condition_operator', path, condition['operator'])
            spec = declared[name]
            if spec.type in {'quantity', 'bound'}:
                raw = condition['value']
                if type(raw) is not dict or set(raw) != {'value', 'unit'}:
                    raise UnmappedProposal('quantity_shape', path, raw)
                amount, base = self.units.to_base(raw['value'], raw['unit'], path + '/unit')
                if self.units.dimension_of(base) not in spec.dimensions:
                    raise UnmappedProposal('unit_dimension_mismatch', path, raw)
                canonical_value = {'value': amount, 'unit': base}
            else:
                if condition['operator'] in {'lt', 'le', 'gt', 'ge'} and spec.type not in {'number', 'date'}:
                    raise UnmappedProposal('ordering_requires_number_or_date', path, name)
                canonical_value, _ = self.value(spec, condition['value'], None, path + '/value')
            out.append({'field': name, 'operator': condition['operator'], 'value': canonical_value})
        out.sort(key=canonical)
        if len({canonical(c) for c in out}) != len(out):
            raise UnmappedProposal('duplicate_condition', 'conditions')
        return tuple(out)

    def scope(self, scope: Any):
        if scope in (None, {}):
            return {}
        if type(scope) is not dict:
            raise UnmappedProposal('scope_object_required', 'scope', scope)
        extra = set(scope) - set(self.scope_keys)
        if extra:
            raise UnmappedProposal('undeclared_scope_key', 'scope', sorted(extra))
        for key, value in scope.items():
            if not isinstance(value, str) or not value.strip():
                raise UnmappedProposal('scope_value_must_be_identifier', f'scope/{key}', value)
        return {k: scope[k].strip() for k in sorted(scope)}

    def proposition(self, proposal: Mapping[str, Any]):
        """Normalize a proposal to the canonical proposition, or raise UnmappedProposal."""
        allowed = {'entity', 'predicate', 'value', 'unit', 'polarity', 'assertion_kind', 'conditions',
                   'condition_status', 'scope', 'valid_from', 'valid_until'}
        extra = set(proposal) - allowed
        if extra:
            raise UnmappedProposal('undeclared_proposal_field', '', sorted(extra))
        predicate = self.predicate(proposal.get('predicate'))
        kind = proposal.get('assertion_kind', 'unknown')
        if kind not in KINDS:
            raise UnmappedProposal('unknown_assertion_kind', 'assertion_kind', kind)
        if kind not in predicate.assertion_kinds:
            raise UnmappedProposal('assertion_kind_not_declared_for_predicate', 'assertion_kind', kind)
        polarity = proposal.get('polarity', 'affirmed')
        if polarity not in {'affirmed', 'negated'}:
            raise UnmappedProposal('unknown_polarity', 'polarity', polarity)
        if 'value' not in proposal:
            raise UnmappedProposal('value_required', 'value')
        value, unit = self.value(predicate.value, proposal['value'], proposal.get('unit'), 'value')
        norm_conditions = list(self.conditions(predicate, proposal.get('conditions')))
        cond_status = proposal.get('condition_status')
        if self.require_explicit_condition_status or cond_status is not None:
            if cond_status not in CONDITION_STATUSES:
                raise UnmappedProposal('condition_status_required' if cond_status is None else 'unknown_condition_status',
                                       'condition_status', cond_status)
            if cond_status == 'conditional' and not norm_conditions:
                raise UnmappedProposal('conditional_requires_conditions', 'conditions', cond_status)
            if cond_status == 'unconditional' and norm_conditions:
                raise UnmappedProposal('unconditional_forbids_conditions', 'conditions', norm_conditions)
        valid_from, valid_until = proposal.get('valid_from'), proposal.get('valid_until')
        try:
            start, end = instant(valid_from), instant(valid_until)
        except (ValueError, TypeError, AttributeError):
            raise UnmappedProposal('invalid_validity_instant', 'valid_from/valid_until') from None
        if start and end and start > end:
            raise UnmappedProposal('inverted_validity_interval', 'valid_from/valid_until')
        out = {'entity': self.entity(proposal.get('entity')), 'predicate': predicate.name,
               'value': value, 'unit': unit, 'polarity': polarity, 'assertion_kind': kind,
               'conditions': norm_conditions,
               'scope': self.scope(proposal.get('scope')),
               'valid_from': valid_from, 'valid_until': valid_until}
        if self.require_explicit_condition_status or cond_status is not None:
            out['condition_status'] = cond_status
        return out

    def normalize_runtime_value(self, spec: FieldSpec, raw: Any, path: str = 'context'):
        """Normalize a runtime value against a declared FieldSpec (requiring explicit units for quantities)."""
        canon, _ = self.value(spec, raw, None, path)
        return canon

    def evaluate_condition(
        self,
        predicate: PredicateSpec,
        condition: Mapping[str, Any],
        context: Mapping[str, Any] | None,
    ) -> str:
        """Evaluate a canonical condition against runtime context: 'true' | 'false' | 'pending' | 'invalid'."""
        name = condition['field']
        if context is None or name not in context or context[name] is None:
            return 'pending'
        declared = dict(predicate.conditions)
        if name not in declared:
            return 'invalid'
        spec = declared[name]
        op = condition['operator']
        expected = condition['value']
        try:
            actual = self.normalize_runtime_value(spec, context[name], f'context/{name}')
        except UnmappedProposal:
            return 'invalid'

        if spec.type == 'quantity':
            if actual['unit'] != expected['unit']:
                return 'invalid'
            lhs, rhs = Decimal(actual['value']), Decimal(expected['value'])
            return 'true' if {'eq': lhs == rhs, 'ne': lhs != rhs, 'lt': lhs < rhs,
                              'le': lhs <= rhs, 'gt': lhs > rhs, 'ge': lhs >= rhs}[op] else 'false'

        if spec.type == 'number':
            lhs, rhs = Decimal(actual), Decimal(expected)
            return 'true' if {'eq': lhs == rhs, 'ne': lhs != rhs, 'lt': lhs < rhs,
                              'le': lhs <= rhs, 'gt': lhs > rhs, 'ge': lhs >= rhs}[op] else 'false'

        if spec.type == 'date':
            if op in {'eq', 'ne'}:
                rel = value_relation(spec, actual, expected)
                if rel == 'equal':
                    return 'true' if op == 'eq' else 'false'
                if rel == 'incompatible':
                    return 'false' if op == 'eq' else 'true'
                return 'pending'
            if date_precision(actual) == date_precision(expected):
                res = {'lt': actual < expected, 'le': actual <= expected,
                       'gt': actual > expected, 'ge': actual >= expected}[op]
                return 'true' if res else 'false'
            a0, a1 = _date_span(actual)
            b0, b1 = _date_span(expected)
            decided = {
                'lt': True if a1 < b0 else (False if a0 >= b1 else None),
                'le': True if a1 <= b0 else (False if a0 > b1 else None),
                'gt': True if a0 > b1 else (False if a1 <= b0 else None),
                'ge': True if a0 >= b1 else (False if a1 < b0 else None),
            }[op]
            if decided is None:
                return 'pending'
            return 'true' if decided else 'false'

        if op not in {'eq', 'ne'}:
            return 'invalid'
        eq = canonical(actual) == canonical(expected)
        return 'true' if (eq if op == 'eq' else not eq) else 'false'

    def normalize_bound_limits(
        self,
        predicate_name: str,
        bound_entry: Any,
        *,
        explicit_unit: str | None = None,
        path: str = 'numeric_bounds',
    ) -> tuple[Decimal | None, Decimal | None, str | None]:
        """Validate and unit-normalize numeric bounds against the declared predicate FieldSpec."""
        pred_spec = self.predicate(predicate_name)
        bound_path = f'{path}/{predicate_name}'
        if isinstance(bound_entry, Mapping):
            raw_lo = bound_entry.get('min')
            raw_hi = bound_entry.get('max')
            entry_unit = bound_entry.get('unit')
        elif isinstance(bound_entry, (tuple, list)) and len(bound_entry) == 3:
            raw_lo, raw_hi, entry_unit = bound_entry
        elif isinstance(bound_entry, (tuple, list)) and len(bound_entry) == 2:
            raw_lo, raw_hi = bound_entry
            entry_unit = None
        else:
            raise ValueError(f"Invalid numeric_bounds entry for '{predicate_name}': {bound_entry!r}")

        if explicit_unit is not None and entry_unit is not None and label_key(explicit_unit) != label_key(entry_unit):
            raise UnmappedProposal(
                'unit_dimension_mismatch',
                f'{bound_path}/unit',
                {'key_unit': explicit_unit, 'entry_unit': entry_unit},
            )
        effective_unit = entry_unit if entry_unit is not None else explicit_unit

        lo = Decimal(decimal_text(raw_lo, f'{bound_path}/min')) if raw_lo is not None else None
        hi = Decimal(decimal_text(raw_hi, f'{bound_path}/max')) if raw_hi is not None else None

        val_spec = pred_spec.value
        if val_spec.type in {'quantity', 'bound'}:
            unit_obj = self.units.lookup(effective_unit, f'{bound_path}/unit')
            if unit_obj.dimension not in val_spec.dimensions:
                raise UnmappedProposal(
                    'unit_dimension_mismatch',
                    f'{bound_path}/unit',
                    {'unit': effective_unit, 'allowed': list(val_spec.dimensions)},
                )
            base_unit = self.units.base(unit_obj.dimension)
            if lo is not None:
                amt_lo, base_unit = self.units.to_base(lo, effective_unit, f'{bound_path}/min')
                lo = Decimal(amt_lo)
            if hi is not None:
                amt_hi, base_unit = self.units.to_base(hi, effective_unit, f'{bound_path}/max')
                hi = Decimal(amt_hi)
            target_unit: str | None = base_unit
        elif val_spec.type == 'number':
            if effective_unit is not None:
                raise UnmappedProposal('unit_not_applicable', f'{bound_path}/unit', effective_unit)
            target_unit = None
        else:
            raise UnmappedProposal('numeric_bounds_unsupported_type', bound_path, val_spec.type)

        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"numeric_bounds[{predicate_name}] lower bound {lo} exceeds upper bound {hi}")
        return lo, hi, target_unit
