"""Provider-neutral semantic contracts. No language-specific parsing."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Protocol

CONTRACT_VERSION = 'claimlayer-semantic-v1'
KINDS = frozenset({'fact', 'definition', 'requirement', 'allegation', 'proposal', 'hypothesis', 'unknown'})
VERDICTS = frozenset({'supported', 'contradicted', 'insufficient_evidence'})


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def instant(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def require_string(value: Any, name: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f'{name} must be a string' + ('' if empty else ' with content'))
    return value


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    text: str
    authority_rank: int = 0
    weight: float = 1.0
    dependence_group: str | None = None
    attribution: str = 'source_statement'

    def __post_init__(self):
        require_string(self.document_id, 'document_id')
        require_string(self.text, 'text')
        require_string(self.attribution, 'attribution')
        if type(self.authority_rank) is not int or not 0 <= self.authority_rank <= 100:
            raise ValueError('authority_rank must be an application-supplied integer in [0,100]')
        if type(self.weight) not in (int, float) or not math.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError('weight must be application-supplied in [0,1]')
        if self.dependence_group is not None:
            require_string(self.dependence_group, 'dependence_group')

    @property
    def revision(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class EvidenceRef:
    document_id: str
    revision: str
    start: int
    end: int
    quote: str

    def __post_init__(self):
        require_string(self.document_id, 'document_id')
        require_string(self.revision, 'revision')
        require_string(self.quote, 'quote')
        if type(self.start) is not int or type(self.end) is not int or not 0 <= self.start < self.end:
            raise ValueError('Evidence offsets must be increasing nonnegative integers')


@dataclass(frozen=True)
class CandidateClaim:
    claim_id: str
    entity: str
    predicate: str
    value: Any
    evidence: tuple[EvidenceRef, ...] = ()
    polarity: str = 'affirmed'
    assertion_kind: str = 'unknown'
    unit: str | None = None
    conditions: tuple[dict[str, Any], ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    valid_from: str | None = None
    valid_until: str | None = None
    premises: tuple[str, ...] = ()
    derivation_rule: str | None = None

    def __post_init__(self):
        for name in ('claim_id', 'entity', 'predicate'):
            require_string(getattr(self, name), name)
        if self.polarity not in {'affirmed', 'negated'} or self.assertion_kind not in KINDS:
            raise ValueError('Unknown polarity or assertion kind')
        if not all(isinstance(r, EvidenceRef) for r in self.evidence):
            raise ValueError('evidence must contain EvidenceRef records')
        if type(self.scope) is not dict or not all(type(c) is dict for c in self.conditions):
            raise ValueError('scope and conditions must be structured objects')
        for condition in self.conditions:
            if set(condition) != {'field', 'operator', 'value'} or condition['operator'] not in {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}:
                raise ValueError('Conditions need field/operator/value with an explicit supported operator')
            require_string(condition['field'], 'condition field')
        if self.unit is not None:
            require_string(self.unit, 'unit')
        if self.derivation_rule is not None:
            require_string(self.derivation_rule, 'derivation_rule')
        for premise in self.premises:
            require_string(premise, 'premise')
        start, end = instant(self.valid_from), instant(self.valid_until)
        if start and end and start > end:
            raise ValueError('Inverted validity interval')
        canonical(asdict(self))

    def proposition(self) -> dict[str, Any]:
        record = asdict(self)
        for key in ('claim_id', 'evidence', 'premises', 'derivation_rule'):
            record.pop(key)
        return record

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> 'CandidateClaim':
        record = dict(value)
        record['evidence'] = tuple(EvidenceRef(**r) for r in record.get('evidence', ()))
        record['conditions'] = tuple(record.get('conditions', ()))
        record['premises'] = tuple(record.get('premises', ()))
        return cls(**record)


@dataclass(frozen=True)
class StructuredQuery:
    entity: str
    predicate: str
    operation: str = 'resolve'
    as_of: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        require_string(self.entity, 'entity')
        require_string(self.predicate, 'predicate')
        if self.operation not in {'resolve', 'list', 'sum', 'count', 'min', 'max', 'average'}:
            raise ValueError('Unsupported query operation')
        instant(self.as_of)
        if type(self.context) is not dict:
            raise ValueError('Query context must be an object')
        canonical(asdict(self))


@dataclass(frozen=True)
class SemanticPolicy:
    mode: str = 'observe'
    allowed_assertion_kinds: tuple[str, ...] = ('fact', 'definition', 'requirement')
    predicate_cardinality: dict[str, str] = field(default_factory=dict)
    allowed_source_attributions: tuple[str, ...] = ('source_statement', 'court_finding', 'audited_record', 'controlling_instrument', 'application_record')
    max_context_chars: int = 24000
    max_context_rounds: int = 1
    max_claims_per_document: int = 100
    version: str = 'semantic-policy-v1'

    def __post_init__(self):
        if self.mode not in {'observe', 'strict'}:
            raise ValueError('mode must be observe or strict')
        if not all(isinstance(v, str) and v for v in self.allowed_source_attributions):
            raise ValueError('Source attributions must be explicit policy labels')
        if not set(self.allowed_assertion_kinds) <= KINDS:
            raise ValueError('Unknown allowed assertion kind')
        if not all(v in {'single', 'multi', 'unknown'} for v in self.predicate_cardinality.values()):
            raise ValueError('Invalid predicate cardinality')
        if type(self.max_context_chars) is not int or self.max_context_chars < 256:
            raise ValueError('Context budget must be >=256 characters')
        if type(self.max_context_rounds) is not int or not 0 <= self.max_context_rounds <= 3:
            raise ValueError('Context rounds must be in [0,3]')
        if type(self.max_claims_per_document) is not int or not 1 <= self.max_claims_per_document <= 1000:
            raise ValueError('Invalid claim count limit')

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


class SemanticReader(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...
    def extract(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def parse_query(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class EvidenceVerifier(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...
    def verify(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
