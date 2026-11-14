"""v4 records: evidence units bound to exact source spans and an acceptance policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from .vocabulary import label_key
from ..semantic.models import EvidenceRef, KINDS, canonical, digest

ORIGINS = frozenset({'reader', 'application', 'rule_derived'})
PROPOSITION_FIELDS = ('entity', 'predicate', 'value', 'unit', 'polarity', 'assertion_kind',
                      'conditions', 'scope', 'valid_from', 'valid_until')
OPTIONAL_PROPOSITION_FIELDS = frozenset({'condition_status'})


def context_key(p, *, include_event=True) -> str:
    """Comparable context: subject, predicate, assertion kind, scope (event), condition set and condition_status."""
    scope = p['scope'] if include_event else {k: v for k, v in p['scope'].items() if k != 'event_id'}
    # Entity identity ignores case, spacing, '_' and '-' (label_key): 'ACME CORP' and 'Acme Corp' are one subject.
    payload = {'entity': label_key(p['entity']), 'predicate': p['predicate'], 'assertion_kind': p['assertion_kind'],
               'scope': scope, 'conditions': p['conditions']}
    if 'condition_status' in p:
        payload['condition_status'] = p['condition_status']
    return canonical(payload)


@dataclass(frozen=True)
class EvidenceUnit:
    """u_i = (id, claim, source locator, context, quality, processing record).

    The identifier denotes an extraction instance: the same proposition read by a
    different producer, from a different span or on a different basis is a new
    instance, while replaying an identical instance is idempotent.
    """
    proposition: dict
    evidence: tuple[EvidenceRef, ...]
    basis: str
    origin: str
    producer: dict
    structural_reliability: float | None = None
    premises: tuple[str, ...] = ()
    rule: str | None = None
    rule_version: str | None = None
    rule_applications: int = 0
    proposal: dict = field(default_factory=dict)

    def __post_init__(self):
        prop_keys = set(self.proposition)
        if not set(PROPOSITION_FIELDS) <= prop_keys or not prop_keys <= (set(PROPOSITION_FIELDS) | OPTIONAL_PROPOSITION_FIELDS):
            raise ValueError('Evidence units need a complete canonical proposition')
        if self.proposition['assertion_kind'] not in KINDS:
            raise ValueError('Unknown assertion kind')
        if self.origin not in ORIGINS:
            raise ValueError('Unknown evidence origin')
        if self.origin == 'rule_derived':
            if not self.premises or not self.rule or self.evidence:
                raise ValueError('Derived units carry premises and a rule, not source spans')
        elif not self.evidence or self.premises or self.rule:
            raise ValueError('Source units need exact evidence and no proof fields')
        if self.structural_reliability is not None and (
                type(self.structural_reliability) not in (int, float)
                or not math.isfinite(self.structural_reliability) or not 0 <= self.structural_reliability <= 1):
            raise ValueError('structural_reliability must be in [0,1] or None (unassessed)')
        canonical(self.record())

    @property
    def unit_id(self) -> str:
        identity = {k: v for k, v in self.record().items() if k != 'proposal'}
        return 'eu-' + digest(identity)

    @property
    def context_key(self) -> str:
        return context_key(self.proposition)

    @property
    def aggregate_key(self) -> str:
        return context_key(self.proposition, include_event=False)

    @property
    def assertion_key(self) -> str:
        p = self.proposition
        return canonical({k: p[k] for k in ('value', 'unit', 'polarity')})

    def record(self) -> dict:
        record = asdict(self)
        record['evidence'] = [asdict(r) for r in self.evidence]
        return record

    @classmethod
    def from_record(cls, record):
        record = dict(record)
        record['evidence'] = tuple(EvidenceRef(**r) for r in record['evidence'])
        record['premises'] = tuple(record['premises'])
        return cls(**record)


@dataclass(frozen=True)
class AcceptancePolicy:
    """Paper reference policy: accept g iff Q_g >= tau and S_g - O_g >= delta.

    Defaults are placeholders, not validated thresholds: select them on
    development data and fix them before any test evaluation.
    """
    tau: float = 0.5
    delta: float = 0.1
    allowed_assertion_kinds: tuple[str, ...] = ('fact', 'definition', 'requirement')
    allowed_source_attributions: tuple[str, ...] = ('source_statement', 'court_finding', 'audited_record',
                                                    'controlling_instrument', 'application_record')
    path_discount: float = 0.95
    thresholds_validated: bool = False
    version: str = 'v4-acceptance-1'

    def __post_init__(self):
        for name, lo_ok in (('tau', True), ('delta', False), ('path_discount', False)):
            val = getattr(self, name)
            if type(val) not in (int, float) or not math.isfinite(val):
                raise ValueError(f'{name} must be a finite number')
        if not 0 <= self.tau <= 1 or not 0 < self.delta <= 1:
            raise ValueError('tau in [0,1] and delta in (0,1] are required')
        if not 0 < self.path_discount <= 1:
            raise ValueError('path_discount must be in (0,1]')
        if not set(self.allowed_assertion_kinds) <= KINDS:
            raise ValueError('Unknown assertion kind')

    @property
    def fingerprint(self):
        return digest(asdict(self))
