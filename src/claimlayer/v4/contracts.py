"""Immutable, content-addressed contracts for Gate's six responsibility layers."""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from typing import Any, ClassVar, Mapping, Sequence
from ..semantic.models import canonical, digest

CONTRACT_VERSION = 'gate-layers-1'


@dataclass(frozen=True)
class Artifact:
    # Keep JSON, not caller-owned mutable dictionaries, as the authoritative state.
    _payload: str = field(repr=False)
    _producer: str = field(repr=False)
    parents: tuple[tuple[str, str], ...] = ()
    schema_version: str = CONTRACT_VERSION
    KIND: ClassVar[str] = 'artifact'

    def __post_init__(self):
        if not isinstance(self._payload, str) or not isinstance(self._producer, str):
            raise TypeError('Artifact state must be immutable canonical JSON strings')
        if not isinstance(json.loads(self._payload), dict) or not isinstance(json.loads(self._producer), dict):
            raise ValueError('Artifact payload and producer must be objects')
        if not isinstance(self.parents, tuple) or any(not isinstance(p, tuple) or len(p)!=2 or not all(isinstance(v,str) for v in p) for p in self.parents):
            raise ValueError('Artifact parents must be immutable ID/digest tuples')

    @classmethod
    def create(cls, payload: Mapping[str, Any], *, producer: Mapping[str, Any],
               parents: Sequence['Artifact'] = ()) -> 'Artifact':
        return cls(canonical(dict(payload)), canonical(dict(producer)),
                   tuple((p.artifact_id, p.content_digest) for p in parents))

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload)

    @property
    def producer(self) -> dict[str, Any]:
        return json.loads(self._producer)

    @property
    def content_digest(self) -> str:
        return digest(dict(kind=self.KIND, schema_version=self.schema_version,
                           payload=self.payload, producer=self.producer, parents=self.parents))

    @property
    def artifact_id(self) -> str:
        return f'{self.KIND}_{self.content_digest}'

    def to_dict(self) -> dict[str, Any]:
        return dict(kind=self.KIND, schema_version=self.schema_version,
                    artifact_id=self.artifact_id, content_digest=self.content_digest,
                    producer=self.producer, parents=[dict(artifact_id=i, content_digest=d) for i,d in self.parents],
                    payload=self.payload)

    def depends_on(self, artifact: 'Artifact') -> bool:
        return (artifact.artifact_id, artifact.content_digest) in self.parents


@dataclass(frozen=True)
class ContextBundle(Artifact): KIND = 'context'
@dataclass(frozen=True)
class TaskSpec(Artifact): KIND = 'task'
@dataclass(frozen=True)
class PremiseSet(Artifact): KIND = 'premises'
@dataclass(frozen=True)
class DerivationPlan(Artifact): KIND = 'plan'
@dataclass(frozen=True)
class VerificationReceipt(Artifact): KIND = 'verification'
@dataclass(frozen=True)
class DecisionReceipt(Artifact): KIND = 'decision'


@dataclass(frozen=True)
class GateResult:
    context: ContextBundle
    task: TaskSpec | None
    premises: PremiseSet | None
    plan: DerivationPlan | None
    verification: VerificationReceipt
    decision: DecisionReceipt
    _trace: str = field(repr=False)
    contract_version: str = CONTRACT_VERSION

    @property
    def trace(self) -> dict[str, Any]:
        return json.loads(self._trace)

    def to_dict(self) -> dict[str, Any]:
        return dict(contract_version=self.contract_version,
                    **{name: getattr(self, name).to_dict() if getattr(self, name) is not None else None
                       for name in ('context','task','premises','plan','verification','decision')},
                    trace=self.trace)


class ContractError(ValueError):
    def __init__(self, code: str, message: str, stage: str, *, repair_owner: str | None = None):
        super().__init__(message)
        self.code, self.stage = code, stage
        self.repair_owner = repair_owner or stage

    def to_dict(self) -> dict[str, Any]:
        return dict(code=self.code, message=str(self), stage=self.stage,
                    detected_by=f'{self.stage}:contract', repair_owner=self.repair_owner,
                    cause='not_determined')


def problem(code: str, message: str, *, stage: str, detected_by: str,
            repair_owner: str, **details: Any) -> dict[str, Any]:
    return dict(code=code, message=message, stage=stage, detected_by=detected_by,
                repair_owner=repair_owner, cause='not_determined', **details)
