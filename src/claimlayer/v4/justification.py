"""Explicit Justification Graph & Epistemic Claim Classes (Phase 1).

Separates raw or derived representations into an explicit DAG of ClaimNodes
and InferenceSteps with mandatory epistemic classification and verification
obligations. An LLM may propose the graph structure, but never validates its
own rules, sets its own source authority, or overrides formal reliability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import EvidenceRef, canonical, digest
from .fidelity import FieldEvidence
from .vocabulary import UnmappedProposal, Vocabulary, label_key, normalize_date

CLAIM_KINDS: tuple[str, ...] = (
    "extraction",
    "derivation",
    "hypothesis",
    "prediction",
    "recommendation",
)

# Epistemic strength order: higher value = stronger epistemic claim class.
# Invariant: derived claims can NEVER be promoted above their weakest premise,
# and hypothesis / prediction / recommendation can NEVER become extraction / derivation.
EPISTEMIC_KIND_RANK: dict[str, int] = {
    "extraction": 5,
    "derivation": 4,
    "prediction": 3,
    "hypothesis": 2,
    "recommendation": 1,
}
_RANK_TO_CLAIM_KIND: dict[int, str] = {v: k for k, v in EPISTEMIC_KIND_RANK.items()}

OBLIGATION_GENERATOR_VERSION = "1.0.0"

MANDATORY_OBLIGATION_FAMILIES: dict[str, str] = {
    "age_and_duration": "temporal",
    "quantities": "quantities",
    "documental_attribution": "document_authority",
    "conditions_and_exceptions": "explicit_logic",
    "aggregations": "aggregation",
    "material_conflict": "document_authority",
}

VERIFIER_FAMILIES: tuple[str, ...] = (
    "arithmetic",
    "quantities",
    "temporal",
    "comparison",
    "aggregation",
    "explicit_logic",
    "relations",
    "document_authority",
    "open_semantics",
)

# Map common InferenceStep.operation names to default verifier families when the caller
# declares an operation without listing explicit VerificationObligation entries.
DEFAULT_OPERATION_FAMILIES: dict[str, tuple[str, ...]] = {
    "add": ("arithmetic", "quantities"),
    "sum": ("arithmetic", "quantities"),
    "sub": ("arithmetic", "quantities"),
    "diff": ("arithmetic", "quantities"),
    "mul": ("arithmetic", "quantities"),
    "prod": ("arithmetic", "quantities"),
    "div": ("arithmetic", "quantities"),
    "ratio": ("arithmetic",),
    "pct": ("arithmetic",),
    "percentage_of": ("arithmetic",),
    "round": ("arithmetic",),
    "arithmetic": ("arithmetic", "quantities"),
    "compound": ("arithmetic",),
    "linear_combination": ("arithmetic", "quantities"),
    "unit_conversion": ("quantities",),
    "quantities": ("quantities",),
    "temporal_age": ("temporal",),
    "age": ("temporal",),
    "temporal_duration": ("temporal",),
    "duration": ("temporal",),
    "temporal_order": ("temporal",),
    "temporal": ("temporal",),
    "comparison": ("comparison", "quantities"),
    "eq": ("comparison", "quantities"),
    "ne": ("comparison", "quantities"),
    "lt": ("comparison", "quantities"),
    "le": ("comparison", "quantities"),
    "gt": ("comparison", "quantities"),
    "ge": ("comparison", "quantities"),
    "between": ("comparison", "quantities"),
    "within_bounds": ("comparison", "quantities"),
    "exceeds_cap": ("comparison", "quantities"),
    "below_floor": ("comparison", "quantities"),
    "aggregation": ("aggregation", "quantities"),
    "agg_sum": ("aggregation", "quantities"),
    "agg_count": ("aggregation",),
    "agg_average": ("aggregation", "quantities"),
    "agg_mean": ("aggregation", "quantities"),
    "agg_min": ("aggregation", "quantities"),
    "agg_max": ("aggregation", "quantities"),
    "rule_application": ("explicit_logic",),
    "explicit_logic": ("explicit_logic",),
    "modus_ponens": ("explicit_logic",),
    "relation_composition": ("relations",),
    "relations": ("relations",),
    "authority_precedence": ("document_authority",),
    "document_authority": ("document_authority",),
    "semantic_entailment": ("open_semantics",),
    "open_semantics": ("open_semantics",),
}

# Metadata keys that must never be trusted if injected by a generative proposal.
_UNTRUSTED_MODEL_METADATA_KEYS = frozenset({
    "trust_score",
    "authority_rank",
    "authority_override",
    "bypass_verification",
    "self_certified",
    "verified",
    "calibrated_probability",
})


def effective_epistemic_kind(
    declared_kind: str,
    premise_kinds: Sequence[str] = (),
    *,
    unverified_attestor: bool = False,
) -> str:
    """Compute effective epistemic class enforcing strict monotonicity.

    Rules:
    - `declared_kind` must belong to `CLAIM_KINDS`.
    - An unverified party/user allegation cannot enter as `extraction` or `derivation`;
      its rank is bounded above by `hypothesis`.
    - Any claim with premises is capped at `derivation` (never raw `extraction`) and
      bounded above by the weakest effective class among its premises.
    - `hypothesis`, `prediction`, and `recommendation` can NEVER be promoted to
      `extraction` or `derivation` regardless of weights or source support.
    """
    if declared_kind not in CLAIM_KINDS:
        raise ValueError(f"Unknown epistemic claim kind {declared_kind!r}; expected one of {CLAIM_KINDS}.")
    for p_kind in premise_kinds:
        if p_kind not in CLAIM_KINDS:
            raise ValueError(f"Unknown premise claim kind {p_kind!r}.")

    rank = EPISTEMIC_KIND_RANK[declared_kind]
    if unverified_attestor:
        rank = min(rank, EPISTEMIC_KIND_RANK["hypothesis"])
    if premise_kinds:
        min_premise_rank = min(EPISTEMIC_KIND_RANK[p] for p in premise_kinds)
        rank = min(rank, EPISTEMIC_KIND_RANK["derivation"], min_premise_rank)
    return _RANK_TO_CLAIM_KIND[rank]


def _coerce_evidence_ref(item: EvidenceRef | Mapping[str, Any]) -> EvidenceRef:
    if isinstance(item, EvidenceRef):
        return item
    if not isinstance(item, Mapping):
        raise ValueError("Evidence item must be EvidenceRef or mapping.")
    return EvidenceRef(
        document_id=str(item["document_id"]),
        revision=str(item.get("revision") or "rev-0"),
        start=int(item["start"]),
        end=int(item["end"]),
        quote=str(item["quote"]),
    )


def _coerce_field_evidence(item: FieldEvidence | Mapping[str, Any]) -> FieldEvidence:
    if isinstance(item, FieldEvidence):
        return item
    if not isinstance(item, Mapping):
        raise ValueError("FieldEvidence item must be FieldEvidence or mapping.")
    ref = _coerce_evidence_ref(item["evidence"])
    return FieldEvidence(field=str(item["field"]), evidence=ref)


@dataclass(frozen=True)
class VerificationObligation:
    """A concrete checkable verification obligation attached to an InferenceStep."""

    obligation_id: str
    family: str
    description: str = ""
    critical: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    origin: str = "proposed"  # 'proposed' | 'engine_mandatory'

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise ValueError("obligation_id must be a non-empty string.")
        if self.family not in VERIFIER_FAMILIES:
            raise ValueError(
                f"Unknown verifier family {self.family!r}; expected one of {VERIFIER_FAMILIES}."
            )
        if self.origin not in ("proposed", "engine_mandatory"):
            raise ValueError(f"Unsupported obligation origin {self.origin!r}.")
        clean_params = dict(self.parameters or {})
        canonical(clean_params)
        object.__setattr__(self, "parameters", clean_params)
        object.__setattr__(self, "critical", bool(self.critical))

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "obligation_id": self.obligation_id,
            "family": self.family,
            "description": self.description,
            "critical": self.critical,
            "parameters": self.parameters,
        }
        if self.origin != "proposed":
            payload["origin"] = self.origin
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.canonical_payload(),
            "origin": self.origin,
        }

    @classmethod
    def from_dict(cls, data: VerificationObligation | Mapping[str, Any]) -> "VerificationObligation":
        if isinstance(data, cls):
            return data
        return cls(
            obligation_id=str(data["obligation_id"]),
            family=str(data["family"]),
            description=str(data.get("description", "")),
            critical=bool(data.get("critical", True)),
            parameters=dict(data.get("parameters") or {}),
            origin=str(data.get("origin", "proposed")),
        )


@dataclass(frozen=True)
class ClaimNode:
    """Typed or provisional proposition node inside an explicit JustificationGraph."""

    claim_id: str
    kind: str = "extraction"
    subject: str = ""
    predicate: str = ""
    value: Any = None
    unit: str | None = None
    polarity: int = 1
    text: str = ""
    assertion_kind: str = "fact"
    evidence: tuple[EvidenceRef, ...] = ()
    field_evidence: tuple[FieldEvidence, ...] = ()
    evidence_unit_ids: tuple[str, ...] = ()
    scope: dict[str, Any] = field(default_factory=dict)
    valid_from: str | None = None
    valid_until: str | None = None
    conditions: tuple[dict[str, Any], ...] = ()
    condition_status: str = "unconditional"
    assumptions: tuple[str, ...] = ()
    attestor: str | None = None
    dependence_group: str | None = None
    structural_reliability: float | None = None
    event_id: str | None = None
    approximate: bool = False
    approx_tolerance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, str) or not self.claim_id.strip():
            raise ValueError("claim_id must be a non-empty string.")
        if self.kind not in CLAIM_KINDS:
            raise ValueError(f"Unknown claim kind {self.kind!r}; expected one of {CLAIM_KINDS}.")
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be 1 or -1.")
        if self.structural_reliability is not None:
            if (
                type(self.structural_reliability) not in (int, float)
                or not math.isfinite(self.structural_reliability)
                or not 0.0 <= float(self.structural_reliability) <= 1.0
            ):
                raise ValueError("structural_reliability must be in [0, 1] or None.")
        if self.approx_tolerance is not None:
            if (
                type(self.approx_tolerance) not in (int, float)
                or not math.isfinite(self.approx_tolerance)
                or float(self.approx_tolerance) < 0.0
            ):
                raise ValueError("approx_tolerance must be non-negative finite number or None.")

        ev_refs = tuple(_coerce_evidence_ref(e) for e in (self.evidence or ()))
        f_refs = tuple(_coerce_field_evidence(fe) for fe in (self.field_evidence or ()))
        u_ids = tuple(str(u) for u in (self.evidence_unit_ids or ()))
        clean_scope = dict(self.scope or {})
        clean_conds = tuple(dict(c) for c in (self.conditions or ()))
        clean_assumptions = tuple(str(a) for a in (self.assumptions or ()))
        # Strip any self-certification / privilege fields if proposed by an external caller.
        raw_meta = dict(self.metadata or {})
        clean_meta = {k: v for k, v in raw_meta.items() if k not in _UNTRUSTED_MODEL_METADATA_KEYS}

        canonical({
            "scope": clean_scope,
            "conditions": clean_conds,
            "value": self.value,
            "metadata": clean_meta,
        })

        object.__setattr__(self, "evidence", ev_refs)
        object.__setattr__(self, "field_evidence", f_refs)
        object.__setattr__(self, "evidence_unit_ids", u_ids)
        object.__setattr__(self, "scope", clean_scope)
        object.__setattr__(self, "conditions", clean_conds)
        object.__setattr__(self, "assumptions", clean_assumptions)
        object.__setattr__(self, "metadata", clean_meta)
        object.__setattr__(self, "approximate", bool(self.approximate))

    def doc_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        seen: set[str] = set()
        for ref in self.evidence:
            if ref.document_id not in seen:
                seen.add(ref.document_id)
                ids.append(ref.document_id)
        if isinstance(self.scope, Mapping) and "doc_id" in self.scope:
            d = str(self.scope["doc_id"])
            if d not in seen:
                seen.add(d)
                ids.append(d)
        return tuple(ids)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "unit": self.unit,
            "polarity": self.polarity,
            "text": self.text,
            "assertion_kind": self.assertion_kind,
            "evidence": [asdict(r) for r in self.evidence],
            "field_evidence": [{"field": fe.field, "evidence": asdict(fe.evidence)} for fe in self.field_evidence],
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "scope": self.scope,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "conditions": list(self.conditions),
            "condition_status": self.condition_status,
            "assumptions": list(self.assumptions),
            "attestor": self.attestor,
            "dependence_group": self.dependence_group,
            "structural_reliability": self.structural_reliability,
            "event_id": self.event_id,
            "approximate": self.approximate,
            "approx_tolerance": self.approx_tolerance,
            "metadata": self.metadata,
        }

    def content_digest(self) -> str:
        return digest(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_payload()

    @classmethod
    def from_dict(cls, data: ClaimNode | Mapping[str, Any]) -> "ClaimNode":
        if isinstance(data, cls):
            return data
        raw_pol = data.get("polarity", 1)
        if isinstance(raw_pol, str):
            polarity = -1 if raw_pol.lower() in ("negated", "-1", "negative") else 1
        else:
            polarity = int(raw_pol)
        return cls(
            claim_id=str(data["claim_id"]),
            kind=str(data.get("kind", "extraction")),
            subject=str(data.get("subject", data.get("entity", ""))),
            predicate=str(data.get("predicate", "")),
            value=data.get("value"),
            unit=data.get("unit"),
            polarity=polarity,
            text=str(data.get("text", "")),
            assertion_kind=str(data.get("assertion_kind", "fact")),
            evidence=tuple(data.get("evidence") or ()),
            field_evidence=tuple(data.get("field_evidence") or ()),
            evidence_unit_ids=tuple(data.get("evidence_unit_ids") or ()),
            scope=dict(data.get("scope") or {}),
            valid_from=data.get("valid_from"),
            valid_until=data.get("valid_until"),
            conditions=tuple(data.get("conditions") or ()),
            condition_status=str(data.get("condition_status", "unconditional")),
            assumptions=tuple(data.get("assumptions") or ()),
            attestor=data.get("attestor"),
            dependence_group=data.get("dependence_group"),
            structural_reliability=data.get("structural_reliability"),
            event_id=data.get("event_id"),
            approximate=bool(data.get("approximate", False)),
            approx_tolerance=data.get("approx_tolerance"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class InferenceStep:
    """Explicit directed inference step connecting premise ClaimNodes to a conclusion ClaimNode."""

    step_id: str
    premises: tuple[str, ...]
    conclusion_id: str
    operation: str
    rule_id: str | None = None
    rule_version: str | None = None
    assumptions: tuple[str, ...] = ()
    obligations: tuple[VerificationObligation, ...] = ()
    critical: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise ValueError("step_id must be a non-empty string.")
        if not isinstance(self.conclusion_id, str) or not self.conclusion_id.strip():
            raise ValueError("conclusion_id must be a non-empty string.")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError("operation must be a non-empty string.")

        prem_tuple = tuple(str(p) for p in self.premises)
        assump_tuple = tuple(str(a) for a in (self.assumptions or ()))
        clean_params = dict(self.parameters or {})
        canonical(clean_params)

        coerced_obs = tuple(VerificationObligation.from_dict(o) for o in (self.obligations or ()))
        if not coerced_obs:
            # Infer default obligations cleanly from operation when none explicitly provided.
            op_key = label_key(self.operation).replace(" ", "_")
            families = DEFAULT_OPERATION_FAMILIES.get(op_key) or DEFAULT_OPERATION_FAMILIES.get(
                self.operation, ("open_semantics",)
            )
            coerced_obs = tuple(
                VerificationObligation(
                    obligation_id=f"{self.step_id}::{fam}",
                    family=fam,
                    description=f"Verify {self.operation} under family '{fam}'",
                    critical=bool(self.critical),
                    parameters=dict(clean_params),
                )
                for fam in families
            )

        object.__setattr__(self, "premises", prem_tuple)
        object.__setattr__(self, "assumptions", assump_tuple)
        object.__setattr__(self, "obligations", coerced_obs)
        object.__setattr__(self, "critical", bool(self.critical))
        object.__setattr__(self, "parameters", clean_params)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "premises": list(self.premises),
            "conclusion_id": self.conclusion_id,
            "operation": self.operation,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "assumptions": list(self.assumptions),
            "obligations": [o.to_dict() for o in self.obligations],
            "critical": self.critical,
            "parameters": self.parameters,
        }

    def step_digest(self) -> str:
        return digest(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return self.canonical_payload()

    @classmethod
    def from_dict(cls, data: InferenceStep | Mapping[str, Any]) -> "InferenceStep":
        if isinstance(data, cls):
            return data
        return cls(
            step_id=str(data["step_id"]),
            premises=tuple(data.get("premises") or ()),
            conclusion_id=str(data.get("conclusion_id")),
            operation=str(data.get("operation", "open_semantics")),
            rule_id=data.get("rule_id"),
            rule_version=data.get("rule_version"),
            assumptions=tuple(data.get("assumptions") or ()),
            obligations=tuple(data.get("obligations") or ()),
            critical=bool(data.get("critical", True)),
            parameters=dict(data.get("parameters") or {}),
        )


@dataclass(frozen=True)
class JustificationGraph:
    """Explicit directed acyclic graph of claims and inference steps."""

    claims: dict[str, ClaimNode]
    steps: tuple[InferenceStep, ...]
    conclusion_id: str

    def __post_init__(self) -> None:
        coerced_claims = {
            str(cid): ClaimNode.from_dict(node) for cid, node in self.claims.items()
        }
        coerced_steps = tuple(InferenceStep.from_dict(s) for s in self.steps)
        object.__setattr__(self, "claims", coerced_claims)
        object.__setattr__(self, "steps", coerced_steps)

    def has_cycle(self) -> tuple[bool, list[str]]:
        """Detect cycles in premise -> conclusion dependencies."""
        adj: dict[str, list[str]] = {cid: [] for cid in self.claims}
        for step in self.steps:
            for pid in step.premises:
                adj.setdefault(pid, []).append(step.conclusion_id)

        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
        path: list[str] = []

        def _dfs(node: str) -> list[str] | None:
            state[node] = 1
            path.append(node)
            for nxt in sorted(adj.get(node, ())):
                st = state.get(nxt, 0)
                if st == 1:
                    idx = path.index(nxt)
                    return path[idx:] + [nxt]
                if st == 0:
                    cycle = _dfs(nxt)
                    if cycle is not None:
                        return cycle
            path.pop()
            state[node] = 2
            return None

        for cid in sorted(adj):
            if state.get(cid, 0) == 0:
                cycle = _dfs(cid)
                if cycle is not None:
                    return True, cycle
        return False, []

    def validate_structure(self) -> list[str]:
        """Return structural validation error messages (empty list means valid DAG)."""
        errors: list[str] = []
        if not self.claims:
            errors.append("JustificationGraph contains no claims.")
            return errors
        if self.conclusion_id not in self.claims:
            errors.append(f"Conclusion claim '{self.conclusion_id}' does not exist in graph.")
        for step in self.steps:
            if step.conclusion_id not in self.claims:
                errors.append(f"Step '{step.step_id}' targets missing conclusion '{step.conclusion_id}'.")
            for pid in step.premises:
                if pid not in self.claims:
                    errors.append(f"Step '{step.step_id}' references missing premise '{pid}'.")
        cyclic, cycle_nodes = self.has_cycle()
        if cyclic:
            errors.append(f"Cycle detected in justification graph: {' -> '.join(cycle_nodes)}.")
        return errors

    def incoming_steps(self, claim_id: str) -> tuple[InferenceStep, ...]:
        return tuple(s for s in self.steps if s.conclusion_id == claim_id)

    def outgoing_steps(self, claim_id: str) -> tuple[InferenceStep, ...]:
        return tuple(s for s in self.steps if claim_id in s.premises)

    def topological_order(self) -> list[str]:
        """Return claim IDs in topological order from leaves (extractions) to root (conclusions)."""
        cyclic, cycle_nodes = self.has_cycle()
        if cyclic:
            raise ValueError(f"Cannot topologically sort cyclic graph: {cycle_nodes}")

        indegree: dict[str, int] = {cid: 0 for cid in self.claims}
        dependents: dict[str, set[str]] = {cid: set() for cid in self.claims}
        for step in self.steps:
            if step.conclusion_id not in indegree:
                continue
            for pid in step.premises:
                if pid in indegree and step.conclusion_id not in dependents[pid]:
                    dependents[pid].add(step.conclusion_id)
                    indegree[step.conclusion_id] += 1

        queue = sorted(cid for cid, deg in indegree.items() if deg == 0)
        order: list[str] = []
        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for dep in sorted(dependents.get(curr, ())):
                indegree[dep] -= 1
                if indegree[dep] == 0:
                    queue.append(dep)
            queue.sort()
        return order

    def dependent_claims(self, source_key: str) -> set[str]:
        """Return transitive closure of claim IDs affected when `source_key` changes.

        `source_key` may be a `claim_id`, `document_id`, or `evidence_unit_id`.
        """
        seed: set[str] = set()
        if source_key in self.claims:
            seed.add(source_key)
        for cid, node in self.claims.items():
            if source_key in node.doc_ids() or source_key in node.evidence_unit_ids:
                seed.add(cid)

        affected = set(seed)
        frontier = list(sorted(seed))
        while frontier:
            curr = frontier.pop(0)
            for step in self.outgoing_steps(curr):
                if step.conclusion_id not in affected:
                    affected.add(step.conclusion_id)
                    frontier.append(step.conclusion_id)
        return affected

    def ancestor_claims(self, claim_id: str) -> set[str]:
        """Return all ancestor claim IDs feeding directly or transitively into `claim_id`."""
        ancestors: set[str] = set()
        frontier = [claim_id]
        visited = {claim_id}
        while frontier:
            curr = frontier.pop(0)
            for step in self.incoming_steps(curr):
                for pid in step.premises:
                    ancestors.add(pid)
                    if pid not in visited:
                        visited.add(pid)
                        frontier.append(pid)
        return ancestors

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "claims": {cid: self.claims[cid].to_dict() for cid in sorted(self.claims)},
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_response(
        cls,
        response: JustificationGraph | Mapping[str, Any],
        *,
        vocabulary: Vocabulary | None = None,
    ) -> "JustificationGraph":
        """Coerce a structured response payload or JustificationGraph into a validated graph."""
        if isinstance(response, cls):
            return response
        if not isinstance(response, Mapping):
            raise ValueError("response must be a JustificationGraph or structured mapping.")

        raw_claims = response.get("claims")
        claims_map: dict[str, ClaimNode] = {}
        if isinstance(raw_claims, Mapping):
            for cid, cdata in raw_claims.items():
                payload = dict(cdata) if isinstance(cdata, Mapping) else {}
                payload.setdefault("claim_id", str(cid))
                claims_map[str(cid)] = ClaimNode.from_dict(payload)
        elif isinstance(raw_claims, Sequence) and not isinstance(raw_claims, (str, bytes)):
            for cdata in raw_claims:
                node = ClaimNode.from_dict(cdata)
                claims_map[node.claim_id] = node
        elif raw_claims is None and "conclusion" in response and isinstance(response["conclusion"], Mapping):
            cnode = ClaimNode.from_dict(
                {**dict(response["conclusion"]), "claim_id": str(response["conclusion"].get("claim_id", "conclusion"))}
            )
            claims_map[cnode.claim_id] = cnode
        else:
            raise ValueError("Structured response must declare 'claims' mapping/sequence.")

        conclusion_id = response.get("conclusion_id")
        if conclusion_id is None and isinstance(response.get("conclusion"), Mapping):
            conc_dict = dict(response["conclusion"])
            conclusion_id = str(conc_dict.get("claim_id") or "conclusion")
            if conclusion_id not in claims_map:
                conc_dict.setdefault("claim_id", conclusion_id)
                claims_map[conclusion_id] = ClaimNode.from_dict(conc_dict)
        elif conclusion_id is None and len(claims_map) == 1:
            conclusion_id = next(iter(claims_map))

        if not conclusion_id or str(conclusion_id) not in claims_map:
            raise ValueError("Response must specify a valid 'conclusion_id' present in 'claims'.")

        if vocabulary is not None:
            # Canonicalize known entities gently where present in declared vocabulary,
            # preserving provisional/unmapped values cleanly when evaluating exploratory claims.
            normalized_claims: dict[str, ClaimNode] = {}
            for cid, node in claims_map.items():
                subj = node.subject
                if subj:
                    try:
                        subj = vocabulary.entity(subj)
                    except UnmappedProposal:
                        pass
                if subj != node.subject:
                    normalized_claims[cid] = ClaimNode(**{**node.to_dict(), "subject": subj})
                else:
                    normalized_claims[cid] = node
            claims_map = normalized_claims

        raw_steps = response.get("steps")
        if raw_steps is None:
            raw_steps = response.get("inference_steps", ())
        steps = tuple(InferenceStep.from_dict(s) for s in (raw_steps or ()))
        return cls(claims=claims_map, steps=steps, conclusion_id=str(conclusion_id))


_GENERIC_LABELS = frozenset({
    "number",
    "value",
    "result",
    "scalar",
    "generic",
    "raw_number",
    "output",
    "answer",
})
_TEMPORAL_PREDICATES = frozenset({
    "age",
    "edad",
    "birth_date",
    "fecha_nacimiento",
    "duration",
    "duracion",
    "elapsed_time",
    "tenure",
    "service_years",
    "age_at_anchor",
    "age_in_year",
    "anchor_age",
    "temporal_age",
    "temporal_duration",
})
_AGGREGATION_PREDICATES = frozenset({
    "total",
    "sum",
    "aggregate",
    "total_amount",
    "total_fee",
    "total_cost",
    "count",
    "average",
    "mean",
    "subtotal",
})
_TEMPORAL_OPS = frozenset({
    "temporal_age",
    "age",
    "temporal_duration",
    "duration",
    "temporal_order",
    "temporal",
    "precedes",
    "meets",
    "overlaps",
})
_AGGREGATION_OPS = frozenset({
    "aggregation",
    "agg_sum",
    "agg_count",
    "agg_average",
    "agg_mean",
    "agg_min",
    "agg_max",
    "count",
    "average",
    "mean",
    "min",
    "max",
})
_AGE_DURATION_TEXT_RE = re.compile(
    r"\b(how old|age|edad|cu[aá]ntos a[nñ]os|a[nñ]os tiene|tiene.*\d+\s*a[nñ]os|"
    r"born|naci[oó]|birthday|cumplea[nñ]os|duration|duraci[oó]n|cu[aá]nto tiempo)\b",
    re.IGNORECASE,
)
_AGGREGATION_TEXT_RE = re.compile(
    r"\b(total|subtotal|sum|suma|sumatoria|average|promedio|media|m[ií]nimo|m[aá]ximo|"
    r"cu[aá]nt(?:os|as)\s+(?:elementos|registros|facturas|[ií]tems|art[ií]culos|eventos|pagos|partidas)|"
    r"how many\s+(?:items|invoices|records|entries|events|payments)|"
    r"all invoices|todas las facturas|todas las partidas)\b",
    re.IGNORECASE,
)
_ISO_DATE_IN_TEXT_RE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")
_YEAR_IN_TEXT_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_DURATION_COMPUTABLE_UNITS = frozenset({"year", "years", "año", "años", "month", "months", "mes", "meses"})


def _is_date_string(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    txt = value.strip()
    if not re.match(r"^\d{4}", txt):
        return False
    try:
        normalize_date(txt)
        return True
    except ValueError:
        return False


def _extract_date_from_text(text: str | None) -> str | None:
    if not text:
        return None
    m_iso = _ISO_DATE_IN_TEXT_RE.findall(text)
    if m_iso:
        return m_iso[-1]
    m_yr = _YEAR_IN_TEXT_RE.findall(text)
    if m_yr:
        return m_yr[-1]
    return None


def infer_conclusion_semantic_domain(
    graph: JustificationGraph,
    *,
    question: str | Mapping[str, Any] | None = None,
    candidate_text: str | None = None,
    domain_profile: Any = None,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Deterministically infer the conclusion semantic domain(s) and detect generic-label evasion."""
    conc = graph.claims[graph.conclusion_id]
    conc_pred_key = label_key(conc.predicate).replace(" ", "_") if conc.predicate else ""
    q_text = (
        question
        if isinstance(question, str)
        else (str(question.get("text") or question.get("question") or "") if isinstance(question, Mapping) else "")
    )
    c_text = candidate_text or conc.text or ""
    anc_ids = graph.ancestor_claims(graph.conclusion_id)
    anc_nodes = [graph.claims[aid] for aid in sorted(anc_ids) if aid in graph.claims]
    inc_steps = graph.incoming_steps(graph.conclusion_id)

    generic_label = conc_pred_key in _GENERIC_LABELS or label_key(conc.metadata.get("field_type", "")) in {"number", "scalar"}

    # Temporal / age & duration discovery (never relies on specific entity names)
    has_temporal_op = any(
        label_key(st.operation).replace(" ", "_") in _TEMPORAL_OPS for st in inc_steps
    )
    has_temporal_pred = conc_pred_key in _TEMPORAL_PREDICATES or any(
        label_key(n.predicate).replace(" ", "_") in _TEMPORAL_PREDICATES for n in anc_nodes
    )
    has_date_premises = any(
        _is_date_string(n.value) or bool(n.valid_from) or bool(n.valid_until)
        for n in anc_nodes
    )
    has_duration_unit = bool(
        (conc.unit and label_key(conc.unit) in _DURATION_COMPUTABLE_UNITS)
        or any(n.unit and label_key(n.unit) in _DURATION_COMPUTABLE_UNITS for n in anc_nodes)
    )
    has_temporal_question = bool(_AGE_DURATION_TEXT_RE.search(q_text) or _AGE_DURATION_TEXT_RE.search(c_text))
    has_temporal_params = any(
        any(k in st.parameters for k in ("start_date", "birth_date", "reference_date", "end_date", "anchor_age", "anchor_date"))
        for st in inc_steps
    )

    is_age_or_duration = bool(
        has_temporal_op
        or has_temporal_pred
        or has_temporal_params
        or (has_date_premises and (has_duration_unit or has_temporal_question))
        or (has_temporal_question and (isinstance(conc.value, (int, float)) or isinstance(conc.value, Mapping) or has_date_premises))
    )

    # Quantity discovery
    has_quantity_units = bool(conc.unit or any(n.unit for n in anc_nodes))
    vocab = getattr(domain_profile, "vocabulary", None)
    conc_pred_spec = None
    if vocab is not None and conc.predicate:
        conc_pred_spec = next((p for p in vocab.predicates if p.name == conc.predicate), None)
    has_quantity_spec = bool(conc_pred_spec is not None and conc_pred_spec.value.type == "quantity")
    has_rounding_params = any(
        any(k in st.parameters for k in ("decimals", "rounding_tolerance", "tolerance"))
        for st in graph.steps
    )
    is_quantity = bool(has_quantity_units or has_quantity_spec or has_rounding_params)

    # Aggregation discovery (inspect conclusion's incoming steps & context, never unrelated upstream ops)
    is_comparison_conc = any(
        label_key(st.operation).replace(" ", "_") in {"lt", "le", "gt", "ge", "eq", "ne", "between"}
        for st in inc_steps
    )
    has_agg_op = any(
        label_key(st.operation).replace(" ", "_") in _AGGREGATION_OPS
        or "expected_population_ids" in st.parameters
        or "population_complete" in st.parameters
        for st in inc_steps
    )
    direct_premises = [graph.claims[pid] for st in inc_steps for pid in st.premises if pid in graph.claims]
    has_event_premises = any(n.event_id is not None for n in direct_premises)
    has_agg_pred = conc_pred_key in _AGGREGATION_PREDICATES or "total" in conc_pred_key or "subtotal" in conc_pred_key
    has_nl_agg_context = bool(
        (q_text or c_text)
        and (_AGGREGATION_TEXT_RE.search(q_text) or _AGGREGATION_TEXT_RE.search(c_text))
    )
    is_aggregation = bool(
        not is_comparison_conc
        and not (is_age_or_duration and not has_agg_op and not has_event_premises)
        and (
            has_agg_op
            or (has_nl_agg_context and (len(anc_nodes) >= 2 or has_agg_pred))
            or (has_agg_pred and (len(direct_premises) >= 3 or has_event_premises))
        )
    )

    # Conditions / exceptions present on any ancestor premise
    has_conditions = any(
        bool(n.conditions) or n.condition_status in ("conditional", "unknown_condition", "unrepresentable_condition")
        for n in anc_nodes
    )

    # Extract temporal anchor parameters when available so temporal obligations cannot be bypassed by `op="add"`
    inferred_temporal_params: dict[str, Any] = {}
    if is_age_or_duration:
        anchor_age_node = next(
            (
                n for n in anc_nodes
                if label_key(n.predicate).replace(" ", "_") in {"age", "age_at_anchor", "anchor_age", "edad"}
                or (n.unit and label_key(n.unit) in _DURATION_COMPUTABLE_UNITS and n.valid_from)
            ),
            None,
        )
        birth_node = next(
            (n for n in anc_nodes if label_key(n.predicate).replace(" ", "_") in {"birth_date", "fecha_nacimiento"} or _is_date_string(n.value)),
            None,
        )
        year_premises = [
            n for n in anc_nodes
            if (isinstance(n.value, int) and 1500 <= n.value <= 2150 and label_key(n.predicate) in {"year", "anchor_year", "reference_year", "target_year", "date"})
            or (_is_date_string(n.value) and n is not birth_node)
        ]

        for st in inc_steps:
            for k in ("start_date", "birth_date", "end_date", "reference_date", "anchor_age", "anchor_date"):
                if k in st.parameters and k not in inferred_temporal_params:
                    inferred_temporal_params[k] = st.parameters[k]

        if anchor_age_node is not None and "anchor_age" not in inferred_temporal_params:
            if isinstance(anchor_age_node.value, (int, float)):
                inferred_temporal_params["anchor_age"] = anchor_age_node.value
            if anchor_age_node.valid_from and "anchor_date" not in inferred_temporal_params:
                inferred_temporal_params["anchor_date"] = anchor_age_node.valid_from
        elif birth_node is not None and "start_date" not in inferred_temporal_params and "birth_date" not in inferred_temporal_params:
            inferred_temporal_params["start_date"] = birth_node.value if _is_date_string(birth_node.value) else birth_node.valid_from

        if "anchor_age" in inferred_temporal_params and "anchor_date" not in inferred_temporal_params and year_premises:
            yp = year_premises[0]
            inferred_temporal_params["anchor_date"] = str(yp.value)

        if "reference_date" not in inferred_temporal_params and "end_date" not in inferred_temporal_params:
            if as_of:
                inferred_temporal_params["reference_date"] = as_of
            elif len(year_premises) >= 2:
                inferred_temporal_params["reference_date"] = str(year_premises[-1].value)
            elif len(year_premises) == 1 and "anchor_age" not in inferred_temporal_params:
                # Only one year premise and no anchor_age: could be reference year if birth_node existed
                inferred_temporal_params["reference_date"] = str(year_premises[0].value)
            else:
                q_date = _extract_date_from_text(q_text)
                if q_date:
                    inferred_temporal_params["reference_date"] = q_date

    discovered_domains: list[str] = []
    if is_age_or_duration:
        discovered_domains.append("age_and_duration")
    if is_quantity:
        discovered_domains.append("quantities")
    if is_aggregation:
        discovered_domains.append("aggregations")
    if has_conditions:
        discovered_domains.append("conditions_and_exceptions")

    # Register uncertainty if predicate is explicitly generic ("number", "value", etc.)
    # and domain could not be cleanly verified via Vocabulary FieldSpec
    uncertain_semantic_type = bool(
        generic_label and conc_pred_spec is None
    )

    return {
        "discovered_domains": tuple(discovered_domains),
        "generic_label_detected": generic_label,
        "semantic_type_uncertain": uncertain_semantic_type,
        "is_age_or_duration": is_age_or_duration,
        "is_quantity": is_quantity,
        "is_aggregation": is_aggregation,
        "has_conditions": has_conditions,
        "temporal_params": inferred_temporal_params,
    }


def generate_mandatory_obligations(
    graph: JustificationGraph,
    *,
    question: str | Mapping[str, Any] | None = None,
    candidate_text: str | None = None,
    documents: Mapping[str, Any] | None = None,
    domain_profile: Any = None,
    as_of: str | None = None,
    has_material_conflict: bool = False,
) -> tuple[VerificationObligation, ...]:
    """Generate versioned, non-removable verification obligations imposed by the engine."""
    sem = infer_conclusion_semantic_domain(
        graph,
        question=question,
        candidate_text=candidate_text,
        domain_profile=domain_profile,
        as_of=as_of,
    )
    conc_id = graph.conclusion_id
    anc_ids = graph.ancestor_claims(conc_id)
    anc_nodes = [graph.claims[aid] for aid in sorted(anc_ids) if aid in graph.claims]
    inc_steps = graph.incoming_steps(conc_id)
    has_cited_sources = bool(
        graph.claims[conc_id].evidence
        or graph.claims[conc_id].evidence_unit_ids
        or any(n.evidence or n.evidence_unit_ids for n in anc_nodes)
        or (documents and len(documents) > 0)
    )

    obligations: list[VerificationObligation] = []

    # 1. Age and duration temporal precision
    if sem["is_age_or_duration"]:
        t_params = dict(sem["temporal_params"])
        t_params["mandatory_check"] = "temporal_precision"
        if not any(label_key(st.operation).replace(" ", "_") in _TEMPORAL_OPS for st in inc_steps):
            t_params.setdefault("temporal_op", "age")
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::age_and_duration",
                family="temporal",
                description="Verify temporal date precision, intervals, and valid derived range.",
                critical=True,
                parameters=t_params,
                origin="engine_mandatory",
            )
        )

    # 2. Quantities: unit compatibility, precision, rounding
    if sem["is_quantity"]:
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::quantities",
                family="quantities",
                description="Verify unit compatibility, registry validity, and explicit rounding/precision.",
                critical=True,
                parameters={
                    "mandatory_check": "quantity_precision_and_units",
                    "temporal_age_mode": sem["is_age_or_duration"],
                },
                origin="engine_mandatory",
            )
        )

    # 3. Documental attribution (non-corroborated narrative sources -> explicit source qualification)
    if has_cited_sources:
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::documental_attribution",
                family="document_authority",
                description="Verify external corroboration or explicit source-dependent qualification.",
                critical=True,
                parameters={"mandatory_check": "source_conditional_attribution"},
                origin="engine_mandatory",
            )
        )

    # 4. Conditions and exceptions preservation
    if sem["has_conditions"]:
        req_conds: list[dict[str, Any]] = []
        for n in anc_nodes:
            for cond in n.conditions:
                req_conds.append(dict(cond))
            if not n.conditions and n.condition_status != "unconditional":
                req_conds.append({"claim_id": n.claim_id, "condition_status": n.condition_status})
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::conditions_and_exceptions",
                family="explicit_logic",
                description="Verify all required conditions and exceptions from premises are preserved.",
                critical=True,
                parameters={
                    "mandatory_check": "condition_preservation",
                    "ancestor_claim_ids": sorted(anc_ids),
                    "required_conditions": req_conds,
                },
                origin="engine_mandatory",
            )
        )

    # 5. Aggregations (subtotal / incomplete population check)
    if sem["is_aggregation"]:
        agg_op = "sum"
        agg_params: dict[str, Any] = {"mandatory_check": "aggregation_completeness"}
        for st in inc_steps:
            op_k = label_key(st.operation).replace(" ", "_")
            if op_k in _AGGREGATION_OPS:
                agg_op = st.operation
                for pk, pv in st.parameters.items():
                    agg_params.setdefault(pk, pv)
                break
        agg_params.setdefault("aggregation_op", agg_op)
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::aggregations",
                family="aggregation",
                description="Verify unit commensurability, event deduplication, and population completeness.",
                critical=True,
                parameters=agg_params,
                origin="engine_mandatory",
            )
        )

    # 6. Material conflict disclosure
    if has_material_conflict:
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::material_conflict",
                family="document_authority",
                description="Verify unresolved material conflicts are not presented unambiguously.",
                critical=True,
                parameters={"mandatory_check": "material_conflict_disclosure"},
                origin="engine_mandatory",
            )
        )

    # 7. Ambiguous or unrecovered generic semantic label uncertainty
    if sem["semantic_type_uncertain"]:
        obligations.append(
            VerificationObligation(
                obligation_id="mandatory::semantic_type_uncertainty",
                family="document_authority",
                description="Record uncertainty when conclusion uses generic label lacking semantic typing.",
                critical=(len(sem["discovered_domains"]) == 0),
                parameters={
                    "mandatory_check": "semantic_type_uncertainty",
                    "discovered_domains": list(sem["discovered_domains"]),
                },
                origin="engine_mandatory",
            )
        )

    return tuple(obligations)
