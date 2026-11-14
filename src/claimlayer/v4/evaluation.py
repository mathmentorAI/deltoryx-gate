"""Graduated Justification Evaluation, Lineage-Safe Propagation & Incremental Cache (Phases 3–4).

Evaluates a proposed response / JustificationGraph without requiring an authorization
policy, verifies what is deterministically checkable across the 9 verifier families,
tracks coverage and pending obligations, enforces correlation-safe support propagation,
supports content-addressed incremental recomputation, and obeys explicit cost/latency budgets.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import math
import time
from typing import Any, Iterable, Mapping, Sequence

from ..semantic.models import EvidenceRef, SourceDocument, canonical, digest
from .correspondence import (
    compute_response_digest,
    verify_response_graph_correspondence,
)
from .domain_profile import DomainProfile
from .engine import EvidenceLayer, evaluate_unit_admissibility
from .fidelity import check_clause_boundary_completeness, verify_extraction_fidelity
from .justification import (
    OBLIGATION_GENERATOR_VERSION,
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    VerificationObligation,
    effective_epistemic_kind,
    generate_mandatory_obligations,
    infer_conclusion_semantic_domain,
)
from .models import AcceptancePolicy, EvidenceUnit
from .verifiers import (
    DEFAULT_VERIFIER_REGISTRY,
    VerifierContext,
    VerifierRegistry,
    VerifierResult,
    _convert_to_base_decimal,
    _parse_date_bounds,
    _to_decimal,
)
from .vocabulary import UnmappedProposal, label_key

EVALUATION_ASSESSMENTS: tuple[str, ...] = (
    "supported",
    "partially_supported",
    "contested",
    "unsupported",
    "refuted",
    "inconclusive",
)


@dataclass(frozen=True)
class VerificationBudget:
    """Explicit compute/cost/latency budget governing graph verification."""

    max_steps: int = 256
    max_depth: int = 16
    max_paths: int = 50
    max_semantic_calls: int = 0
    max_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 0 or self.max_depth < 0 or self.max_paths < 1 or self.max_semantic_calls < 0:
            raise ValueError("Budget bounds must be non-negative (max_paths >= 1).")
        if self.max_latency_ms is not None and (not math.isfinite(self.max_latency_ms) or self.max_latency_ms < 0):
            raise ValueError("max_latency_ms must be a non-negative finite number or None.")


@dataclass
class _CachedStepEntry:
    cache_key: str
    effective_node_digest: str
    results: tuple[VerifierResult, ...]
    lineages: tuple[tuple[frozenset[str], float], ...]
    source_deps: frozenset[str]
    refuted: bool


@dataclass
class _CachedLeafEntry:
    cache_key: str
    effective_node_digest: str
    checks: tuple[VerifierResult, ...]
    lineages: tuple[tuple[frozenset[str], float], ...]
    source_deps: frozenset[str]
    refuted: bool
    unverified_attestor: bool
    effective_kind: str


class VerificationCache:
    """Content-addressed incremental cache with dependency-driven invalidation."""

    def __init__(self) -> None:
        self._leaf_entries: dict[str, _CachedLeafEntry] = {}
        self._step_entries: dict[str, _CachedStepEntry] = {}
        # source_key (doc_id / unit_id / claim_id) -> set of claim/step cache keys
        self._source_to_leaf_keys: dict[str, set[str]] = {}
        self._source_to_step_keys: dict[str, set[str]] = {}
        self._last_source_digests: dict[str, str] = {}

    def clear(self) -> None:
        self._leaf_entries.clear()
        self._step_entries.clear()
        self._source_to_leaf_keys.clear()
        self._source_to_step_keys.clear()
        self._last_source_digests.clear()

    def register_leaf(self, key: str, entry: _CachedLeafEntry) -> None:
        self._leaf_entries[key] = entry
        for dep in entry.source_deps:
            self._source_to_leaf_keys.setdefault(dep, set()).add(key)

    def register_step(self, key: str, entry: _CachedStepEntry) -> None:
        self._step_entries[key] = entry
        for dep in entry.source_deps:
            self._source_to_step_keys.setdefault(dep, set()).add(key)

    def get_leaf(self, key: str) -> _CachedLeafEntry | None:
        return self._leaf_entries.get(key)

    def get_step(self, key: str) -> _CachedStepEntry | None:
        return self._step_entries.get(key)

    def invalidate_changed_sources(self, current_source_digests: Mapping[str, str]) -> set[str]:
        """Invalidate only entries depending on sources whose digest changed.

        Returns the set of invalidated cache keys across leaves and steps.
        """
        invalidated: set[str] = set()
        for source_id, new_digest in current_source_digests.items():
            old = self._last_source_digests.get(source_id)
            if old is not None and old != new_digest:
                for lk in list(self._source_to_leaf_keys.get(source_id, ())):
                    self._leaf_entries.pop(lk, None)
                    invalidated.add(lk)
                for sk in list(self._source_to_step_keys.get(source_id, ())):
                    self._step_entries.pop(sk, None)
                    invalidated.add(sk)
            self._last_source_digests[source_id] = str(new_digest)
        return invalidated


@dataclass(frozen=True)
class ResponseAssessment:
    """Graduated evaluation output decoupled from operational authorization."""

    conclusion_id: str
    conclusion: ClaimNode
    effective_epistemic_kind: str
    assessment: str
    evidence_support_index: float
    conflict_index: float
    uncertainty_index: float
    verification_coverage: float
    critical_unverified_steps: tuple[str, ...]
    supporting_evidence: tuple[dict[str, Any], ...]
    contradicting_evidence: tuple[dict[str, Any], ...]
    inference_steps: tuple[dict[str, Any], ...]
    premises: tuple[dict[str, Any], ...]
    completed_checks: tuple[VerifierResult, ...]
    pending_obligations: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...]
    calibrated_correctness_probability: float | None = None
    domain_profile_id: str = ""
    domain_profile_version: str = ""
    domain_profile_fingerprint: str = ""
    assessment_digest: str = ""
    cache_metrics: dict[str, int] = field(default_factory=dict)
    supporting_lineage_count: int = 0
    contradicting_lineage_count: int = 0
    response_digest: str | None = None
    question: str | None = None
    candidate_response: str | None = None
    representation_mode: str = "graph_only"
    response_correspondence_verified: bool = False
    obligation_generator_version: str = OBLIGATION_GENERATOR_VERSION
    proposed_obligations: tuple[dict[str, Any], ...] = ()
    mandatory_obligations: tuple[dict[str, Any], ...] = ()
    semantic_domain_info: dict[str, Any] = field(default_factory=dict)
    correspondence_report: dict[str, Any] = field(default_factory=dict)
    critical_coverage: float = 1.0

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "conclusion_id": self.conclusion_id,
            "conclusion": self.conclusion.to_dict(),
            "effective_epistemic_kind": self.effective_epistemic_kind,
            "assessment": self.assessment,
            "evidence_support_index": round(self.evidence_support_index, 6),
            "conflict_index": round(self.conflict_index, 6),
            "uncertainty_index": round(self.uncertainty_index, 6),
            "verification_coverage": round(self.verification_coverage, 6),
            "critical_coverage": round(self.critical_coverage, 6),
            "critical_unverified_steps": list(self.critical_unverified_steps),
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "inference_steps": list(self.inference_steps),
            "premises": list(self.premises),
            "completed_checks": [c.to_dict() for c in self.completed_checks],
            "pending_obligations": list(self.pending_obligations),
            "assumptions": list(self.assumptions),
            "calibrated_correctness_probability": self.calibrated_correctness_probability,
            "domain_profile_id": self.domain_profile_id,
            "domain_profile_version": self.domain_profile_version,
            "domain_profile_fingerprint": self.domain_profile_fingerprint,
            "assessment_digest": self.assessment_digest,
            "cache_metrics": dict(self.cache_metrics),
            "response_digest": self.response_digest,
            "question": self.question,
            "candidate_response": self.candidate_response,
            "representation_mode": self.representation_mode,
            "response_correspondence_verified": self.response_correspondence_verified,
            "obligation_generator_version": self.obligation_generator_version,
            "proposed_obligations": list(self.proposed_obligations),
            "mandatory_obligations": list(self.mandatory_obligations),
            "semantic_domain_info": dict(self.semantic_domain_info),
            "correspondence_report": dict(self.correspondence_report),
        }


# ---------------------------------------------------------------------------
# Correlation-safe lineage fusion & pairwise claim conflict helpers
# ---------------------------------------------------------------------------

def _fuse_correlated_derivations(derivations: Sequence[tuple[frozenset[str], float]]) -> tuple[float, int]:
    """Combine derivations using idempotent `max` within overlapping source lineages
    and Noisy-OR only across disjoint source-lineage connected components.

    Guarantees that correlated multi-hop paths or multiple reformulations citing the
    same document / dependence_group NEVER inflate support.
    """
    valid = [(lineages, max(0.0, min(1.0, float(w)))) for lineages, w in derivations if w > 0.0]
    if not valid:
        return 0.0, 0

    n = len(valid)
    visited = [False] * n
    component_maxima: list[float] = []

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp_indices: list[int] = []
        comp_lineages: set[str] = set(valid[start][0])
        while stack:
            idx = stack.pop()
            comp_indices.append(idx)
            l_i = valid[idx][0]
            comp_lineages |= l_i
            for j in range(n):
                if not visited[j]:
                    # Connected iff sharing ANY underlying source / dependence_group
                    if l_i & valid[j][0]:
                        visited[j] = True
                        stack.append(j)
        component_maxima.append(max(valid[k][1] for k in comp_indices))

    remaining = 1.0
    for m in component_maxima:
        remaining *= 1.0 - m
    return round(1.0 - remaining, 6), len(component_maxima)


def _node_lineages(
    node: ClaimNode,
    documents: Mapping[str, SourceDocument],
    profile: DomainProfile,
) -> frozenset[str]:
    """Return all underlying dependence groups / document IDs for a leaf ClaimNode."""
    groups: set[str] = set()
    if node.dependence_group:
        groups.add(f"dg:{label_key(node.dependence_group)}")
    for did in node.doc_ids():
        if did in profile.policy.known_source_dependencies:
            groups.add(f"dg:{label_key(profile.policy.known_source_dependencies[did])}")
        elif did in documents and documents[did].dependence_group:
            groups.add(f"dg:{label_key(documents[did].dependence_group)}")
        else:
            groups.add(f"doc:{did}")
    if not groups:
        groups.add(f"claim:{node.claim_id}")
    return frozenset(groups)


def _temporal_windows_overlap(a: ClaimNode, b: ClaimNode) -> bool:
    as_b = _parse_date_bounds(a.valid_from)
    ae_b = _parse_date_bounds(a.valid_until)
    bs_b = _parse_date_bounds(b.valid_from)
    be_b = _parse_date_bounds(b.valid_until)
    if (ae_b and bs_b and ae_b[1] < bs_b[0]) or (be_b and as_b and be_b[1] < as_b[0]):
        return False
    return True


def _claims_contradict(a: ClaimNode, b: ClaimNode, profile: DomainProfile) -> bool:
    """Typed three-valued contradiction check respecting UnitRegistry and temporal windows."""
    if not a.subject or not b.subject or not a.predicate or not b.predicate:
        return False
    if label_key(a.subject) != label_key(b.subject) or a.predicate != b.predicate:
        return False
    if not _temporal_windows_overlap(a, b):
        return False

    reg = profile.effective_unit_registry()
    na, nb = _to_decimal(a.value), _to_decimal(b.value)
    if na is not None and nb is not None:
        try:
            ba, da, _ = _convert_to_base_decimal(na, a.unit, reg)
            bb, db, _ = _convert_to_base_decimal(nb, b.unit, reg)
            rel_tol = Decimal("1e-9") * max(Decimal(1), abs(ba), abs(bb))
            same_val = (da == db) and (abs(ba - bb) <= rel_tol)
        except UnmappedProposal:
            same_val = canonical(a.value) == canonical(b.value) and a.unit == b.unit
        if a.polarity != b.polarity:
            return same_val
        return not same_val

    same_canonical = canonical(a.value) == canonical(b.value) and a.unit == b.unit
    if a.polarity != b.polarity:
        return same_canonical
    return not same_canonical


def _normalize_evidence_inputs(
    evidence: EvidenceLayer | Mapping[str, SourceDocument] | Sequence[SourceDocument] | Mapping[str, str] | None,
) -> tuple[dict[str, SourceDocument], EvidenceLayer | None]:
    if evidence is None:
        return {}, None
    if isinstance(evidence, EvidenceLayer):
        return {d.document_id: d for d in evidence.documents}, evidence
    if isinstance(evidence, Mapping):
        docs: dict[str, SourceDocument] = {}
        for k, v in evidence.items():
            if isinstance(v, SourceDocument):
                docs[v.document_id] = v
            elif isinstance(v, str):
                docs[str(k)] = SourceDocument(document_id=str(k), text=v)
            elif isinstance(v, Mapping):
                docs[str(k)] = SourceDocument(
                    document_id=str(v.get("document_id", k)),
                    text=str(v["text"]),
                    authority_rank=int(v.get("authority_rank", 60)),
                    weight=float(v.get("weight", 1.0)),
                    dependence_group=v.get("dependence_group"),
                    attribution=str(v.get("attribution", "source_statement")),
                )
        return docs, None
    if isinstance(evidence, Sequence):
        return {d.document_id: d for d in evidence}, None
    raise ValueError("Unsupported evidence container.")


def _evaluate_leaf_claim(
    node: ClaimNode,
    *,
    documents: Mapping[str, SourceDocument],
    profile: DomainProfile,
    layer: EvidenceLayer | None,
    as_of: str | None,
) -> tuple[tuple[VerifierResult, ...], tuple[tuple[frozenset[str], float], ...], bool, bool, str]:
    """Evaluate a leaf ClaimNode (grounding, fidelity, authority, reliability)."""
    unverified_roles = {label_key(r) for r in profile.policy.unverified_attestor_roles}
    unverified_attestor = bool(node.attestor and label_key(node.attestor) in unverified_roles)
    eff_kind = effective_epistemic_kind(node.kind, (), unverified_attestor=unverified_attestor)

    checks: list[VerifierResult] = []
    refuted = False
    lineages = _node_lineages(node, documents, profile)

    # 1. If linked to an EvidenceLayer unit, run full v4 evaluate_unit_admissibility()
    if layer is not None and node.evidence_unit_ids and profile.vocabulary is not None:
        layer_snap = layer.snapshot()
        units_by_id = {u.unit_id: u for u in layer_snap["units"]}
        layer_docs_map = layer_snap["documents"]
        acc_pol = AcceptancePolicy(tau=profile.policy.tau, delta=profile.policy.delta)
        for uid in node.evidence_unit_ids:
            u = units_by_id.get(uid)
            if u is None:
                refuted = True
                checks.append(
                    VerifierResult(
                        obligation_id=f"leaf::{node.claim_id}::{uid}",
                        step_id=f"leaf::{node.claim_id}",
                        family="document_authority",
                        status="refuted",
                        guarantee_kind="deterministic_formal",
                        reasons=("MISSING_EVIDENCE_UNIT",),
                        verifier_id="leaf_ledger_admissibility",
                        verifier_version="1.0.0",
                    )
                )
            else:
                uev = evaluate_unit_admissibility(
                    u,
                    vocabulary=profile.vocabulary,
                    policy=acc_pol,
                    profile=profile.epistemic_profile,
                    units_by_id=units_by_id,
                    documents=layer_docs_map,
                    stored_field_evidences=layer_snap.get("field_evidences"),
                    as_of=as_of,
                )
                if not uev.admissible:
                    is_unassessed = uev.issue_code == "UNASSESSED_READER_RELIABILITY"
                    if not is_unassessed:
                        refuted = True
                    checks.append(
                        VerifierResult(
                            obligation_id=f"leaf::{node.claim_id}::{uid}",
                            step_id=f"leaf::{node.claim_id}",
                            family="document_authority",
                            status="unknown" if is_unassessed else "refuted",
                            guarantee_kind="deterministic_formal",
                            evidence={"unit_id": uid, "reason": uev.reason, "issue_code": uev.issue_code},
                            reasons=(str(uev.issue_code or uev.reason or "UNIT_INADMISSIBLE"),),
                            verifier_id="leaf_ledger_admissibility",
                            verifier_version="1.0.0",
                        )
                    )
                else:
                    checks.append(
                        VerifierResult(
                            obligation_id=f"leaf::{node.claim_id}::{uid}",
                            step_id=f"leaf::{node.claim_id}",
                            family="document_authority",
                            status="verified",
                            guarantee_kind="deterministic_formal",
                            evidence={"unit_id": uid, "support": uev.support},
                            reasons=("UNIT_ADMISSIBILITY_VERIFIED",),
                            verifier_id="leaf_ledger_admissibility",
                            verifier_version="1.0.0",
                        )
                    )

    # 2. If leaf carries EvidenceRefs and documents map is available, verify anchors/revision/fidelity
    if node.evidence and documents:
        for idx, ref in enumerate(node.evidence):
            doc = documents.get(ref.document_id)
            if doc is None:
                checks.append(
                    VerifierResult(
                        obligation_id=f"leaf::{node.claim_id}::ref{idx}",
                        step_id=f"leaf::{node.claim_id}",
                        family="document_authority",
                        status="unknown",
                        guarantee_kind="deterministic_formal",
                        evidence={"document_id": ref.document_id},
                        reasons=("UNREAD_OR_MISSING_DOCUMENT",),
                        verifier_id="leaf_span_grounding",
                        verifier_version="1.0.0",
                    )
                )
                continue
            if ref.revision != "rev-0" and ref.revision != doc.revision:
                refuted = True
                checks.append(
                    VerifierResult(
                        obligation_id=f"leaf::{node.claim_id}::ref{idx}",
                        step_id=f"leaf::{node.claim_id}",
                        family="document_authority",
                        status="refuted",
                        guarantee_kind="deterministic_formal",
                        evidence={"document_id": ref.document_id},
                        reasons=("STALE_DOCUMENT_REVISION",),
                        verifier_id="leaf_span_grounding",
                        verifier_version="1.0.0",
                    )
                )
                continue
            cond_check = node.condition_status == "unconditional"
            findings = check_clause_boundary_completeness(doc.text, ref, check_conditional=cond_check)
            if findings:
                refuted = True
                checks.append(
                    VerifierResult(
                        obligation_id=f"leaf::{node.claim_id}::ref{idx}",
                        step_id=f"leaf::{node.claim_id}",
                        family="document_authority",
                        status="refuted",
                        guarantee_kind="deterministic_formal",
                        evidence={"codes": [f.code for f in findings]},
                        reasons=tuple(f.code for f in findings),
                        verifier_id="leaf_span_grounding",
                        verifier_version="1.0.0",
                    )
                )
                continue

        # Also run full v4 extraction fidelity check when predicate & entity belong to Vocabulary
        if (
            not refuted
            and profile.vocabulary is not None
            and any(p.name == node.predicate for p in profile.vocabulary.predicates)
            and node.subject
        ):
            try:
                canon_ent = profile.vocabulary.entity(node.subject)
                doc0 = documents.get(node.evidence[0].document_id)
                if doc0 is not None:
                    prop_payload: dict[str, Any] = {
                        "entity": canon_ent,
                        "predicate": node.predicate,
                        "value": (
                            {"value": node.value, "unit": node.unit}
                            if node.unit and not isinstance(node.value, Mapping)
                            else node.value
                        ),
                        "unit": node.unit,
                        "polarity": "affirmed" if node.polarity == 1 else "negated",
                        "assertion_kind": node.assertion_kind,
                        "conditions": list(node.conditions),
                        "condition_status": node.condition_status,
                        "scope": dict(node.scope),
                        "valid_from": node.valid_from,
                        "valid_until": node.valid_until,
                    }
                    tmp_unit = EvidenceUnit(
                        proposition=prop_payload,
                        evidence=node.evidence,
                        basis="explicit",
                        origin="reader",
                        producer={"reader_id": node.attestor or "graph_proposal"},
                        structural_reliability=node.structural_reliability,
                    )
                    fid_findings = verify_extraction_fidelity(
                        doc0.text,
                        tmp_unit,
                        field_evidence=node.field_evidence or None,
                        vocabulary=profile.vocabulary,
                        documents=documents,
                    )
                    if fid_findings:
                        refuted = True
                        checks.append(
                            VerifierResult(
                                obligation_id=f"leaf::{node.claim_id}::fidelity",
                                step_id=f"leaf::{node.claim_id}",
                                family="document_authority",
                                status="refuted",
                                guarantee_kind="deterministic_formal",
                                evidence={"codes": [f.code for f in fid_findings]},
                                reasons=tuple(f.code for f in fid_findings),
                                verifier_id="leaf_extraction_fidelity",
                                verifier_version="1.0.0",
                            )
                        )
            except (UnmappedProposal, ValueError, KeyError):
                # Do not crash if a claim holds provisional/derived/date representations
                # not yet registered in a narrow Vocabulary field schema.
                pass

    # Check document superseded metadata
    if node.metadata.get("superseded") or node.metadata.get("superseded_by"):
        refuted = True
        checks.append(
            VerifierResult(
                obligation_id=f"leaf::{node.claim_id}::superseded",
                step_id=f"leaf::{node.claim_id}",
                family="document_authority",
                status="refuted",
                guarantee_kind="deterministic_formal",
                reasons=("SUPERSEDED_OR_INACTIVE_DOCUMENT",),
                verifier_id="leaf_document_authority",
                verifier_version="1.0.0",
            )
        )

    if unverified_attestor and node.kind in ("extraction", "derivation"):
        checks.append(
            VerifierResult(
                obligation_id=f"leaf::{node.claim_id}::attestor",
                step_id=f"leaf::{node.claim_id}",
                family="document_authority",
                status="unknown",
                guarantee_kind="deterministic_formal",
                evidence={"attestor": node.attestor},
                reasons=("UNVERIFIED_ATTESTOR_ROLE",),
                verifier_id="leaf_document_authority",
                verifier_version="1.0.0",
            )
        )

    if not checks and not refuted:
        if node.evidence or node.evidence_unit_ids:
            checks.append(
                VerifierResult(
                    obligation_id=f"leaf::{node.claim_id}::grounding",
                    step_id=f"leaf::{node.claim_id}",
                    family="document_authority",
                    status="verified",
                    guarantee_kind="deterministic_formal",
                    reasons=("LEAF_GROUNDING_VERIFIED",),
                    verifier_id="leaf_span_grounding",
                    verifier_version="1.0.0",
                )
            )
        elif node.kind == "extraction":
            checks.append(
                VerifierResult(
                    obligation_id=f"leaf::{node.claim_id}::missing_evidence",
                    step_id=f"leaf::{node.claim_id}",
                    family="document_authority",
                    status="unknown",
                    guarantee_kind="deterministic_formal",
                    reasons=("MISSING_EVIDENCE_CITATIONS",),
                    verifier_id="leaf_span_grounding",
                    verifier_version="1.0.0",
                )
            )

    if refuted:
        return tuple(checks), (), True, unverified_attestor, eff_kind

    # Compute support weight for leaf claim (NEVER treated as calibrated probability)
    doc_weights = [documents[did].weight for did in node.doc_ids() if did in documents]
    base_doc_w = min(doc_weights) if doc_weights else (0.90 if node.evidence else 0.0)
    if node.structural_reliability is not None:
        rel_w = float(node.structural_reliability)
    elif node.attestor and node.attestor in profile.calibration.extractor_reliabilities:
        rel_w = float(profile.calibration.extractor_reliabilities[node.attestor])
    elif profile.calibration.default_extractor_reliability is not None:
        rel_w = float(profile.calibration.default_extractor_reliability)
    else:
        rel_w = 1.0 if (node.evidence or node.evidence_unit_ids) else 0.0

    weight = max(0.0, min(1.0, base_doc_w * rel_w))
    if unverified_attestor:
        weight = min(weight, 0.25)
    elif eff_kind in ("hypothesis", "prediction", "recommendation") and not (node.evidence or node.evidence_unit_ids):
        weight = 0.0

    # Authority rank adjustment for low-ranked memos/allegations if attribution ranks lower than policy expectations
    for did in node.doc_ids():
        d = documents.get(did)
        if d is not None:
            attr_rank = profile.policy.authority_ranks.get(d.attribution)
            if attr_rank is not None and attr_rank <= 20:
                weight = min(weight, 0.30)

    leaf_derivations = ((lineages, round(weight, 6)),) if weight > 0.0 else ()
    return tuple(checks), leaf_derivations, False, unverified_attestor, eff_kind


def _obligation_record(ob: VerificationObligation, step_id: str) -> dict[str, Any]:
    """Serialize an obligation with origin and criticality metadata."""
    return {
        "obligation_id": ob.obligation_id,
        "step_id": step_id,
        "family": ob.family,
        "origin": ob.origin,
        "critical": ob.critical,
        "mandatory_check": ob.parameters.get("mandatory_check"),
        "parameters": canonical(dict(ob.parameters)),
    }


def evaluate_response(
    response: JustificationGraph | Mapping[str, Any],
    evidence: EvidenceLayer | Mapping[str, SourceDocument] | Sequence[SourceDocument] | Mapping[str, str] | None = None,
    domain_profile: DomainProfile | None = None,
    *,
    cache: VerificationCache | None = None,
    budget: VerificationBudget | None = None,
    as_of: str | None = None,
    clause_relations: Sequence[Any] = (),
    verifier_registry: VerifierRegistry | None = None,
    question: str | Mapping[str, Any] | None = None,
    candidate_response: str | None = None,
    representation_mode: str | None = None,
) -> ResponseAssessment:
    """Evaluate a generated response / JustificationGraph without requiring an authorization policy."""
    t_start = time.monotonic()
    profile = domain_profile or DomainProfile(profile_id="default")
    profile.validate_profile()
    active_budget = budget or VerificationBudget()
    registry = verifier_registry or DEFAULT_VERIFIER_REGISTRY
    documents, layer = _normalize_evidence_inputs(evidence)

    if isinstance(question, Mapping):
        q_str: str | None = str(question.get("text") or question.get("question") or "").strip() or None
    elif question is not None:
        q_str = str(question)
    else:
        q_str = None

    allowed_modes = ("graph_only", "supplied_graph_with_response", "adapter_extracted_graph")
    if representation_mode is None:
        rep_mode = "graph_only" if candidate_response is None else "supplied_graph_with_response"
    else:
        if representation_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported representation_mode {representation_mode!r}; expected one of {allowed_modes}"
            )
        rep_mode = representation_mode

    resp_digest = compute_response_digest(candidate_response, question)

    graph = JustificationGraph.from_response(response, vocabulary=profile.vocabulary)
    structural_errors = graph.validate_structure()
    if structural_errors:
        # Handle cyclic or structurally invalid graph deterministically as refuted/inconclusive
        conc = graph.claims.get(graph.conclusion_id) or next(iter(graph.claims.values()))
        check = VerifierResult(
            obligation_id="graph::structure",
            step_id="graph::structure",
            family="explicit_logic",
            status="refuted",
            guarantee_kind="deterministic_formal",
            evidence={"errors": structural_errors},
            reasons=("INVALID_JUSTIFICATION_GRAPH_STRUCTURE",),
            verifier_id="graph_structure_validator",
            verifier_version="1.0.0",
        )
        return ResponseAssessment(
            conclusion_id=conc.claim_id,
            conclusion=conc,
            effective_epistemic_kind=conc.kind,
            assessment="refuted",
            evidence_support_index=0.0,
            conflict_index=1.0,
            uncertainty_index=0.0,
            verification_coverage=1.0,
            critical_coverage=1.0,
            critical_unverified_steps=(),
            supporting_evidence=(),
            contradicting_evidence=(),
            inference_steps=tuple(s.to_dict() for s in graph.steps),
            premises=(),
            completed_checks=(check,),
            pending_obligations=(),
            assumptions=(),
            calibrated_correctness_probability=None,
            domain_profile_id=profile.profile_id,
            domain_profile_version=profile.version,
            domain_profile_fingerprint=profile.fingerprint(),
            assessment_digest=digest({
                "conclusion_id": conc.claim_id,
                "errors": structural_errors,
                "response_digest": resp_digest,
                "representation_mode": rep_mode,
                "obligation_generator_version": OBLIGATION_GENERATOR_VERSION,
            }),
            cache_metrics={"cache_hits": 0, "cache_misses": 1, "recomputed_nodes": 1, "reused_nodes": 0, "budget_deferred_steps": 0},
            response_digest=resp_digest,
            question=q_str,
            candidate_response=candidate_response,
            representation_mode=rep_mode,
            response_correspondence_verified=False,
            obligation_generator_version=OBLIGATION_GENERATOR_VERSION,
        )

    profile_fp = profile.fingerprint()
    verifier_versions = registry.versions_snapshot()
    topo_ids = graph.topological_order()

    # Invalidate changed source revisions cleanly if cache is active
    if cache is not None and documents:
        cache.invalidate_changed_sources({did: doc.revision for did, doc in documents.items()})

    semantic_domain_info = infer_conclusion_semantic_domain(
        graph,
        question=question,
        candidate_text=candidate_response,
        domain_profile=profile,
        as_of=as_of,
    )
    preliminary_corr: dict[str, Any]
    if candidate_response is not None:
        preliminary_corr = verify_response_graph_correspondence(
            candidate_response,
            graph,
            question=question,
            documents=documents,
            unit_registry=profile.vocabulary.units if profile.vocabulary is not None else None,
            has_material_conflict=False,
        )
    else:
        preliminary_corr = {
            "response_checked": False,
            "correspondence_ok": False,
            "issues": (),
        }
    base_mandatory_obs = generate_mandatory_obligations(
        graph,
        question=question,
        candidate_text=candidate_response,
        documents=documents,
        domain_profile=profile,
        as_of=as_of,
        has_material_conflict=False,
    )
    non_conflict_mandatory_obs = tuple(
        ob for ob in base_mandatory_obs if ob.parameters.get("mandatory_check") != "conflict_disclosure"
    )
    evaluated_mandatory_ids: set[str] = set()
    proposed_obligations_records: list[dict[str, Any]] = []
    mandatory_obligations_records: list[dict[str, Any]] = []
    critical_check_keys: set[tuple[str, str]] = set()
    conc_extra_steps: set[str] = set()

    # Compute node depth from leaves
    node_depth: dict[str, int] = {}
    for cid in topo_ids:
        inc = graph.incoming_steps(cid)
        if not inc:
            node_depth[cid] = 0
        else:
            node_depth[cid] = 1 + max((node_depth.get(pid, 0) for st in inc for pid in st.premises), default=0)

    node_derivations: dict[str, list[tuple[frozenset[str], float]]] = {cid: [] for cid in graph.claims}
    node_effective_kind: dict[str, str] = {}
    node_effective_digest: dict[str, str] = {}
    node_refuted: dict[str, bool] = {cid: False for cid in graph.claims}
    all_checks: list[VerifierResult] = []
    pending_obligations: list[dict[str, Any]] = []
    critical_unverified: list[str] = []
    all_assumptions: list[str] = []
    step_summaries: list[dict[str, Any]] = []

    cache_hits = 0
    cache_misses = 0
    recomputed_nodes = 0
    reused_nodes = 0
    deferred_steps = 0
    executed_steps = 0
    semantic_calls_used = 0
    mandatory_cache_hits = 0
    mandatory_cache_misses = 0

    def _budget_time_exceeded() -> bool:
        if active_budget.max_latency_ms is None:
            return False
        return (time.monotonic() - t_start) * 1000.0 > active_budget.max_latency_ms

    for cid in topo_ids:
        node = graph.claims[cid]
        inc_steps = graph.incoming_steps(cid)
        all_assumptions.extend(node.assumptions)

        if not inc_steps:
            # Leaf ClaimNode
            doc_revs = {did: documents[did].revision for did in node.doc_ids() if did in documents}
            leaf_key = digest({
                "kind": "leaf",
                "node": node.content_digest(),
                "doc_revs": doc_revs,
                "unit_ids": list(node.evidence_unit_ids),
                "as_of": as_of,
                "profile_fp": profile_fp,
            })
            cached = cache.get_leaf(leaf_key) if cache is not None else None
            if cached is not None:
                cache_hits += 1
                reused_nodes += 1
                leaf_checks, leaf_derivs, is_ref, _, eff_kind = (
                    cached.checks,
                    cached.lineages,
                    cached.refuted,
                    cached.unverified_attestor,
                    cached.effective_kind,
                )
                node_effective_digest[cid] = cached.effective_node_digest
            else:
                cache_misses += 1
                recomputed_nodes += 1
                leaf_checks, leaf_derivs, is_ref, unv_att, eff_kind = _evaluate_leaf_claim(
                    node, documents=documents, profile=profile, layer=layer, as_of=as_of
                )
                eff_dig = digest({"leaf_key": leaf_key, "refuted": is_ref, "eff_kind": eff_kind})
                node_effective_digest[cid] = eff_dig
                if cache is not None:
                    deps = frozenset(list(node.doc_ids()) + list(node.evidence_unit_ids) + [cid])
                    cache.register_leaf(
                        leaf_key,
                        _CachedLeafEntry(
                            cache_key=leaf_key,
                            effective_node_digest=eff_dig,
                            checks=leaf_checks,
                            lineages=leaf_derivs,
                            source_deps=deps,
                            refuted=is_ref,
                            unverified_attestor=unv_att,
                            effective_kind=eff_kind,
                        ),
                    )

            all_checks.extend(leaf_checks)
            node_derivations[cid].extend(leaf_derivs)
            node_refuted[cid] = is_ref
            node_effective_kind[cid] = eff_kind
            for chk in leaf_checks:
                all_assumptions.extend(chk.assumptions)
                critical_check_keys.add((chk.step_id, chk.obligation_id))
                if chk.status == "unknown":
                    pending_obligations.append({
                        "obligation_id": chk.obligation_id,
                        "step_id": chk.step_id,
                        "family": chk.family,
                        "reasons": list(chk.reasons),
                        "critical": True,
                        "origin": "engine_mandatory",
                    })
            continue

        # Derived ClaimNode: compute effective epistemic kind bounded by weakest premise across all incoming steps
        premise_eff_kinds = [
            node_effective_kind[pid] for st in inc_steps for pid in st.premises
        ]
        eff_kind = effective_epistemic_kind(node.kind, premise_eff_kinds)
        node_effective_kind[cid] = eff_kind

        step_eff_digests: list[str] = []
        for step in inc_steps:
            all_assumptions.extend(step.assumptions)
            step_depth = 1 + max((node_depth.get(pid, 0) for pid in step.premises), default=0)
            is_conclusion_step = cid == graph.conclusion_id
            effective_step_obligations: list[VerificationObligation] = list(step.obligations)
            for ob in step.obligations:
                if ob.origin == "engine_mandatory":
                    mandatory_obligations_records.append(_obligation_record(ob, step.step_id))
                    evaluated_mandatory_ids.add(ob.obligation_id)
                else:
                    proposed_obligations_records.append(_obligation_record(ob, step.step_id))
            # Keep probabilistic open_semantics steps isolated so PROBABILISTIC_ONLY_CRITICAL_STEP stays intact;
            # engine-imposed conclusion mandatory obligations are evaluated on dedicated mandatory::conclusion below.
            prem_digests = [node_effective_digest[pid] for pid in step.premises]
            step_cache_key = digest({
                "kind": "step",
                "step": step.step_digest(),
                "conclusion": node.content_digest(),
                "premise_eff_digests": prem_digests,
                "as_of": as_of,
                "profile_fp": profile_fp,
                "verifier_versions": verifier_versions,
                "obligation_generator_version": OBLIGATION_GENERATOR_VERSION,
                "mandatory_obs": [
                    digest(o.to_dict())
                    for o in effective_step_obligations
                    if o.origin == "engine_mandatory"
                ],
                "rep_mode": rep_mode if is_conclusion_step else "graph_only",
                "resp_digest": resp_digest if is_conclusion_step else None,
            })

            cached_step = cache.get_step(step_cache_key) if cache is not None else None
            if cached_step is not None:
                cache_hits += 1
                reused_nodes += 1
                step_results = list(cached_step.results)
                step_lineages = list(cached_step.lineages)
                step_refuted = cached_step.refuted
                step_eff_digests.append(cached_step.effective_node_digest)
            else:
                # Enforce depth, step count, and latency budgets
                if (
                    executed_steps >= active_budget.max_steps
                    or step_depth > active_budget.max_depth
                    or _budget_time_exceeded()
                ):
                    deferred_steps += 1
                    step_results = [
                        VerifierResult(
                            obligation_id=ob.obligation_id,
                            step_id=step.step_id,
                            family=ob.family,
                            status="unknown",
                            guarantee_kind=(
                                "probabilistic_model" if ob.family == "open_semantics" else "deterministic_formal"
                            ),
                            reasons=("PENDING_BUDGET",),
                            verifier_id="budget_guard",
                            verifier_version="1.0.0",
                        )
                        for ob in effective_step_obligations
                    ]
                    step_lineages = []
                    step_refuted = False
                else:
                    cache_misses += 1
                    recomputed_nodes += 1
                    executed_steps += 1

                    # Separate cheap deterministic obligations from expensive open_semantics
                    det_obs = [o for o in effective_step_obligations if o.family != "open_semantics"]
                    sem_obs = [o for o in effective_step_obligations if o.family == "open_semantics"]

                    premises_nodes = tuple(graph.claims[pid] for pid in step.premises)
                    det_ctx = VerifierContext(
                        step=step,
                        conclusion=node,
                        premises=premises_nodes,
                        domain_profile=profile,
                        documents=documents,
                        layer=layer,
                        clause_relations=tuple(clause_relations),
                        as_of=as_of,
                        allow_semantic_call=False,
                        question=question,
                        candidate_response=candidate_response if is_conclusion_step else None,
                        correspondence_facts=preliminary_corr if is_conclusion_step else {},
                    )

                    step_results = []
                    if len(det_obs) > 1:
                        with ThreadPoolExecutor(max_workers=min(4, len(det_obs))) as pool:
                            step_results.extend(
                                pool.map(lambda o: registry.verify_obligation(o, det_ctx), det_obs)
                            )
                    else:
                        for ob in det_obs:
                            step_results.append(registry.verify_obligation(ob, det_ctx))

                    for ob in sem_obs:
                        can_call_sem = (
                            semantic_calls_used < active_budget.max_semantic_calls
                            and not _budget_time_exceeded()
                        )
                        if can_call_sem and profile.semantic_evaluator.enabled:
                            semantic_calls_used += 1
                        sem_ctx = VerifierContext(
                            step=step,
                            conclusion=node,
                            premises=premises_nodes,
                            domain_profile=profile,
                            documents=documents,
                            layer=layer,
                            clause_relations=tuple(clause_relations),
                            as_of=as_of,
                            allow_semantic_call=can_call_sem,
                            question=question,
                            candidate_response=candidate_response if is_conclusion_step else None,
                            correspondence_facts=preliminary_corr if is_conclusion_step else {},
                        )
                        step_results.append(registry.verify_obligation(ob, sem_ctx))

                    any_prem_refuted = any(node_refuted[pid] for pid in step.premises)
                    any_check_refuted = any(r.status == "refuted" for r in step_results)
                    step_refuted = any_prem_refuted or any_check_refuted

                    applicable = [r for r in step_results if r.status != "not_applicable"]
                    all_obs_verified = bool(applicable) and all(r.status == "verified" for r in applicable)

                    step_lineages = []
                    if not step_refuted and all_obs_verified:
                        # Compute support via Gödel weakest-link over premises multiplied by step discount
                        if any(r.guarantee_kind == "probabilistic_model" for r in applicable):
                            discount = profile.policy.inference_discounts.get("probabilistic_model", 0.65)
                        elif step.operation in ("relation_composition", "relations"):
                            discount = profile.policy.inference_discounts.get("relation_composition", 0.92)
                        else:
                            discount = profile.policy.inference_discounts.get("deterministic_formal", 1.0)

                        premise_strengths: list[float] = []
                        combined_lineage_ids: set[str] = set()
                        for pid in step.premises:
                            p_derivs = node_derivations[pid]
                            p_sup, _ = _fuse_correlated_derivations(p_derivs)
                            premise_strengths.append(p_sup)
                            for lin_set, _ in p_derivs:
                                combined_lineage_ids |= lin_set
                        if premise_strengths:
                            weakest = min(premise_strengths)
                            step_strength = round(max(0.0, min(1.0, weakest * discount)), 6)
                            if eff_kind == "recommendation":
                                step_strength = min(step_strength, 0.50)
                            elif eff_kind in ("hypothesis", "prediction"):
                                step_strength = min(step_strength, 0.60)
                            if step_strength > 0.0:
                                step_lineages.append((frozenset(combined_lineage_ids or {cid}), step_strength))

                    s_eff_dig = digest({
                        "step_cache_key": step_cache_key,
                        "refuted": step_refuted,
                        "results": [r.to_dict() for r in step_results],
                        "lineages": [(sorted(l), w) for l, w in step_lineages],
                    })
                    step_eff_digests.append(s_eff_dig)
                    if cache is not None:
                        anc_sources: set[str] = {cid} | set(step.premises)
                        for pid in step.premises:
                            anc_sources |= set(graph.claims[pid].doc_ids())
                            for aid in graph.ancestor_claims(pid):
                                anc_sources.add(aid)
                                anc_sources |= set(graph.claims[aid].doc_ids())
                        cache.register_step(
                            step_cache_key,
                            _CachedStepEntry(
                                cache_key=step_cache_key,
                                effective_node_digest=s_eff_dig,
                                results=tuple(step_results),
                                lineages=tuple(step_lineages),
                                source_deps=frozenset(anc_sources),
                                refuted=step_refuted,
                            ),
                        )

            all_checks.extend(step_results)
            if step_refuted:
                node_refuted[cid] = True
            else:
                node_derivations[cid].extend(step_lineages)

            has_step_pending = False
            for r in step_results:
                all_assumptions.extend(r.assumptions)
                ob_match = next((o for o in effective_step_obligations if o.obligation_id == r.obligation_id), None)
                is_crit = bool(
                    (ob_match.critical if ob_match else True)
                    and (step.critical or (ob_match is not None and ob_match.origin == "engine_mandatory"))
                )
                if is_crit:
                    critical_check_keys.add((r.step_id, r.obligation_id))
                if r.status == "unknown":
                    pending_obligations.append({
                        "obligation_id": r.obligation_id,
                        "step_id": step.step_id,
                        "family": r.family,
                        "reasons": list(r.reasons),
                        "critical": is_crit,
                        "origin": ob_match.origin if ob_match else "engine_mandatory",
                    })
                    if is_crit:
                        has_step_pending = True

            if has_step_pending and step.step_id not in critical_unverified:
                critical_unverified.append(step.step_id)

            step_summaries.append({
                "step_id": step.step_id,
                "operation": step.operation,
                "premises": list(step.premises),
                "conclusion_id": step.conclusion_id,
                "refuted": step_refuted,
                "checks": [r.to_dict() for r in step_results],
            })

        node_effective_digest[cid] = digest({"cid": cid, "steps": step_eff_digests, "eff_kind": eff_kind})

    # Target conclusion evaluation & opposing claim detection
    conc_id = graph.conclusion_id
    conc_node = graph.claims[conc_id]
    conc_eff_kind = node_effective_kind[conc_id]
    conc_derivations = list(node_derivations[conc_id])
    conc_ancestors = graph.ancestor_claims(conc_id)

    # Identify contradicting claims inside graph for conclusion (excluding direct deriving ancestors)
    contradicting_derivations: list[tuple[frozenset[str], float]] = []
    contradicting_records: list[dict[str, Any]] = []
    for other_id, other_node in sorted(graph.claims.items()):
        if other_id == conc_id or other_id in conc_ancestors or node_refuted.get(other_id, False):
            continue
        if _claims_contradict(conc_node, other_node, profile):
            o_derivs = node_derivations.get(other_id, ())
            o_sup, _ = _fuse_correlated_derivations(o_derivs)
            if o_sup > 0.0:
                contradicting_derivations.extend(o_derivs)
                contradicting_records.append({
                    "claim_id": other_id,
                    "subject": other_node.subject,
                    "predicate": other_node.predicate,
                    "value": other_node.value,
                    "unit": other_node.unit,
                    "support": o_sup,
                })

    has_material_conflict = bool(contradicting_records)
    correspondence_report: dict[str, Any]
    if candidate_response is not None:
        correspondence_report = verify_response_graph_correspondence(
            candidate_response,
            graph,
            question=question,
            documents=documents,
            unit_registry=profile.vocabulary.units if profile.vocabulary is not None else None,
            has_material_conflict=has_material_conflict,
            semantic_evaluator=profile.semantic_evaluator,
        )
    else:
        correspondence_report = {
            "response_checked": False,
            "correspondence_ok": False,
            "issues": (),
        }

    final_mandatory_obs = generate_mandatory_obligations(
        graph,
        question=question,
        candidate_text=candidate_response,
        documents=documents,
        domain_profile=profile,
        as_of=as_of,
        has_material_conflict=has_material_conflict,
    )
    remaining_mandatory = [ob for ob in final_mandatory_obs if ob.obligation_id not in evaluated_mandatory_ids]
    existing_conc_steps = graph.incoming_steps(conc_id)
    if remaining_mandatory:
        target_step_id = "mandatory::conclusion"
        conc_extra_steps.add(target_step_id)
        merged_premises = tuple(dict.fromkeys(pid for st in existing_conc_steps for pid in st.premises))
        merged_params: dict[str, Any] = {}
        primary_op = "mandatory_conclusion_verification"
        if existing_conc_steps:
            primary_op = existing_conc_steps[0].operation
            for st in existing_conc_steps:
                for pk, pv in st.parameters.items():
                    merged_params.setdefault(pk, pv)
        synthetic_step = InferenceStep(
            step_id=target_step_id,
            operation=primary_op,
            premises=merged_premises,
            conclusion_id=conc_id,
            obligations=tuple(remaining_mandatory),
            parameters=merged_params,
            critical=True,
        )
        mandatory_cache_key = digest({
            "kind": "mandatory_conclusion",
            "conclusion": conc_node.content_digest(),
            "premise_eff_digests": [node_effective_digest[pid] for pid in merged_premises if pid in node_effective_digest],
            "doc_revs": {did: doc.revision for did, doc in documents.items()},
            "as_of": as_of,
            "profile_fp": profile_fp,
            "verifier_versions": verifier_versions,
            "obligation_generator_version": OBLIGATION_GENERATOR_VERSION,
            "mandatory_obs": [digest(o.to_dict()) for o in remaining_mandatory],
            "rep_mode": rep_mode,
            "resp_digest": resp_digest,
            "has_material_conflict": has_material_conflict,
        })
        cached_mand = cache.get_step(mandatory_cache_key) if cache is not None else None
        if cached_mand is not None:
            mandatory_cache_hits += 1
            mand_results = list(cached_mand.results)
            for mob in remaining_mandatory:
                mandatory_obligations_records.append(_obligation_record(mob, target_step_id))
                evaluated_mandatory_ids.add(mob.obligation_id)
        else:
            mandatory_cache_misses += 1
            post_ctx = VerifierContext(
                step=synthetic_step,
                conclusion=conc_node,
                premises=tuple(graph.claims[pid] for pid in synthetic_step.premises),
                domain_profile=profile,
                documents=documents,
                layer=layer,
                clause_relations=tuple(clause_relations),
                as_of=as_of,
                allow_semantic_call=False,
                question=question,
                candidate_response=candidate_response,
                correspondence_facts=correspondence_report,
            )
            mand_results = []
            for mob in remaining_mandatory:
                mandatory_obligations_records.append(_obligation_record(mob, target_step_id))
                evaluated_mandatory_ids.add(mob.obligation_id)
                mand_results.append(registry.verify_obligation(mob, post_ctx))
            if cache is not None:
                m_deps: set[str] = {conc_id} | set(conc_ancestors)
                for cid_m in list(m_deps):
                    if cid_m in graph.claims:
                        m_deps |= set(graph.claims[cid_m].doc_ids())
                m_eff_dig = digest({"key": mandatory_cache_key, "results": [r.to_dict() for r in mand_results]})
                cache.register_step(
                    mandatory_cache_key,
                    _CachedStepEntry(
                        cache_key=mandatory_cache_key,
                        effective_node_digest=m_eff_dig,
                        results=tuple(mand_results),
                        lineages=(),
                        source_deps=frozenset(m_deps),
                        refuted=any(r.status == "refuted" for r in mand_results),
                    ),
                )

        mob_by_id = {mob.obligation_id: mob for mob in remaining_mandatory}
        for m_res in mand_results:
            mob = mob_by_id.get(m_res.obligation_id)
            is_mob_crit = bool(mob.critical) if mob is not None else True
            is_conflict_ob = bool(mob is not None and mob.parameters.get("mandatory_check") == "conflict_disclosure")
            all_checks.append(m_res)
            all_assumptions.extend(m_res.assumptions)
            if is_mob_crit:
                critical_check_keys.add((m_res.step_id, m_res.obligation_id))
            if m_res.status == "refuted":
                node_refuted[conc_id] = True
                conc_derivations = []
            elif m_res.status == "unknown":
                pending_obligations.append({
                    "obligation_id": m_res.obligation_id,
                    "step_id": m_res.step_id,
                    "family": m_res.family,
                    "reasons": list(m_res.reasons),
                    "critical": is_mob_crit,
                    "origin": "engine_mandatory",
                })
                if is_mob_crit and m_res.step_id not in critical_unverified:
                    critical_unverified.append(m_res.step_id)
                if not is_conflict_ob:
                    conc_derivations = []

    corr_issues = tuple(correspondence_report.get("issues", ()))
    if candidate_response is not None:
        corr_step_id = "mandatory::response_correspondence"
        conc_extra_steps.add(corr_step_id)
        if corr_issues:
            has_refuted_corr_issue = False
            for idx, issue in enumerate(corr_issues):
                code = str(issue.get("code") or f"CORRESPONDENCE_ISSUE_{idx}")
                issue_status = str(issue.get("status") or "refuted")
                if issue_status not in ("refuted", "unknown"):
                    issue_status = "refuted"
                is_sem_pending = code in (
                    "RESPONSE_CORRESPONDENCE_PENDING_SEMANTIC",
                    "CONDITION_CORRESPONDENCE_UNRESOLVED",
                    "RELATIONAL_CORRESPONDENCE_UNRESOLVED",
                )
                ob_id = f"mandatory::correspondence::{code.lower()}::{idx}"
                chk = VerifierResult(
                    obligation_id=ob_id,
                    step_id=corr_step_id,
                    family="open_semantics" if (issue_status == "unknown" and is_sem_pending) else "explicit_logic",
                    status=issue_status,
                    guarantee_kind=(
                        "probabilistic_model"
                        if (issue_status == "unknown" and is_sem_pending)
                        else "deterministic_formal"
                    ),
                    evidence=dict(issue),
                    reasons=(code,),
                    verifier_id="response_correspondence_verifier",
                    verifier_version="1.0.0",
                )
                all_checks.append(chk)
                critical_check_keys.add((chk.step_id, chk.obligation_id))
                if issue_status == "unknown":
                    pending_obligations.append({
                        "obligation_id": chk.obligation_id,
                        "step_id": chk.step_id,
                        "family": chk.family,
                        "reasons": list(chk.reasons),
                        "critical": True,
                        "origin": "engine_mandatory",
                    })
                    if chk.step_id not in critical_unverified:
                        critical_unverified.append(chk.step_id)
                else:
                    has_refuted_corr_issue = True
            if has_refuted_corr_issue:
                node_refuted[conc_id] = True
            conc_derivations = []
        else:
            chk = VerifierResult(
                obligation_id="mandatory::response_correspondence",
                step_id=corr_step_id,
                family="explicit_logic",
                status="verified",
                guarantee_kind="deterministic_formal",
                evidence={"response_digest": resp_digest, "representation_mode": rep_mode},
                reasons=("RESPONSE_GRAPH_CORRESPONDENCE_VERIFIED",),
                verifier_id="response_correspondence_verifier",
                verifier_version="1.0.0",
            )
            all_checks.append(chk)
            critical_check_keys.add((chk.step_id, chk.obligation_id))

    raw_support, sup_lineages = _fuse_correlated_derivations(conc_derivations)
    raw_opposition, opp_lineages = _fuse_correlated_derivations(contradicting_derivations)

    # Check if any critical check to conclusion was formally refuted
    conc_scope_nodes = conc_ancestors | {conc_id}
    conc_steps = {s.step_id for s in graph.steps if s.conclusion_id in conc_scope_nodes} | conc_extra_steps
    has_formal_refutation = bool(
        node_refuted.get(conc_id, False)
        or any(
            c.status == "refuted" and (c.step_id in conc_steps or c.step_id == f"leaf::{conc_id}" or any(c.step_id == f"leaf::{a}" for a in conc_scope_nodes))
            for c in all_checks
        )
    )
    response_correspondence_verified = bool(
        candidate_response is not None
        and correspondence_report.get("correspondence_ok")
        and not corr_issues
        and not has_formal_refutation
    )

    # Compute verification coverage over all non-not_applicable checks and critical/mandatory subset
    applicable_checks = [c for c in all_checks if c.status != "not_applicable"]
    completed_determinate = [c for c in applicable_checks if c.status in ("verified", "refuted")]
    if applicable_checks:
        all_coverage = round(len(completed_determinate) / len(applicable_checks), 6)
    else:
        all_coverage = 1.0

    critical_applicable = [
        c for c in applicable_checks if (c.step_id, c.obligation_id) in critical_check_keys
    ]
    if critical_applicable:
        critical_determinate = [c for c in critical_applicable if c.status in ("verified", "refuted")]
        critical_coverage = round(len(critical_determinate) / len(critical_applicable), 6)
    else:
        critical_coverage = 1.0

    coverage = min(all_coverage, critical_coverage)

    # Compute orthodox orthogonal Support / Conflict / Uncertainty decomposition
    if has_formal_refutation:
        support_idx = 0.0
        conflict_idx = 1.0
        uncertainty_idx = 0.0
    elif raw_support <= 0.0 and raw_opposition <= 0.0:
        support_idx = 0.0
        conflict_idx = 0.0
        uncertainty_idx = 1.0
    else:
        total_mass = 1.0 - (1.0 - raw_support) * (1.0 - raw_opposition)
        b_full = total_mass * (raw_support / (raw_support + raw_opposition))
        d_full = total_mass * (raw_opposition / (raw_support + raw_opposition))
        u_evid = (1.0 - raw_support) * (1.0 - raw_opposition)

        # Partial verification coverage limits realized support and moves residual mass to uncertainty
        realized_b = b_full * coverage
        unverified_gap = b_full - realized_b
        support_idx = round(max(0.0, min(1.0, realized_b)), 6)
        conflict_idx = round(max(0.0, min(1.0, d_full)), 6)
        uncertainty_idx = round(max(0.0, min(1.0, u_evid + unverified_gap)), 6)

    # Classify overall graduated assessment label
    if has_formal_refutation:
        assessment_label = "refuted"
    elif conflict_idx >= profile.policy.max_conflict_index:
        assessment_label = "contested"
    elif (
        coverage == 1.0
        and not critical_unverified
        and not pending_obligations
        and support_idx >= profile.policy.tau
        and conflict_idx < profile.policy.max_conflict_index
        and conc_eff_kind in ("extraction", "derivation")
    ):
        assessment_label = "supported"
    elif support_idx > 0.0:
        assessment_label = "partially_supported"
    elif pending_obligations or critical_unverified:
        assessment_label = "inconclusive"
    else:
        assessment_label = "unsupported"

    # Supporting premise / evidence summary records
    supporting_records: list[dict[str, Any]] = []
    for pid in sorted(graph.ancestor_claims(conc_id) | {conc_id}):
        pnode = graph.claims[pid]
        if pnode.evidence or pnode.evidence_unit_ids:
            p_sup, _ = _fuse_correlated_derivations(node_derivations.get(pid, ()))
            if p_sup > 0.0:
                supporting_records.append({
                    "claim_id": pid,
                    "kind": node_effective_kind.get(pid, pnode.kind),
                    "subject": pnode.subject,
                    "predicate": pnode.predicate,
                    "value": pnode.value,
                    "unit": pnode.unit,
                    "doc_ids": list(pnode.doc_ids()),
                    "support": p_sup,
                })

    premise_records = [
        {
            "claim_id": pid,
            "kind": node_effective_kind.get(pid, graph.claims[pid].kind),
            "subject": graph.claims[pid].subject,
            "predicate": graph.claims[pid].predicate,
            "value": graph.claims[pid].value,
            "unit": graph.claims[pid].unit,
        }
        for pid in sorted(graph.ancestor_claims(conc_id))
    ]

    calibrated_prob = profile.calibration.calibrate_correctness(support_idx)
    dedup_assumptions = tuple(dict.fromkeys(all_assumptions))

    cache_metrics = {
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "recomputed_nodes": recomputed_nodes,
        "reused_nodes": reused_nodes,
        "budget_deferred_steps": deferred_steps,
        "mandatory_cache_hits": mandatory_cache_hits,
        "mandatory_cache_misses": mandatory_cache_misses,
    }

    summary_for_digest = {
        "conclusion_id": conc_id,
        "assessment": assessment_label,
        "effective_epistemic_kind": conc_eff_kind,
        "evidence_support_index": support_idx,
        "conflict_index": conflict_idx,
        "uncertainty_index": uncertainty_idx,
        "verification_coverage": coverage,
        "critical_coverage": critical_coverage,
        "critical_unverified_steps": critical_unverified,
        "profile_fp": profile_fp,
        "response_digest": resp_digest,
        "representation_mode": rep_mode,
        "obligation_generator_version": OBLIGATION_GENERATOR_VERSION,
        "mandatory_obligations": mandatory_obligations_records,
    }

    return ResponseAssessment(
        conclusion_id=conc_id,
        conclusion=conc_node,
        effective_epistemic_kind=conc_eff_kind,
        assessment=assessment_label,
        evidence_support_index=support_idx,
        conflict_index=conflict_idx,
        uncertainty_index=uncertainty_idx,
        verification_coverage=coverage,
        critical_coverage=critical_coverage,
        critical_unverified_steps=tuple(critical_unverified),
        supporting_evidence=tuple(supporting_records),
        contradicting_evidence=tuple(contradicting_records),
        inference_steps=tuple(step_summaries),
        premises=tuple(premise_records),
        completed_checks=tuple(all_checks),
        pending_obligations=tuple(pending_obligations),
        assumptions=dedup_assumptions,
        calibrated_correctness_probability=calibrated_prob,
        domain_profile_id=profile.profile_id,
        domain_profile_version=profile.version,
        domain_profile_fingerprint=profile_fp,
        assessment_digest=digest(summary_for_digest),
        cache_metrics=cache_metrics,
        supporting_lineage_count=sup_lineages,
        contradicting_lineage_count=opp_lineages,
        response_digest=resp_digest,
        question=q_str,
        candidate_response=candidate_response,
        representation_mode=rep_mode,
        response_correspondence_verified=response_correspondence_verified,
        obligation_generator_version=OBLIGATION_GENERATOR_VERSION,
        proposed_obligations=tuple(proposed_obligations_records),
        mandatory_obligations=tuple(mandatory_obligations_records),
        semantic_domain_info=semantic_domain_info,
        correspondence_report=correspondence_report,
    )
