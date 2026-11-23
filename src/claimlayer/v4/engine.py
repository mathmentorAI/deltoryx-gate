"""ClaimLayer v4 engine: the model proposes, deterministic policy decides.

Faithful to the Evidence Intelligence contract:
* a language model only *proposes* typed evidence units bound to exact spans;
* every unit must conform to an application vocabulary (no free-form keys);
* support, conflicts, authority, aggregation and derivations are computed by
  code over the ledger (dependence-aware Noisy-OR, Q = S^2/(S+O), margin rule);
* the declared epistemic table weighs how a unit was obtained and blocks bases such
  as narrative or abductive assumptions;
* no model ever issues a supported/contradicted verdict.

Results remain conditional on extraction fidelity, vocabulary design and the
application's authority metadata; nothing here certifies factual truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
import copy
import json
from pathlib import Path
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Sequence

from ..semantic.models import KINDS, EvidenceRef, SourceDocument, StructuredQuery, canonical, digest, instant
from ..semantic.store import VerificationStore
from .epistemics import DEFAULT_PROFILE, EXPLICIT, EpistemicProfile
from .fidelity import FieldEvidence, FidelityFinding, verify_extraction_fidelity
from .models import AcceptancePolicy, EvidenceUnit, context_key
from .vocabulary import NORMALIZER_VERSION, UnmappedProposal, Vocabulary, decimal_text, label_key, value_relation

ENGINE_VERSION = '4.0.0'
CONTRACT = 'claimlayer-evidence-v4'
TIER_PRIMARY, TIER_ANNEX, TIER_EXTERNAL = 100, 50, 10  # source tiers declared by the application


def _identity_rule_check(conclusion, premises) -> bool:
    return len(premises) == 1 and all(
        canonical({k: v for k, v in conclusion.items() if k not in ('valid_from', 'valid_until')})
        == canonical({k: v for k, v in p.items() if k not in ('valid_from', 'valid_until')})
        for p in premises
    )


DEFAULT_RULES: dict[str, tuple[str, Any]] = {'identity': ('1', _identity_rule_check)}


def _noisy_or(values):
    remaining = 1.0
    for value in values:
        remaining *= 1.0 - value
    return round(1.0 - remaining, 6)


def _q_index(support, opposition):
    return round(support ** 2 / (support + opposition), 6) if support + opposition > 0 else 0.0


@dataclass(frozen=True)
class UnitEvaluation:
    """Single deterministic unit admissibility & provenance evaluation shared across v4."""

    unit: EvidenceUnit
    admissible: bool
    reason: str | None = None
    issue_code: str | None = None
    issue_stage: str | None = None
    issue_disposition: str | None = None
    issue_message: str | None = None
    issue_detail: Any = None
    fidelity_findings: tuple[FidelityFinding, ...] = ()
    documents: dict[str, SourceDocument] = field(default_factory=dict)
    support: float | None = None
    components: dict[str, Any] = field(default_factory=dict)
    dependence: frozenset[str] = frozenset()
    rank: int = 0
    pending_conditions: list[dict[str, Any]] = field(default_factory=list)
    pending_scope: dict[str, Any] = field(default_factory=dict)
    premise_ids: tuple[str, ...] = ()
    normalizer_version: str = NORMALIZER_VERSION

    def as_basis(self) -> dict[str, Any]:
        return {
            'unit': self.unit,
            'documents': self.documents,
            'support': self.support,
            'components': self.components,
            'dependence': self.dependence,
            'rank': self.rank,
            'pending_conditions': self.pending_conditions,
            'pending_scope': self.pending_scope,
            'premise_bases': list(self.premise_ids),
        }


def evaluate_unit_admissibility(
    unit: EvidenceUnit,
    *,
    vocabulary: Vocabulary,
    policy: AcceptancePolicy,
    profile: EpistemicProfile = DEFAULT_PROFILE,
    units_by_id: Mapping[str, EvidenceUnit] | None = None,
    rules: Mapping[str, tuple[str, Any]] | None = None,
    documents: Mapping[str, SourceDocument] | None = None,
    document_revisions: Mapping[tuple[str, str], SourceDocument] | None = None,
    as_of: str | None = None,
    validity: str = 'filter',
    allowed_assertion_kinds: Iterable[str] | None = None,
    query_scope: Mapping[str, Any] | None = None,
    query_conditions: Mapping[str, Any] | None = None,
    condition_mode: str = 'report',
    field_evidences: Mapping[str, Sequence[FieldEvidence]] | None = None,
    stored_field_evidences: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    require_field_evidence: Sequence[str] = (),
    source_tiers: Mapping[str, int] | None = None,
    source_weights: Mapping[str, float] | None = None,
    check_evidence_refs: bool = True,
    skip_fidelity: bool = False,
    seen: frozenset[str] = frozenset(),
) -> UnitEvaluation:
    """Evaluate a single EvidenceUnit under one unified admissibility, provenance, fidelity & condition core."""
    if unit.unit_id in seen:
        return UnitEvaluation(unit=unit, admissible=False, reason='cyclic_dependency')

    p = unit.proposition
    admitted_kinds = tuple(allowed_assertion_kinds) if allowed_assertion_kinds is not None else policy.allowed_assertion_kinds
    if p['assertion_kind'] not in admitted_kinds:
        return UnitEvaluation(unit=unit, admissible=False, reason='assertion_kind_not_admitted')

    if validity == 'filter' and as_of is not None:
        at = instant(as_of)
        start, end = instant(p['valid_from']), instant(p['valid_until'])
        if (start and at < start) or (end and at > end):
            return UnitEvaluation(unit=unit, admissible=False, reason='outside_validity')

    if not profile.admissible(unit.basis):
        return UnitEvaluation(unit=unit, admissible=False, reason='basis_not_admissible_under_profile')

    pred_spec = vocabulary.predicate(p['predicate'])
    q_scope = query_scope or {}
    pending_scope: dict[str, Any] = {}
    for key, value in p.get('scope', {}).items():
        if key in {'event_id', 'clause_id'}:
            if key in q_scope:
                qval = q_scope[key]
                if not isinstance(qval, str) or label_key(qval) != label_key(value):
                    return UnitEvaluation(unit=unit, admissible=False, reason='scope_mismatch')
            continue
        if key in q_scope:
            qval = q_scope[key]
            if isinstance(qval, str):
                if label_key(qval) != label_key(value):
                    return UnitEvaluation(unit=unit, admissible=False, reason='scope_mismatch')
            elif canonical(qval) != canonical(value):
                return UnitEvaluation(unit=unit, admissible=False, reason='scope_mismatch')
        else:
            pending_scope[key] = value

    if condition_mode == 'discharge' and pending_scope:
        return UnitEvaluation(unit=unit, admissible=False, reason='scope_not_discharged', pending_scope=pending_scope)

    active_rules = rules if rules is not None else DEFAULT_RULES
    resolved_docs: dict[str, SourceDocument] = {}
    premise_ids: tuple[str, ...] = ()

    if unit.origin == 'rule_derived':
        installed = active_rules.get(unit.rule)
        if installed is None or installed[0] != unit.rule_version:
            return UnitEvaluation(
                unit=unit,
                admissible=False,
                reason='rule_version_unavailable',
                issue_code='RULE_VERSION_UNAVAILABLE',
                issue_stage='epistemic_integrity',
                issue_disposition='RETRY',
                issue_message=f"Rule '{unit.rule}' version '{unit.rule_version}' is not available.",
                issue_detail={'unit_id': unit.unit_id, 'rule': unit.rule, 'version': unit.rule_version},
                pending_scope=pending_scope,
            )
        premise_evals: list[UnitEvaluation] = []
        for pid in unit.premises:
            premise = units_by_id.get(pid) if units_by_id is not None else None
            if premise is None:
                return UnitEvaluation(
                    unit=unit,
                    admissible=False,
                    reason='premise_missing',
                    issue_code='MISSING_PREMISE_UNIT',
                    issue_stage='epistemic_integrity',
                    issue_disposition='RETRY',
                    issue_message=f"Derived unit {unit.unit_id} references missing premise '{pid}'.",
                    issue_detail={'unit_id': unit.unit_id, 'premise_id': pid},
                    pending_scope=pending_scope,
                )
            p_eval = evaluate_unit_admissibility(
                premise,
                vocabulary=vocabulary,
                policy=policy,
                profile=profile,
                units_by_id=units_by_id,
                rules=active_rules,
                documents=documents,
                document_revisions=document_revisions,
                as_of=as_of,
                validity=validity,
                allowed_assertion_kinds=admitted_kinds,
                query_scope=query_scope,
                query_conditions=query_conditions,
                condition_mode=condition_mode,
                field_evidences=field_evidences,
                stored_field_evidences=stored_field_evidences,
                require_field_evidence=require_field_evidence,
                source_tiers=source_tiers,
                source_weights=source_weights,
                check_evidence_refs=check_evidence_refs,
                skip_fidelity=skip_fidelity,
                seen=seen | {unit.unit_id},
            )
            if not p_eval.admissible:
                return UnitEvaluation(
                    unit=unit,
                    admissible=False,
                    reason='premise_' + (p_eval.reason or 'ineligible'),
                    issue_code=p_eval.issue_code or 'PREMISE_INADMISSIBLE',
                    issue_stage=p_eval.issue_stage or 'epistemic_integrity',
                    issue_disposition=p_eval.issue_disposition or 'RETRY',
                    issue_message=p_eval.issue_message or f"Premise {pid} inadmissible: {p_eval.reason}",
                    issue_detail={'unit_id': unit.unit_id, 'premise_id': pid, 'premise_reason': p_eval.reason, 'detail': p_eval.issue_detail},
                    fidelity_findings=p_eval.fidelity_findings,
                    pending_scope=pending_scope,
                )
            if p_eval.support is None:
                return UnitEvaluation(
                    unit=unit,
                    admissible=False,
                    reason='premise_unassessed',
                    issue_code='UNASSESSED_READER_RELIABILITY',
                    issue_stage='epistemic_integrity',
                    issue_disposition='ESCALATE',
                    issue_message=f"Premise {pid} of derived unit {unit.unit_id} has unassessed structural_reliability=None.",
                    issue_detail={'unit_id': unit.unit_id, 'premise_id': pid},
                    pending_scope=pending_scope,
                )
            premise_evals.append(p_eval)
        try:
            rule_valid = installed[1](p, tuple(pe.unit.proposition for pe in premise_evals)) is True
        except Exception:
            rule_valid = False
        if not rule_valid:
            return UnitEvaluation(
                unit=unit,
                admissible=False,
                reason='invalid_derivation',
                issue_code='INVALID_DERIVATION',
                issue_stage='epistemic_integrity',
                issue_disposition='RETRY',
                issue_message=f"Derived unit {unit.unit_id} fails rule '{unit.rule}' check.",
                issue_detail={'unit_id': unit.unit_id, 'rule': unit.rule},
                pending_scope=pending_scope,
            )
        weakest = min(pe.support for pe in premise_evals if pe.support is not None)
        support = round((policy.path_discount ** unit.rule_applications) * weakest, 6)
        components = {
            'weakest_premise': weakest,
            'rule_applications': unit.rule_applications,
            'path_discount': policy.path_discount,
        }
        for pe in premise_evals:
            resolved_docs.update(pe.documents)
        dependence = frozenset(d for pe in premise_evals for d in pe.dependence) or frozenset({f'unit:{unit.unit_id}'})
        rank = min((pe.rank for pe in premise_evals), default=0)
        premise_ids = tuple(pe.unit.unit_id for pe in premise_evals)
    else:
        if not unit.evidence:
            return UnitEvaluation(
                unit=unit,
                admissible=False,
                reason='missing_evidence',
                issue_code='MISSING_EVIDENCE_DOCUMENT',
                issue_stage='epistemic_integrity',
                issue_disposition='RETRY',
                issue_message=f"Unit {unit.unit_id} has no evidence references.",
                issue_detail={'unit_id': unit.unit_id},
                pending_scope=pending_scope,
            )
        if documents is not None:
            for ref in unit.evidence:
                if document_revisions is not None and (ref.document_id, ref.revision) not in document_revisions and ref.document_id not in documents:
                    return UnitEvaluation(
                        unit=unit,
                        admissible=False,
                        reason='stale_evidence',
                        issue_code='MISSING_EVIDENCE_DOCUMENT',
                        issue_stage='epistemic_integrity',
                        issue_disposition='RETRY',
                        issue_message=f"Evidence document '{ref.document_id}' revision '{ref.revision}' unavailable.",
                        issue_detail={'unit_id': unit.unit_id, 'document_id': ref.document_id, 'revision': ref.revision},
                        pending_scope=pending_scope,
                    )
                doc = documents.get(ref.document_id)
                if doc is None:
                    return UnitEvaluation(
                        unit=unit,
                        admissible=False,
                        reason='stale_evidence',
                        issue_code='MISSING_EVIDENCE_DOCUMENT',
                        issue_stage='epistemic_integrity',
                        issue_disposition='RETRY',
                        issue_message=f"Evidence document '{ref.document_id}' unavailable.",
                        issue_detail={'unit_id': unit.unit_id, 'document_id': ref.document_id, 'revision': ref.revision},
                        pending_scope=pending_scope,
                    )
                if doc.revision != ref.revision:
                    return UnitEvaluation(
                        unit=unit,
                        admissible=False,
                        reason='stale_evidence',
                        issue_code='STALE_EVIDENCE_REVISION',
                        issue_stage='epistemic_integrity',
                        issue_disposition='RETRY',
                        issue_message=f"Unit {unit.unit_id} cites stale revision '{ref.revision}' of '{ref.document_id}'.",
                        issue_detail={'unit_id': unit.unit_id, 'document_id': ref.document_id, 'cited_revision': ref.revision, 'current_revision': doc.revision},
                        pending_scope=pending_scope,
                    )
                if check_evidence_refs and (ref.start < 0 or ref.end > len(doc.text) or doc.text[ref.start:ref.end] != ref.quote):
                    return UnitEvaluation(
                        unit=unit,
                        admissible=False,
                        reason='stale_evidence',
                        issue_code='QUOTE_SPAN_MISMATCH',
                        issue_stage='epistemic_integrity',
                        issue_disposition='RETRY',
                        issue_message='Evidence quote does not match stored document slice.',
                        issue_detail={'unit_id': unit.unit_id, 'document_id': ref.document_id},
                        pending_scope=pending_scope,
                    )
                resolved_docs[ref.document_id] = doc

            if any(d.attribution not in policy.allowed_source_attributions for d in resolved_docs.values()):
                return UnitEvaluation(unit=unit, admissible=False, reason='source_attribution_not_admitted', documents=resolved_docs, pending_scope=pending_scope)

            if not skip_fidelity:
                fe_by_field: dict[str, FieldEvidence] = {}
                raw_stored_fe = stored_field_evidences.get(unit.unit_id) if stored_field_evidences else None
                if raw_stored_fe:
                    for fname, ref_dict in raw_stored_fe.items():
                        try:
                            fe_by_field[fname] = FieldEvidence(fname, EvidenceRef(**ref_dict))
                        except (TypeError, ValueError) as exc:
                            return UnitEvaluation(
                                unit=unit,
                                admissible=False,
                                reason='fidelity:INVALID_FIELD_EVIDENCE',
                                issue_code='INVALID_FIELD_EVIDENCE',
                                issue_stage='epistemic_integrity',
                                issue_disposition='RETRY',
                                issue_message=f"Invalid stored FieldEvidence for '{fname}': {exc}",
                                issue_detail={'unit_id': unit.unit_id, 'field': fname},
                                documents=resolved_docs,
                                pending_scope=pending_scope,
                            )
                if field_evidences and unit.unit_id in field_evidences:
                    for caller_fe in field_evidences[unit.unit_id]:
                        fe_by_field[caller_fe.field] = caller_fe
                merged_fe = list(fe_by_field.values()) if fe_by_field else None
                primary_doc = resolved_docs[unit.evidence[0].document_id]
                f_findings = verify_extraction_fidelity(
                    primary_doc.text,
                    unit,
                    field_evidence=merged_fe,
                    require_field_evidence=require_field_evidence,
                    vocabulary=vocabulary,
                    documents=resolved_docs,
                    table_context=(
                        unit.proposal.get('table_context')
                        if isinstance(unit.proposal, Mapping)
                        else None
                    ),
                )
                if f_findings:
                    return UnitEvaluation(
                        unit=unit,
                        admissible=False,
                        reason=f'fidelity:{f_findings[0].code}',
                        issue_code=f_findings[0].code,
                        issue_stage='epistemic_integrity',
                        issue_disposition='RETRY',
                        issue_message=f_findings[0].message,
                        issue_detail={'unit_id': unit.unit_id, 'field': f_findings[0].field, 'detail': f_findings[0].detail},
                        fidelity_findings=tuple(f_findings),
                        documents=resolved_docs,
                        pending_scope=pending_scope,
                    )

        if resolved_docs:
            doc_w = min(
                source_weights[d.document_id] if (source_weights and d.document_id in source_weights) else d.weight
                for d in resolved_docs.values()
            )
            rank = min(
                source_tiers[d.document_id] if (source_tiers and d.document_id in source_tiers) else d.authority_rank
                for d in resolved_docs.values()
            )
            dependence = frozenset(d.dependence_group or ('document:' + d.document_id) for d in resolved_docs.values())
        else:
            doc_w = min(
                (source_weights[r.document_id] if (source_weights and r.document_id in source_weights) else 1.0)
                for r in unit.evidence
            )
            rank = min(
                (source_tiers[r.document_id] if (source_tiers and r.document_id in source_tiers) else 0)
                for r in unit.evidence
            )
            dependence = frozenset('document:' + r.document_id for r in unit.evidence)

        components = {
            'span': 1.0,
            'structure': unit.structural_reliability,
            'applicability': doc_w,
            'basis': profile.weight(unit.basis),
        }
        support = None if any(v is None for v in components.values()) else round(min(components.values()), 6)

    cstatus = p.get('condition_status')
    if cstatus == 'unknown_condition':
        return UnitEvaluation(
            unit=unit,
            admissible=False,
            reason='unknown_condition',
            issue_code='UNKNOWN_CONDITION',
            issue_stage='epistemic_integrity',
            issue_disposition='ESCALATE',
            issue_message=f"Unit {unit.unit_id} has 'unknown_condition' for {p['entity']}.{p['predicate']}; cannot authorize.",
            issue_detail={'unit_id': unit.unit_id},
            documents=resolved_docs,
            pending_scope=pending_scope,
        )
    if cstatus == 'unrepresentable_condition':
        return UnitEvaluation(
            unit=unit,
            admissible=False,
            reason='unrepresentable_condition',
            issue_code='UNREPRESENTABLE_CONDITION',
            issue_stage='epistemic_integrity',
            issue_disposition='ESCALATE',
            issue_message=f"Unit {unit.unit_id} has 'unrepresentable_condition' for {p['entity']}.{p['predicate']}; cannot authorize.",
            issue_detail={'unit_id': unit.unit_id},
            documents=resolved_docs,
            pending_scope=pending_scope,
        )

    pending_conditions: list[dict[str, Any]] = []
    outcomes = [(cond, vocabulary.evaluate_condition(pred_spec, cond, query_conditions)) for cond in p.get('conditions', ())]
    if any(out == 'invalid' for _, out in outcomes):
        return UnitEvaluation(
            unit=unit,
            admissible=False,
            reason='condition_invalid',
            issue_code='INVALID_CONDITION_VALUE',
            issue_stage='epistemic_integrity',
            issue_disposition='ESCALATE',
            issue_message=f"Condition for {p['entity']}.{p['predicate']} received an invalid or incommensurable runtime value.",
            issue_detail={'unit_id': unit.unit_id, 'conditions': list(p.get('conditions', ()))},
            documents=resolved_docs,
            pending_scope=pending_scope,
        )
    if any(out == 'false' for _, out in outcomes):
        return UnitEvaluation(
            unit=unit,
            admissible=False,
            reason='condition_not_met',
            documents=resolved_docs,
            pending_scope=pending_scope,
        )
    pending_conditions = [cond for cond, out in outcomes if out == 'pending']
    if condition_mode == 'discharge' and pending_conditions:
        return UnitEvaluation(
            unit=unit,
            admissible=False,
            reason='condition_not_discharged',
            issue_code='PENDING_CONDITION',
            issue_stage='epistemic_integrity',
            issue_disposition='ESCALATE',
            issue_message=f"Condition for {p['entity']}.{p['predicate']} cannot be evaluated because operational context is missing.",
            issue_detail={'unit_id': unit.unit_id, 'conditions': pending_conditions},
            documents=resolved_docs,
            pending_conditions=pending_conditions,
            pending_scope=pending_scope,
        )

    return UnitEvaluation(
        unit=unit,
        admissible=True,
        documents=resolved_docs,
        support=support,
        components=components,
        dependence=dependence,
        rank=rank,
        pending_conditions=pending_conditions,
        pending_scope=pending_scope,
        premise_ids=premise_ids,
    )


class EvidenceLayer:
    """Append-only evidence ledger with deterministic, versioned views."""

    def __init__(self, db_path=None, *, vocabulary: Vocabulary, project_id='default', reader=None,
                 profile: EpistemicProfile = DEFAULT_PROFILE, policy: AcceptancePolicy | None = None,
                 rules: Mapping[str, tuple[str, Any]] | None = None, transitive_rules: Iterable = ()):
        if not isinstance(vocabulary, Vocabulary):
            raise TypeError('v4 requires an explicit application Vocabulary')
        self._tmpdir = tempfile.TemporaryDirectory() if db_path is None else None
        path = Path(self._tmpdir.name) / 'evidence_v4.db' if db_path is None else Path(db_path)
        self.store = VerificationStore(path, project_id)
        self.vocabulary, self.reader, self.profile = vocabulary, reader, profile
        self.policy = policy or AcceptancePolicy()
        self._rules = {'identity': ('1', lambda conclusion, premises: len(premises) == 1 and all(
            canonical({k: v for k, v in conclusion.items() if k not in ('valid_from', 'valid_until')})
            == canonical({k: v for k, v in p.items() if k not in ('valid_from', 'valid_until')}) for p in premises))}
        for name, specification in (rules or {}).items():
            version, checker = specification
            if name == 'identity' or not isinstance(version, str) or not version or not callable(checker):
                raise ValueError('Rules need a new name, an explicit version and a trusted callable')
            self._rules[name] = (version, checker)
        self._transitive = {}
        for rule in transitive_rules:
            from .paths import TransitiveRule
            if not isinstance(rule, TransitiveRule):
                raise TypeError('Use TransitiveRule declarations')
            spec = self.vocabulary.predicate(rule.predicate)
            if spec.value.type != 'entity':
                raise ValueError('Transitive predicates must have entity values')
            self._transitive[rule.predicate] = rule
            self._rules[rule.rule_name] = (rule.version, rule.check)

    # ------------------------------------------------------------ configuration
    @property
    def configuration(self):
        return {'engine_version': ENGINE_VERSION, 'contract': CONTRACT,
                'normalizer_version': NORMALIZER_VERSION,
                'vocabulary': {'name': self.vocabulary.name, 'version': self.vocabulary.version,
                               'fingerprint': self.vocabulary.fingerprint},
                'profile': {'name': self.profile.name, 'version': self.profile.version,
                            'fingerprint': self.profile.fingerprint},
                'policy': asdict(self.policy),
                'rules': {name: version for name, (version, _) in sorted(self._rules.items())}}

    # ---------------------------------------------------------------- documents
    def _document(self, document_id, revision=None, *, current=True):
        active, _ = self.store.head('document', document_id)
        revision = revision or active
        record = self.store.get('document', revision) if revision else None
        if record is None or record['document_id'] != document_id or (current and active != revision):
            raise ValueError('Missing or stale source revision')
        document = SourceDocument(**record)
        if document.revision != revision:
            raise ValueError('Source identity mismatch')
        return document

    def _refs_valid(self, refs, *, docs_cache=None):
        if not refs:
            raise ValueError('Source units need exact evidence references')
        for ref in refs:
            cache_key = (ref.document_id, ref.revision)
            document = docs_cache.get(cache_key) if docs_cache is not None else None
            if document is None:
                document = self._document(ref.document_id, ref.revision)
                if docs_cache is not None:
                    docs_cache[cache_key] = document
            if ref.end > len(document.text) or document.text[ref.start:ref.end] != ref.quote:
                raise ValueError('Quote and offsets do not match the stored source')

    def _event(self, payload):
        event_id = uuid.uuid4().hex
        self.store.put('event', event_id, {**json.loads(canonical(payload)), 'event_id': event_id,
                                           'created_at': datetime.now(timezone.utc).isoformat()})
        return event_id

    def _record_doc_read_status(self, document_id: str, revision: str, *, unread: bool,
                                errors: list[dict[str, Any]]) -> None:
        payload = {'document_id': document_id, 'revision': revision,
                   'unread': bool(unread), 'errors': list(errors)}
        self.store.put('doc_read_status', digest(payload), payload, head=document_id)

    def _store_field_evidence(self, unit_id: str, raw_fe: Any) -> None:
        normalized: dict[str, dict[str, Any]] = {}
        if isinstance(raw_fe, Mapping):
            for fname, ref in raw_fe.items():
                ref_dict = asdict(ref) if isinstance(ref, EvidenceRef) else dict(ref)
                normalized[str(fname)] = ref_dict
        elif isinstance(raw_fe, (list, tuple)):
            for entry in raw_fe:
                if hasattr(entry, 'field') and hasattr(entry, 'evidence'):
                    normalized[str(entry.field)] = asdict(entry.evidence)
                elif isinstance(entry, Mapping) and 'field' in entry and 'evidence' in entry:
                    ref = entry['evidence']
                    normalized[str(entry['field'])] = asdict(ref) if isinstance(ref, EvidenceRef) else dict(ref)
        if normalized:
            clean = json.loads(canonical(normalized))
            self.store.put('field_evidence', digest({'unit_id': unit_id, 'fe': clean}), clean, head=unit_id)

    @property
    def documents(self) -> list[SourceDocument]:
        return [SourceDocument(**r) for _, _, r in self.store.heads('document')]

    @property
    def unread_documents(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, _, st in self.store.heads('doc_read_status') if st and st.get('unread')))

    @property
    def read_errors(self) -> tuple[dict[str, Any], ...]:
        out: list[dict[str, Any]] = []
        for _, _, st in self.store.heads('doc_read_status'):
            if st and st.get('errors'):
                out.extend(st['errors'])
        return tuple(out)

    @property
    def document_read_statuses(self) -> dict[str, dict[str, Any]]:
        """Per-document read status: 'read_ok', 'read_failed' (with errors), or 'not_read'."""
        status_by_id = {name: st for name, _, st in self.store.heads('doc_read_status') if st}
        out: dict[str, dict[str, Any]] = {}
        for doc in self.documents:
            st = status_by_id.get(doc.document_id)
            errs = list(st.get('errors') or ()) if st else []
            unread = True if st is None else bool(st.get('unread'))
            if errs:
                state = 'read_failed'
            elif unread:
                state = 'not_read'
            else:
                state = 'read_ok'
            out[doc.document_id] = {
                'document_id': doc.document_id,
                'revision': doc.revision,
                'authority_rank': int(doc.authority_rank),
                'status': state,
                'errors': errs,
            }
        return out

    @property
    def field_evidences(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {uid: rec for uid, _, rec in self.store.heads('field_evidence') if rec}

    def snapshot(self) -> dict[str, Any]:
        """Capture a consistent single-transaction snapshot of documents, units, and read state."""
        raw = self.store.snapshot()
        heads = raw['heads']
        docs = {doc_id: SourceDocument(**rec) for doc_id, _, rec in heads.get('document', [])}
        doc_revs = {
            (rec['document_id'], rev): SourceDocument(**rec)
            for rev, rec in raw['document_revisions'].items()
        }
        units = [EvidenceUnit.from_record(rec) for _, _, rec in heads.get('unit', [])]
        read_status_heads = heads.get('doc_read_status', [])
        unread = tuple(sorted(name for name, _, st in read_status_heads if st and st.get('unread')))
        read_errors: list[dict[str, Any]] = []
        for _, _, st in read_status_heads:
            if st and st.get('errors'):
                read_errors.extend(st['errors'])
        field_evidences = {uid: rec for uid, _, rec in heads.get('field_evidence', []) if rec}
        rejections = [{'rejection_id': rid, **rec} for rid, rec in raw['rejections']]
        ledger_state_digest = digest({
            'documents': {doc_id: doc.revision for doc_id, doc in sorted(docs.items())},
            'document_revisions': sorted(raw['document_revisions'].keys()),
            'units': sorted(u.unit_id for u in units),
            'unread_documents': list(unread),
            'read_errors': read_errors,
            'field_evidences': {uid: digest(rec) for uid, rec in sorted(field_evidences.items())},
            'rejections': [r['rejection_id'] for r in rejections],
        })
        return {
            'configuration': self.configuration,
            'documents': docs,
            'document_revisions': doc_revs,
            'units': units,
            'unread_documents': unread,
            'read_errors': tuple(read_errors),
            'field_evidences': field_evidences,
            'rejections': rejections,
            'ledger_state_digest': ledger_state_digest,
        }

    def ingest(self, documents):
        """Store sources; if a reader is configured, record its proposals and admit conforming units."""
        if isinstance(documents, SourceDocument):
            documents = [documents]
        else:
            documents = list(documents)
        for document in documents:
            if not isinstance(document, SourceDocument):
                raise TypeError('Use SourceDocument; authority, weight and dependence are application metadata')

        # Run external/uncached reader proposals outside SQLite write lock, then persist the whole batch atomically.
        prepared = []
        local_read_cache: dict[str, dict[str, Any]] = {}
        for document in documents:
            if self.reader is None:
                prepared.append((document, None, None, None, None))
                continue
            request = {'task': 'propose', 'contract': CONTRACT, 'reader': self.reader.identity,
                       'document': {'document_id': document.document_id, 'revision': document.revision,
                                    'text': document.text},
                       'vocabulary': self.vocabulary.describe(), 'profile': self.profile.describe()}
            key = digest(request)
            cached = local_read_cache.get(key)
            already_stored = cached is not None
            if cached is None:
                _, cached = self.store.head('read', key)
                already_stored = cached is not None
            if cached is not None:
                prepared.append((document, key, cached, None, already_stored))
                continue
            try:
                result = self.reader.propose(copy.deepcopy(request))
                canonical(result)
            except Exception as exc:  # recorded, never converted into evidence
                prepared.append((document, key, None, type(exc).__name__, getattr(exc, 'diagnostics', None)))
            else:
                cached = {'request_key': key, 'result': json.loads(canonical(result))}
                local_read_cache[key] = cached
                prepared.append((document, key, cached, None, False))

        summary = {'documents_ingested': 0, 'units_admitted': 0, 'proposals_rejected': 0,
                   'rejection_reasons': {}, 'unread_documents': [], 'read_errors': []}
        with self.store.transaction(mode='IMMEDIATE'):
            for document, key, cached, error_type, diag_or_flag in prepared:
                self.store.put('document', document.revision, asdict(document), head=document.document_id)
                summary['documents_ingested'] += 1
                if self.reader is None:
                    summary['unread_documents'].append(document.document_id)
                    self._record_doc_read_status(document.document_id, document.revision, unread=True, errors=[])
                    continue
                if error_type is not None:
                    self._event({'task': 'propose', 'request_key': key, 'error_type': error_type,
                                 'diagnostics': diag_or_flag})
                    err = {'document_id': document.document_id, 'error_type': error_type}
                    summary['read_errors'].append(err)
                    summary['unread_documents'].append(document.document_id)
                    self._record_doc_read_status(document.document_id, document.revision, unread=True, errors=[err])
                    continue
                if not diag_or_flag:
                    self.store.put('read', key, cached, head=key)
                result = cached['result']
                producer = {'kind': 'reader', 'identity': self.reader.identity, 'read_key': key}
                doc_errors = [{'document_id': document.document_id, **e} for e in result.get('errors', ())]
                if doc_errors:
                    summary['read_errors'].extend(doc_errors)
                    summary['unread_documents'].append(document.document_id)
                self._record_doc_read_status(document.document_id, document.revision,
                                             unread=bool(doc_errors), errors=doc_errors)
                for rejected in result.get('rejected', []):
                    self._reject(rejected.get('proposal'), rejected.get('basis'), producer,
                                 {'reason': rejected.get('reason', 'reader_contract'), 'path': '', 'detail': rejected.get('detail')},
                                 summary)
                for item in result.get('items', []):
                    uid = self._admit(item.get('proposal'), item.get('basis'), item.get('evidence'), 'reader', producer,
                                      getattr(self.reader, 'structural_reliability', None), summary)
                    if uid is not None and item.get('field_evidence'):
                        self._store_field_evidence(uid, item['field_evidence'])
        return summary

    def _split_proposal_table_context(self, proposal, extra_table_context=None):
        if type(proposal) is not dict:
            raise UnmappedProposal('proposal_object_required', '', proposal)
        prop_clean = dict(proposal)
        raw_tc = prop_clean.pop('table_context', None)
        if extra_table_context is not None:
            raw_tc = extra_table_context
        stored_proposal = json.loads(canonical(prop_clean))
        if raw_tc is not None:
            if not isinstance(raw_tc, Mapping) or not raw_tc:
                raise UnmappedProposal('invalid_table_context', 'table_context', raw_tc)
            stored_proposal['table_context'] = json.loads(canonical(dict(raw_tc)))
        return prop_clean, stored_proposal

    def ingest_units(self, items, *, attestor: str, structural_reliability: float | None = 1.0):
        """Admit application-supplied typed proposals bound to exact spans (no model involved)."""
        if not isinstance(attestor, str) or not attestor.strip():
            raise ValueError('An application attestor is required')
        summary = {'units_admitted': 0, 'proposals_rejected': 0, 'rejection_reasons': {}, 'unit_ids': []}
        producer = {'kind': 'application', 'attestor': attestor}
        docs_cache = {}
        with self.store.transaction(mode='IMMEDIATE'):
            for item in items:
                evidence = [asdict(r) if isinstance(r, EvidenceRef) else r for r in item.get('evidence', ())]
                unit_id = self._admit(
                    item.get('proposal'),
                    item.get('basis', EXPLICIT),
                    evidence,
                    'application',
                    producer,
                    structural_reliability,
                    summary,
                    table_context=item.get('table_context'),
                    docs_cache=docs_cache,
                )
                if unit_id is not None:
                    if item.get('field_evidence'):
                        self._store_field_evidence(unit_id, item['field_evidence'])
                    for ev in evidence:
                        doc_id = ev.get('document_id') if isinstance(ev, Mapping) else None
                        if doc_id:
                            _, st = self.store.head('doc_read_status', doc_id)
                            if st and st.get('unread') and not st.get('errors'):
                                self._record_doc_read_status(doc_id, st['revision'], unread=False, errors=[])
                summary['unit_ids'].append(unit_id)
        return summary

    def admit_reread_items(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        producer: Mapping[str, Any],
        structural_reliability: float | None = None,
    ) -> dict[str, Any]:
        """Validate and admit targeted re-read items under full schema, evidence, and fidelity checks."""
        summary: dict[str, Any] = {
            'units_admitted': 0,
            'proposals_rejected': 0,
            'rejection_reasons': {},
            'unit_ids': [],
        }
        docs_map = {d.document_id: d for d in self.documents}
        with self.store.transaction(mode='IMMEDIATE'):
            for item in items:
                proposal = item.get('proposal')
                basis = item.get('basis', EXPLICIT)
                evidence = [asdict(r) if isinstance(r, EvidenceRef) else r for r in item.get('evidence', ())]
                try:
                    prop_clean, stored_proposal = self._split_proposal_table_context(
                        proposal, item.get('table_context')
                    )
                    proposition = self.vocabulary.proposition(prop_clean)
                    if not self.profile.admissible(basis):
                        raise UnmappedProposal(
                            'basis_blocked' if basis in self.profile.blocked else 'unknown_basis',
                            'basis',
                            basis,
                        )
                    refs = tuple(EvidenceRef(**r) for r in (evidence or ()))
                    self._refs_valid(refs)
                    unit = EvidenceUnit(
                        proposition,
                        refs,
                        basis,
                        'reader',
                        dict(producer),
                        structural_reliability,
                        proposal=stored_proposal,
                    )
                    fe_list: list[FieldEvidence] = []
                    raw_fe = item.get('field_evidence')
                    if isinstance(raw_fe, Mapping):
                        for fname, ref in raw_fe.items():
                            r_obj = ref if isinstance(ref, EvidenceRef) else EvidenceRef(**ref)
                            fe_list.append(FieldEvidence(str(fname), r_obj))
                except UnmappedProposal as exc:
                    self._reject(proposal, basis, dict(producer), exc.as_dict(), summary)
                    continue
                except (ValueError, TypeError, KeyError) as exc:
                    self._reject(
                        proposal,
                        basis,
                        dict(producer),
                        {'reason': 'evidence_binding', 'path': 'evidence', 'detail': str(exc)},
                        summary,
                    )
                    continue

                ev_res = evaluate_unit_admissibility(
                    unit,
                    vocabulary=self.vocabulary,
                    policy=self.policy,
                    profile=self.profile,
                    documents=docs_map,
                    field_evidences={unit.unit_id: fe_list} if fe_list else None,
                    condition_mode='report',
                    skip_fidelity=False,
                )
                if not ev_res.admissible:
                    self._reject(
                        proposal,
                        basis,
                        dict(producer),
                        {
                            'reason': ev_res.reason or 'reread_inadmissible',
                            'issue_code': ev_res.issue_code,
                            'message': ev_res.issue_message,
                        },
                        summary,
                    )
                    continue

                self._put_unit_if_new(unit, summary)
                if fe_list:
                    self._store_field_evidence(unit.unit_id, {fe.field: fe.evidence for fe in fe_list})
                summary['unit_ids'].append(unit.unit_id)
        return summary

    def _put_unit_if_new(self, unit: EvidenceUnit, summary: dict[str, Any]) -> str:
        """Persist a unit once per semantic identity without failing on differing raw proposal dicts."""
        rec = unit.record()
        _, existing = self.store.head('unit', unit.unit_id)
        if existing is not None:
            ex_id = {k: v for k, v in existing.items() if k != 'proposal'}
            new_id = {k: v for k, v in rec.items() if k != 'proposal'}
            if canonical(ex_id) == canonical(new_id):
                return unit.unit_id
        self.store.put('unit', unit.unit_id, rec, head=unit.unit_id)
        summary['units_admitted'] += 1
        return unit.unit_id

    def _reject(self, proposal, basis, producer, error, summary):
        hint = {}
        if isinstance(proposal, dict):
            for key in ('entity', 'predicate'):
                try:
                    hint[key] = (self.vocabulary.entity(proposal.get(key)) if key == 'entity'
                                 else self.vocabulary.predicate(proposal.get(key)).name)
                except UnmappedProposal:
                    hint[key] = proposal.get(key) if isinstance(proposal.get(key), str) else None
        record = {'proposal': proposal, 'basis': basis, 'producer': producer, 'error': error, 'hint': hint,
                  'vocabulary_fingerprint': self.vocabulary.fingerprint}
        record = json.loads(canonical(record))
        self.store.put('rejection', 'rj-' + digest(record), record)
        summary['proposals_rejected'] += 1
        summary['rejection_reasons'][error['reason']] = summary['rejection_reasons'].get(error['reason'], 0) + 1
        return None

    def _admit(self, proposal, basis, evidence, origin, producer, structural_reliability, summary, *,
               table_context=None, docs_cache=None):
        try:
            prop_clean, stored_proposal = self._split_proposal_table_context(proposal, table_context)
            proposition = self.vocabulary.proposition(prop_clean)
            if not self.profile.admissible(basis):
                raise UnmappedProposal('basis_blocked' if basis in self.profile.blocked else 'unknown_basis', 'basis', basis)
            refs = tuple(EvidenceRef(**r) for r in (evidence or ()))
            self._refs_valid(refs, docs_cache=docs_cache)
            unit = EvidenceUnit(proposition, refs, basis, origin, producer, structural_reliability,
                                proposal=stored_proposal)
        except UnmappedProposal as exc:
            return self._reject(proposal, basis, producer, exc.as_dict(), summary)
        except (ValueError, TypeError, KeyError) as exc:
            return self._reject(proposal, basis, producer, {'reason': 'evidence_binding', 'path': 'evidence',
                                                            'detail': str(exc)}, summary)
        return self._put_unit_if_new(unit, summary)

    # ------------------------------------------------------------------ ledger
    @property
    def units(self):
        return [EvidenceUnit.from_record(r) for _, _, r in self.store.heads('unit')]

    def unit(self, unit_id):
        _, record = self.store.head('unit', unit_id)
        if record is None:
            raise KeyError(unit_id)
        return EvidenceUnit.from_record(record)

    def rejections(self, *, entity=None, predicate=None):
        out = []
        for rid, record in self.store.by_kind('rejection'):
            if entity is not None and record['hint'].get('entity') != entity:
                continue
            if predicate is not None and record['hint'].get('predicate') != predicate:
                continue
            out.append({'rejection_id': rid, **record})
        return out

    # -------------------------------------------------------------- derivations
    def derive(self, proposal, premises: Iterable[str], rule: str):
        """Register a conclusion justified by an installed, versioned rule over existing units."""
        if rule not in self._rules:
            raise ValueError('Unknown rule: install a trusted, versioned checker first')
        premise_ids = tuple(premises)
        if not premise_ids:
            raise ValueError('Derivations need premises; empty-premise axioms are not supported')
        with self.store.transaction(mode='IMMEDIATE'):
            units = [self.unit(pid) for pid in premise_ids]
            proposition = self.vocabulary.proposition(proposal)
            version, checker = self._rules[rule]
            try:
                valid = checker(proposition, tuple(u.proposition for u in units)) is True
            except Exception:
                valid = False
            if not valid:
                raise ValueError('Rule does not justify this conclusion from these premises')
            applications = (len(units) - 1 if rule.startswith('transitive:') else 1) + max(u.rule_applications for u in units)
            unit = EvidenceUnit(proposition, (), EXPLICIT, 'rule_derived', {'kind': 'rule', 'rule': rule},
                                None, premise_ids, rule, version, applications, proposal=json.loads(canonical(proposal)))
            if unit.unit_id in premise_ids:
                raise ValueError('Cyclic derivation')
            self.store.put('unit', unit.unit_id, unit.record(), head=unit.unit_id)
            return unit.unit_id

    def derive_paths(self, entity, predicate, *, as_of=None, max_hops=3, max_paths=20):
        from .paths import derive_paths
        return derive_paths(self, entity, predicate, as_of=as_of, max_hops=max_hops, max_paths=max_paths)

    def find_paths(self, entity, *, predicates=None, as_of=None, max_hops=3, max_paths=20):
        from .paths import find_paths
        return find_paths(self, entity, predicates=predicates, as_of=as_of, max_hops=max_hops, max_paths=max_paths)

    # -------------------------------------------------------------- eligibility
    def _condition_state(self, condition, context, predicate):
        """Return True (met), False (violated), None (pending), or 'invalid' (malformed runtime value)."""
        outcome = self.vocabulary.evaluate_condition(predicate, condition, context)
        if outcome == 'true':
            return True
        if outcome == 'false':
            return False
        if outcome == 'invalid':
            return 'invalid'
        return None

    def _eligible(
        self,
        unit,
        query,
        mode,
        seen=frozenset(),
        kinds=None,
        validity='filter',
        *,
        units_by_id=None,
        documents_override=None,
        document_revisions_override=None,
        field_evidences_map=None,
        field_evidence_overrides=None,
        require_field_evidence=(),
        skip_fidelity=False,
    ):
        """Return (basis, None) or (None, reason) for a unit under the unified v4 admissibility core."""
        docs = {d.document_id: d for d in self.documents} if documents_override is None else documents_override
        units_map = {u.unit_id: u for u in self.units} if units_by_id is None else units_by_id
        fe_map = self.field_evidences if field_evidences_map is None else field_evidences_map
        ev = evaluate_unit_admissibility(
            unit,
            vocabulary=self.vocabulary,
            policy=self.policy,
            profile=self.profile,
            units_by_id=units_map,
            rules=self._rules,
            documents=docs,
            document_revisions=document_revisions_override,
            as_of=query.as_of,
            validity=validity,
            allowed_assertion_kinds=kinds,
            query_scope=query.context,
            query_conditions=query.context,
            condition_mode=mode,
            field_evidences=field_evidence_overrides,
            stored_field_evidences=fe_map,
            require_field_evidence=require_field_evidence,
            check_evidence_refs=True,
            skip_fidelity=skip_fidelity,
            seen=seen,
        )
        if not ev.admissible:
            return None, ev.reason
        return ev.as_basis(), None

    # ------------------------------------------------------------------ queries
    def _query(self, query, *, validity='filter'):
        if isinstance(query, dict):
            query = StructuredQuery(**query)
        if not isinstance(query, StructuredQuery):
            raise TypeError('Use StructuredQuery or its dictionary; interpret natural language before asking')
        as_of = query.as_of if (query.as_of is not None or validity == 'ignore') else datetime.now(timezone.utc).isoformat()
        return replace(query, entity=self.vocabulary.entity(query.entity), as_of=as_of)

    def ask(self, query, *, conditions='report', assertion_kinds=None, validity='filter'):
        """Resolve, list or aggregate over eligible units. Never invokes a model.

        conditions='report' returns conditional answers labelled with their
        pending conditions; 'discharge' admits only units whose conditions and
        scope are satisfied by query.context. assertion_kinds overrides the
        policy's admitted kinds for this query; results always carry their kind.
        """
        if assertion_kinds is not None and not set(assertion_kinds) <= set(KINDS):
            raise ValueError('Unknown assertion kind')
        if conditions not in {'report', 'discharge'}:
            raise ValueError("conditions must be 'report' or 'discharge'")
        if validity not in {'filter', 'ignore'}:
            raise ValueError("validity must be 'filter' (as_of) or 'ignore' (document review across periods)")
        try:
            query = self._query(query, validity=validity)
            spec = self.vocabulary.predicate(query.predicate)
        except UnmappedProposal as exc:
            return self._issue({'status': 'invalid_query', 'abstain': True, 'reason': exc.as_dict(), 'results': []})
        try:
            snap = self.snapshot()
            fe_map = snap['field_evidences']
            docs_map = snap['documents']
            doc_revs_map = snap['document_revisions']
            units_list = snap['units']
            units_map = {u.unit_id: u for u in units_list}
            query_entity_key = label_key(query.entity)
            bases, excluded = [], []
            for unit in units_list:
                p = unit.proposition
                if label_key(p['entity']) != query_entity_key or p['predicate'] != query.predicate:
                    continue
                basis, reason = self._eligible(
                    unit,
                    query,
                    conditions,
                    kinds=assertion_kinds,
                    validity=validity,
                    units_by_id=units_map,
                    documents_override=docs_map,
                    document_revisions_override=doc_revs_map,
                    field_evidences_map=fe_map,
                )
                if basis is None:
                    excluded.append({'unit_id': unit.unit_id, 'reason': reason})
                else:
                    bases.append(basis)
            rejected = [
                r for r in snap['rejections']
                if r['hint'].get('entity') == query.entity and r['hint'].get('predicate') == query.predicate
            ]
            coverage = {'eligible_units': len(bases), 'excluded_units': len(excluded),
                        'rejected_proposals': len(rejected), 'rejection_ids': [r['rejection_id'] for r in rejected],
                        'scope': 'ledger for this project and vocabulary; not completeness of the source corpus'}
            if query.operation in {'sum', 'count', 'min', 'max', 'average'}:
                results, reason = self._aggregate(bases, query, spec, coverage)
                status = 'answered' if results else ('incomplete' if reason == 'coverage_incomplete' else
                                                      'unresolved_conflict' if reason == 'unresolved_event_values' else
                                                      'insufficient_evidence')
            else:
                results = self._resolve(bases, query, spec)
                reason = None
                accepted = [r for r in results if r['status'] == 'accepted']
                status = ('answered' if accepted else 'unresolved_conflict'
                          if any(r['status'] in {'unresolved_conflict', 'unresolved_comparison'} for r in results)
                          else 'insufficient_evidence')
            used = {b['unit'].unit_id for b in bases}
            for basis in bases:
                used.update(basis.get('premise_bases', ()))
            revisions = sorted({d.revision for b in bases for d in b['documents'].values()})
            unit_digests = {
                uid: digest((units_map[uid] if uid in units_map else self.unit(uid)).record())
                for uid in sorted(used)
            }
            return self._issue({'query': asdict(query), 'conditions_mode': conditions, 'validity_mode': validity, 'status': status,
                                'abstain': status != 'answered', 'reason': reason, 'results': results,
                                'excluded': excluded, 'coverage': coverage, 'configuration': self.configuration,
                                'basis': {'unit_record_digests': unit_digests, 'source_revisions': revisions},
                                'support_kind': 'policy index, not a calibrated probability',
                                'source_authentication': 'not_verified'})
        except Exception as exc:
            return self._issue_error_escalated({
                'query': asdict(query),
                'conditions_mode': conditions,
                'validity_mode': validity,
                'status': 'error_escalated',
                'disposition': 'ESCALATE',
                'abstain': True,
                'reason': 'internal_error',
                'error': {'type': type(exc).__name__, 'detail': str(exc)[:300]},
                'results': [],
                'excluded': [],
                'coverage': {},
                'configuration': self.configuration,
                'support_kind': 'policy index, not a calibrated probability',
                'source_authentication': 'not_verified',
            })

    def _support(self, members):
        """Max within dependence groups, Noisy-OR across groups; shared origins never double count."""
        assessed = sorted((m for m in members if m['support'] is not None),
                          key=lambda m: (-m['support'], m['unit'].unit_id))
        used, selected = set(), []
        for member in assessed:
            if not member['dependence'] & used:
                used |= member['dependence']
                selected.append(member)
        if not assessed:
            return None, []
        return _noisy_or([m['support'] for m in selected]), [m['unit'].unit_id for m in selected]

    def _opposition(self, groups):
        by_dependence = {}
        for group in groups:
            for member in group['members']:
                if member['support'] is None:
                    continue
                key = tuple(sorted(member['dependence']))
                by_dependence[key] = max(by_dependence.get(key, 0.0), member['support'])
        # Overlapping dependence keys collapse to their maximum (conservative).
        merged = {}
        for key, value in by_dependence.items():
            clash = [k for k in merged if set(k) & set(key)]
            for k in clash:
                value = max(value, merged.pop(k))
                key = tuple(sorted(set(key) | set(k)))
            merged[key] = value
        return _noisy_or(list(merged.values()))

    def _relation(self, spec, a, b):
        """Relation of two value groups that a single-valued predicate would otherwise treat as conflicting.

        'incompatible' (a real conflict), 'compatible' (one states the other less precisely or less fully),
        or 'unknown' (undecidable without information the values lack, e.g. a unit dimension or time zone)."""
        if a['unit'] is not None and b['unit'] is not None:
            units = self.vocabulary.units
            if units.dimension_of(a['unit']) != units.dimension_of(b['unit']):
                return 'unknown'
        relation = value_relation(spec.value, a['value'], b['value'])
        if relation == 'equal':  # the same value (possibly written differently): conflicting only if negated
            return 'incompatible' if a['polarity'] != b['polarity'] else 'compatible'
        return relation

    def _resolve(self, bases, query, spec):
        contexts = {}
        for basis in bases:
            key = canonical({'context': json.loads(basis['unit'].context_key),
                             'pending_conditions': basis['pending_conditions'],
                             'pending_scope': basis['pending_scope']})
            contexts.setdefault(key, []).append(basis)
        results = []
        for key in sorted(contexts):
            members = contexts[key]
            groups = {}
            for basis in members:
                groups.setdefault(basis['unit'].assertion_key, []).append(basis)
            table = []
            for assertion, group_members in sorted(groups.items()):
                support, selected = self._support(group_members)
                p = group_members[0]['unit'].proposition
                assessed = [m for m in group_members if m['support'] is not None]
                table.append({'assertion': assertion, 'value': p['value'], 'unit': p['unit'], 'polarity': p['polarity'],
                              'members': group_members, 'support': support, 'selected': selected,
                              'rank': max(m['rank'] for m in assessed) if assessed else max(m['rank'] for m in group_members)})
            context = json.loads(key)
            results.extend(self._decide(table, spec, query, context))
        return results

    def _incompatible(self, spec, a, b):
        if a['assertion'] == b['assertion']:
            return False
        same_value = canonical(a['value']) == canonical(b['value']) and a['unit'] == b['unit']
        if same_value and a['polarity'] != b['polarity']:
            return True  # explicit negation
        return spec.cardinality == 'single' and a['polarity'] == b['polarity'] == 'affirmed'

    def _compare_group_competitors(self, spec, group, others):
        """Compare *group* against potential conflicting peers, returning `(competitors, opposing)`."""
        competitors, opposing = [], []
        for other in others:
            if other is group:
                continue
            value_relation_ = self._relation(spec, group, other) if self._incompatible(spec, group, other) else 'compatible'
            if value_relation_ == 'compatible':
                competitors.append(self._competitor(other, 'compatible'))
                continue
            if value_relation_ == 'unknown':
                relation = 'unresolved'
            elif other['rank'] < group['rank']:
                relation = 'defeated_by_authority'
            elif other['rank'] > group['rank']:
                relation = 'outranks' if other['support'] is not None else 'unresolved'
            else:
                relation = 'opposing' if other['support'] is not None else 'unresolved'
            competitors.append(self._competitor(other, relation))
            if relation in {'opposing', 'outranks'}:
                opposing.append(other)
        return competitors, opposing

    def _score_and_build_result(self, group, competitors, opposing, context):
        if group['support'] is None:
            status, opposition = 'unassessed', 0.0
        else:
            opposition = self._opposition(opposing)
            q = _q_index(group['support'], opposition)
            if any(c['relation'] == 'outranks' for c in competitors):
                status = 'defeated_by_authority'
            elif any(c['relation'] == 'unresolved' for c in competitors):
                status = 'unresolved_comparison'
            elif q >= self.policy.tau and group['support'] - opposition >= self.policy.delta:
                status = 'accepted'
            elif opposition > 0:
                status = 'unresolved_conflict'
            else:
                status = 'below_threshold'
        return self._result(group, status, context, competitors, opposition)

    def _decide(self, table, spec, query, context):
        if spec.cardinality == 'multi':
            # Distinct values in a multi-valued predicate never compete; only identical
            # (canonical(value), unit) groups with opposite polarity can conflict (O(N)).
            by_val_unit: dict[tuple[str, Any], list[dict[str, Any]]] = {}
            group_keys: list[tuple[str, Any]] = []
            for group in table:
                key = (canonical(group['value']), group['unit'])
                group_keys.append(key)
                by_val_unit.setdefault(key, []).append(group)
            out = []
            for group, key in zip(table, group_keys):
                others = [
                    other for other in by_val_unit[key]
                    if other is not group and other['polarity'] != group['polarity']
                    and other['assertion'] != group['assertion']
                ]
                competitors, opposing = self._compare_group_competitors(spec, group, others)
                out.append(self._score_and_build_result(group, competitors, opposing, context))
            return out

        out = []
        candidates = table if query.operation == 'list' else None
        if candidates is None:
            assessed = [g for g in table if g['support'] is not None]
            if not assessed:
                return [self._result(g, 'unassessed', context, [], 0.0) for g in table]
            top_rank = max(g['rank'] for g in assessed)
            top = sorted([g for g in assessed if g['rank'] == top_rank], key=lambda g: (-g['support'], g['assertion']))
            candidates = top[:1]
        for group in candidates:
            competitors, opposing = self._compare_group_competitors(spec, group, table)
            out.append(self._score_and_build_result(group, competitors, opposing, context))
        return out

    def _competitor(self, group, relation):
        return {'relation': relation, 'value': group['value'], 'unit': group['unit'], 'polarity': group['polarity'],
                'support': group['support'], 'authority_rank': group['rank'],
                'unit_ids': sorted(m['unit'].unit_id for m in group['members'])}

    def _result(self, group, status, context, competitors, opposition):
        support = group['support']
        evidence = []
        for member in sorted(group['members'], key=lambda m: m['unit'].unit_id):
            unit = member['unit']
            evidence.append({'unit_id': unit.unit_id, 'origin': unit.origin, 'basis': unit.basis,
                             'support_index': member['support'], 'components': member['components'],
                             'authority_rank': member['rank'], 'dependence': sorted(member['dependence']),
                             'evidence': [asdict(r) for r in unit.evidence], 'premises': list(unit.premises),
                             'rule': unit.rule})
        return {'status': status, 'value': group['value'], 'unit': group['unit'], 'polarity': group['polarity'],
                'context': context['context'], 'conditional': bool(context['pending_conditions'] or context['pending_scope']),
                'pending_conditions': context['pending_conditions'], 'pending_scope': context['pending_scope'],
                'support': {'S': support, 'O': opposition,
                            'Q': None if support is None else _q_index(support, opposition),
                            'margin': None if support is None else round(support - opposition, 6),
                            'tau': self.policy.tau, 'delta': self.policy.delta,
                            'thresholds_validated': self.policy.thresholds_validated},
                'authority_rank': group['rank'], 'selected_unit_ids': group['selected'],
                'competitors': competitors, 'evidence': evidence}

    def _aggregate(self, bases, query, spec, coverage):
        if spec.value.type not in {'quantity', 'number'} and query.operation != 'count':
            return [], 'numeric_values_required'
        if any(b['pending_conditions'] or b['pending_scope'] for b in bases):
            return [], 'conditions_not_discharged'
        if not bases:
            return [], 'no_eligible_evidence'
        if len({b['unit'].aggregate_key for b in bases}) != 1:
            return [], 'heterogeneous_contexts'
        events = {}
        for basis in bases:
            event = basis['unit'].proposition['scope'].get('event_id')
            if not event:
                return [], 'event_identity_required'
            events.setdefault(event, []).append(basis)
        chosen, details = [], []
        single = replace(spec, cardinality='single')
        resolve_query = replace(query, operation='resolve')
        for event in sorted(events):
            groups = {}
            for basis in events[event]:
                groups.setdefault(basis['unit'].assertion_key, []).append(basis)
            table = []
            for assertion, members in sorted(groups.items()):
                support, selected = self._support(members)
                p = members[0]['unit'].proposition
                assessed = [m for m in members if m['support'] is not None]
                table.append({'assertion': assertion, 'value': p['value'], 'unit': p['unit'], 'polarity': p['polarity'],
                              'members': members, 'support': support, 'selected': selected,
                              'rank': max(m['rank'] for m in assessed) if assessed else 0})
            decision = self._decide(table, single, resolve_query, {'context': {}, 'pending_conditions': [], 'pending_scope': {}})
            details.append({'event_id': event, 'decision': decision})
            if not decision or decision[0]['status'] != 'accepted':
                return [], 'unresolved_event_values'
            chosen.append(decision[0])
        if any(c['polarity'] != 'affirmed' for c in chosen):
            return [], 'negated_amounts'
        if coverage['rejected_proposals']:
            return [], 'coverage_incomplete'
        if query.operation == 'count':
            value, unit = str(len(chosen)), 'count'
        else:
            if len({c['unit'] for c in chosen}) != 1:
                return [], 'incompatible_units'
            amounts = [Decimal(c['value']['value'] if type(c['value']) is dict else c['value']) for c in chosen]
            value = {'sum': lambda: sum(amounts), 'min': lambda: min(amounts), 'max': lambda: max(amounts),
                     'average': lambda: sum(amounts) / len(amounts)}[query.operation]()
            value, unit = decimal_text(value, 'aggregate'), chosen[0]['unit']
        return [{'status': 'accepted', 'operation': query.operation, 'value': value, 'unit': unit,
                 'event_ids': sorted(events), 'events': details}], None

    # --------------------------------------------------------------- claim check
    def check_claim(self, proposal, *, as_of=None, context=None, validity='filter', _ask_cache=None):
        """Deterministically compare an external assertion with the ledger. No model call.

        supported: an accepted ledger value in the same context equals it.
        contradicted: an accepted value in the same context is incompatible with it.
        not_supported: the ledger holds the entity/predicate only in other contexts
                       (e.g. the assertion drops or inverts a condition, or changes the kind).
        not_representable: the assertion violates the vocabulary (e.g. unit dimension).
        error_escalated: fail-closed internal error during claim resolution (escalates, never aborts callers).
        """
        try:
            proposition = self.vocabulary.proposition(proposal)
        except UnmappedProposal as exc:
            return self._issue_error_escalated({'operation': 'check_claim', 'status': 'not_representable',
                                                'proposal': proposal, 'reason': exc.as_dict(),
                                                'configuration': self.configuration})
        try:
            query = {'entity': proposition['entity'], 'predicate': proposition['predicate'], 'operation': 'list',
                     'as_of': as_of, 'context': dict(context or {})}
            assertion_kinds = tuple(sorted(set(self.policy.allowed_assertion_kinds) | {proposition['assertion_kind']}))
            response = None
            cache_key = None
            if _ask_cache is not None:
                cache_key = (canonical(query), 'report', assertion_kinds, validity)
                response = _ask_cache.get(cache_key)
            if response is None:
                response = self.ask(query, conditions='report', validity=validity, assertion_kinds=assertion_kinds)
                if _ask_cache is not None and response.get('status') != 'error_escalated':
                    _ask_cache[cache_key] = response
            if response.get('status') == 'error_escalated':
                return self._issue_error_escalated({
                    'operation': 'check_claim',
                    'status': 'error_escalated',
                    'disposition': 'ESCALATE',
                    'abstain': True,
                    'proposition': proposition,
                    'matched_result': None,
                    'differences': [],
                    'relation': None,
                    'error': response.get('error'),
                    'normalizer_version': NORMALIZER_VERSION,
                    'ledger_response_id': response.get('response_id'),
                    'configuration': self.configuration,
                    'decision_kind': 'fail-closed escalation on internal resolution error',
                })
            target_context = json.loads(context_key(proposition))
            same = [r for r in response['results'] if r['context'] == target_context]
            claim_value = {'value': proposition['value'], 'unit': proposition['unit'], 'polarity': proposition['polarity']}
            target_val_canon = canonical(claim_value['value'])
            spec = self.vocabulary.predicate(proposition['predicate'])
            status, matched, differences = 'insufficient_evidence', None, []
            accepted = [r for r in same if r['status'] == 'accepted']
            for result in accepted:
                same_value = (
                    result['unit'] == claim_value['unit']
                    and result['polarity'] == claim_value['polarity']
                    and canonical(result['value']) == target_val_canon
                )
                if not same_value and result['unit'] == claim_value['unit'] and result['polarity'] == claim_value['polarity']:
                    same_value = value_relation(spec.value, result['value'], claim_value['value']) == 'equal'
                if same_value:
                    status, matched = 'supported', result
                    break
            relation = 'equal' if status == 'supported' else None
            if status != 'supported':
                weaker = None
                for result in accepted:
                    a = {'assertion': 'x', **{k: result[k] for k in ('value', 'unit', 'polarity')}}
                    b = {'assertion': 'y', **claim_value}
                    if spec.cardinality == 'multi':
                        incompatible = (
                            result['unit'] == claim_value['unit']
                            and result['polarity'] != claim_value['polarity']
                            and canonical(result['value']) == target_val_canon
                        )
                    else:
                        incompatible = self._incompatible(spec, a, b)
                    kind = self._relation(spec, a, b) if incompatible else 'compatible'
                    if kind == 'incompatible':
                        status, matched, relation = 'contradicted', result, 'incompatible'
                        differences = self._differences(claim_value, {k: result[k] for k in ('value', 'unit', 'polarity')})
                        break
                    if weaker is None or (kind == 'unknown' and weaker[0] == 'compatible'):
                        weaker = (kind, result)
                if status != 'contradicted' and weaker is not None:
                    # Not decidable from the ledger: e.g. a less precise date or a fuller name. Shown, never a verdict.
                    relation, matched = weaker[0], weaker[1]
                    differences = self._differences(claim_value, {k: weaker[1][k] for k in ('value', 'unit', 'polarity')})
            if status == 'insufficient_evidence' and same and any(r['status'] in {'unresolved_conflict', 'unresolved_comparison'} for r in same):
                status = 'unresolved_conflict'
            if status == 'insufficient_evidence' and not same and response['results']:
                status = 'not_supported'
                nearest = min(response['results'], key=lambda r: len(self._context_differences(target_context, proposition, r)))
                matched = nearest
                differences = self._context_differences(target_context, proposition, nearest)
                if nearest.get('status') == 'accepted':
                    if (
                        nearest['unit'] == claim_value['unit']
                        and nearest['polarity'] == claim_value['polarity']
                        and canonical(nearest['value']) == target_val_canon
                    ):
                        relation = 'equal'
                    elif nearest['unit'] == claim_value['unit'] and nearest['polarity'] == claim_value['polarity']:
                        relation = value_relation(spec.value, nearest['value'], claim_value['value'])
                    else:
                        a = {'assertion': 'x', **{k: nearest[k] for k in ('value', 'unit', 'polarity')}}
                        b = {'assertion': 'y', **claim_value}
                        relation = self._relation(spec, a, b) if self._incompatible(spec, a, b) else 'compatible'
            return self._issue({'operation': 'check_claim', 'status': status, 'proposition': proposition,
                                'matched_result': matched, 'differences': differences, 'relation': relation,
                                'normalizer_version': NORMALIZER_VERSION,
                                'ledger_response_id': response['response_id'], 'configuration': self.configuration,
                                'decision_kind': 'deterministic comparison with ledger; truth not certified'})
        except Exception as exc:
            return self._issue_error_escalated({
                'operation': 'check_claim',
                'status': 'error_escalated',
                'disposition': 'ESCALATE',
                'abstain': True,
                'proposition': proposition,
                'matched_result': None,
                'differences': [],
                'relation': None,
                'error': {'type': type(exc).__name__, 'detail': str(exc)[:300]},
                'normalizer_version': NORMALIZER_VERSION,
                'configuration': self.configuration,
                'decision_kind': 'fail-closed escalation on internal resolution error',
            })

    @staticmethod
    def _differences(expected, observed, path=''):
        if canonical(expected) == canonical(observed):
            return []
        if type(expected) is dict and type(observed) is dict:
            out = []
            for key in sorted(set(expected) | set(observed)):
                here = f'{path}/{key}'
                if key not in expected or key not in observed:
                    out.append({'path': here, 'claim': expected.get(key), 'ledger': observed.get(key)})
                else:
                    out.extend(EvidenceLayer._differences(expected[key], observed[key], here))
            return out
        return [{'path': path or '/', 'claim': expected, 'ledger': observed}]

    def _context_differences(self, target, proposition, result):
        out = []
        ledger = dict(result['context'])
        if result['conditional']:
            ledger['conditions'] = result['pending_conditions']
        for key in ('assertion_kind', 'scope', 'conditions', 'condition_status'):
            if canonical(target.get(key)) != canonical(ledger.get(key)):
                out.append({'path': '/' + key, 'claim': target.get(key), 'ledger': ledger.get(key)})
        out.extend(self._differences({k: proposition[k] for k in ('value', 'unit', 'polarity')},
                                     {k: result[k] for k in ('value', 'unit', 'polarity')}))
        return out

    # ---------------------------------------------------------------- integrity
    def _issue(self, payload):
        payload = json.loads(canonical(payload))
        response = {**payload, 'certificate': {'scheme': 'CLAIMLAYER-EVIDENCE-INTEGRITY-v4', 'scope': 'integrity',
                                               'payload_digest': digest(payload)}}
        response_id = digest(response)
        self.store.put('response', response_id, response)
        return {**response, 'response_id': response_id}

    def _issue_error_escalated(self, payload):
        """Issue a certified fail-closed response even if store persistence itself raises."""
        try:
            return self._issue(payload)
        except Exception:
            clean = json.loads(canonical(payload))
            response = {**clean, 'certificate': {'scheme': 'CLAIMLAYER-EVIDENCE-INTEGRITY-v4', 'scope': 'integrity',
                                                 'payload_digest': digest(clean)}}
            return {**response, 'response_id': digest(response)}

    def verify_response_integrity(self, response):
        try:
            if type(response) is not dict or 'response_id' not in response:
                return False
            body = {k: v for k, v in response.items() if k != 'response_id'}
            if digest(body) != response['response_id'] or self.store.get('response', response['response_id']) != body:
                return False
            basis = response.get('basis', {})
            for uid, fingerprint in basis.get('unit_record_digests', {}).items():
                if digest(self.unit(uid).record()) != fingerprint:
                    return False
            for revision in basis.get('source_revisions', []):
                record = self.store.get('document', revision)
                if record is None or SourceDocument(**record).revision != revision:
                    return False
            return True
        except (ValueError, TypeError, KeyError):
            return False
