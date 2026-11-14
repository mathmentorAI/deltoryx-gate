"""ClaimLayer v4 engine: the model proposes, deterministic policy decides.

Faithful to the Evidence Intelligence contract:
* a language model only *proposes* typed evidence units bound to exact spans;
* every unit must conform to an application vocabulary (no free-form keys);
* support, conflicts, authority, aggregation and derivations are computed by
  code over the ledger (dependence-aware Noisy-OR, Q = S^2/(S+O), margin rule);
* the v2 epistemic table weighs how a unit was obtained and blocks bases such
  as narrative or abductive assumptions;
* no model ever issues a supported/contradicted verdict.

Results remain conditional on extraction fidelity, vocabulary design and the
application's authority metadata; nothing here certifies factual truth.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
import copy
import json
from pathlib import Path
import tempfile
import uuid
from typing import Any, Iterable, Mapping

from ..semantic.models import KINDS, EvidenceRef, SourceDocument, StructuredQuery, canonical, digest, instant
from ..semantic.store import VerificationStore
from .epistemics import DEFAULT_PROFILE, EXPLICIT, EpistemicProfile
from .models import AcceptancePolicy, EvidenceUnit, context_key
from .vocabulary import UnmappedProposal, Vocabulary, decimal_text, label_key, value_relation

ENGINE_VERSION = '4.0.0a1'
CONTRACT = 'claimlayer-evidence-v4'
TIER_PRIMARY, TIER_ANNEX, TIER_EXTERNAL = 100, 50, 10  # v2 tiers, now declared by the application


def _noisy_or(values):
    remaining = 1.0
    for value in values:
        remaining *= 1.0 - value
    return round(1.0 - remaining, 6)


def _q_index(support, opposition):
    return round(support ** 2 / (support + opposition), 6) if support + opposition > 0 else 0.0


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

    def _refs_valid(self, refs):
        if not refs:
            raise ValueError('Source units need exact evidence references')
        for ref in refs:
            document = self._document(ref.document_id, ref.revision)
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
        return {
            'configuration': self.configuration,
            'documents': docs,
            'document_revisions': doc_revs,
            'units': units,
            'unread_documents': unread,
            'read_errors': tuple(read_errors),
            'field_evidences': field_evidences,
            'rejections': rejections,
        }

    def ingest(self, documents):
        """Store sources; if a reader is configured, record its proposals and admit conforming units."""
        if isinstance(documents, SourceDocument):
            documents = [documents]
        summary = {'documents_ingested': 0, 'units_admitted': 0, 'proposals_rejected': 0,
                   'rejection_reasons': {}, 'unread_documents': [], 'read_errors': []}
        for document in documents:
            if not isinstance(document, SourceDocument):
                raise TypeError('Use SourceDocument; authority, weight and dependence are application metadata')
            self.store.put('document', document.revision, asdict(document), head=document.document_id)
            summary['documents_ingested'] += 1
            if self.reader is None:
                summary['unread_documents'].append(document.document_id)
                self._record_doc_read_status(document.document_id, document.revision, unread=True, errors=[])
                continue
            request = {'task': 'propose', 'contract': CONTRACT, 'reader': self.reader.identity,
                       'document': {'document_id': document.document_id, 'revision': document.revision,
                                    'text': document.text},
                       'vocabulary': self.vocabulary.describe(), 'profile': self.profile.describe()}
            key = digest(request)
            _, cached = self.store.head('read', key)
            if cached is None:
                try:
                    result = self.reader.propose(copy.deepcopy(request))
                    canonical(result)
                except Exception as exc:  # recorded, never converted into evidence
                    self._event({'task': 'propose', 'request_key': key, 'error_type': type(exc).__name__,
                                 'diagnostics': getattr(exc, 'diagnostics', None)})
                    err = {'document_id': document.document_id, 'error_type': type(exc).__name__}
                    summary['read_errors'].append(err)
                    summary['unread_documents'].append(document.document_id)
                    self._record_doc_read_status(document.document_id, document.revision, unread=True, errors=[err])
                    continue
                cached = {'request_key': key, 'result': json.loads(canonical(result))}
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

    def ingest_units(self, items, *, attestor: str, structural_reliability: float | None = 1.0):
        """Admit application-supplied typed proposals bound to exact spans (no model involved)."""
        if not isinstance(attestor, str) or not attestor.strip():
            raise ValueError('An application attestor is required')
        summary = {'units_admitted': 0, 'proposals_rejected': 0, 'rejection_reasons': {}, 'unit_ids': []}
        producer = {'kind': 'application', 'attestor': attestor}
        for item in items:
            evidence = [asdict(r) if isinstance(r, EvidenceRef) else r for r in item.get('evidence', ())]
            unit_id = self._admit(item.get('proposal'), item.get('basis', EXPLICIT), evidence, 'application',
                                  producer, structural_reliability, summary)
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

    def _admit(self, proposal, basis, evidence, origin, producer, structural_reliability, summary):
        try:
            if type(proposal) is not dict:
                raise UnmappedProposal('proposal_object_required', '', proposal)
            proposition = self.vocabulary.proposition(proposal)
            if not self.profile.admissible(basis):
                raise UnmappedProposal('basis_blocked' if basis in self.profile.blocked else 'unknown_basis', 'basis', basis)
            refs = tuple(EvidenceRef(**r) for r in (evidence or ()))
            self._refs_valid(refs)
            unit = EvidenceUnit(proposition, refs, basis, origin, producer, structural_reliability,
                                proposal=json.loads(canonical(proposal)))
        except UnmappedProposal as exc:
            return self._reject(proposal, basis, producer, exc.as_dict(), summary)
        except (ValueError, TypeError, KeyError) as exc:
            return self._reject(proposal, basis, producer, {'reason': 'evidence_binding', 'path': 'evidence',
                                                            'detail': str(exc)}, summary)
        self.store.put('unit', unit.unit_id, unit.record(), head=unit.unit_id)
        summary['units_admitted'] += 1
        return unit.unit_id

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
        with self.store.connect() as db:
            ids = [r[0] for r in db.execute('SELECT record_id FROM semantic_records WHERE project_id=? AND kind=? ORDER BY rowid',
                                          (self.store.project_id, 'rejection'))]
        out = []
        for rid in ids:
            record = self.store.get('rejection', rid)
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
        """Return True (met), False (violated) or None (not discharged by runtime context)."""
        name = condition['field']
        if name not in context:
            return None
        spec = dict(predicate.conditions)[name]
        right, op = condition['value'], condition['operator']
        try:
            if spec.type == 'quantity':
                raw = context[name]
                amount, base = self.vocabulary.units.to_base(raw['value'], raw['unit'], 'context')
                if base != right['unit']:
                    return False
                left, right = Decimal(amount), Decimal(right['value'])
            else:
                left, _ = self.vocabulary.value(spec, context[name], None, 'context')
                if spec.type == 'number':
                    left, right = Decimal(left), Decimal(right)
                elif spec.type != 'date' and op in {'lt', 'le', 'gt', 'ge'}:
                    return False
        except (UnmappedProposal, TypeError, KeyError):
            return False
        if op in {'eq', 'ne'}:
            equal = left == right if isinstance(left, Decimal) else canonical(left) == canonical(right)
            return equal if op == 'eq' else not equal
        return {'lt': left < right, 'le': left <= right, 'gt': left > right, 'ge': left >= right}[op]

    def _eligible(self, unit, query, mode, seen=frozenset(), kinds=None, validity='filter'):
        """Return (basis, None) or (None, reason) for a unit under a query and policy."""
        if unit.unit_id in seen:
            return None, 'cyclic_dependency'
        p = unit.proposition
        if p['assertion_kind'] not in (kinds or self.policy.allowed_assertion_kinds):
            return None, 'assertion_kind_not_admitted'
        if validity == 'filter':
            at = instant(query.as_of)
            start, end = instant(p['valid_from']), instant(p['valid_until'])
            if (start and at < start) or (end and at > end):
                return None, 'outside_validity'
        pending_scope = {}
        predicate = self.vocabulary.predicate(p['predicate'])
        for key, value in p['scope'].items():
            if key == 'event_id':
                if key in query.context and canonical(query.context[key]) != canonical(value):
                    return None, 'scope_mismatch'
                continue
            if key in query.context:
                if canonical(query.context[key]) != canonical(value):
                    return None, 'scope_mismatch'
            else:
                pending_scope[key] = value
        pending_conditions = []
        for condition in p['conditions']:
            state = self._condition_state(condition, query.context, predicate)
            if state is False:
                return None, 'condition_not_met'
            if state is None:
                pending_conditions.append(condition)
        if mode == 'discharge' and (pending_conditions or pending_scope):
            return None, 'condition_not_discharged' if pending_conditions else 'scope_not_discharged'
        if unit.origin == 'rule_derived':
            installed = self._rules.get(unit.rule)
            if installed is None or installed[0] != unit.rule_version:
                return None, 'rule_version_unavailable'
            bases = []
            for pid in unit.premises:
                try:
                    premise = self.unit(pid)
                except KeyError:
                    return None, 'premise_missing'
                basis, reason = self._eligible(premise, query, mode, seen | {unit.unit_id}, kinds, validity)
                if basis is None:
                    return None, 'premise_' + reason
                if basis['support'] is None:
                    return None, 'premise_unassessed'
                bases.append(basis)
            try:
                valid = installed[1](p, tuple(b['unit'].proposition for b in bases)) is True
            except Exception:
                valid = False
            if not valid:
                return None, 'invalid_derivation'
            weakest = min(b['support'] for b in bases)
            support = round((self.policy.path_discount ** unit.rule_applications) * weakest, 6)
            documents = {k: v for b in bases for k, v in b['documents'].items()}
            return {'unit': unit, 'documents': documents, 'support': support,
                    'components': {'weakest_premise': weakest, 'rule_applications': unit.rule_applications,
                                   'path_discount': self.policy.path_discount},
                    'dependence': frozenset(d for b in bases for d in b['dependence']),
                    'rank': min(b['rank'] for b in bases), 'pending_conditions': pending_conditions,
                    'pending_scope': pending_scope, 'premise_bases': [b['unit'].unit_id for b in bases]}, None
        try:
            self._refs_valid(unit.evidence)
        except ValueError:
            return None, 'stale_evidence'
        documents = {r.document_id: self._document(r.document_id, r.revision) for r in unit.evidence}
        if any(d.attribution not in self.policy.allowed_source_attributions for d in documents.values()):
            return None, 'source_attribution_not_admitted'
        if not self.profile.admissible(unit.basis):
            return None, 'basis_not_admissible_under_profile'
        components = {'span': 1.0, 'structure': unit.structural_reliability,
                      'applicability': min(d.weight for d in documents.values()),
                      'basis': self.profile.weight(unit.basis)}
        support = None if any(v is None for v in components.values()) else round(min(components.values()), 6)
        dependence = frozenset(d.dependence_group or 'document:' + d.document_id for d in documents.values())
        return {'unit': unit, 'documents': documents, 'support': support, 'components': components,
                'dependence': dependence, 'rank': min(d.authority_rank for d in documents.values()),
                'pending_conditions': pending_conditions, 'pending_scope': pending_scope}, None

    # ------------------------------------------------------------------ queries
    def _query(self, query):
        if isinstance(query, dict):
            query = StructuredQuery(**query)
        if not isinstance(query, StructuredQuery):
            raise TypeError('Use StructuredQuery or its dictionary; interpret natural language before asking')
        return replace(query, entity=self.vocabulary.entity(query.entity),
                       as_of=query.as_of or datetime.now(timezone.utc).isoformat())

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
            query = self._query(query)
            spec = self.vocabulary.predicate(query.predicate)
        except UnmappedProposal as exc:
            return self._issue({'status': 'invalid_query', 'abstain': True, 'reason': exc.as_dict(), 'results': []})
        bases, excluded = [], []
        for unit in self.units:
            p = unit.proposition
            if label_key(p['entity']) != label_key(query.entity) or p['predicate'] != query.predicate:
                continue
            basis, reason = self._eligible(unit, query, conditions, kinds=assertion_kinds, validity=validity)
            if basis is None:
                excluded.append({'unit_id': unit.unit_id, 'reason': reason})
            else:
                bases.append(basis)
        rejected = self.rejections(entity=query.entity, predicate=query.predicate)
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
        return self._issue({'query': asdict(query), 'conditions_mode': conditions, 'validity_mode': validity, 'status': status,
                            'abstain': status != 'answered', 'reason': reason, 'results': results,
                            'excluded': excluded, 'coverage': coverage, 'configuration': self.configuration,
                            'basis': {'unit_record_digests': {uid: digest(self.unit(uid).record()) for uid in sorted(used)},
                                      'source_revisions': revisions},
                            'support_kind': 'policy index, not a calibrated probability',
                            'source_authentication': 'not_verified'})

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

    def _decide(self, table, spec, query, context):
        out = []
        candidates = table if (query.operation == 'list' or spec.cardinality == 'multi') else None
        if candidates is None:
            assessed = [g for g in table if g['support'] is not None]
            if not assessed:
                return [self._result(g, 'unassessed', context, [], 0.0) for g in table]
            top_rank = max(g['rank'] for g in assessed)
            top = sorted([g for g in assessed if g['rank'] == top_rank], key=lambda g: (-g['support'], g['assertion']))
            candidates = top[:1]
        for group in candidates:
            competitors, opposing = [], []
            for other in table:
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
            out.append(self._result(group, status, context, competitors, opposition))
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
    def check_claim(self, proposal, *, as_of=None, context=None, validity='filter'):
        """Deterministically compare an external assertion with the ledger. No model call.

        supported: an accepted ledger value in the same context equals it.
        contradicted: an accepted value in the same context is incompatible with it.
        not_supported: the ledger holds the entity/predicate only in other contexts
                       (e.g. the assertion drops or inverts a condition, or changes the kind).
        not_representable: the assertion violates the vocabulary (e.g. unit dimension).
        """
        try:
            proposition = self.vocabulary.proposition(proposal)
        except UnmappedProposal as exc:
            return self._issue({'operation': 'check_claim', 'status': 'not_representable', 'proposal': proposal,
                                'reason': exc.as_dict(), 'configuration': self.configuration})
        query = {'entity': proposition['entity'], 'predicate': proposition['predicate'], 'operation': 'list',
                 'as_of': as_of, 'context': dict(context or {})}
        response = self.ask(query, conditions='report', validity=validity,
                            assertion_kinds=tuple(sorted(set(self.policy.allowed_assertion_kinds) | {proposition['assertion_kind']})))
        target_context = json.loads(context_key(proposition))
        same = [r for r in response['results'] if r['context'] == target_context]
        claim_value = {'value': proposition['value'], 'unit': proposition['unit'], 'polarity': proposition['polarity']}
        spec = self.vocabulary.predicate(proposition['predicate'])
        status, matched, differences = 'insufficient_evidence', None, []
        accepted = [r for r in same if r['status'] == 'accepted']
        for result in accepted:
            same_value = canonical({'value': result['value'], 'unit': result['unit'], 'polarity': result['polarity']}) == canonical(claim_value)
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
                kind = self._relation(spec, a, b) if self._incompatible(spec, a, b) else 'compatible'
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
        return self._issue({'operation': 'check_claim', 'status': status, 'proposition': proposition,
                            'matched_result': matched, 'differences': differences, 'relation': relation,
                            'ledger_response_id': response['response_id'], 'configuration': self.configuration,
                            'decision_kind': 'deterministic comparison with ledger; truth not certified'})

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
        for key in ('assertion_kind', 'scope', 'conditions'):
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
