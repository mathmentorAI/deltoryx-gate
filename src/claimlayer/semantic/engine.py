"""Semantic ingestion and snapshot resolution. Queries never call a model.

Language interpretation belongs to injected readers/verifiers. This module uses
only typed contracts, source coordinates, declared policies and stored verdicts.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
import copy
from decimal import Decimal
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping
from urllib.parse import urlsplit
import uuid

from .models import (CONTRACT_VERSION, KINDS, VERDICTS, CandidateClaim, EvidenceRef,
                     SemanticPolicy, SourceDocument, StructuredQuery, canonical, digest, instant)
from .store import VerificationStore


class SemanticContractError(ValueError):
    pass


def _identity(adapter) -> dict[str, Any]:
    value = copy.deepcopy(dict(adapter.identity))
    if not {'provider', 'model', 'prompt_version'} <= set(value):
        raise SemanticContractError('Adapters must identify provider, model and prompt_version')
    if set(value) - {'provider', 'model', 'prompt_version', 'contract_version', 'endpoint', 'parameters'}:
        raise SemanticContractError('Unexpected adapter identity field')
    for key in ('provider', 'model', 'prompt_version'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise SemanticContractError('Invalid adapter identity')
    if 'endpoint' in value:
        url = urlsplit(value['endpoint'])
        if url.username or url.password or url.query or url.fragment:
            raise SemanticContractError('Credentials must not appear in identity URLs')
    def inspect(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if str(key).lower() in {'api_key', 'authorization', 'password', 'secret', 'access_token'}:
                    raise SemanticContractError('Credentials must not appear in persisted identity')
                inspect(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                inspect(item)
    inspect(value)
    canonical(value)
    return value


def _noisy_or(values):
    remaining = 1.0
    for value in values:
        remaining *= 1.0 - value
    return round(1.0 - remaining, 6)


class ClaimLayer:
    """Provider-neutral semantic ledger (v3 preview).

    Default policy observes model assessments. Set SemanticPolicy(mode='strict')
    explicitly to resolve semantically supported claims. No model means no
    guessed parsing and no silently accepted natural-language evidence.
    """
    def __init__(self, db_path=None, *, project_id='default', semantic_reader=None,
                 evidence_verifier=None, policy=None, rule_checkers=None, transitive_relations=(),
                 require_quote_alignment=None):
        if require_quote_alignment is not None:
            raise ValueError('require_quote_alignment is a v2 contract; use SemanticPolicy explicitly')
        self._tmpdir = tempfile.TemporaryDirectory() if db_path is None else None
        path = Path(self._tmpdir.name) / 'semantic.db' if db_path is None else Path(db_path)
        self.store = VerificationStore(path, project_id)
        self.reader = semantic_reader
        self.verifier = evidence_verifier
        self.policy = SemanticPolicy(**asdict(policy or SemanticPolicy()))
        self._rules = {'identity': ('1', lambda c, ps: len(ps) == 1 and canonical(c.proposition()) == canonical(ps[0].proposition()))}
        for name, specification in (rule_checkers or {}).items():
            version, checker = specification
            if not isinstance(version, str) or not version or not callable(checker):
                raise ValueError('Rule checkers need an explicit version and trusted callable')
            if name == 'identity':
                raise ValueError('The built-in identity rule cannot be replaced')
            self._rules[name] = (version, checker)
        from .multihop import TransitiveRelation
        self._transitive_relations = {}
        for relation in transitive_relations:
            if not isinstance(relation, TransitiveRelation):
                raise ValueError('Use explicit application TransitiveRelation specifications')
            if relation.predicate in self._transitive_relations or relation.rule_name in self._rules:
                raise ValueError('Duplicate or conflicting transitive rule')
            self._transitive_relations[relation.predicate] = relation
            self._rules[relation.rule_name] = (relation.version, relation.check)
        self.migration_required = self.store.legacy_present() and not self.store.heads('document')

    @property
    def core_claims(self):
        return [CandidateClaim.from_dict(r['claim']) for _, _, r in self.store.heads('claim')]

    def _document(self, document_id, revision=None, *, current=True):
        active, _ = self.store.head('document', document_id)
        revision = revision or active
        record = self.store.get('document', revision) if revision else None
        if record is None or record['document_id'] != document_id or (current and active != revision):
            raise SemanticContractError('Missing or stale source revision')
        doc = SourceDocument(**record)
        if doc.revision != revision:
            raise SemanticContractError('Source identity mismatch')
        return doc

    def _claim_record(self, claim_id):
        _, record = self.store.head('claim', claim_id)
        if record is None:
            raise KeyError(claim_id)
        return record

    def _refs_valid(self, refs, bundle=None):
        if not refs:
            raise SemanticContractError('Source claims need exact evidence references')
        for ref in refs:
            doc = self._document(ref.document_id, ref.revision)
            if ref.end > len(doc.text) or doc.text[ref.start:ref.end] != ref.quote:
                raise SemanticContractError('Quote and offsets do not match the stored source')
            if bundle is not None and not any(
                f['document_id'] == ref.document_id and f['revision'] == ref.revision
                and f['start'] <= ref.start < ref.end <= f['end'] for f in bundle['fragments']
            ):
                raise SemanticContractError('A cited span was not supplied to the verifier')

    def _call(self, adapter, method, request):
        request = json.loads(canonical(request))
        record = {'request': request, 'created_at': datetime.now(timezone.utc).isoformat(),
                  'event_id': uuid.uuid4().hex}
        try:
            response = getattr(adapter, method)(copy.deepcopy(request))
            if type(response) is not dict:
                raise SemanticContractError('Adapter response must be an object')
            canonical(response)
            record['response'] = copy.deepcopy(response)
            output = response.get('output', response)
            if type(output) is not dict:
                raise SemanticContractError('Adapter output must be an object')
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            diagnostics = getattr(exc, 'diagnostics', None)
            if diagnostics is not None:
                record['error_diagnostics'] = copy.deepcopy(diagnostics)
            self.store.put('event', record['event_id'], record)
            return None, record
        self.store.put('event', record['event_id'], record)
        return output, record

    def ingest(self, documents):
        if isinstance(documents, (str, SourceDocument)):
            documents = [documents]
        summary = {'documents_ingested': 0, 'claims_extracted': 0, 'claims_rejected': 0,
                   'unverified_documents': [], 'errors': []}
        for doc in documents:
            if isinstance(doc, str):
                doc = SourceDocument('text-' + digest(doc), doc)
            if not isinstance(doc, SourceDocument):
                raise TypeError('Use SourceDocument or text; typed claims use ingest_claims()')
            self.store.put('document', doc.revision, asdict(doc), head=doc.document_id)
            summary['documents_ingested'] += 1
            if self.reader is None:
                summary['unverified_documents'].append(doc.document_id)
                continue
            if len(doc.text) > self.policy.max_context_chars:
                summary['errors'].append({'document_id': doc.document_id, 'error_type': 'ContextBudgetExceeded'})
                summary['unverified_documents'].append(doc.document_id)
                continue
            request = {'task': 'extract', 'contract_version': CONTRACT_VERSION,
                       'adapter': _identity(self.reader), 'document': asdict(doc),
                       'revision': doc.revision, 'document_span': {'start': 0, 'end': len(doc.text)},
                       'max_claims': self.policy.max_claims_per_document}
            request_key = digest(request)
            _, cached = self.store.head('read', request_key)
            if cached:
                output = cached['output']
            else:
                output, event = self._call(self.reader, 'extract', request)
                if output is None:
                    summary['errors'].append({'document_id': doc.document_id, 'error_type': event['error_type']})
                    summary['unverified_documents'].append(doc.document_id)
                    continue
            try:
                if set(output) != {'claims'} or type(output['claims']) is not list or len(output['claims']) > self.policy.max_claims_per_document:
                    raise SemanticContractError('Invalid extraction schema or count')
                candidates = []
                for proposal in output['claims']:
                    if type(proposal) is not dict or set(proposal) & {'claim_id', 'premises', 'derivation_rule'}:
                        raise SemanticContractError('A reader cannot assign IDs or proof rules')
                    candidate = CandidateClaim.from_dict({'claim_id': 'read-' + digest(proposal), **proposal})
                    self._refs_valid(candidate.evidence)
                    if any(r.document_id != doc.document_id or r.revision != doc.revision for r in candidate.evidence):
                        raise SemanticContractError('Reader cited evidence outside its source')
                    candidates.append(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                summary['errors'].append({'document_id': doc.document_id, 'error_type': type(exc).__name__})
                summary['unverified_documents'].append(doc.document_id)
                continue
            if not cached:
                self.store.put('read', request_key, {'request': request, 'output': output}, head=request_key)
            ingested = self.ingest_claims(candidates)
            summary['claims_extracted'] += ingested['claims_extracted']
            summary['claims_rejected'] += ingested['claims_rejected']
        return summary

    def migrate_legacy_sources(self, sources):
        """Explicitly reingest original sources; never promote old scores or verdicts.

        The v2 ledger stores excerpts, not complete originals. The caller must
        supply a SourceDocument for every legacy filename in this project.
        Authority and dependence metadata remain application decisions.
        """
        if not self.store.legacy_present():
            raise ValueError('No legacy project to migrate')
        sources = list(sources)
        with self.store.connect() as db:
            expected = {r[0] for r in db.execute(
                'SELECT filename FROM documents WHERE project_id=?', (self.store.project_id,))}
        if (any(not isinstance(s, SourceDocument) for s in sources)
                or len({s.document_id for s in sources}) != len(sources)
                or {s.document_id for s in sources} != expected):
            raise ValueError('Supply original SourceDocuments using every legacy filename as document_id')
        result = self.ingest(sources)
        self.migration_required = False
        return {**result, 'legacy_assessments_imported': 0,
                'migration': 'original_sources_reingested; legacy ledger preserved'}

    def ingest_claims(self, claims, *, verify=True):
        counts = {'claims_extracted': 0, 'claims_rejected': 0}
        for claim in claims:
            if not isinstance(claim, CandidateClaim):
                raise TypeError('Expected CandidateClaim')
            if claim.premises or claim.derivation_rule:
                raise ValueError('Formal conclusions must use derive()')
            self._refs_valid(claim.evidence)
            old_id, old = self.store.head('claim', claim.claim_id)
            if old is not None and old['claim'] != json.loads(canonical(asdict(claim))):
                raise ValueError('Claim IDs are immutable; submit a new ID for a revision')
            record = {'claim': asdict(claim), 'origin': 'source'}
            self.store.put('claim', claim.fingerprint, record, head=claim.claim_id)
            counts['claims_extracted'] += 1
            if verify:
                assessment = self.verify_claim(claim.claim_id)
                if assessment['status'] in {'contradicted', 'invalid_response', 'error'}:
                    counts['claims_rejected'] += 1
        return counts

    def _bundle(self, claim, related, extra=()):
        allowed = sorted({r.document_id for r in claim.evidence} | set(related))
        docs = {name: self._document(name) for name in allowed}
        intervals = {name: [] for name in docs}
        for ref in claim.evidence:
            self._refs_valid([ref])
            intervals[ref.document_id].append((max(0, ref.start - 1000), min(len(docs[ref.document_id].text), ref.end + 1000)))
        for req in extra:
            if set(req) != {'document_id', 'start', 'end'} or req['document_id'] not in docs:
                raise SemanticContractError('Invalid or unauthorized context request')
            start, end = req['start'], req['end']
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(docs[req['document_id']].text):
                raise SemanticContractError('Invalid context interval')
            intervals[req['document_id']].append((start, end))
        for name, doc in docs.items():
            if not intervals[name]:
                intervals[name].append((0, min(len(doc.text), 2000)))
        fragments = []
        for name in allowed:
            merged = []
            for start, end in sorted(intervals[name]):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            for start, end in merged:
                doc = docs[name]
                fragments.append({'document_id': name, 'revision': doc.revision, 'start': start,
                                  'end': end, 'text': doc.text[start:end], 'attribution': doc.attribution})
        size = sum(len(f['text']) for f in fragments)
        if size > self.policy.max_context_chars:
            raise SemanticContractError('Requested context exceeds the declared budget')
        return {'fragments': fragments, 'allowed_documents': [
            {'document_id': name, 'revision': doc.revision, 'characters': len(doc.text)} for name, doc in docs.items()],
            'omitted_characters': sum(len(d.text) for d in docs.values()) - size,
            'max_context_chars': self.policy.max_context_chars}

    def verify_claim(self, claim_id, *, related_document_ids=(), force=False):
        claim = CandidateClaim.from_dict(self._claim_record(claim_id)['claim'])
        if claim.premises:
            raise ValueError('Derived claims use trusted proof rules, not model verification')
        if self.verifier is None:
            return {'status': 'unverified', 'reason': 'verifier_not_configured'}
        extra = []
        attempts = []
        for round_index in range(self.policy.max_context_rounds + 1):
            try:
                bundle = self._bundle(claim, related_document_ids, extra)
                request = {'task': 'verify', 'contract_version': CONTRACT_VERSION,
                           'context_progress_version': 'v1',
                           'adapter': _identity(self.verifier), 'policy': asdict(self.policy),
                           'claim': asdict(claim), 'bundle': bundle}
                request_key = digest(request)
                _, cached = self.store.head('evaluation_cache', request_key)
                cached_record = self.store.get('evaluation', cached['evaluation_id']) if cached else None
                if cached_record and cached_record['status'] not in {'error', 'invalid_response'} and not force:
                    record = cached_record
                    self._activate_evaluation(claim_id, record)
                    if (record['status'] == 'insufficient_evidence'
                            and record.get('context_progress', {}).get('action') == 'expand'
                            and round_index < self.policy.max_context_rounds):
                        extra.extend(record['context_progress']['new_requests'])
                        attempts.append(record['evaluation_id'])
                        continue
                    return record
                output, event = self._call(self.verifier, 'verify', request)
                if output is None:
                    record = {'status': 'error', 'reason': event['error_type'], 'output': None}
                else:
                    try:
                        status = self._validate_verdict(output, claim, bundle)
                        record = {'status': status, 'output': output}
                        if status == 'insufficient_evidence' and output.get('context_requests'):
                            record['context_progress'] = self._context_progress(output['context_requests'], bundle)
                            if record['context_progress']['action'] == 'stop':
                                record['reason'] = 'context_already_supplied'
                            elif round_index >= self.policy.max_context_rounds:
                                record['context_progress']['round_limit_reached'] = True
                    except (KeyError, TypeError, ValueError) as exc:
                        record = {'status': 'invalid_response', 'reason': type(exc).__name__, 'output': output}
                record.update({'request': request, 'request_key': request_key,
                               'evaluation_id': uuid.uuid4().hex, 'event_id': event['event_id'],
                               'previous_attempts': list(attempts)})
            except (KeyError, TypeError, ValueError) as exc:
                record = {'status': 'insufficient_evidence', 'reason': type(exc).__name__,
                          'context_error': {'stage': 'bundle', 'detail': str(exc)},
                          'output': None, 'claim_fingerprint': claim.fingerprint,
                          'evaluation_id': uuid.uuid4().hex, 'previous_attempts': list(attempts)}
            self.store.put('evaluation', record['evaluation_id'], record)
            self._activate_evaluation(claim_id, record)
            if 'request_key' in record:
                self.store.put('evaluation_cache', record['evaluation_id'], {'evaluation_id': record['evaluation_id']}, head=record['request_key'])
            if (record['status'] == 'insufficient_evidence' and record['output']
                    and record.get('context_progress', {}).get('action') == 'expand'
                    and round_index < self.policy.max_context_rounds):
                extra.extend(record['context_progress']['new_requests'])
                attempts.append(record['evaluation_id'])
                continue
            return record
        return record

    @staticmethod
    def _context_progress(requests, bundle):
        """Identify new evidence ranges, without judging the model's semantics."""
        new, supplied = [], []
        for request in requests:
            cursor = request['start']
            spans = sorted((f['start'], f['end']) for f in bundle['fragments']
                           if f['document_id'] == request['document_id'])
            for start, end in spans:
                if start > cursor:
                    break
                cursor = max(cursor, end)
            (supplied if cursor >= request['end'] else new).append(request)
        return {'action': 'expand' if new else 'stop', 'new_requests': new,
                'already_supplied': supplied,
                'meaning': 'Context delivery only; no semantic endorsement'}

    def _activate_evaluation(self, claim_id, record):
        self.store.put('assessment', record['evaluation_id'], {'evaluation_id': record['evaluation_id']}, head=claim_id)

    def _validate_verdict(self, output, claim, bundle):
        required = {'verdict', 'assertion_kind', 'scope_preserved', 'conditions_preserved',
                    'references', 'context_requests', 'rationale'}
        if 'subject_check' in output:
            required.add('subject_check')
        if 'proposition_check' in output:
            required.add('proposition_check')
        if set(output) != required or output['verdict'] not in VERDICTS or output['assertion_kind'] not in KINDS:
            raise SemanticContractError('Invalid verdict schema')
        if type(output['scope_preserved']) is not bool or type(output['conditions_preserved']) is not bool:
            raise SemanticContractError('Scope and condition flags must be booleans')
        if not isinstance(output['rationale'], str) or len(output['rationale']) > 4000:
            raise SemanticContractError('Rationale must be brief text')
        if type(output['references']) is not list or type(output['context_requests']) is not list or len(output['context_requests']) > 8:
            raise SemanticContractError('Invalid reference or context request list')
        refs = [EvidenceRef(**r) for r in output['references']]
        if refs:
            self._refs_valid(refs, bundle)
        subject_ok = True
        if 'subject_check' in output:
            check = output['subject_check']
            fields = {'claim_entity','claim_predicate','evidence_entity','entity_role',
                      'relationship','status','references','rationale'}
            if type(check) is not dict or set(check) != fields:
                raise SemanticContractError('Invalid subject check schema')
            if check['claim_entity'] != claim.entity or check['claim_predicate'] != claim.predicate:
                raise SemanticContractError('Subject check must evaluate the unchanged proposition')
            if check['status'] not in {'matched','mismatched','unclear'}:
                raise SemanticContractError('Invalid subject match status')
            for field in ('evidence_entity','entity_role'):
                if check[field] is not None and (type(check[field]) is not str or not check[field].strip() or len(check[field])>4000):
                    raise SemanticContractError('Invalid subject identity or role')
            for field in ('relationship','rationale'):
                if type(check[field]) is not str or not check[field].strip() or len(check[field])>4000:
                    raise SemanticContractError('Subject check requires brief relationship and rationale')
            if type(check['references']) is not list:
                raise SemanticContractError('Invalid subject evidence')
            subject_refs = [EvidenceRef(**r) for r in check['references']]
            if subject_refs:
                self._refs_valid(subject_refs, bundle)
            subject_ok = (check['status']=='matched' and bool(subject_refs)
                          and bool(check['evidence_entity']) and bool(check['entity_role']))
        proposition_ok = True
        if 'proposition_check' in output:
            from .proposition import compare
            proposition_ok = compare(claim, output['proposition_check'])
        if output['verdict'] == 'supported':
            if not refs:
                raise SemanticContractError('Support requires an exact supplied reference')
            if (not proposition_ok or not subject_ok or output['assertion_kind'] != claim.assertion_kind or not output['scope_preserved']
                    or not output['conditions_preserved'] or output['context_requests']):
                return 'insufficient_evidence'
        # Validate requested sources and integer ranges before a possible next call.
        docs = {d['document_id']: d for d in bundle['allowed_documents']}
        for req in output['context_requests']:
            if type(req) is not dict or set(req) != {'document_id', 'start', 'end'} or req['document_id'] not in docs:
                raise SemanticContractError('Unauthorized context request')
            if (type(req['start']) is not int or type(req['end']) is not int
                    or not 0 <= req['start'] < req['end'] <= docs[req['document_id']]['characters']):
                raise SemanticContractError('Invalid context range')
        return output['verdict']

    def derive(self, claim):
        if not isinstance(claim, CandidateClaim) or not claim.premises or claim.derivation_rule not in self._rules:
            raise ValueError('Explicit premises and an installed trusted rule are required')
        if claim.claim_id in claim.premises or self.store.head('claim', claim.claim_id)[1] is not None:
            raise ValueError('Cycles and replacing claim IDs are forbidden')
        premises = [CandidateClaim.from_dict(self._claim_record(pid)['claim']) for pid in claim.premises]
        version, checker = self._rules[claim.derivation_rule]
        try:
            valid = checker(claim, tuple(premises)) is True
        except Exception:
            valid = False
        if not valid:
            raise ValueError('Rule does not justify this proposition and scope')
        record = {'claim': asdict(claim), 'origin': 'rule_derived', 'rule_version': version}
        self.store.put('claim', claim.fingerprint, record, head=claim.claim_id)
        return {'status': 'rule_derived', 'claim_id': claim.claim_id}

    def derive_paths(self, entity, predicate, **options):
        from .multihop import derive_paths
        return derive_paths(self, entity, predicate, **options)

    def attest_claim(self, claim_id, *, attestor):
        """Explicit trusted-application input; never an LLM-authentication shortcut."""
        if not isinstance(attestor, str) or not attestor.strip():
            raise ValueError('An application attestor is required')
        claim = CandidateClaim.from_dict(self._claim_record(claim_id)['claim'])
        self._refs_valid(claim.evidence)
        record = {'evaluation_id': uuid.uuid4().hex, 'status': 'application_attested',
                  'claim_fingerprint': claim.fingerprint, 'attestor': attestor}
        self.store.put('evaluation', record['evaluation_id'], record)
        self._activate_evaluation(claim_id, record)
        return record

    def _condition(self, condition, context):
        field = condition['field']
        if field not in context:
            return False
        left, right, op = context[field], condition['value'], condition['operator']
        if op in {'eq', 'ne'}:
            equal = canonical(left) == canonical(right)
            return equal if op == 'eq' else not equal
        if type(left) not in (int, float) or type(right) not in (int, float):
            return False
        return {'lt': lambda: left < right, 'le': lambda: left <= right,
                'gt': lambda: left > right, 'ge': lambda: left >= right}[op]()

    def _eligible(self, claim_id, query, seen=None, *, as_premise=False, check_conflicts=True):
        seen = set() if seen is None else seen
        if claim_id in seen:
            return None, 'cyclic_dependency'
        record = self._claim_record(claim_id)
        claim = CandidateClaim.from_dict(record['claim'])
        if claim.assertion_kind not in self.policy.allowed_assertion_kinds:
            return None, 'assertion_kind_not_admitted'
        at = instant(query.as_of)
        start, end = instant(claim.valid_from), instant(claim.valid_until)
        if (start and at < start) or (end and at > end):
            return None, 'outside_validity'
        for key, value in claim.scope.items():
            if key == 'event_id' and query.operation in {'list', 'sum', 'count', 'min', 'max', 'average'}:
                continue
            if key not in query.context or canonical(query.context[key]) != canonical(value):
                return None, 'scope_not_discharged'
        if not all(self._condition(c, query.context) for c in claim.conditions):
            return None, 'condition_not_discharged'
        if record['origin'] == 'rule_derived':
            version_checker = self._rules.get(claim.derivation_rule)
            if not version_checker or version_checker[0] != record['rule_version']:
                return None, 'rule_version_unavailable'
            bases = []
            for pid in claim.premises:
                basis, reason = self._eligible(pid, query, seen | {claim_id},
                                               as_premise=True, check_conflicts=check_conflicts)
                if not basis:
                    return None, 'premise_' + reason
                bases.append(basis)
            try:
                valid = version_checker[1](claim, tuple(b['claim'] for b in bases)) is True
            except Exception:
                valid = False
            if not valid:
                return None, 'invalid_derivation'
            basis = {'claim': claim, 'documents': {k: v for b in bases for k, v in b['documents'].items()},
                     'evaluations': sorted({e for b in bases for e in b['evaluations']}), 'status': 'rule_derived'}
            if as_premise and check_conflicts:
                return self._check_premise_conflicts(basis, query, seen)
            return basis, None
        try:
            self._refs_valid(claim.evidence)
        except ValueError:
            return None, 'stale_evidence'
        _, assessment = self.store.head('assessment', claim_id)
        evaluation = self.store.get('evaluation', assessment['evaluation_id']) if assessment else None
        if evaluation is None:
            return None, 'unverified'
        if evaluation['status'] == 'application_attested':
            if evaluation['claim_fingerprint'] != claim.fingerprint:
                return None, 'stale_assessment'
        elif evaluation['status'] == 'supported':
            if self.policy.mode != 'strict':
                return None, 'semantic_observation_only'
            request = evaluation['request']
            if request['claim'] != json.loads(canonical(asdict(claim))) or request['policy'] != json.loads(canonical(asdict(self.policy))):
                return None, 'stale_assessment'
            for item in request['bundle']['allowed_documents']:
                try:
                    self._document(item['document_id'], item['revision'])
                except ValueError:
                    return None, 'stale_context'
            if self.verifier is not None and request['adapter'] != _identity(self.verifier):
                return None, 'verifier_version_changed'
        else:
            return None, evaluation['status']
        docs = {r.document_id: self._document(r.document_id, r.revision) for r in claim.evidence}
        # Attribution from the application cannot be promoted by generated cue words.
        if any(d.attribution not in self.policy.allowed_source_attributions for d in docs.values()):
            return None, 'source_attribution_not_admitted'
        basis = {'claim': claim, 'documents': docs, 'evaluations': [evaluation['evaluation_id']],
                 'status': evaluation['status']}
        if as_premise and check_conflicts:
            return self._check_premise_conflicts(basis, query, seen)
        return basis, None

    def _check_premise_conflicts(self, basis, query, seen):
        """Block contested proof inputs under declared single-value semantics.

        Ordinary source queries still expose alternatives. This conservative
        guard does not compute a full argumentation fixed point.
        """
        claim = basis['claim']
        single = self.policy.predicate_cardinality.get(claim.predicate) == 'single'
        rank = min(d.authority_rank for d in basis['documents'].values())
        contested = False
        for other in self.core_claims:
            if (other.claim_id == claim.claim_id or other.claim_id in seen
                    or other.entity != claim.entity or other.predicate != claim.predicate
                    or other.unit != claim.unit or other.scope != claim.scope
                    or other.conditions != claim.conditions
                    or other.assertion_kind != claim.assertion_kind):
                continue
            same_value = canonical(other.value) == canonical(claim.value)
            explicit_negation = same_value and other.polarity != claim.polarity
            if not explicit_negation and (not single or same_value):
                continue
            competitor, _ = self._eligible(other.claim_id, query, seen | {claim.claim_id},
                                           check_conflicts=False)
            if competitor is None:
                continue
            other_rank = min(d.authority_rank for d in competitor['documents'].values())
            if other_rank > rank:
                return None, 'defeated_by_authority'
            if other_rank == rank:
                contested = True
        return (None, 'contested_at_equal_authority') if contested else (basis, None)

    def prepare_query(self, query):
        if self.reader is None:
            return {'status': 'unverified', 'reason': 'query_reader_not_configured'}
        request = {'task': 'parse_query', 'contract_version': CONTRACT_VERSION,
                   'adapter': _identity(self.reader), 'query': query,
                   'entities': sorted({c.entity for c in self.core_claims}),
                   'predicates': sorted({c.predicate for c in self.core_claims})}
        key = digest(request)
        _, cached = self.store.head('prepared_query', key)
        if cached is None:
            output, event = self._call(self.reader, 'parse_query', request)
            if output is None:
                return {'status': 'error', 'reason': event['error_type']}
            try:
                if output == {'insufficient': True}:
                    return {'status': 'insufficient_evidence'}
                parsed = StructuredQuery(**output)
                if parsed.entity not in request['entities'] or parsed.predicate not in request['predicates']:
                    raise SemanticContractError('Query invented a ledger entity or predicate')
            except (ValueError, TypeError):
                return {'status': 'invalid_response'}
            # Model-inferred condition values are a proposal, not trusted runtime facts.
            cached = {'request': request, 'query': asdict(replace(parsed, context={})),
                      'proposed_context': parsed.context, 'event_id': event['event_id']}
            self.store.put('prepared_query', key, cached, head=key)
        self.store.put('query_head', key, {'prepared_key': key}, head=query)
        return {'status': 'prepared', 'query': cached['query'], 'proposed_context': cached['proposed_context']}

    def _support(self, bases):
        selected, used = [], set()
        weighted = []
        for basis in bases:
            docs = list(basis['documents'].values())
            dependencies = {d.dependence_group or digest(d.text) for d in docs}
            weighted.append((min((d.weight for d in docs), default=0.0), basis['claim'].claim_id, dependencies, basis))
        for weight, cid, dependencies, basis in sorted(weighted, key=lambda x: (-x[0], x[1])):
            if not dependencies & used:
                used |= dependencies
                selected.append((weight, basis))
        return _noisy_or([w for w, _ in selected]), [b['claim'].claim_id for _, b in selected]

    def ask(self, query, *, context=None, as_of=None):
        if isinstance(query, str):
            _, head = self.store.head('query_head', query)
            if head is None:
                return self._issue({'query': query, 'results': [], 'abstain': True,
                                    'reason': 'query_not_prepared', 'migration_required': self.migration_required})
            prepared = self.store.get('prepared_query', head['prepared_key'])
            if (prepared['request']['entities'] != sorted({c.entity for c in self.core_claims})
                    or prepared['request']['predicates'] != sorted({c.predicate for c in self.core_claims})
                    or (self.reader is not None and prepared['request']['adapter'] != _identity(self.reader))):
                return self._issue({'query': query, 'results': [], 'abstain': True, 'reason': 'prepared_query_stale'})
            query = StructuredQuery(**prepared['query'])
        elif isinstance(query, dict):
            query = StructuredQuery(**query)
        if not isinstance(query, StructuredQuery):
            raise TypeError('Use StructuredQuery, its dictionary, or explicitly prepare natural-language queries')
        query = replace(query, context=copy.deepcopy(query.context if context is None else context),
                        as_of=as_of or query.as_of or datetime.now(timezone.utc).isoformat())
        candidates = [c for c in self.core_claims if c.entity == query.entity and c.predicate == query.predicate]
        bases, excluded = [], []
        for claim in candidates:
            basis, reason = self._eligible(claim.claim_id, query)
            if basis:
                bases.append(basis)
            else:
                excluded.append({'claim_id': claim.claim_id, 'reason': reason})
        results, reason = [], 'no_verified_evidence'
        if bases:
            if query.operation in {'sum', 'count', 'min', 'max', 'average'}:
                results, reason = self._aggregate(bases, query)
            else:
                results = self._resolve(bases, query)
                reason = 'resolved'
        basis_set = set()
        def add_ancestors(cid):
            if cid in basis_set:
                return
            basis_set.add(cid)
            for parent in CandidateClaim.from_dict(self._claim_record(cid)['claim']).premises:
                add_ancestors(parent)
        for basis in bases:
            add_ancestors(basis['claim'].claim_id)
        basis_ids = sorted(basis_set)
        evaluations = sorted({e for b in bases for e in b['evaluations']})
        source_revisions = {d.revision for b in bases for d in b['documents'].values()}
        evaluation_digests = {}
        for evaluation_id in evaluations:
            record = self.store.get('evaluation', evaluation_id)
            evaluation_digests[evaluation_id] = digest(record)
            source_revisions.update(item['revision'] for item in
                record.get('request', {}).get('bundle', {}).get('allowed_documents', []))
        source_revisions = sorted(source_revisions)
        return self._issue({'query': asdict(query), 'results': results, 'abstain': not bool(results),
                            'reason': reason, 'excluded': excluded, 'policy': asdict(self.policy),
                            'basis': {'claims': basis_ids, 'evaluations': evaluations, 'source_revisions': source_revisions,
                                      'claim_record_digests': {cid: digest(self._claim_record(cid)) for cid in basis_ids},
                                      'evaluation_digests': evaluation_digests},
                            'source_authentication': 'not_verified', 'confidence_kind': 'evidence_support_not_probability'})

    def _resolve(self, bases, query):
        grouped = {}
        for basis in bases:
            proposition = basis['claim'].proposition()
            proposition.pop('valid_from')
            proposition.pop('valid_until')
            grouped.setdefault(canonical(proposition), []).append(basis)
        groups = []
        for key, members in grouped.items():
            score, selected = self._support(members)
            rank = max(min(d.authority_rank for d in b['documents'].values()) for b in members)
            groups.append({'key': key, 'members': members, 'support': score, 'rank': rank, 'selected': selected})
        groups.sort(key=lambda g: (-g['rank'], -g['support'], g['key']))
        cardinality = self.policy.predicate_cardinality.get(query.predicate, 'unknown')
        chosen = groups if query.operation == 'list' or cardinality == 'multi' else groups[:1]
        outputs = []
        for winner in chosen:
            rep = winner['members'][0]['claim']
            contradictions, compatible, unresolved, opposing = [], [], [], {}
            for other in groups:
                if other is winner:
                    continue
                candidate = other['members'][0]['claim']
                entry = {'value': candidate.value, 'unit': candidate.unit, 'polarity': candidate.polarity,
                         'claim_ids': [b['claim'].claim_id for b in other['members']]}
                comparable = (cardinality == 'single' and candidate.unit == rep.unit
                              and candidate.scope == rep.scope and candidate.conditions == rep.conditions
                              and candidate.assertion_kind == rep.assertion_kind
                              and bool(candidate.valid_from or candidate.valid_until) == bool(rep.valid_from or rep.valid_until))
                if cardinality == 'multi':
                    compatible.append(entry)
                elif comparable:
                    contradictions.append(entry)
                    attenuation = min(1.0, other['rank'] / winner['rank']) if winner['rank'] else 1.0
                    for b in other['members']:
                        for d in b['documents'].values():
                            dep = d.dependence_group or digest(d.text)
                            opposing[dep] = max(opposing.get(dep, 0.0), d.weight * attenuation)
                else:
                    unresolved.append(entry)
            opposition = _noisy_or(list(opposing.values()))
            score = winner['support'] ** 2 / (winner['support'] + opposition) if winner['support'] + opposition else 0.0
            outputs.append({'entity': rep.entity, 'predicate': rep.predicate, 'value': rep.value,
                            'unit': rep.unit, 'polarity': rep.polarity, 'assertion_kind': rep.assertion_kind,
                            'support_score': round(score, 6), 'settled': not contradictions and not unresolved,
                            'contradictions': contradictions, 'compatible_values': compatible,
                            'unresolved_comparisons': unresolved, 'selected_claim_ids': winner['selected'],
                            'supporting_evidence': [{'claim': asdict(b['claim']), 'verification_status': b['status'],
                                'evaluation_ids': b['evaluations']} for b in winner['members']]})
        return outputs

    def _aggregate(self, bases, query):
        events = {}
        for basis in bases:
            event = basis['claim'].scope.get('event_id')
            if not isinstance(event, str) or not event:
                return [], 'event_identity_required'
            events.setdefault(event, []).append(basis)
        selected = []
        for event, members in events.items():
            # Aggregation never chooses a conflicting amount by linguistic guesswork.
            values = {canonical({'value': b['claim'].value, 'unit': b['claim'].unit,
                                 'polarity': b['claim'].polarity}) for b in members}
            if len(values) != 1:
                return [], 'unresolved_event_values'
            selected.append(members[0]['claim'])
        if len({c.unit for c in selected}) != 1 or any(c.polarity != 'affirmed' for c in selected):
            return [], 'incompatible_units_or_polarity'
        if query.operation == 'count':
            value, unit = len(selected), 'count'
        else:
            if any(type(c.value) not in (int, float) for c in selected):
                return [], 'numeric_values_required'
            values = [Decimal(str(c.value)) for c in selected]
            if query.operation == 'sum':
                value = sum(values)
            elif query.operation == 'min':
                value = min(values)
            elif query.operation == 'max':
                value = max(values)
            else:
                value = sum(values) / len(values)
            value = int(value) if value == value.to_integral_value() else float(value)
            unit = selected[0].unit
        return [{'entity': query.entity, 'predicate': query.predicate, 'value': value, 'unit': unit,
                 'operation': query.operation, 'event_ids': sorted(events), 'settled': True}], 'resolved'

    def _issue(self, payload):
        payload = json.loads(canonical(payload))
        certificate = {'scheme': 'CLAIMLAYER-SEMANTIC-INTEGRITY-v1', 'scope': 'integrity',
                       'payload_digest': digest(payload)}
        response = {**payload, 'certificate': certificate}
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
            for cid, fingerprint in basis.get('claim_record_digests', {}).items():
                if digest(self._claim_record(cid)) != fingerprint:
                    return False
            for eid, fingerprint in basis.get('evaluation_digests', {}).items():
                if digest(self.store.get('evaluation', eid)) != fingerprint:
                    return False
            for revision in basis.get('source_revisions', []):
                source = self.store.get('document', revision)
                if source is None or SourceDocument(**source).revision != revision:
                    return False
            return True
        except (ValueError, TypeError, KeyError):
            return False

    def assessment_history(self, claim_id):
        """Return all stored assessments for this immutable claim, including failures."""
        fingerprint = CandidateClaim.from_dict(self._claim_record(claim_id)['claim']).fingerprint
        with self.store.connect() as db:
            ids = [r[0] for r in db.execute('SELECT record_id FROM semantic_records WHERE project_id=? AND kind=? ORDER BY rowid',
                                          (self.store.project_id, 'evaluation'))]
        records = [self.store.get('evaluation', rid) for rid in ids]
        return [r for r in records if r.get('claim_fingerprint') == fingerprint
                or r.get('request', {}).get('claim', {}).get('claim_id') == claim_id]
