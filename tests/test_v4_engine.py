"""v4 engine contracts on application-attested units (no model involved)."""
import pytest

from claimlayer.v4 import (AcceptancePolicy, DEFAULT_PROFILE, EvidenceLayer, EvidenceRef, FieldSpec, PredicateSpec,
                           SourceDocument, TIER_ANNEX, TIER_EXTERNAL, TIER_PRIMARY, TransitiveRule, Vocabulary)
from claimlayer.v4.epistemics import BLOCKED_INFERENCE_TYPES, INFERENCE_WEIGHTS

VOCAB = Vocabulary('test', '1', (
    PredicateSpec('amount', FieldSpec('quantity', dimensions=('currency:EUR', 'currency:USD'))),
    PredicateSpec('payment_term', FieldSpec('quantity', dimensions=('duration_calendar_days', 'duration_business_days'))),
    PredicateSpec('term', FieldSpec('quantity', dimensions=('duration_months',))),
    PredicateSpec('office', FieldSpec('string'), cardinality='multi'),
    PredicateSpec('overtime_multiplier', FieldSpec('quantity', dimensions=('multiplier',)),
                  conditions=(('employee_is_nonexempt', FieldSpec('boolean')),
                              ('hours_worked', FieldSpec('quantity', dimensions=('duration_hours',))))),
    PredicateSpec('subsidiary_of', FieldSpec('entity')),
    PredicateSpec('director', FieldSpec('entity')),
))


class Ledger:
    def __init__(self, tmp_path=None, **kwargs):
        self.e = EvidenceLayer(None if tmp_path is None else tmp_path / 'v4.db', vocabulary=VOCAB, **kwargs)
        self.n = 0

    def add(self, predicate, value, *, entity='ACME', unit=None, rank=TIER_PRIMARY, weight=0.9, group=None,
            basis='explicit', kind='fact', conditions=(), scope=None, valid_from=None, valid_until=None,
            structural_reliability=1.0, doc=None):
        self.n += 1
        text = f'Record {self.n}: {entity} {predicate} {value} {unit or ""}.'
        document = SourceDocument(doc or f'doc-{self.n}.txt', text, authority_rank=rank, weight=weight, dependence_group=group)
        self.e.ingest(document)
        proposal = {'entity': entity, 'predicate': predicate, 'value': value, 'unit': unit, 'assertion_kind': kind,
                    'conditions': list(conditions), 'scope': scope or {}, 'valid_from': valid_from, 'valid_until': valid_until}
        out = self.e.ingest_units([{'proposal': proposal, 'basis': basis,
                                    'evidence': [EvidenceRef(document.document_id, document.revision, 0, len(text), text)]}],
                                  attestor='test-fixture', structural_reliability=structural_reliability)
        return out['unit_ids'][0], out


def ask(e, predicate, entity='ACME', **kw):
    return e.ask({'entity': entity, 'predicate': predicate, 'as_of': kw.pop('as_of', '2026-01-01'),
                  'operation': kw.pop('operation', 'resolve'), 'context': kw.pop('context', {})}, **kw)


def test_epistemic_table_is_the_v2_registry():
    for basis, weight in INFERENCE_WEIGHTS.items():
        if basis not in BLOCKED_INFERENCE_TYPES:
            assert DEFAULT_PROFILE.weight(basis) == weight
    assert DEFAULT_PROFILE.weight('explicit') == 1.0
    assert all(not DEFAULT_PROFILE.admissible(b) for b in BLOCKED_INFERENCE_TYPES)


def test_blocked_basis_never_enters_the_ledger():
    l = Ledger()
    unit_id, out = l.add('payment_term', 30, unit='days', basis='narrative')
    assert unit_id is None and out['rejection_reasons'] == {'basis_blocked': 1}
    r = ask(l.e, 'payment_term')
    assert r['abstain'] and r['coverage']['rejected_proposals'] == 1


def test_support_index_is_min_of_components_and_basis_weight():
    l = Ledger()
    l.add('payment_term', 30, unit='days', weight=0.9, basis='common_sense')
    result = ask(l.e, 'payment_term')['results'][0]
    assert result['evidence'][0]['components'] == {'span': 1.0, 'structure': 1.0, 'applicability': 0.9, 'basis': 0.65}
    assert result['support']['S'] == 0.65


def test_dependence_groups_max_within_noisy_or_across():
    l = Ledger()
    l.add('payment_term', 30, unit='days', weight=0.8, group='seller-copy')
    l.add('payment_term', 30, unit='days', weight=0.7, group='seller-copy')
    l.add('payment_term', 30, unit='days', weight=0.5, group='independent')
    s = ask(l.e, 'payment_term')['results'][0]['support']['S']
    assert s == pytest.approx(1 - (1 - 0.8) * (1 - 0.5))


def test_paper_q_index_and_margin_rule_for_equal_support():
    l = Ledger()
    l.add('payment_term', 30, unit='days', weight=1.0)
    l.add('payment_term', 45, unit='days', weight=1.0)
    result = ask(l.e, 'payment_term')
    assert result['status'] == 'unresolved_conflict' and result['abstain']
    support = result['results'][0]['support']
    assert support['S'] == support['O'] == 1.0 and support['Q'] == 0.5 and support['margin'] == 0


def test_explicit_fact_beats_weak_inference_at_equal_authority():
    l = Ledger()
    l.add('payment_term', 30, unit='days', weight=0.9, basis='explicit')
    l.add('payment_term', 45, unit='days', weight=0.9, basis='common_sense')
    result = ask(l.e, 'payment_term')
    top = result['results'][0]
    assert result['status'] == 'answered' and top['value']['value'] == '30'
    assert top['support']['O'] == 0.65 and top['competitors'][0]['relation'] == 'opposing'


@pytest.mark.parametrize('rank,expected', [(TIER_EXTERNAL, '30'), (TIER_ANNEX, '30')])
def test_authority_is_declared_by_application_and_defeats_lower_tiers(rank, expected):
    l = Ledger()
    l.add('payment_term', 30, unit='days', rank=TIER_PRIMARY, doc='annex-injected-note.txt')
    l.add('payment_term', 90, unit='days', rank=rank, doc='primary-contract.txt')
    result = ask(l.e, 'payment_term')
    assert result['status'] == 'answered' and result['results'][0]['value']['value'] == expected
    assert result['results'][0]['competitors'][0]['relation'] == 'defeated_by_authority'


def test_multivalued_predicates_keep_compatible_values():
    l = Ledger()
    l.add('office', 'Madrid')
    l.add('office', 'Barcelona')
    result = ask(l.e, 'office')
    assert sorted(r['value'] for r in result['results'] if r['status'] == 'accepted') == ['Barcelona', 'Madrid']
    assert all(c['relation'] == 'compatible' for r in result['results'] for c in r['competitors'])


def test_explicit_negation_conflicts_even_for_multivalued():
    l = Ledger()
    l.add('office', 'Madrid')
    doc = SourceDocument('neg.txt', 'ACME has no office in Madrid.', authority_rank=TIER_PRIMARY, weight=0.9)
    l.e.ingest(doc)
    l.e.ingest_units([{'proposal': {'entity': 'ACME', 'predicate': 'office', 'value': 'Madrid', 'polarity': 'negated',
                                    'assertion_kind': 'fact'},
                       'evidence': [EvidenceRef(doc.document_id, doc.revision, 0, len(doc.text), doc.text)]}],
                     attestor='t')
    statuses = {r['polarity']: r['status'] for r in ask(l.e, 'office')['results']}
    assert statuses == {'affirmed': 'unresolved_conflict', 'negated': 'unresolved_conflict'}


def test_different_dimensions_are_unresolved_not_contradictory():
    l = Ledger()
    l.add('payment_term', 45, unit='business days')
    l.add('payment_term', 45, unit='calendar days')
    result = ask(l.e, 'payment_term')
    assert result['status'] == 'unresolved_conflict'
    assert result['results'][0]['status'] == 'unresolved_comparison'


def test_disjoint_validity_is_succession_not_conflict():
    l = Ledger()
    l.add('payment_term', 30, unit='days', valid_from='2023-01-01', valid_until='2023-12-31')
    l.add('payment_term', 45, unit='days', valid_from='2024-01-01', valid_until='2030-12-31')
    assert ask(l.e, 'payment_term', as_of='2023-06-01')['results'][0]['value']['value'] == '30'
    current = ask(l.e, 'payment_term', as_of='2026-01-01')
    assert current['status'] == 'answered' and current['results'][0]['value']['value'] == '45'


def test_conditions_are_reported_or_discharged_never_dropped():
    l = Ledger()
    l.add('overtime_multiplier', 1.5, unit='times', kind='requirement',
          conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True},
                      {'field': 'hours_worked', 'operator': 'gt', 'value': {'value': 40, 'unit': 'hours'}}])
    reported = ask(l.e, 'overtime_multiplier')
    assert reported['status'] == 'answered' and reported['results'][0]['conditional']
    assert len(reported['results'][0]['pending_conditions']) == 2
    strict = ask(l.e, 'overtime_multiplier', conditions='discharge')
    assert strict['abstain'] and strict['excluded'][0]['reason'] == 'condition_not_discharged'
    met = ask(l.e, 'overtime_multiplier', conditions='discharge',
              context={'employee_is_nonexempt': True, 'hours_worked': {'value': 41, 'unit': 'hours'}})
    assert met['status'] == 'answered' and not met['results'][0]['conditional']
    violated = ask(l.e, 'overtime_multiplier', context={'employee_is_nonexempt': False})
    assert violated['abstain'] and violated['excluded'][0]['reason'] == 'condition_not_met'


def test_unassessed_structure_prevents_acceptance():
    l = Ledger()
    l.add('payment_term', 30, unit='days', structural_reliability=None)
    result = ask(l.e, 'payment_term')
    assert result['abstain'] and result['results'][0]['status'] == 'unassessed'


def test_aggregation_requires_event_identity_and_counts_each_event_once():
    l = Ledger()
    l.add('amount', 25000, unit='EUR', scope={'event_id': 'Q1'})
    l.add('amount', 25000, unit='EUR', scope={'event_id': 'Q2'})
    l.add('amount', 25000, unit='EUR', scope={'event_id': 'Q3'})
    l.add('amount', 25000, unit='EUR', scope={'event_id': 'Q3'}, group='copy')
    total = ask(l.e, 'amount', operation='sum')
    assert total['status'] == 'answered' and total['results'][0]['value'] == '75000'
    assert ask(l.e, 'amount', operation='count')['results'][0]['value'] == '3'


def test_aggregation_abstains_without_identity_mixed_currency_or_conflict():
    no_id = Ledger()
    no_id.add('amount', 25000, unit='EUR')
    no_id.add('amount', 25000, unit='EUR')
    assert ask(no_id.e, 'amount', operation='sum')['reason'] == 'event_identity_required'
    mixed = Ledger()
    mixed.add('amount', 10, unit='EUR', scope={'event_id': 'a'})
    mixed.add('amount', 20, unit='USD', scope={'event_id': 'b'})
    assert ask(mixed.e, 'amount', operation='sum')['reason'] == 'incompatible_units'
    conflict = Ledger()
    conflict.add('amount', 10, unit='EUR', scope={'event_id': 'a'}, weight=0.9)
    conflict.add('amount', 20, unit='EUR', scope={'event_id': 'a'}, weight=0.9)
    assert ask(conflict.e, 'amount', operation='sum')['reason'] == 'unresolved_event_values'


def test_event_conflict_is_resolved_by_declared_authority():
    l = Ledger()
    l.add('amount', 10, unit='EUR', scope={'event_id': 'a'}, rank=TIER_PRIMARY)
    l.add('amount', 99, unit='EUR', scope={'event_id': 'a'}, rank=TIER_EXTERNAL)
    l.add('amount', 5, unit='EUR', scope={'event_id': 'b'})
    assert ask(l.e, 'amount', operation='sum')['results'][0]['value'] == '15'


def test_declared_conversions_aggregate_exactly():
    l = Ledger()
    l.add('term', 1, unit='years', scope={'event_id': 'a'})
    l.add('term', 6, unit='months', scope={'event_id': 'b'})
    result = ask(l.e, 'term', operation='sum')
    assert result['results'][0]['value'] == '18' and result['results'][0]['unit'] == 'months'


def test_rejected_proposals_make_totals_incomplete():
    l = Ledger()
    l.add('amount', 10, unit='EUR', scope={'event_id': 'a'})
    l.add('amount', 20, unit='furlongs', scope={'event_id': 'b'})
    result = ask(l.e, 'amount', operation='sum')
    assert result['status'] == 'incomplete' and result['coverage']['rejected_proposals'] == 1


def test_replay_is_idempotent_and_reopen_is_identical(tmp_path):
    l = Ledger(tmp_path)
    l.add('payment_term', 30, unit='days', doc='same.txt')
    first = ask(l.e, 'payment_term')
    text = 'Record 1: ACME payment_term 30 days.'
    doc = SourceDocument('same.txt', text, authority_rank=TIER_PRIMARY, weight=0.9)
    l.e.ingest_units([{'proposal': {'entity': 'ACME', 'predicate': 'payment_term', 'value': 30, 'unit': 'days',
                                    'assertion_kind': 'fact', 'conditions': [], 'scope': {}, 'valid_from': None, 'valid_until': None},
                       'evidence': [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)]}], attestor='test-fixture')
    assert len(l.e.units) == 1
    again = ask(l.e, 'payment_term')
    reopened = ask(EvidenceLayer(tmp_path / 'v4.db', vocabulary=VOCAB), 'payment_term')
    strip = lambda r: {k: v for k, v in r.items() if k not in {'response_id', 'certificate'}}
    assert strip(first) == strip(again) == strip(reopened)


def test_changed_source_revision_invalidates_units():
    l = Ledger()
    l.add('payment_term', 30, unit='days', doc='contract.txt')
    l.e.ingest(SourceDocument('contract.txt', 'Rewritten contract without that clause.'))
    result = ask(l.e, 'payment_term')
    assert result['abstain'] and result['excluded'][0]['reason'] == 'stale_evidence'


def test_integrity_detects_tampering():
    l = Ledger()
    l.add('payment_term', 30, unit='days')
    result = ask(l.e, 'payment_term')
    assert l.e.verify_response_integrity(result)
    tampered = {**result, 'results': [{**result['results'][0], 'value': 'TAMPERED'}]}
    assert not l.e.verify_response_integrity(tampered)


# ------------------------------------------------------------------ check_claim
def overtime_ledger():
    l = Ledger()
    l.add('overtime_multiplier', 1.5, unit='times', kind='requirement',
          conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True},
                      {'field': 'hours_worked', 'operator': 'gt', 'value': {'value': 40, 'unit': 'hours'}}])
    return l


def claim(**kw):
    base = {'entity': 'ACME', 'predicate': 'overtime_multiplier', 'value': 1.5, 'unit': 'multiple',
            'assertion_kind': 'requirement',
            'conditions': [{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True},
                           {'field': 'hours_worked', 'operator': 'gt', 'value': {'value': 40, 'unit': 'hours'}}]}
    base.update(kw)
    return base


def test_check_claim_supported_with_alias_units_and_reordered_conditions():
    l = overtime_ledger()
    reordered = claim(conditions=list(reversed(claim()['conditions'])), unit='x')
    assert l.e.check_claim(reordered, as_of='2026-01-01')['status'] == 'supported'


@pytest.mark.parametrize('change,status', [
    (dict(value=2), 'contradicted'),
    (dict(unit='USD'), 'not_representable'),
    (dict(conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': False},
                      {'field': 'hours_worked', 'operator': 'gt', 'value': {'value': 40, 'unit': 'hours'}}]), 'not_supported'),
    (dict(conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True}]), 'not_supported'),
    (dict(conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True},
                      {'field': 'hours_worked', 'operator': 'ge', 'value': {'value': 40, 'unit': 'hours'}}]), 'not_supported'),
    (dict(assertion_kind='fact'), 'not_supported'),
    (dict(entity='Everyone'), 'insufficient_evidence'),
])
def test_check_claim_rejects_each_single_field_perturbation(change, status):
    result = overtime_ledger().e.check_claim(claim(**change), as_of='2026-01-01')
    assert result['status'] == status
    if status in {'contradicted', 'not_supported'}:
        assert result['differences']


def test_check_claim_never_calls_a_model():
    class Exploding:
        identity = {'model': 'must-not-be-called'}

        def propose(self, request):
            raise AssertionError('reader called during check_claim')
    l = overtime_ledger()
    l.e.reader = Exploding()
    assert l.e.check_claim(claim(), as_of='2026-01-01')['status'] == 'supported'


# ------------------------------------------------------------- derivations/paths
def test_derivation_requires_installed_rule_and_scores_weakest_link():
    l = Ledger()
    premise, _ = l.add('payment_term', 30, unit='days', weight=0.75, doc='p.txt')
    with pytest.raises(ValueError):
        l.e.derive({'entity': 'ACME', 'predicate': 'payment_term', 'value': 30, 'unit': 'days', 'assertion_kind': 'fact'},
                   [premise], 'invented_rule')
    with pytest.raises(ValueError):
        l.e.derive({'entity': 'ACME', 'predicate': 'payment_term', 'value': 31, 'unit': 'days', 'assertion_kind': 'fact'},
                   [premise], 'identity')
    derived = l.e.derive({'entity': 'ACME', 'predicate': 'payment_term', 'value': 30, 'unit': 'days',
                          'assertion_kind': 'fact'}, [premise], 'identity')
    result = ask(l.e, 'payment_term')['results'][0]
    by_id = {e['unit_id']: e for e in result['evidence']}
    assert by_id[derived]['support_index'] == pytest.approx(0.95 * 0.75)
    # the copy shares the premise's dependence group: it does not add support
    assert result['support']['S'] == 0.75


def test_derived_unit_fails_when_premise_is_invalidated():
    l = Ledger()
    premise, _ = l.add('payment_term', 30, unit='days', doc='p.txt')
    derived = l.e.derive({'entity': 'ACME', 'predicate': 'payment_term', 'value': 30, 'unit': 'days',
                          'assertion_kind': 'fact'}, [premise], 'identity')
    l.e.ingest(SourceDocument('p.txt', 'changed'))
    reasons = {x['unit_id']: x['reason'] for x in ask(l.e, 'payment_term')['excluded']}
    assert reasons[derived] == 'premise_stale_evidence'


def paper_path_ledger():
    l = Ledger(transitive_rules=[TransitiveRule('subsidiary_of', '1')])
    l.add('subsidiary_of', 'B', entity='A', weight=0.9)
    l.add('subsidiary_of', 'C', entity='B', weight=0.8)
    l.add('subsidiary_of', 'D', entity='C', weight=0.75)
    l.add('director', 'Jane Roe', entity='C', weight=0.9)
    return l


def test_transitive_paths_use_paper_weakest_link_score():
    l = paper_path_ledger()
    response = l.e.derive_paths('A', 'subsidiary_of', as_of='2026-01-01')
    scores = {tuple(p['nodes']): p['path_score'] for p in response['paths']}
    assert scores[('A', 'B', 'C', 'D')] == pytest.approx(0.676875)  # 0.75 * 0.95^2, paper Section 6
    assert scores[('A', 'B', 'C')] == pytest.approx(0.8 * 0.95)
    derived = ask(l.e, 'subsidiary_of', entity='A', operation='list')
    assert {r['value'] for r in derived['results']} >= {'B', 'C', 'D'}


def test_undeclared_relations_only_return_candidate_paths():
    l = paper_path_ledger()
    with pytest.raises(ValueError):
        l.e.derive_paths('A', 'director', as_of='2026-01-01')
    candidates = l.e.find_paths('A', as_of='2026-01-01')
    nodes = [tuple(p['nodes']) for p in candidates['paths']]
    assert ('A', 'B', 'C', 'Jane Roe') in nodes
    assert all(p['status'] == 'candidate_path' for p in candidates['paths'])
    assert not [u for u in l.e.units if u.origin == 'rule_derived']


def test_document_review_can_ignore_validity_windows_explicitly():
    l = Ledger()
    l.add('payment_term', 30, unit='days', valid_from='2030-01-01', valid_until='2034-12-31')
    assert ask(l.e, 'payment_term', as_of='2026-01-01')['excluded'][0]['reason'] == 'outside_validity'
    review = ask(l.e, 'payment_term', as_of='2026-01-01', validity='ignore')
    assert review['status'] == 'answered' and review['validity_mode'] == 'ignore'
    check = l.e.check_claim({'entity': 'ACME', 'predicate': 'payment_term', 'value': 30, 'unit': 'days',
                             'assertion_kind': 'fact'}, as_of='2026-01-01', validity='ignore')
    assert check['status'] == 'supported'
