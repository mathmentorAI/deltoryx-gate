"""Value relations: different precision or an undecidable time zone is never reported as a contradiction."""
import pytest

from claimlayer.v4 import EvidenceLayer, EvidenceRef, FieldSpec, PredicateSpec, SourceDocument, Vocabulary
from claimlayer.v4.vocabulary import UnmappedProposal, date_relation, normalize_date, value_relation

VOCAB = Vocabulary('rel', '1', (
    PredicateSpec('effective_date', FieldSpec('date')),
    PredicateSpec('event_time', FieldSpec('date')),
    PredicateSpec('registered_name', FieldSpec('string')),
    PredicateSpec('counterparty', FieldSpec('entity')),
    PredicateSpec('fee', FieldSpec('quantity', dimensions=('currency:USD',))),
))


def test_dates_keep_the_stated_precision():
    assert normalize_date('2019') == '2019'
    assert normalize_date('2019-01') == '2019-01'
    assert normalize_date('2019-01-14') == '2019-01-14'
    assert normalize_date('2023-03-14T23:47:00') == '2023-03-14T23:47:00'
    assert normalize_date('2023-03-14T23:47:00-07:00') == '2023-03-15T06:47:00Z'
    for bad in ('2019-13', '19-01-14', 'January 2019', '2019-02-30'):
        with pytest.raises(ValueError):
            normalize_date(bad)
    v = VOCAB.proposition({'entity': 'X', 'predicate': 'effective_date', 'value': '2019-01', 'assertion_kind': 'fact'})
    assert v['value'] == '2019-01'
    with pytest.raises(UnmappedProposal):
        VOCAB.proposition({'entity': 'X', 'predicate': 'effective_date', 'value': 'Jan 2019', 'assertion_kind': 'fact'})


@pytest.mark.parametrize('a,b,expected', [
    ('2019-01', '2019-01-14', 'compatible'),           # month stated vs a day inside it
    ('2019', '2019-07-01', 'compatible'),
    ('2019-02', '2019-01-14', 'incompatible'),
    ('2019-01-01', '2019-01-14', 'incompatible'),       # same precision, different value
    ('2023-03-14T23:47:00', '2023-03-15T06:47:00Z', 'unknown'),   # local time vs instant: zone undecidable
    ('2023-03-14T23:47:00', '2023-03-17T06:47:00Z', 'incompatible'),  # too far apart for any zone
    ('2023-03-14', '2023-03-14T10:00:00Z', 'compatible'),
    ('2023-03-14', '2023-03-15T06:47:00Z', 'unknown'),
    ('2023-03-14', '2023-03-14T23:47:00', 'compatible'),
    ('2023-03-15T06:47:00Z', '2023-03-15T07:47:00Z', 'incompatible'),
])
def test_date_relation(a, b, expected):
    assert date_relation(a, b) == expected
    assert date_relation(b, a) == expected


def test_text_relation_is_containment_not_similarity():
    spec = FieldSpec('string')
    assert value_relation(spec, 'Ridgeline', 'Ridgeline Capital Advisors LLC') == 'compatible'
    assert value_relation(spec, 'EUIPO Trade Mark Registration No. 018432156', '018432156') == 'compatible'
    assert value_relation(spec, 'Delaware', 'delaware') == 'equal'
    assert value_relation(spec, 'Texas', 'Delaware') == 'incompatible'
    assert value_relation(spec, 'Ridge', 'Ridgeline') == 'incompatible'  # whole words only


class Ledger:
    def __init__(self):
        self.e = EvidenceLayer(None, vocabulary=VOCAB)
        self.n = 0

    def add(self, predicate, value, unit=None, entity='ACME'):
        self.n += 1
        text = f'Record {self.n}: {entity} {predicate} {value}.'
        doc = SourceDocument(f'doc-{self.n}.txt', text, authority_rank=100, weight=0.9)
        self.e.ingest(doc)
        proposal = {'entity': entity, 'predicate': predicate, 'value': value, 'unit': unit, 'assertion_kind': 'fact'}
        self.e.ingest_units([{'proposal': proposal, 'basis': 'explicit',
                              'evidence': [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)]}],
                            attestor='fixture')


def check(ledger, predicate, value, unit=None, entity='ACME'):
    return ledger.e.check_claim({'entity': entity, 'predicate': predicate, 'value': value, 'unit': unit,
                                 'assertion_kind': 'fact'}, validity='ignore')


def test_claims_of_other_precision_or_zone_are_not_contradicted():
    l = Ledger()
    l.add('effective_date', '2019-01')
    l.add('event_time', '2023-03-14T23:47:00-07:00')
    l.add('counterparty', 'Ridgeline Capital Advisors LLC')
    month = check(l, 'effective_date', '2019-01-14')
    assert month['status'] == 'insufficient_evidence' and month['relation'] == 'compatible'
    local = check(l, 'event_time', '2023-03-14T23:47:00')
    assert local['status'] == 'insufficient_evidence' and local['relation'] == 'unknown'
    assert check(l, 'event_time', '2023-03-15T06:47:00Z')['status'] == 'supported'
    assert check(l, 'counterparty', 'Ridgeline')['status'] == 'insufficient_evidence'
    assert check(l, 'effective_date', '2019-03-14')['status'] == 'contradicted'
    assert check(l, 'counterparty', 'Northwind LLC')['status'] == 'contradicted'


def test_sources_with_compatible_values_do_not_conflict():
    l = Ledger()
    l.add('effective_date', '2019-01')
    l.add('effective_date', '2019-01-14')
    result = l.e.ask({'entity': 'ACME', 'predicate': 'effective_date', 'operation': 'list', 'as_of': '2026-01-01'},
                     validity='ignore')
    assert all(r['status'] == 'accepted' for r in result['results'])
    assert check(l, 'effective_date', '2019-01-14')['status'] == 'supported'


def test_entity_identity_and_text_values_ignore_case_and_spacing():
    l = Ledger()
    l.add('counterparty', 'Ridgeline Capital Advisors LLC', entity='ACME CORP')
    assert check(l, 'counterparty', 'RIDGELINE CAPITAL  ADVISORS LLC', entity='Acme Corp')['status'] == 'supported'
    l.add('registered_name', 'Acme Corp', entity='ACME')
    l.add('registered_name', 'ACME CORP', entity='acme')
    result = l.e.ask({'entity': 'Acme', 'predicate': 'registered_name', 'operation': 'list', 'as_of': '2026-01-01'},
                     validity='ignore')
    assert result['results'] and all(r['status'] == 'accepted' for r in result['results'])
    assert check(l, 'registered_name', 'Beta LLC', entity='ACME')['status'] == 'contradicted'
