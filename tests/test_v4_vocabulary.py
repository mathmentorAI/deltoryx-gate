"""v4 vocabulary contract: declared labels only, exact conversions only."""
import pytest

from claimlayer.v4 import FieldSpec, PredicateSpec, STANDARD_UNITS, Unit, UnmappedProposal, Vocabulary


def vocab(**kwargs):
    return Vocabulary('test', '1', (
        PredicateSpec('rate', FieldSpec('quantity', dimensions=('annual_rate',))),
        PredicateSpec('deadline', FieldSpec('quantity', dimensions=('duration_business_days', 'duration_calendar_days'))),
        PredicateSpec('term', FieldSpec('quantity', dimensions=('duration_months',))),
        PredicateSpec('hours', FieldSpec('bound', dimensions=('hours_per_week',))),
        PredicateSpec('flag', FieldSpec('boolean'), conditions=(('nonexempt', FieldSpec('boolean')),)),
        PredicateSpec('territory', FieldSpec('set', enum=('United States', 'Canada', 'Worldwide'))),
        PredicateSpec('rule', FieldSpec.record(duration=FieldSpec('quantity', dimensions=('duration_months',)),
                                               passive=FieldSpec('boolean'))),
    ), **kwargs)


def prop(**kw):
    return {'entity': 'X', 'assertion_kind': 'requirement', **kw}


def test_unit_aliases_normalize_to_one_declared_unit():
    v = vocab()
    a = v.proposition(prop(predicate='rate', value=0.75, unit='percent per annum'))
    b = v.proposition(prop(predicate='rate', value='0.750', unit='Percent_Per_Annum'))
    c = v.proposition(prop(predicate='rate', value=0.75, unit='% p.a.'))
    assert a == b == c
    assert a['value'] == {'value': '0.75', 'unit': 'percent_per_annum'}


def test_unit_of_another_dimension_is_not_representable():
    with pytest.raises(UnmappedProposal) as err:
        vocab().proposition(prop(predicate='rate', value=0.75, unit='USD'))
    assert err.value.reason == 'unit_dimension_mismatch'


def test_unknown_unit_and_unknown_predicate_are_rejected():
    with pytest.raises(UnmappedProposal) as err:
        vocab().proposition(prop(predicate='rate', value=1, unit='furlongs'))
    assert err.value.reason == 'unknown_unit'
    with pytest.raises(UnmappedProposal) as err:
        vocab().proposition(prop(predicate='post_termination_years', value=24))
    assert err.value.reason == 'unknown_predicate'


def test_exact_conversions_only():
    v = vocab()
    assert v.proposition(prop(predicate='term', value=2, unit='years'))['value'] == {'value': '24', 'unit': 'months'}
    # business days and calendar days are different dimensions: both allowed, never equal
    business = v.proposition(prop(predicate='deadline', value=45, unit='business days'))
    calendar = v.proposition(prop(predicate='deadline', value=45, unit='calendar_days'))
    assert business['value'] != calendar['value']
    with pytest.raises(UnmappedProposal):
        v.proposition(prop(predicate='term', value=30, unit='days'))


def test_bounds_keep_their_operator():
    v = vocab()
    ge = v.proposition(prop(predicate='hours', value={'operator': 'ge', 'value': 30, 'unit': 'hours_per_week'}))
    gt = v.proposition(prop(predicate='hours', value={'operator': 'gt', 'value': 30, 'unit': 'hours_per_week'}))
    assert ge['value'] != gt['value']
    with pytest.raises(UnmappedProposal) as err:
        v.proposition(prop(predicate='hours', value={'operator': 'ge', 'value': 30, 'unit': 'hours_per_day'}))
    assert err.value.reason == 'unit_dimension_mismatch'


def test_conditions_must_be_declared_and_are_order_insensitive():
    v = vocab()
    with pytest.raises(UnmappedProposal) as err:
        v.proposition(prop(predicate='flag', value=True, conditions=[{'field': 'invented', 'operator': 'eq', 'value': True}]))
    assert err.value.reason == 'undeclared_condition'
    one = v.proposition(prop(predicate='flag', value=True, conditions=[{'field': 'nonexempt', 'operator': 'eq', 'value': True}]))
    assert one['conditions'] == [{'field': 'nonexempt', 'operator': 'eq', 'value': True}]


def test_records_reject_free_keys_and_missing_fields():
    v = vocab()
    with pytest.raises(UnmappedProposal) as err:
        v.proposition(prop(predicate='rule', value={'duration': {'value': 24, 'unit': 'months'}, 'passive': True,
                                                    'post_termination_years': 24}))
    assert err.value.reason == 'undeclared_field'
    with pytest.raises(UnmappedProposal) as err:
        v.proposition(prop(predicate='rule', value={'passive': True}))
    assert err.value.reason == 'missing_field'
    ok = v.proposition(prop(predicate='rule', value={'duration': {'value': 2, 'unit': 'years'}, 'passive': True}))
    assert ok['value']['duration'] == {'value': '24', 'unit': 'months'}


def test_sets_are_order_insensitive_and_closed():
    v = vocab()
    a = v.proposition(prop(predicate='territory', value=['Canada', 'united states']))
    b = v.proposition(prop(predicate='territory', value=['United States', 'Canada']))
    assert a == b
    with pytest.raises(UnmappedProposal):
        v.proposition(prop(predicate='territory', value=['Mars']))


def test_entities_use_declared_aliases_and_optional_closed_world():
    v = vocab(entities=(('ACME Corp', ('ACME', 'Acme Corporation')),), closed_entities=True)
    assert v.proposition(prop(predicate='flag', value=True, entity='acme corporation'))['entity'] == 'ACME Corp'
    with pytest.raises(UnmappedProposal):
        v.proposition(prop(predicate='flag', value=True, entity='Other'))


def test_registry_refuses_ambiguous_labels():
    with pytest.raises(ValueError):
        STANDARD_UNITS.extend([Unit('days_alt', 'duration_calendar_days', '2', ('days',))])


def test_vocabulary_fingerprint_changes_with_contract():
    assert vocab().fingerprint == vocab().fingerprint
    assert vocab().fingerprint != vocab(closed_entities=True).fingerprint


def test_dates_are_iso_calendar_dates_only():
    v = Vocabulary('dates', '1', (PredicateSpec('effective_date', FieldSpec('date')),))
    assert v.proposition(prop(predicate='effective_date', value='2027-01-31'))['value'] == '2027-01-31'
    stamp = v.proposition(prop(predicate='effective_date', value='2023-03-08T02:14:00-07:00'))['value']
    assert stamp == '2023-03-08T09:14:00Z'
    for bad in ('31/01/2027', 'January 31, 2027', '2027-02-30', 20270131):
        with pytest.raises(UnmappedProposal) as err:
            v.proposition(prop(predicate='effective_date', value=bad))
        assert err.value.reason == 'iso_date_required'


def test_description_tells_readers_the_json_shape():
    described = vocab().describe()['predicates']
    shapes = {p['name']: p['value']['json_shape'] for p in described}
    assert '"unit"' in shapes['rate'] and '"operator"' in shapes['hours']
