"""Engineering regression checks for source primitives retained by the current core."""
from claimlayer.v4._source_models import Source
from claimlayer.v4._source_quantities import extract_dates, extract_quantities
from claimlayer.v4._source_diff import build_comparison_policy, extract_divergent_sections_with_coverage


def test_quantity_and_date_extraction_primitives():
    values = extract_quantities('The budget is 750 USD, due on 2026-11-23.')
    assert any(q.magnitude == 750 and q.unit == 'USD' for q in values)
    assert '2026-11-23' in extract_dates('Due on 2026-11-23.')


def test_source_comparison_uses_retained_primitives():
    governing = Source(id='a', text='Section 1: Amount is 750 USD.', tier='governing')
    comparison = Source(id='b', text='Section 1: Amount is 850 USD.')
    report = extract_divergent_sections_with_coverage(governing, comparison)
    assert report.divergences
    policy, divergences = build_comparison_policy(governing, comparison)
    assert policy.required_source_ids
