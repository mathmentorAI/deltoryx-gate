"""Verification triage: lanes, the number/date guard and reading load (model double, no network)."""
import json

from claimlayer.v4 import EvidenceLayer, EvidenceRef, FieldSpec, LLMProposalReader, PredicateSpec, SourceDocument, Vocabulary
from claimlayer.v4.triage import claim_values, segment_text, value_mentions, verify_draft

VOCAB = Vocabulary('triage', '1', (
    PredicateSpec('base_rent', FieldSpec('quantity', dimensions=('currency:USD',))),
    PredicateSpec('lease_term', FieldSpec('quantity', dimensions=('duration_months',))),
    PredicateSpec('governing_law', FieldSpec('string')),
    PredicateSpec('commencement_date', FieldSpec('date')),
))


def test_segments_split_sentences_and_keep_table_rows():
    text = ('# Memo\nThe Tenant is Whitecliff Capital Partners LLC. Base rent is $52.00 per RSF (see Sec. 4). '
            'Term: 10 years.\n| Item | Value |\n|---|---|\n| Rent | $52.00 |\n')
    segs = [s['text'] for s in segment_text(text)]
    assert segs == ['# Memo', 'The Tenant is Whitecliff Capital Partners LLC.', 'Base rent is $52.00 per RSF (see Sec. 4).',
                    'Term: 10 years.', '| Item | Value |', '| Rent | $52.00 |']
    for s in segment_text(text):
        assert text[s['start']:s['end']] == s['text']


def test_value_mentions_skip_references_and_list_markers():
    found = value_mentions('3. Under Section 11.4, rent of $4.6 million is due on March 3, 2025 and 12% after 30 days in 2026.')
    texts = [m['text'] for m in found]
    assert 'March 3, 2025' in texts and '11.4' not in texts and '3.' not in texts
    money = next(m for m in found if m['text'].startswith('$4.6'))
    assert '4600000' in money['forms'] and '4.6' in money['forms']
    assert next(m for m in found if m['text'] == '12%')['forms'] == ['0.12', '12']
    assert next(m for m in found if m['text'] == '2026')['kind'] == 'year'
    assert claim_values({'value': 52, 'unit': 'USD'}) == {'52'}


SOURCE = ('Lease Agreement. Section 4. Base Rent shall be $52.00 per rentable square foot in Year 1. '
          'Section 2. The Term is ten (10) years. Section 30. This Lease is governed by the laws of Delaware. '
          'The Commencement Date is March 1, 2025.')
DRAFT = ('Summary\nBase rent is $52.00 per RSF. The term is 10 years and starts on March 1, 2025. '
         'The lease is governed by Texas law. We recommend accepting the markup.\n')


class Model:
    identity = {'provider': 'double', 'model': 'scripted', 'endpoint': 'local', 'parameters': {}}

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def complete(self, system, payload):
        return {'output': self.outputs.pop(0)(payload), 'usage': {'total_tokens': 1}}


def draft_reading(payload):
    eid = payload['fragments'][0]['evidence_id']
    p = lambda pred, value, unit, quote: {'entity': 'Lease', 'predicate': pred, 'value': value, 'unit': unit,
                                          'assertion_kind': 'fact', 'basis': 'explicit', 'evidence': [eid], 'quote': quote}
    return {'proposals': [p('base_rent', 52, 'USD', 'Base rent is $52.00 per RSF'),
                          p('lease_term', 10, 'years', 'The term is 10 years'),
                          p('governing_law', 'Texas', None, 'governed by Texas law')]}


def ledger():
    e = EvidenceLayer(vocabulary=VOCAB)
    doc = SourceDocument('lease.docx', SOURCE, authority_rank=100, weight=0.9)
    e.ingest(doc)
    def ref(phrase):
        i = SOURCE.index(phrase)
        return [EvidenceRef(doc.document_id, doc.revision, i, i + len(phrase), phrase)]
    e.ingest_units([
        {'proposal': {'entity': 'Lease', 'predicate': 'base_rent', 'value': 52, 'unit': 'USD', 'assertion_kind': 'fact'},
         'evidence': ref('Base Rent shall be $52.00')},
        {'proposal': {'entity': 'Lease', 'predicate': 'lease_term', 'value': 10, 'unit': 'years', 'assertion_kind': 'fact'},
         'evidence': ref('The Term is ten (10) years')},
        {'proposal': {'entity': 'Lease', 'predicate': 'governing_law', 'value': 'Delaware', 'assertion_kind': 'fact'},
         'evidence': ref('governed by the laws of Delaware')},
    ], attestor='fixture')
    return e


def test_triage_lanes_guard_and_reading_load():
    reader = LLMProposalReader(Model([draft_reading]), fragment_chars=2000, structural_reliability=0.9)
    report = verify_draft(ledger(), DRAFT, reader=reader)
    lanes = {s['text']: s['lane'] for s in report['segments']}
    assert lanes['Base rent is $52.00 per RSF.'] == 'verified'
    # the term is supported, but the date in the same sentence is not accounted for by any supported claim
    assert lanes['The term is 10 years and starts on March 1, 2025.'] == 'unverified'
    contradicted = next(s for s in report['segments'] if s['lane'] == 'contradicted')
    assert contradicted['text'] == 'The lease is governed by Texas law.'
    assert contradicted['claims'][0]['sources'][0]['quote'] == 'governed by the laws of Delaware'
    assert lanes['We recommend accepting the markup.'] == 'no_claim'
    r = report['reading']
    assert r['alarms_only']['words'] < r['skip_verified']['words'] < r['read_all']['words'] + 10
    unguarded = verify_draft(ledger(), DRAFT, reader=LLMProposalReader(Model([draft_reading]), fragment_chars=2000,
                                                                          structural_reliability=0.9), guard=False)
    assert {s['text']: s['lane'] for s in unguarded['segments']}['The term is 10 years and starts on March 1, 2025.'] == 'verified'


def test_draft_vocabulary_hints_carry_the_sources_entity_names():
    from claimlayer.v4 import draft_vocabulary
    v = draft_vocabulary(ledger())
    described = {p['name']: p['description'] for p in v.describe()['predicates']}
    assert 'Lease' in described['base_rent'] and 'Entities named for this attribute' in described['base_rent']
    assert v.entity('lease') == 'Lease'


def test_same_value_read_as_another_speech_act_is_verified_with_a_note():
    def reading(payload):
        eid = payload['fragments'][0]['evidence_id']
        return {'proposals': [{'entity': 'Lease', 'predicate': 'base_rent', 'value': 52, 'unit': 'USD',
                               'assertion_kind': 'allegation', 'basis': 'explicit', 'evidence': [eid],
                               'quote': 'Base rent is $52.00 per RSF'}]}
    reader = LLMProposalReader(Model([reading]), fragment_chars=2000, structural_reliability=0.9)
    report = verify_draft(ledger(), 'Base rent is $52.00 per RSF.\n', reader=reader)
    segment = report['segments'][0]
    assert segment['lane'] == 'verified'
    assert segment['claims'][0]['check_status'] == 'not_supported' and segment['claims'][0]['note'] == 'assertion_kind_differs'


def test_draft_vocabulary_inherits_closed_entities_aliases_and_condition_status():
    from claimlayer.v4 import draft_vocabulary

    # 1. Single-entity closed vocabulary with explicit condition_status, scope_keys, register_status_terms
    closed_single = Vocabulary(
        name='code_closed_1',
        version='1.0',
        predicates=(PredicateSpec('changes_file', FieldSpec('string'), cardinality='multi'),),
        entities=(('pull request', ('PR', 'this PR')),),
        closed_entities=True,
        scope_keys=('event_id', 'repo'),
        require_explicit_condition_status=True,
        register_status_terms=('active',),
    )
    l1 = EvidenceLayer(vocabulary=closed_single)
    d1 = SourceDocument('diff.txt', 'The pull request changes file src/app.py.', authority_rank=100, weight=1.0)
    l1.ingest(d1)
    l1.ingest_units([{
        'proposal': {
            'entity': 'pull request',
            'predicate': 'changes_file',
            'value': 'src/app.py',
            'assertion_kind': 'fact',
            'condition_status': 'unconditional',
        },
        'evidence': [EvidenceRef(d1.document_id, d1.revision, 0, len(d1.text), d1.text)],
    }], attestor='fixture')

    dv1 = draft_vocabulary(l1)
    assert dv1.closed_entities is True
    assert dv1.require_explicit_condition_status is True
    assert dv1.scope_keys == ('event_id', 'repo')
    assert dv1.register_status_terms == ('active',)
    assert dv1.entities == closed_single.entities
    assert dv1.fingerprint == closed_single.fingerprint

    # Reader on verify_draft must reject undeclared entity & missing condition_status, and accept alias
    draft_txt = 'This PR modifies src/app.py.'

    def mixed_proposals(payload):
        eid = payload['fragments'][0]['evidence_id']
        return {'proposals': [
            {
                'entity': 'pull request #1538',
                'predicate': 'changes_file',
                'value': 'src/app.py',
                'assertion_kind': 'fact',
                'condition_status': 'unconditional',
                'basis': 'explicit',
                'evidence': [eid],
                'quote': draft_txt,
            },
            {
                'entity': 'this PR',
                'predicate': 'changes_file',
                'value': 'src/app.py',
                'assertion_kind': 'fact',
                'basis': 'explicit',  # missing condition_status -> rejected
                'evidence': [eid],
                'quote': draft_txt,
            },
            {
                'entity': 'this PR',
                'predicate': 'changes_file',
                'value': 'src/app.py',
                'assertion_kind': 'fact',
                'condition_status': 'unconditional',
                'basis': 'explicit',
                'evidence': [eid],
                'quote': draft_txt,
            },
        ]}

    rep1 = verify_draft(l1, draft_txt, reader=LLMProposalReader(Model([mixed_proposals]), fragment_chars=2000,
                                                              structural_reliability=0.9), guard=False)
    assert rep1['draft_reading']['units_admitted'] == 1
    assert rep1['draft_reading']['rejection_reasons'] == {
        'unknown_entity': 1,
        'condition_status_required': 1,
    }
    assert rep1['segments'][0]['lane'] == 'verified'

    # 2. Multi-entity closed vocabulary keeps all declared entities/aliases + inherits flags + filters hints
    closed_multi = Vocabulary(
        name='code_closed_2',
        version='1.0',
        predicates=(PredicateSpec('role', FieldSpec('string')),),
        entities=(('Acme', ('Acme Corp',)), ('Beta', ())),
        closed_entities=True,
        scope_keys=('event_id', 'repo'),
        require_explicit_condition_status=True,
        register_status_terms=('active',),
    )
    l2 = EvidenceLayer(vocabulary=closed_multi)
    d2 = SourceDocument('s.txt', 'Acme is borrower.', authority_rank=100, weight=1.0)
    l2.ingest(d2)
    l2.ingest_units([{
        'proposal': {'entity': 'Acme', 'predicate': 'role', 'value': 'borrower',
                     'assertion_kind': 'fact', 'condition_status': 'unconditional'},
        'evidence': [EvidenceRef(d2.document_id, d2.revision, 0, len(d2.text), d2.text)],
    }], attestor='fixture')
    dv2 = draft_vocabulary(l2)
    assert dv2.closed_entities is True
    assert dv2.require_explicit_condition_status is True
    assert dv2.scope_keys == ('event_id', 'repo')
    assert dv2.register_status_terms == ('active',)
    assert dv2.entities == closed_multi.entities
    assert dv2.entity('Acme Corp') == 'Acme'
    assert dv2.entity('Beta') == 'Beta'
    rep2 = verify_draft(
        l2,
        'Gamma is borrower.',
        reader=LLMProposalReader(Model([lambda pl: {'proposals': [{
            'entity': 'Gamma', 'predicate': 'role', 'value': 'borrower',
            'assertion_kind': 'fact', 'condition_status': 'unconditional',
            'basis': 'explicit', 'evidence': [pl['fragments'][0]['evidence_id']], 'quote': 'Gamma is borrower.',
        }]}]), fragment_chars=2000, structural_reliability=0.9),
        guard=False,
    )
    assert rep2['draft_reading']['units_admitted'] == 0
    assert rep2['draft_reading']['rejection_reasons'] == {'unknown_entity': 1}
