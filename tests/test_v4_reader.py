"""v4 reader contract with a model double: proposals only, exact binding, no verdicts."""
import json

from claimlayer.v4 import (READER_PROMPT, EvidenceLayer, FieldSpec, LLMProposalReader, PredicateSpec,
                           SourceDocument, Vocabulary)

VOCAB = Vocabulary('reader-test', '1', (
    PredicateSpec('sofr_floor', FieldSpec('quantity', dimensions=('annual_rate',)), assertion_kinds=('proposal', 'fact')),
))
TEXT = 'Lender markup. SOFR Floor. Term SOFR shall be subject to a floor of 0.75% per annum. ' * 3


class Model:
    """Returns scripted outputs and records every payload it receives."""

    def __init__(self, script):
        self.script, self.payloads = list(script), []

    identity = {'provider': 'double', 'model': 'scripted', 'endpoint': 'local', 'parameters': {}}

    def complete(self, system, payload):
        self.payloads.append((system, json.loads(json.dumps(payload))))
        output = self.script.pop(0)
        if isinstance(output, Exception):
            raise output
        return {'output': output(payload) if callable(output) else output, 'usage': {'total_tokens': 1}}


def proposal(payload, **kw):
    first = payload['fragments'][0]['evidence_id']
    return {'proposals': [{'entity': 'Lender markup', 'predicate': 'sofr_floor', 'value': 0.75,
                           'unit': '% per annum', 'assertion_kind': 'proposal', 'basis': 'explicit',
                           'evidence': [first], **kw}]}


def layer(model, reliability=0.9):
    reader = LLMProposalReader(model, fragment_chars=200, window_fragments=1, structural_reliability=reliability)
    return EvidenceLayer(vocabulary=VOCAB, reader=reader)


def test_reader_binds_ids_to_exact_spans_and_engine_admits():
    model = Model([proposal, {'proposals': []}])
    e = layer(model)
    summary = e.ingest(SourceDocument('markup.docx', TEXT, weight=0.8))
    assert summary['units_admitted'] == 1 and summary['proposals_rejected'] == 0
    unit = e.units[0]
    assert TEXT[unit.evidence[0].start:unit.evidence[0].end] == unit.evidence[0].quote
    r = e.ask({'entity': 'Lender markup', 'predicate': 'sofr_floor', 'as_of': '2026-01-01'}, assertion_kinds=('proposal',))
    assert r['status'] == 'answered' and r['results'][0]['support']['S'] == 0.8


def test_prompt_carries_vocabulary_but_no_candidate_claim_or_verdict_request():
    model = Model([{'proposals': []}, {'proposals': []}])
    layer(model).ingest(SourceDocument('markup.docx', TEXT))
    system, payload = model.payloads[0]
    assert system == READER_PROMPT
    assert set(payload) == {'document_id', 'fragments', 'vocabulary', 'admissible_bases', 'max_proposals'}
    assert payload['vocabulary']['predicates'][0]['name'] == 'sofr_floor'
    assert 'narrative' not in payload['admissible_bases'] and 'explicit' in payload['admissible_bases']
    assert 'authority_rank' not in json.dumps(payload)


def test_foreign_ids_free_keys_and_verdict_fields_are_rejected():
    model = Model([
        lambda p: {'proposals': [proposal(p)['proposals'][0] | {'evidence': ['ev-invented']},
                                 proposal(p)['proposals'][0] | {'verdict': 'supported'},
                                 proposal(p)['proposals'][0] | {'value': {'post_termination_years': 24}}]},
        {'proposals': []},
    ])
    e = layer(model)
    summary = e.ingest(SourceDocument('markup.docx', TEXT))
    assert summary['units_admitted'] == 0 and summary['proposals_rejected'] == 3
    reasons = sorted(r['error']['reason'] for r in e.rejections())
    assert reasons == ['quantity_shape', 'reader_schema', 'unknown_or_foreign_evidence_id']


def test_wrong_dimension_from_reader_is_rejected_not_rescued():
    model = Model([lambda p: proposal(p, unit='USD'), {'proposals': []}])
    e = layer(model)
    assert e.ingest(SourceDocument('markup.docx', TEXT))['rejection_reasons'] == {'unit_dimension_mismatch': 1}


def test_unmeasured_reader_reliability_keeps_units_unassessed():
    model = Model([proposal, {'proposals': []}])
    e = layer(model, reliability=None)
    e.ingest(SourceDocument('markup.docx', TEXT))
    r = e.ask({'entity': 'Lender markup', 'predicate': 'sofr_floor', 'as_of': '2026-01-01'}, assertion_kinds=('proposal',))
    assert r['abstain'] and r['results'][0]['status'] == 'unassessed'


def test_transport_errors_are_recorded_and_reads_are_cached():
    model = Model([TimeoutError('slow'), {'proposals': []}])
    e = layer(model)
    summary = e.ingest(SourceDocument('markup.docx', TEXT))
    assert summary['read_errors'][0]['error_type'] == 'TimeoutError' and summary['units_admitted'] == 0
    calls = len(model.payloads)
    e.ingest(SourceDocument('markup.docx', TEXT))  # identical request: cached, no new model call
    assert len(model.payloads) == calls


def test_quote_narrows_evidence_to_the_located_words_only():
    model = Model([lambda p: proposal(p, quote='floor of 0.75%  per   annum'), {'proposals': []}])
    e = layer(model)
    e.ingest(SourceDocument('markup.docx', TEXT, weight=0.8))
    ref = e.units[0].evidence[0]
    assert ref.quote == 'floor of 0.75% per annum' and TEXT[ref.start:ref.end] == ref.quote


def test_paraphrased_quote_keeps_fragment_level_evidence():
    model = Model([lambda p: proposal(p, quote='a 0.75 percent floor applies'), {'proposals': []}])
    e = layer(model)
    e.ingest(SourceDocument('markup.docx', TEXT, weight=0.8))
    ref = e.units[0].evidence[0]
    assert ref.end - ref.start == 200  # the whole cited fragment


def test_locate_quote_ignores_presentation_only():
    from claimlayer.v4.spans import locate_quote
    fragment = 'Section 4.2: the **Tenant** shall pay \\$52.00 per RSF — “Base Rent”.'
    span = locate_quote(fragment, 'tenant shall pay $52.00 per RSF - "Base Rent"')
    assert span is not None and fragment[span[0]:span[1]].startswith('Tenant')
    assert locate_quote(fragment, 'tenant must pay $52.00') is None


def test_truncated_window_is_split_and_read_again():
    class Truncating(Model):
        def complete(self, system, payload):
            if len(payload['fragments']) > 1:
                exc = ValueError('truncated')
                exc.diagnostics = {'attempts': [{'attempt': 1, 'finish_reason': 'length'}]}
                raise exc
            return {'output': proposal(payload), 'usage': {'total_tokens': 1}}
    reader = LLMProposalReader(Truncating([]), fragment_chars=200, window_fragments=4, structural_reliability=0.9)
    e = EvidenceLayer(vocabulary=VOCAB, reader=reader)
    summary = e.ingest(SourceDocument('markup.docx', TEXT, weight=0.8))
    fragments = -(-len(TEXT) // 200)
    assert summary['read_errors'] == [] and summary['units_admitted'] == fragments
