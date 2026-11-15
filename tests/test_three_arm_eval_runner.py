"""Protocol safeguards for the local three-arm runner (no live endpoint required)."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parents[1] / 'examples' / 'harness2_securitization'
sys.path.insert(0, str(HERE))
from run_three_arm_eval import chunks, parse_candidate, compact_notes, Client, retrieve_windows
from harness2_v4_pipeline import build_synthetic_securitization_corpus, evaluate_securitization_candidate, build_section_comparison_packet


def test_chunks_reconstruct_documents_without_missing_tail():
    corpus=build_synthetic_securitization_corpus()
    pieces=list(chunks(corpus,37))
    for name,doc in corpus.documents.items():
        parts=[p for p in pieces if p['document_id']==name]
        assert ''.join(p['text'] for p in parts)==doc.text
        assert parts[0]['start']==0 and parts[-1]['end']==len(doc.text)
        assert all(a['end']==b['start'] for a,b in zip(parts,parts[1:]))


def test_graph_quotes_cannot_be_fabricated_or_ambiguous():
    corpus=build_synthetic_securitization_corpus()
    name=next(iter(corpus.documents))
    graph={'claims':[{'evidence':[{'document_id':name,'quote':'Not in any source'}]}]}
    with pytest.raises(ValueError,match='quote'):
        parse_candidate(json.dumps({'graph':graph}),'report',corpus)


def test_response_text_is_not_replaced_by_extractor():
    corpus=build_synthetic_securitization_corpus()
    result=parse_candidate('{"report_text":"invented","graph":null}','original',corpus)
    assert result['report_text']=='original'


def test_v3_does_not_depend_on_v4_graph_validity():
    corpus = build_synthetic_securitization_corpus()
    raw = json.dumps({'assertions': [], 'graph': {'claims': [
        {'evidence': [{'document_id': 'missing', 'quote': 'fabricated'}]}]}})
    candidate = parse_candidate(raw, 'original', corpus, include_graph=False)
    assert candidate['graph'] is None
    assert candidate['report_text'] == 'original'


def test_live_v4_requires_graph_even_when_prose_exists():
    result=evaluate_securitization_candidate('Review incomplete.',build_synthetic_securitization_corpus(),require_graph=True)
    assert not result.accepted
    assert 'MISSING_JUSTIFICATION_GRAPH' in {i['code'] for i in result.issues}


def test_live_v3_cannot_accept_empty_extraction():
    result = evaluate_securitization_candidate(
        {'report_text': 'Review incomplete.', 'assertions': []},
        build_synthetic_securitization_corpus(), gate_mode='v3_only', require_assertions=True)
    assert not result.accepted
    assert 'MISSING_STRUCTURED_ASSERTIONS' in {i['code'] for i in result.issues}


def test_retrieval_shares_budget_between_sources_and_preserves_offsets():
    corpus = SimpleNamespace(documents={
        name: SimpleNamespace(text=' '.join(str(n) for n in range(100, 120)))
        for name in ('email', 'contract', 'disclosure')})
    result = retrieve_windows('Values 101 and 112 differ from 119.', corpus, budget=900)
    assert {w['document_id'] for w in result['selected']} == set(corpus.documents)
    for window in result['selected']:
        assert window['text'] == corpus.documents[window['document_id']].text[window['start']:window['end']]
    assert len(json.dumps(result['selected']).encode()) <= 900


def test_retrieval_does_not_find_number_inside_other_number():
    corpus = SimpleNamespace(documents={'doc': SimpleNamespace(text='The value is 123 or 2.5.')})
    assert retrieve_windows('The value is 2.', corpus)['selected'] == []


def test_compact_packet_declares_truncation():
    packet=build_section_comparison_packet(build_synthetic_securitization_corpus(),max_section_chars=10)
    assert any(not d['full_text_included'] for d in packet['documents'])


def test_context_overflow_is_rejected_without_request(tmp_path):
    client=Client(SimpleNamespace(max_calls=3,max_seconds=30,context_length=1024),tmp_path)
    with pytest.raises(ValueError,match='context'):
        client.call('test','x'*2000)
    assert not client.calls


def test_reduction_does_not_silently_truncate_nonshrinking_notes():
    class Echo:
        def call(self,*args):return 'x'*6000
    with pytest.raises(RuntimeError,match='shrink'):
        compact_notes(Echo(),['x'*4000])


def test_resume_preserves_provenance_and_ignores_incomplete_scans(tmp_path):
    from run_three_arm_eval import reusable_scans, sha, save
    old = tmp_path / 'old'
    save(old / 'protocol.json', {'corpus_sha256': 'corpus'})
    complete = {'stage': 'scan', 'seconds': 1,
                'request': {'messages': ['same input'], 'model': 'model',
                            'temperature': 0, 'reasoning_effort': 'none'},
                'response': {'choices': [{'finish_reason': 'stop',
                                         'message': {'content': 'complete notes'}}]}}
    save(old / 'calls' / '0.json', complete)
    incomplete = json.loads(json.dumps(complete))
    incomplete['response']['choices'][0]['finish_reason'] = 'length'
    incomplete['response']['choices'][0]['message']['content'] = 'truncated'
    save(old / 'calls' / '1.json', incomplete)
    found = reusable_scans(old, 'corpus', 'model')
    assert list(found.values())[0][0] == 'complete notes'
    assert reusable_scans(old, 'corpus', 'other-model') == {}
    resumed = tmp_path / 'resumed'
    save(resumed / 'protocol.json', {'corpus_sha256': 'corpus'})
    save(resumed / 'reused_scan_calls.json', [
        {'source': str(old / 'calls' / '0.json'),
         'sha256': sha((old / 'calls' / '0.json').read_bytes())}])
    assert reusable_scans(resumed, 'corpus', 'model') == found
    (old / 'calls' / '0.json').write_text('{}')
    with pytest.raises(ValueError, match='provenance'):
        reusable_scans(resumed, 'corpus', 'model')


def test_complete_reduction_can_be_reused_only_for_identical_input(tmp_path):
    from run_three_arm_eval import SYSTEM, sha
    notes = ['x' * 4000]
    prompt = ('Consolidate these notes in at most 180 words. Preserve source '
              'citations, differing values and limitations. Do not invent.\n' + notes[0])
    key = sha([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}])
    client = Client(SimpleNamespace(), tmp_path)
    def reject_call(*args):
        raise AssertionError('Unexpected model call')
    client.call = reject_call
    cache = {key: ('retained summary', '/old/call.json', 'hash', {'seconds': 1})}
    assert compact_notes(client, notes, cache) == 'retained summary'
    assert len(client.reused_reductions) == 1
    with pytest.raises(AssertionError, match='Unexpected'):
        compact_notes(client, ['y' * 4000], cache)


def test_three_arm_flow_never_promotes_missing_graph_to_acceptance(tmp_path, monkeypatch):
    import run_three_arm_eval as runner
    corpus = build_synthetic_securitization_corpus()
    monkeypatch.setattr(runner, 'load_case_documents', lambda _: corpus)
    stages = []

    def fake_call(self, stage, prompt, max_tokens=700):
        stages.append(stage)
        self.calls.append({'stage': stage, 'seconds': 0.0})
        if stage.startswith('extract_'):
            return '{"assertions": [], "graph": null}'
        if stage == 'scan':
            return 'Partial source notes; further comparison required.'
        return 'The comparison remains incomplete; no conclusion is established.'

    monkeypatch.setattr(runner.Client, 'call', fake_call)
    args = SimpleNamespace(out=str(tmp_path / 'run'), case='cart_2024_2',
                           chunk_chars=4800, dry_run=False, reuse_scan=None,
                           iterations=2)
    runner.run(args)
    out = Path(args.out)
    assert json.loads((out / 'completion.json').read_text())['completed']
    baseline = json.loads((out / 'baseline.json').read_text())
    gated = json.loads((out / 'v3_v4.json').read_text())
    assert baseline['accepted'] is None
    assert baseline['final_disposition'] == 'NOT_VERIFIED'
    scores = json.loads((out / 'scores.json').read_text())
    assert scores[0]['accepted'] is None
    assert scores[0]['metrics']['false_accepts']['rate'] is None
    assert gated['accepted'] is False
    assert len(gated['attempts']) == 2
    assert 'repair_v3_v4' in stages
    assert any(i['code'] == 'MISSING_JUSTIFICATION_GRAPH'
               for attempt in gated['attempts'] for i in attempt['issues'])


def test_parse_candidate_resolves_repeated_and_whitespace_normalized_quotes():
    corpus = build_synthetic_securitization_corpus()
    name = next(iter(corpus.documents))
    base_text = corpus.v4_sources[name].text
    snippet = base_text[:60]
    # Repeated quote or whitespace-collapsed quote should resolve cleanly to exact span
    collapsed = ' '.join(snippet.split())
    graph = {'claims': {'c1': {'claim_id': 'c1', 'evidence': [{'document_id': name, 'quote': collapsed}]}}}
    parsed = parse_candidate(json.dumps({'graph': graph}), 'report', corpus)
    ref = parsed['graph']['claims']['c1']['evidence'][0]
    assert ref['quote'] == base_text[ref['start']:ref['end']]


def test_retrieve_windows_covers_sections_and_multiple_occurrences():
    doc_text = (
        "Table of Contents: Section 2.04 Interest Rate 6.75% ... "
        + ("filler " * 80)
        + "Operative Clause Section 2.04: Class C Notes coupon rate is 6.75% per annum."
    )
    corpus = SimpleNamespace(documents={'indenture': SimpleNamespace(text=doc_text, authority_rank=100)})
    result = retrieve_windows('See Section 2.04 for 6.75% coupon.', corpus, budget=2500)
    joined = '\n'.join(w['text'] for w in result['selected'])
    assert 'Operative Clause Section 2.04' in joined
    assert result['selected_windows'] >= 2


def test_percent_per_annum_canonicalization_and_subordinate_role():
    from claimlayer.verifiers.quote_quantity import canonicalize_unit
    from claimlayer import Assertion, Ledger, Policy, Source, verify
    from harness2_v4_pipeline import SecuritizationReportExtractor, build_structured_repair_feedback, evaluate_securitization_candidate

    assert canonicalize_unit('PERCENT_PER_ANNUM') == 'PERCENT'
    ledger = Ledger([Source(id='doc1', text='Class C Notes bear interest at 6.75% per annum.')])
    v = verify(
        assertions=[Assertion(subject='Class C Notes', predicate='coupon_rate', value='6.75', unit='PERCENT_PER_ANNUM', source_id='doc1', quote='Class C Notes bear interest at 6.75% per annum.')],
        ledger=ledger,
        policy=Policy(policy_id='p1'),
    )
    assert v.status == 'PASS'


def test_client_retries_transient_http_errors_with_backoff(tmp_path, monkeypatch):
    import io
    from urllib.error import HTTPError
    import run_three_arm_eval as runner

    attempts = {'count': 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}],
                'usage': {'total_tokens': 10},
            }).encode()

    def fake_urlopen(req, timeout=10):
        attempts['count'] += 1
        if attempts['count'] == 1:
            raise HTTPError(req.full_url, 429, 'Rate limit', {}, io.BytesIO(b''))
        return FakeResponse()

    monkeypatch.setattr(runner, 'urlopen', fake_urlopen)
    monkeypatch.setattr(runner.time, 'sleep', lambda s: sleeps.append(s))
    args = SimpleNamespace(
        base_url='http://localhost:9999/v1',
        model='test-model',
        thinking_mode='nvidia_off',
        context_length=8192,
        max_calls=5,
        max_seconds=60,
        timeout=10,
        max_retries=2,
        backoff_base_s=0.25,
    )
    client = runner.Client(args, tmp_path)
    out = client.call('test', 'hello', 50)
    assert out == 'ok'
    assert attempts['count'] == 2
    assert sleeps == [0.25]
    assert client.calls[0]['retries'] == 1

    corpus = build_synthetic_securitization_corpus()
    om_doc = next(d for d in corpus.documents.values() if d.authority_rank < 100)
    ext = SecuritizationReportExtractor()
    extracted = ext.extract(
        {
            'report_text': 'Report',
            'assertions': [{
                'subject': 'Class C Notes',
                'predicate': 'coupon_rate',
                'value': '6.50%',
                'unit': 'PERCENT',
                'role': 'subordinate_discrepancy',
                'source_id': om_doc.filename,
                'quote': om_doc.text[:80],
            }],
        },
        corpus,
    )
    assert extracted.assertions[0].predicate.startswith('om_discrepancy_')

    bad_candidate = {'report_text': 'Incomplete.', 'assertions': [{'subject': 'Class C Notes', 'predicate': 'coupon_rate', 'value': '99%', 'source_id': om_doc.filename, 'quote': 'missing'}]}
    verdict = evaluate_securitization_candidate(bad_candidate, corpus, gate_mode='v3_only', require_assertions=True)
    fb = build_structured_repair_feedback(verdict, bad_candidate, max_chars=1200)
    parsed_fb = json.loads(fb)
    assert len(fb) <= 1200
    assert parsed_fb['disposition'] == 'RETRY'
    assert parsed_fb['failing_assertions']


def test_day_count_predicate_and_pandoc_dollar_quote_grounding():
    from claimlayer import Assertion, Ledger, Policy, Source, verify
    ledger = Ledger([Source(id='doc1', text='Class C Notes: \\$50,000,000 --- 6.75% per annum, calculated on a 30/360 day count basis.')])
    v = verify(
        assertions=[
            Assertion(subject='Deal', predicate='governing_day_count', value='30/360', source_id='doc1', quote='calculated on a 30/360 day count basis.'),
            Assertion(subject='Class C Notes', predicate='note_principal', value='$50,000,000', source_id='doc1', quote='Class C Notes: $50,000,000 --- 6.75% per annum'),
        ],
        ledger=ledger,
        policy=Policy(policy_id='p2'),
    )
    assert v.status == 'PASS'


def test_client_retries_transient_http_errors_with_backoff(tmp_path, monkeypatch):
    import io
    from urllib.error import HTTPError
    import run_three_arm_eval as runner

    attempts = {'count': 0}
    sleeps = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}],
                'usage': {'total_tokens': 10},
            }).encode()

    def fake_urlopen(req, timeout=10):
        attempts['count'] += 1
        if attempts['count'] == 1:
            raise HTTPError(req.full_url, 429, 'Rate limit', {}, io.BytesIO(b''))
        return FakeResponse()

    monkeypatch.setattr(runner, 'urlopen', fake_urlopen)
    monkeypatch.setattr(runner.time, 'sleep', lambda s: sleeps.append(s))
    args = SimpleNamespace(
        base_url='http://localhost:9999/v1',
        model='test-model',
        thinking_mode='nvidia_off',
        context_length=8192,
        max_calls=5,
        max_seconds=60,
        timeout=10,
        max_retries=2,
        backoff_base_s=0.25,
    )
    client = runner.Client(args, tmp_path)
    out = client.call('test', 'hello', 50)
    assert out == 'ok'
    assert attempts['count'] == 2
    assert sleeps == [0.25]
    assert client.calls[0]['retries'] == 1
