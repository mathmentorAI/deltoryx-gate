"""Standalone deterministic v4 library contracts on the preserved structured inputs.

No language model, no extraction, no network. Only the v4 arm runs here.
Historical multi-arm results and their protocol are retained as provenance.
This measures engine contracts (what each library computes from given evidence),
not extraction quality or end-to-end answer quality.

Block A: the 20 shared functional contracts of benchmarks/functional_parity (unchanged fixtures).
Block B: 11 contracts taken from the revised paper (19 Oct 2026), fixed before running.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
# SQLite needs to create and delete journal files; pass --out to a local directory when the
# repository lives on a mount that forbids deletion, then copy the finished results.
OUT = Path(sys.argv[sys.argv.index('--out') + 1]).resolve() if '--out' in sys.argv else REPO / 'benchmarks' / 'library_contracts_v4_results'

from claimlayer import v4  # noqa: E402
from claimlayer.v4 import StructuredQuery  # noqa: E402

AS_OF = '2026-01-01'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')


def number(x):
    if isinstance(x, dict) and 'value' in x:
        x = x['value']
    try:
        return float(str(x).split()[0])
    except (TypeError, ValueError):
        return x


# ======================================================================= Block A
AMOUNT_VOCAB = v4.Vocabulary('functional-parity', '1', (
    v4.PredicateSpec('amount', v4.FieldSpec('quantity', dimensions=('currency:EUR', 'currency:USD'))),))


def v4_shared(case, folder):
    db = folder / 'v4.sqlite'
    e = v4.EvidenceLayer(db, vocabulary=AMOUNT_VOCAB)
    ingestion, ids = [], {}
    for c in case['items']:
        text = f"ACME amount is {c['value']} {c['unit']}."
        d = v4.SourceDocument('source-' + c['id'] + '.txt', text, authority_rank=c['rank'], weight=.9)
        e.ingest(d)
        scope = {'event_id': c['event_id']} if c.get('event_id') and case['operation'] != 'resolve' else {}
        out = e.ingest_units([{'proposal': {'entity': c['entity'], 'predicate': c['predicate'], 'value': c['value'],
                                            'unit': c['unit'], 'assertion_kind': 'fact', 'scope': scope,
                                            'valid_from': c.get('valid_from'), 'valid_until': c.get('valid_until')},
                               'evidence': [v4.EvidenceRef(d.document_id, d.revision, 0, len(text), text)]}],
                             attestor='library-contract-fixture;not-model-verified')
        ids[c['id']] = out['unit_ids'][0]
        ingestion.append(out)
    dependency = None
    if case.get('dependency'):
        mode = case['dependency']
        premise = 'unknown' if mode == 'missing-premise' else ids['root']
        value = 31 if mode == 'changed-value' else 30
        try:
            derived = e.derive({'entity': 'ACME', 'predicate': 'amount', 'value': value, 'unit': 'EUR',
                                'assertion_kind': 'fact'}, [premise], 'identity')
            query = StructuredQuery('ACME', 'amount', as_of=case.get('as_of', AS_OF))
            basis, reason = e._eligible(e.unit(derived), e._query(query), 'report')
            dependency = {'eligible': basis is not None, 'reason': reason, 'creation_rejected': False}
        except (ValueError, KeyError) as exc:
            dependency = {'eligible': False, 'reason': str(exc), 'creation_rejected': True}
    if case.get('reopen'):
        e = v4.EvidenceLayer(db, vocabulary=AMOUNT_VOCAB)
    q = {'entity': 'ACME', 'predicate': 'amount', 'operation': case['operation'], 'as_of': case.get('as_of', AS_OF)}
    response = e.ask(q)
    reopened = v4.EvidenceLayer(db, vocabulary=AMOUNT_VOCAB).ask(q)
    strip = lambda r: {k: v for k, v in r.items() if k not in {'response_id', 'certificate'}}
    tampered = copy.deepcopy(response)
    if tampered.get('results'):
        tampered['results'][0]['value'] = 'TAMPERED'
    else:
        tampered['reason'] = 'TAMPERED'
    return {'response': response, 'integrity': e.verify_response_integrity(response),
            'tamper_detected': not e.verify_response_integrity(tampered), 'reopened_equal': strip(response) == strip(reopened),
            'dependency': dependency, 'ingestion': ingestion}


def v4_evaluate(case, result):
    if case['kind'] == 'dependency':
        return result['dependency']['eligible'] == (case['dependency'] == 'valid')
    response = result['response']
    if case['kind'] == 'abstain':
        return response['abstain']
    accepted = [r for r in response['results'] if r['status'] == 'accepted']
    if case['operation'] == 'list':
        return sorted(number(r['value']) for r in accepted) == sorted(case['expected'])
    return bool(accepted) and number(accepted[0]['value']) == case['expected']


# ======================================================================= Block B
def item(**kw):
    base = {'entity': 'ACME', 'predicate': 'payment_term', 'unit': 'days', 'rank': 100, 'weight': 0.9,
            'group': None, 'basis': 'explicit', 'conditions': [], 'event_id': None, 'filename': None,
            'cardinality': 'single'}
    base.update(kw)
    return base


PAPER_CASES = [
    {'id': 'P01-dependent-copies-do-not-corroborate', 'paper': 'Sec. 5.2 dependence groups: b_{g,d}=max',
     'items': [item(id='a', value=30, weight=0.8, group='seller-memo'), item(id='b', value=30, weight=0.7, group='seller-memo')],
     'operation': 'resolve', 'gold': {'top_value': 30, 'selected_count': 1}},
    {'id': 'P02-equal-support-conflict-is-unresolved', 'paper': 'Sec. 5.3 S=O gives Q=0.5 and fails the margin',
     'items': [item(id='a', value=30, weight=1.0), item(id='b', value=45, weight=1.0)],
     'operation': 'resolve', 'gold': {'settled': False}},
    {'id': 'P03-explicit-statement-outweighs-weak-inference', 'paper': 'v2 epistemic weights inside a_i',
     'items': [item(id='a', value=30, basis='explicit'), item(id='b', value=45, basis='common_sense')],
     'operation': 'resolve', 'gold': {'top_value': 30}},
    {'id': 'P04-blocked-inference-basis-is-excluded', 'paper': 'v2 BLOCKED_INFERENCE_TYPES',
     'items': [item(id='a', value=45, basis='narrative')],
     'operation': 'resolve', 'gold': {'abstain': True}},
    {'id': 'P05-authority-from-application-metadata-not-filename', 'paper': 'Sec. 6.3 authority controlled outside untrusted content',
     'items': [item(id='a', value=45, rank=100, weight=0.8, filename='annex-amended-contract.txt'),
               item(id='b', value=30, rank=50, weight=0.9, filename='base-terms.txt')],
     'operation': 'resolve', 'gold': {'top_value': 45}},
    {'id': 'P06-multivalued-values-are-compatible', 'paper': 'Sec. 4.2 compatible alternatives are not opposition',
     'items': [item(id='a', predicate='office', value='Madrid', unit=None, cardinality='multi'),
               item(id='b', predicate='office', value='Barcelona', unit=None, cardinality='multi')],
     'operation': 'list', 'gold': {'values': ['Barcelona', 'Madrid'], 'contradiction': False}},
    {'id': 'P07-unit-labels-canonicalized-under-declared-policy', 'paper': 'Sec. 4.1 canonicalization under a versioned policy',
     'items': [item(id='a', predicate='sofr_floor', value=0.75, unit='percent_per_annum', group='original'),
               item(id='b', predicate='sofr_floor', value=0.75, unit='percent per annum', group='markup')],
     'operation': 'resolve', 'gold': {'top_value': 0.75, 'settled': True, 'selected_count': 2}},
    {'id': 'P08-declared-exact-conversion-before-aggregation', 'paper': 'Sec. 7 conversions require recorded rules',
     'items': [item(id='a', predicate='term', value=1, unit='years', event_id='e1'),
               item(id='b', predicate='term', value=6, unit='months', event_id='e2')],
     'operation': 'sum', 'gold': {'aggregate': 18, 'aggregate_unit': 'months'}},
    {'id': 'P09-conditions-stay-attached-to-answers', 'paper': 'Sec. 3.1 context chi is part of the claim',
     'items': [item(id='a', predicate='overtime_multiplier', value=1.5, unit='times',
                    conditions=[{'field': 'employee_is_nonexempt', 'operator': 'eq', 'value': True}])],
     'operation': 'resolve', 'gold': {'top_value': 1.5, 'conditional_label': True}},
    {'id': 'P10-weakest-link-path-ranking', 'paper': 'Sec. 6.2 R(pi)=gamma^h min s, example 0.676875',
     'items': [item(id='a', entity='Alpha Holding', predicate='subsidiary_of', value='Beta Sub', unit=None, weight=0.9),
               item(id='b', entity='Beta Sub', predicate='subsidiary_of', value='Gamma Sub', unit=None, weight=0.8),
               item(id='c', entity='Gamma Sub', predicate='director', value='Jane Roe', unit=None, weight=0.75)],
     'operation': 'path', 'path': {'start': 'Alpha Holding', 'target_predicate': 'director', 'target': 'Jane Roe'},
     'gold': {'path_score': 0.676875}},
    # Added after the initial run (amendment 2): reversed value order, so a pass cannot come from tie-breaking.
    {'id': 'P03b-explicit-outweighs-inference-reversed-order', 'paper': 'v2 epistemic weights inside a_i (post-hoc sensitivity)',
     'items': [item(id='a', value=45, basis='explicit'), item(id='b', value=30, basis='common_sense')],
     'operation': 'resolve', 'gold': {'top_value': 45}},
]

PAPER_VOCAB = v4.Vocabulary('paper-contracts', '1', (
    v4.PredicateSpec('payment_term', v4.FieldSpec('quantity', dimensions=('duration_calendar_days',))),
    v4.PredicateSpec('office', v4.FieldSpec('string'), cardinality='multi'),
    v4.PredicateSpec('sofr_floor', v4.FieldSpec('quantity', dimensions=('annual_rate',))),
    v4.PredicateSpec('term', v4.FieldSpec('quantity', dimensions=('duration_months',))),
    v4.PredicateSpec('overtime_multiplier', v4.FieldSpec('quantity', dimensions=('multiplier',)),
                     conditions=(('employee_is_nonexempt', v4.FieldSpec('boolean')),)),
    v4.PredicateSpec('subsidiary_of', v4.FieldSpec('entity')),
    v4.PredicateSpec('director', v4.FieldSpec('entity')),
))


def doc_text(c):
    return f"{c['entity']} {c['predicate']} {c['value']} {c['unit'] or ''}".strip() + '.'


def observe_v4(case, folder):
    e = v4.EvidenceLayer(folder / 'v4.sqlite', vocabulary=PAPER_VOCAB)
    for c in case['items']:
        text = doc_text(c)
        d = v4.SourceDocument(c['filename'] or f"source-{c['id']}.txt", text, authority_rank=c['rank'],
                              weight=c['weight'], dependence_group=c['group'])
        e.ingest(d)
        e.ingest_units([{'proposal': {'entity': c['entity'], 'predicate': c['predicate'], 'value': c['value'],
                                      'unit': c['unit'], 'assertion_kind': 'fact', 'conditions': c['conditions'],
                                      'scope': {'event_id': c['event_id']} if c['event_id'] else {}},
                         'basis': c['basis'], 'evidence': [v4.EvidenceRef(d.document_id, d.revision, 0, len(text), text)]}],
                       attestor='library-contract-fixture')
    if case['operation'] == 'path':
        p = case['path']
        response = e.find_paths(p['start'], as_of=AS_OF)
        scores = [x['path_score'] for x in response['paths'] if x['nodes'][-1] == p['target']]
        return {'path_score': max(scores) if scores else None}, response
    c0 = case['items'][0]
    response = e.ask({'entity': c0['entity'], 'predicate': c0['predicate'], 'operation': case['operation'], 'as_of': AS_OF})
    accepted = [r for r in response['results'] if r['status'] == 'accepted']
    results = response['results']
    top = accepted[0] if accepted else (results[0] if results else None)
    obs = {'abstain': response['abstain'], 'settled': bool(accepted) and len(accepted) == len(results),
           'top_value': number(top['value']) if top and case['operation'] != 'sum' else None,
           'selected_count': len(top['selected_unit_ids']) if top and 'selected_unit_ids' in top else None,
           'values': sorted(r['value'] for r in accepted) if case['operation'] == 'list' else None,
           'contradiction': any(c['relation'] in {'opposing', 'outranks'} for r in results for c in r.get('competitors', [])),
           'conditional_label': bool(top and top.get('conditional') and top.get('pending_conditions'))}
    if case['operation'] == 'sum':
        obs['aggregate'] = number(results[0]['value']) if accepted else None
        obs['aggregate_unit'] = results[0]['unit'] if accepted else None
    return obs, response


def passes(gold, obs):
    checks = {}
    for key, expected in gold.items():
        actual = obs.get(key)
        if key in {'top_value', 'aggregate', 'path_score'} and isinstance(expected, (int, float)):
            # path scores: v2 rounds its output to 4 decimals, so compare at 1e-4 (amendment 1)
            tolerance = 1e-4 if key == 'path_score' else 1e-6
            checks[key] = actual is not None and isinstance(actual, (int, float)) and abs(actual - expected) < tolerance
        else:
            checks[key] = actual == expected
    return all(checks.values()), checks


def run():
    if (OUT / 'protocol.json').exists():
        raise SystemExit('Preserve the existing run; choose a new results directory')
    shared = json.loads((Path(__file__).parent / "cases_block_a.json").read_text())
    save(OUT / 'cases_block_a.json', shared)
    save(OUT / 'cases_block_b.json', PAPER_CASES)
    save(OUT / 'protocol.json', {
        'execution_mode': 'standalone-v4; historical v2/v3 arms not executed',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'runner_sha256': sha(Path(__file__).read_bytes()),
        'block_a_sha256': sha((OUT / 'cases_block_a.json').read_bytes()),
        'block_b_sha256': sha((OUT / 'cases_block_b.json').read_bytes()),
        'library_sha256': {str(f.relative_to(REPO)): sha(f.read_bytes()) for folder in ['src/claimlayer', 'src/core']
                           for f in sorted((REPO / folder).rglob('*.py'))},
        'scope': 'Standalone v4 engine contracts on application-supplied typed records. '
                 'No model, extraction or end-to-end accuracy evaluation. Block A loads the preserved '
                 '20 fixtures; block B runs the 11 preserved paper contracts. Historical v2/v3 '
                 'results in the input directory were produced in another repository and are not rerun.',
        'historical_protocol': 'benchmarks/library_contracts_v4/protocol.json',
    })
    rows = []
    for case in shared:
        row = {'block': 'A', 'id': case['id'], 'kind': case['kind'], 'arms': {}}
        for arm, fn, evaluate in [('v4', v4_shared, v4_evaluate)]:
            folder = OUT / 'cases' / case['id'] / arm
            folder.mkdir(parents=True, exist_ok=True)
            try:
                result = fn(case, folder)
                result['contract_pass'] = evaluate(case, result)
            except Exception as exc:
                result = {'error_type': type(exc).__name__, 'error': str(exc), 'trace': traceback.format_exc(), 'contract_pass': False}
            save(folder / 'result.json', result)
            row['arms'][arm] = {'pass': result['contract_pass'], 'error_type': result.get('error_type'),
                                'integrity': result.get('integrity'), 'tamper_detected': result.get('tamper_detected')}
        rows.append(row)
        print(row['id'], {a: r['pass'] for a, r in row['arms'].items()}, flush=True)
    for case in PAPER_CASES:
        row = {'block': 'B', 'id': case['id'], 'paper': case['paper'], 'gold': case['gold'], 'arms': {}}
        for arm, fn in [('v4', observe_v4)]:
            folder = OUT / 'cases' / case['id'] / arm
            folder.mkdir(parents=True, exist_ok=True)
            try:
                obs, response = fn(case, folder)
                ok, checks = passes(case['gold'], obs)
                result = {'observation': obs, 'checks': checks, 'pass': ok, 'response': response}
            except Exception as exc:
                result = {'error_type': type(exc).__name__, 'error': str(exc), 'trace': traceback.format_exc(), 'pass': False, 'checks': {}}
            save(folder / 'result.json', result)
            row['arms'][arm] = {'pass': result['pass'], 'checks': result['checks'], 'error_type': result.get('error_type'),
                                'observation': {k: v for k, v in result.get('observation', {}).items() if k != 'ingestion'}}
        rows.append(row)
        print(row['id'], {a: r['pass'] for a, r in row['arms'].items()}, flush=True)
    summary = {'blocks': {}, 'per_case': rows,
               'scope': 'Contract outcomes on fixed structured fixtures; not accuracy, extraction quality or a global score.'}
    for block in ['A', 'B']:
        subset = [r for r in rows if r['block'] == block]
        summary['blocks'][block] = {'cases': len(subset), **{arm: dict(Counter('pass' if r['arms'][arm]['pass'] else 'fail' for r in subset))
                                                             for arm in ['v4']}}
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary['blocks'], indent=1))
    if any(not row['arms']['v4']['pass'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    run()
