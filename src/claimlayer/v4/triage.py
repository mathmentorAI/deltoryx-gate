"""Verification triage: tell a human which parts of a draft they still need to read, and why.

The bottleneck with agent output is human attention. ``verify_draft`` reads a draft into typed claims, checks
each one deterministically against an evidence ledger built from the sources, and sorts the draft's segments
(sentences, table rows) into lanes:

* ``contradicted``      — a claim conflicts with an accepted source value (both spans are shown);
* ``sources_disagree``  — the sources themselves conflict on the claim;
* ``unverified``        — a claim has no accepted source value, or (with ``guard``) a number or date in the
                          segment is not accounted for by a supported claim;
* ``verified``          — every located claim is supported, and every number/date is accounted for;
* ``no_claim``          — no checkable claim was read there (headings, analysis, advice).

A reviewer who trusts ``verified`` reads everything else; one who only wants alarms reads the first two lanes.
Nothing here certifies truth: it is conditional on the readings, the vocabulary and the ledger.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
import re

from ..semantic.models import SourceDocument
from .autovocab import with_entities
from .engine import EvidenceLayer

LANES = ('contradicted', 'sources_disagree', 'unverified', 'verified', 'no_claim')
READING_WPM = 250  # the reading speed Anthropic's employment report uses to convert output into reading time
_STATUS_LANE = {'supported': 'verified', 'contradicted': 'contradicted', 'unresolved_conflict': 'sources_disagree'}

# ------------------------------------------------------------------ segments
# Abbreviations after which a capitalised word does not start a new sentence. Company suffixes (Inc., LLC.)
# are not listed: a capitalised word after them usually does start one.
_ABBREVIATIONS = {'mr', 'ms', 'mrs', 'dr', 'no', 'nos', 'u.s', 'e.g', 'i.e', 'v', 'vs', 'sec', 'secs', 'art', 'st',
                  'jr', 'sr', 'approx', 'cf', 'al', 'p', 'pp', 'para', 'fig', 'u.k', 'e.u', 'jan', 'feb', 'mar',
                  'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec'}
_BREAK = re.compile(r'(?<=[.!?;])["”’)\]]?\s+(?=["“(\[$€£]?[A-Z0-9])')


def segment_text(text: str):
    """Split into reading units: sentences within lines; a table row is one unit. Offsets index into text."""
    segments = []
    for line in re.finditer(r'[^\n]+', text):
        body, base = line.group(), line.start()
        if not re.search(r'[A-Za-z0-9]', body) or re.fullmatch(r'[\s|:\-+=]*', body):
            continue
        if body.lstrip().startswith('|'):
            segments.append((base, line.end()))
            continue
        start = 0
        for match in _BREAK.finditer(body):
            before = body[start:match.start()].rstrip('"”’)]')
            word = re.search(r'([A-Za-z.]+)[.!?;]$', before)
            if word and (word.group(1).lower().rstrip('.') in _ABBREVIATIONS or re.fullmatch(r'[A-Z]', word.group(1))):
                continue
            segments.append((base + start, base + match.start()))
            start = match.end()
        segments.append((base + start, line.end()))
    out = []
    for start, end in segments:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start and re.search(r'[A-Za-z0-9]', text[start:end]):
            out.append({'index': len(out), 'start': start, 'end': end, 'text': text[start:end],
                        'words': len(text[start:end].split())})
    return out


# ------------------------------------------------------- numbers and dates
_MONTHS = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                       'september', 'october', 'november', 'december'], 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
_MONTHS['sept'] = 9
_MONTH = r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?'
_DATE_PATTERNS = [
    (re.compile(_MONTH + r'\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', re.I), 'mdy'),
    (re.compile(r'(\d{1,2})(?:st|nd|rd|th)?\s+' + _MONTH + r',?\s+(\d{4})', re.I), 'dmy'),
    (re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'), 'iso'),
    (re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'), 'slash'),
    (re.compile(_MONTH + r',?\s+(\d{4})\b', re.I), 'my'),
]
_NUMBER = re.compile(r'(?<![\w.])([$€£])?\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?(%|percent\b|per cent\b)?'
                     r'(\s?(?:million|billion|thousand|m\b|bn\b|k\b))?', re.I)
_REFERENCE = re.compile(r'(section|sections|§|article|clause|exhibit|schedule|rule|item|part|appendix|annex|'
                        r'paragraph|para\.|page|p\.|note|footnote|tier|issue|finding|comp|no\.)\s*\(?$', re.I)
_LIST_MARK = re.compile(r'^\s*(?:\(?\d{1,2}[.)]|\(?[a-z][.)]|[ivx]{1,4}[.)])\s')


def _iso(year, month, day=None):
    try:
        return date(int(year), int(month), int(day)).isoformat() if day else f'{int(year):04d}-{int(month):02d}'
    except ValueError:
        return None


def value_mentions(segment: str):
    """Numbers and dates written in a segment, each with the canonical forms it may stand for.

    Section-style references, list markers and numbers inside dates are not values."""
    mentions, taken = [], []
    for pattern, kind in _DATE_PATTERNS:
        for m in pattern.finditer(segment):
            if any(s < m.end() and m.start() < e for s, e in taken):
                continue
            g = m.groups()
            if kind == 'mdy':
                forms = {_iso(g[2], _MONTHS[g[0].lower().rstrip('.')], g[1])}
            elif kind == 'dmy':
                forms = {_iso(g[2], _MONTHS[g[1].lower().rstrip('.')], g[0])}
            elif kind == 'iso':
                forms = {_iso(*g)}
            elif kind == 'slash':
                forms = {_iso(g[2], g[0], g[1]), _iso(g[2], g[1], g[0])}
            else:
                forms = {_iso(g[1], _MONTHS[g[0].lower().rstrip('.')])}
            forms.discard(None)
            if forms:
                mentions.append({'kind': 'date', 'text': m.group(), 'forms': sorted(forms)})
                taken.append((m.start(), m.end()))
    list_mark = _LIST_MARK.match(segment)
    for m in _NUMBER.finditer(segment):
        if any(s < m.end() and m.start() < e for s, e in taken):
            continue
        if list_mark and m.start() < list_mark.end():
            continue
        if _REFERENCE.search(segment[max(0, m.start() - 14):m.start()]):
            continue
        try:
            number = Decimal(m.group(2).replace(',', '') + (m.group(3) or ''))
        except InvalidOperation:
            continue
        forms = {number}
        scale = (m.group(5) or '').strip().lower()
        if scale:
            forms.add(number * {'million': 10 ** 6, 'm': 10 ** 6, 'billion': 10 ** 9, 'bn': 10 ** 9,
                                'thousand': 10 ** 3, 'k': 10 ** 3}[scale])
        if m.group(4):
            forms.add(number / 100)
        kind = 'year' if not (m.group(1) or m.group(3) or m.group(4) or scale) and 1900 <= number <= 2100 else 'number'
        mentions.append({'kind': kind, 'text': m.group().strip(), 'forms': sorted({_canon(f) for f in forms})})
    return mentions


def _canon(number: Decimal) -> str:
    text = format(number.normalize(), 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def claim_values(value):
    """Numbers and ISO dates a claim states, from the reader's raw value (before unit conversion)."""
    out = set()
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        try:
            out.add(_canon(Decimal(str(value))))
        except InvalidOperation:
            pass
    elif isinstance(value, str):
        if re.fullmatch(r'\d{4}(-\d{2}(-\d{2}(T.*)?)?)?', value.strip()):
            out.add(value.strip()[:10])
        for token in re.findall(r'\d[\d,]*(?:\.\d+)?', value):
            try:
                out.add(_canon(Decimal(token.replace(',', ''))))
            except InvalidOperation:
                pass
    elif isinstance(value, dict):
        for v in value.values():
            out |= claim_values(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            out |= claim_values(v)
    return out


def _covered(mention, values):
    if mention['kind'] == 'date':
        # the claim must be at least as precise as the text: a supported month does not vouch for a day
        return any(v.startswith(f) for f in mention['forms'] for v in values if '-' in v)
    if mention['kind'] == 'year':
        return any(v.startswith(mention['forms'][0]) for v in values) or mention['forms'][0] in values
    return any(f in values for f in mention['forms'])


# ------------------------------------------------------------------ triage
def _sources_of(result):
    refs = []
    for evidence in (result or {}).get('evidence', []):
        for ref in evidence.get('evidence', []):
            refs.append({'document_id': ref['document_id'], 'start': ref['start'], 'end': ref['end'],
                         'quote': ref['quote']})
    return refs


def triage(ledger: EvidenceLayer, draft_layer: EvidenceLayer, draft_text: str, *, guard: bool = True,
           max_segments_per_claim: int = 2):
    segments = segment_text(draft_text)
    for segment in segments:
        segment['claims'] = []
    unlocated = []
    for unit in draft_layer.units:
        result = ledger.check_claim({**unit.proposal, 'entity': unit.proposition['entity']}, validity='ignore')
        status, note = result['status'], None
        if status == 'not_supported' and {d['path'] for d in result.get('differences', [])} == {'/assertion_kind'}:
            # same value, conditions and scope; only the speech act was read differently (e.g. the draft reports
            # as a fact what a source alleges). The value is supported; the difference is shown, not hidden.
            status, note = 'supported', 'assertion_kind_differs'
        claim = {'unit_id': unit.unit_id, 'status': status, 'check_status': result['status'], 'note': note,
                 'relation': result.get('relation'),
                 'proposition': {k: result.get('proposition', {}).get(k) for k in ('entity', 'predicate', 'value', 'unit', 'polarity')},
                 'raw_value': unit.proposal.get('value'), 'differences': result.get('differences', []),
                 'sources': _sources_of(result.get('matched_result')),
                 'draft_span': [(r.start, r.end) for r in unit.evidence]}
        hits = [s for s in segments if any(s['start'] < end and start < s['end'] for start, end in claim['draft_span'])]
        if not hits or len(hits) > max_segments_per_claim:
            unlocated.append(claim)
            continue
        for segment in hits:
            segment['claims'].append(claim)
    for segment in segments:
        lanes = {_STATUS_LANE.get(c['status'], 'unverified') for c in segment['claims']}
        segment['uncovered'] = []
        if not lanes:
            segment['lane'] = 'no_claim'
            continue
        lane = next(l for l in LANES if l in lanes)
        if lane == 'verified' and guard:
            values = set()
            for c in segment['claims']:
                values |= claim_values(c['raw_value'])
            segment['uncovered'] = [m['text'] for m in value_mentions(segment['text']) if not _covered(m, values)]
            if segment['uncovered']:
                lane = 'unverified'
        segment['lane'] = lane
    lanes = {lane: {'segments': 0, 'words': 0} for lane in LANES}
    for segment in segments:
        lanes[segment['lane']]['segments'] += 1
        lanes[segment['lane']]['words'] += segment['words']

    def evidence_words(selected):
        quotes = {(r['document_id'], r['start'], r['end']): r['quote'] for s in selected for c in s['claims']
                  if c['status'] in {'contradicted', 'unresolved_conflict'} for r in c['sources']}
        return sum(len(q.split()) for q in quotes.values())

    alarms = [s for s in segments if s['lane'] in {'contradicted', 'sources_disagree'}]
    to_read = [s for s in segments if s['lane'] != 'verified']
    total = sum(s['words'] for s in segments)
    reading = {
        'read_all': {'words': total},
        'skip_verified': {'words': sum(s['words'] for s in to_read) + evidence_words(alarms)},
        'alarms_only': {'words': sum(s['words'] for s in alarms) + evidence_words(alarms)},
    }
    for value in reading.values():
        value['minutes'] = round(value['words'] / READING_WPM, 2)
    return {'segments': segments, 'unlocated_claims': unlocated, 'lanes': lanes, 'reading': reading,
            'guard': guard, 'claims': sum(len(s['claims']) for s in segments) + len(unlocated),
            'reading_wpm': READING_WPM}


def draft_vocabulary(ledger: EvidenceLayer, *, max_entities: int = 200, per_predicate: int = 12):
    """The ledger's vocabulary plus the entity names its sources use, most frequent first, overall and per
    predicate, so that the draft is read with the sources' names for the same things."""
    overall, by_predicate = Counter(), {}
    for unit in ledger.units:
        entity, predicate = unit.proposition['entity'], unit.proposition['predicate']
        overall[entity] += 1
        by_predicate.setdefault(predicate, Counter())[entity] += 1
    entities = [e for e, _ in overall.most_common(max_entities)]
    hints = {p: [e for e, _ in c.most_common(per_predicate)] for p, c in by_predicate.items()}
    return with_entities(ledger.vocabulary, entities, hints)


def verify_draft(ledger: EvidenceLayer, draft_text: str, *, reader, draft_id: str = 'draft', db_path=None,
                 guard: bool = True):
    """Read a draft with the ledger's vocabulary (plus the sources' entity names) and triage it."""
    layer = EvidenceLayer(db_path, vocabulary=draft_vocabulary(ledger), reader=reader, policy=ledger.policy)
    reading = layer.ingest(SourceDocument(draft_id, draft_text, weight=0.9))
    report = triage(ledger, layer, draft_text, guard=guard)
    report['draft_reading'] = reading
    return report
