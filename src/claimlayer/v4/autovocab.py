"""Model-proposed vocabulary, validated by code.

When an application has no hand-written vocabulary, a model may propose typed predicates from the assignment
and source excerpts. Code admits only well-formed predicates and records every dropped proposal. The rules are
generic: what can be checked against sources (facts stated there), at the granularity of the thing a value
belongs to; never the author's evaluations.
"""
from __future__ import annotations

from dataclasses import replace
import re

from ..semantic.models import digest
from .vocabulary import STANDARD_UNITS, FieldSpec, PredicateSpec, Vocabulary, label_key

VALUE_TYPES = ('quantity', 'bound', 'boolean', 'enum', 'set', 'string', 'entity', 'number', 'date')
CONDITION_TYPES = ('boolean', 'number', 'enum', 'entity', 'date')
# Evaluations cannot be checked against sources; predicates named after them are dropped (documented, generic).
EVALUATIVE_STEMS = ('priority', 'severity', 'recommend', 'assessment', 'risk_level', 'risk_rating')

VOCAB_PROMPT = '''You design a typed vocabulary that software will use to check statements against source
documents. You receive an assignment and, for each source document, its beginning, passages taken from
across the rest of it and its headings. Cover the whole documents, not only their first pages. Do not perform
the assignment.

Propose predicates for facts that the source documents state and that a reader could check against them:
amounts, prices, rates, percentages, durations, deadlines, dates, thresholds, counts, identifiers (registration,
account, section or clause numbers), parties and their roles, governing law, obligations, rights, permissions,
prohibitions, and the conditions or exceptions that limit them.

Rules:
- Never propose predicates for evaluations, opinions or advice (priority, severity, risk rating, whether a
  term is market standard, recommended response): they cannot be checked against sources.
- Name a predicate after an attribute of a specific thing (base_rent, notice_period, governing_law,
  share_price), not after a document or an analysis.
- Attributes that an entity can hold several of at once (roles, titles, addresses, products) use
  cardinality "multi".
- Conditions limit when a value applies (a scenario, an election, a threshold, a period). Never use a
  condition to restate which entity or document a statement is about.

Return JSON only: {"predicates":[{"name":snake_case,"description":string,"value_type":string,
"dimensions":[string],"enum":[string],"cardinality":"single"|"multi",
"conditions":[{"field":snake_case,"type":string,"enum":[string]}]}]}

value_type is one of quantity, bound, boolean, enum, set, string, entity, number, date.
- quantity and bound need "dimensions" chosen only from allowed_dimensions; a bound is a threshold with an
  operator (lt, le, gt, ge). date values are ISO dates at the precision the text states.
- enum and set need their members in "enum".
- Prefer quantity, bound, boolean, enum or date over string: strings are compared verbatim.
- cardinality single means one value per entity in a given context.
- condition type is one of boolean, number, enum, entity, date.
At most max_predicates predicates and at most four conditions per predicate.'''


def _snake(name):
    return isinstance(name, str) and re.fullmatch(r'[a-z][a-z0-9_]{1,63}', name) is not None


def allowed_dimensions():
    out = {}
    for unit in STANDARD_UNITS.describe():
        out.setdefault(unit['dimension'], []).append(unit['name'])
    return out


def build_vocabulary(proposal, name: str):
    """Validate model-proposed predicates into a Vocabulary; invalid entries are dropped and recorded."""
    dims = allowed_dimensions()
    accepted, dropped = [], []
    for raw in (proposal.get('predicates') if isinstance(proposal, dict) else None) or []:
        try:
            if not isinstance(raw, dict) or not _snake(raw.get('name')):
                raise ValueError('invalid name')
            if any(stem in raw['name'] for stem in EVALUATIVE_STEMS):
                raise ValueError('evaluative predicate')
            vt = raw.get('value_type')
            if vt not in VALUE_TYPES:
                raise ValueError(f'value_type {vt!r}')
            kwargs = {}
            if vt in {'quantity', 'bound'}:
                d = tuple(sorted({x for x in raw.get('dimensions') or [] if x in dims}))
                if not d:
                    raise ValueError('no allowed dimension')
                kwargs['dimensions'] = d
            if vt in {'enum', 'set'}:
                members = tuple(dict.fromkeys(str(x) for x in raw.get('enum') or [] if str(x).strip()))
                if not members:
                    raise ValueError('enum without members')
                if len({label_key(m) for m in members}) != len(members):
                    raise ValueError('ambiguous enum members')
                kwargs['enum'] = members
            conditions = []
            for c in (raw.get('conditions') or [])[:4]:
                if not isinstance(c, dict) or not _snake(c.get('field')) or c.get('type') not in CONDITION_TYPES:
                    continue
                ckw = {}
                if c['type'] == 'enum':
                    members = tuple(dict.fromkeys(str(x) for x in c.get('enum') or [] if str(x).strip()))
                    if not members:
                        continue
                    ckw['enum'] = members
                conditions.append((c['field'], FieldSpec(c['type'], **ckw)))
            conditions = list(dict(conditions).items())
            description = str(raw.get('description', ''))[:300]
            spec = PredicateSpec(raw['name'], FieldSpec(vt, description=description, **kwargs),
                                 cardinality='multi' if raw.get('cardinality') == 'multi' else 'single',
                                 conditions=tuple(conditions), description=description)
            if spec.name in {p.name for p in accepted}:
                raise ValueError('duplicate predicate')
            accepted.append(spec)
        except (ValueError, TypeError) as exc:
            dropped.append({'proposal': raw, 'reason': str(exc)})
    if not accepted:
        raise ValueError('No valid predicates proposed')
    return Vocabulary('auto:' + name, digest(proposal)[:16], tuple(accepted)), dropped


_HEADING = re.compile(r'^\s*(#{1,6}\s+\S|(ARTICLE|Article|SECTION|Section|PART|Part|ITEM|Item|SCHEDULE|Schedule|'
                      r'EXHIBIT|Exhibit|ANNEX|Annex)\b|\d+(\.\d+)*[.)]?\s+[A-Z])')


def document_overview(text: str, *, head_chars=1500, samples=4, sample_chars=600, max_headings=40):
    """What the vocabulary designer sees of one document: its beginning, passages spread over its whole length
    and its headings, so long documents are not described by their cover page alone."""
    headings = []
    for line in text.splitlines():
        line = line.strip()
        if 3 <= len(line) <= 120 and (_HEADING.match(line) or (line.isupper() and sum(c.isalpha() for c in line) >= 4)):
            if line not in headings:
                headings.append(line)
    if len(headings) > max_headings:
        step = len(headings) / max_headings
        headings = [headings[int(i * step)] for i in range(max_headings)]
    rest = len(text) - head_chars
    spread = []
    if rest > sample_chars:
        for i in range(samples):
            start = head_chars + int(rest * (i + 0.5) / samples) - sample_chars // 2
            spread.append(text[max(head_chars, start):max(head_chars, start) + sample_chars])
    return {'characters': len(text), 'beginning': text[:head_chars], 'passages_from_the_rest': spread,
            'headings': headings}


def propose_vocabulary(model, assignment: str, documents: dict, *, name='documents', max_predicates=30):
    """Ask the model for predicates (one call), then validate. Returns (Vocabulary, report)."""
    payload = {'assignment': assignment,
               'documents': [{'name': n, **document_overview(t)} for n, t in documents.items()],
               'allowed_dimensions': allowed_dimensions(), 'max_predicates': max_predicates}
    response = model.complete(VOCAB_PROMPT, payload)
    vocabulary, dropped = build_vocabulary(response['output'], name)
    return vocabulary, {'response': response, 'dropped': dropped, 'prompt_digest': digest(VOCAB_PROMPT)}


def with_entities(vocabulary: Vocabulary, entities, hints=None):
    """The same contract plus entity names, so a second reading can reuse the first one's names.

    hints maps a predicate to the entity names the first reading used for it; they are added to the predicate's
    description so that a reader names the same thing the same way (it remains free to name something else)."""
    unique = {}
    for entity in entities:
        unique.setdefault(label_key(entity), entity)
    predicates = vocabulary.predicates
    if hints:
        predicates = tuple(
            replace(p, description=(p.description + ' ' if p.description else '') +
                    'Entities named for this attribute in the sources: ' + '; '.join(hints[p.name]) + '.')
            if hints.get(p.name) else p for p in predicates)
    return Vocabulary(vocabulary.name, vocabulary.version + '+entities', predicates,
                      units=vocabulary.units, entities=tuple((e, ()) for e in unique.values()),
                      scope_keys=vocabulary.scope_keys)
