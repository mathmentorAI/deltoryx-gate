"""Graph paths: candidate retrieval for any entity-valued predicates, proofs only for declared rules.

Paper contract: a path identifies relevant records but does not prove a relation.
Only predicates declared as TransitiveRule produce derived units; every other
path is returned as a candidate. Ranking uses the weakest-link score
R(pi) = gamma^h * min(s_z), h = number of rule applications (k-1 for k edges).
It is a ranking heuristic, not a probability.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..semantic.models import canonical, instant, require_string


@dataclass(frozen=True)
class TransitiveRule:
    predicate: str
    version: str

    def __post_init__(self):
        require_string(self.predicate, 'predicate')
        require_string(self.version, 'version')

    @property
    def rule_name(self):
        return 'transitive:' + self.predicate

    def conclusion(self, premises):
        if not 2 <= len(premises) <= 8:
            raise ValueError('A path needs two to eight declared edges')
        first = premises[0]
        for edge in premises:
            if (edge['predicate'] != self.predicate or edge['polarity'] != 'affirmed' or edge['unit'] is not None
                    or edge['assertion_kind'] != first['assertion_kind']
                    or canonical(edge['conditions']) != canonical(first['conditions'])
                    or canonical({k: v for k, v in edge['scope'].items() if k != 'event_id'})
                    != canonical({k: v for k, v in first['scope'].items() if k != 'event_id'})
                    or not isinstance(edge['value'], str)):
                raise ValueError('Incompatible transitive edge contract')
        nodes = [first['entity']] + [edge['value'] for edge in premises]
        if len(set(nodes)) != len(nodes) or any(a['value'] != b['entity'] for a, b in zip(premises, premises[1:])):
            raise ValueError('Path must be connected, acyclic and exact in entity identity')
        starts = [e['valid_from'] for e in premises if e['valid_from']]
        ends = [e['valid_until'] for e in premises if e['valid_until']]
        start = max(starts, key=instant) if starts else None
        end = min(ends, key=instant) if ends else None
        if start and end and instant(start) > instant(end):
            raise ValueError('No shared validity interval')
        return {'entity': first['entity'], 'predicate': self.predicate, 'value': premises[-1]['value'], 'unit': None,
                'polarity': 'affirmed', 'assertion_kind': first['assertion_kind'],
                'conditions': first['conditions'],
                'scope': {k: v for k, v in first['scope'].items() if k != 'event_id'},
                'valid_from': start, 'valid_until': end}

    def check(self, conclusion, premises):
        try:
            expected = self.conclusion(list(premises))
        except (ValueError, TypeError, KeyError):
            return False
        return canonical(expected) == canonical(conclusion)


def _edges(engine, predicates, query, mode):
    edges, excluded = {}, []
    for unit in sorted(engine.units, key=lambda u: u.unit_id):
        p = unit.proposition
        if p['predicate'] not in predicates or unit.origin == 'rule_derived':
            continue
        if p['polarity'] != 'affirmed' or not isinstance(p['value'], str):
            excluded.append({'unit_id': unit.unit_id, 'reason': 'unsupported_edge_contract'})
            continue
        basis, reason = engine._eligible(unit, replace(query, entity=p['entity'], predicate=p['predicate']), mode)
        if basis is None or basis['support'] is None:
            excluded.append({'unit_id': unit.unit_id, 'reason': reason or 'unassessed'})
            continue
        edges.setdefault(p['entity'], []).append((unit, basis))
    return edges, excluded


def _walk(start, edges, max_hops, min_hops, max_expansions=10000):
    paths, stack, expansions = [], [((start,), ())], 0
    while stack:
        nodes, chain = stack.pop()
        if len(chain) >= min_hops:
            paths.append((nodes, chain))
        if len(chain) >= max_hops:
            continue
        for unit, basis in reversed(edges.get(nodes[-1], [])):
            expansions += 1
            if expansions > max_expansions:
                return paths, 'max_expansions'
            target = unit.proposition['value']
            if target in nodes:
                continue
            stack.append((nodes + (target,), chain + ((unit, basis),)))
    return paths, None


def _score(engine, chain):
    weakest = min(b['support'] for _, b in chain)
    return round((engine.policy.path_discount ** (len(chain) - 1)) * weakest, 6), weakest


def derive_paths(engine, entity, predicate, *, as_of=None, max_hops=3, max_paths=20):
    rule = engine._transitive.get(predicate)
    if rule is None:
        raise ValueError('The application has not declared this predicate transitive')
    if type(max_hops) is not int or not 2 <= max_hops <= 8 or type(max_paths) is not int or not 1 <= max_paths <= 100:
        raise ValueError('Bounded search required: 2<=max_hops<=8, 1<=max_paths<=100')
    query = engine._query({'entity': entity, 'predicate': predicate, 'operation': 'list', 'as_of': as_of})
    edges, excluded = _edges(engine, {predicate}, query, 'discharge')
    walked, stopped = _walk(query.entity, edges, max_hops, 2)
    found = []
    for nodes, chain in walked:
        premises = [u.proposition for u, _ in chain]
        try:
            conclusion = rule.conclusion(premises)
        except ValueError:
            excluded.append({'unit_ids': [u.unit_id for u, _ in chain], 'reason': 'incompatible_path_contract'})
            continue
        score, weakest = _score(engine, chain)
        unit_id = engine.derive(conclusion, [u.unit_id for u, _ in chain], rule.rule_name)
        found.append({'unit_id': unit_id, 'nodes': list(nodes), 'premises': [u.unit_id for u, _ in chain],
                      'hops': len(chain), 'rule_applications': len(chain) - 1, 'weakest_support': weakest,
                      'path_score': score})
    found.sort(key=lambda x: (-x['path_score'], x['hops'], x['unit_id']))
    return engine._issue({'operation': 'derive_paths', 'entity': query.entity, 'predicate': predicate,
                          'as_of': query.as_of, 'rule': {'name': rule.rule_name, 'version': rule.version},
                          'paths': found[:max_paths], 'truncated': len(found) > max_paths or stopped is not None,
                          'excluded_edges': excluded, 'configuration': engine.configuration,
                          'scope': 'bounded discovery over a declared transitive relation; derived units remain '
                                   'subject to ask() conflict resolution; score is a ranking index, not a probability'})


def find_paths(engine, entity, *, predicates=None, as_of=None, max_hops=3, max_paths=20):
    """Heterogeneous joins over entity-valued predicates. Returns candidates, never proofs."""
    allowed = {p.name for p in engine.vocabulary.predicates if p.value.type == 'entity'}
    predicates = allowed if predicates is None else set(predicates)
    if not predicates <= allowed:
        raise ValueError('Paths can only follow entity-valued predicates')
    if type(max_hops) is not int or not 1 <= max_hops <= 8 or type(max_paths) is not int or not 1 <= max_paths <= 100:
        raise ValueError('Bounded search required: 1<=max_hops<=8, 1<=max_paths<=100')
    query = engine._query({'entity': entity, 'predicate': sorted(predicates)[0], 'operation': 'list', 'as_of': as_of})
    edges, excluded = _edges(engine, predicates, query, 'report')
    walked, stopped = _walk(query.entity, edges, max_hops, 1)
    found = []
    for nodes, chain in walked:
        score, weakest = _score(engine, chain)
        found.append({'status': 'candidate_path', 'nodes': list(nodes),
                      'edges': [{'unit_id': u.unit_id, 'predicate': u.proposition['predicate'],
                                 'conditional': bool(b['pending_conditions'] or b['pending_scope'])} for u, b in chain],
                      'hops': len(chain), 'weakest_support': weakest, 'path_score': score})
    found.sort(key=lambda x: (-x['path_score'], x['hops'], x['nodes']))
    return engine._issue({'operation': 'find_paths', 'entity': query.entity, 'predicates': sorted(predicates),
                          'as_of': query.as_of, 'paths': found[:max_paths],
                          'truncated': len(found) > max_paths or stopped is not None, 'excluded_edges': excluded,
                          'configuration': engine.configuration,
                          'scope': 'candidate retrieval: a path does not establish that any predicate composes'})
