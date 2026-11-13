"""Bounded search over application-authorized homogeneous transitive relations."""
from dataclasses import dataclass,asdict
import heapq
import json
from .models import CandidateClaim,StructuredQuery,canonical,digest,instant,require_string

@dataclass(frozen=True)
class TransitiveRelation:
    predicate: str
    version: str

    def __post_init__(self):
        require_string(self.predicate,'predicate')
        require_string(self.version,'version')

    @property
    def rule_name(self):
        return 'transitive:' + self.predicate

    def conclusion(self, premises):
        if not 2 <= len(premises) <= 8:
            raise ValueError('A path needs two to eight declared edges')
        first=premises[0]
        for edge in premises:
            if (edge.predicate!=self.predicate or edge.assertion_kind!='fact'
                    or edge.polarity!='affirmed' or edge.unit is not None
                    or type(edge.value) is not str or not edge.value
                    or canonical(edge.scope)!=canonical(first.scope)
                    or canonical(edge.conditions)!=canonical(first.conditions)):
                raise ValueError('Incompatible transitive edge contract')
        nodes=[first.entity]+[edge.value for edge in premises]
        if len(set(nodes))!=len(nodes) or any(a.value!=b.entity for a,b in zip(premises,premises[1:])):
            raise ValueError('Path must be connected, acyclic and exact in entity identity')
        starts=[e.valid_from for e in premises if e.valid_from]
        ends=[e.valid_until for e in premises if e.valid_until]
        start=max(starts,key=instant) if starts else None
        end=min(ends,key=instant) if ends else None
        if start and end and instant(start)>instant(end):
            raise ValueError('No shared validity interval')
        identity={'rule':self.rule_name,'version':self.version,'premises':[p.claim_id for p in premises]}
        return CandidateClaim('path-'+digest(identity),first.entity,self.predicate,premises[-1].value,
                              assertion_kind='fact',scope=first.scope,conditions=first.conditions,
                              valid_from=start,valid_until=end,premises=tuple(p.claim_id for p in premises),
                              derivation_rule=self.rule_name)

    def check(self, claim, premises):
        try:expected=self.conclusion(premises)
        except (ValueError,TypeError):return False
        return not claim.evidence and canonical(claim.proposition())==canonical(expected.proposition())


def derive_paths(engine,entity,predicate,*,target=None,as_of,context=None,max_hops=3,max_paths=20,max_expansions=1000):
    require_string(entity,'entity');require_string(predicate,'predicate')
    if target is not None:require_string(target,'target')
    if (type(max_hops) is not int or not 2<=max_hops<=8
            or type(max_paths) is not int or not 1<=max_paths<=100
            or type(max_expansions) is not int or not 1<=max_expansions<=10000):
        raise ValueError('Search requires bounded hops, paths and expansions')
    relation=engine._transitive_relations.get(predicate)
    if relation is None:raise ValueError('Application has not authorized this transitive predicate')
    query=StructuredQuery(entity,predicate,as_of=as_of,context={} if context is None else context)
    if not query.as_of:raise ValueError('An explicit as_of is required')
    edges={};excluded=[]
    for claim in sorted(engine.core_claims,key=lambda c:c.claim_id):
        # Only original asserted edges: do not discover paths over old derived shortcuts.
        if claim.predicate!=predicate or engine._claim_record(claim.claim_id)['origin']=='rule_derived':continue
        if claim.assertion_kind!='fact' or claim.polarity!='affirmed' or claim.unit is not None or type(claim.value) is not str:
            excluded.append({'claim_id':claim.claim_id,'reason':'unsupported_edge_contract'});continue
        basis,reason=engine._eligible(claim.claim_id,query,as_premise=True)
        if basis is None:excluded.append({'claim_id':claim.claim_id,'reason':reason});continue
        rank=min(d.authority_rank for d in basis['documents'].values())
        weight=min(d.weight for d in basis['documents'].values())
        edges.setdefault(claim.entity,[]).append((claim,rank,weight))
    # Max-min source metadata priority; these weights are NOT calibrated probabilities.
    frontier=[(-101,-1.0,0,(),(entity,))];paths=[];expansions=0;stopped=None
    while frontier:
        neg_rank,neg_weight,depth,ids,nodes=heapq.heappop(frontier)
        if depth>=2 and (target is None or nodes[-1]==target):
            premises=tuple(CandidateClaim.from_dict(engine._claim_record(cid)['claim']) for cid in ids)
            try:derived=relation.conclusion(premises)
            except ValueError:
                excluded.append({'claim_ids':list(ids),'reason':'incompatible_path_contract'});continue
            old=engine.store.head('claim',derived.claim_id)[1]
            if old is None:engine.derive(derived)
            elif old.get('rule_version')!=relation.version or old['claim']!=json.loads(canonical(asdict(derived))):
                raise ValueError('Derived path identity collision')
            basis,reason=engine._eligible(derived.claim_id,query)
            if basis is None:
                excluded.append({'claim_ids':list(ids),'reason':reason});continue
            paths.append({'claim_id':derived.claim_id,'nodes':list(nodes),'premises':list(ids),'hops':depth,
                          'authority_rank':-neg_rank,'bottleneck_source_weight':-neg_weight,
                          'weight_interpretation':'application metadata, not probability'})
            if len(paths)>=max_paths:
                stopped='max_paths' if frontier or depth<max_hops else None;break
        if depth>=max_hops or (target is not None and nodes[-1]==target):continue
        for edge,rank,weight in edges.get(nodes[-1],[]):
            if expansions>=max_expansions:
                stopped='max_expansions';break
            expansions+=1
            if edge.value in nodes:continue
            heapq.heappush(frontier,(-min(-neg_rank,rank),-min(-neg_weight,weight),depth+1,
                                     ids+(edge.claim_id,),nodes+(edge.value,)))
        if stopped=='max_expansions':break
    selected_ids={cid for path in paths for cid in [path['claim_id'],*path['premises']]}
    records={cid:engine._claim_record(cid) for cid in sorted(selected_ids)}
    revisions={ref.revision for record in records.values() for ref in CandidateClaim.from_dict(record['claim']).evidence}
    evaluation_ids=set()
    for cid in selected_ids:
        _,head=engine.store.head('assessment',cid)
        if head:evaluation_ids.add(head['evaluation_id'])
    basis={'claim_record_digests':{cid:digest(record) for cid,record in records.items()},
           'evaluation_digests':{eid:digest(engine.store.get('evaluation',eid)) for eid in sorted(evaluation_ids)},
           'source_revisions':sorted(revisions)}
    payload={'operation':'derive_paths','entity':entity,'predicate':predicate,'target':target,'as_of':as_of,
             'context':query.context,'paths':paths,'excluded_edges':excluded,'expansions':expansions,
             'limits':{'max_hops':max_hops,'max_paths':max_paths,'max_expansions':max_expansions},
             'stopped_reason':stopped,'exhaustive':False,'scope':'bounded authorized-path discovery, not completeness or semantic truth',
             'rule':{'name':relation.rule_name,'version':relation.version},
             'basis':basis,'proof_claims':{cid:record['claim'] for cid,record in records.items()}}
    return engine._issue(payload)
