"""Layer 5: check source bindings and execute structured goals. No LLM or original claim input."""
from __future__ import annotations
from dataclasses import asdict
import time
from typing import Any, Mapping
from ..semantic.models import EvidenceRef, canonical
from .contracts import ContextBundle, TaskSpec, PremiseSet, DerivationPlan, VerificationReceipt, problem
from .context import documents
from .premise_builder import bind_premise
from ._formal_utils import _topological_steps, _values_match, _comparable, _value_kind
from .operations import DEFAULT_OPERATION_CATALOG, OperationCatalog
from .justification import ClaimNode, InferenceStep, JustificationGraph, VerificationObligation
from .evaluation import evaluate_response, VerificationBudget
from .verifiers import DEFAULT_VERIFIER_REGISTRY, VerifierRegistry, VerifierResult
from .reconstruction import parse_proposed_operations, proposed_operation_specs, step_grade, weakest

CORE_VERSION='verification-core-2'


class _CatalogVerifier:
    """Keep existing v4 formal obligations; use the declared catalog for extended operators."""
    def __init__(self,base,catalog):
        self.base,self.catalog=base,catalog
        self.family,self.verifier_id,self.version,self.guarantee_kind=base.family,base.verifier_id,base.version,base.guarantee_kind

    def verify(self,obligation,context):
        spec=self.catalog.get(context.step.operation)
        proposed=spec is not None and spec.version.startswith('proposed-') and obligation.origin!='engine_mandatory'
        if not proposed:
            result=self.base.verify(obligation,context)
            if result.status!='not_applicable' or obligation.origin=='engine_mandatory':return result
        if spec is None or spec.family!=self.family:
            return self.base.verify(obligation,context)
        out=self.catalog.execute(context.step.operation,context.premises,{**context.step.parameters,**obligation.parameters},context.domain_profile)
        matched=out.ok and _values_match(out.value,out.unit,context.conclusion.value,context.conclusion.unit,context.domain_profile.effective_unit_registry())
        return VerifierResult(obligation.obligation_id,context.step.step_id,self.family,
                              'verified' if matched else 'unknown' if out.error_kind=='missing_input' else 'refuted',
                              'deterministic_formal',evidence=out.to_dict(),
                              reasons=('CATALOG_OPERATION_VERIFIED' if matched else str(out.error_code or 'CATALOG_OPERATION_MISMATCH'),),
                              verifier_id=f'catalog:{spec.name}',verifier_version=spec.version)


class _NoOpenSemantics:
    family='open_semantics';verifier_id='core:no-open-semantics';version=CORE_VERSION;guarantee_kind='probabilistic_model'
    def verify(self,obligation,context):
        return VerifierResult(obligation.obligation_id,context.step.step_id,self.family,'unknown',
                              'probabilistic_model',reasons=('SEMANTIC_INTERPRETATION_NOT_CERTIFIED_BY_CORE',),
                              verifier_id=self.verifier_id,verifier_version=self.version)


def _formal_registry(base,catalog):
    registry=base or DEFAULT_VERIFIER_REGISTRY
    return VerifierRegistry([_NoOpenSemantics() if f=='open_semantics' else _CatalogVerifier(registry.get(f),catalog) for f in registry.families()])


def _revision_ok(actual,expected):
    return actual==expected or expected in ('source','change') or (expected=='head' and actual in ('added','context')) or (expected=='base' and actual in ('removed','context'))


def _signature(node):
    return (node.subject,node.predicate,canonical(node.scope),canonical(node.conditions))


def verify_derivation(context: ContextBundle, task: TaskSpec, premises: PremiseSet, plan: DerivationPlan,
                      *, profile: Any, catalog: OperationCatalog=DEFAULT_OPERATION_CATALOG,
                      verifier_registry=None, count_definitions=None) -> VerificationReceipt:
    """No semantic comparison to original_claim. Formal inputs are untrusted and rechecked."""
    start=time.monotonic();issues=[];registry=profile.effective_unit_registry();docs=documents(context)
    def bad(code,message,owner='planning',**details):
        return problem(code,message,stage='verification',detected_by='verification_core',repair_owner=owner,**details)
    if not task.depends_on(context) or not premises.depends_on(context) or not premises.depends_on(task) or not plan.depends_on(task) or not plan.depends_on(premises):
        issues.append(bad('STALE_ARTIFACT_DEPENDENCY','Input artifact parent hashes do not match','orchestration'))
    # Do not trust stored admitted values or a model's assertion of successful grounding.
    records={};source_issues={}
    for raw in premises.payload.get('proposed',[]):
        try:
            rec=bind_premise(raw,docs,registry)
            if rec['claim_id'] in records:raise ValueError('DUPLICATE_PREMISE_ID')
            records[rec['claim_id']]=rec
        except (ValueError,TypeError,KeyError) as exc:
            pid=str(raw.get('claim_id') or '') if isinstance(raw,dict) else ''
            source_issues[pid]=bad(str(exc),'Source binding failed','premises',claim_id=pid)
    if canonical(list(records.values()))!=canonical(premises.payload.get('admitted',[])):
        issues.append(bad('PREMISE_ARTIFACT_MISMATCH','Stored premise values do not match independent source checks','premises'))
    proposed,op_defects=parse_proposed_operations(plan.payload.get('proposed_operations'),core_catalog=catalog,registry=registry)
    issues.extend(bad(d['code'],d['message']) for d in op_defects)
    opmap={o.name:o for o in proposed}
    active=catalog.extend(proposed_operation_specs([o for o in proposed if o.grade=='C']))
    results=[]
    goal_plans={p['goal_id']:p for p in plan.payload['goals']}
    for goal in task.payload['goals']:
        row=dict(goal_id=goal['goal_id'],target=goal,derivation_status='not_run',formal_comparison='not_evaluated',
                 conclusion=None,verified_steps=[],issues=list(issues),missing_premises=[],assumptions=[],
                 formal_validity_scope='Conditional on the stated premises and declared operation meanings',
                 semantic_responsibility='model',semantic_review='not_assessed',reliability=None)
        gp=goal_plans.get(goal['goal_id'])
        if issues:
            row['derivation_status']='invalid';results.append(row);continue
        if gp is not None:
            by_conclusion={str(st.get('conclusion_id')):st for st in gp.get('steps',[])}
            needed=set();stack=[str(gp.get('conclusion_id'))]
            while stack:
                node=stack.pop()
                if node in needed:continue
                needed.add(node)
                if node in by_conclusion:stack.extend(str(x) for x in by_conclusion[node].get('premises',[]))
            relevant=[issue for pid,issue in source_issues.items() if pid in needed]
            if relevant:
                row.update(derivation_status='invalid',issues=relevant)
                results.append(row);continue
        if gp is None or gp.get('conclusion_id') is None:
            row.update(derivation_status='incomplete',formal_comparison='undetermined',
                       missing_premises=(gp or {}).get('missing_premises',[]) or [dict(role='derivation',description='The model did not establish a derivation for this goal')])
            results.append(row);continue
        row['missing_premises']=gp.get('missing_premises',[])
        try:
            _execute_goal(row,gp,goal,records,profile,active,opmap,docs,context.payload.get('as_of'),verifier_registry,count_definitions or {})
        except (ValueError,TypeError,KeyError,IndexError,ArithmeticError) as exc:
            row['issues'].append(bad('INVALID_STRUCTURED_PLAN',str(exc)))
            row.update(derivation_status='invalid',formal_comparison='not_evaluated')
        results.append(row)
    statuses={r['derivation_status'] for r in results}
    aggregate='invalid' if 'invalid' in statuses else 'incomplete' if statuses!={'valid'} else 'valid'
    payload=dict(derivation_status=aggregate,goals=results,issues=issues,
                 evidence_checks=list(records.values()),source_issues=list(source_issues.values()),proposed_operations=[o.describe() for o in proposed],
                 declared_goal_count=len(results),goals_with_conclusion=sum(r['conclusion'] is not None for r in results),
                 semantic_coverage='not_assessed',semantic_correctness_probability=None,
                 calibrated_correctness_probability=None,calibration_status='uncalibrated',
                 catalog_id=catalog.catalog_id,catalog_version=catalog.version,catalog_fingerprint=catalog.fingerprint(),
                 verifier_latency_ms=round((time.monotonic()-start)*1000,3),
                 guarantee='Execution and source checks for structured goals, not semantic equivalence to the original request')
    return VerificationReceipt.create(payload,producer=dict(component='verification',version=CORE_VERSION),parents=[context,task,premises,plan])


def _execute_goal(row,gp,goal,records,profile,catalog,proposed,docs,as_of,verifier_registry,count_defs):
    def bad(code,message,**details):
        row['issues'].append(problem(code,message,stage='verification',detected_by='verification_core',repair_owner='planning',**details))
    registry=profile.effective_unit_registry()
    if goal.get('unit') is not None:
        try:registry.lookup(goal['unit'])
        except ValueError:
            bad('UNKNOWN_TARGET_UNIT',str(goal['unit']))
            row['derivation_status']='invalid';return
    ordered,errors=_topological_steps(gp.get('steps',[]),set(records))
    ids=[s.get('step_id') for s in ordered]
    if errors or None in ids or len(ids)!=len(set(ids)):
        bad('INVALID_GRAPH','; '.join(errors) or 'Missing or duplicate step IDs')
        row['derivation_status']='invalid';return
    nodes={};sem={};grades={};graph_steps=[]
    for pid,r in records.items():
        md={'premise_kind':r['kind'],'event':r['event'],'direction':r['direction'],'anchor_reference':r['anchor_reference']}
        if r['diff']:md['diff']=r['diff']
        nodes[pid]=ClaimNode(pid,subject=r['entity'],predicate=r['predicate'],value=r['value'],unit=r['unit'],
                            polarity=r['polarity'],text=r['evidence']['quote'],evidence=(EvidenceRef(**r['evidence']),),
                            conditions=tuple(r['conditions']),scope=r['scope'],metadata=md,
                            condition_status='conditional' if r['conditions'] else 'unconditional')
        sem[pid]=r['revision_mode'];grades[pid]=r['grade']
    for st in ordered:
        sid=st['step_id'];op=str(st.get('operation') or '');spec=catalog.get(op)
        if op in proposed and proposed[op].grade=='D':
            row['missing_premises'].append(dict(role='unexecuted_model_assertion',description=f'{sid}: model assertion without executable definition'))
            row.update(derivation_status='incomplete',formal_comparison='undetermined');return
        if spec is None:
            bad('UNKNOWN_OPERATION',op,step_id=sid);row['derivation_status']='invalid';return
        pids=st.get('premises',[]);pn=tuple(nodes[p] for p in pids);params=dict(st.get('parameters') or {})
        if not pn:raise ValueError('A derivation step requires at least one premise')
        if any(str(k).startswith('__') for k in params):raise ValueError('Reserved operation parameter')
        if st.get('rule_id'):params['rule_id']=st['rule_id']
        if st.get('rule_version'):params['rule_version']=st['rule_version']
        if spec.name=='rule_application':
            rule=profile.declared_rules.get(str(params.get('rule_id')))
            if rule is None or (params.get('rule_version') and params['rule_version']!=rule.version):
                raise ValueError('UNDECLARED_RULE_OR_VERSION')
        if spec.name=='count_lines' and params.get('definition'):
            definition=count_defs.get(str(params['definition']))
            if definition is None:raise ValueError('UNDECLARED_COUNT_DEFINITION')
            params.update(contains=definition.contains,definition_version=definition.version)
            if definition.line_prefix:params['line_prefix']=definition.line_prefix
        out=catalog.execute(op,pn,params,profile)
        if not out.ok:
            bad(str(out.error_code),f'Cannot execute {op}',step_id=sid,details=out.details)
            row['derivation_status']='incomplete' if out.error_kind=='missing_input' else 'invalid'
            row['formal_comparison']='undetermined' if out.error_kind=='missing_input' else 'not_evaluated';return
        declared_type=spec.output_type
        actual_type=_value_kind(out.value)
        expected_types={'number':{'number'},'number_or_quantity':{'number'},'quantity':{'number'},
                        'integer':{'number'},'boolean':{'boolean'},'weekday':{'weekday'},'date':{'date'},
                        'string':{'text','weekday','date'},'text':{'text','weekday','date'}}
        if declared_type in expected_types and actual_type not in expected_types[declared_type]:
            bad('OPERATION_OUTPUT_TYPE_MISMATCH',f'{spec.name} declares {declared_type} but returns {actual_type}',step_id=sid)
            row['derivation_status']='invalid';return
        if op in proposed and declared_type not in {*expected_types,'any','object','array'}:
            bad('UNSUPPORTED_PROPOSED_OUTPUT_TYPE',declared_type,step_id=sid)
            row['derivation_status']='invalid';return
        output=dict(st.get('output') or {})
        if spec.name=='identity':
            source=pn[0]
            expected=dict(subject=source.subject,predicate=source.predicate,scope=source.scope,conditions=list(source.conditions))
            if any(k in output and canonical(output[k])!=canonical(v) for k,v in expected.items()):
                bad('IDENTITY_CHANGES_PROPOSITION','identity preserves subject, predicate, scope and conditions',step_id=sid)
                row['derivation_status']='invalid';return
            output={**output,**expected,'revision':sem[pids[0]]}
        elif spec.name=='count_lines':
            if output.get('predicate','line_count')!='line_count' or output.get('subject',pn[0].subject)!=pn[0].subject:
                bad('COUNT_CHANGES_MEASUREMENT','count_lines measures line_count of its cited subject',step_id=sid)
                row['derivation_status']='invalid';return
            output={**output,'predicate':'line_count','subject':pn[0].subject}
            if out.details.get('truncated') or out.details.get('convention_sensitive'):
                row['missing_premises'].append(dict(role='count_scope_or_convention',description='Source is truncated or newline convention is ambiguous'))
        if not output.get('subject') or not output.get('predicate'):raise ValueError('MISSING_STEP_OUTPUT_TYPE')
        inherited={canonical(c) for p in pn for c in p.conditions}
        conditions=output.get('conditions',[])
        if not inherited <= {canonical(c) for c in conditions}:raise ValueError('DROPPED_REQUIRED_CONDITION')
        cid=st['conclusion_id']
        nodes[cid]=ClaimNode(cid,kind='derivation',subject=output['subject'],predicate=output['predicate'],
                             value=out.value,unit=out.unit,polarity=pn[0].polarity if spec.name=='identity' else 1,
                             text=f'{spec.name}({",".join(pids)}) = {out.value}',conditions=tuple(conditions),
                             condition_status='conditional' if conditions else 'unconditional',scope=output.get('scope',{}),
                             assumptions=tuple(out.assumptions)+tuple(st.get('assumptions',[])))
        sem[cid]=output.get('revision',goal['revision']);grade=step_grade(spec.name,proposed)
        grades[cid]=weakest([grade,*[grades[p] for p in pids]])
        ob=VerificationObligation(f'{sid}::{spec.family}',spec.family,description=f'Check {spec.name}',parameters=params)
        graph_steps.append(InferenceStep(sid,tuple(pids),cid,spec.name,rule_id=st.get('rule_id'),
                                         rule_version=st.get('rule_version'),parameters=params,obligations=(ob,),
                                         assumptions=nodes[cid].assumptions))
        row['verified_steps'].append(dict(step_id=sid,operation=spec.name,operation_version=spec.version,premises=pids,
                                          conclusion_id=cid,recomputed_value=out.value,recomputed_unit=out.unit,
                                          details=out.details,output={**output,'revision':sem[cid]},grade=grade,
                                          semantic_mapping='preserved' if spec.name=='identity' else 'source_measurement' if spec.name=='count_lines' else 'model_proposed',
                                          model_proposed_value=st.get('proposed_value'),
                                          model_disagreed=st.get('proposed_value') is not None and not _values_match(out.value,out.unit,st['proposed_value'],st.get('proposed_unit'),profile.effective_unit_registry())))
        row['assumptions'].extend(nodes[cid].assumptions)
    cid=str(gp['conclusion_id'])
    if cid not in nodes:raise ValueError('MISSING_CONCLUSION')
    conc=nodes[cid]
    row['conclusion']=dict(claim_id=cid,subject=conc.subject,predicate=conc.predicate,value=conc.value,unit=conc.unit,
                           scope=conc.scope,conditions=list(conc.conditions),polarity=conc.polarity,revision=sem[cid])
    if _signature(conc)!=(goal['subject'],goal['predicate'],canonical(goal['scope']),canonical(goal['conditions'])) or not _revision_ok(sem[cid],goal['revision']):
        bad('FORMAL_TARGET_MISMATCH','The computed proposition has a different structured subject, predicate, revision, scope or conditions')
        row['derivation_status']='invalid';return
    # Revision compatibility is checked from actual source diff sides, not model labels.
    by_conclusion={str(s['conclusion_id']):s for s in ordered}
    leaves=set();visited=set();stack=[cid]
    while stack:
        node=stack.pop()
        if node in visited:continue
        visited.add(node)
        if node in by_conclusion:stack.extend(by_conclusion[node].get('premises',[]))
        elif node in records:leaves.add(node)
    diff_sides=[set((records[p].get('diff_binding') or {}).get('sides',[])) for p in leaves if records[p].get('diff_binding')]
    mode=goal['revision']
    if mode in ('head','added','change') and diff_sides:
        if mode in ('head','added') and any(s=={'-'} for s in diff_sides):
            bad('BASE_ONLY_EVIDENCE_FOR_HEAD','A dependency is supported only by removed lines')
        if mode=='added' and not any('+' in s for s in diff_sides):
            bad('NO_ADDED_LINE_EVIDENCE','The target requires added lines')
        if mode=='change' and not any(s & {'+','-'} for s in diff_sides):
            bad('NO_CHANGED_LINE_EVIDENCE','The target concerns change but dependencies are unchanged context')
        if row['issues']:
            row['derivation_status']='invalid';return
    if _value_kind(conc.value)=='number' and ((conc.unit is None)!=(goal['unit'] is None)):
        bad('MISSING_COMPARISON_UNIT','Both numeric quantities must declare compatible units, or both must be dimensionless')
        row['derivation_status']='invalid';return
    if conc.unit is not None and goal['unit'] is not None:
        if registry.lookup(conc.unit).dimension!=registry.lookup(goal['unit']).dimension:
            bad('INCOMPATIBLE_TARGET_DIMENSIONS','A dimension mismatch is not a refutation')
            row['derivation_status']='invalid';return
    # Run existing v4 mandatory formal obligations with semantic calls disabled.
    if graph_steps:
        graph=JustificationGraph(nodes,tuple(graph_steps),cid)
        assessment=evaluate_response(graph,evidence=docs,domain_profile=profile,
                                     verifier_registry=_formal_registry(verifier_registry,catalog),
                                     budget=VerificationBudget(max_semantic_calls=0),as_of=as_of)
        row['graph']=graph.to_dict() if hasattr(graph,'to_dict') else graph.canonical_payload()
        row['formal_checks']=assessment.to_summary_dict()
        row['assumptions'].extend(assessment.assumptions)
        # A deferred mandatory check must never disappear merely because it did
        # not produce a completed VerifierResult (e.g. graph resource limits).
        if assessment.pending_obligations or assessment.critical_unverified_steps:
            row['missing_premises'].append(dict(
                role='pending_formal_obligations',
                description='Mandatory checks are pending or critical steps remain unverified',
                obligations=list(assessment.pending_obligations),
                critical_steps=list(assessment.critical_unverified_steps)))
        failures=[c for c in assessment.completed_checks if c.status=='refuted']
        unknown=[c for c in assessment.completed_checks if c.status=='unknown']
        if failures:
            for c in failures:bad('FORMAL_OBLIGATION_REFUTED',';'.join(c.reasons),step_id=c.step_id)
            row['derivation_status']='invalid';return
        if unknown:
            row['missing_premises'].extend(dict(role='formal_obligation',description=';'.join(c.reasons)) for c in unknown)
    row['reliability']=dict(grade=grades[cid],meaning='Weakest execution/source dependency; not semantic correctness',
                             semantic_mapping_responsibility='model')
    if row['missing_premises']:
        row.update(derivation_status='incomplete',formal_comparison='undetermined');return
    if not _comparable(conc.value,goal['value']):
        bad('INCOMPARABLE_TARGET_VALUE','Computed and asserted values have incompatible kinds')
        row['derivation_status']='invalid';return
    matched=_values_match(conc.value,conc.unit,goal['value'],goal['unit'],profile.effective_unit_registry()) and conc.polarity==goal['polarity']
    row.update(derivation_status='valid',formal_comparison='satisfied' if matched else 'contradicted')
