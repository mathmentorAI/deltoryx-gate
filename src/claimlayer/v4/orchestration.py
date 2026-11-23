"""Bounded live synthesis/repair over the six explicit layer contracts."""
from __future__ import annotations
from dataclasses import replace
import json
import time
from typing import Any
from ..semantic.models import canonical, digest
from .contracts import GateResult, VerificationReceipt, ContractError, problem
from .context import build_context, documents
from .task_interpretation import capture_task
from .premise_builder import build_premises
from .plan_synthesis import ClaimDerivationPlanner, capture_plan
from .verification_core import verify_derivation, CORE_VERSION
from .decision_policy import decide
from ._formal_utils import _resolve_derivation_profile
from .operations import DEFAULT_OPERATION_CATALOG
from .count_definitions import count_definition_registry


class ClaimVerifier:
    def __init__(self, *, model=None, planner=None, catalog=DEFAULT_OPERATION_CATALOG,
                 domain_profile=None, verifier_registry=None, max_retries=1, unit_profile='auto',
                 count_definitions=None, action_policy=None):
        if model is None and planner is None:raise ValueError('Provide a real model or a live planner adapter')
        if type(max_retries) is not int or not 0<=max_retries<=5:raise ValueError('max_retries must be in [0,5]')
        if model is not None and planner is not None:raise ValueError('Provide model or planner, not both')
        self.count_definitions=count_definition_registry(count_definitions)
        self.planner=planner if planner is not None else ClaimDerivationPlanner(model=model,count_definitions=self.count_definitions)
        self.catalog,self.domain_profile,self.verifier_registry=catalog,domain_profile,verifier_registry
        self.max_retries,self.unit_profile=max_retries,unit_profile
        self.action_policy=action_policy

    def verify(self,sources,claim,*,as_of=None,context=None,operation=None):
        started=time.monotonic();bundle=build_context(sources,claim,context=context,as_of=as_of)
        profile,label,unit_fp=_resolve_derivation_profile(self.domain_profile,self.unit_profile,claim,documents(bundle))
        traces=[];task=premises=plan=None;feedback=None;verification=None;infrastructure=False
        for attempt in range(1,self.max_retries+2):
            call=self.planner.plan(context=bundle,catalog=self.catalog,domain_profile=profile,
                                   repair_feedback=feedback,frozen_task=task)
            producer={**dict(self.planner.identity),'attempt':attempt}
            event=dict(attempt=attempt,producer=producer,usage=dict(call.usage),planning_latency_ms=round(call.latency_ms,3),
                       raw_proposal=call.proposal,provider_response=call.provider_response,
                       input_context_digest=bundle.content_digest,issues=[],status='received')
            if not call.ok or call.proposal is None:
                infrastructure=call.error_kind=='provider_error'
                event['issues']=[problem(call.error_kind or 'MALFORMED_PROPOSAL',call.error_message or 'No proposal',
                                        stage='infrastructure' if infrastructure else 'planning',
                                        detected_by='provider_adapter',repair_owner='infrastructure' if infrastructure else 'planning')]
                event['status']='provider_error' if infrastructure else 'rejected'
                traces.append(event);feedback=event['issues']
                if infrastructure:break
                continue
            raw=call.proposal
            try:
                candidate=capture_task(raw.get('task'),bundle,producer)
                if task is not None and canonical(candidate.payload)!=canonical(task.payload):
                    if raw.get('reinterpretation') is not True:
                        raise ContractError('TASK_CHANGED_IN_REPAIR','Task is frozen; explicitly request reinterpretation to invalidate dependent artifacts','interpretation')
                    event['invalidated_artifacts']=[a.artifact_id for a in (task,premises,plan,verification) if a is not None]
                    task=candidate;premises=plan=verification=None
                elif task is None:task=candidate
                premises=build_premises(raw.get('premises'),bundle,task,profile,producer)
                plan=capture_plan(raw,task,premises,producer)
                verification=verify_derivation(bundle,task,premises,plan,profile=profile,catalog=self.catalog,
                                               verifier_registry=self.verifier_registry,count_definitions=self.count_definitions)
                vp=verification.payload
                issues=[*premises.payload['rejected'],*vp['issues'],*[i for g in vp['goals'] for i in g['issues']]]
                issues=list({canonical(i):i for i in issues}.values())
                event.update(status='rejected' if vp['derivation_status']=='invalid' else 'completed',
                             issues=issues,verification=verification.to_dict(),artifacts=dict(
                                 task=task.to_dict(),premises=premises.to_dict(),plan=plan.to_dict()))
                traces.append(event);feedback=issues
                if vp['derivation_status']!='invalid':break
            except (ContractError,ValueError,TypeError,KeyError,IndexError) as exc:
                issue=exc.to_dict() if isinstance(exc,ContractError) else problem('MALFORMED_PROPOSAL',str(exc),stage='planning',detected_by='layer_contracts',repair_owner='planning')
                event.update(status='rejected',issues=[issue]);traces.append(event);feedback=[issue]
                # Never reuse a successful receipt after a later failed attempt.
                verification=None
        if verification is None or infrastructure or traces[-1]['status']=='rejected' and 'verification' not in traces[-1]:
            verification=VerificationReceipt.create(dict(derivation_status='not_run',goals=[],issues=feedback or [],
                                                         evidence_checks=[],semantic_coverage='not_assessed',verifier_latency_ms=0,
                                                         guarantee='No completed formal verification for the current attempt'),
                                                     producer=dict(component='verification',version=CORE_VERSION),
                                                     parents=[bundle]+([task] if task else [])+([premises] if premises else [])+([plan] if plan else []))
        decision=decide(bundle,task,verification,action_policy=self.action_policy,operation=operation,infrastructure=infrastructure)
        trace=dict(orchestrator_version='orchestration-1',status='blocked' if infrastructure else 'completed',
                   infrastructure_error=feedback[0] if infrastructure and feedback else None,
                   calls=len(traces),retries=max(0,len(traces)-1),attempts=traces,
                   unit_profile=label,unit_registry_fingerprint=unit_fp,
                   planning_latency_ms=round(sum(t['planning_latency_ms'] for t in traces),3),
                   verifier_latency_ms=round(sum(t.get('verification',{}).get('payload',{}).get('verifier_latency_ms',0) for t in traces),3),
                   total_latency_ms=round((time.monotonic()-started)*1000,3),
                   usage={k:sum(int(t['usage'].get(k) or 0) for t in traces) for k in ('prompt_tokens','completion_tokens','total_tokens')})
        return GateResult(bundle,task,premises,plan,verification,decision,canonical(trace))


def verify_claim(sources,claim,*,model=None,planner=None,catalog=DEFAULT_OPERATION_CATALOG,
                 domain_profile=None,verifier_registry=None,as_of=None,max_retries=1,unit_profile='auto',
                 count_definitions=None,context=None,operation=None,action_policy=None):
    return ClaimVerifier(model=model,planner=planner,catalog=catalog,domain_profile=domain_profile,
                         verifier_registry=verifier_registry,max_retries=max_retries,unit_profile=unit_profile,
                         count_definitions=count_definitions,
                         action_policy=action_policy).verify(sources,claim,as_of=as_of,context=context,operation=operation)


def infrastructure_result(sources, claim, issue, *, context=None, as_of=None):
    """Represent a deployment failure without fabricating a planner or evidence result."""
    bundle=build_context(sources,claim,context=context,as_of=as_of)
    verification=VerificationReceipt.create(
        dict(derivation_status='not_run',goals=[],issues=[issue],evidence_checks=[],verifier_latency_ms=0,
             guarantee='Provider/configuration unavailable; no semantic or formal result'),
        producer=dict(component='verification',version=CORE_VERSION),parents=[bundle])
    decision=decide(bundle,None,verification,infrastructure=True)
    return GateResult(bundle,None,None,None,verification,decision,
                      canonical(dict(status='blocked',infrastructure_error=issue,calls=0,retries=0,attempts=[])))
