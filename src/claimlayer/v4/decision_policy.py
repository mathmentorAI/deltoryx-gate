"""Layer 6: explicit application authorization; default decisions only inform."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
from ..semantic.models import digest
from .contracts import ContextBundle, TaskSpec, VerificationReceipt, DecisionReceipt
from .reconstruction import GRADES, grade_at_least


@dataclass(frozen=True)
class ActionPolicy:
    policy_id: str
    version: str
    operation_id: str
    payload_digest: str
    accept_model_interpretation: bool
    context_digest: str
    pass_min_grade: str = 'B'

    def __post_init__(self):
        if not all(isinstance(v,str) and v.strip() for v in (self.policy_id,self.version,self.operation_id,self.payload_digest,self.context_digest)):
            raise ValueError('An action policy must identify policy, version, operation and payload digest')
        if type(self.accept_model_interpretation) is not bool or self.pass_min_grade not in GRADES:
            raise ValueError('Invalid action policy')


def decide(context: ContextBundle, task: TaskSpec | None, verification: VerificationReceipt, *,
           action_policy: ActionPolicy | None = None, operation: Mapping[str,Any] | None = None,
           infrastructure: bool = False) -> DecisionReceipt:
    vp=verification.payload;goals=vp.get('goals',[]);reasons=[]
    disposition='INFORM';authorized=False;scope='report'
    dependencies_current=verification.depends_on(context) and (task is None or verification.depends_on(task))
    if not dependencies_current:
        disposition='ESCALATE';reasons.append('VERIFICATION_SNAPSHOT_NOT_CURRENT')
    elif infrastructure or vp['derivation_status']!='valid':
        disposition='ESCALATE';reasons.append('PROVIDER_ERROR' if infrastructure else 'INCOMPLETE_OR_INVALID_DERIVATION')
    elif action_policy is None:
        reasons.append('INFORMATION_ONLY_NO_ACTION_POLICY')
    else:
        scope='action'
        if action_policy.context_digest!=context.content_digest:reasons.append('POLICY_SNAPSHOT_MISMATCH')
        if not action_policy.accept_model_interpretation:reasons.append('MODEL_INTERPRETATION_NOT_ACCEPTED')
        if not operation or operation.get('operation_id')!=action_policy.operation_id or digest(operation.get('payload'))!=action_policy.payload_digest:
            reasons.append('ACTION_PAYLOAD_NOT_BOUND_TO_POLICY')
        if not goals or any(g['formal_comparison']!='satisfied' for g in goals):reasons.append('ACTION_REQUIRES_ALL_GOALS_SATISFIED')
        if any(not g.get('reliability') or not grade_at_least(g['reliability']['grade'],action_policy.pass_min_grade) for g in goals):
            reasons.append('EXECUTION_GRADE_BELOW_POLICY')
        authorized=not reasons;disposition='PASS' if authorized else 'ESCALATE'
        if authorized:reasons.append('EXPLICIT_ACTION_POLICY_SATISFIED')
    payload=dict(disposition=disposition,authorized=authorized,scope=scope,reasons=reasons,
                 policy=action_policy.__dict__ if action_policy else None,
                 operation_digest=digest(operation) if operation is not None else None,
                 context_digest=context.content_digest,verification_digest=verification.content_digest,
                 interpretation='model_proposed' if task is not None else 'not_available',
                 presentation='Formal result for the model-structured goals and explicit premises; original-language interpretation is not certified by the core')
    return DecisionReceipt.create(payload,producer=dict(component='decision_policy',version='policy-1'),
                                   parents=[context,verification]+([task] if task else []))


def decision_is_current(decision: DecisionReceipt, context: ContextBundle, verification: VerificationReceipt,
                        operation: Mapping[str,Any] | None = None,
                        action_policy: ActionPolicy | None = None) -> bool:
    p=decision.payload
    if p['authorized'] and (action_policy is None or p['policy']!=action_policy.__dict__):
        return False
    return (verification.depends_on(context) and decision.depends_on(context) and decision.depends_on(verification)
            and p['context_digest']==context.content_digest and p['verification_digest']==verification.content_digest
            and p['operation_digest']==(digest(operation) if operation is not None else None))
