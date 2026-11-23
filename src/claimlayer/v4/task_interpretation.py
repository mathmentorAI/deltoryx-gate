"""Layer 2: capture the model's interpretation; never certify its semantic fidelity."""
from __future__ import annotations
from typing import Any, Mapping
from .contracts import ContextBundle, TaskSpec, ContractError

TASK_VERSION='task-1'


def capture_task(raw: Any, context: ContextBundle, producer: Mapping[str,Any]) -> TaskSpec:
    if not isinstance(raw,Mapping) or not isinstance(raw.get('goals'),list) or not 1<=len(raw['goals'])<=64:
        raise ContractError('INVALID_TASK','task.goals must contain 1..64 structured goals','interpretation')
    claim=context.payload['original_claim'];seen=set();goals=[]
    for goal in raw['goals']:
        if not isinstance(goal,Mapping):
            raise ContractError('INVALID_GOAL','Each goal must be an object','interpretation')
        g=dict(goal)
        for k in ('goal_id','subject','predicate','original_quote'):
            if not isinstance(g.get(k),str) or not g[k].strip():
                raise ContractError('INVALID_GOAL',f'Goal requires nonempty {k}','interpretation')
        if g['goal_id'] in seen:
            raise ContractError('DUPLICATE_GOAL_ID',g['goal_id'],'interpretation')
        seen.add(g['goal_id'])
        if g['original_quote'] not in claim:
            raise ContractError('INVALID_CLAIM_REFERENCE','original_quote must be an exact substring of the original claim','interpretation')
        if 'value' not in g or type(g.get('polarity',1)) is not int or g.get('polarity',1) not in (1,-1):
            raise ContractError('INVALID_GOAL','value and polarity are required','interpretation')
        if g.get('unit') is not None and not isinstance(g['unit'],str):
            raise ContractError('INVALID_GOAL','unit must be text or null','interpretation')
        if not isinstance(g.get('scope',{}),dict) or not isinstance(g.get('conditions',[]),list):
            raise ContractError('INVALID_GOAL','scope must be an object and conditions a list','interpretation')
        if g.get('revision','source') not in ('source','head','base','change','added','removed'):
            raise ContractError('INVALID_GOAL','Unsupported revision mode','interpretation')
        kind=g.get('kind','fact')
        if kind not in ('fact','behavior','policy','recommendation'):
            raise ContractError('INVALID_GOAL','Unsupported goal kind','interpretation')
        goals.append({k:g.get(k,default) for k,default in dict(goal_id=None,subject=None,predicate=None,value=None,
                      unit=None,polarity=1,revision='source',scope={},conditions=[],kind=kind,original_quote=None).items()})
    return TaskSpec.create(dict(goals=goals,ambiguities=raw.get('ambiguities',[]),
                                semantic_responsibility='model',semantic_review='not_assessed',
                                coverage_meaning='Model-declared goals; completeness relative to original text is not certified'),
                           producer={**producer,'component':'interpretation','version':TASK_VERSION},parents=[context])
