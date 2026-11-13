"""Compare an independently proposed source projection with the immutable claim.

No natural-language parsing: interpretation remains the adapter's responsibility.
"""
from dataclasses import asdict
from decimal import Decimal
from .models import CandidateClaim,canonical,digest

FIELDS={'entity','predicate','value','polarity','assertion_kind','unit','conditions','scope','valid_from','valid_until'}

def shape(value):
    """Expose field names/types, never scalar candidate values to the projector."""
    if type(value) is dict:return {'type':'object','fields':{k:shape(v) for k,v in value.items()}}
    if type(value) is list or type(value) is tuple:
        return {'type':'array','item_shapes':[shape(v) for v in value]}
    return {'type':'null' if value is None else 'boolean' if type(value) is bool else 'number' if type(value) in (int,float) else 'string'}

def comparable(value):
    if type(value) in (int,float):return ('number',str(Decimal(str(value)).normalize()))
    if type(value) is dict:return ('object',tuple((k,comparable(v)) for k,v in sorted(value.items())))
    if type(value) in (list,tuple):return ('array',tuple(comparable(v) for v in value))
    return (type(value).__name__,value)

def differences(expected,observed,path=''):
    if comparable(expected)==comparable(observed):return []
    if type(expected) is dict and type(observed) is dict:
        result=[]
        for key in sorted(set(expected)|set(observed)):
            here=path+'/'+key.replace('~','~0').replace('/','~1')
            if key not in expected or key not in observed:
                result.append({'path':here,'claim_present':key in expected,'source_present':key in observed,
                               'claim':expected.get(key),'source':observed.get(key)})
            else:result.extend(differences(expected[key],observed[key],here))
        return result
    return [{'path':path,'claim_present':True,'source_present':True,'claim':expected,'source':observed}]


def ordered_conditions(projection):
    if isinstance(projection.get('conditions'), (list,tuple)):
        return {**projection,'conditions':sorted(projection['conditions'],key=canonical)}
    return projection

def compare(claim,check):
    required={'candidate_fingerprint','source_projection','projection_complete','unmapped_restrictions','field_differences'}
    if type(check) is not dict or set(check)!=required:raise ValueError('Invalid proposition check schema')
    if check['candidate_fingerprint']!=claim.fingerprint:raise ValueError('Proposition check belongs to another immutable claim')
    if type(check['projection_complete']) is not bool or type(check['unmapped_restrictions']) is not list:
        raise ValueError('Invalid projection completeness or restrictions')
    projection=check['source_projection']
    if type(projection) is not dict or set(projection)-FIELDS:raise ValueError('Invalid source projection fields')
    # Incomplete interpretations remain inspectable, never eligible for support.
    # A complete projection must satisfy the full typed contract without repair.
    if check['projection_complete']:
        if set(projection)!=FIELDS:raise ValueError('Incomplete complete-projection schema')
        CandidateClaim.from_dict({'claim_id':'projection-validation',**projection})
    expected=claim.proposition()
    # conditions are a conjunction; ordering alone must not create a mismatch.
    expected['conditions']=sorted(expected['conditions'],key=canonical)
    projection=ordered_conditions(projection)
    delta=differences(expected,projection)
    if check['field_differences']!=delta:raise ValueError('Altered proposition difference audit')
    return check['projection_complete'] and not check['unmapped_restrictions'] and not delta

def make_check(claim,projection,complete,restrictions):
    expected=claim.proposition();expected['conditions']=sorted(expected['conditions'],key=canonical)
    if type(projection) is not dict:raise ValueError('Source projection must be an object')
    source=ordered_conditions(projection)
    return {'candidate_fingerprint':claim.fingerprint,'source_projection':projection,
            'projection_complete':complete,'unmapped_restrictions':restrictions,
            'field_differences':differences(expected,source)}
