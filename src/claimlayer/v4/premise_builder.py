"""Layer 3: semantic premises with independently checkable source binding."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Mapping
from ..semantic.models import EvidenceRef, SourceDocument
from .contracts import ContextBundle, TaskSpec, PremiseSet, ContractError, problem
from .context import documents
from .diff_evidence import parse_unified_diff, verify_diff_presence, diff_premise_metadata, find_revision_pair
from .premise_grounding import verify_premise_grounding
from .reconstruction import premise_fidelity


def _reference(raw,doc):
    if raw.get('revision') and raw['revision']!=doc.revision:
        raise ValueError('SOURCE_REVISION_MISMATCH')
    a,b=raw.get('line_start'),raw.get('line_end')
    if a is not None or b is not None:
        lines=doc.text.split('\n')
        if type(a) is not int or type(b) is not int or not 1<=a<=b<=len(lines):
            raise ValueError('INVALID_LINE_RANGE')
        start=sum(len(x)+1 for x in lines[:a-1]);end=sum(len(x)+1 for x in lines[:b])-1
        end=min(end,len(doc.text))
        quote=doc.text[start:end]
        if raw.get('quote') is not None and raw['quote']!=quote:
            raise ValueError('QUOTE_RANGE_MISMATCH')
    else:
        quote=raw.get('quote')
        if not isinstance(quote,str) or not quote:
            raise ValueError('MISSING_SOURCE_REFERENCE')
        start=doc.text.find(quote);end=start+len(quote)
        if start<0:raise ValueError('QUOTE_NOT_FOUND_IN_SOURCE')
    if not quote:raise ValueError('EMPTY_SOURCE_REFERENCE')
    return EvidenceRef(doc.document_id,doc.revision,start,end,quote)


def bind_premise(raw: Mapping[str,Any], docs: Mapping[str,SourceDocument], registry: Any) -> dict[str,Any]:
    """Pure source checks. Values interpreted by the model remain explicitly unverified."""
    pid=str(raw.get('claim_id') or '')
    if not pid:raise ValueError('MISSING_PREMISE_ID')
    doc=docs.get(str(raw.get('document_id') or ''))
    if doc is None:raise ValueError('SOURCE_DOCUMENT_NOT_FOUND')
    ref=_reference(raw,doc)
    kind=raw.get('kind','literal')
    if kind not in ('literal','interpreted','quote','diff_presence'):
        raise ValueError('UNKNOWN_PREMISE_KIND')
    dmap=parse_unified_diff(doc.text)
    diff=None;presence=None;binding=None;checks=['source_document_and_revision','verbatim_span_bound']
    if dmap is not None:
        binding=dmap.binding_for_span(ref.start,ref.end)
        if binding.get('document_lines'):
            diff=diff_premise_metadata(dmap,doc.document_id,*binding['document_lines'])
    subject=str(raw.get('entity') or raw.get('subject') or '')
    predicate=str(raw.get('predicate') or '')
    value=raw.get('value');unit=raw.get('unit');fixed=False
    if kind=='diff_presence':
        defects,presence=verify_diff_presence(raw,doc.text,dmap,find_revision_pair(d.text for d in docs.values()))
        if defects:raise ValueError(';'.join(d['code'] for d in defects))
        subject=presence['file'];predicate='text_present';value=True;unit=None;fixed=True
        checks.append('diff_presence_verified')
    elif kind=='quote':
        subject=next(iter(dmap.files()),doc.document_id) if dmap is not None else doc.document_id
        predicate='text_span';value=ref.quote;unit=None;fixed=True
    elif kind=='literal':
        gr=verify_premise_grounding({**raw,'quote':ref.quote,'revision':doc.revision},docs,unit_registry=registry)
        if not gr.grounded:raise ValueError(';'.join(d.code for d in gr.defects))
        value,unit=gr.normalized_value,gr.normalized_unit
        checks.extend(gr.checks_passed)
        fixed=premise_fidelity(value)=='literal'
    if not subject or not predicate:raise ValueError('MISSING_PREMISE_SUBJECT_OR_PREDICATE')
    if unit is not None:registry.lookup(str(unit))
    conditions=raw.get('conditions',[]);scope=raw.get('scope',{})
    if not isinstance(conditions,list) or not isinstance(scope,dict):raise ValueError('INVALID_PREMISE_SCOPE_OR_CONDITIONS')
    if kind in ('diff_presence','quote'):
        # The model cannot attach behavioral conditions to a mechanical span fact.
        conditions=[];scope={}
    if type(raw.get('polarity',1)) is not int or raw.get('polarity',1) not in (1,-1):raise ValueError('INVALID_PREMISE_POLARITY')
    return dict(claim_id=pid,kind=kind,entity=subject,predicate=predicate,value=value,unit=unit,
                polarity=raw.get('polarity',1),conditions=conditions,scope=scope,
                revision_mode=presence['side'] if presence else raw.get('revision_mode','source'),evidence=asdict(ref),
                checks_passed=list(dict.fromkeys(checks)),value_fixed_by_source=fixed,
                semantic_origin='source_procedure' if kind in ('quote','diff_presence') else 'model',
                grade='A' if fixed else 'E',diff=diff,diff_binding=binding,diff_presence=presence,
                event=raw.get('event',''),direction=raw.get('direction'),anchor_reference=raw.get('anchor_reference'))


def build_premises(raw: Any, context: ContextBundle, task: TaskSpec, profile: Any,
                   producer: Mapping[str,Any]) -> PremiseSet:
    if not isinstance(raw,list) or len(raw)>512:
        raise ContractError('INVALID_PREMISES','premises must be a list of at most 512 entries','premises')
    docs=documents(context);records=[];errors=[];seen=set()
    for p in raw:
        try:
            if not isinstance(p,dict):raise ValueError('INVALID_PREMISE')
            if p.get('claim_id') in seen:raise ValueError('DUPLICATE_PREMISE_ID')
            seen.add(p.get('claim_id'))
            records.append(bind_premise(p,docs,profile.effective_unit_registry()))
        except (ValueError,TypeError,KeyError) as exc:
            errors.append(problem(str(exc),'Premise source binding rejected',stage='premises',
                                  detected_by='source_binding',repair_owner='premises',claim_id=p.get('claim_id') if isinstance(p,dict) else None,
                                  required_change='When using a line range, omit quote or copy the exact source range including diff prefixes' if str(exc)=='QUOTE_RANGE_MISMATCH' else 'Repair the cited source reference or premise fields'))
    return PremiseSet.create(dict(proposed=raw,admitted=records,rejected=errors),
                             producer={**producer,'component':'premises','version':'premises-1'},parents=[context,task])
