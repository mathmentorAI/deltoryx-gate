"""Layer 1: preserve source identity, source coordinates and available context."""
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Mapping
from ..semantic.models import SourceDocument, digest
from .contracts import ContextBundle

CONTEXT_VERSION = 'context-1'
SOURCE_METADATA = ('path','reviewed_revision','revision','kind','completeness','line_start','line_end','coordinate_system')
CONTEXT_METADATA = ('path','reviewed_revision','line_start','line_end','coordinate_system','completeness')


def build_context(sources: Any, claim: str, *, context: Mapping[str, Any] | None = None,
                  as_of: str | None = None) -> ContextBundle:
    if not isinstance(claim,str) or not claim.strip():
        raise ValueError('claim must be nonempty text')
    if isinstance(sources,Mapping):
        rows=list(sources.items())
    elif isinstance(sources,(list,tuple)):
        rows=[(f'doc_{i}',v) for i,v in enumerate(sources,1)]
    else:
        raise ValueError('sources must be a mapping or sequence')
    if not rows:
        raise ValueError('At least one source is required')
    docs=[];seen=set()
    for key,value in rows:
        meta={}
        if isinstance(value,SourceDocument):
            doc=value
        elif isinstance(value,str):
            doc=SourceDocument(str(key),value,authority_rank=80,weight=1.0)
        elif isinstance(value,Mapping):
            did=str(value.get('document_id') or value.get('id') or key)
            doc=SourceDocument(did,str(value['text']),authority_rank=int(value.get('authority_rank',80)),
                               weight=float(value.get('weight',1.0)),
                               dependence_group=value.get('dependence_group'),
                               attribution=str(value.get('attribution','source_statement')))
            meta={k:value[k] for k in SOURCE_METADATA if k in value}
        else:
            raise ValueError(f'Unsupported source {key!r}')
        if doc.document_id in seen:
            raise ValueError(f'Duplicate document_id: {doc.document_id}')
        seen.add(doc.document_id)
        docs.append(dict(document=asdict(doc), revision=doc.revision, content_digest=digest(doc.text),
                         metadata={**meta,'source_coordinate_system':meta.get('coordinate_system','unknown'),
                                   'coordinate_system':'document_lines_1_based',
                                   'completeness':meta.get('completeness','unknown')}))
    return ContextBundle.create(dict(original_claim=claim,claim_digest=digest(claim),as_of=as_of,
                                     metadata={k:v for k,v in (context or {}).items() if k in CONTEXT_METADATA},
                                     sources=docs),producer=dict(component='context',version=CONTEXT_VERSION))


def documents(bundle: ContextBundle) -> dict[str,SourceDocument]:
    return {d['document']['document_id']:SourceDocument(**d['document']) for d in bundle.payload['sources']}
