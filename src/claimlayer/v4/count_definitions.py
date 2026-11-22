"""Declared count definitions: what a reproducible count measures.

`count_lines` reproduces how many lines match a criterion. Whether that criterion counts the
attribute a claim talks about (e.g. "test cases") is a separate fact. When the planner chooses the
criterion itself, that link is an interpretation and the verifier abstains. A CountDefinition is a
declared rule (grade B) supplied by the caller or domain profile — never by the planner — stating
which attribute a criterion counts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .reconstruction import content_tokens

# Attribute words for which a structural line count (no filter) is the definition itself.
LINE_ATTRIBUTE_TERMS = frozenset({"linea", "lineas", "line", "lines", "loc"})


@dataclass(frozen=True)
class CountDefinition:
    definition_id: str
    version: str
    attribute_terms: frozenset[str]
    contains: str | None = None
    line_prefix: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"definition_id": self.definition_id, "version": self.version,
                "attribute_terms": sorted(self.attribute_terms), "contains": self.contains,
                "line_prefix": self.line_prefix, "description": self.description}

    def counts_attribute(self, claim_attribute: Any) -> bool:
        return bool(content_tokens(claim_attribute) & set(self.attribute_terms))


def count_definition_registry(defs: Iterable[CountDefinition] | Mapping[str, CountDefinition] | None) -> dict[str, CountDefinition]:
    if defs is None:
        return {}
    if isinstance(defs, Mapping):
        return dict(defs)
    return {d.definition_id: d for d in defs}


def structural_line_count_matches(claim_attribute: Any) -> bool:
    return bool(content_tokens(claim_attribute) & LINE_ATTRIBUTE_TERMS)
