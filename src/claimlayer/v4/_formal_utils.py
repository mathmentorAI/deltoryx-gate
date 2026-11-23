"""Pure source normalization, unit selection and formal comparison utilities."""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence
from ..semantic.models import SourceDocument, canonical, digest
from .domain_profile import DomainProfile
from .operations import normalize_weekday
from .vocabulary import STANDARD_UNITS, UNIT_PROFILES, UnitRegistry, UnmappedProposal, label_key, normalize_date

def normalize_sources(
    sources: (
        Mapping[str, SourceDocument]
        | Mapping[str, str]
        | Mapping[str, Mapping[str, Any]]
        | Sequence[SourceDocument | Mapping[str, Any] | str]
    ),
) -> tuple[dict[str, SourceDocument], list[SourceDocument]]:
    """Coerce flexible consumer source containers into validated SourceDocument instances."""
    docs_map: dict[str, SourceDocument] = {}
    docs_list: list[SourceDocument] = []
    if isinstance(sources, Mapping):
        for key, val in sources.items():
            if isinstance(val, SourceDocument):
                doc = val
            elif isinstance(val, str):
                doc = SourceDocument(document_id=str(key), text=val, authority_rank=80, weight=1.0)
            elif isinstance(val, Mapping):
                doc = SourceDocument(
                    document_id=str(val.get("document_id") or val.get("id") or key),
                    text=str(val["text"]),
                    authority_rank=int(val.get("authority_rank", 80)),
                    weight=float(val.get("weight", 1.0)),
                    dependence_group=val.get("dependence_group"),
                    attribution=str(val.get("attribution", "source_statement")),
                )
            else:
                raise ValueError(f"Unsupported source entry for '{key}': {type(val).__name__}.")
            docs_map[doc.document_id] = doc
            docs_list.append(doc)
        return docs_map, docs_list

    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for idx, item in enumerate(sources):
            if isinstance(item, SourceDocument):
                doc = item
            elif isinstance(item, str):
                doc = SourceDocument(document_id=f"doc_{idx + 1}", text=item, authority_rank=80, weight=1.0)
            elif isinstance(item, Mapping):
                doc = SourceDocument(
                    document_id=str(item.get("document_id") or item.get("id") or f"doc_{idx + 1}"),
                    text=str(item["text"]),
                    authority_rank=int(item.get("authority_rank", 80)),
                    weight=float(item.get("weight", 1.0)),
                    dependence_group=item.get("dependence_group"),
                    attribution=str(item.get("attribution", "source_statement")),
                )
            else:
                raise ValueError(f"Unsupported source entry at index {idx}: {type(item).__name__}.")
            docs_map[doc.document_id] = doc
            docs_list.append(doc)
        return docs_map, docs_list

    raise ValueError("sources must be a mapping or sequence of SourceDocuments/strings.")

def _to_decimal_safe(val: Any) -> Decimal | None:
    if isinstance(val, bool) or val is None:
        return None
    try:
        d = Decimal(str(val).strip())
        return d if d.is_finite() else None
    except (InvalidOperation, ValueError):
        return None

def _values_match(
    val_a: Any,
    unit_a: str | None,
    val_b: Any,
    unit_b: str | None,
    registry: UnitRegistry,
    *,
    claim_text: str | None = None,
) -> bool:
    wa = normalize_weekday(val_a) if isinstance(val_a, str) else None
    wb = normalize_weekday(val_b) if isinstance(val_b, str) else None
    if wa is not None and wb is not None:
        return wa[0] == wb[0]
    if (wa is not None) != (wb is not None):
        return False

    if isinstance(val_a, bool) or isinstance(val_b, bool):
        return isinstance(val_a, bool) and isinstance(val_b, bool) and (val_a is val_b)

    da = _to_decimal_safe(val_a)
    db = _to_decimal_safe(val_b)
    if da is not None and db is not None:
        # Check units carefully: if both units provided, dimensions must match; if claim_text states foreign currency/unit, reject
        eff_ub = unit_b
        if eff_ub is None and unit_a is not None and claim_text:
            low_claim = claim_text.casefold()
            try:
                ua_obj = registry.lookup(unit_a)
                for other_u in registry._units.values():
                    if other_u.dimension == ua_obj.dimension:
                        continue
                    for alias in (other_u.name, *other_u.aliases):
                        al_low = alias.casefold()
                        if al_low and al_low in low_claim and not (
                            ua_obj.name.casefold() in low_claim or any(a.casefold() in low_claim for a in ua_obj.aliases)
                        ):
                            return False
            except UnmappedProposal:
                pass
        if unit_a is not None and eff_ub is not None:
            try:
                ua_obj = registry.lookup(unit_a)
                ub_obj = registry.lookup(eff_ub)
                if ua_obj.dimension != ub_obj.dimension:
                    return False
                ba = da * Decimal(ua_obj.factor)
                bb = db * Decimal(ub_obj.factor)
                tol = Decimal("1e-9") * max(Decimal(1), abs(ba), abs(bb))
                return abs(ba - bb) <= tol
            except UnmappedProposal:
                return label_key(unit_a) == label_key(eff_ub) and da == db
        return da == db

    if isinstance(val_a, str) and isinstance(val_b, str):
        try:
            na = normalize_date(val_a)
            nb = normalize_date(val_b)
            return na == nb
        except ValueError:
            pass
        return label_key(val_a) == label_key(val_b)

    return canonical(val_a) == canonical(val_b) and unit_a == unit_b

def _value_kind(val: Any) -> str:
    if isinstance(val, bool):
        return "boolean"
    if isinstance(val, str) and normalize_weekday(val) is not None:
        return "weekday"
    if _to_decimal_safe(val) is not None:
        return "number"
    if isinstance(val, str):
        try:
            normalize_date(val)
            return "date"
        except ValueError:
            return "text"
    return "structured" if val is not None else "null"

def _comparable(val_a: Any, val_b: Any) -> bool:
    """A computed conclusion can only confirm or contradict a claim value of the same kind.
    Mismatched kinds (e.g. boolean vs text) are a planning defect, never a refutation."""
    return _value_kind(val_a) == _value_kind(val_b) and _value_kind(val_a) != "null"

def _coerce_polarity(raw: Any) -> int:
    if isinstance(raw, str):
        return -1 if raw.strip().lower() in {"-1", "negated", "negative"} else 1
    try:
        return -1 if int(raw) == -1 else 1
    except (TypeError, ValueError):
        return 1

def _topological_steps(
    steps: Sequence[Mapping[str, Any]],
    leaf_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate step IDs, missing inputs, and cycles, returning steps in execution order."""
    errors: list[str] = []
    produced_by: dict[str, int] = {}
    step_dicts = [dict(s) for s in steps]
    for idx, st in enumerate(step_dicts):
        cid = str(st.get("conclusion_id") or "").strip()
        if not cid:
            errors.append(f"Step at index {idx} has empty conclusion_id.")
        elif cid in leaf_ids:
            errors.append(f"Step conclusion_id '{cid}' collides with a premise claim_id.")
        elif cid in produced_by:
            errors.append(f"Duplicate step conclusion_id '{cid}'.")
        else:
            produced_by[cid] = idx

    if errors:
        return [], errors

    available: set[str] = set(leaf_ids)
    remaining = list(range(len(step_dicts)))
    ordered: list[dict[str, Any]] = []
    while remaining:
        progressed = False
        next_remaining: list[int] = []
        for idx in remaining:
            st = step_dicts[idx]
            prems = [str(p) for p in (st.get("premises") or ())]
            if all(p in available for p in prems):
                ordered.append(st)
                available.add(str(st["conclusion_id"]))
                progressed = True
            else:
                next_remaining.append(idx)
        if not progressed:
            unresolved_details = []
            for idx in next_remaining:
                st = step_dicts[idx]
                missing = [str(p) for p in (st.get("premises") or ()) if str(p) not in available]
                unresolved_details.append(f"{st.get('step_id', idx)} missing/cyclic {missing}")
            errors.append(f"Cyclic or unresolved step dependencies: {'; '.join(unresolved_details)}.")
            return [], errors
        remaining = next_remaining
    return ordered, []

UNIT_PROFILE_CHOICES = ("auto", "standard", "es")

_ES_MARKERS = frozenset(
    "el la los las del que y es fue era son hoy ayer mañana días dias antes después despues "
    "con por una un su sus se para al lo como quedé quedamos cuando entre sobre está esta".split()
)

_EN_MARKERS = frozenset(
    "the and of is was were are today yesterday tomorrow days before after with by an his her "
    "its for to as on at that this when between".split()
)

def detect_text_language(*texts: str) -> str:
    """Deterministic lexical language guess used ONLY to choose a unit-label profile.

    Returns 'es', 'en' or 'und'. It never touches premises or verdicts: it only decides which
    surface labels (e.g. 'días', 'dólares') the unit registry recognises. The choice is recorded
    in telemetry so every result stays reproducible.
    """
    joined = " ".join(t for t in texts if isinstance(t, str)).lower()
    tokens = re.findall(r"[a-záéíóúüñ]+", joined)
    es = sum(1 for t in tokens if t in _ES_MARKERS)
    en = sum(1 for t in tokens if t in _EN_MARKERS)
    if re.search(r"[ñ¿¡áéíóú]", joined):
        es += 2
    if es > en:
        return "es"
    if en > es:
        return "en"
    return "und"

def resolve_unit_profile(unit_profile: str, claim: str, source_texts: Sequence[str]) -> tuple[str, UnitRegistry]:
    """Map 'auto' | 'standard' | 'es' to a named registry from UNIT_PROFILES."""
    if unit_profile not in UNIT_PROFILE_CHOICES:
        raise ValueError(f"unit_profile must be one of {UNIT_PROFILE_CHOICES}, got {unit_profile!r}")
    name = unit_profile
    if name == "auto":
        name = "es" if detect_text_language(claim, *source_texts) == "es" else "standard"
    return name, UNIT_PROFILES[name]

def _resolve_derivation_profile(
    domain_profile: DomainProfile | None,
    unit_profile: str,
    claim: str,
    docs_map: Mapping[str, SourceDocument],
) -> tuple[DomainProfile, str, str]:
    """Return (profile, unit_profile_label, unit_registry_fingerprint).

    An explicitly supplied DomainProfile always wins (its own units are used unchanged).
    Otherwise a default profile is built over the named unit profile.
    """
    if domain_profile is not None:
        reg = domain_profile.effective_unit_registry()
        return domain_profile, f"domain_profile:{domain_profile.profile_id}", digest(reg.describe())
    name, reg = resolve_unit_profile(unit_profile, claim, [d.text for d in docs_map.values()])
    return DomainProfile(profile_id="default", unit_registry=reg), name, digest(reg.describe())
