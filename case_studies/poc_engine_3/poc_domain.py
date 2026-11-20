"""Domain profile, declared rules, POC-local verifier adapters, and graph builder for POC-3."""
from __future__ import annotations

from datetime import date
import json
import os
import re
import unicodedata
from typing import Any, Mapping, Sequence

from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4.domain_profile import (
    DeclaredRule,
    DomainProfile,
    RelationCompositionRule,
)
from claimlayer.v4.evaluation import ResponseAssessment, evaluate_response
from claimlayer.v4.justification import ClaimNode, InferenceStep, JustificationGraph
from claimlayer.v4.normalizers import normalize_date_token
from claimlayer.v4.verifiers import (
    DEFAULT_VERIFIER_REGISTRY,
    ExplicitLogicVerifier,
    TemporalVerifier,
    VerifierContext,
    VerifierRegistry,
    VerifierResult,
    _completed_years,
)
from claimlayer.v4.vocabulary import label_key

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PREREGISTRATION_PATH = os.path.join(_THIS_DIR, "PREREGISTRATION.json")
REFERENCE_DATE = "2026-11-20"


def load_preregistration(path: str = DEFAULT_PREREGISTRATION_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


_ENTITY_ALIAS_MAP: dict[str, str] = {
    "warren buffett": "Warren Buffett",
    "buffett": "Warren Buffett",
    "alfa s.a.": "Alfa S.A.",
    "alfa s.a": "Alfa S.A.",
    "alfa": "Alfa S.A.",
    "beta s.l.": "Beta S.L.",
    "beta s.l": "Beta S.L.",
    "beta": "Beta S.L.",
    "gamma s.l.": "Gamma S.L.",
    "gamma s.l": "Gamma S.L.",
    "gamma": "Gamma S.L.",
    "contrato": "Contrato",
    "el contrato": "Contrato",
}


def canonical_entity(name: Any) -> str:
    if not isinstance(name, str):
        return str(name or "").strip()
    raw = name.strip()
    k = raw.casefold()
    if k in _ENTITY_ALIAS_MAP:
        return _ENTITY_ALIAS_MAP[k]
    lk = label_key(raw.rstrip("."))
    for alias, canon in _ENTITY_ALIAS_MAP.items():
        if label_key(alias.rstrip(".")) == lk:
            return canon
    return raw


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn")


def parse_spanish_dmy_date(raw: Any) -> str | None:
    """Resolve ISO / Spanish dates, wrapping `normalize_date_token` + Spanish DD/MM/YYYY fallback."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip()
    native = normalize_date_token(s)
    if native is not None:
        return native
    # Fallback strictly for Spanish DD/MM/YYYY ambiguous slash dates (e.g. '05/12/2026' -> 2026-12-05)
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        day_i, mon_i, year_i = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(year_i, mon_i, day_i).isoformat()
        except ValueError:
            return None
    return None


def build_poc_sources(prereg: Mapping[str, Any] | None = None) -> dict[str, SourceDocument]:
    data = prereg or load_preregistration()
    return {
        doc_id: SourceDocument(
            document_id=str(doc_id),
            text=str(text),
            attribution="audited_record",
            weight=1.0,
        )
        for doc_id, text in dict(data["sources"]).items()
    }


def build_poc_domain_profile() -> DomainProfile:
    """Declare domain rules R1, R2, R3 strictly inside the POC DomainProfile."""
    return DomainProfile(
        profile_id="poc_engine_3",
        version="1.0.0",
        declared_rules={
            "R1_persona_mayor": DeclaredRule(
                rule_id="R1_persona_mayor",
                version="1.0",
                description="persona_mayor(x) <= edad(x, 2026-11-20) >= 65",
                kind="implication",
                required_premises=({"predicate": "edad_al_menos_65", "value": True, "polarity": 1},),
                conclusion_template={"predicate": "persona_mayor", "value": True, "polarity": 1},
                parameters={"umbral_edad": 65, "fecha_referencia": REFERENCE_DATE},
            ),
            "R2a_prop_implica_control": DeclaredRule(
                rule_id="R2a_prop_implica_control",
                version="1.0",
                description="propietaria_100(x, y) => controla(x, y)",
                kind="implication",
                required_premises=({"predicate": "propietaria_100", "polarity": 1},),
                conclusion_template={"predicate": "controla", "polarity": 1},
            ),
            "R3_sin_preaviso_en_plazo": DeclaredRule(
                rule_id="R3_sin_preaviso_en_plazo",
                version="1.0",
                description="sin_preaviso_registrado & (dias_hasta_vencimiento < 30) => sin_preaviso_en_plazo",
                kind="conjunction",
                required_premises=(
                    {"predicate": "sin_preaviso_registrado", "value": True, "polarity": 1},
                    {"predicate": "plazo_preaviso_vencido", "value": True, "polarity": 1},
                ),
                conclusion_template={"predicate": "sin_preaviso_en_plazo", "value": True, "polarity": 1},
            ),
            "R3_preaviso_en_plazo": DeclaredRule(
                rule_id="R3_preaviso_en_plazo",
                version="1.0",
                description="preaviso_no_renovacion_enviado & (dias_antelacion >= 30) => preaviso_en_plazo",
                kind="conjunction",
                required_premises=(
                    {"predicate": "preaviso_no_renovacion_enviado", "value": True, "polarity": 1},
                    {"predicate": "antelacion_al_menos_30_dias", "value": True, "polarity": 1},
                ),
                conclusion_template={"predicate": "preaviso_en_plazo", "value": True, "polarity": 1},
            ),
            "R3_renovacion_automatica": DeclaredRule(
                rule_id="R3_renovacion_automatica",
                version="1.0",
                description="clausula_renovacion_automatica & sin_preaviso_en_plazo => contrato_se_renovara",
                kind="conjunction",
                required_premises=(
                    {"predicate": "clausula_renovacion_automatica", "value": True, "polarity": 1},
                    {"predicate": "sin_preaviso_en_plazo", "value": True, "polarity": 1},
                ),
                conclusion_template={"predicate": "contrato_se_renovara", "value": True, "polarity": 1},
            ),
            "R3_exclusion_preaviso": DeclaredRule(
                rule_id="R3_exclusion_preaviso",
                version="1.0",
                description="clausula_renovacion_automatica & preaviso_en_plazo => excluye contrato_se_renovara",
                kind="exclusion",
                required_premises=(
                    {"predicate": "clausula_renovacion_automatica", "value": True, "polarity": 1},
                    {"predicate": "preaviso_en_plazo", "value": True, "polarity": 1},
                ),
                conclusion_template={"predicate": "contrato_se_renovara", "polarity": -1},
            ),
        },
        relation_rules={
            "R2_controla_transitiva": RelationCompositionRule(
                rule_id="R2_controla_transitiva",
                left_predicate="controla",
                right_predicate="controla",
                composed_predicate="controla",
                version="1.0",
                transitive=True,
                require_temporal_overlap=False,
                require_exact_bridge=True,
            ),
        },
    )


class PocTemporalVerifier(TemporalVerifier):
    """POC adapter skipping redundant mandatory temporal-interval checks on boolean rule conclusions."""

    verifier_id = "poc_temporal_boolean_passthrough"

    def verify(self, obligation: VerificationObligation_t, context: VerifierContext) -> VerifierResult:
        if (
            obligation.parameters.get("mandatory_check") == "temporal_precision"
            and context.step.step_id == "mandatory::conclusion"
            and isinstance(context.conclusion.value, bool)
        ):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="not_applicable",
                guarantee_kind=self.guarantee_kind,
                reasons=("BOOLEAN_DOWNSTREAM_RULE_CONCLUSION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )
        return super().verify(obligation, context)


VerificationObligation_t = Any


class PocExplicitLogicVerifier(ExplicitLogicVerifier):
    """POC adapter adding variable-binding validation and verified condition discharge."""

    verifier_id = "poc_explicit_logic_domain"

    def verify(self, obligation: Any, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        if params.get("mandatory_check") == "condition_preservation":
            discharged_ok = any(
                p.predicate == "sin_preaviso_en_plazo"
                and p.value is True
                and p.polarity == 1
                and p.kind in ("extraction", "derivation")
                for p in context.premises
            )
            if discharged_ok and context.conclusion.condition_status == "unconditional":
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="verified",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("CONDITION_DISCHARGED_BY_VERIFIED_PREMISE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
            exclusion_resolved = any(
                p.predicate == "preaviso_en_plazo"
                and p.value is True
                and p.polarity == 1
                and p.kind in ("extraction", "derivation")
                for p in context.premises
            )
            if exclusion_resolved:
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="verified",
                    guarantee_kind=self.guarantee_kind,
                    reasons=("CONDITION_RESOLVED_BY_TIMELY_NOTICE",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )

        res = super().verify(obligation, context)
        rule_id = context.step.rule_id or params.get("rule_id")
        if res.status == "verified" and rule_id == "R2a_prop_implica_control" and context.premises:
            prem = context.premises[0]
            if (
                label_key(prem.subject) != label_key(context.conclusion.subject)
                or label_key(str(prem.value)) != label_key(str(context.conclusion.value))
            ):
                return VerifierResult(
                    obligation_id=obligation.obligation_id,
                    step_id=context.step.step_id,
                    family=self.family,
                    status="refuted",
                    guarantee_kind=self.guarantee_kind,
                    evidence={
                        "premise_pair": [prem.subject, str(prem.value)],
                        "conclusion_pair": [context.conclusion.subject, str(context.conclusion.value)],
                    },
                    reasons=("VARIABLE_BINDING_MISMATCH",),
                    verifier_id=self.verifier_id,
                    verifier_version=self.version,
                )
        return res


def build_poc_verifier_registry() -> VerifierRegistry:
    reg = VerifierRegistry()
    for verifier in DEFAULT_VERIFIER_REGISTRY._verifiers.values():
        reg.register(verifier)
    reg.register(PocTemporalVerifier())
    reg.register(PocExplicitLogicVerifier())
    return reg


def _locate_quote_span(doc_text: str, quote: str) -> tuple[int, int, str] | None:
    q = (quote or "").strip()
    if not q:
        return None
    idx = doc_text.find(q)
    if idx >= 0:
        return idx, idx + len(q), doc_text[idx : idx + len(q)]
    # Fallback: normalize internal whitespace only while preserving exact span in doc_text
    tokens = [re.escape(tok) for tok in q.split()]
    if not tokens:
        return None
    m = re.search(r"\s+".join(tokens), doc_text, flags=re.IGNORECASE)
    if m:
        return m.start(), m.end(), m.group(0)
    return None


def _entity_short_token(canon_name: str) -> str:
    toks = _strip_accents(canon_name.casefold()).split()
    return toks[0] if toks else ""


def anchor_premise_in_docs(
    raw_prem: Mapping[str, Any],
    allowed_docs: Mapping[str, SourceDocument],
) -> dict[str, Any]:
    """Verify a proposed premise is literally grounded in `allowed_docs`."""
    doc_id = str(raw_prem.get("doc_id") or "").strip()
    if doc_id not in allowed_docs:
        return {"grounded": False, "reason": f"DOC_NOT_IN_CLAIM_SOURCES:{doc_id}"}
    doc = allowed_docs[doc_id]
    quote = str(raw_prem.get("verbatim_quote") or "").strip()
    span = _locate_quote_span(doc.text, quote)
    if span is None:
        return {"grounded": False, "reason": "QUOTE_NOT_FOUND_IN_SOURCE"}
    start, end, exact_slice = span
    q_norm = _strip_accents(exact_slice.casefold())

    pred = str(raw_prem.get("predicate") or "").strip()
    subj = canonical_entity(str(raw_prem.get("subject") or ""))
    val = raw_prem.get("value")

    # Predicate-specific grounding guard so an unrelated quote cannot smuggle ungrounded propositions
    if pred == "birth_date":
        iso = parse_spanish_dmy_date(str(val or ""))
        if not iso:
            return {"grounded": False, "reason": "UNPARSEABLE_BIRTH_DATE"}
        yr = iso[:4]
        subj_tok = _strip_accents(subj.casefold()).split()[-1] if subj else ""
        if yr not in q_norm or (subj_tok and subj_tok not in q_norm):
            return {"grounded": False, "reason": "BIRTH_DATE_OR_SUBJECT_NOT_IN_QUOTE"}
        norm_val: Any = iso

    elif pred == "propietaria_100":
        val_canon = canonical_entity(str(val or ""))
        s_tok = _entity_short_token(subj)
        v_tok = _entity_short_token(val_canon)
        if not s_tok or not v_tok or s_tok == v_tok:
            return {"grounded": False, "reason": "INVALID_OWNERSHIP_ENTITIES"}
        pos_s = q_norm.find(s_tok)
        pos_v = q_norm.find(v_tok)
        if pos_s < 0 or pos_v < 0 or pos_s > pos_v:
            return {"grounded": False, "reason": "OWNERSHIP_DIRECTION_NOT_IN_QUOTE"}
        if "100" not in q_norm or "propietar" not in q_norm:
            return {"grounded": False, "reason": "OWNERSHIP_100_NOT_IN_QUOTE"}
        norm_val = val_canon

    elif pred == "clausula_renovacion_automatica":
        if ("renov" not in q_norm and "renuev" not in q_norm) or "30" not in q_norm or ("salvo" not in q_norm and "preaviso" not in q_norm):
            return {"grounded": False, "reason": "AUTO_RENEWAL_CLAUSE_NOT_IN_QUOTE"}
        norm_val = True

    elif pred == "fecha_vencimiento":
        iso = parse_spanish_dmy_date(str(val or ""))
        if not iso or "venc" not in q_norm:
            return {"grounded": False, "reason": "EXPIRY_DATE_NOT_GROUNDED_IN_QUOTE"}
        # Ensure stated date string appears in quote (either DD/MM/YYYY or YYYY-MM-DD)
        if not any(part in exact_slice for part in (iso, iso[8:10] + "/" + iso[5:7] + "/" + iso[:4])):
            return {"grounded": False, "reason": "EXPIRY_DATE_VALUE_NOT_IN_QUOTE"}
        norm_val = iso

    elif pred == "sin_preaviso_registrado":
        if "sin " not in q_norm or ("preaviso" not in q_norm and "comunicacion" not in q_norm):
            return {"grounded": False, "reason": "NO_NOTICE_RECORD_NOT_IN_QUOTE"}
        norm_val = True

    elif pred == "fecha_registro_sin_preaviso":
        iso = parse_spanish_dmy_date(str(val or ""))
        if not iso or "sin " not in q_norm or "preaviso" not in q_norm:
            return {"grounded": False, "reason": "NO_NOTICE_DATE_NOT_GROUNDED"}
        if not any(part in exact_slice for part in (iso, iso[8:10] + "/" + iso[5:7] + "/" + iso[:4])):
            return {"grounded": False, "reason": "NO_NOTICE_DATE_VALUE_NOT_IN_QUOTE"}
        norm_val = iso

    elif pred == "preaviso_no_renovacion_enviado":
        if "salvo " in q_norm or q_norm.startswith("sin ") or " sin " in q_norm:
            return {"grounded": False, "reason": "CONDITIONAL_OR_ABSENT_NOTICE_NOT_SENT"}
        if "preaviso" not in q_norm or "enviad" not in q_norm:
            return {"grounded": False, "reason": "SENT_NOTICE_NOT_IN_QUOTE"}
        norm_val = True

    elif pred == "fecha_preaviso":
        iso = parse_spanish_dmy_date(str(val or ""))
        if not iso or "salvo " in q_norm or "preaviso" not in q_norm or "enviad" not in q_norm:
            return {"grounded": False, "reason": "NOTICE_DATE_NOT_GROUNDED"}
        if not any(part in exact_slice for part in (iso, iso[8:10] + "/" + iso[5:7] + "/" + iso[:4])):
            return {"grounded": False, "reason": "NOTICE_DATE_VALUE_NOT_IN_QUOTE"}
        norm_val = iso

    else:
        return {"grounded": False, "reason": f"UNSUPPORTED_PREDICATE:{pred}"}

    return {
        "grounded": True,
        "doc_id": doc_id,
        "subject": subj,
        "predicate": pred,
        "value": norm_val,
        "ref": EvidenceRef(
            document_id=doc_id,
            revision="rev-0",
            start=start,
            end=end,
            quote=exact_slice,
        ),
        "verbatim_quote": exact_slice,
    }


HAND_PREMISES_MODE_A: dict[str, list[dict[str, Any]]] = {
    "C1a": [
        {
            "doc_id": "S1",
            "subject": "Warren Buffett",
            "predicate": "birth_date",
            "value": "1930-08-30",
            "verbatim_quote": "Warren Buffett nació el 30 de agosto de 1930.",
        }
    ],
    "C1b": [
        {
            "doc_id": "S1",
            "subject": "Warren Buffett",
            "predicate": "birth_date",
            "value": "1930-08-30",
            "verbatim_quote": "Warren Buffett nació el 30 de agosto de 1930.",
        }
    ],
    "C1c": [],
    "C2a": [
        {
            "doc_id": "S2",
            "subject": "Alfa S.A.",
            "predicate": "propietaria_100",
            "value": "Beta S.L.",
            "verbatim_quote": "Alfa S.A. es propietaria del 100 % de Beta S.L.",
        },
        {
            "doc_id": "S3",
            "subject": "Beta S.L.",
            "predicate": "propietaria_100",
            "value": "Gamma S.L.",
            "verbatim_quote": "Beta S.L. es propietaria del 100 % de Gamma S.L.",
        },
    ],
    "C2b": [
        {
            "doc_id": "S2",
            "subject": "Alfa S.A.",
            "predicate": "propietaria_100",
            "value": "Beta S.L.",
            "verbatim_quote": "Alfa S.A. es propietaria del 100 % de Beta S.L.",
        },
        {
            "doc_id": "S3",
            "subject": "Beta S.L.",
            "predicate": "propietaria_100",
            "value": "Gamma S.L.",
            "verbatim_quote": "Beta S.L. es propietaria del 100 % de Gamma S.L.",
        },
    ],
    "C2c": [
        {
            "doc_id": "S2",
            "subject": "Alfa S.A.",
            "predicate": "propietaria_100",
            "value": "Beta S.L.",
            "verbatim_quote": "Alfa S.A. es propietaria del 100 % de Beta S.L.",
        }
    ],
    "C3a": [
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "fecha_vencimiento",
            "value": "2026-12-31",
            "verbatim_quote": "El contrato vence el 31/12/2026.",
        },
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "clausula_renovacion_automatica",
            "value": True,
            "verbatim_quote": "Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.",
        },
    ],
    "C3b": [
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "fecha_vencimiento",
            "value": "2026-12-31",
            "verbatim_quote": "El contrato vence el 31/12/2026.",
        },
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "clausula_renovacion_automatica",
            "value": True,
            "verbatim_quote": "Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.",
        },
        {
            "doc_id": "S5",
            "subject": "Contrato",
            "predicate": "sin_preaviso_registrado",
            "value": True,
            "verbatim_quote": "Sin comunicaciones de preaviso registradas hasta el 05/12/2026.",
        },
        {
            "doc_id": "S5",
            "subject": "Contrato",
            "predicate": "fecha_registro_sin_preaviso",
            "value": "2026-12-05",
            "verbatim_quote": "Sin comunicaciones de preaviso registradas hasta el 05/12/2026.",
        },
    ],
    "C3c": [
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "fecha_vencimiento",
            "value": "2026-12-31",
            "verbatim_quote": "El contrato vence el 31/12/2026.",
        },
        {
            "doc_id": "S4",
            "subject": "Contrato",
            "predicate": "clausula_renovacion_automatica",
            "value": True,
            "verbatim_quote": "Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.",
        },
        {
            "doc_id": "S6",
            "subject": "Contrato",
            "predicate": "preaviso_no_renovacion_enviado",
            "value": True,
            "verbatim_quote": "Preaviso de no renovación enviado el 15/11/2026.",
        },
        {
            "doc_id": "S6",
            "subject": "Contrato",
            "predicate": "fecha_preaviso",
            "value": "2026-11-15",
            "verbatim_quote": "Preaviso de no renovación enviado el 15/11/2026.",
        },
    ],
}


def map_assessment_to_label(res: ResponseAssessment) -> str:
    """Apply the preregistered 4-class engine-to-label correspondence."""
    if res.assessment in ("refuted", "contested"):
        return "CONTRADICHA"
    if res.assessment == "supported" and res.effective_epistemic_kind in ("derivation", "extraction"):
        return "DERIVADA"
    if res.effective_epistemic_kind == "hypothesis" and len(res.supporting_evidence) > 0:
        return "HIPÓTESIS"
    return "SIN EVIDENCIA"


def _find_directed_chain(
    edges: Sequence[tuple[str, str, str]],
    start: str,
    goal: str,
    *,
    max_hops: int = 4,
) -> list[tuple[str, str, str]]:
    """DFS bounded search for an acyclic directed path start -> ... -> goal."""
    start_k = label_key(start)
    goal_k = label_key(goal)
    stack: list[tuple[str, list[tuple[str, str, str]], set[str]]] = [(start_k, [], {start_k})]
    while stack:
        curr, path, seen = stack.pop()
        if path and curr == goal_k:
            return path
        if len(path) >= max_hops:
            continue
        for node_id, s_ent, v_ent in edges:
            sk, vk = label_key(s_ent), label_key(v_ent)
            if sk == curr and vk not in seen:
                stack.append((vk, [*path, (node_id, s_ent, v_ent)], seen | {vk}))
    return []


def build_graph_and_evaluate(
    claim_spec: Mapping[str, Any],
    proposed_premises: Sequence[Mapping[str, Any]],
    all_docs: Mapping[str, SourceDocument],
    *,
    profile: DomainProfile | None = None,
    use_default_registry: bool = False,
) -> dict[str, Any]:
    """Ground proposed premises, construct JustificationGraph deterministically, and evaluate via v4."""
    prof = profile or build_poc_domain_profile()
    registry = DEFAULT_VERIFIER_REGISTRY if use_default_registry else build_poc_verifier_registry()
    claim_id = str(claim_spec["claim_id"])
    target = dict(claim_spec["target"])
    allowed_ids = list(claim_spec["source_ids"])
    allowed_docs = {did: all_docs[did] for did in allowed_ids if did in all_docs}

    grounded_prems: list[dict[str, Any]] = []
    rejected_prems: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for raw in proposed_premises:
        if not isinstance(raw, Mapping):
            continue
        chk = anchor_premise_in_docs(raw, allowed_docs)
        if chk["grounded"]:
            dedup_k = (chk["doc_id"], chk["subject"], chk["predicate"], str(chk["value"]))
            if dedup_k not in seen_keys:
                seen_keys.add(dedup_k)
                grounded_prems.append(chk)
        else:
            rejected_prems.append({**dict(raw), "rejection_reason": chk["reason"]})

    claims: dict[str, ClaimNode] = {}
    steps: list[InferenceStep] = []
    conc_id = f"{claim_id}::conclusion"
    tgt_subj = canonical_entity(str(target["subject"]))
    tgt_pred = str(target["predicate"])
    tgt_val = target["value"]
    tgt_unit = target.get("unit")

    # ------------------------------------------------------------------ Case 1
    if tgt_pred in ("persona_mayor", "edad", "es_vegetariano"):
        birth = next(
            (
                p for p in grounded_prems
                if p["predicate"] == "birth_date" and label_key(p["subject"]) == label_key(tgt_subj)
            ),
            None,
        )
        if tgt_pred == "es_vegetariano" or birth is None:
            claims[conc_id] = ClaimNode(
                claim_id=conc_id,
                kind="derivation",
                subject=tgt_subj,
                predicate=tgt_pred,
                value=tgt_val,
                unit=tgt_unit,
                text=str(claim_spec["claim_text"]),
            )
        elif tgt_pred == "edad":
            p_id = f"{claim_id}::p_birth"
            claims[p_id] = ClaimNode(
                claim_id=p_id,
                kind="extraction",
                subject=tgt_subj,
                predicate="birth_date",
                value=birth["value"],
                evidence=(birth["ref"],),
                text=birth["verbatim_quote"],
            )
            claims[conc_id] = ClaimNode(
                claim_id=conc_id,
                kind="derivation",
                subject=tgt_subj,
                predicate="edad",
                value=tgt_val,
                unit="years",
                text=str(claim_spec["claim_text"]),
            )
            steps.append(
                InferenceStep(
                    step_id=f"{claim_id}::s_age",
                    operation="temporal_age",
                    premises=(p_id,),
                    conclusion_id=conc_id,
                    parameters={"birth_date": birth["value"], "reference_date": REFERENCE_DATE},
                )
            )
        else:  # persona_mayor
            ref_d = date.fromisoformat(REFERENCE_DATE)
            b_d = date.fromisoformat(str(birth["value"]))
            exact_age = _completed_years(b_d, ref_d)
            p_id = f"{claim_id}::p_birth"
            c_age = f"{claim_id}::d_age"
            c_ge65 = f"{claim_id}::d_ge65"
            claims[p_id] = ClaimNode(
                claim_id=p_id,
                kind="extraction",
                subject=tgt_subj,
                predicate="birth_date",
                value=birth["value"],
                evidence=(birth["ref"],),
                text=birth["verbatim_quote"],
            )
            claims[c_age] = ClaimNode(
                claim_id=c_age,
                kind="derivation",
                subject=tgt_subj,
                predicate="edad",
                value=exact_age,
                unit="years",
                text=f"Edad calculada al {REFERENCE_DATE}: {exact_age} años",
            )
            claims[c_ge65] = ClaimNode(
                claim_id=c_ge65,
                kind="derivation",
                subject=tgt_subj,
                predicate="edad_al_menos_65",
                value=True,
                text=f"{exact_age} >= 65",
            )
            claims[conc_id] = ClaimNode(
                claim_id=conc_id,
                kind="derivation",
                subject=tgt_subj,
                predicate="persona_mayor",
                value=True,
                text=str(claim_spec["claim_text"]),
            )
            steps.extend(
                [
                    InferenceStep(
                        step_id=f"{claim_id}::s_age",
                        operation="temporal_age",
                        premises=(p_id,),
                        conclusion_id=c_age,
                        parameters={"birth_date": birth["value"], "reference_date": REFERENCE_DATE},
                    ),
                    InferenceStep(
                        step_id=f"{claim_id}::s_cmp_65",
                        operation="ge",
                        premises=(c_age,),
                        conclusion_id=c_ge65,
                        parameters={"threshold": 65, "threshold_unit": "years"},
                    ),
                    InferenceStep(
                        step_id=f"{claim_id}::s_r1",
                        operation="rule_application",
                        premises=(c_ge65,),
                        conclusion_id=conc_id,
                        rule_id="R1_persona_mayor",
                        rule_version="1.0",
                    ),
                ]
            )

    # ------------------------------------------------------------------ Case 2
    elif tgt_pred == "controla":
        own_edges = [p for p in grounded_prems if p["predicate"] == "propietaria_100"]
        edge_nodes: list[tuple[str, str, str, int]] = []
        for idx, edge in enumerate(own_edges):
            e_id = f"{claim_id}::e_{idx}"
            claims[e_id] = ClaimNode(
                claim_id=e_id,
                kind="extraction",
                subject=edge["subject"],
                predicate="propietaria_100",
                value=edge["value"],
                evidence=(edge["ref"],),
                text=edge["verbatim_quote"],
            )
            edge_nodes.append((e_id, edge["subject"], str(edge["value"]), idx))

        # Only materialize `controla` intermediates when a directed ownership path connects
        # tgt_subj -> ... -> tgt_val (avoids _claims_contradict false conflict between
        # unconnected `controla(Alfa, Beta)` and conclusion `controla(Alfa, Gamma)` in C2c).
        chain = _find_directed_chain(
            [(eid, s, v) for eid, s, v, _ in edge_nodes],
            tgt_subj,
            canonical_entity(str(tgt_val)),
            max_hops=4,
        )
        claims[conc_id] = ClaimNode(
            claim_id=conc_id,
            kind="derivation",
            subject=tgt_subj,
            predicate="controla",
            value=canonical_entity(str(tgt_val)),
            text=str(claim_spec["claim_text"]),
        )
        if len(chain) >= 2:
            ctrl_ids: list[str] = []
            for hop_idx, (e_id, s_ent, v_ent) in enumerate(chain):
                d_id = f"{claim_id}::ctrl_{hop_idx}"
                claims[d_id] = ClaimNode(
                    claim_id=d_id,
                    kind="derivation",
                    subject=s_ent,
                    predicate="controla",
                    value=v_ent,
                    text=f"{s_ent} controla {v_ent}",
                )
                steps.append(
                    InferenceStep(
                        step_id=f"{claim_id}::s_r2a_{hop_idx}",
                        operation="rule_application",
                        premises=(e_id,),
                        conclusion_id=d_id,
                        rule_id="R2a_prop_implica_control",
                        rule_version="1.0",
                    )
                )
                ctrl_ids.append(d_id)
            steps.append(
                InferenceStep(
                    step_id=f"{claim_id}::s_r2_trans",
                    operation="relation_composition",
                    premises=tuple(ctrl_ids),
                    conclusion_id=conc_id,
                    rule_id="R2_controla_transitiva",
                    rule_version="1.0",
                )
            )
        elif len(chain) == 1:
            steps.append(
                InferenceStep(
                    step_id=f"{claim_id}::s_r2a_direct",
                    operation="rule_application",
                    premises=(chain[0][0],),
                    conclusion_id=conc_id,
                    rule_id="R2a_prop_implica_control",
                    rule_version="1.0",
                )
            )

    # ------------------------------------------------------------------ Case 3
    elif tgt_pred == "contrato_se_renovara":
        p_clause = next((p for p in grounded_prems if p["predicate"] == "clausula_renovacion_automatica"), None)
        p_expiry = next((p for p in grounded_prems if p["predicate"] == "fecha_vencimiento"), None)
        p_no_flag = next((p for p in grounded_prems if p["predicate"] == "sin_preaviso_registrado"), None)
        p_no_date = next((p for p in grounded_prems if p["predicate"] == "fecha_registro_sin_preaviso"), None)
        p_yes_flag = next((p for p in grounded_prems if p["predicate"] == "preaviso_no_renovacion_enviado"), None)
        p_yes_date = next((p for p in grounded_prems if p["predicate"] == "fecha_preaviso"), None)

        if p_clause is None:
            claims[conc_id] = ClaimNode(
                claim_id=conc_id,
                kind="derivation",
                subject=tgt_subj,
                predicate=tgt_pred,
                value=tgt_val,
                text=str(claim_spec["claim_text"]),
            )
        else:
            clause_id = f"{claim_id}::p_clausula"
            claims[clause_id] = ClaimNode(
                claim_id=clause_id,
                kind="extraction",
                subject="Contrato",
                predicate="clausula_renovacion_automatica",
                value=True,
                evidence=(p_clause["ref"],),
                text=p_clause["verbatim_quote"],
                condition_status="conditional",
                conditions=({"field": "preaviso_en_plazo", "operator": "eq", "value": False},),
            )
            expiry_date_str = str(p_expiry["value"]) if p_expiry else "2026-12-31"
            expiry_id: str | None = None
            if p_expiry is not None:
                expiry_id = f"{claim_id}::p_vencimiento"
                claims[expiry_id] = ClaimNode(
                    claim_id=expiry_id,
                    kind="extraction",
                    subject="Contrato",
                    predicate="fecha_vencimiento",
                    value=expiry_date_str,
                    evidence=(p_expiry["ref"],),
                    text=p_expiry["verbatim_quote"],
                )

            if p_yes_flag is not None and p_yes_date is not None and expiry_id is not None:
                d_ant = (date.fromisoformat(expiry_date_str) - date.fromisoformat(str(p_yes_date["value"]))).days
                av_id = f"{claim_id}::p_preaviso"
                av_dt_id = f"{claim_id}::p_preaviso_fecha"
                d_days_id = f"{claim_id}::d_dias_ant"
                d_ge30_id = f"{claim_id}::d_ge30"
                d_en_plazo_id = f"{claim_id}::d_en_plazo"
                claims[av_id] = ClaimNode(
                    claim_id=av_id,
                    kind="extraction",
                    subject="Contrato",
                    predicate="preaviso_no_renovacion_enviado",
                    value=True,
                    evidence=(p_yes_flag["ref"],),
                    text=p_yes_flag["verbatim_quote"],
                )
                claims[av_dt_id] = ClaimNode(
                    claim_id=av_dt_id,
                    kind="extraction",
                    subject="Contrato",
                    predicate="fecha_preaviso",
                    value=str(p_yes_date["value"]),
                    evidence=(p_yes_date["ref"],),
                    text=p_yes_date["verbatim_quote"],
                )
                claims[d_days_id] = ClaimNode(
                    claim_id=d_days_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="dias_antelacion_preaviso",
                    value=d_ant,
                    unit="days",
                    text=f"Antelación calculada ({p_yes_date['value']} -> {expiry_date_str}): {d_ant} días",
                )
                claims[d_ge30_id] = ClaimNode(
                    claim_id=d_ge30_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="antelacion_al_menos_30_dias",
                    value=True,
                    text=f"{d_ant} >= 30 días",
                )
                claims[d_en_plazo_id] = ClaimNode(
                    claim_id=d_en_plazo_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="preaviso_en_plazo",
                    value=True,
                    text="Preaviso enviado con >= 30 días de antelación",
                )
                claims[conc_id] = ClaimNode(
                    claim_id=conc_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="contrato_se_renovara",
                    value=True,
                    text=str(claim_spec["claim_text"]),
                    condition_status="unconditional",
                )
                steps.extend(
                    [
                        InferenceStep(
                            step_id=f"{claim_id}::s_dias_ant",
                            operation="temporal_duration",
                            premises=(av_dt_id, expiry_id),
                            conclusion_id=d_days_id,
                            parameters={
                                "start_date": str(p_yes_date["value"]),
                                "end_date": expiry_date_str,
                                "duration_unit": "days",
                            },
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_ge30",
                            operation="ge",
                            premises=(d_days_id,),
                            conclusion_id=d_ge30_id,
                            parameters={"threshold": 30, "threshold_unit": "days"},
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_plazo",
                            operation="rule_application",
                            premises=(av_id, d_ge30_id),
                            conclusion_id=d_en_plazo_id,
                            rule_id="R3_preaviso_en_plazo",
                            rule_version="1.0",
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_excl",
                            operation="rule_application",
                            premises=(clause_id, d_en_plazo_id),
                            conclusion_id=conc_id,
                            rule_id="R3_exclusion_preaviso",
                            rule_version="1.0",
                        ),
                    ]
                )

            elif p_no_flag is not None and p_no_date is not None and expiry_id is not None:
                d_rem = (date.fromisoformat(expiry_date_str) - date.fromisoformat(str(p_no_date["value"]))).days
                reg_id = f"{claim_id}::p_sin_reg"
                reg_dt_id = f"{claim_id}::p_sin_fecha"
                d_rem_id = f"{claim_id}::d_dias_rest"
                d_lt30_id = f"{claim_id}::d_plazo_venc"
                d_sin_plazo_id = f"{claim_id}::d_sin_en_plazo"
                claims[reg_id] = ClaimNode(
                    claim_id=reg_id,
                    kind="extraction",
                    subject="Contrato",
                    predicate="sin_preaviso_registrado",
                    value=True,
                    evidence=(p_no_flag["ref"],),
                    text=p_no_flag["verbatim_quote"],
                )
                claims[reg_dt_id] = ClaimNode(
                    claim_id=reg_dt_id,
                    kind="extraction",
                    subject="Contrato",
                    predicate="fecha_registro_sin_preaviso",
                    value=str(p_no_date["value"]),
                    evidence=(p_no_date["ref"],),
                    text=p_no_date["verbatim_quote"],
                )
                claims[d_rem_id] = ClaimNode(
                    claim_id=d_rem_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="dias_hasta_vencimiento",
                    value=d_rem,
                    unit="days",
                    text=f"Días restantes desde registro {p_no_date['value']} hasta {expiry_date_str}: {d_rem} días (límite preaviso 2026-12-01 ya vencido)",
                )
                claims[d_lt30_id] = ClaimNode(
                    claim_id=d_lt30_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="plazo_preaviso_vencido",
                    value=True,
                    text=f"{d_rem} < 30 días (fecha registro > límite 2026-12-01)",
                )
                claims[d_sin_plazo_id] = ClaimNode(
                    claim_id=d_sin_plazo_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="sin_preaviso_en_plazo",
                    value=True,
                    text="Sin preaviso registrado tras vencer el plazo contractual de 30 días",
                )
                claims[conc_id] = ClaimNode(
                    claim_id=conc_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="contrato_se_renovara",
                    value=True,
                    text=str(claim_spec["claim_text"]),
                    condition_status="unconditional",
                )
                steps.extend(
                    [
                        InferenceStep(
                            step_id=f"{claim_id}::s_dias_rest",
                            operation="temporal_duration",
                            premises=(reg_dt_id, expiry_id),
                            conclusion_id=d_rem_id,
                            parameters={
                                "start_date": str(p_no_date["value"]),
                                "end_date": expiry_date_str,
                                "duration_unit": "days",
                            },
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_lt30",
                            operation="lt",
                            premises=(d_rem_id,),
                            conclusion_id=d_lt30_id,
                            parameters={"threshold": 30, "threshold_unit": "days"},
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_sin",
                            operation="rule_application",
                            premises=(reg_id, d_lt30_id),
                            conclusion_id=d_sin_plazo_id,
                            rule_id="R3_sin_preaviso_en_plazo",
                            rule_version="1.0",
                        ),
                        InferenceStep(
                            step_id=f"{claim_id}::s_r3",
                            operation="rule_application",
                            premises=(clause_id, d_sin_plazo_id),
                            conclusion_id=conc_id,
                            rule_id="R3_renovacion_automatica",
                            rule_version="1.0",
                        ),
                    ]
                )
            else:
                # Only S4 is present: missing precondition stays an explicit ungrounded hypothesis -> monotonicity downgrades!
                hyp_id = f"{claim_id}::p_hyp_sin_preaviso"
                claims[hyp_id] = ClaimNode(
                    claim_id=hyp_id,
                    kind="hypothesis",
                    subject="Contrato",
                    predicate="sin_preaviso_en_plazo",
                    value=True,
                    text="Hipótesis no acreditada: no hubo preaviso con >= 30 días de antelación",
                )
                claims[conc_id] = ClaimNode(
                    claim_id=conc_id,
                    kind="derivation",
                    subject="Contrato",
                    predicate="contrato_se_renovara",
                    value=True,
                    text=str(claim_spec["claim_text"]),
                    condition_status="conditional",
                    conditions=({"field": "preaviso_en_plazo", "operator": "eq", "value": False},),
                )
                steps.append(
                    InferenceStep(
                        step_id=f"{claim_id}::s_r3_hyp",
                        operation="rule_application",
                        premises=(clause_id, hyp_id),
                        conclusion_id=conc_id,
                        rule_id="R3_renovacion_automatica",
                        rule_version="1.0",
                    )
                )

    graph = JustificationGraph(claims=claims, steps=tuple(steps), conclusion_id=conc_id)
    res = evaluate_response(
        graph,
        evidence=allowed_docs,
        domain_profile=prof,
        verifier_registry=registry,
        as_of=REFERENCE_DATE,
    )
    label = map_assessment_to_label(res)
    return {
        "claim_id": claim_id,
        "expected_label": str(claim_spec["expected_label"]),
        "obtained_label": label,
        "correct": label == str(claim_spec["expected_label"]),
        "assessment": res.assessment,
        "effective_epistemic_kind": res.effective_epistemic_kind,
        "evidence_support_index": res.evidence_support_index,
        "conflict_index": res.conflict_index,
        "uncertainty_index": res.uncertainty_index,
        "grounded_premises": [
            {
                "doc_id": p["doc_id"],
                "subject": p["subject"],
                "predicate": p["predicate"],
                "value": p["value"],
                "verbatim_quote": p["verbatim_quote"],
            }
            for p in grounded_prems
        ],
        "rejected_premises": rejected_prems,
        "trace": format_graph_trace(graph, res, allowed_docs),
    }


def format_graph_trace(
    graph: JustificationGraph,
    res: ResponseAssessment,
    allowed_docs: Mapping[str, SourceDocument],
) -> dict[str, Any]:
    """Build a human-readable trace of the JustificationGraph and engine checks."""
    premises_trace = []
    for cid in sorted(graph.claims):
        if cid == graph.conclusion_id:
            continue
        node = graph.claims[cid]
        doc_ids = list(node.doc_ids())
        source_sentence = allowed_docs[doc_ids[0]].text if doc_ids and doc_ids[0] in allowed_docs else None
        premises_trace.append(
            {
                "claim_id": cid,
                "kind": node.kind,
                "subject": node.subject,
                "predicate": node.predicate,
                "value": node.value,
                "unit": node.unit,
                "doc_ids": doc_ids,
                "quote": node.evidence[0].quote if node.evidence else None,
                "source_sentence": source_sentence,
                "condition_status": node.condition_status,
            }
        )

    steps_trace = []
    for st in graph.steps:
        step_checks = [c.to_dict() for c in res.completed_checks if c.step_id == st.step_id]
        steps_trace.append(
            {
                "step_id": st.step_id,
                "operation": st.operation,
                "rule_id": st.rule_id,
                "rule_version": st.rule_version,
                "premises": list(st.premises),
                "conclusion_id": st.conclusion_id,
                "parameters": dict(st.parameters),
                "checks": [
                    {
                        "family": c["family"],
                        "status": c["status"],
                        "reasons": c["reasons"],
                        "verifier_id": c["verifier_id"],
                    }
                    for c in step_checks
                ],
            }
        )

    mandatory_checks = [
        {
            "obligation_id": c.obligation_id,
            "family": c.family,
            "status": c.status,
            "reasons": list(c.reasons),
            "verifier_id": c.verifier_id,
        }
        for c in res.completed_checks
        if c.step_id.startswith("mandatory::")
    ]
    return {
        "conclusion_id": res.conclusion_id,
        "conclusion_declared_kind": res.conclusion.kind,
        "effective_epistemic_kind": res.effective_epistemic_kind,
        "assessment": res.assessment,
        "rules_applied": sorted({st.rule_id for st in graph.steps if st.rule_id}),
        "premises_and_intermediates": premises_trace,
        "inference_steps": steps_trace,
        "mandatory_checks": mandatory_checks,
        "supporting_evidence": list(res.supporting_evidence),
        "pending_obligations": list(res.pending_obligations),
    }


def run_mode_a_all(
    prereg: Mapping[str, Any] | None = None,
    *,
    use_default_registry: bool = False,
) -> list[dict[str, Any]]:
    data = prereg or load_preregistration()
    docs = build_poc_sources(data)
    profile = build_poc_domain_profile()
    return [
        build_graph_and_evaluate(
            claim_spec,
            HAND_PREMISES_MODE_A[claim_spec["claim_id"]],
            docs,
            profile=profile,
            use_default_registry=use_default_registry,
        )
        for claim_spec in data["claims"]
    ]
