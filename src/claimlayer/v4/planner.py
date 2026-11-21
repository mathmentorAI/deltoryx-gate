"""End-to-end LLM Derivation Planner, Controlled Recalculation & Claim Verification API (v4).

Nemotron proposes structured premises and an execution plan over an explicit
OperationCatalog. Deltoryx validates the schema, checks premise-to-evidence
correspondence, recomputes every step deterministically using controlled
operations, compares the recomputed conclusion against the original claim, and
separates epistemic verdict, operational routing, and certainty/calibration.
"""
from __future__ import annotations

import re

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError

from ..semantic.adapters import OpenAICompatibleModel
from ..semantic.models import SourceDocument, canonical, digest
from .authorization import authorize
from .domain_profile import DomainProfile
from .evaluation import ResponseAssessment, evaluate_response
from .justification import (
    ClaimNode,
    DEFAULT_OPERATION_FAMILIES,
    InferenceStep,
    JustificationGraph,
    VERIFIER_FAMILIES,
    VerificationObligation,
)
from .operations import (
    DEFAULT_OPERATION_CATALOG,
    OperationCatalog,
    OperationOutput,
    normalize_weekday,
)
from .premise_grounding import (
    PremiseGroundingResult,
    _extract_explicit_anchor_phrase,
    diagnose_alternate_anchors,
    verify_claim_target_correspondence,
    verify_premise_grounding,
    verify_step_anchor_correspondence,
)
from .verifiers import DEFAULT_VERIFIER_REGISTRY, VerifierContext, VerifierRegistry, VerifierResult
from .vocabulary import STANDARD_UNITS, UNIT_PROFILES, UnitRegistry, UnmappedProposal, label_key, normalize_date
from .reconstruction import (
    DEFAULT_RELIABILITY_POLICY,
    EXPRESSION_PRIMITIVES,
    ProposedOperation,
    ReliabilityPolicy,
    compute_reliability,
    parse_proposed_operations,
    proposed_operation_specs,
    step_grade,
)

DERIVATION_PLANNER_PROMPT_VERSION_V2 = "v4-derivation-planner-2"

DERIVATION_PLANNER_PROMPT_V2 = """Eres el planificador estructurado de derivaciones de Deltoryx Gate.
Recibes `sources` (documentos con `document_id`, `revision` y `text`), una `claim` (afirmación a contrastar), un catálogo explícito `operations_catalog`, opcionalmente `declared_rules` y `units`, y opcionalmente `repair_feedback` de un intento previo fallido.

Tu tarea es proponer premisas verificables extraídas SOLO de `sources` y un plan acíclico de pasos sobre `operations_catalog` para recalcular el hecho relevante y compararlo con `claim`. Un motor determinista comprobará todas las citas, ejecutará las operaciones y decidirá si `claim` se sostiene, queda refutada o carece de evidencia.

Reglas estrictas:
1. `claim` es EXCLUSIVAMENTE el objeto a contrastar. NUNCA uses `claim` como premisa, cita o evidencia, ni tomes datos del reloj del sistema. Extrae en `claim_target` lo que afirma literalmente `claim` sin cambiar su valor, unidad ni polaridad.
2. Cada premisa en `premises` debe citar un `document_id` y `revision` existentes en `sources` y copiar en `quote` un fragmento LITERAL exacto (subcadena continua) del `text` de ese documento.
3. Mantén la correspondencia por cláusula elemental: la `quote` debe respaldar exactamente `entity`, `event`, `value`, `unit`, `direction` y `anchor_reference`. No mezcles valores de una cláusula con entidades o eventos de otra cláusula. Cuando la cláusula nombre un sujeto o evento, usa esas mismas palabras literales en `entity`/`event` (no etiquetas abstractas). Para días de la semana usa `unit: null`.
4. En desplazamientos relativos: expresiones hacia atrás ("antes", "atrás", "previo", "hace", "disminuye", "before", "earlier", "ago", "minus") requieren signo NEGATIVO en `value` (ej. -2) y `direction="before"`; expresiones hacia adelante ("después", "posterior", "luego", "aumenta", "after", "later", "plus") requieren signo POSITIVO y `direction="after"`. Si el texto indica respecto a qué evento/fecha es el desplazamiento, ponlo en `anchor_reference`.
5. Usa SOLO operaciones declaradas en `operations_catalog` (y `rule_id` declarados en `declared_rules` cuando uses `rule_application`). Respeta estrictamente el orden de operandos del catálogo:
   - `weekday_shift`: `premises` = `[id_ancla_dia_semana, id_desplazamiento]`
   - `date_offset`: `premises` = `[id_ancla_fecha_iso, id_desplazamiento]`
   - `sub`: `premises` = `[id_minuendo, id_sustraendo]`
   - `div`: `premises` = `[id_numerador, id_denominador]`
6. Si falta un dato indispensable en `sources` (por ejemplo, falta el ancla temporal o falta un operando necesario), NO lo inventes ni uses el reloj del sistema: deja `steps` vacío, `derived_conclusion_id` en `null` y detalla la carencia en `missing_premises`.
7. Si necesitas una regla u operación ausente de `operations_catalog` y `declared_rules`, regístrala en `suggested_rules` y no inventes pasos no admitidos.
8. NUNCA emitas puntuaciones de confianza, probabilidades, dictámenes (`supported`/`refuted`) ni código ejecutable.

Devuelve ÚNICAMENTE un objeto JSON válido con este esquema exacto:
{
  "claim_target": {
    "subject": "string",
    "predicate": "string",
    "value": "JSON",
    "unit": "string|null",
    "polarity": 1
  },
  "premises": [
    {
      "claim_id": "p1",
      "document_id": "string",
      "revision": "string",
      "quote": "subcadena literal exacta de sources",
      "entity": "string",
      "event": "string",
      "predicate": "string",
      "value": "JSON",
      "unit": "string|null",
      "direction": "before|after|neutral|null",
      "anchor_reference": "string|null",
      "polarity": 1
    }
  ],
  "steps": [
    {
      "step_id": "s1",
      "operation": "nombre_operacion_catalogo",
      "premises": ["p1", "p2"],
      "conclusion_id": "c1",
      "rule_id": "string|null",
      "rule_version": "string|null",
      "parameters": {},
      "assumptions": [],
      "proposed_value": "JSON|null",
      "proposed_unit": "string|null"
    }
  ],
  "derived_conclusion_id": "c1|null",
  "missing_premises": [
    {"role": "string", "description": "string"}
  ],
  "assumptions": ["string"],
  "suggested_rules": [
    {"rule_id": "string", "description": "string", "premises": ["string"], "conclusion": "string"}
  ]
}"""

# v3: the planner may propose operations (closed expression language or model-asserted),
# and is told that interpreted premise values lower reliability.
DERIVATION_PLANNER_PROMPT_VERSION = "v4-derivation-planner-3"

DERIVATION_PLANNER_PROMPT = """Eres el planificador estructurado de derivaciones de Deltoryx Gate.
Recibes `sources` (documentos con `document_id`, `revision` y `text`), una `claim` (afirmación a contrastar), un catálogo explícito `operations_catalog`, opcionalmente `declared_rules` y `units`, y opcionalmente `repair_feedback` de un intento previo fallido.

Tu tarea es proponer premisas verificables extraídas SOLO de `sources` y un plan acíclico de pasos sobre `operations_catalog` para recalcular el hecho relevante y compararlo con `claim`. Un motor determinista comprobará todas las citas, ejecutará las operaciones y decidirá si `claim` se sostiene, queda refutada o carece de evidencia.

Reglas estrictas:
1. `claim` es EXCLUSIVAMENTE el objeto a contrastar. NUNCA uses `claim` como premisa, cita o evidencia, ni tomes datos del reloj del sistema. Extrae en `claim_target` lo que afirma literalmente `claim` sin cambiar su valor, unidad ni polaridad.
2. Cada premisa en `premises` debe citar un `document_id` y `revision` existentes en `sources` y copiar en `quote` un fragmento LITERAL exacto (subcadena continua) del `text` de ese documento.
3. Mantén la correspondencia por cláusula elemental: la `quote` debe respaldar exactamente `entity`, `event`, `value`, `unit`, `direction` y `anchor_reference`. No mezcles valores de una cláusula con entidades o eventos de otra cláusula. Cuando la cláusula nombre un sujeto o evento, usa esas mismas palabras literales en `entity`/`event` (no etiquetas abstractas). Para días de la semana usa `unit: null`.
4. En desplazamientos relativos: expresiones hacia atrás ("antes", "atrás", "previo", "hace", "disminuye", "before", "earlier", "ago", "minus") requieren signo NEGATIVO en `value` (ej. -2) y `direction="before"`; expresiones hacia adelante ("después", "posterior", "luego", "aumenta", "after", "later", "plus") requieren signo POSITIVO y `direction="after"`. Si el texto indica respecto a qué evento/fecha es el desplazamiento, ponlo en `anchor_reference`.
5. Usa SOLO operaciones declaradas en `operations_catalog` (y `rule_id` declarados en `declared_rules` cuando uses `rule_application`). Respeta estrictamente el orden de operandos del catálogo:
   - `weekday_shift`: `premises` = `[id_ancla_dia_semana, id_desplazamiento]`
   - `date_offset`: `premises` = `[id_ancla_fecha_iso, id_desplazamiento]`
   - `sub`: `premises` = `[id_minuendo, id_sustraendo]`
   - `div`: `premises` = `[id_numerador, id_denominador]`
6. Si falta un dato indispensable en `sources` (por ejemplo, falta el ancla temporal o falta un operando necesario), NO lo inventes ni uses el reloj del sistema: deja `steps` vacío, `derived_conclusion_id` en `null` y detalla la carencia en `missing_premises`.
7. Si necesitas una operación ausente de `operations_catalog` y `declared_rules`, PROPÓNLA en `proposed_operations` y úsala en `steps` con su `name`. Siempre que puedas, defínela con `expression`, un árbol JSON sobre las primitivas de `expression_primitives`: nodos `{"input": "<nombre de entrada>"}`, `{"const": <escalar>}` u `{"op": "<primitiva>", "args": [...]}`. Las entradas se asignan en orden a los `premises` del paso. Deltoryx interpretará y re-ejecutará la expresión (fiabilidad C). Si no puede expresarse así, deja `expression` en `null` y pon el resultado en `proposed_value` del paso: quedará registrado como afirmación tuya, de fiabilidad baja (D). Nunca escribas código de programación.
8b. Prefiere premisas cuyo `value` aparezca literalmente en la `quote` (números, nombres, rutas, identificadores). Un valor que no esté en la cita (por ejemplo `true` como interpretación de una línea) se registra como interpretación tuya y rebaja la fiabilidad al grado más bajo (E).
8. NUNCA emitas puntuaciones de confianza, probabilidades, dictámenes (`supported`/`refuted`) ni código ejecutable.

Devuelve ÚNICAMENTE un objeto JSON válido con este esquema exacto:
{
  "claim_target": {
    "subject": "string",
    "predicate": "string",
    "value": "JSON",
    "unit": "string|null",
    "polarity": 1
  },
  "premises": [
    {
      "claim_id": "p1",
      "document_id": "string",
      "revision": "string",
      "quote": "subcadena literal exacta de sources",
      "entity": "string",
      "event": "string",
      "predicate": "string",
      "value": "JSON",
      "unit": "string|null",
      "direction": "before|after|neutral|null",
      "anchor_reference": "string|null",
      "polarity": 1
    }
  ],
  "steps": [
    {
      "step_id": "s1",
      "operation": "nombre_operacion_catalogo",
      "premises": ["p1", "p2"],
      "conclusion_id": "c1",
      "rule_id": "string|null",
      "rule_version": "string|null",
      "parameters": {},
      "assumptions": [],
      "proposed_value": "JSON|null",
      "proposed_unit": "string|null"
    }
  ],
  "derived_conclusion_id": "c1|null",
  "missing_premises": [
    {"role": "string", "description": "string"}
  ],
  "assumptions": ["string"],
  "suggested_rules": [
    {"rule_id": "string", "description": "string", "premises": ["string"], "conclusion": "string"}
  ],
  "proposed_operations": [
    {"name": "identificador", "description": "string", "inputs": ["a", "b"], "expression": "árbol JSON o null", "output_type": "string", "output_unit": "string|inherit|null"}
  ]
}"""

_UNTRUSTED_PLAN_KEYS: frozenset[str] = frozenset({
    "confidence",
    "trust_score",
    "authority_rank",
    "authority_override",
    "bypass_verification",
    "self_certified",
    "verified",
    "verdict",
    "epistemic_verdict",
    "calibrated_probability",
    "calibrated_correctness_probability",
})


@dataclass(frozen=True)
class PlannerCallResult:
    """Raw output and transport telemetry of one planner invocation."""

    ok: bool
    proposal: dict[str, Any] | None = None
    error_kind: str | None = None  # 'provider_error' | 'malformed_proposal' | None
    error_message: str | None = None
    latency_ms: float = 0.0
    usage: dict[str, int] = field(default_factory=dict)
    raw_model: str | None = None


class DerivationPlanner(Protocol):
    @property
    def identity(self) -> Mapping[str, Any]: ...

    def plan(
        self,
        *,
        sources: Sequence[SourceDocument],
        claim: str,
        catalog: OperationCatalog,
        domain_profile: DomainProfile,
        repair_feedback: Sequence[Mapping[str, Any]] | None = None,
    ) -> PlannerCallResult: ...


@dataclass(frozen=True)
class ClaimDerivationPlanner:
    """Planner adapter backed by claimlayer.semantic.adapters.OpenAICompatibleModel."""

    model: OpenAICompatibleModel
    prompt: str = DERIVATION_PLANNER_PROMPT
    prompt_version: str = DERIVATION_PLANNER_PROMPT_VERSION

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            **self.model.identity,
            "prompt_version": self.prompt_version,
            "prompt_digest": digest(self.prompt),
        }

    def plan(
        self,
        *,
        sources: Sequence[SourceDocument],
        claim: str,
        catalog: OperationCatalog,
        domain_profile: DomainProfile,
        repair_feedback: Sequence[Mapping[str, Any]] | None = None,
    ) -> PlannerCallResult:
        registry = domain_profile.effective_unit_registry()
        payload: dict[str, Any] = {
            "sources": [
                {
                    "document_id": doc.document_id,
                    "revision": doc.revision,
                    "text": doc.text,
                }
                for doc in sources
            ],
            "claim": claim,
            "operations_catalog": catalog.describe_for_planner(),
            "declared_rules": [
                domain_profile.declared_rules[k].describe()
                for k in sorted(domain_profile.declared_rules)
            ],
            "units": [u.name for u in sorted(registry._units.values(), key=lambda x: x.name)],
        }
        if "proposed_operations" in self.prompt:
            payload["expression_primitives"] = {k: v["doc"] for k, v in EXPRESSION_PRIMITIVES.items()}
        if repair_feedback:
            payload["repair_feedback"] = list(repair_feedback)

        t0 = time.monotonic()
        try:
            res = self.model.complete(self.prompt, payload)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            dt_ms = (time.monotonic() - t0) * 1000.0
            diag = getattr(exc, "diagnostics", None)
            detail = f"{type(exc).__name__}"
            if isinstance(exc, HTTPError):
                detail += f":{exc.code}"
            return PlannerCallResult(
                ok=False,
                error_kind="provider_error",
                error_message=f"{detail} ({diag})" if diag else detail,
                latency_ms=round(dt_ms, 3),
            )
        except Exception as exc:
            dt_ms = (time.monotonic() - t0) * 1000.0
            diag = getattr(exc, "diagnostics", None) or {}
            prov_resp = diag.get("provider_response") if isinstance(diag, Mapping) else None
            usage = _extract_usage(prov_resp.get("usage") if isinstance(prov_resp, Mapping) else None)
            # Distinguish connection/transport stage vs malformed response contract
            attempts = diag.get("attempts", ()) if isinstance(diag, Mapping) else ()
            is_contract_err = any(isinstance(a, Mapping) and a.get("stage") == "response_contract" for a in attempts)
            err_kind = "malformed_proposal" if (is_contract_err or isinstance(exc, (ValueError, KeyError, TypeError, IndexError))) else "provider_error"
            return PlannerCallResult(
                ok=False,
                error_kind=err_kind,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=round(dt_ms, 3),
                usage=usage,
            )

        dt_ms = (time.monotonic() - t0) * 1000.0
        output = res.get("output")
        usage = _extract_usage(res.get("usage"))
        if not isinstance(output, Mapping):
            return PlannerCallResult(
                ok=False,
                error_kind="malformed_proposal",
                error_message="Planner model did not return a JSON object.",
                latency_ms=round(dt_ms, 3),
                usage=usage,
                raw_model=res.get("response_model"),
            )
        return PlannerCallResult(
            ok=True,
            proposal=dict(output),
            latency_ms=round(dt_ms, 3),
            usage=usage,
            raw_model=res.get("response_model"),
        )


def _extract_usage(raw_usage: Any) -> dict[str, int]:
    if not isinstance(raw_usage, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = int(raw_usage.get("prompt_tokens") or 0)
    ct = int(raw_usage.get("completion_tokens") or 0)
    tt = int(raw_usage.get("total_tokens") or (pt + ct))
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}


@dataclass(frozen=True)
class OperationalRouting:
    """Operational routing kept separate from epistemic truth judgment."""

    status: str  # 'PASS' | 'RETRY' | 'ESCALATE'
    allowed: bool
    requires_human: bool
    reasons: tuple[str, ...] = ()
    authorization_verdict: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allowed": self.allowed,
            "requires_human": self.requires_human,
            "reasons": list(self.reasons),
            "authorization_verdict": self.authorization_verdict,
        }


@dataclass(frozen=True)
class CertaintyReport:
    """Explicit separation of formal calculation validity, extraction fidelity, source support, and calibration."""

    derivation_validity: dict[str, Any]
    extraction_fidelity: dict[str, Any]
    source_support: dict[str, Any]
    calibrated_correctness_probability: float | None
    calibration_status: str  # 'calibrated' | 'uncalibrated'
    guarantee_explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "derivation_validity": dict(self.derivation_validity),
            "extraction_fidelity": dict(self.extraction_fidelity),
            "source_support": dict(self.source_support),
            "calibrated_correctness_probability": self.calibrated_correctness_probability,
            "calibration_status": self.calibration_status,
            "guarantee_explanation": self.guarantee_explanation,
        }


@dataclass(frozen=True)
class PlannerTelemetry:
    """Call, token, retry, and latency accounting for one claim verification."""

    model_identity: dict[str, Any]
    prompt_version: str
    prompt_digest: str
    catalog_id: str
    catalog_version: str
    catalog_fingerprint: str
    calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    planning_latency_ms: float = 0.0
    verifier_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    attempt_traces: tuple[dict[str, Any], ...] = ()
    unit_profile: str | None = None
    unit_registry_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_identity": dict(self.model_identity),
            "prompt_version": self.prompt_version,
            "prompt_digest": self.prompt_digest,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "catalog_fingerprint": self.catalog_fingerprint,
            "calls": self.calls,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "planning_latency_ms": round(self.planning_latency_ms, 3),
            "verifier_latency_ms": round(self.verifier_latency_ms, 3),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "attempt_traces": list(self.attempt_traces),
            "unit_profile": self.unit_profile,
            "unit_registry_fingerprint": self.unit_registry_fingerprint,
        }


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Complete, reproducible end-to-end verification artifact for a natural-language claim."""

    original_claim: str
    claim_target: dict[str, Any] | None
    computed_conclusion: dict[str, Any] | None
    epistemic_verdict: str  # 'derived' | 'refuted' | 'insufficient_evidence' | 'unresolved'
    outcome_category: str
    operational_routing: OperationalRouting
    premises: tuple[dict[str, Any], ...]
    rejected_premises: tuple[dict[str, Any], ...]
    verified_steps: tuple[dict[str, Any], ...]
    assumptions: tuple[str, ...]
    missing_premises: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    defects: tuple[dict[str, Any], ...]
    anchor_diagnostics: tuple[dict[str, Any], ...]
    certainty: CertaintyReport
    explanation: str
    raw_proposal: dict[str, Any] | None
    graph_admitted: bool
    graph: JustificationGraph | None
    assessment: ResponseAssessment | None
    telemetry: PlannerTelemetry
    reliability: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_claim": self.original_claim,
            "claim_target": self.claim_target,
            "computed_conclusion": self.computed_conclusion,
            "epistemic_verdict": self.epistemic_verdict,
            "outcome_category": self.outcome_category,
            "operational_routing": self.operational_routing.to_dict(),
            "premises": list(self.premises),
            "rejected_premises": list(self.rejected_premises),
            "verified_steps": list(self.verified_steps),
            "assumptions": list(self.assumptions),
            "missing_premises": list(self.missing_premises),
            "limitations": list(self.limitations),
            "defects": list(self.defects),
            "anchor_diagnostics": list(self.anchor_diagnostics),
            "certainty": self.certainty.to_dict(),
            "explanation": self.explanation,
            "raw_proposal": self.raw_proposal,
            "graph_admitted": self.graph_admitted,
            "graph": self.graph.to_dict() if self.graph is not None else None,
            "assessment": self.assessment.to_summary_dict() if self.assessment is not None else None,
            "telemetry": self.telemetry.to_dict(),
            "reliability": self.reliability,
        }


# ---------------------------------------------------------------------------
# Catalog-aware Verifier Bridge (supports extended OperationCatalog seamlessly)
# ---------------------------------------------------------------------------

class _CatalogFallbackVerifier:
    """Delegates to the base v4 family verifier and falls back to OperationCatalog for extended ops."""

    def __init__(self, base_verifier: Any, catalog: OperationCatalog) -> None:
        self._base = base_verifier
        self._catalog = catalog
        self.family = base_verifier.family
        self.verifier_id = base_verifier.verifier_id
        self.version = base_verifier.version
        self.guarantee_kind = base_verifier.guarantee_kind

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        spec = self._catalog.get(context.step.operation)
        # Planner-proposed operations are re-executed by Deltoryx's closed interpreter (grade C)
        # or replay the model-asserted value (grade D); they are never routed to rule verifiers.
        is_proposed = (
            spec is not None
            and spec.version.startswith("proposed-")
            and spec.family == self.family
            and obligation.origin != "engine_mandatory"
        )
        if not is_proposed:
            res = self._base.verify(obligation, context)
            if res.status != "not_applicable" or obligation.origin == "engine_mandatory":
                return res
            if spec is None or spec.family != self.family:
                return res
        out = self._catalog.execute(
            context.step.operation,
            context.premises,
            {**context.step.parameters, **obligation.parameters},
            context.domain_profile,
        )
        if not out.ok:
            status = "unknown" if out.error_kind == "missing_input" else "refuted"
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status=status,
                guarantee_kind="deterministic_formal",
                evidence=dict(out.details),
                reasons=(str(out.error_code or "CATALOG_OPERATION_ERROR"),),
                verifier_id=f"catalog:{spec.name}",
                verifier_version=spec.version,
            )
        reg = context.domain_profile.effective_unit_registry()
        matched = _values_match(
            out.value,
            out.unit,
            context.conclusion.value,
            context.conclusion.unit,
            reg,
        )
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified" if matched else "refuted",
            guarantee_kind="deterministic_formal",
            evidence={"expected": out.value, "expected_unit": out.unit, "proposed": context.conclusion.value},
            assumptions=tuple(out.assumptions),
            reasons=("CATALOG_OPERATION_VERIFIED" if matched else "CATALOG_OPERATION_MISMATCH",),
            verifier_id=f"catalog:{spec.name}",
            verifier_version=spec.version,
        )


def _effective_verifier_registry(
    base_registry: VerifierRegistry | None,
    catalog: OperationCatalog,
) -> VerifierRegistry:
    base = base_registry or DEFAULT_VERIFIER_REGISTRY
    wrapped: list[Any] = []
    for fam in base.families():
        v = base.get(fam)
        if fam == "open_semantics":
            wrapped.append(v)
        else:
            wrapped.append(_CatalogFallbackVerifier(v, catalog))
    return VerifierRegistry(wrapped)


# ---------------------------------------------------------------------------
# Normalization & Comparison Helpers
# ---------------------------------------------------------------------------

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


def _strip_untrusted_keys(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {
            str(k): _strip_untrusted_keys(v)
            for k, v in obj.items()
            if str(k) not in _UNTRUSTED_PLAN_KEYS
        }
    if isinstance(obj, list):
        return [_strip_untrusted_keys(v) for v in obj]
    return obj


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


def _build_certainty_report(
    *,
    epistemic_verdict: str,
    outcome_category: str,
    grounded_count: int,
    rejected_count: int,
    checks_applied: Sequence[str],
    docs_map: Mapping[str, SourceDocument],
    profile: DomainProfile,
    assessment: ResponseAssessment | None,
) -> CertaintyReport:
    if epistemic_verdict in ("derived", "refuted") and assessment is not None and assessment.assessment == "supported":
        derivation_validity = {
            "status": "exact_under_explicit_premises",
            "guarantee_kind": "deterministic_formal",
            "explanation": (
                "Deducción determinista exacta bajo las premisas explícitas extraídas y las operaciones controladas."
            ),
        }
    else:
        derivation_validity = {
            "status": "not_established",
            "guarantee_kind": "none",
            "explanation": f"No se completó una derivación válida ({outcome_category}).",
        }

    extraction_fidelity = {
        "status": "verified_grounding" if (grounded_count > 0 and rejected_count == 0) else (
            "partial_or_rejected" if rejected_count > 0 else "no_premises_extracted"
        ),
        "grounded_premises": grounded_count,
        "rejected_premises": rejected_count,
        "checks_applied": list(dict.fromkeys(checks_applied)),
        "note": (
            "Fidelidad comprobada contra citas literales y cláusulas elementales de las fuentes; "
            "no sustituye a una corroboración externa ni mide probabilidad calibrada."
        ),
    }

    attributions = {did: doc.attribution for did, doc in sorted(docs_map.items())}
    if assessment is not None:
        source_support: dict[str, Any] = {
            "evidence_support_index": assessment.evidence_support_index,
            "conflict_index": assessment.conflict_index,
            "uncertainty_index": assessment.uncertainty_index,
            "verification_coverage": assessment.verification_coverage,
            "is_policy_index": True,
            "source_attributions": attributions,
        }
        calibrated_prob = (
            profile.calibration.calibrate_correctness(assessment.evidence_support_index)
            if epistemic_verdict == "derived"
            else None
        )
    else:
        source_support = {
            "evidence_support_index": 0.0,
            "conflict_index": 0.0,
            "uncertainty_index": 1.0,
            "verification_coverage": 0.0,
            "is_policy_index": True,
            "source_attributions": attributions,
        }
        calibrated_prob = None

    calibration_status = "calibrated" if calibrated_prob is not None else "uncalibrated"
    if calibrated_prob is not None:
        guarantee_explanation = (
            f"Deducción determinista exacta bajo premisas explícitas; probabilidad empírica calibrada="
            f"{calibrated_prob} (dataset={profile.calibration.calibration_dataset_id})."
        )
    elif epistemic_verdict in ("derived", "refuted"):
        guarantee_explanation = (
            "Deducción aritmética/calendárica exacta bajo las premisas explícitas extraídas; "
            "calibrated_correctness_probability=None porque no existe calibración empírica aplicable "
            "(el índice de soporte es una métrica de política, no una probabilidad)."
        )
    else:
        guarantee_explanation = (
            f"Sin derivación formal completa ({outcome_category}); calibrated_correctness_probability=None."
        )

    return CertaintyReport(
        derivation_validity=derivation_validity,
        extraction_fidelity=extraction_fidelity,
        source_support=source_support,
        calibrated_correctness_probability=calibrated_prob,
        calibration_status=calibration_status,
        guarantee_explanation=guarantee_explanation,
    )


def _base_telemetry(
    *,
    planner_identity: Mapping[str, Any],
    prompt_version: str,
    prompt_digest: str,
    catalog: OperationCatalog,
    verifier_ms: float,
    external_telemetry: Mapping[str, Any] | None = None,
    unit_profile: str | None = None,
    unit_registry_fingerprint: str | None = None,
) -> PlannerTelemetry:
    ext = dict(external_telemetry or {})
    plan_ms = float(ext.get("planning_latency_ms") or 0.0)
    total_ms = float(ext.get("total_latency_ms") or (plan_ms + verifier_ms))
    return PlannerTelemetry(
        model_identity=dict(ext.get("model_identity") or planner_identity),
        prompt_version=str(ext.get("prompt_version") or prompt_version),
        prompt_digest=str(ext.get("prompt_digest") or prompt_digest),
        catalog_id=catalog.catalog_id,
        catalog_version=catalog.version,
        catalog_fingerprint=catalog.fingerprint(),
        calls=int(ext.get("calls") or 0),
        retries=int(ext.get("retries") or 0),
        prompt_tokens=int(ext.get("prompt_tokens") or 0),
        completion_tokens=int(ext.get("completion_tokens") or 0),
        total_tokens=int(ext.get("total_tokens") or 0),
        planning_latency_ms=plan_ms,
        verifier_latency_ms=verifier_ms,
        total_latency_ms=total_ms,
        attempt_traces=tuple(ext.get("attempt_traces") or ()),
        unit_profile=unit_profile,
        unit_registry_fingerprint=unit_registry_fingerprint,
    )



# ---------------------------------------------------------------------------
# Explicit unit-profile selection
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Deterministic Plan Verification & Recalculation Engine
# ---------------------------------------------------------------------------

def verify_plan(
    sources: (
        Mapping[str, SourceDocument]
        | Mapping[str, str]
        | Mapping[str, Mapping[str, Any]]
        | Sequence[SourceDocument | Mapping[str, Any] | str]
    ),
    claim: str,
    proposal: Mapping[str, Any],
    *,
    catalog: OperationCatalog = DEFAULT_OPERATION_CATALOG,
    domain_profile: DomainProfile | None = None,
    verifier_registry: VerifierRegistry | None = None,
    as_of: str | None = None,
    telemetry: Mapping[str, Any] | None = None,
    unit_profile: str = "auto",
    reliability_policy: ReliabilityPolicy | None = None,
) -> ClaimVerificationResult:
    """Deterministically validate a structured proposal, ground premises, execute operations, and contrast `claim`."""
    t_ver0 = time.monotonic()
    docs_map, _ = normalize_sources(sources)
    profile, unit_profile_label, unit_registry_fp = _resolve_derivation_profile(
        domain_profile, unit_profile, claim, docs_map
    )
    registry = profile.effective_unit_registry()
    eff_verifiers = _effective_verifier_registry(verifier_registry, catalog)
    rel_policy = reliability_policy or DEFAULT_RELIABILITY_POLICY
    proposed_ops: dict[str, ProposedOperation] = {}

    def _finish(
        *,
        claim_target: dict[str, Any] | None,
        computed_conclusion: dict[str, Any] | None,
        epistemic_verdict: str,
        outcome_category: str,
        routing: OperationalRouting,
        premises_out: Sequence[dict[str, Any]] = (),
        rejected_out: Sequence[dict[str, Any]] = (),
        steps_out: Sequence[dict[str, Any]] = (),
        assumptions_out: Sequence[str] = (),
        missing_out: Sequence[dict[str, Any]] = (),
        limitations_out: Sequence[str] = (),
        defects_out: Sequence[dict[str, Any]] = (),
        anchor_diags: Sequence[dict[str, Any]] = (),
        checks_applied: Sequence[str] = (),
        explanation: str,
        raw_clean: dict[str, Any] | None,
        graph_admitted: bool,
        graph: JustificationGraph | None,
        assessment: ResponseAssessment | None,
        reliability: dict[str, Any] | None = None,
    ) -> ClaimVerificationResult:
        ver_ms = round((time.monotonic() - t_ver0) * 1000.0, 3)
        certainty = _build_certainty_report(
            epistemic_verdict=epistemic_verdict,
            outcome_category=outcome_category,
            grounded_count=len(premises_out),
            rejected_count=len(rejected_out),
            checks_applied=checks_applied,
            docs_map=docs_map,
            profile=profile,
            assessment=assessment,
        )
        tel = _base_telemetry(
            planner_identity={"provider": "offline_plan"},
            prompt_version=DERIVATION_PLANNER_PROMPT_VERSION,
            prompt_digest=digest(DERIVATION_PLANNER_PROMPT),
            catalog=catalog,
            verifier_ms=ver_ms,
            external_telemetry=telemetry,
            unit_profile=unit_profile_label,
            unit_registry_fingerprint=unit_registry_fp,
        )
        return ClaimVerificationResult(
            original_claim=claim,
            claim_target=claim_target,
            computed_conclusion=computed_conclusion,
            epistemic_verdict=epistemic_verdict,
            outcome_category=outcome_category,
            operational_routing=routing,
            premises=tuple(premises_out),
            rejected_premises=tuple(rejected_out),
            verified_steps=tuple(steps_out),
            assumptions=tuple(dict.fromkeys(assumptions_out)),
            missing_premises=tuple(missing_out),
            limitations=tuple(dict.fromkeys(limitations_out)),
            defects=tuple(defects_out),
            anchor_diagnostics=tuple(anchor_diags),
            certainty=certainty,
            explanation=explanation,
            raw_proposal=raw_clean,
            graph_admitted=graph_admitted,
            graph=graph,
            assessment=assessment,
            telemetry=tel,
            reliability=reliability,
        )

    if not isinstance(proposal, Mapping):
        return _finish(
            claim_target=None,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False, reasons=("MALFORMED_PROPOSAL_NOT_MAPPING",)),
            defects_out=[{"code": "MALFORMED_PROPOSAL", "message": "Proposal must be a JSON object."}],
            explanation="La propuesta del planificador no es un objeto JSON válido.",
            raw_clean=None,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    raw_clean = dict(_strip_untrusted_keys(dict(proposal)))
    raw_target = raw_clean.get("claim_target")
    raw_premises = raw_clean.get("premises")
    raw_steps = raw_clean.get("steps")
    derived_conclusion_id = raw_clean.get("derived_conclusion_id")
    raw_missing = raw_clean.get("missing_premises") or []
    raw_assumptions = [str(a) for a in (raw_clean.get("assumptions") or []) if str(a).strip()]
    raw_suggested_rules = raw_clean.get("suggested_rules") or []

    if raw_premises is None:
        raw_premises = []
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_premises, list) or not isinstance(raw_steps, list):
        return _finish(
            claim_target=dict(raw_target) if isinstance(raw_target, Mapping) else None,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False, reasons=("MALFORMED_PROPOSAL_SCHEMA",)),
            defects_out=[{"code": "MALFORMED_PROPOSAL_SCHEMA", "message": "'premises' and 'steps' must be lists."}],
            explanation="La propuesta incumple el esquema esperado ('premises' y 'steps' deben ser listas).",
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    # 1. Validate claim_target correspondence with original claim
    target_dict = dict(raw_target) if isinstance(raw_target, Mapping) else None
    target_findings = verify_claim_target_correspondence(claim, target_dict or {}, unit_registry=registry)
    if target_findings:
        defects = [
            {"code": f.code, "message": f.message, "field": f.field, "detail": f.detail}
            for f in target_findings
        ]
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(
                status="RETRY",
                allowed=False,
                requires_human=False,
                reasons=tuple(f.code for f in target_findings),
            ),
            defects_out=defects,
            assumptions_out=raw_assumptions,
            explanation=f"El objetivo estructurado `claim_target` no coincide con la afirmación original: {target_findings[0].message}",
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    # 2a. Planner-proposed operations: validated as data, executed only through the closed interpreter
    prop_list, prop_defects = parse_proposed_operations(
        raw_clean.get("proposed_operations"), core_catalog=catalog, registry=registry
    )
    if prop_defects:
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False,
                                       reasons=tuple(d["code"] for d in prop_defects)),
            defects_out=prop_defects,
            assumptions_out=raw_assumptions,
            explanation="Operaciones propuestas inválidas: " + "; ".join(d["message"] for d in prop_defects),
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )
    if prop_list:
        proposed_ops = {op.name: op for op in prop_list}
        catalog = catalog.extend(proposed_operation_specs(prop_list))
        eff_verifiers = _effective_verifier_registry(verifier_registry, catalog)

    # 2. Check for model-suggested rules or unknown operations / undeclared rules before execution
    limitations: list[str] = []
    if isinstance(raw_suggested_rules, list) and raw_suggested_rules:
        for sr in raw_suggested_rules:
            if isinstance(sr, Mapping):
                rid = sr.get("rule_id") or sr.get("name") or "suggested_rule"
                desc = sr.get("description") or ""
                limitations.append(f"UNVERIFIED_SUGGESTED_RULE:{rid}" + (f" ({desc})" if desc else ""))
                raw_assumptions.append(f"unverified_suggested_rule:{rid}")
            else:
                limitations.append(f"UNVERIFIED_SUGGESTED_RULE:{sr}")

    for idx, st in enumerate(raw_steps):
        if not isinstance(st, Mapping):
            return _finish(
                claim_target=target_dict,
                computed_conclusion=None,
                epistemic_verdict="unresolved",
                outcome_category="malformed_proposal",
                routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False, reasons=("MALFORMED_STEP",)),
                defects_out=[{"code": "MALFORMED_STEP", "message": f"Step {idx} must be an object."}],
                explanation="Uno de los pasos propuestos no es un objeto válido.",
                raw_clean=raw_clean,
                graph_admitted=False,
                graph=None,
                assessment=None,
            )
        op_name = str(st.get("operation") or "").strip()
        spec = catalog.get(op_name)
        if spec is None:
            limitations.append(f"UNKNOWN_OPERATION:{op_name or 'empty'}")
        elif spec.name == "rule_application":
            rule_id = st.get("rule_id") or (st.get("parameters") or {}).get("rule_id")
            rule_ver = st.get("rule_version") or (st.get("parameters") or {}).get("rule_version")
            rule_obj = profile.declared_rules.get(str(rule_id)) if rule_id else None
            if rule_obj is None:
                limitations.append(f"UNDECLARED_RULE:{rule_id}")
            elif rule_ver is not None and str(rule_ver) != rule_obj.version:
                limitations.append(f"RULE_VERSION_MISMATCH:{rule_id}@{rule_ver}")

    # 3. Validate premises & enforce anti-claim-as-premise invariant
    grounded_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    groundings_by_id: dict[str, PremiseGroundingResult] = {}
    raw_premises_by_id: dict[str, Mapping[str, Any]] = {}
    all_defects: list[dict[str, Any]] = []
    checks_applied: list[str] = ["claim_target_correspondence"]
    seen_premise_ids: set[str] = set()

    for idx, rp in enumerate(raw_premises):
        if not isinstance(rp, Mapping):
            all_defects.append({"code": "MALFORMED_PREMISE", "message": f"Premise {idx} is not an object."})
            continue
        pid = str(rp.get("claim_id") or f"p_{idx + 1}").strip()
        if pid in seen_premise_ids:
            all_defects.append({"code": "DUPLICATE_PREMISE_ID", "message": f"Duplicate premise claim_id '{pid}'."})
            continue
        seen_premise_ids.add(pid)
        raw_premises_by_id[pid] = rp

        gr = verify_premise_grounding(rp, docs_map, unit_registry=registry, claim_text=claim)
        checks_applied.extend(gr.checks_passed)
        # Detect if an ungrounded quote came from the claim text itself
        q_str = str(rp.get("quote") or rp.get("verbatim_quote") or "").strip()
        claim_copied = any(d.code == "FORBIDDEN_NON_CORPUS_SOURCE" for d in gr.defects) or (
            any(d.code == "QUOTE_NOT_FOUND_IN_SOURCE" for d in gr.defects)
            and bool(q_str)
            and label_key(q_str) in label_key(claim)
        )
        p_defects = list(gr.defects)
        if claim_copied:
            from .fidelity import FidelityFinding
            p_defects.append(
                FidelityFinding(
                    code="CLAIM_USED_AS_PREMISE",
                    message="Original claim cannot be cited as a premise or evidence source.",
                    field="evidence",
                    detail={"claim_id": pid},
                )
            )
        if p_defects:
            d_dicts = [
                {"claim_id": pid, "code": d.code, "message": d.message, "field": d.field, "detail": d.detail}
                for d in p_defects
            ]
            all_defects.extend(d_dicts)
            rejected_records.append({**dict(rp), "claim_id": pid, "defects": d_dicts})
        else:
            groundings_by_id[pid] = gr
            grounded_records.append({
                "claim_id": pid,
                "entity": str(rp.get("entity") or rp.get("subject") or ""),
                "event": str(rp.get("event") or ""),
                "predicate": str(rp.get("predicate") or ""),
                "value": gr.normalized_value,
                "unit": gr.normalized_unit,
                "direction": rp.get("direction"),
                "anchor_reference": rp.get("anchor_reference"),
                "polarity": _coerce_polarity(rp.get("polarity", 1)),
                "evidence": gr.to_dict()["evidence"],
                "matched_clauses": list(gr.matched_clauses),
                "checks_passed": list(gr.checks_passed),
            })

    # Check if any step tries to reference claim_target / claim directly as a premise
    for st in raw_steps:
        if isinstance(st, Mapping):
            prems_ref = [str(p) for p in (st.get("premises") or ())]
            for p_ref in prems_ref:
                if label_key(p_ref) in {"claim", "claim_target", "afirmacion", "afirmación"}:
                    all_defects.append({
                        "code": "CLAIM_USED_AS_PREMISE",
                        "message": f"Step '{st.get('step_id')}' attempts to use '{p_ref}' (the claim itself) as a premise.",
                        "field": "steps.premises",
                    })

    if limitations:
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="unsupported_operation_or_rule",
            routing=OperationalRouting(
                status="ESCALATE",
                allowed=False,
                requires_human=True,
                reasons=tuple(l.split(" ", 1)[0] for l in limitations),
            ),
            premises_out=grounded_records,
            rejected_out=rejected_records,
            assumptions_out=raw_assumptions,
            missing_out=[dict(m) for m in raw_missing if isinstance(m, Mapping)],
            limitations_out=limitations,
            defects_out=all_defects,
            checks_applied=checks_applied,
            explanation="El plan propuesto requiere operaciones o reglas no declaradas en el catálogo controlado: " + "; ".join(limitations),
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    if all_defects:
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="premise_grounding_defect",
            routing=OperationalRouting(
                status="RETRY",
                allowed=False,
                requires_human=False,
                reasons=tuple(dict.fromkeys(d["code"] for d in all_defects)),
            ),
            premises_out=grounded_records,
            rejected_out=rejected_records,
            assumptions_out=raw_assumptions,
            missing_out=[dict(m) for m in raw_missing if isinstance(m, Mapping)],
            defects_out=all_defects,
            checks_applied=checks_applied,
            explanation="Se detectaron defectos de anclaje o correspondencia en las premisas extraídas: "
            + "; ".join(f"{d['code']} ({d['message']})" for d in all_defects),
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    missing_list = [dict(m) for m in raw_missing if isinstance(m, Mapping)]

    # 4. Handle explicit corpus insufficiency (no steps / null conclusion + missing_premises)
    if not raw_steps or derived_conclusion_id is None:
        if missing_list or derived_conclusion_id is None:
            desc = "; ".join(str(m.get("description") or m.get("role") or "") for m in missing_list if m)
            if not missing_list:
                missing_list = [{"role": "derivation_gap", "description": "Las fuentes no aportan todas las premisas necesarias."}]
                desc = missing_list[0]["description"]
            return _finish(
                claim_target=target_dict,
                computed_conclusion=None,
                epistemic_verdict="insufficient_evidence",
                outcome_category="corpus_insufficient_evidence",
                routing=OperationalRouting(
                    status="ESCALATE",
                    allowed=False,
                    requires_human=True,
                    reasons=("MISSING_REQUIRED_PREMISES",),
                ),
                premises_out=grounded_records,
                rejected_out=rejected_records,
                assumptions_out=raw_assumptions,
                missing_out=missing_list,
                checks_applied=checks_applied,
                explanation=f"Indeterminada por evidencia insuficiente en las fuentes (sin usar reloj del sistema ni suponer premisas): {desc}",
                raw_clean=raw_clean,
                graph_admitted=False,
                graph=None,
                assessment=None,
            )

    # 5. Topological order and cycle/dependency validation
    ordered_steps, dag_errors = _topological_steps(raw_steps, set(groundings_by_id.keys()))
    if dag_errors:
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(
                status="RETRY",
                allowed=False,
                requires_human=False,
                reasons=("INVALID_DERIVATION_DAG",),
            ),
            premises_out=grounded_records,
            assumptions_out=raw_assumptions,
            missing_out=missing_list,
            defects_out=[{"code": "INVALID_DERIVATION_DAG", "message": err} for err in dag_errors],
            checks_applied=checks_applied,
            explanation="El grafo propuesto contiene ciclos o referencias inválidas: " + "; ".join(dag_errors),
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    conc_id = str(derived_conclusion_id)
    produced_ids = {str(st["conclusion_id"]) for st in ordered_steps}
    if conc_id not in produced_ids and conc_id not in groundings_by_id:
        return _finish(
            claim_target=target_dict,
            computed_conclusion=None,
            epistemic_verdict="unresolved",
            outcome_category="malformed_proposal",
            routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False, reasons=("MISSING_DERIVED_CONCLUSION_NODE",)),
            premises_out=grounded_records,
            defects_out=[{"code": "MISSING_DERIVED_CONCLUSION_NODE", "message": f"derived_conclusion_id '{conc_id}' is not produced by any step."}],
            explanation=f"El nodo de conclusión '{conc_id}' no es producido por ningún paso del plan.",
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=None,
            assessment=None,
        )

    # 6. Build grounded leaf ClaimNodes and deterministically execute each step
    claims_map: dict[str, ClaimNode] = {}
    for pid, gr in groundings_by_id.items():
        rp = raw_premises_by_id[pid]
        assert gr.evidence_ref is not None
        claims_map[pid] = ClaimNode(
            claim_id=pid,
            kind="extraction",
            subject=str(rp.get("entity") or rp.get("subject") or ""),
            predicate=str(rp.get("predicate") or ""),
            value=gr.normalized_value,
            unit=gr.normalized_unit,
            polarity=_coerce_polarity(rp.get("polarity", 1)),
            text=gr.evidence_ref.quote,
            evidence=(gr.evidence_ref,),
            condition_status=str(rp.get("condition_status") or "unconditional"),
            conditions=tuple(rp.get("conditions") or ()),
            scope=dict(rp.get("scope") or {}),
            metadata={"event": str(rp.get("event") or ""), "direction": rp.get("direction"), "anchor_reference": rp.get("anchor_reference")},
        )

    graph_steps: list[InferenceStep] = []
    verified_steps_out: list[dict[str, Any]] = []
    all_assumptions = list(raw_assumptions)
    anchor_used_info: dict[str, dict[str, Any]] = {}

    for idx, st in enumerate(ordered_steps):
        step_id = str(st.get("step_id") or f"s_{idx + 1}")
        op_name = str(st["operation"])
        spec = catalog.get(op_name)
        assert spec is not None
        prem_ids = tuple(str(p) for p in (st.get("premises") or ()))
        prem_nodes = tuple(claims_map[p] for p in prem_ids)
        step_conc_id = str(st["conclusion_id"])
        params = dict(st.get("parameters") or {})
        if st.get("rule_id") and "rule_id" not in params:
            params["rule_id"] = st["rule_id"]
        if op_name in proposed_ops and proposed_ops[op_name].grade == "D":
            params["__asserted_value"] = st.get("proposed_value")
            params["__asserted_unit"] = st.get("proposed_unit")
        if st.get("rule_version") and "rule_version" not in params:
            params["rule_version"] = st["rule_version"]

        # Step-level explicit anchor correspondence guard for weekday_shift / date_offset
        if spec.name in {"weekday_shift", "date_offset"} and len(prem_ids) == 2:
            p0_id, p1_id = prem_ids
            if p0_id in groundings_by_id and p1_id in groundings_by_id:
                g0, g1 = groundings_by_id[p0_id], groundings_by_id[p1_id]
                is_0_anchor = (normalize_weekday(g0.normalized_value) is not None) if spec.name == "weekday_shift" else (isinstance(g0.normalized_value, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}$", g0.normalized_value)))
                is_1_anchor = (normalize_weekday(g1.normalized_value) is not None) if spec.name == "weekday_shift" else (isinstance(g1.normalized_value, str) and bool(re.match(r"^\d{4}-\d{2}-\d{2}$", g1.normalized_value)))
                if is_0_anchor and not is_1_anchor:
                    a_id, o_id = p0_id, p1_id
                elif is_1_anchor and not is_0_anchor:
                    a_id, o_id = p1_id, p0_id
                else:
                    a_id, o_id = None, None
                if a_id and o_id:
                    checks_applied.append("step_anchor_correspondence")
                    a_findings = verify_step_anchor_correspondence(
                        step_id,
                        groundings_by_id[a_id],
                        raw_premises_by_id[a_id],
                        groundings_by_id[o_id],
                        raw_premises_by_id[o_id],
                    )
                    if a_findings:
                        d_dicts = [
                            {"step_id": step_id, "code": f.code, "message": f.message, "field": f.field, "detail": f.detail}
                            for f in a_findings
                        ]
                        return _finish(
                            claim_target=target_dict,
                            computed_conclusion=None,
                            epistemic_verdict="unresolved",
                            outcome_category="premise_grounding_defect",
                            routing=OperationalRouting(
                                status="RETRY",
                                allowed=False,
                                requires_human=False,
                                reasons=tuple(f.code for f in a_findings),
                            ),
                            premises_out=grounded_records,
                            rejected_out=rejected_records,
                            assumptions_out=all_assumptions,
                            defects_out=d_dicts,
                            checks_applied=checks_applied,
                            explanation=a_findings[0].message,
                            raw_clean=raw_clean,
                            graph_admitted=False,
                            graph=None,
                            assessment=None,
                        )
                    anchor_used_info[step_conc_id] = {
                        "anchor_id": a_id,
                        "offset_id": o_id,
                        "anchor_grounding": groundings_by_id[a_id],
                        "offset_grounding": groundings_by_id[o_id],
                    }

        op_out = catalog.execute(op_name, prem_nodes, params, profile)
        if not op_out.ok:
            if op_out.error_kind == "missing_input":
                err_missing = [
                    *missing_list,
                    {"role": str(op_out.error_code or "missing_input"), "description": f"Paso '{step_id}' ({spec.name}) requiere entrada ausente: {op_out.error_code}."},
                ]
                return _finish(
                    claim_target=target_dict,
                    computed_conclusion=None,
                    epistemic_verdict="insufficient_evidence",
                    outcome_category="corpus_insufficient_evidence",
                    routing=OperationalRouting(
                        status="ESCALATE",
                        allowed=False,
                        requires_human=True,
                        reasons=(str(op_out.error_code or "MISSING_INPUT"),),
                    ),
                    premises_out=grounded_records,
                    assumptions_out=all_assumptions,
                    missing_out=err_missing,
                    checks_applied=checks_applied,
                    explanation=f"Evidencia insuficiente para ejecutar '{spec.name}' en el paso '{step_id}': {op_out.error_code}.",
                    raw_clean=raw_clean,
                    graph_admitted=False,
                    graph=None,
                    assessment=None,
                )
            if op_out.error_kind == "undeclared_rule":
                lim = str(op_out.error_code or f"UNDECLARED_RULE:{step_id}")
                return _finish(
                    claim_target=target_dict,
                    computed_conclusion=None,
                    epistemic_verdict="unresolved",
                    outcome_category="unsupported_operation_or_rule",
                    routing=OperationalRouting(
                        status="ESCALATE",
                        allowed=False,
                        requires_human=True,
                        reasons=(lim,),
                    ),
                    premises_out=grounded_records,
                    assumptions_out=all_assumptions,
                    limitations_out=[lim],
                    checks_applied=checks_applied,
                    explanation=f"Regla u operación no declarada en '{step_id}': {lim}.",
                    raw_clean=raw_clean,
                    graph_admitted=False,
                    graph=None,
                    assessment=None,
                )
            # Formal execution error (e.g. incompatible unit dimensions, division by zero)
            return _finish(
                claim_target=target_dict,
                computed_conclusion=None,
                epistemic_verdict="unresolved",
                outcome_category="premise_grounding_defect",
                routing=OperationalRouting(
                    status="RETRY",
                    allowed=False,
                    requires_human=False,
                    reasons=(str(op_out.error_code or "FORMAL_EXECUTION_ERROR"),),
                ),
                premises_out=grounded_records,
                assumptions_out=all_assumptions,
                defects_out=[{
                    "step_id": step_id,
                    "code": str(op_out.error_code or "FORMAL_EXECUTION_ERROR"),
                    "message": f"Fallo formal al ejecutar '{spec.name}' en '{step_id}': {op_out.error_code}.",
                    "detail": op_out.details,
                }],
                checks_applied=checks_applied,
                explanation=f"Error formal en el paso '{step_id}' ({spec.name}): {op_out.error_code}.",
                raw_clean=raw_clean,
                graph_admitted=False,
                graph=None,
                assessment=None,
            )

        # Construct recomputed derived ClaimNode (NEVER trust step.proposed_value!)
        step_assumps = tuple(str(a) for a in (st.get("assumptions") or ())) + op_out.assumptions
        all_assumptions.extend(step_assumps)
        conc_subj = str(target_dict.get("subject") or prem_nodes[0].subject)
        conc_pred = str(target_dict.get("predicate") or f"{spec.name}_result")
        claims_map[step_conc_id] = ClaimNode(
            claim_id=step_conc_id,
            kind="derivation",
            subject=conc_subj,
            predicate=conc_pred,
            value=op_out.value,
            unit=op_out.unit,
            polarity=1,
            text=f"{spec.name}({', '.join(prem_ids)}) = {op_out.value}" + (f" {op_out.unit}" if op_out.unit else ""),
            assumptions=step_assumps,
        )
        op_key = label_key(spec.name).replace(" ", "_")
        explicit_obs: tuple[VerificationObligation, ...] = ()
        if op_key not in DEFAULT_OPERATION_FAMILIES and spec.name not in DEFAULT_OPERATION_FAMILIES:
            explicit_obs = (
                VerificationObligation(
                    obligation_id=f"{step_id}::{spec.family}",
                    family=spec.family,
                    description=f"Verify controlled catalog operation '{spec.name}'",
                    critical=True,
                    parameters=dict(params),
                ),
            )
        inf_step = InferenceStep(
            step_id=step_id,
            premises=prem_ids,
            conclusion_id=step_conc_id,
            operation=spec.name,
            rule_id=st.get("rule_id"),
            rule_version=st.get("rule_version"),
            assumptions=step_assumps,
            obligations=explicit_obs,
            parameters=params,
        )
        graph_steps.append(inf_step)

        prop_val = st.get("proposed_value")
        prop_unit = st.get("proposed_unit")
        disagreed = (
            prop_val is not None
            and not _values_match(op_out.value, op_out.unit, prop_val, prop_unit, registry)
        )
        verified_steps_out.append({
            "step_id": step_id,
            "operation": spec.name,
            "operation_version": spec.version,
            "premises": list(prem_ids),
            "conclusion_id": step_conc_id,
            "recomputed_value": op_out.value,
            "recomputed_unit": op_out.unit,
            "details": op_out.details,
            "model_proposed_value": prop_val,
            "model_proposed_unit": prop_unit,
            "model_disagreed": disagreed,
            "grade": step_grade(spec.name, proposed_ops),
            "proposed_by_planner": spec.name in proposed_ops,
        })

    # 7. Evaluate the recomputed JustificationGraph via v4 evaluate_response
    graph = JustificationGraph(
        claims=claims_map,
        steps=tuple(graph_steps),
        conclusion_id=conc_id,
    )
    assessment = evaluate_response(
        graph,
        evidence=docs_map,
        domain_profile=profile,
        verifier_registry=eff_verifiers,
        as_of=as_of,
        question=claim,
    )
    all_assumptions.extend(assessment.assumptions)

    conc_node = graph.claims[conc_id]
    computed_conc_dict = {
        "claim_id": conc_id,
        "value": conc_node.value,
        "unit": conc_node.unit,
        "text": conc_node.text,
        "polarity": conc_node.polarity,
    }

    if assessment.assessment != "supported":
        refuted_reasons = [
            r for c in assessment.completed_checks if c.status in ("refuted", "unknown") for r in c.reasons
        ]
        routing_status = "RETRY" if assessment.assessment == "refuted" else "ESCALATE"
        return _finish(
            claim_target=target_dict,
            computed_conclusion=computed_conc_dict,
            epistemic_verdict="unresolved",
            outcome_category="premise_grounding_defect" if assessment.assessment == "refuted" else "corpus_insufficient_evidence",
            routing=OperationalRouting(
                status=routing_status,
                allowed=False,
                requires_human=(routing_status == "ESCALATE"),
                reasons=tuple(dict.fromkeys(refuted_reasons or [f"ASSESSMENT_{assessment.assessment.upper()}"])),
            ),
            premises_out=grounded_records,
            steps_out=verified_steps_out,
            assumptions_out=all_assumptions,
            defects_out=[{"code": r, "message": f"Verificador v4 marcó {assessment.assessment}: {r}"} for r in refuted_reasons],
            checks_applied=checks_applied,
            explanation=f"La evaluación formal del grafo arrojó '{assessment.assessment}' ({', '.join(refuted_reasons)}).",
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=graph,
            assessment=assessment,
        )

    # 8. Compare recomputed conclusion against verified claim_target
    target_val = target_dict.get("value")
    target_unit = target_dict.get("unit")
    target_pol = _coerce_polarity(target_dict.get("polarity", 1))
    reliability = compute_reliability(
        conclusion_id=conc_id,
        premises={r["claim_id"]: r for r in grounded_records},
        steps=verified_steps_out,
        proposed=proposed_ops,
        policy=rel_policy,
    )
    if not _comparable(conc_node.value, target_val):
        kinds = {"computed": _value_kind(conc_node.value), "claim_target": _value_kind(target_val)}
        return _finish(
            claim_target=target_dict,
            computed_conclusion=computed_conc_dict,
            epistemic_verdict="unresolved",
            outcome_category="incomparable_conclusion",
            routing=OperationalRouting(status="RETRY", allowed=False, requires_human=False,
                                       reasons=("INCOMPARABLE_CONCLUSION_AND_CLAIM",)),
            premises_out=grounded_records,
            steps_out=verified_steps_out,
            assumptions_out=all_assumptions,
            defects_out=[{"code": "INCOMPARABLE_CONCLUSION_AND_CLAIM",
                          "message": f"La conclusión calculada ({kinds['computed']}: {conc_node.value!r}) no es comparable con el valor afirmado ({kinds['claim_target']}: {target_val!r}).",
                          "detail": kinds}],
            checks_applied=checks_applied,
            explanation=(f"Sin resolver: la derivación produce un valor de tipo {kinds['computed']} y la afirmación espera "
                         f"{kinds['claim_target']}; tipos incomparables no confirman ni refutan."),
            raw_clean=raw_clean,
            graph_admitted=False,
            graph=graph,
            assessment=assessment,
            reliability=reliability,
        )
    val_matches = _values_match(
        conc_node.value,
        conc_node.unit,
        target_val,
        target_unit,
        registry,
        claim_text=claim,
    )
    claim_holds = val_matches if target_pol == 1 else (not val_matches)

    if claim_holds:
        auth = authorize(assessment, operation="verify_claim", policy=profile)
        routing = OperationalRouting(
            status=auth.status,
            allowed=auth.allowed,
            requires_human=auth.requires_human,
            reasons=tuple(i.code for i in auth.issues) if auth.issues else ("DERIVATION_VERIFIED_AND_MATCHES_CLAIM",),
            authorization_verdict={
                "status": auth.status,
                "allowed": auth.allowed,
                "requires_human": auth.requires_human,
                "snapshot_digest": auth.snapshot_digest,
                "issues": [i.code for i in auth.issues],
            },
        )
        if not reliability["pass_allowed_by_grade"]:
            routing = OperationalRouting(
                status="ESCALATE",
                allowed=False,
                requires_human=True,
                reasons=(f"RELIABILITY_BELOW_PASS_THRESHOLD:{reliability['grade']}<{rel_policy.pass_min_grade}",
                         *routing.reasons),
                authorization_verdict=routing.authorization_verdict,
            )
        step_desc = "; ".join(
            f"{s['operation']}({', '.join(s['premises'])}) -> {s['recomputed_value']}"
            + (f" {s['recomputed_unit']}" if s['recomputed_unit'] else "")
            for s in verified_steps_out
        )
        return _finish(
            claim_target=target_dict,
            computed_conclusion=computed_conc_dict,
            epistemic_verdict="derived",
            outcome_category="verified_derivation",
            routing=routing,
            premises_out=grounded_records,
            steps_out=verified_steps_out,
            assumptions_out=all_assumptions,
            checks_applied=checks_applied + ["conclusion_claim_comparison"],
            explanation=(
                f"Afirmación derivada y verificada de forma determinista: {step_desc}, "
                f"coincidiendo con la afirmación original ({target_val!r}). Fiabilidad {reliability['grade']}: {reliability['guarantee']}"
            ),
            raw_clean=raw_clean,
            graph_admitted=True,
            graph=graph,
            assessment=assessment,
            reliability=reliability,
        )

    # Refuted: recomputed conclusion is valid under explicit premises and contradicts claim_target
    anchor_diags: list[dict[str, Any]] = []
    if conc_id in anchor_used_info:
        aui = anchor_used_info[conc_id]
        a_gr: PremiseGroundingResult = aui["anchor_grounding"]
        o_gr: PremiseGroundingResult = aui["offset_grounding"]
        if normalize_weekday(a_gr.normalized_value) is not None:
            off_val = _to_decimal_safe(o_gr.normalized_value)
            if off_val is not None:
                from .operations import normalize_day_offset
                off_int, _ = normalize_day_offset(off_val, o_gr.normalized_unit, registry)
                if off_int is not None:
                    anchor_diags = diagnose_alternate_anchors(
                        docs_map,
                        offset_days=off_int,
                        used_anchor_weekday=str(a_gr.normalized_value),
                        used_anchor_clauses=a_gr.matched_clauses,
                        offset_clauses=o_gr.matched_clauses,
                        claimed_value=target_val,
                        explicit_anchor_phrase=(o_gr.explicit_anchor or {}).get("phrase"),
                    )

    comp_str = f"{conc_node.value}" + (f" {conc_node.unit}" if conc_node.unit else "")
    diag_suffix = f" {anchor_diags[0]['explanation']}" if anchor_diags else ""
    routing = OperationalRouting(
        status="RETRY",
        allowed=False,
        requires_human=False,
        reasons=("CLAIM_CONTRADICTS_COMPUTED_CONCLUSION",),
    )
    if not reliability["pass_allowed_by_grade"]:
        routing = OperationalRouting(
            status="ESCALATE",
            allowed=False,
            requires_human=True,
            reasons=(f"RELIABILITY_BELOW_PASS_THRESHOLD:{reliability['grade']}<{rel_policy.pass_min_grade}",
                     "CLAIM_CONTRADICTS_COMPUTED_CONCLUSION"),
        )
    return _finish(
        claim_target=target_dict,
        computed_conclusion=computed_conc_dict,
        epistemic_verdict="refuted",
        outcome_category="verified_refutation",
        routing=routing,
        premises_out=grounded_records,
        steps_out=verified_steps_out,
        assumptions_out=all_assumptions,
        anchor_diags=anchor_diags,
        checks_applied=checks_applied + ["conclusion_claim_comparison"],
        explanation=(
            f"Afirmación refutada: el cálculo controlado a partir de las premisas verificadas produce "
            f"'{comp_str}', que contradice lo afirmado ({target_val!r}).{diag_suffix} Fiabilidad {reliability['grade']}: {reliability['guarantee']}"
        ),
        raw_clean=raw_clean,
        graph_admitted=True,
        graph=graph,
        assessment=assessment,
        reliability=reliability,
    )


# ---------------------------------------------------------------------------
# End-to-End Public Consumer Interface
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """Consumer-facing verifier taking sources + a natural-language claim and running plan + verify."""

    def __init__(
        self,
        *,
        model: OpenAICompatibleModel | None = None,
        planner: DerivationPlanner | None = None,
        catalog: OperationCatalog = DEFAULT_OPERATION_CATALOG,
        domain_profile: DomainProfile | None = None,
        verifier_registry: VerifierRegistry | None = None,
        max_retries: int = 1,
        unit_profile: str = "auto",
        reliability_policy: ReliabilityPolicy | None = None,
    ) -> None:
        self.reliability_policy = reliability_policy or DEFAULT_RELIABILITY_POLICY
        if unit_profile not in UNIT_PROFILE_CHOICES:
            raise ValueError(f"unit_profile must be one of {UNIT_PROFILE_CHOICES}, got {unit_profile!r}")
        if planner is not None:
            self.planner: DerivationPlanner = planner
        elif model is not None:
            self.planner = ClaimDerivationPlanner(model=model)
        else:
            raise ValueError("Provide either 'model' (OpenAICompatibleModel) or 'planner' (DerivationPlanner).")
        if max_retries < 0 or max_retries > 5:
            raise ValueError("max_retries must be in [0, 5].")
        self.catalog = catalog
        self.domain_profile = domain_profile or DomainProfile(profile_id="default")
        self._explicit_domain_profile = domain_profile
        self.unit_profile = unit_profile
        self.verifier_registry = verifier_registry
        self.max_retries = max_retries

    def verify(
        self,
        sources: (
            Mapping[str, SourceDocument]
            | Mapping[str, str]
            | Mapping[str, Mapping[str, Any]]
            | Sequence[SourceDocument | Mapping[str, Any] | str]
        ),
        claim: str,
        *,
        as_of: str | None = None,
    ) -> ClaimVerificationResult:
        t_total0 = time.monotonic()
        docs_map, docs_list = normalize_sources(sources)
        profile, unit_profile_label, unit_registry_fp = _resolve_derivation_profile(
            self._explicit_domain_profile, self.unit_profile, claim, docs_map
        )

        calls = 0
        retries = 0
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        planning_ms = 0.0
        verifier_ms_acc = 0.0
        attempt_traces: list[dict[str, Any]] = []
        repair_feedback: list[dict[str, Any]] | None = None

        last_result: ClaimVerificationResult | None = None
        for attempt_idx in range(self.max_retries + 1):
            call_res = self.planner.plan(
                sources=docs_list,
                claim=claim,
                catalog=self.catalog,
                domain_profile=profile,
                repair_feedback=repair_feedback,
            )
            calls += 1
            if attempt_idx > 0:
                retries += 1
            planning_ms += call_res.latency_ms
            prompt_tokens += int(call_res.usage.get("prompt_tokens") or 0)
            completion_tokens += int(call_res.usage.get("completion_tokens") or 0)
            total_tokens += int(call_res.usage.get("total_tokens") or 0)

            if not call_res.ok or call_res.proposal is None:
                is_provider_err = call_res.error_kind == "provider_error"
                outcome_cat = "provider_error" if is_provider_err else "malformed_proposal"
                routing = OperationalRouting(
                    status="ESCALATE" if is_provider_err else "RETRY",
                    allowed=False,
                    requires_human=is_provider_err,
                    reasons=("PROVIDER_ERROR" if is_provider_err else "MALFORMED_PLANNER_OUTPUT",),
                )
                attempt_traces.append({
                    "attempt": attempt_idx + 1,
                    "ok": False,
                    "outcome_category": outcome_cat,
                    "error_message": call_res.error_message,
                    "latency_ms": call_res.latency_ms,
                    "usage": dict(call_res.usage),
                })
                if not is_provider_err and attempt_idx < self.max_retries:
                    repair_feedback = [{"code": "MALFORMED_PLANNER_OUTPUT", "message": call_res.error_message or "Invalid JSON object"}]
                    continue

                total_ms = round((time.monotonic() - t_total0) * 1000.0, 3)
                p_ident = dict(self.planner.identity)
                tel = PlannerTelemetry(
                    model_identity=p_ident,
                    prompt_version=str(p_ident.get("prompt_version") or DERIVATION_PLANNER_PROMPT_VERSION),
                    prompt_digest=str(p_ident.get("prompt_digest") or digest(DERIVATION_PLANNER_PROMPT)),
                    catalog_id=self.catalog.catalog_id,
                    catalog_version=self.catalog.version,
                    catalog_fingerprint=self.catalog.fingerprint(),
                    calls=calls,
                    retries=retries,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    planning_latency_ms=round(planning_ms, 3),
                    verifier_latency_ms=round(verifier_ms_acc, 3),
                    total_latency_ms=total_ms,
                    attempt_traces=tuple(attempt_traces),
                    unit_profile=unit_profile_label,
                    unit_registry_fingerprint=unit_registry_fp,
                )
                certainty = _build_certainty_report(
                    epistemic_verdict="unresolved",
                    outcome_category=outcome_cat,
                    grounded_count=0,
                    rejected_count=0,
                    checks_applied=(),
                    docs_map=docs_map,
                    profile=profile,
                    assessment=None,
                )
                return ClaimVerificationResult(
                    original_claim=claim,
                    claim_target=None,
                    computed_conclusion=None,
                    epistemic_verdict="unresolved",
                    outcome_category=outcome_cat,
                    operational_routing=routing,
                    premises=(),
                    rejected_premises=(),
                    verified_steps=(),
                    assumptions=(),
                    missing_premises=(),
                    limitations=(),
                    defects=({"code": outcome_cat.upper(), "message": call_res.error_message or outcome_cat},),
                    anchor_diagnostics=(),
                    certainty=certainty,
                    explanation=(
                        f"Fallo del proveedor de inferencia (distinto de ausencia de evidencia): {call_res.error_message}"
                        if is_provider_err
                        else f"Salida malformada del planificador: {call_res.error_message}"
                    ),
                    raw_proposal=None,
                    graph_admitted=False,
                    graph=None,
                    assessment=None,
                    telemetry=tel,
                )

            ver_res = verify_plan(
                docs_map,
                claim,
                call_res.proposal,
                catalog=self.catalog,
                domain_profile=self._explicit_domain_profile,
                verifier_registry=self.verifier_registry,
                as_of=as_of,
                unit_profile=self.unit_profile,
                reliability_policy=self.reliability_policy,
            )
            verifier_ms_acc += ver_res.telemetry.verifier_latency_ms
            attempt_traces.append({
                "attempt": attempt_idx + 1,
                "ok": ver_res.epistemic_verdict in ("derived", "refuted", "insufficient_evidence"),
                "outcome_category": ver_res.outcome_category,
                "epistemic_verdict": ver_res.epistemic_verdict,
                "defects": [d.get("code") for d in ver_res.defects],
                "latency_ms": call_res.latency_ms,
                "verifier_latency_ms": ver_res.telemetry.verifier_latency_ms,
                "usage": dict(call_res.usage),
            })
            last_result = ver_res

            # Retry ONLY on repairable planner/grounding defects
            if (
                ver_res.outcome_category in {"malformed_proposal", "premise_grounding_defect", "incomparable_conclusion"}
                and attempt_idx < self.max_retries
            ):
                repair_feedback = [
                    {
                        "code": d.get("code"),
                        "message": d.get("message"),
                        "detail": d.get("detail"),
                    }
                    for d in ver_res.defects
                ]
                continue
            break

        assert last_result is not None
        total_ms = round((time.monotonic() - t_total0) * 1000.0, 3)
        p_ident = dict(self.planner.identity)
        final_tel = PlannerTelemetry(
            model_identity=p_ident,
            prompt_version=str(p_ident.get("prompt_version") or DERIVATION_PLANNER_PROMPT_VERSION),
            prompt_digest=str(p_ident.get("prompt_digest") or digest(DERIVATION_PLANNER_PROMPT)),
            catalog_id=self.catalog.catalog_id,
            catalog_version=self.catalog.version,
            catalog_fingerprint=self.catalog.fingerprint(),
            calls=calls,
            retries=retries,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            planning_latency_ms=round(planning_ms, 3),
            verifier_latency_ms=round(verifier_ms_acc, 3),
            total_latency_ms=total_ms,
            attempt_traces=tuple(attempt_traces),
            unit_profile=unit_profile_label,
            unit_registry_fingerprint=unit_registry_fp,
        )
        return ClaimVerificationResult(
            original_claim=last_result.original_claim,
            claim_target=last_result.claim_target,
            computed_conclusion=last_result.computed_conclusion,
            epistemic_verdict=last_result.epistemic_verdict,
            outcome_category=last_result.outcome_category,
            operational_routing=last_result.operational_routing,
            premises=last_result.premises,
            rejected_premises=last_result.rejected_premises,
            verified_steps=last_result.verified_steps,
            assumptions=last_result.assumptions,
            missing_premises=last_result.missing_premises,
            limitations=last_result.limitations,
            defects=last_result.defects,
            anchor_diagnostics=last_result.anchor_diagnostics,
            certainty=last_result.certainty,
            explanation=last_result.explanation,
            raw_proposal=last_result.raw_proposal,
            graph_admitted=last_result.graph_admitted,
            graph=last_result.graph,
            assessment=last_result.assessment,
            telemetry=final_tel,
            reliability=last_result.reliability,
        )


DEFAULT_PLANNER_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PLANNER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def build_derivation_planner_model(
    api_key: str,
    *,
    base_url: str = DEFAULT_PLANNER_BASE_URL,
    model: str = DEFAULT_PLANNER_MODEL,
    timeout: float = 120.0,
    enable_thinking: bool = False,
    max_tokens: int = 2048,
) -> OpenAICompatibleModel:
    """Provider configuration used by the live derivation benchmark
    (`benchmarks/v4_derivation_planner_eval.py`): temperature 0, max_tokens 2048,
    2 transport retries, thinking disabled. Applications (API, CLI) should build the
    planner model here so that their results stay comparable with the benchmark."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key is required to build the derivation planner model.")
    return OpenAICompatibleModel(
        base_url=base_url,
        model=model,
        api_key=api_key.strip(),
        timeout=float(timeout),
        max_tokens=int(max_tokens),
        retries=2,
        extra_body={"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
    )


def verify_claim(
    sources: (
        Mapping[str, SourceDocument]
        | Mapping[str, str]
        | Mapping[str, Mapping[str, Any]]
        | Sequence[SourceDocument | Mapping[str, Any] | str]
    ),
    claim: str,
    *,
    model: OpenAICompatibleModel | None = None,
    planner: DerivationPlanner | None = None,
    catalog: OperationCatalog = DEFAULT_OPERATION_CATALOG,
    domain_profile: DomainProfile | None = None,
    verifier_registry: VerifierRegistry | None = None,
    as_of: str | None = None,
    max_retries: int = 1,
    unit_profile: str = "auto",
    reliability_policy: ReliabilityPolicy | None = None,
) -> ClaimVerificationResult:
    """Public single-call entrypoint verifying a natural-language claim against sources via planner + v4 core."""
    verifier = ClaimVerifier(
        model=model,
        planner=planner,
        catalog=catalog,
        domain_profile=domain_profile,
        verifier_registry=verifier_registry,
        max_retries=max_retries,
        unit_profile=unit_profile,
        reliability_policy=reliability_policy,
    )
    return verifier.verify(sources, claim, as_of=as_of)
