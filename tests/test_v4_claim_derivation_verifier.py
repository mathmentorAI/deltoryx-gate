"""Deterministic unit & end-to-end tests for v4 derivation planning, grounding, and CLI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError

import pytest

import deltoryx_gate.v4 as dg_v4
from claimlayer.cli import main as cli_main
from claimlayer.semantic.adapters import OpenAICompatibleModel
from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    ClaimNode,
    ClaimVerifier,
    DEFAULT_OPERATION_CATALOG,
    DeclaredRule,
    DomainProfile,
    EmpiricalCalibration,
    InferenceStep,
    JustificationGraph,
    OperationCatalog,
    OperationOutput,
    OperationSpec,
    PlannerCallResult,
    compute_date_offset,
    compute_weekday_shift,
    evaluate_response,
    normalize_weekday,
    verify_claim,
    verify_plan,
)


class _ScriptedPlanner:
    """Offline deterministic test double implementing the DerivationPlanner protocol."""

    def __init__(self, responses: Sequence[PlannerCallResult | dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "test_double", "model": "scripted-planner"}

    def plan(
        self,
        *,
        sources: Sequence[SourceDocument],
        claim: str,
        catalog: OperationCatalog,
        domain_profile: DomainProfile,
        repair_feedback: Sequence[Mapping[str, Any]] | None = None,
    ) -> PlannerCallResult:
        idx = len(self.calls)
        self.calls.append({
            "sources": [d.document_id for d in sources],
            "claim": claim,
            "catalog_ops": catalog.names(),
            "repair_feedback": list(repair_feedback) if repair_feedback else None,
        })
        item = self._responses[min(idx, len(self._responses) - 1)]
        if isinstance(item, PlannerCallResult):
            return item
        return PlannerCallResult(
            ok=True,
            proposal=item,
            latency_ms=12.5,
            usage={"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
        )


# ---------------------------------------------------------------------------
# 1. OperationCatalog, Calendar & Arithmetic Core
# ---------------------------------------------------------------------------

def test_operation_catalog_calendar_and_arithmetic() -> None:
    assert normalize_weekday("Miércoles") == (2, "miércoles", "es")
    assert normalize_weekday("Friday") == (4, "viernes", "en")

    sh_fri = compute_weekday_shift("viernes", -2)
    assert sh_fri["computed_weekday"] == "miércoles"
    assert sh_fri["crossed_week_boundary"] is False

    sh_tue = compute_weekday_shift("martes", -2)
    assert sh_tue["computed_weekday"] == "domingo"
    assert sh_tue["crossed_week_boundary"] is True

    sh_en = compute_weekday_shift("Friday", 3)
    assert sh_en["computed_weekday"] == "monday"
    assert sh_en["crossed_week_boundary"] is True

    d_res = compute_date_offset("2026-11-20", -2)
    assert d_res["computed_date"] == "2026-11-18"
    assert d_res["computed_weekday"] == "miércoles"

    # weeks unit converts cleanly x7
    p_anchor = ClaimNode(claim_id="a", value="viernes")
    p_weeks = ClaimNode(claim_id="w", value=-1, unit="weeks")
    out_w = DEFAULT_OPERATION_CATALOG.execute("weekday_shift", (p_anchor, p_weeks))
    assert out_w.ok and out_w.value == "viernes"

    # business_days is a distinct dimension and MUST NOT be silently converted as calendar_days
    p_bus = ClaimNode(claim_id="b", value=-2, unit="business_days")
    out_bad_u = DEFAULT_OPERATION_CATALOG.execute("weekday_shift", (p_anchor, p_bus))
    assert not out_bad_u.ok
    assert out_bad_u.error_kind == "formal_error"
    assert (out_bad_u.error_code or "").startswith("INCOMPATIBLE_OFFSET_UNIT")

    # Unknown operation returns explicit undeclared_rule limitation
    out_unk = DEFAULT_OPERATION_CATALOG.execute("magic_guess", (p_anchor,))
    assert not out_unk.ok
    assert out_unk.error_kind == "undeclared_rule"
    assert out_unk.error_code == "UNKNOWN_OPERATION:magic_guess"

    # Catalog extensibility without changing query interface
    def _exec_abs_diff(prems: Sequence[ClaimNode], params: Mapping[str, Any], prof: DomainProfile) -> OperationOutput:
        return OperationOutput(ok=True, value=abs(int(prems[0].value) - int(prems[1].value)), unit=prems[0].unit)

    extended = DEFAULT_OPERATION_CATALOG.extend(
        [
            OperationSpec(
                name="abs_diff",
                version="1.0.0",
                family="arithmetic",
                description="Absolute integer difference.",
                inputs=({"role": "a", "type": "number"}, {"role": "b", "type": "number"}),
                output_type="number",
                executor=_exec_abs_diff,
            )
        ],
        catalog_id="extended_test",
    )
    assert extended.has("abs_diff")
    assert not DEFAULT_OPERATION_CATALOG.has("abs_diff")
    assert extended.execute("abs_diff", (ClaimNode(claim_id="x", value=10), ClaimNode(claim_id="y", value=14))).value == 4


def test_direct_temporal_verifier_weekday_shift_and_date_offset() -> None:
    doc = SourceDocument(
        document_id="s1",
        text="Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños.",
        weight=1.0,
    )
    ref_bday = EvidenceRef("s1", doc.revision, doc.text.index("Su cumpleaños fue el martes"), doc.text.index("Su cumpleaños fue el martes") + len("Su cumpleaños fue el martes"), "Su cumpleaños fue el martes")
    ref_off = EvidenceRef("s1", doc.revision, doc.text.index("quedé con ella dos días antes de su cumpleaños"), len(doc.text) - 1, "quedé con ella dos días antes de su cumpleaños")

    g_ok = JustificationGraph(
        claims={
            "p_a": ClaimNode(claim_id="p_a", kind="extraction", subject="su cumpleaños", predicate="weekday", value="martes", evidence=(ref_bday,)),
            "p_o": ClaimNode(claim_id="p_o", kind="extraction", subject="cita", predicate="offset_days", value=-2, unit="calendar_days", evidence=(ref_off,)),
            "c": ClaimNode(claim_id="c", kind="derivation", subject="cita", predicate="weekday", value="domingo"),
        },
        steps=(InferenceStep(step_id="s1", operation="weekday_shift", premises=("p_a", "p_o"), conclusion_id="c"),),
        conclusion_id="c",
    )
    res_ok = evaluate_response(g_ok, evidence={"s1": doc}, question="La cita fue el domingo.")
    assert res_ok.assessment == "supported"
    assert any("WEEKDAY_SHIFT_VERIFIED" in c.reasons for c in res_ok.completed_checks)

    g_bad = JustificationGraph(
        claims={
            **g_ok.claims,
            "c": ClaimNode(claim_id="c", kind="derivation", subject="cita", predicate="weekday", value="miércoles"),
        },
        steps=g_ok.steps,
        conclusion_id="c",
    )
    res_bad = evaluate_response(g_bad, evidence={"s1": doc}, question="La cita fue el miércoles.")
    assert res_bad.assessment == "refuted"
    assert any("WEEKDAY_SHIFT_MISMATCH" in c.reasons for c in res_bad.completed_checks)

    # date_offset via TemporalVerifier
    d_doc = SourceDocument(document_id="d1", text="Fecha de firma: 2026-11-20. Vence 2 días antes.", weight=1.0)
    r1 = EvidenceRef("d1", d_doc.revision, 0, 26, "Fecha de firma: 2026-11-20")
    r2 = EvidenceRef("d1", d_doc.revision, 28, 46, "Vence 2 días antes")
    g_date = JustificationGraph(
        claims={
            "p1": ClaimNode(claim_id="p1", kind="extraction", subject="contrato", predicate="fecha", value="2026-11-20", evidence=(r1,)),
            "p2": ClaimNode(claim_id="p2", kind="extraction", subject="contrato", predicate="offset_days", value=-2, unit="calendar_days", evidence=(r2,)),
            "c": ClaimNode(claim_id="c", kind="derivation", subject="vencimiento", predicate="fecha", value="2026-11-18"),
        },
        steps=(InferenceStep(step_id="sd", operation="date_offset", premises=("p1", "p2"), conclusion_id="c"),),
        conclusion_id="c",
    )
    res_date = evaluate_response(g_date, evidence={"d1": d_doc})
    assert res_date.assessment == "supported"
    assert any("DATE_OFFSET_VERIFIED" in c.reasons for c in res_date.completed_checks)


# ---------------------------------------------------------------------------
# 2. Five Temporal Acceptance Cases via Public Entrypoint
# ---------------------------------------------------------------------------

def _temporal_plan_simple(claimed_day: str, proposed_step_val: str = "miércoles") -> dict[str, Any]:
    return {
        "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": claimed_day, "unit": None, "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": "Hoy es viernes",
                "entity": "hoy",
                "event": "hoy",
                "predicate": "dia_semana",
                "value": "viernes",
                "unit": None,
                "direction": "neutral",
                "anchor_reference": None,
                "polarity": 1,
            },
            {
                "claim_id": "p2",
                "document_id": "doc_1",
                "quote": "quedé con ella dos días antes",
                "entity": "cita",
                "event": "cita",
                "predicate": "desplazamiento_dias",
                "value": -2,
                "unit": "calendar_days",
                "direction": "before",
                "anchor_reference": None,
                "polarity": 1,
            },
        ],
        "steps": [
            {
                "step_id": "s1",
                "operation": "weekday_shift",
                "premises": ["p1", "p2"],
                "conclusion_id": "c1",
                "proposed_value": proposed_step_val,
            }
        ],
        "derived_conclusion_id": "c1",
        "missing_premises": [],
        "assumptions": [],
        "suggested_rules": [],
    }


def test_acceptance_five_temporal_cases_via_public_api() -> None:
    src_simple = {"doc_1": "Hoy es viernes y quedé con ella dos días antes."}

    # Case 1: Derived Wednesday
    r1 = verify_claim(src_simple, "La cita fue el miércoles.", planner=_ScriptedPlanner([_temporal_plan_simple("miércoles")]))
    assert r1.epistemic_verdict == "derived"
    assert r1.outcome_category == "verified_derivation"
    assert r1.operational_routing.status == "PASS"
    assert r1.operational_routing.allowed is True
    assert r1.computed_conclusion is not None
    assert r1.computed_conclusion["value"] == "miércoles"
    assert r1.verified_steps[0]["details"]["anchor_weekday"] == "viernes"
    assert r1.verified_steps[0]["details"]["offset_days"] == -2
    assert r1.certainty.calibrated_correctness_probability is None
    assert r1.certainty.calibration_status == "uncalibrated"
    assert r1.certainty.derivation_validity["status"] == "exact_under_explicit_premises"

    # Case 2: Refuted Thursday — even if model wrongly proposes 'jueves' on step, Deltoryx recomputes 'miércoles'
    r2 = verify_claim(
        src_simple,
        "La cita fue el jueves.",
        planner=_ScriptedPlanner([_temporal_plan_simple("jueves", proposed_step_val="jueves")]),
    )
    assert r2.epistemic_verdict == "refuted"
    assert r2.outcome_category == "verified_refutation"
    assert r2.operational_routing.status == "RETRY"
    assert r2.computed_conclusion is not None
    assert r2.computed_conclusion["value"] == "miércoles"
    assert r2.verified_steps[0]["model_disagreed"] is True

    # Case 3: Missing anchor -> Indeterminate, system clock (2026-11-20 Friday) NEVER injected!
    src_missing = {"doc_1": "Quedé con ella dos días antes."}
    plan_missing = {
        "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": "miércoles", "unit": None, "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": "Quedé con ella dos días antes",
                "entity": "cita",
                "event": "cita",
                "predicate": "desplazamiento_dias",
                "value": -2,
                "unit": "calendar_days",
                "direction": "before",
                "anchor_reference": None,
                "polarity": 1,
            }
        ],
        "steps": [],
        "derived_conclusion_id": None,
        "missing_premises": [{"role": "anchor_weekday", "description": "Falta el día de la semana de referencia."}],
        "assumptions": [],
        "suggested_rules": [],
    }
    r3 = verify_claim(src_missing, "La cita fue el miércoles.", planner=_ScriptedPlanner([plan_missing]))
    assert r3.epistemic_verdict == "insufficient_evidence"
    assert r3.outcome_category == "corpus_insufficient_evidence"
    assert r3.operational_routing.status == "ESCALATE"
    assert r3.computed_conclusion is None
    assert len(r3.missing_premises) == 1

    # Case 3 variant: partial step with missing anchor also resolves to insufficient_evidence
    plan_missing_partial_step = {
        **plan_missing,
        "steps": [{"step_id": "s1", "operation": "weekday_shift", "premises": ["p1"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
        "missing_premises": [],
    }
    r3_alt = verify_claim(src_missing, "La cita fue el miércoles.", planner=_ScriptedPlanner([plan_missing_partial_step]))
    assert r3_alt.epistemic_verdict == "insufficient_evidence"
    assert r3_alt.outcome_category == "corpus_insufficient_evidence"
    assert r3_alt.computed_conclusion is None

    # Cases 4 & 5: Birthday anchor crossing week boundary
    src_bday = {"doc_1": "Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños."}

    def _bday_plan(claimed_day: str) -> dict[str, Any]:
        return {
            "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": claimed_day, "unit": None, "polarity": 1},
            "premises": [
                {
                    "claim_id": "p_bday",
                    "document_id": "doc_1",
                    "quote": "Su cumpleaños fue el martes",
                    "entity": "su cumpleaños",
                    "event": "cumpleaños",
                    "predicate": "dia_semana",
                    "value": "martes",
                    "unit": None,
                    "direction": "neutral",
                    "anchor_reference": None,
                    "polarity": 1,
                },
                {
                    "claim_id": "p_off",
                    "document_id": "doc_1",
                    "quote": "quedé con ella dos días antes de su cumpleaños",
                    "entity": "cita",
                    "event": "cita",
                    "predicate": "desplazamiento_dias",
                    "value": -2,
                    "unit": "calendar_days",
                    "direction": "before",
                    "anchor_reference": "su cumpleaños",
                    "polarity": 1,
                },
            ],
            "steps": [
                {"step_id": "s1", "operation": "weekday_shift", "premises": ["p_bday", "p_off"], "conclusion_id": "c1"}
            ],
            "derived_conclusion_id": "c1",
            "missing_premises": [],
            "assumptions": [],
            "suggested_rules": [],
        }

    r4 = verify_claim(src_bday, "La cita fue el domingo.", planner=_ScriptedPlanner([_bday_plan("domingo")]))
    assert r4.epistemic_verdict == "derived"
    assert r4.operational_routing.status == "PASS"
    assert r4.computed_conclusion is not None and r4.computed_conclusion["value"] == "domingo"
    assert r4.verified_steps[0]["details"]["crossed_week_boundary"] is True

    r5 = verify_claim(src_bday, "La cita fue el miércoles.", planner=_ScriptedPlanner([_bday_plan("miércoles")]))
    assert r5.epistemic_verdict == "refuted"
    assert r5.operational_routing.status == "RETRY"
    assert r5.computed_conclusion is not None and r5.computed_conclusion["value"] == "domingo"
    assert len(r5.anchor_diagnostics) == 1
    assert r5.anchor_diagnostics[0]["incorrect_anchor_weekday"] == "viernes"
    assert r5.anchor_diagnostics[0]["correct_anchor_weekday"] == "martes"


# ---------------------------------------------------------------------------
# 3. Non-Temporal Cross-Document Track + Catalog Extension
# ---------------------------------------------------------------------------

def test_nontemporal_cross_document_difference_and_extended_catalog() -> None:
    docs = {
        "doc_budget": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
        "doc_spend": "Gasto ejecutado del Proyecto Norte: 900 EUR.",
    }

    def _sub_plan(claimed_val: int, *, include_spend: bool = True) -> dict[str, Any]:
        prems = [
            {
                "claim_id": "p_budget",
                "document_id": "doc_budget",
                "quote": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
                "entity": "Proyecto Norte",
                "event": "presupuesto aprobado",
                "predicate": "presupuesto_aprobado",
                "value": 1250,
                "unit": "EUR",
                "direction": "neutral",
                "polarity": 1,
            }
        ]
        steps: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        conc_id: str | None = "c1"
        if include_spend:
            prems.append({
                "claim_id": "p_spend",
                "document_id": "doc_spend",
                "quote": "Gasto ejecutado del Proyecto Norte: 900 EUR.",
                "entity": "Proyecto Norte",
                "event": "gasto ejecutado",
                "predicate": "gasto_ejecutado",
                "value": 900,
                "unit": "EUR",
                "direction": "neutral",
                "polarity": 1,
            })
            steps.append({
                "step_id": "s_diff",
                "operation": "sub",
                "premises": ["p_budget", "p_spend"],
                "conclusion_id": "c1",
            })
        else:
            conc_id = None
            missing.append({"role": "gasto_ejecutado", "description": "No consta documento de gasto ejecutado."})

        return {
            "claim_target": {"subject": "Proyecto Norte", "predicate": "margen_restante", "value": claimed_val, "unit": "EUR", "polarity": 1},
            "premises": prems,
            "steps": steps,
            "derived_conclusion_id": conc_id,
            "missing_premises": missing,
            "assumptions": [],
            "suggested_rules": [],
        }

    # Unseen derived figure (350 EUR appears in neither document!)
    r_ok = dg_v4.verify_claim(
        docs,
        "El margen restante del Proyecto Norte es de 350 EUR.",
        planner=_ScriptedPlanner([_sub_plan(350)]),
    )
    assert r_ok.epistemic_verdict == "derived"
    assert r_ok.operational_routing.status == "PASS"
    assert r_ok.computed_conclusion is not None
    assert r_ok.computed_conclusion["value"] == 350
    assert r_ok.computed_conclusion["unit"] == "EUR"
    assert "350" not in docs["doc_budget"] and "350" not in docs["doc_spend"]

    # Wrong figure -> Refuted, returns 350 EUR
    r_bad = verify_claim(
        docs,
        "El margen restante del Proyecto Norte es de 450 EUR.",
        planner=_ScriptedPlanner([_sub_plan(450)]),
    )
    assert r_bad.epistemic_verdict == "refuted"
    assert r_bad.operational_routing.status == "RETRY"
    assert r_bad.computed_conclusion is not None and r_bad.computed_conclusion["value"] == 350

    # Missing second document -> Indeterminate
    r_miss = verify_claim(
        {"doc_budget": docs["doc_budget"]},
        "El margen restante del Proyecto Norte es de 350 EUR.",
        planner=_ScriptedPlanner([_sub_plan(350, include_spend=False)]),
    )
    assert r_miss.epistemic_verdict == "insufficient_evidence"
    assert r_miss.outcome_category == "corpus_insufficient_evidence"
    assert r_miss.operational_routing.status == "ESCALATE"

    # Extended catalog works seamlessly through the same verify_claim interface
    custom_cat = DEFAULT_OPERATION_CATALOG.extend(
        [
            OperationSpec(
                name="net_margin_diff",
                version="1.0.0",
                family="arithmetic",
                description="Compute budget minus spend.",
                inputs=({"role": "budget", "type": "quantity"}, {"role": "spend", "type": "quantity"}),
                output_type="quantity",
                executor=lambda prems, params, prof: OperationOutput(
                    ok=True,
                    value=int(prems[0].value) - int(prems[1].value),
                    unit=prems[0].unit,
                ),
            )
        ],
        catalog_id="project_catalog",
    )
    ext_plan = _sub_plan(350)
    ext_plan["steps"][0]["operation"] = "net_margin_diff"
    r_ext = verify_claim(
        docs,
        "El margen restante del Proyecto Norte es de 350 EUR.",
        planner=_ScriptedPlanner([ext_plan]),
        catalog=custom_cat,
    )
    assert r_ext.epistemic_verdict == "derived"
    assert r_ext.operational_routing.status == "PASS"
    assert r_ext.telemetry.catalog_id == "project_catalog"


# ---------------------------------------------------------------------------
# 4. Adversarial Grounding Defects, Anti-Cheating & Calibration
# ---------------------------------------------------------------------------

def test_adversarial_authentic_quote_defects_and_anti_cheating() -> None:
    src_simple = {"doc_1": "Hoy es viernes y quedé con ella dos días antes."}

    # 1. Inverted sign (+2 on "dos días antes")
    bad_sign = _temporal_plan_simple("domingo")
    bad_sign["premises"][1]["value"] = 2
    r_sign = verify_plan(src_simple, "La cita fue el domingo.", bad_sign)
    assert r_sign.epistemic_verdict == "unresolved"
    assert r_sign.outcome_category == "premise_grounding_defect"
    assert any(d["code"] == "DIRECTION_SIGN_MISMATCH" for d in r_sign.defects)

    # 2. Wrong event on authentic quote ('cumpleaños' assigned to 'viernes' on birthday source)
    src_bday = {"doc_1": "Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños."}
    bad_event = _temporal_plan_simple("miércoles")
    bad_event["premises"][0]["quote"] = "Hoy es viernes"
    bad_event["premises"][0]["event"] = "cumpleaños"
    r_ev = verify_plan(src_bday, "La cita fue el miércoles.", bad_event)
    assert r_ev.epistemic_verdict == "unresolved"
    assert any(d["code"] == "EVENT_VALUE_MISMATCH" for d in r_ev.defects)

    # 3. Wrong anchor wired into "dos días antes de su cumpleaños" even when quoting full paragraph
    bad_anchor = {
        "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": "miércoles", "unit": None, "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": src_bday["doc_1"],
                "entity": "hoy",
                "event": "hoy",
                "predicate": "dia_semana",
                "value": "viernes",
                "unit": None,
                "polarity": 1,
            },
            {
                "claim_id": "p2",
                "document_id": "doc_1",
                "quote": "quedé con ella dos días antes de su cumpleaños",
                "entity": "cita",
                "event": "cita",
                "predicate": "desplazamiento_dias",
                "value": -2,
                "unit": "calendar_days",
                "direction": "before",
                "polarity": 1,
            },
        ],
        "steps": [{"step_id": "s1", "operation": "weekday_shift", "premises": ["p1", "p2"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
    }
    r_anc = verify_plan(src_bday, "La cita fue el miércoles.", bad_anchor)
    assert r_anc.epistemic_verdict == "unresolved"
    assert r_anc.outcome_category == "premise_grounding_defect"
    assert any(d["code"] == "ANCHOR_MISMATCH" for d in r_anc.defects)

    # 4. Entity substitution & co-occurrence trap across clauses
    src_coocc = {"doc_1": "Gasto del Proyecto Norte: 900 EUR. Gasto del Proyecto Sur: 400 EUR."}
    coocc_plan = {
        "claim_target": {"subject": "Proyecto Norte", "predicate": "gasto", "value": 400, "unit": "EUR", "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": src_coocc["doc_1"],
                "entity": "Proyecto Norte",
                "event": "gasto",
                "predicate": "gasto",
                "value": 400,
                "unit": "EUR",
                "polarity": 1,
            }
        ],
        "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
    }
    r_coocc = verify_plan(src_coocc, "El gasto del Proyecto Norte fue de 400 EUR.", coocc_plan)
    assert r_coocc.epistemic_verdict == "unresolved"
    assert any(d["code"] == "ENTITY_VALUE_CLAUSE_MISMATCH" for d in r_coocc.defects)

    # 5. Non-temporal event substitution ('presupuesto aprobado' bound to 'multa máxima 200 EUR')
    src_ev_swap = {"doc_1": "Proyecto Norte: presupuesto aprobado 1.250 EUR; multa máxima 200 EUR."}
    ev_swap_plan = {
        "claim_target": {"subject": "Proyecto Norte", "predicate": "presupuesto", "value": 200, "unit": "EUR", "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": src_ev_swap["doc_1"],
                "entity": "Proyecto Norte",
                "event": "presupuesto aprobado",
                "predicate": "presupuesto",
                "value": 200,
                "unit": "EUR",
                "polarity": 1,
            }
        ],
        "steps": [{"step_id": "s1", "operation": "identity", "premises": ["p1"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
    }
    r_ev_swap = verify_plan(src_ev_swap, "El presupuesto del Proyecto Norte es de 200 EUR.", ev_swap_plan)
    assert r_ev_swap.epistemic_verdict == "unresolved"
    assert any(d["code"] == "EVENT_VALUE_MISMATCH" for d in r_ev_swap.defects)

    # 6. Missing source / fabricated quote / stale revision
    no_src = _temporal_plan_simple("miércoles")
    no_src["premises"][0]["document_id"] = ""
    assert any(d["code"] == "MISSING_SOURCE_REFERENCE" for d in verify_plan(src_simple, "La cita fue el miércoles.", no_src).defects)

    fab_q = _temporal_plan_simple("miércoles")
    fab_q["premises"][0]["quote"] = "Hoy es sábado"
    assert any(d["code"] == "QUOTE_NOT_FOUND_IN_SOURCE" for d in verify_plan(src_simple, "La cita fue el miércoles.", fab_q).defects)

    stale_rev = _temporal_plan_simple("miércoles")
    stale_rev["premises"][0]["revision"] = "stale-rev-999"
    assert any(d["code"] == "STALE_DOCUMENT_REVISION" for d in verify_plan(src_simple, "La cita fue el miércoles.", stale_rev).defects)

    # 7. Anti-self-evidence: claim used as premise & rewritten claim_target
    claim_as_prem = _temporal_plan_simple("miércoles")
    claim_as_prem["premises"][0]["document_id"] = "claim"
    claim_as_prem["premises"][0]["quote"] = "La cita fue el miércoles."
    r_self = verify_plan(src_simple, "La cita fue el miércoles.", claim_as_prem)
    assert any(d["code"] == "CLAIM_USED_AS_PREMISE" for d in r_self.defects)

    tampered_target = _temporal_plan_simple("miércoles")
    r_tamp = verify_plan(src_simple, "La cita fue el jueves.", tampered_target)
    assert r_tamp.outcome_category == "malformed_proposal"
    assert any(d["code"] == "CLAIM_TARGET_MISMATCH_WITH_CLAIM_TEXT" for d in r_tamp.defects)


def test_unknown_operation_suggested_rule_and_calibration_separation() -> None:
    src = {"doc_1": "Hoy es viernes y quedé con ella dos días antes."}

    # Unknown operation & suggested rule produce explicit limitations (never executed)
    unk_plan = _temporal_plan_simple("miércoles")
    unk_plan["steps"][0]["operation"] = "python_eval_calendar"
    unk_plan["suggested_rules"] = [{"rule_id": "custom_rule_x", "description": "Inferir día"}]
    r_lim = verify_plan(src, "La cita fue el miércoles.", unk_plan)
    assert r_lim.epistemic_verdict == "unresolved"
    assert r_lim.outcome_category == "unsupported_operation_or_rule"
    assert r_lim.operational_routing.status == "ESCALATE"
    assert any("UNKNOWN_OPERATION:python_eval_calendar" in l for l in r_lim.limitations)
    assert any("UNVERIFIED_SUGGESTED_RULE:custom_rule_x" in l for l in r_lim.limitations)

    # Undeclared rule_id in rule_application
    undec_plan = _temporal_plan_simple("miércoles")
    undec_plan["steps"][0] = {
        "step_id": "s1",
        "operation": "rule_application",
        "premises": ["p1", "p2"],
        "conclusion_id": "c1",
        "rule_id": "R_invented",
    }
    r_undec = verify_plan(src, "La cita fue el miércoles.", undec_plan)
    assert r_undec.outcome_category == "unsupported_operation_or_rule"
    assert "UNDECLARED_RULE:R_invented" in r_undec.limitations

    # Untrusted confidence stripped; calibration uses EmpiricalCalibration when configured
    noisy_plan = {**_temporal_plan_simple("miércoles"), "confidence": 0.99, "verified": True}
    r_uncal = verify_plan(src, "La cita fue el miércoles.", noisy_plan)
    assert r_uncal.certainty.calibrated_correctness_probability is None
    assert r_uncal.certainty.calibration_status == "uncalibrated"

    cal_profile = DomainProfile(
        profile_id="calibrated_test",
        calibration=EmpiricalCalibration(
            calibration_bins=((0.0, 0.5, 0.35), (0.5, 1.0, 0.91)),
            calibration_dataset_id="dev_cal_v1",
        ),
    )
    r_cal = verify_plan(src, "La cita fue el miércoles.", noisy_plan, domain_profile=cal_profile)
    assert r_cal.certainty.calibrated_correctness_probability == pytest.approx(0.91)
    assert r_cal.certainty.calibration_status == "calibrated"


def test_provider_error_malformed_json_and_retry_recovery() -> None:
    src = {"doc_1": "Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños."}

    # Provider error distinguished from corpus insufficiency
    r_prov = verify_claim(
        src,
        "La cita fue el domingo.",
        planner=_ScriptedPlanner([PlannerCallResult(ok=False, error_kind="provider_error", error_message="URLError: offline", latency_ms=5.0)]),
    )
    assert r_prov.epistemic_verdict == "unresolved"
    assert r_prov.outcome_category == "provider_error"
    assert r_prov.operational_routing.status == "ESCALATE"

    # Malformed proposal (cyclic graph)
    cyclic = _temporal_plan_simple("domingo")
    cyclic["steps"] = [
        {"step_id": "s1", "operation": "weekday_shift", "premises": ["c2", "p2"], "conclusion_id": "c1"},
        {"step_id": "s2", "operation": "identity", "premises": ["c1"], "conclusion_id": "c2"},
    ]
    r_cyc = verify_plan({"doc_1": "Hoy es viernes y quedé con ella dos días antes."}, "La cita fue el domingo.", cyclic)
    assert r_cyc.outcome_category == "malformed_proposal"
    assert r_cyc.operational_routing.status == "RETRY"

    # Bounded retry recovery: attempt 1 uses wrong anchor -> feedback -> attempt 2 succeeds
    bad_attempt = {
        "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": "domingo", "unit": None, "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": "Hoy es viernes",
                "entity": "hoy",
                "event": "hoy",
                "predicate": "dia_semana",
                "value": "viernes",
                "polarity": 1,
            },
            {
                "claim_id": "p2",
                "document_id": "doc_1",
                "quote": "quedé con ella dos días antes de su cumpleaños",
                "entity": "cita",
                "event": "cita",
                "predicate": "desplazamiento_dias",
                "value": -2,
                "unit": "calendar_days",
                "direction": "before",
                "polarity": 1,
            },
        ],
        "steps": [{"step_id": "s1", "operation": "weekday_shift", "premises": ["p1", "p2"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
    }
    good_attempt = {
        "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": "domingo", "unit": None, "polarity": 1},
        "premises": [
            {
                "claim_id": "p1",
                "document_id": "doc_1",
                "quote": "Su cumpleaños fue el martes",
                "entity": "su cumpleaños",
                "event": "cumpleaños",
                "predicate": "dia_semana",
                "value": "martes",
                "polarity": 1,
            },
            bad_attempt["premises"][1],
        ],
        "steps": [{"step_id": "s1", "operation": "weekday_shift", "premises": ["p1", "p2"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
    }
    planner = _ScriptedPlanner([bad_attempt, good_attempt])
    r_rec = verify_claim(src, "La cita fue el domingo.", planner=planner, max_retries=1)
    assert r_rec.epistemic_verdict == "derived"
    assert r_rec.operational_routing.status == "PASS"
    assert r_rec.telemetry.calls == 2
    assert r_rec.telemetry.retries == 1
    assert planner.calls[1]["repair_feedback"][0]["code"] == "ANCHOR_MISMATCH"


# ---------------------------------------------------------------------------
# 5. Public CLI (`verify-claim` & Backwards Compatibility)
# ---------------------------------------------------------------------------

def test_cli_verify_claim_offline_exit_codes_and_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_pass = tmp_path / "plan_pass.json"
    plan_pass.write_text(json.dumps(_temporal_plan_simple("miércoles"), ensure_ascii=False), encoding="utf-8")
    out_file = tmp_path / "result.json"

    rc0 = cli_main([
        "verify-claim",
        "--source", "doc_1=Hoy es viernes y quedé con ella dos días antes.",
        "--claim", "La cita fue el miércoles.",
        "--plan-json", str(plan_pass),
        "--out", str(out_file),
    ])
    assert rc0 == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["epistemic_verdict"] == "derived"
    assert captured["computed_conclusion"]["value"] == "miércoles"
    assert captured["operational_routing"]["status"] == "PASS"
    assert out_file.exists()

    plan_retry = tmp_path / "plan_retry.json"
    plan_retry.write_text(json.dumps(_temporal_plan_simple("jueves"), ensure_ascii=False), encoding="utf-8")
    rc2 = cli_main([
        "verify-claim",
        "--source", "doc_1=Hoy es viernes y quedé con ella dos días antes.",
        "--claim", "La cita fue el jueves.",
        "--plan-json", str(plan_retry),
    ])
    assert rc2 == 2
    captured2 = json.loads(capsys.readouterr().out)
    assert captured2["epistemic_verdict"] == "refuted"
    assert captured2["computed_conclusion"]["value"] == "miércoles"

    plan_esc = tmp_path / "plan_esc.json"
    plan_esc.write_text(
        json.dumps(
            {
                "claim_target": {"subject": "cita", "predicate": "dia_semana", "value": "miércoles", "polarity": 1},
                "premises": [],
                "steps": [],
                "derived_conclusion_id": None,
                "missing_premises": [{"role": "anchor_weekday", "description": "Falta el día de referencia."}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rc3 = cli_main([
        "verify-claim",
        "--source", "doc_1=Quedé con ella dos días antes.",
        "--claim", "La cita fue el miércoles.",
        "--plan-json", str(plan_esc),
    ])
    assert rc3 == 3
    captured3 = json.loads(capsys.readouterr().out)
    assert captured3["epistemic_verdict"] == "insufficient_evidence"

    # Existing CLI subcommand remains backwards-compatible
    assert cli_main(["demo"]) == 0
