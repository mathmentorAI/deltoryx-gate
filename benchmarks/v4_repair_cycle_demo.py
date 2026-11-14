"""Executable Demo & Benchmark for ClaimLayer v4 Evaluation, Repair & Reverification Cycle.

Demonstrates:
1. Case 1 (Temporal + Source-Conditional Narrative):
   - Evidence: "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933."
   - Question: "¿Cuántos años tiene si estamos en 2026?"
   - Initial response: "Tiene 106 años." -> RETRY with specific obligations -> REVERIFY ->
     exploratory ACCEPT (with incremental cache reuse) while operational authorize() remains ESCALATE.
2. Case 2 (Incomplete Invoice Aggregation Subtotal + Condition Preservation):
   - Evidence: "Factura A es 400 USD y Factura B es 600 USD, salvo disputa activa."
   - Question: "¿Cuál es el total de facturas a pagar?"
   - Initial response: "El total completo definitivo es 1000 USD." -> RETRY -> REVERIFY ->
     exploratory ACCEPT while operational authorize() remains ESCALATE.

Supports:
- Default mode: deterministic `RuleGuidedRepairAdapter` (no LLM needed).
- `--live-llm`: live OpenAI-compatible model endpoint (e.g. local LM Studio) via `OpenAICompatibleModel`.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    LLMRepairAgent,
    OpenAICompatibleModel,
    RepairBudget,
    RepairOrchestrator,
    RuleGuidedRepairAdapter,
)


def _doc(doc_id: str, text: str, *, weight: float = 0.95) -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        authority_rank=85,
        weight=weight,
    )


def _ref(doc: SourceDocument, quote: str) -> EvidenceRef:
    start = doc.text.index(quote)
    return EvidenceRef(
        document_id=doc.document_id,
        revision=doc.revision,
        start=start,
        end=start + len(quote),
        quote=quote,
    )


def _build_case1_temporal_buffett() -> dict[str, Any]:
    doc = _doc(
        "d_buffett",
        "Warren Buffet compró su primera acción cuando tenía 13 años y era 1933.",
    )
    p_age = ClaimNode(
        claim_id="p_age",
        kind="extraction",
        subject="Warren Buffet",
        predicate="anchor_age",
        value=13,
        unit="years",
        evidence=(_ref(doc, "tenía 13 años"),),
        structural_reliability=0.95,
    )
    p_yr = ClaimNode(
        claim_id="p_yr",
        kind="extraction",
        subject="Warren Buffet",
        predicate="anchor_year",
        value="1933",
        evidence=(_ref(doc, "era 1933"),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="conc_age",
        kind="derivation",
        subject="Warren Buffet",
        predicate="age",
        value=106,
        unit="years",
        approximate=False,
    )
    step = InferenceStep(
        step_id="s_age",
        operation="temporal_age",
        premises=("p_age", "p_yr"),
        conclusion_id="conc_age",
        parameters={"anchor_age": 13, "anchor_date": "1933", "reference_date": "2026"},
    )
    graph = JustificationGraph(
        claims={"p_age": p_age, "p_yr": p_yr, "conc_age": conc},
        steps=(step,),
        conclusion_id="conc_age",
    )
    return {
        "case_id": "temporal_incomplete_dates",
        "question": "¿Cuántos años tiene si estamos en 2026?",
        "initial_response": "Tiene 106 años.",
        "evidence": {"d_buffett": doc},
        "graph": graph,
        "operation_id": "certify_exact_age",
    }


def _build_case2_incomplete_aggregation_and_condition() -> dict[str, Any]:
    doc = _doc(
        "d_inv",
        "Factura A es 400 USD y Factura B es 600 USD, salvo en caso de disputa activa.",
    )
    i1 = ClaimNode(
        claim_id="inv_a",
        kind="extraction",
        subject="Factura A",
        predicate="charge",
        value=400,
        unit="USD",
        event_id="evt-a",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "Factura A es 400 USD"),),
        structural_reliability=0.95,
    )
    i2 = ClaimNode(
        claim_id="inv_b",
        kind="extraction",
        subject="Factura B",
        predicate="charge",
        value=600,
        unit="USD",
        event_id="evt-b",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
        evidence=(_ref(doc, "Factura B es 600 USD"),),
        structural_reliability=0.95,
    )
    conc = ClaimNode(
        claim_id="tot_inv",
        kind="derivation",
        subject="Facturas",
        predicate="total_amount",
        value=1000,
        unit="USD",
        conditions=({"field": "exception", "value": "salvo en caso de disputa activa"},),
        condition_status="conditional",
    )
    step = InferenceStep(
        step_id="s_agg",
        operation="agg_sum",
        premises=("inv_a", "inv_b"),
        conclusion_id="tot_inv",
    )
    graph = JustificationGraph(
        claims={"inv_a": i1, "inv_b": i2, "tot_inv": conc},
        steps=(step,),
        conclusion_id="tot_inv",
    )
    return {
        "case_id": "incomplete_subtotal_and_condition",
        "question": "¿Cuál es el total de facturas?",
        "initial_response": "El total completo definitivo de las facturas es 1000 USD.",
        "evidence": {"d_inv": doc},
        "graph": graph,
        "operation_id": "approve_invoice_payout",
    }


def _summarize_cycle(case_id: str, cycle: Any, *, mode: str) -> dict[str, Any]:
    attempts = []
    for att in cycle.attempts:
        attempts.append({
            "iteration": att.iteration,
            "disposition": att.review.disposition,
            "response": att.candidate_response,
            "issues": [i.code for i in att.review.issues],
            "required_repairs": [r.issue_code for r in att.review.required_repairs],
            "cache_hits": att.cache_metrics.get("cache_hits", 0),
            "cache_misses": att.cache_metrics.get("cache_misses", 0),
            "reused_nodes": att.cache_metrics.get("reused_nodes", 0),
            "recomputed_nodes": att.cache_metrics.get("recomputed_nodes", 0),
        })
    auth = cycle.operational_authorization
    return {
        "case_id": case_id,
        "mode": mode,
        "accepted": cycle.accepted,
        "final_disposition": cycle.final_disposition,
        "stop_reason": cycle.stop_reason,
        "iterations_used": cycle.iterations_used,
        "final_response": cycle.final_response,
        "critical_coverage": cycle.final_review.assessment.critical_coverage,
        "obligation_generator_version": cycle.final_review.assessment.obligation_generator_version,
        "operational_authorization": (
            {
                "status": auth.status,
                "allowed": auth.allowed,
                "issues": [i.code for i in auth.issues],
            }
            if auth is not None
            else None
        ),
        "attempts": attempts,
    }


def run_demo(*, live_llm: bool = False, base_url: str = "http://127.0.0.1:1234/v1", model: str = "qwen/qwen3.5-9b") -> list[dict[str, Any]]:
    if live_llm:
        repair_agent = LLMRepairAgent(
            OpenAICompatibleModel(base_url=base_url, model=model, api_key="lm-studio")
        )
        mode = f"live_llm:{model}"
    else:
        repair_agent = RuleGuidedRepairAdapter()
        mode = "deterministic_rule_guided"

    results = []
    for builder in (_build_case1_temporal_buffett, _build_case2_incomplete_aggregation_and_condition):
        case = builder()
        orch = RepairOrchestrator(
            budget=RepairBudget(max_iterations=3),
            authorize_operation_id=case["operation_id"],
        )
        cycle = orch.run(
            question=case["question"],
            initial_response=case["initial_response"],
            evidence=case["evidence"],
            graph_or_adapter=case["graph"],
            repair_agent=repair_agent,
        )
        results.append(_summarize_cycle(case["case_id"], cycle, mode=mode))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="ClaimLayer v4 evaluation & repair cycle demo")
    parser.add_argument("--live-llm", action="store_true", help="Use live OpenAI-compatible LLM endpoint")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="qwen/qwen3.5-9b", help="Model name for live LLM mode")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    results = run_demo(live_llm=args.live_llm, base_url=args.base_url, model=args.model)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    for item in results:
        print(f"=== {item['case_id']} ({item['mode']}) ===")
        print(f"final_disposition={item['final_disposition']} accepted={item['accepted']} stop_reason={item['stop_reason']} iterations={item['iterations_used']}")
        print(f"operational_authorization={item['operational_authorization']}")
        for att in item["attempts"]:
            print(
                f"  [iter {att['iteration']}] {att['disposition']} issues={att['issues']} "
                f"cache_hits={att['cache_hits']} reused_nodes={att['reused_nodes']}"
            )
            print(f"    response: {att['response']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
