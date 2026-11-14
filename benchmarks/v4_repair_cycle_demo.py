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
    LLMGraphProposalAdapter,
    LLMRepairAgent,
    OpenAICompatibleModel,
    RepairBudget,
    RepairOrchestrator,
    RuleGuidedRepairAdapter,
    review_response,
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


def run_demo(
    *,
    live_llm: bool = False,
    supplied_graph: bool = False,
    base_url: str = "http://127.0.0.1:1234/v1",
    model: str = "qwen/qwen3.5-9b",
) -> list[dict[str, Any]]:
    llm_client = None
    if live_llm:
        llm_client = OpenAICompatibleModel(base_url=base_url, model=model, api_key="lm-studio")
        repair_agent = LLMRepairAgent(llm_client)
        mode = f"live_llm:{model}" + (":supplied_graph" if supplied_graph else ":adapter_extracted_graph")
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
        graph_or_adapter = (
            LLMGraphProposalAdapter(llm_client)
            if live_llm and llm_client is not None and not supplied_graph
            else case["graph"]
        )
        cycle = orch.run(
            question=case["question"],
            initial_response=case["initial_response"],
            evidence=case["evidence"],
            graph_or_adapter=graph_or_adapter,
            repair_agent=repair_agent,
        )
        results.append(_summarize_cycle(case["case_id"], cycle, mode=mode))
    return results


def run_perturbation_benchmark() -> dict[str, Any]:
    """Evaluate transformation perturbations (adversarial vs valid paraphrases) across Case 1 and Case 2."""
    c1 = _build_case1_temporal_buffett()
    c2 = _build_case2_incomplete_aggregation_and_condition()

    matrix = [
        # Adversarial perturbations (must NOT be accepted)
        (
            "false_equation",
            c1,
            (
                "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
                "sin fechas completas no se puede determinar la edad exacta. "
                "El cálculo es 13 + (2026 − 1933) = 13."
            ),
            False,
        ),
        (
            "ungrounded_equation",
            c1,
            (
                "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026; "
                "sin fechas completas no se puede determinar la edad exacta. "
                "El cálculo es 50 + 50 = 100."
            ),
            False,
        ),
        (
            "conclusion_negation",
            c1,
            (
                "Según el texto, Warren Buffet no tendría aproximadamente 106 años en 2026; "
                "faltan fechas completas para precisar la edad."
            ),
            False,
        ),
        (
            "ungrounded_claim_addition",
            c1,
            (
                "Según el texto, Warren Buffet tendría aproximadamente 106 años en 2026 y es astronauta; "
                "faltan el día y el mes para precisar su edad."
            ),
            False,
        ),
        (
            "condition_inversion",
            c2,
            (
                "Según el texto, el subtotal parcial de las facturas listadas es 1000 USD "
                "sin constancia de completitud de población, incluso en caso de disputa activa."
            ),
            False,
        ),
        (
            "unrelated_condition",
            c2,
            (
                "Según el texto, el subtotal parcial de las facturas listadas es 1000 USD "
                "sin constancia de completitud de población, si hace sol."
            ),
            False,
        ),
        (
            "role_swap_age_year",
            c1,
            (
                "Según el texto, Warren Buffet tenía 1933 años en 13 (aproximadamente 106 años en 2026); "
                "faltan el día y el mes para precisar su edad."
            ),
            False,
        ),
        (
            "predicate_noun_reassignment_shares",
            c1,
            (
                "Según el texto, Warren Buffet compró 106 acciones en 1933 (aproximadamente 106 años en 2026); "
                "faltan el día y el mes para precisar su edad."
            ),
            False,
        ),
        (
            "entity_value_swap_invoices",
            c2,
            (
                "Según el texto, Factura A es 600 USD y Factura B es 400 USD, por lo que el subtotal parcial "
                "de las facturas listadas es 1000 USD sin constancia de completitud de población, "
                "salvo en caso de disputa activa."
            ),
            False,
        ),
        # Valid paraphrases (MUST be accepted with response_correspondence_verified=True)
        (
            "valid_paraphrase_temporal_est_and_granularity",
            c1,
            (
                "A partir del pasaje facilitado, la estimación para Warren Buffet es de unos 106 años en 2026; "
                "faltan el día y el mes para precisar su edad."
            ),
            True,
        ),
        (
            "valid_paraphrase_temporal_with_grounded_equation",
            c1,
            (
                "Conforme al fragmento aportado, la edad de Warren Buffet rondaría los 106 años en 2026 "
                "(13 + (2026 - 1933) = 106), aunque no hay fecha de nacimiento completa ni verificación histórica externa."
            ),
            True,
        ),
        (
            "valid_paraphrase_subtotal_prerequisite_neg",
            c2,
            (
                "De acuerdo con el documento proporcionado, el subtotal parcial de las facturas listadas "
                "es 1000 USD, sin constancia de completitud de toda la población, solo si no existe disputa activa."
            ),
            True,
        ),
        (
            "valid_paraphrase_subtotal_unless",
            c2,
            (
                "Según el pasaje facilitado, el subtotal parcial de las facturas listadas es 1000 USD "
                "(400 + 600 = 1000), sin constancia de completitud de población, a menos que haya disputa activa."
            ),
            True,
        ),
        (
            "valid_paraphrase_subtotal_with_correct_item_relations",
            c2,
            (
                "De acuerdo con el texto, Factura A es 400 USD y Factura B es 600 USD, por lo que el subtotal parcial "
                "de las facturas listadas es 1000 USD sin constancia de completitud de población, "
                "salvo en caso de disputa activa."
            ),
            True,
        ),
    ]

    case_rows: list[dict[str, Any]] = []
    false_accepts = 0
    false_rejects = 0
    negative_total = 0
    positive_total = 0

    for label, case, resp, expected_accept in matrix:
        rv = review_response(case["question"], resp, case["evidence"], case["graph"])
        actual_accept = bool(rv.disposition == "ACCEPT" and rv.assessment.response_correspondence_verified)
        if not expected_accept:
            negative_total += 1
            if actual_accept:
                false_accepts += 1
        else:
            positive_total += 1
            if not actual_accept:
                false_rejects += 1
        case_rows.append({
            "case": label,
            "expected_accept": expected_accept,
            "actual_accept": actual_accept,
            "disposition": rv.disposition,
            "response_correspondence_verified": rv.assessment.response_correspondence_verified,
            "issues": [i.code for i in rv.issues],
        })

    return {
        "cases": case_rows,
        "negative_count": negative_total,
        "positive_count": positive_total,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "false_accept_rate": (false_accepts / negative_total) if negative_total else 0.0,
        "false_reject_rate": (false_rejects / positive_total) if positive_total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ClaimLayer v4 evaluation & repair cycle demo")
    parser.add_argument("--live-llm", action="store_true", help="Use live OpenAI-compatible LLM endpoint")
    parser.add_argument("--supplied-graph", action="store_true", help="Keep supplied fixture graph in --live-llm mode instead of LLMGraphProposalAdapter")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--model", default="qwen/qwen3.5-9b", help="Model name for live LLM mode")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    results = run_demo(
        live_llm=args.live_llm,
        supplied_graph=args.supplied_graph,
        base_url=args.base_url,
        model=args.model,
    )
    perturb = run_perturbation_benchmark()
    if args.json:
        print(json.dumps({"cycles": results, "perturbation_benchmark": perturb}, ensure_ascii=False, indent=2))
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

    print("=== perturbation_benchmark ===")
    print(
        f"negative={perturb['negative_count']} positive={perturb['positive_count']} "
        f"false_accept_rate={perturb['false_accept_rate']:.3f} false_reject_rate={perturb['false_reject_rate']:.3f}"
    )
    for row in perturb["cases"]:
        print(
            f"  - {row['case']}: expected_accept={row['expected_accept']} actual_accept={row['actual_accept']} "
            f"disposition={row['disposition']} issues={row['issues']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
