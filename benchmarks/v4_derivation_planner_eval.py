"""Live & reproducible evaluation harness for the v4 Derivation Planner + ClaimVerifier.

Runs:
1. Core Acceptance Cases (5 temporal + 3 non-temporal cross-document).
2. Held-out Paraphrases (6 cases covering forward/backward shifts, week wraps,
   competing anchors, missing anchors, and cross-document USD arithmetic) using
   the exact same frozen `DERIVATION_PLANNER_PROMPT` and `DEFAULT_OPERATION_CATALOG`.

Usage:
    NVIDIA_API_KEY=... PYTHONPATH=claim-layer/src .venv/bin/python claim-layer/benchmarks/v4_derivation_planner_eval.py --phase all
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from claimlayer.semantic.adapters import OpenAICompatibleModel  # noqa: E402
from claimlayer.semantic.models import SourceDocument, digest  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    DEFAULT_OPERATION_CATALOG,
    DERIVATION_PLANNER_PROMPT,
    DERIVATION_PLANNER_PROMPT_VERSION,
    DERIVATION_PLANNER_PROMPT_V2,
    DERIVATION_PLANNER_PROMPT_VERSION_V2,
    ClaimDerivationPlanner,
    verify_claim,
)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_OUT_PATH = _THIS_DIR / "v4_derivation_planner_live_results.json"
PROMPT, PROMPT_VERSION = DERIVATION_PLANNER_PROMPT, DERIVATION_PLANNER_PROMPT_VERSION

PARAPHRASE_SELECTION_PROTOCOL = (
    "Pre-declared 6-case held-out paraphrase suite selected prior to held-out execution along orthogonal "
    "variation axes: (H1) forward shift +3 crossing Sunday->Tuesday boundary, (H2) forward shift with missing "
    "anchor in paraphrased wording ('tres jornadas más tarde'), (H3) backward shift with competing anchor "
    "('Ayer era miércoles' vs 'lanzamiento oficial ocurrió el lunes') crossing week boundary to Saturday, "
    "(H4) refuted variant of H3 where distractor anchor minus 2 yields claimed Monday, (H5) refuted backward "
    "shift on Monday ('hoy es lunes' -> expected Saturday vs claimed Friday), and (H6) non-temporal cross-document "
    "USD margin subtraction (2,400 - 1,650 = 750 unseen in sources)."
)

CORE_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "T1_simple_derived_wednesday",
        "track": "temporal_core",
        "sources": {"doc_1": "Hoy es viernes y quedé con ella dos días antes."},
        "claim": "La cita fue el miércoles.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": "miércoles",
        "expected_computed_unit": None,
    },
    {
        "case_id": "T2_simple_refuted_thursday",
        "track": "temporal_core",
        "sources": {"doc_1": "Hoy es viernes y quedé con ella dos días antes."},
        "claim": "La cita fue el jueves.",
        "expected_epistemic": "refuted",
        "expected_routing": "RETRY",
        "expected_computed_value": "miércoles",
        "expected_computed_unit": None,
    },
    {
        "case_id": "T3_missing_anchor_indeterminate",
        "track": "temporal_core",
        "sources": {"doc_1": "Quedé con ella dos días antes."},
        "claim": "La cita fue el miércoles.",
        "expected_epistemic": "insufficient_evidence",
        "expected_routing": "ESCALATE",
        "expected_computed_value": None,
        "expected_computed_unit": None,
    },
    {
        "case_id": "T4_birthday_anchor_derived_sunday",
        "track": "temporal_core",
        "sources": {"doc_1": "Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños."},
        "claim": "La cita fue el domingo.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": "domingo",
        "expected_computed_unit": None,
    },
    {
        "case_id": "T5_birthday_anchor_refuted_wednesday",
        "track": "temporal_core",
        "sources": {"doc_1": "Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños."},
        "claim": "La cita fue el miércoles.",
        "expected_epistemic": "refuted",
        "expected_routing": "RETRY",
        "expected_computed_value": "domingo",
        "expected_computed_unit": None,
    },
    {
        "case_id": "NT1_cross_doc_sub_derived_350",
        "track": "nontemporal_core",
        "sources": {
            "doc_budget": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
            "doc_spend": "Gasto ejecutado del Proyecto Norte: 900 EUR.",
        },
        "claim": "El margen restante del Proyecto Norte es de 350 EUR.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": 350,
        "expected_computed_unit": "EUR",
    },
    {
        "case_id": "NT2_cross_doc_sub_refuted_450",
        "track": "nontemporal_core",
        "sources": {
            "doc_budget": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
            "doc_spend": "Gasto ejecutado del Proyecto Norte: 900 EUR.",
        },
        "claim": "El margen restante del Proyecto Norte es de 450 EUR.",
        "expected_epistemic": "refuted",
        "expected_routing": "RETRY",
        "expected_computed_value": 350,
        "expected_computed_unit": "EUR",
    },
    {
        "case_id": "NT3_cross_doc_sub_missing_spend",
        "track": "nontemporal_core",
        "sources": {
            "doc_budget": "Presupuesto aprobado para el Proyecto Norte: 1.250 EUR.",
        },
        "claim": "El margen restante del Proyecto Norte es de 350 EUR.",
        "expected_epistemic": "insufficient_evidence",
        "expected_routing": "ESCALATE",
        "expected_computed_value": None,
        "expected_computed_unit": None,
    },
)

HELDOUT_PARAPHRASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "H1_forward_shift_week_wrap_derived",
        "track": "heldout_paraphrase",
        "sources": {"doc_1": "El informe se entregó el sábado y la reunión se celebró tres días después."},
        "claim": "La reunión tuvo lugar el martes.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": "martes",
        "expected_computed_unit": None,
    },
    {
        "case_id": "H2_forward_missing_anchor_indeterminate",
        "track": "heldout_paraphrase",
        "sources": {"doc_1": "Nos vimos tres jornadas más tarde."},
        "claim": "El encuentro ocurrió el lunes.",
        "expected_epistemic": "insufficient_evidence",
        "expected_routing": "ESCALATE",
        "expected_computed_value": None,
        "expected_computed_unit": None,
    },
    {
        "case_id": "H3_competing_anchor_paraphrase_derived",
        "track": "heldout_paraphrase",
        "sources": {"doc_1": "Ayer era miércoles, aunque el lanzamiento oficial ocurrió el lunes; firmamos el acuerdo dos días antes del lanzamiento oficial."},
        "claim": "El acuerdo se firmó el sábado.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": "sábado",
        "expected_computed_unit": None,
    },
    {
        "case_id": "H4_competing_anchor_paraphrase_refuted",
        "track": "heldout_paraphrase",
        "sources": {"doc_1": "Ayer era miércoles, aunque el lanzamiento oficial ocurrió el lunes; firmamos el acuerdo dos días antes del lanzamiento oficial."},
        "claim": "El acuerdo se firmó el lunes.",
        "expected_epistemic": "refuted",
        "expected_routing": "RETRY",
        "expected_computed_value": "sábado",
        "expected_computed_unit": None,
    },
    {
        "case_id": "H5_monday_backward_refuted",
        "track": "heldout_paraphrase",
        "sources": {"doc_1": "Hoy es lunes y la entrega venció dos días antes."},
        "claim": "La entrega venció el viernes.",
        "expected_epistemic": "refuted",
        "expected_routing": "RETRY",
        "expected_computed_value": "sábado",
        "expected_computed_unit": None,
    },
    {
        "case_id": "H6_cross_doc_usd_paraphrase_derived",
        "track": "heldout_paraphrase",
        "sources": {
            "contrato": "Techo presupuestario de la Fase Beta: 2,400 USD.",
            "factura": "Importe facturado en la Fase Beta: 1,650 USD.",
        },
        "claim": "Quedan 750 USD disponibles en la Fase Beta.",
        "expected_epistemic": "derived",
        "expected_routing": "PASS",
        "expected_computed_value": 750,
        "expected_computed_unit": "USD",
    },
)


def _load_env_fallback() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return key
    for candidate in (_REPO_ROOT.parent / ".env", _REPO_ROOT / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#") and s.startswith("NVIDIA_API_KEY="):
                    val = s.split("=", 1)[1].strip().strip("\"'")
                    if val:
                        return val
    return ""


def _case_passed(case: Mapping[str, Any], res_dict: Mapping[str, Any]) -> bool:
    if res_dict["epistemic_verdict"] != case["expected_epistemic"]:
        return False
    if res_dict["operational_routing"]["status"] != case["expected_routing"]:
        return False
    exp_val = case["expected_computed_value"]
    comp = res_dict.get("computed_conclusion")
    if exp_val is None:
        return comp is None
    if comp is None or comp.get("value") != exp_val:
        return False
    if case["expected_computed_unit"] is not None and comp.get("unit") != case["expected_computed_unit"]:
        return False
    return True


def _run_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    model: OpenAICompatibleModel,
    max_retries: int,
    interval_sec: float,
    prompt: str = DERIVATION_PLANNER_PROMPT,
    prompt_version: str = DERIVATION_PLANNER_PROMPT_VERSION,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        if idx > 0 and interval_sec > 0:
            time.sleep(interval_sec)
        docs = {
            did: SourceDocument(document_id=did, text=txt, authority_rank=80, weight=1.0)
            for did, txt in case["sources"].items()
        }
        res = verify_claim(
            docs,
            case["claim"],
            planner=ClaimDerivationPlanner(model=model, prompt=prompt, prompt_version=prompt_version),
            catalog=DEFAULT_OPERATION_CATALOG,
            max_retries=max_retries,
        )
        res_d = res.to_dict()
        passed = _case_passed(case, res_d)
        rows.append({
            "case_id": case["case_id"],
            "track": case["track"],
            "sources": dict(case["sources"]),
            "claim": case["claim"],
            "expected": {
                "epistemic_verdict": case["expected_epistemic"],
                "operational_routing": case["expected_routing"],
                "computed_value": case["expected_computed_value"],
                "computed_unit": case["expected_computed_unit"],
            },
            "observed": {
                "epistemic_verdict": res_d["epistemic_verdict"],
                "outcome_category": res_d["outcome_category"],
                "operational_routing": res_d["operational_routing"],
                "computed_conclusion": res_d["computed_conclusion"],
                "graph_admitted": res_d["graph_admitted"],
                "certainty": res_d["certainty"],
                "defects": res_d["defects"],
                "limitations": res_d["limitations"],
                "missing_premises": res_d["missing_premises"],
                "anchor_diagnostics": res_d["anchor_diagnostics"],
                "explanation": res_d["explanation"],
            },
            "passed": passed,
            "raw_proposal": res_d["raw_proposal"],
            "graph": res_d["graph"],
            "telemetry": res_d["telemetry"],
        })
        print(
            f"[{case['case_id']}] pass={passed} epistemic={res_d['epistemic_verdict']} "
            f"routing={res_d['operational_routing']['status']} "
            f"computed={(res_d['computed_conclusion'] or {}).get('value')} "
            f"calls={res_d['telemetry']['calls']} tokens={res_d['telemetry']['total_tokens']} "
            f"total_ms={res_d['telemetry']['total_latency_ms']}"
        )
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    calls = sum(int(r["telemetry"]["calls"]) for r in rows)
    retries = sum(int(r["telemetry"]["retries"]) for r in rows)
    prompt_toks = sum(int(r["telemetry"]["prompt_tokens"]) for r in rows)
    comp_toks = sum(int(r["telemetry"]["completion_tokens"]) for r in rows)
    total_toks = sum(int(r["telemetry"]["total_tokens"]) for r in rows)
    plan_ms = [float(r["telemetry"]["planning_latency_ms"]) for r in rows]
    ver_ms = [float(r["telemetry"]["verifier_latency_ms"]) for r in rows]
    tot_ms = [float(r["telemetry"]["total_latency_ms"]) for r in rows]
    sum_tot_sec = sum(tot_ms) / 1000.0 if tot_ms else 0.0
    claims_per_min = round((passed / sum_tot_sec) * 60.0, 2) if sum_tot_sec > 0 else 0.0
    false_accepts = sum(
        1
        for r in rows
        if r["expected"]["epistemic_verdict"] != "derived" and r["observed"]["epistemic_verdict"] == "derived"
    )
    provider_errs = sum(1 for r in rows if r["observed"]["outcome_category"] == "provider_error")
    return {
        "total_cases": total,
        "passed_cases": passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "false_accepts": false_accepts,
        "provider_errors": provider_errs,
        "calls": calls,
        "retries": retries,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "total_tokens": total_toks,
        "planning_latency_mean_ms": round(statistics.mean(plan_ms), 2) if plan_ms else 0.0,
        "verifier_latency_mean_ms": round(statistics.mean(ver_ms), 3) if ver_ms else 0.0,
        "total_latency_mean_ms": round(statistics.mean(tot_ms), 2) if tot_ms else 0.0,
        "total_latency_p50_ms": round(statistics.median(tot_ms), 2) if tot_ms else 0.0,
        "total_latency_p95_ms": round(sorted(tot_ms)[min(len(tot_ms) - 1, int(0.95 * len(tot_ms)))], 2) if tot_ms else 0.0,
        "total_wall_ms": round(sum(tot_ms), 2),
        "verified_claims_per_minute": claims_per_min,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate live v4 derivation planner on core & held-out cases.")
    ap.add_argument("--phase", choices=("core", "heldout", "all"), default="all")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--interval-sec", type=float, default=0.4)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    ap.add_argument("--prompt", choices=("v2", "v3"), default="v3",
                    help="v2 reproduces the recorded 14/14 evidence; v3 allows planner-proposed operations")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()
    global PROMPT, PROMPT_VERSION
    PROMPT, PROMPT_VERSION = ((DERIVATION_PLANNER_PROMPT_V2, DERIVATION_PLANNER_PROMPT_VERSION_V2) if args.prompt == "v2"
                              else (DERIVATION_PLANNER_PROMPT, DERIVATION_PLANNER_PROMPT_VERSION))

    api_key = _load_env_fallback()
    if not api_key:
        raise SystemExit("ERROR: NVIDIA_API_KEY is required in environment or .env.")

    model = OpenAICompatibleModel(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        timeout=float(args.timeout),
        max_tokens=int(args.max_tokens),
        retries=2,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    existing: dict[str, Any] = {}
    if args.out.exists():
        try:
            existing = json.loads(args.out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    core_rows = existing.get("core_cases") or []
    heldout_rows = existing.get("heldout_paraphrases") or []

    if args.phase in ("core", "all"):
        print("=== Running Core Acceptance Cases (8) ===")
        core_rows = _run_cases(CORE_CASES, model=model, max_retries=args.max_retries, interval_sec=args.interval_sec, prompt=PROMPT, prompt_version=PROMPT_VERSION)

    if args.phase in ("heldout", "all"):
        print("\n=== Running Frozen-Prompt Held-Out Paraphrases (6) ===")
        heldout_rows = _run_cases(HELDOUT_PARAPHRASES, model=model, max_retries=args.max_retries, interval_sec=args.interval_sec, prompt=PROMPT, prompt_version=PROMPT_VERSION)

    all_rows = list(core_rows) + list(heldout_rows)
    artifact = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "provider": model.provider,
            "base_url": model.base_url,
            "model": model.model,
            "temperature": 0,
            "max_tokens": model.max_tokens,
            "retries": model.retries,
            "extra_body": dict(model.extra_body),
            "planner_max_retries": args.max_retries,
        },
        "prompt_version": PROMPT_VERSION,
        "prompt_digest": digest(PROMPT),
        "catalog": {
            "catalog_id": DEFAULT_OPERATION_CATALOG.catalog_id,
            "version": DEFAULT_OPERATION_CATALOG.version,
            "fingerprint": DEFAULT_OPERATION_CATALOG.fingerprint(),
            "operations": list(DEFAULT_OPERATION_CATALOG.names()),
        },
        "paraphrase_selection_protocol": PARAPHRASE_SELECTION_PROTOCOL,
        "summary_core": _summarize_rows(core_rows),
        "summary_heldout": _summarize_rows(heldout_rows),
        "summary_total": _summarize_rows(all_rows),
        "core_cases": core_rows,
        "heldout_paraphrases": heldout_rows,
    }
    args.out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Summary ===")
    print(json.dumps({
        "prompt_digest": artifact["prompt_digest"],
        "summary_core": artifact["summary_core"],
        "summary_heldout": artifact["summary_heldout"],
        "summary_total": artifact["summary_total"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
