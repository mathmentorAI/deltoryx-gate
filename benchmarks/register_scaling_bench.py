"""Scaling benchmark for ``build_corpus_reference_register`` (Priority 1, Requirement #4).

Measures wall-clock latency across identical-document x2 dense workloads of
50, 100, 200, 400, 1000, and 2000 admitted units, computes the empirical log-log
scaling exponent, and verifies linear/near-linear scaling (exponent <= 1.3,
2000 units < 60s).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from claimlayer.semantic.models import SourceDocument  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    EvidenceLayer,
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_VOCABULARY,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domains.securitization import SECURITIZATION_DOMAIN_PROFILE  # noqa: E402
from claimlayer.v4.reader import _bind_raw_proposals, _build_document_fragments  # noqa: E402


def build_dense_identical_layer(n_units: int, *, shared_entity: bool = False, include_divergence: bool = False) -> tuple[EvidenceLayer, Any]:
    if n_units % 2 != 0 or n_units <= 0:
        raise ValueError("n_units must be a positive even integer")
    n_per_doc = n_units // 2
    lines_fixed: list[str] = []
    for i in range(n_per_doc):
        rate = 5.0 if shared_entity else round(5.0 + (i % 40) * 0.05, 2)
        # 59 chars + '\n' = 60 chars/line so 30 lines align cleanly on 1800-char fragments
        s = f"Tranche {i:04d} Notes bear interest at {rate:.2f}% per annum"
        s = f"{s} ({i % 9 + 1:01d})."
        extra = 59 - len(s)
        if extra > 0:
            s = s[:-1] + " " + ("n" * max(0, extra - 1)) + "."
        lines_fixed.append(s)
    lines2 = list(lines_fixed)
    if include_divergence and n_per_doc > 0:
        s0 = "Tranche 0000 Notes bear interest at 5.99% per annum (1)."
        extra0 = 59 - len(s0)
        if extra0 > 0:
            s0 = s0[:-1] + " " + ("n" * max(0, extra0 - 1)) + "."
        lines2[0] = s0

    t1 = "\n".join(lines_fixed)
    t2 = "\n".join(lines2)
    d1 = SourceDocument(document_id="doc_a.docx", text=t1, authority_rank=100, attribution="controlling_instrument")
    d2 = SourceDocument(document_id="doc_b.docx", text=t2, authority_rank=80, attribution="source_statement")

    class _DenseBenchmarkReader:
        identity = {"provider": "bench", "model": f"dense_{n_units}_{shared_entity}_{include_divergence}", "prompt_version": "t", "prompt_digest": "d"}

        def propose(self, req: dict[str, Any]) -> dict[str, Any]:
            doc = req["document"]
            frags = _build_document_fragments(doc, 1800)
            frag_map = dict(frags)
            raw: list[dict[str, Any]] = []
            seen_q: set[str] = set()
            for fid, fobj in frags:
                for line in fobj["quote"].split("\n"):
                    line_s = line.strip()
                    if not (line_s.startswith("Tranche ") and line_s.endswith(".")) or line_s in seen_q:
                        continue
                    seen_q.add(line_s)
                    parts = line_s.split()
                    idx = int(parts[1])
                    pct_tok = parts[6]
                    pct = float(pct_tok.rstrip("%"))
                    ent = "Class Notes" if shared_entity else f"Tranche {idx:04d} Notes"
                    fe_ent = f"Tranche {idx:04d} Notes" if not shared_entity else "Notes"
                    raw.append({
                        "entity": ent,
                        "predicate": "coupon_rate",
                        "value": {"value": pct, "unit": "percent_per_annum"},
                        "assertion_kind": "fact",
                        "conditions": [],
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [fid],
                        "quote": line_s,
                        "field_evidence": {"entity": fe_ent, "value": f"{pct_tok} per annum"},
                    })
            items: list[Any] = []
            rej: list[Any] = []
            _bind_raw_proposals(raw, frag_map, doc["text"], items, rej)
            return {"items": items, "rejected": rej, "calls": [], "errors": [], "fragments": len(frags), "windows": 1}

    layer = EvidenceLayer(vocabulary=SECURITIZATION_VOCABULARY, policy=SECURITIZATION_ACCEPTANCE_POLICY, reader=_DenseBenchmarkReader())
    layer.ingest([d1, d2])
    div_coll = collect_reference_divergences([d1, d2], domain_profile=SECURITIZATION_DOMAIN_PROFILE) if include_divergence else None
    return layer, div_coll


def ols_log_log_slope(points: list[tuple[int, float]]) -> float:
    xs = [math.log(max(1, n)) for n, _ in points]
    ys = [math.log(max(1e-6, dt)) for _, dt in points]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


def run_scaling_suite(
    sizes: tuple[int, ...] = (50, 100, 200, 400, 1000, 2000),
    *,
    include_divergence: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    pts: list[tuple[int, float]] = []
    for n in sizes:
        layer, div_coll = build_dense_identical_layer(n, include_divergence=include_divergence)
        t0 = time.perf_counter()
        reg = build_corpus_reference_register(layer, SECURITIZATION_VOCABULARY, divergence_collection=div_coll)
        dt = time.perf_counter() - t0
        actual_u = len(layer.units)
        assert actual_u == n, f"Expected {n} admitted units, got {actual_u}"
        pts.append((actual_u, dt))
        rows.append({
            "target_units": n,
            "actual_units": actual_u,
            "elapsed_seconds": round(dt, 4),
            "obligations": len(reg.obligations),
            "gaps": len(reg.gap_divergences),
        })
    slope = ols_log_log_slope(pts)
    return {
        "sizes": list(sizes),
        "include_divergence": include_divergence,
        "rows": rows,
        "ols_log_log_exponent": round(slope, 4),
        "max_elapsed_seconds": round(max(r["elapsed_seconds"] for r in rows), 4),
    }


def main() -> int:
    rep = run_scaling_suite()
    print(json.dumps(rep, indent=2))
    assert rep["ols_log_log_exponent"] <= 1.3, f"Exponent too high: {rep['ols_log_log_exponent']}"
    assert rep["rows"][-1]["elapsed_seconds"] < 60.0, f"2000-unit runtime exceeded 60s: {rep['rows'][-1]}"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
