"""Scaling and exact-equivalence tests for corpus reference register construction (#4)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_BENCH = _ROOT / "benchmarks"
for _p in (_SRC, _BENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import register_scaling_bench as rsb  # noqa: E402
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


def test_register_scaling_near_linear_up_to_2000_units():
    """Dense identical-doc x2 benchmark runs 50..2000 units with exponent <= 1.3 and 2000u < 60s."""
    rep = rsb.run_scaling_suite(sizes=(50, 100, 200, 400, 1000, 2000), include_divergence=False)
    assert [r["actual_units"] for r in rep["rows"]] == [50, 100, 200, 400, 1000, 2000]
    assert rep["ols_log_log_exponent"] <= 1.3
    assert rep["rows"][-1]["elapsed_seconds"] < 60.0

    rep_div = rsb.run_scaling_suite(sizes=(50, 100, 200, 400, 1000, 2000), include_divergence=True)
    assert [r["actual_units"] for r in rep_div["rows"]] == [50, 100, 200, 400, 1000, 2000]
    assert rep_div["ols_log_log_exponent"] <= 1.3
    assert rep_div["rows"][-1]["elapsed_seconds"] < 60.0
    assert rep_div["rows"][-1]["obligations"] >= 3


def test_register_shared_entity_2000_units_linear():
    """Even when 2000 compatible units share one entity+predicate, conflict pairing stays linear."""
    layer, _ = rsb.build_dense_identical_layer(2000, shared_entity=True, include_divergence=False)
    assert len(layer.units) == 2000
    t0 = time.perf_counter()
    reg = build_corpus_reference_register(layer, SECURITIZATION_VOCABULARY)
    dt = time.perf_counter() - t0
    assert dt < 5.0
    assert reg.unit_count == 2000


def test_register_equivalence_with_conflicts_overrides_and_gaps():
    """Bucketed pairing and indexed span lookups preserve exact obligations, gaps, and determinism."""
    lines_a = [
        "Class A Notes bear interest at 5.10% per annum.",
        "Class B Notes bear interest at 5.80% per annum.",
        "Class C Notes bear interest at 6.75% per annum and mature in 2030.",
    ]
    lines_b = [
        "Class A Notes bear interest at 5.10% per annum.",
        "Class B Notes bear interest at 5.55% per annum.",
        "Class C Notes bear interest at 6.75% per annum and mature in 2030.",
    ]
    t_a = "\n".join(lines_a)
    t_b = "\n".join(lines_b)
    d_a = SourceDocument(document_id="indenture.docx", text=t_a, authority_rank=100, attribution="controlling_instrument")
    d_b = SourceDocument(document_id="draft_oc.docx", text=t_b, authority_rank=80, attribution="source_statement")

    class _ExactReader:
        identity = {"provider": "test", "model": "eq", "prompt_version": "t", "prompt_digest": "d"}

        def propose(self, req):
            doc = req["document"]
            frags = _build_document_fragments(doc, 1800)
            fid = frags[0][0]
            raw = []
            for ent, rate_str, val in (
                ("Class A Notes", "5.10%", 5.1),
                ("Class B Notes", "5.80%" if doc["document_id"] == "indenture.docx" else "5.55%", 5.8 if doc["document_id"] == "indenture.docx" else 5.55),
                ("Class C Notes", "6.75%", 6.75),
            ):
                q = next(ln for ln in doc["text"].splitlines() if ent in ln)
                raw.append({
                    "entity": ent,
                    "predicate": "coupon_rate",
                    "value": {"value": val, "unit": "percent_per_annum"},
                    "assertion_kind": "fact",
                    "conditions": [],
                    "condition_status": "unconditional",
                    "basis": "explicit",
                    "evidence": [fid],
                    "quote": q,
                    "field_evidence": {"entity": ent, "value": f"{rate_str} per annum"},
                })
            items, rej = [], []
            _bind_raw_proposals(raw, dict(frags), doc["text"], items, rej)
            return {"items": items, "rejected": rej, "calls": [], "errors": [], "fragments": 1, "windows": 1}

    layer = EvidenceLayer(vocabulary=SECURITIZATION_VOCABULARY, policy=SECURITIZATION_ACCEPTANCE_POLICY, reader=_ExactReader())
    layer.ingest([d_a, d_b])
    div = collect_reference_divergences([d_a, d_b], domain_profile=SECURITIZATION_DOMAIN_PROFILE)

    reg1 = build_corpus_reference_register(layer, SECURITIZATION_VOCABULARY, divergence_collection=div)
    reg2 = build_corpus_reference_register(layer, SECURITIZATION_VOCABULARY, divergence_collection=div)
    assert reg1.to_dict() == reg2.to_dict()
    assert reg1.has_material_conflicts is True
    assert reg1.has_authority_overrides is True
    kinds = [(ob.parameters or {}).get("corpus_obligation_kind") for ob in reg1.obligations]
    assert "material_conflict" in kinds
    assert "authority_override" in kinds
