"""Linear O(N) scaling for cardinality='multi' and fail-closed ESCALATE on internal errors."""
from __future__ import annotations

import sqlite3
import time
import tracemalloc

from claimlayer.v4 import (
    EvidenceLayer,
    EvidenceRef,
    FieldSpec,
    LLMProposalReader,
    PredicateSpec,
    SourceDocument,
    Vocabulary,
)
from claimlayer.v4.triage import verify_draft


def _build_multi_layer(n: int, *, add_negated_conflict: bool = False) -> EvidenceLayer:
    vocab = Vocabulary("multi_bench", "1", (PredicateSpec("item", FieldSpec("string"), cardinality="multi"),))
    lines = [f"Acme Corp has item val_{i:05d}." for i in range(n)]
    text = "\n".join(lines)
    doc = SourceDocument("d_aff", text, authority_rank=100, weight=1.0)
    layer = EvidenceLayer(None, vocabulary=vocab)
    layer.ingest(doc)
    items = []
    pos = 0
    for i, ln in enumerate(lines):
        items.append(
            {
                "proposal": {
                    "entity": "Acme Corp",
                    "predicate": "item",
                    "value": f"val_{i:05d}",
                    "assertion_kind": "fact",
                },
                "evidence": [EvidenceRef(doc.document_id, doc.revision, pos, pos + len(ln), ln)],
            }
        )
        pos += len(ln) + 1
    layer.ingest_units(items, attestor="app")

    if add_negated_conflict:
        neg_text = "Acme Corp does not have item val_00042."
        neg_doc = SourceDocument("d_neg", neg_text, authority_rank=100, weight=1.0)
        layer.ingest(neg_doc)
        layer.ingest_units(
            [
                {
                    "proposal": {
                        "entity": "Acme Corp",
                        "predicate": "item",
                        "value": "val_00042",
                        "polarity": "negated",
                        "assertion_kind": "fact",
                    },
                    "evidence": [EvidenceRef(neg_doc.document_id, neg_doc.revision, 0, len(neg_text), neg_text)],
                }
            ],
            attestor="app",
        )
    return layer


def test_multi_cardinality_2000_values_linear_time_and_memory():
    times: dict[int, float] = {}
    peaks_bytes: dict[int, int] = {}

    for n in (500, 1000, 2000):
        layer_t = _build_multi_layer(n)
        t0 = time.perf_counter()
        res_t = layer_t.ask({"entity": "Acme Corp", "predicate": "item", "operation": "list"}, validity="ignore")
        times[n] = time.perf_counter() - t0
        assert res_t["status"] == "answered"
        assert len(res_t["results"]) == n
        assert all(r["competitors"] == [] for r in res_t["results"])

        layer_m = _build_multi_layer(n)
        tracemalloc.start()
        res_m = layer_m.ask({"entity": "Acme Corp", "predicate": "item", "operation": "list"}, validity="ignore")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks_bytes[n] = peak
        assert len(res_m["results"]) == n

        if n == 2000:
            chk = layer_t.check_claim(
                {"entity": "Acme Corp", "predicate": "item", "value": "val_01999", "assertion_kind": "fact"},
                validity="ignore",
            )
            assert chk["status"] == "supported"

    assert times[2000] < 15.0
    assert times[2000] / max(times[500], 1e-4) <= 7.0
    assert peaks_bytes[2000] < 80 * 1024 * 1024
    assert peaks_bytes[2000] / max(peaks_bytes[500], 1) <= 6.0


def test_multi_cardinality_2000_values_preserves_opposite_polarity_conflict():
    layer = _build_multi_layer(2000, add_negated_conflict=True)
    res = layer.ask({"entity": "Acme Corp", "predicate": "item", "operation": "list"}, validity="ignore")
    assert res["status"] == "answered"
    assert len(res["results"]) == 2001

    conflicts = [r for r in res["results"] if r["value"] == "val_00042"]
    assert len(conflicts) == 2
    assert {r["polarity"]: r["status"] for r in conflicts} == {
        "affirmed": "unresolved_conflict",
        "negated": "unresolved_conflict",
    }
    assert all(len(r["competitors"]) == 1 and r["competitors"][0]["relation"] == "opposing" for r in conflicts)

    non_conflicts = [r for r in res["results"] if r["value"] != "val_00042"]
    assert len(non_conflicts) == 1999
    assert all(r["status"] == "accepted" and r["competitors"] == [] for r in non_conflicts)

    chk_conflict = layer.check_claim(
        {"entity": "Acme Corp", "predicate": "item", "value": "val_00042", "assertion_kind": "fact"},
        validity="ignore",
    )
    assert chk_conflict["status"] == "unresolved_conflict"

    chk_ok = layer.check_claim(
        {"entity": "Acme Corp", "predicate": "item", "value": "val_01999", "assertion_kind": "fact"},
        validity="ignore",
    )
    assert chk_ok["status"] == "supported"


def test_internal_exception_in_ask_check_claim_and_triage_escalates_fail_closed(monkeypatch):
    # 1. ask() internal failure -> error_escalated + ESCALATE
    l1 = _build_multi_layer(3)

    def boom_decide(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(l1, "_decide", boom_decide)
    r_ask = l1.ask({"entity": "Acme Corp", "predicate": "item", "operation": "list"}, validity="ignore")
    assert r_ask["status"] == "error_escalated"
    assert r_ask["disposition"] == "ESCALATE"
    assert r_ask["abstain"] is True
    assert r_ask["error"]["type"] == "RuntimeError"

    # 2. check_claim() when ask() raises -> error_escalated + ESCALATE
    l2 = _build_multi_layer(3)

    def overflow_ask(*args, **kwargs):
        raise OverflowError("string longer than INT_MAX bytes")

    monkeypatch.setattr(l2, "ask", overflow_ask)
    r_chk = l2.check_claim(
        {"entity": "Acme Corp", "predicate": "item", "value": "val_00000", "assertion_kind": "fact"},
        validity="ignore",
    )
    assert r_chk["status"] == "error_escalated"
    assert r_chk["disposition"] == "ESCALATE"
    assert r_chk["abstain"] is True
    assert r_chk["error"]["type"] == "OverflowError"

    # 3. store.put failing with sqlite3.DataError during _issue -> check_claim still returns ESCALATE cleanly
    l3 = _build_multi_layer(3)

    def failing_put(*args, **kwargs):
        raise sqlite3.DataError("string or blob too big")

    monkeypatch.setattr(l3.store, "put", failing_put)
    r_store = l3.check_claim(
        {"entity": "Acme Corp", "predicate": "item", "value": "val_00000", "assertion_kind": "fact"},
        validity="ignore",
    )
    assert r_store["status"] == "error_escalated"
    assert r_store["disposition"] == "ESCALATE"
    assert r_store["abstain"] is True
    assert r_store["error"]["type"] == "DataError"

    # 4. triage()/verify_draft() when claim 1 raises sqlite3.DataError -> claim 1 escalates, claim 2 still verified
    l4 = _build_multi_layer(3)
    draft = "Acme Corp has item val_00000. Acme Corp has item val_00001."

    class TwoClaimModel:
        identity = {"provider": "test", "model": "two-claim", "parameters": {}}

        def complete(self, system, payload):
            eid = payload["fragments"][0]["evidence_id"]
            return {
                "output": {
                    "proposals": [
                        {
                            "entity": "Acme Corp",
                            "predicate": "item",
                            "value": "val_00000",
                            "assertion_kind": "fact",
                            "basis": "explicit",
                            "evidence": [eid],
                            "quote": "Acme Corp has item val_00000.",
                        },
                        {
                            "entity": "Acme Corp",
                            "predicate": "item",
                            "value": "val_00001",
                            "assertion_kind": "fact",
                            "basis": "explicit",
                            "evidence": [eid],
                            "quote": "Acme Corp has item val_00001.",
                        },
                    ]
                },
                "usage": {},
            }

    orig_check_claim = EvidenceLayer.check_claim

    def selective_error(proposal, *, as_of=None, context=None, validity="filter", _ask_cache=None):
        if proposal.get("value") == "val_00000":
            raise sqlite3.DataError("string or blob too big")
        return orig_check_claim(l4, proposal, as_of=as_of, context=context, validity=validity, _ask_cache=_ask_cache)

    monkeypatch.setattr(l4, "check_claim", selective_error)
    reader = LLMProposalReader(TwoClaimModel(), fragment_chars=2000, structural_reliability=0.9)
    report = verify_draft(l4, draft, reader=reader, guard=False)

    assert report["disposition"] == "ESCALATE"
    assert len(report["escalations"]) == 1
    assert report["escalations"][0]["status"] == "error_escalated"
    assert report["escalations"][0]["disposition"] == "ESCALATE"

    by_val = {s["claims"][0]["proposition"]["value"]: s for s in report["segments"] if s["claims"]}
    assert by_val["val_00000"]["lane"] == "unverified"
    assert by_val["val_00000"]["claims"][0]["status"] == "error_escalated"
    assert by_val["val_00000"]["claims"][0]["disposition"] == "ESCALATE"
    assert by_val["val_00001"]["lane"] == "verified"
    assert by_val["val_00001"]["claims"][0]["status"] == "supported"
