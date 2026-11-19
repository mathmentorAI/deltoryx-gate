"""Reproducibility and offline replay tests for v4 reader, triage, and receipts."""
from __future__ import annotations

import hashlib
import json
import time

from claimlayer import Assertion, Ledger, Source, verify
from claimlayer.semantic.models import canonical
from claimlayer.v4 import (
    EvidenceLayer,
    EvidenceRef,
    FieldSpec,
    LLMProposalReader,
    PredicateSpec,
    SourceDocument,
    Vocabulary,
)
from claimlayer.v4.triage import draft_vocabulary, verify_draft

VOCAB = Vocabulary(
    name="code_change_replay_test",
    version="1.0",
    predicates=(
        PredicateSpec("changes_file", FieldSpec("string"), cardinality="multi"),
        PredicateSpec("adds_symbol", FieldSpec("string"), cardinality="multi"),
        PredicateSpec("modifies_symbol", FieldSpec("string"), cardinality="multi"),
    ),
)


def _cache_key(system: str, payload: dict) -> str:
    encoded = system + json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _make_ledger(text: str, units: list[tuple[str, str, str]]) -> EvidenceLayer:
    layer = EvidenceLayer(None, vocabulary=VOCAB)
    doc = SourceDocument("diff_pr_1", text, authority_rank=100, weight=1.0)
    layer.ingest(doc)
    items = []
    pos = 0
    for pred, val, line in units:
        idx = text.find(line, pos)
        assert idx >= 0
        pos = idx + len(line)
        items.append(
            {
                "proposal": {
                    "entity": "pull request",
                    "predicate": pred,
                    "value": val,
                    "assertion_kind": "fact",
                },
                "evidence": [EvidenceRef(doc.document_id, doc.revision, idx, idx + len(line), line)],
            }
        )
    layer.ingest_units(items, attestor="app")
    return layer


def _build_live_and_cache_models(truncate_lengths: set[int], proposal_fn):
    cache: dict[str, dict] = {}

    class LiveCachingModel:
        identity = {"provider": "test-live", "model": "mock-llm", "parameters": {"temperature": 0}}

        def complete(self, system, payload):
            key = _cache_key(system, payload)
            if key in cache:
                return cache[key]
            frags = payload["fragments"]
            if len(frags) in truncate_lengths:
                exc = ValueError("Response truncated due to max_tokens")
                exc.diagnostics = {"attempts": [{"attempt": 1, "finish_reason": "length"}]}
                raise exc
            result = {"output": proposal_fn(payload), "usage": {"total_tokens": 10}}
            cache[key] = result
            return result

    class ReplayOnlyModel:
        identity = {"provider": "test-live", "model": "mock-llm", "parameters": {"temperature": 0}}

        def complete(self, system, payload):
            key = _cache_key(system, payload)
            if key in cache:
                return cache[key]
            raise KeyError(f"replay: missing call {key[:12]}")

    return LiveCachingModel(), ReplayOnlyModel(), cache


def test_split_window_replay_matches_live_and_is_bit_identical():
    # Case A: 2-fragment window 0+2 truncates live -> splits into 0+1 and 1+1
    # Case B: multi-window document (12 fragments) where 5+5 truncates -> 5+2 and 7+3 (which truncates -> 7+1, 8+2)
    source_lines = [
        "The pull request changes file src/auth.py.",
        "The pull request adds symbol refresh_token.",
        "The pull request modifies symbol verify_session.",
    ]
    ledger = _make_ledger(
        "\n".join(source_lines),
        [
            ("changes_file", "src/auth.py", source_lines[0]),
            ("adds_symbol", "refresh_token", source_lines[1]),
            ("modifies_symbol", "verify_session", source_lines[2]),
        ],
    )

    def make_proposals(payload):
        props = []
        for fr in payload["fragments"]:
            if "src/auth.py" in fr["text"]:
                props.append(
                    {
                        "entity": "pull request",
                        "predicate": "changes_file",
                        "value": "src/auth.py",
                        "assertion_kind": "fact",
                        "basis": "explicit",
                        "evidence": [fr["evidence_id"]],
                        "quote": "Updated src/auth.py in the module.",
                    }
                )
            if "refresh_token" in fr["text"]:
                props.append(
                    {
                        "entity": "pull request",
                        "predicate": "adds_symbol",
                        "value": "refresh_token",
                        "assertion_kind": "fact",
                        "basis": "explicit",
                        "evidence": [fr["evidence_id"]],
                        "quote": "Added refresh_token helper.",
                    }
                )
            if "missing_helper" in fr["text"]:
                props.append(
                    {
                        "entity": "pull request",
                        "predicate": "adds_symbol",
                        "value": "missing_helper",
                        "assertion_kind": "fact",
                        "basis": "explicit",
                        "evidence": [fr["evidence_id"]],
                        "quote": "Added missing_helper function.",
                    }
                )
        return {"proposals": props}

    # Case A: 2-fragment draft (fragment_chars=200, len ~ 320 -> 2 fragments: 0+2 splits into 0+1, 1+1)
    draft_2 = (
        "Summary of changes. Updated src/auth.py in the module. "
        + ("Filler prose describing refactoring details. " * 4)
        + "Added refresh_token helper. Added missing_helper function."
    )
    live_a, replay_a, cache_a = _build_live_and_cache_models({2}, make_proposals)
    r_live_a = LLMProposalReader(live_a, fragment_chars=200, window_fragments=5, structural_reliability=0.9)
    r_rep_a = LLMProposalReader(replay_a, fragment_chars=200, window_fragments=5, structural_reliability=0.9)

    rep_live_a = verify_draft(ledger, draft_2, reader=r_live_a, draft_id="draft_2")
    assert len(cache_a) == 2
    rep_1a = verify_draft(ledger, draft_2, reader=r_rep_a, draft_id="draft_2")
    time.sleep(0.02)
    rep_2a = verify_draft(ledger, draft_2, reader=r_rep_a, draft_id="draft_2")

    assert rep_1a["draft_reading"]["read_errors"] == []
    assert rep_2a["draft_reading"]["read_errors"] == []
    assert canonical(rep_live_a) == canonical(rep_1a) == canonical(rep_2a)
    assert rep_1a["claims"] == 3

    # Case B: 12-fragment draft with nested split 5+5 -> 5+2 and 7+3 -> 7+1, 8+2
    filler_block = "Context sentence explaining architecture choices without symbols. " * 3  # ~201 chars
    draft_12 = (
        "Updated src/auth.py in the module. "
        + filler_block * 5
        + "Added refresh_token helper. "
        + filler_block * 2
        + "Added missing_helper function. "
        + filler_block * 3
    )
    live_b, replay_b, cache_b = _build_live_and_cache_models({5, 3}, make_proposals)
    # Only fail the second 5-fragment window (starting at fragment 5) and its 3-fragment right child
    # by testing any 3-frag and testing exact 5-frag windows in truncate_lengths={5, 3}:
    r_live_b = LLMProposalReader(live_b, fragment_chars=200, window_fragments=5, structural_reliability=0.9)
    r_rep_b = LLMProposalReader(replay_b, fragment_chars=200, window_fragments=5, structural_reliability=0.9)

    rep_live_b = verify_draft(ledger, draft_12, reader=r_live_b, draft_id="draft_12")
    assert rep_live_b["draft_reading"]["read_errors"] == []
    rep_1b = verify_draft(ledger, draft_12, reader=r_rep_b, draft_id="draft_12")
    time.sleep(0.02)
    rep_2b = verify_draft(ledger, draft_12, reader=r_rep_b, draft_id="draft_12")

    assert rep_1b["draft_reading"]["read_errors"] == []
    assert canonical(rep_live_b) == canonical(rep_1b) == canonical(rep_2b)
    assert rep_1b["claims"] == 3

    # Receipts from ask(..., validity='ignore'), check_claim(..., validity='ignore'), and v3 verify()
    # must also be bit-for-bit identical across separate runs.
    ask_1 = ledger.ask({"entity": "pull request", "predicate": "changes_file"}, validity="ignore")
    time.sleep(0.02)
    ask_2 = ledger.ask({"entity": "pull request", "predicate": "changes_file"}, validity="ignore")
    assert ask_1["query"]["as_of"] is None
    assert canonical(ask_1) == canonical(ask_2)
    assert ask_1["response_id"] == ask_2["response_id"]
    assert ledger.verify_response_integrity(ask_1)

    chk_1 = ledger.check_claim(
        {
            "entity": "pull request",
            "predicate": "changes_file",
            "value": "src/auth.py",
            "assertion_kind": "fact",
        },
        validity="ignore",
    )
    time.sleep(0.02)
    chk_2 = ledger.check_claim(
        {
            "entity": "pull request",
            "predicate": "changes_file",
            "value": "src/auth.py",
            "assertion_kind": "fact",
        },
        validity="ignore",
    )
    assert canonical(chk_1) == canonical(chk_2)
    assert chk_1["response_id"] == chk_2["response_id"]
    assert chk_1["ledger_response_id"] == chk_2["ledger_response_id"]

    # v3 receipt determinism across two identical verify() runs
    v3_ledger = Ledger(sources=[Source(id="diff", text="def add(a, b):\n    return a + b", tier="authoritative")])
    v3_assertions = [
        Assertion(
            subject="pull_request",
            predicate="contains_change",
            value="def add(a, b):",
            source_id="diff",
            quote="def add(a, b): return a + b",
        )
    ]
    v1 = verify(v3_assertions, v3_ledger)
    time.sleep(0.02)
    v2 = verify(v3_assertions, v3_ledger)
    assert canonical(v1.to_dict()) == canonical(v2.to_dict())
    assert v1.receipt.signature == v2.receipt.signature


def test_replay_missing_leaf_still_fails_closed():
    class MissingLeafModel:
        identity = {"provider": "replay", "model": "mock", "parameters": {}}

        def complete(self, system, payload):
            raise KeyError("replay: missing leaf")

    reader = LLMProposalReader(MissingLeafModel(), fragment_chars=200, window_fragments=3, structural_reliability=0.9)
    layer = EvidenceLayer(None, vocabulary=VOCAB, reader=reader)
    summary = layer.ingest(SourceDocument("doc1", "Alpha beta gamma. " * 25))
    assert summary["units_admitted"] == 0
    assert summary["unread_documents"] == ["doc1"]
    assert len(summary["read_errors"]) >= 1
    assert all(err["error_type"] == "KeyError" for err in summary["read_errors"])


def test_live_contract_keyerror_with_diagnostics_does_not_split():
    calls = []

    class ContractKeyErrorModel:
        identity = {"provider": "live", "model": "mock", "parameters": {}}

        def complete(self, system, payload):
            calls.append(len(payload["fragments"]))
            exc = KeyError("choices")
            exc.diagnostics = {
                "attempts": [{"attempt": 1, "error_type": "KeyError", "stage": "response_contract"}]
            }
            raise exc

    reader = LLMProposalReader(
        ContractKeyErrorModel(), fragment_chars=200, window_fragments=4, structural_reliability=0.9
    )
    layer = EvidenceLayer(None, vocabulary=VOCAB, reader=reader)
    summary = layer.ingest(SourceDocument("doc2", "Some statement about code. " * 20))
    assert calls == [3]  # did NOT split into smaller windows
    assert summary["unread_documents"] == ["doc2"]
    assert summary["read_errors"][0]["error_type"] == "KeyError"


def test_draft_vocabulary_deterministic_tie_breaking():
    doc_a = SourceDocument("d", "Entity_Zeta adds foo. Entity_Alpha adds bar.")
    doc_b = SourceDocument("d", "Entity_Zeta adds foo. Entity_Alpha adds bar.")

    def build(order):
        layer = EvidenceLayer(None, vocabulary=VOCAB)
        layer.ingest(doc_a if order[0] == "Entity_Zeta" else doc_b)
        items = []
        for ent, tok in order:
            line = f"{ent} adds {tok}."
            i = doc_a.text.index(line)
            items.append(
                {
                    "proposal": {
                        "entity": ent,
                        "predicate": "adds_symbol",
                        "value": tok,
                        "assertion_kind": "fact",
                    },
                    "evidence": [EvidenceRef(doc_a.document_id, doc_a.revision, i, i + len(line), line)],
                }
            )
        layer.ingest_units(items, attestor="app")
        return draft_vocabulary(layer)

    v1 = build([("Entity_Zeta", "foo"), ("Entity_Alpha", "bar")])
    v2 = build([("Entity_Alpha", "bar"), ("Entity_Zeta", "foo")])
    assert v1.describe() == v2.describe()
    assert v1.fingerprint == v2.fingerprint
