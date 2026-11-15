"""Tests for structured list values, gap-targeted re-reading, and token-level textual diff cross-check."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_HARNESS = _ROOT / "examples" / "harness2_securitization"
for _p in (_SRC, _HARNESS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claimlayer.semantic.models import EvidenceRef, SourceDocument, StructuredQuery, digest  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    AcceptancePolicy,
    DEFAULT_PROFILE,
    EvidenceLayer,
    EvidenceUnit,
    EXPLICIT,
    FieldEvidence,
    FieldSpec,
    LLMProposalReader,
    PredicateSpec,
    REREAD_PROMPT,
    REREAD_PROMPT_VERSION,
    RecordedProposalReader,
    STRICT_READER_PROMPT,
    SECURITIZATION_VOCABULARY,
    TargetedRereadConfig,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
    evaluate_unit_admissibility,
    verify_extraction_fidelity,
)
from claimlayer.v4.corpus_obligations import verify_corpus_obligations_coverage  # noqa: E402
from claimlayer.v4.domains.securitization import (  # noqa: E402
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_DOMAIN_PROFILE,
)
from claimlayer.v4.justification import ClaimNode, JustificationGraph  # noqa: E402
from claimlayer.v4.reader import _bind_raw_proposals, _build_document_fragments  # noqa: E402
from claimlayer.v4.vocabulary import value_relation  # noqa: E402
from harness2_v4_pipeline import LoadedDocument  # noqa: E402


def _make_doc(doc_id: str, text: str, *, rank: int = 100) -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        attribution="controlling_instrument" if rank >= 90 else "source_statement",
        authority_rank=rank,
    )


def _list_vocab() -> Vocabulary:
    return Vocabulary(
        name="list-test",
        version="1.0.0",
        require_explicit_condition_status=True,
        predicates=(
            PredicateSpec(
                name="distribution_order",
                value=FieldSpec.list(
                    FieldSpec.record(
                        step=FieldSpec("string"),
                        recipient=FieldSpec("string"),
                    ),
                    min_items=1,
                ),
                assertion_kinds=("requirement", "definition", "fact"),
                aliases=("priority of payments", "payments", "steps"),
            ),
        ),
    )


def test_list_value_type_admission_and_fingerprint_and_relation():
    v1 = Vocabulary(
        name="fp-test",
        version="1.0.0",
        predicates=(
            PredicateSpec(
                name="ordered_terms",
                value=FieldSpec.list(FieldSpec("string"), min_items=1),
                assertion_kinds=("fact",),
            ),
        ),
    )
    v2 = Vocabulary(
        name="fp-test",
        version="1.0.0",
        predicates=(
            PredicateSpec(
                name="ordered_terms",
                value=FieldSpec.list(FieldSpec("string"), min_items=2),
                assertion_kinds=("fact",),
            ),
        ),
    )
    assert v1.fingerprint != v2.fingerprint

    spec = v1.predicate("ordered_terms").value
    a = ["Step 1 Trustee", "Step 2 Reserve", "Step 3 Residual"]
    b_swapped = ["Step 1 Trustee", "Step 3 Residual", "Step 2 Reserve"]
    b_shorter = ["Step 1 Trustee", "Step 2 Reserve"]
    assert value_relation(spec, a, list(a)) == "equal"
    assert value_relation(spec, a, b_swapped) == "incompatible"
    assert value_relation(spec, a, b_shorter) == "incompatible"

    # Empty list or wrong element shape rejected by Vocabulary.value
    with pytest.raises(Exception):
        v1.proposition({
            "entity": "Waterfall",
            "predicate": "ordered_terms",
            "value": [],
            "assertion_kind": "fact",
        })


def test_list_per_element_fidelity_rejects_missing_and_out_of_order_spans():
    vocab = _list_vocab()
    text = (
        "Priority of payments applies as follows: "
        "Step 1 Servicing Fee; Step 2 Class A Interest; Step 3 Reserve Account; Step 4 Residual."
    )
    doc = _make_doc("master.docx", text)
    s0 = text.index("Step 1 Servicing Fee")
    s1 = text.index("Step 2 Class A Interest")
    s2 = text.index("Step 3 Reserve Account")
    s3 = text.index("Step 4 Residual")
    items = [
        ("Step 1 Servicing Fee", s0, s0 + len("Step 1 Servicing Fee"), "1", "Servicing Fee"),
        ("Step 2 Class A Interest", s1, s1 + len("Step 2 Class A Interest"), "2", "Class A Interest"),
        ("Step 3 Reserve Account", s2, s2 + len("Step 3 Reserve Account"), "3", "Reserve Account"),
        ("Step 4 Residual", s3, s3 + len("Step 4 Residual"), "4", "Residual"),
    ]
    prop = {
        "entity": "Priority of payments",
        "predicate": "distribution_order",
        "value": [{"step": st, "recipient": rec} for _, _, _, st, rec in items],
        "assertion_kind": "requirement",
        "condition_status": "unconditional",
    }
    full_ref = EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)
    unit = EvidenceUnit(vocab.proposition(prop), (full_ref,), EXPLICIT, "application", {"attestor": "t"})

    # 1) Valid ordered FieldEvidence passes fidelity
    good_fe = [
        FieldEvidence(f"value[{i}]", EvidenceRef(doc.document_id, doc.revision, st_o, en_o, q))
        for i, (q, st_o, en_o, _, _) in enumerate(items)
    ]
    assert verify_extraction_fidelity(text, unit, field_evidence=good_fe, vocabulary=vocab, documents={doc.document_id: doc}) == []

    # 2) Missing one element's FieldEvidence fails fidelity
    missing_fe = good_fe[:-1]
    findings_missing = verify_extraction_fidelity(
        text, unit, field_evidence=missing_fe, vocabulary=vocab, documents={doc.document_id: doc}
    )
    assert any(f.code == "MISSING_FIELD_EVIDENCE" and f.field == "value[3]" for f in findings_missing)

    # 3) Mismatched element quote fails VALUE_NOT_IN_FIELD_EVIDENCE
    bad_val_fe = list(good_fe)
    bad_val_fe[2] = FieldEvidence("value[2]", EvidenceRef(doc.document_id, doc.revision, s1, s1 + len("Step 2 Class A Interest"), "Step 2 Class A Interest"))
    findings_bad_val = verify_extraction_fidelity(
        text, unit, field_evidence=bad_val_fe, vocabulary=vocab, documents={doc.document_id: doc}
    )
    assert any(f.code == "VALUE_NOT_IN_FIELD_EVIDENCE" and f.field == "value[2]" for f in findings_bad_val)

    # 4) Out-of-order list values vs source offsets fails LIST_ORDER_MISMATCH
    swapped_prop = dict(prop)
    swapped_val = list(prop["value"])
    swapped_val[1], swapped_val[2] = swapped_val[2], swapped_val[1]
    swapped_prop["value"] = swapped_val
    swapped_unit = EvidenceUnit(vocab.proposition(swapped_prop), (full_ref,), EXPLICIT, "application", {"attestor": "t"})
    swapped_fe = [good_fe[0], FieldEvidence("value[1]", good_fe[2].evidence), FieldEvidence("value[2]", good_fe[1].evidence), good_fe[3]]
    findings_order = verify_extraction_fidelity(
        text, swapped_unit, field_evidence=swapped_fe, vocabulary=vocab, documents={doc.document_id: doc}
    )
    assert any(f.code == "LIST_ORDER_MISMATCH" and f.field == "value[2]" for f in findings_order)


def test_list_predicate_bind_and_covers_n_waterfall_steps_without_step_gaps():
    vocab = SECURITIZATION_VOCABULARY
    master_text = (
        "Section 5.01 Waterfall. Available Funds distribute: "
        "Step 1 Servicing Fee; Step 2 Trustee Fee; Step 3 Class A Interest; "
        "Step 4 Class B Principal; Step 5 Reserve Account Replenishment; Step 6 Residual."
    )
    draft_text = (
        "Section 5.01 Waterfall. Available Funds distribute: "
        "Step 1 Servicing Fee; Step 2 Trustee Fee; Step 3 Reserve Account Replenishment; "
        "Step 4 Class A Interest; Step 5 Class B Principal; Step 6 Residual."
    )
    master = _make_doc("master.docx", master_text, rank=100)
    draft = _make_doc("draft.docx", draft_text, rank=60)

    def _raw_wf_proposal(doc: SourceDocument, steps: list[tuple[str, str, str]]) -> dict[str, Any]:
        frags = _build_document_fragments({"document_id": doc.document_id, "revision": doc.revision, "text": doc.text}, 1800)
        ev_id = frags[0][0]
        raw = {
            "entity": "Available Funds",
            "predicate": "waterfall_schedule",
            "value": [{"step": st, "beneficiary": ben} for st, ben, _ in steps],
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "basis": "explicit",
            "evidence": [ev_id],
            "quote": "Available Funds distribute:",
            "field_evidence": {
                "entity": "Available Funds",
                "items": [q for _, _, q in steps],
            },
        }
        items, rejected = [], []
        _bind_raw_proposals([raw], dict(frags), doc.text, items, rejected)
        assert not rejected
        return items[0]

    layer = EvidenceLayer(
        vocabulary=vocab,
        policy=SECURITIZATION_ACCEPTANCE_POLICY,
    )
    layer.ingest([master, draft])
    layer.ingest_units(
        [
            _raw_wf_proposal(master, [
                ("1", "Servicing Fee", "Step 1 Servicing Fee"),
                ("2", "Trustee Fee", "Step 2 Trustee Fee"),
                ("3", "Class A Interest", "Step 3 Class A Interest"),
                ("4", "Class B Principal", "Step 4 Class B Principal"),
                ("5", "Reserve Account Replenishment", "Step 5 Reserve Account Replenishment"),
                ("6", "Residual", "Step 6 Residual"),
            ]),
            _raw_wf_proposal(draft, [
                ("1", "Servicing Fee", "Step 1 Servicing Fee"),
                ("2", "Trustee Fee", "Step 2 Trustee Fee"),
                ("3", "Reserve Account Replenishment", "Step 3 Reserve Account Replenishment"),
                ("4", "Class A Interest", "Step 4 Class A Interest"),
                ("5", "Class B Principal", "Step 5 Class B Principal"),
                ("6", "Residual", "Step 6 Residual"),
            ]),
        ],
        attestor="test_reader",
    )

    # Verify both units pass strict admissibility + fidelity
    assert len(layer.units) == 2
    docs_map = {master.document_id: master, draft.document_id: draft}
    for u in layer.units:
        ev_res = evaluate_unit_admissibility(
            u,
            vocabulary=vocab,
            policy=layer.policy,
            profile=layer.profile,
            documents=docs_map,
            stored_field_evidences=layer.field_evidences,
        )
        assert ev_res.admissible, f"Expected admissible list unit, got {ev_res.reason}"

    loaded_docs = {
        "master.docx": LoadedDocument(
            doc_key="D1",
            filename="master.docx",
            role="governing_operative_agreement",
            authority_tier="authoritative",
            authority_rank=100,
            governs_on_conflict=True,
            text=master_text,
            start_line=1,
            sections=(),
        ),
        "draft.docx": LoadedDocument(
            doc_key="D2",
            filename="draft.docx",
            role="disclosure_document",
            authority_tier="standard",
            authority_rank=60,
            governs_on_conflict=False,
            text=draft_text,
            start_line=1,
            sections=(),
        ),
    }
    divs = collect_reference_divergences(loaded_docs, SECURITIZATION_DOMAIN_PROFILE)
    reg = build_corpus_reference_register(
        layer,
        vocab,
        divergence_collection=divs,
        expected_documents={"master.docx": master, "draft.docx": draft},
    )
    assert reg.available is True
    assert reg.has_material_conflicts is True
    assert reg.has_authority_overrides is True
    # Every step number is inside verified list item spans -> 0 quantity step gaps!
    assert [g for g in reg.gap_divergences if g.get("kind") == "quantity"] == []


class _StubTargetedModel:
    """Deterministic stub OpenAICompatibleModel supporting targeted re-read tests."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, Any]] = []

    @property
    def identity(self) -> dict[str, Any]:
        return {"provider": "stub", "model": "stub-reread-model"}

    def complete(self, system_prompt: str, payload: Any) -> dict[str, Any]:
        self.calls.append((system_prompt, payload))
        return {"output": {"proposals": self.handler(system_prompt, payload)}}


def test_targeted_reread_closes_gap_blind_with_fidelity_and_budget_and_provenance():
    master_text = (
        "Section 2.04 Rates: Tranche B carries 5.80% per annum.\n"
        "Section 5.01 Distribution Sequence: Step 1 Trustee Fee; Step 2 Reserve Deposit; Step 3 Class B Principal."
    )
    draft_text = (
        "Section 2.04 Rates: Tranche B carries 5.55% per annum.\n"
        "Section 5.01 Distribution Sequence: Step 1 Trustee Fee; Step 2 Class B Principal; Step 3 Reserve Deposit."
    )
    master = _make_doc("indenture.docx", master_text, rank=100)
    draft = _make_doc("memorandum.docx", draft_text, rank=60)
    docs_loaded = {
        "indenture.docx": LoadedDocument(
            doc_key="D1",
            filename="indenture.docx",
            role="governing_operative_agreement",
            authority_tier="authoritative",
            authority_rank=100,
            governs_on_conflict=True,
            text=master_text,
            start_line=1,
            sections=(),
        ),
        "memorandum.docx": LoadedDocument(
            doc_key="D2",
            filename="memorandum.docx",
            role="disclosure_document",
            authority_tier="standard",
            authority_rank=60,
            governs_on_conflict=False,
            text=draft_text,
            start_line=1,
            sections=(),
        ),
    }

    # Initial reader captures Tranche B rates ONLY -> leaves Step 1/2/3 waterfall gaps!
    def _handler(sys_prompt: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if "target_sentence" not in payload:
            # First pass: ONLY return coupon_rate
            ev_id = payload["fragments"][0]["evidence_id"]
            frag = payload["fragments"][0]["text"]
            rate = "5.80%" if "5.80%" in frag else "5.55%"
            val = 5.80 if "5.80%" in frag else 5.55
            return [{
                "entity": "Tranche B",
                "predicate": "coupon_rate",
                "value": val,
                "unit": "percent",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "basis": "explicit",
                "evidence": [ev_id],
                "quote": f"Tranche B carries {rate} per annum.",
                "field_evidence": {"entity": "Tranche B", "value": f"{rate} per annum"},
            }]
        # Second targeted read: MUST NOT contain any response/graph keys
        assert "candidate_response" not in payload
        assert "response" not in payload
        assert "graph" not in payload
        assert payload["target_sentence"]
        assert payload["target_tokens"]
        ev_id = payload["fragments"][0]["evidence_id"]
        did = payload["document_id"]
        if did == "indenture.docx":
            items = [
                {"step": "1", "beneficiary": "Trustee Fee"},
                {"step": "2", "beneficiary": "Reserve Deposit"},
                {"step": "3", "beneficiary": "Class B Principal"},
            ]
            quotes = ["Step 1 Trustee Fee", "Step 2 Reserve Deposit", "Step 3 Class B Principal"]
        else:
            items = [
                {"step": "1", "beneficiary": "Trustee Fee"},
                {"step": "2", "beneficiary": "Class B Principal"},
                {"step": "3", "beneficiary": "Reserve Deposit"},
            ]
            quotes = ["Step 1 Trustee Fee", "Step 2 Class B Principal", "Step 3 Reserve Deposit"]
        return [{
            "entity": "Distribution Sequence",
            "predicate": "waterfall_schedule",
            "value": items,
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "basis": "explicit",
            "evidence": [ev_id],
            "quote": payload["target_sentence"],
            "field_evidence": {"entity": "Distribution Sequence", "items": quotes},
        }]

    model = _StubTargetedModel(_handler)
    reader = LLMProposalReader(
        model=model,
        prompt=STRICT_READER_PROMPT,
        prompt_version="v4-proposals-4",
        reread_prompt=REREAD_PROMPT,
        reread_prompt_version=REREAD_PROMPT_VERSION,
        structural_reliability=0.95,
    )
    layer = EvidenceLayer(vocabulary=SECURITIZATION_VOCABULARY, policy=SECURITIZATION_ACCEPTANCE_POLICY, reader=reader)
    layer.ingest([master, draft])
    divs = collect_reference_divergences(docs_loaded, SECURITIZATION_DOMAIN_PROFILE)

    # First verify gaps exist when reread is disabled
    reg_no_rr = build_corpus_reference_register(
        layer,
        SECURITIZATION_VOCABULARY,
        divergence_collection=divs,
        expected_documents={"indenture.docx": master, "memorandum.docx": draft},
        reread_config=TargetedRereadConfig(enabled=False),
    )
    assert len(reg_no_rr.gap_divergences) >= 2
    assert reg_no_rr.reread_provenance["performed"] is False

    # Now enable targeted re-reading
    reg_rr = build_corpus_reference_register(
        layer,
        SECURITIZATION_VOCABULARY,
        divergence_collection=divs,
        expected_documents={"indenture.docx": master, "memorandum.docx": draft},
        reread_config=TargetedRereadConfig(enabled=True, max_attempts_per_document=3),
    )
    assert reg_rr.reread_provenance["performed"] is True
    assert reg_rr.reread_provenance["closed_gap_count"] >= 2
    assert reg_rr.reread_provenance["open_gap_count"] == 0
    assert reg_rr.reread_provenance["reader"]["prompt_version"] == REREAD_PROMPT_VERSION
    assert reg_rr.gap_divergences == ()


def test_targeted_reread_unclosed_or_unfaithful_or_budget_exceeded_still_escalates():
    master_text = "Clause 1: Minimum reserve $10,000,000.\nClause 2: Delinquency cap 4.50%."
    draft_text = "Clause 1: Minimum reserve $12,000,000.\nClause 2: Delinquency cap 5.25%."
    master = _make_doc("indenture.docx", master_text, rank=100)
    draft = _make_doc("memorandum.docx", draft_text, rank=60)
    docs_loaded = {
        "indenture.docx": LoadedDocument("D1", "indenture.docx", "governing_operative_agreement", "authoritative", 100, True, master_text, 1, ()),
        "memorandum.docx": LoadedDocument("D2", "memorandum.docx", "disclosure_document", "standard", 60, False, draft_text, 1, ()),
    }

    # Re-reader returns a hallucinated wrong figure (fails fidelity -> rejected!) on Clause 2
    def _bad_reread_handler(sys_prompt: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
        ev_id = payload["fragments"][0]["evidence_id"]
        if "target_sentence" not in payload:
            # First pass covers Clause 1 reserve only
            amt = 10000000 if "$10,000,000" in payload["fragments"][0]["text"] else 12000000
            raw_amt = "$10,000,000" if amt == 10000000 else "$12,000,000"
            return [{
                "entity": "Minimum reserve",
                "predicate": "note_principal",
                "value": amt,
                "unit": "USD",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "basis": "explicit",
                "evidence": [ev_id],
                "quote": f"Minimum reserve {raw_amt}.",
                "field_evidence": {"entity": "Minimum reserve", "value": raw_amt},
            }]
        # Bad value 9.99% NOT present in doc -> fidelity must reject!
        return [{
            "entity": "Delinquency cap",
            "predicate": "delinquency_trigger",
            "value": 9.99,
            "unit": "percent",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "basis": "explicit",
            "evidence": [ev_id],
            "quote": payload["target_sentence"],
            "field_evidence": {"entity": "Delinquency cap", "value": "9.99%"},
        }]

    model = _StubTargetedModel(_bad_reread_handler)
    reader = LLMProposalReader(model=model, prompt=STRICT_READER_PROMPT, structural_reliability=0.95)
    layer = EvidenceLayer(vocabulary=SECURITIZATION_VOCABULARY, policy=SECURITIZATION_ACCEPTANCE_POLICY, reader=reader)
    layer.ingest([master, draft])
    divs = collect_reference_divergences(docs_loaded, SECURITIZATION_DOMAIN_PROFILE)

    # Budget max_attempts_per_document=1 restricts attempts per doc, AND hallucinated proposal is rejected!
    reg = build_corpus_reference_register(
        layer,
        SECURITIZATION_VOCABULARY,
        divergence_collection=divs,
        expected_documents={"indenture.docx": master, "memorandum.docx": draft},
        reread_config=TargetedRereadConfig(enabled=True, max_attempts_per_document=1),
    )
    assert reg.reread_provenance["performed"] is True
    assert reg.reread_provenance["closed_gap_count"] == 0
    assert reg.reread_provenance["open_gap_count"] >= 2
    assert any(a["proposals_rejected"] >= 1 for a in reg.reread_provenance["attempts"])

    # Unclosed gaps MUST still emit REFERENCE_REGISTER_GAP -> ESCALATE
    dummy_graph = JustificationGraph(claims={}, steps=[], conclusion_id="")
    cov = verify_corpus_obligations_coverage(reg, dummy_graph, candidate_response="Minimum reserve is $10,000,000 per indenture.docx.")
    assert any(i["code"] == "REFERENCE_REGISTER_GAP" and not i["repairable"] for i in cov.issues)


def test_textual_diff_same_coupon_plus_subordination_and_joint_numeric_shift():
    """Mere overlap with non-differing unit OR joint numeric shift MUST emit kind='textual' gaps."""
    from claimlayer.v4.corpus_obligations import build_corpus_reference_register
    from claimlayer.v4.domain_profile import DomainProfile, PolicyConfiguration

    vocab = Vocabulary(
        name="textual-diff-test",
        version="1.0.0",
        units=SECURITIZATION_VOCABULARY.units,
        predicates=(
            PredicateSpec(
                name="coupon_rate",
                value=FieldSpec("quantity", dimensions=("ratio", "annual_rate")),
                assertion_kinds=("requirement", "fact"),
                required=True,
            ),
            PredicateSpec(
                name="seniority",
                value=FieldSpec("enum", enum=("senior", "subordinated")),
                assertion_kinds=("requirement", "fact"),
            ),
        ),
    )

    # Case 1: Coupon matches (5.80% on both) + overlapping coupon spans the whole sentence,
    # while "senior" vs "subordinated" differs -> MUST emit textual gap!
    m_text = "Class B Notes bear interest at 5.80% and rank senior to Class C."
    d_text = "Class B Notes bear interest at 5.80% and rank subordinated to Class C."
    m_doc = _make_doc("indenture.docx", m_text, rank=100)
    d_doc = _make_doc("draft_oc.docx", d_text, rank=60)
    layer1 = EvidenceLayer(vocabulary=vocab, policy=SECURITIZATION_ACCEPTANCE_POLICY)
    layer1.ingest([m_doc, d_doc])

    def _prop(doc: SourceDocument, pred: str, val: Any, unit: str | None, quote: str) -> dict[str, Any]:
        frags = _build_document_fragments({"document_id": doc.document_id, "revision": doc.revision, "text": doc.text}, 1800)
        p: dict[str, Any] = {
            "entity": "Class B Notes",
            "predicate": pred,
            "value": val,
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "basis": "explicit",
            "evidence": [frags[0][0]],
            "quote": quote,
            "field_evidence": {"entity": "Class B Notes"},
        }
        if unit is not None:
            p["unit"] = unit
        items, rej = [], []
        _bind_raw_proposals([p], dict(frags), doc.text, items, rej)
        assert not rej
        return items[0]

    layer1.ingest_units(
        [
            _prop(m_doc, "coupon_rate", 5.8, "percent_per_annum", m_text),
            _prop(d_doc, "coupon_rate", 5.8, "percent_per_annum", d_text),
        ],
        attestor="test_reader",
    )
    prof = DomainProfile(
        profile_id="txt_diff_prof",
        vocabulary=vocab,
        policy=PolicyConfiguration(authority_ranks={"indenture.docx": 100, "draft_oc.docx": 60}),
    )
    loaded1 = {
        "indenture.docx": LoadedDocument("D1", "indenture.docx", "governing_operative_agreement", "authoritative", 100, True, m_text, 1, ()),
        "draft_oc.docx": LoadedDocument("D2", "draft_oc.docx", "disclosure_document", "standard", 60, False, d_text, 1, ()),
    }
    reg1 = build_corpus_reference_register(
        layer1,
        vocab,
        divergence_collection=collect_reference_divergences(loaded1, prof),
        expected_documents={"indenture.docx": m_doc, "draft_oc.docx": d_doc},
        reread_config=TargetedRereadConfig(enabled=False),
    )
    txt_gaps1 = [g for g in reg1.gap_divergences if g.get("kind") == "textual"]
    assert txt_gaps1, f"Expected textual gap for senior vs subordinated, got {reg1.gap_divergences}"
    assert any("senior" in str(g.get("changed_text", "")) or "subordinated" in str(g.get("changed_text", "")) for g in txt_gaps1)
    cov1 = verify_corpus_obligations_coverage(reg1, JustificationGraph(claims={}, steps=[], conclusion_id=""), candidate_response="Class B Notes 5.80%.")
    assert any(i["code"] == "REFERENCE_REGISTER_GAP" for i in cov1.issues)

    # Case 2: Joint numeric shift (5.80% -> 5.55%) AND textual shift ("senior" -> "subordinated")
    # Even when coupon_rate conflict covers the rate span, uncovered "senior" <-> "subordinated" MUST emit textual gap!
    d2_text = "Class B Notes bear interest at 5.55% and rank subordinated to Class C."
    d2_doc = _make_doc("draft_oc.docx", d2_text, rank=60)
    layer2 = EvidenceLayer(vocabulary=vocab, policy=SECURITIZATION_ACCEPTANCE_POLICY)
    layer2.ingest([m_doc, d2_doc])
    layer2.ingest_units(
        [
            _prop(m_doc, "coupon_rate", 5.8, "percent_per_annum", "bear interest at 5.80%"),
            _prop(d2_doc, "coupon_rate", 5.55, "percent_per_annum", "bear interest at 5.55%"),
        ],
        attestor="test_reader",
    )
    loaded2 = {
        "indenture.docx": loaded1["indenture.docx"],
        "draft_oc.docx": LoadedDocument("D2", "draft_oc.docx", "disclosure_document", "standard", 60, False, d2_text, 1, ()),
    }
    reg2 = build_corpus_reference_register(
        layer2,
        vocab,
        divergence_collection=collect_reference_divergences(loaded2, prof),
        expected_documents={"indenture.docx": m_doc, "draft_oc.docx": d2_doc},
        reread_config=TargetedRereadConfig(enabled=False),
    )
    txt_gaps2 = [g for g in reg2.gap_divergences if g.get("kind") == "textual"]
    assert txt_gaps2, f"Expected textual gap when sentence has both rate diff and priority diff, got {reg2.gap_divergences}"
    assert any("senior" in str(g.get("changed_text", "")) or "subordinated" in str(g.get("changed_text", "")) for g in txt_gaps2)

    # Case 3: When differing seniority conflict IS extracted covering "senior"/"subordinated" spans -> 0 textual gaps!
    layer3 = EvidenceLayer(vocabulary=vocab, policy=SECURITIZATION_ACCEPTANCE_POLICY)
    layer3.ingest([m_doc, d_doc])
    layer3.ingest_units(
        [
            _prop(m_doc, "coupon_rate", 5.8, "percent_per_annum", "bear interest at 5.80%"),
            _prop(d_doc, "coupon_rate", 5.8, "percent_per_annum", "bear interest at 5.80%"),
            _prop(m_doc, "seniority", "senior", None, "rank senior to Class C"),
            _prop(d_doc, "seniority", "subordinated", None, "rank subordinated to Class C"),
        ],
        attestor="test_reader",
    )
    reg3 = build_corpus_reference_register(
        layer3,
        vocab,
        divergence_collection=collect_reference_divergences(loaded1, prof),
        expected_documents={"indenture.docx": m_doc, "draft_oc.docx": d_doc},
        reread_config=TargetedRereadConfig(enabled=False),
    )
    assert reg3.gap_divergences == ()


def test_textual_diff_catches_non_numeric_variants_and_ignores_formatting_and_doc_ids():
    """Textual diff catches condition/party/timeframe shifts while ignoring formatting & structural doc ids."""
    from claimlayer.v4.corpus_obligations import _divergence_gaps_for_item, build_corpus_reference_register
    from claimlayer.v4.domain_profile import DomainProfile, PolicyConfiguration

    vocab = Vocabulary(
        name="txt-variants",
        version="1.0.0",
        units=SECURITIZATION_VOCABULARY.units,
        predicates=(
            PredicateSpec("fee", value=FieldSpec("quantity", dimensions=("ratio", "annual_rate")), assertion_kinds=("requirement",), required=True),
        ),
    )
    prof = DomainProfile(
        profile_id="var_prof",
        vocabulary=vocab,
        policy=PolicyConfiguration(authority_ranks={"master_indenture.pdf": 100, "draft_summary.pdf": 60}),
    )

    # 1. Varied non-numeric shifts MUST emit textual gaps
    for pri_sent, sec_sent, expected_frag in (
        (
            "Servicer shall remit collections upon closing.",
            "Master Servicer shall remit collections upon closing.",
            "Master",
        ),
        (
            "Interest distributions are payable monthly.",
            "Interest distributions are payable monthly unless waived.",
            "unless waived",
        ),
        (
            "Notice takes effect upon closing.",
            "Notice takes effect after written notice.",
            "closing",
        ),
    ):
        loaded = {
            "master_indenture.pdf": LoadedDocument("D1", "master_indenture.pdf", "governing_operative_agreement", "authoritative", 100, True, pri_sent, 1, ()),
            "draft_summary.pdf": LoadedDocument("D2", "draft_summary.pdf", "disclosure_document", "standard", 60, False, sec_sent, 1, ()),
        }
        coll = collect_reference_divergences(loaded, prof)
        assert coll.divergences
        gaps = _divergence_gaps_for_item(
            coll.divergences[0],
            doc_texts={"master_indenture.pdf": pri_sent, "draft_summary.pdf": sec_sent},
            admitted_units=(),
            unit_spans_by_doc={},
            reader_obligations=(),
            doc_metadata=coll.document_metadata,
        )
        txt_gaps = [g for g in gaps if g.get("kind") == "textual"]
        assert txt_gaps, f"Expected textual gap for {pri_sent!r} vs {sec_sent!r}"
        assert any(expected_frag.lower() in str(g.get("changed_text", "")).lower() or expected_frag.lower() in str(g.get("counterpart_changed_text", "")).lower() for g in txt_gaps)

    # 2. Formatting-only (whitespace, punctuation, casing) AND structural doc-id aliases -> 0 textual gaps
    fmt_pri = "Per Master Indenture, Class B Notes bear interest at 5.80% per annum!"
    fmt_sec = "Per  draft summary,  class b notes,   bear interest at 5.80 % per annum."
    loaded_fmt = {
        "master_indenture.pdf": LoadedDocument("D1", "master_indenture.pdf", "governing_operative_agreement", "authoritative", 100, True, fmt_pri, 1, ()),
        "draft_summary.pdf": LoadedDocument("D2", "draft_summary.pdf", "disclosure_document", "standard", 60, False, fmt_sec, 1, ()),
    }
    coll_fmt = collect_reference_divergences(loaded_fmt, prof)
    for div in coll_fmt.divergences:
        gaps = _divergence_gaps_for_item(
            div,
            doc_texts={"master_indenture.pdf": fmt_pri, "draft_summary.pdf": fmt_sec},
            admitted_units=(),
            unit_spans_by_doc={},
            reader_obligations=(),
            doc_metadata=coll_fmt.document_metadata,
        )
        assert gaps == []


def test_recorded_reads_v2_replay_preserves_ablation_and_closes_gaps():
    """Offline replay of live nvidia/nemotron-3-ultra-550b-a55b v2 + reread recordings stays 10/10 on ablation and deterministic."""
    import sys

    bench_dir = Path(__file__).resolve().parents[1] / "benchmarks"
    if str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    import measure_reader_recall as mrr

    rec_v2_dir = Path(__file__).resolve().parents[1] / "examples" / "harness2_securitization" / "recorded_reads_v2"
    ann_path = bench_dir / "conflict_annotations.json"
    rep = mrr.measure_recall(annotations_path=ann_path, recordings_dir=rec_v2_dir, mode="recorded")
    summary = rep["summary"]
    assert summary["documents_total"] == 15
    assert summary["documents_read_ok"] == 15
    assert summary["ablation"]["conflicts_removed"] == 10
    assert summary["ablation"]["gaps_detected"] == 10
    assert summary["gaps_closed_by_reread_total"] == 2
    assert summary["gaps_by_kind"]["quantity"] == 4
    assert summary["clean_corpus_false_gaps"]["false_gaps_total"] == 0
    assert summary["clean_candidate_escalation_rate_wilson_95"]["k"] == 0
    per_corpus = {c["corpus_id"]: c for c in rep["corpora"]}
    assert per_corpus["meridian"]["gaps_closed_by_reread"] == 2
    assert per_corpus["apex"]["reference_register_gaps"] == 0
    assert per_corpus["ridgeline"]["reference_register_gaps"] == 0


def test_duplicate_raw_proposals_and_authoritative_reread_attribution():
    """Duplicate semantic proposals with differing raw dicts do not fail store immutability, and authoritative_instrument rereads admit."""
    doc = _make_doc(
        "indenture.docx",
        "Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum.",
        rank=100,
    )
    doc = SourceDocument(
        document_id=doc.document_id,
        text=doc.text,
        authority_rank=doc.authority_rank,
        attribution="authoritative_instrument",
    )
    frags = _build_document_fragments(
        {"document_id": doc.document_id, "revision": doc.revision, "text": doc.text},
        1800,
    )
    fid = frags[0][0]

    class _DupReader:
        identity = {"provider": "test", "model": "dup", "prompt_version": "t", "prompt_digest": "d"}

        def propose(self, req):
            items, rej = [], []
            raw = [
                {
                    "entity": "Class B Notes",
                    "predicate": "coupon_rate",
                    "value": {"value": 5.8, "unit": "percent_per_annum"},
                    "assertion_kind": "fact",
                    "condition_status": "unconditional",
                    "basis": "explicit",
                    "evidence": [fid],
                    "quote": "Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum.",
                    "field_evidence": {"entity": "Class B Notes", "value": "5.80% per annum"},
                },
                {
                    "entity": "Class B Notes",
                    "predicate": "coupon_rate",
                    "value": {"value": 5.8, "unit": "percent_per_annum"},
                    "assertion_kind": "fact",
                    "conditions": [],
                    "condition_status": "unconditional",
                    "basis": "explicit",
                    "evidence": [fid],
                    "quote": "Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum.",
                    "field_evidence": {"entity": "Class B Notes", "value": "5.80% per annum"},
                },
            ]
            _bind_raw_proposals(raw, dict(frags), doc.text, items, rej)
            assert not rej
            return {"items": items, "rejected": [], "calls": [], "errors": [], "fragments": 1, "windows": 1}

    layer = EvidenceLayer(vocabulary=SECURITIZATION_VOCABULARY, policy=SECURITIZATION_ACCEPTANCE_POLICY, reader=_DupReader())
    res = layer.ingest([doc])
    assert res["units_admitted"] == 1
    assert not res["read_errors"]

    rr_items, rr_rej = [], []
    _bind_raw_proposals(
        [
            {
                "entity": "Class B Notes",
                "predicate": "note_principal",
                "value": {"value": 40000000, "unit": "USD"},
                "assertion_kind": "fact",
                "condition_status": "unconditional",
                "basis": "explicit",
                "evidence": [fid],
                "quote": "Class B Notes ($40,000,000 initial principal) bear interest at 5.80% per annum.",
                "field_evidence": {"entity": "Class B Notes", "value": "$40,000,000"},
            }
        ],
        dict(frags),
        doc.text,
        rr_items,
        rr_rej,
    )
    assert not rr_rej
    rr_sum = layer.admit_reread_items(rr_items, producer={"kind": "reread_reader", "identity": {"model": "dup"}})
    assert rr_sum["units_admitted"] == 1
    assert not rr_sum["rejection_reasons"]
