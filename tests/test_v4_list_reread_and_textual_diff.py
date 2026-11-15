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
    PredicateSpec,
    STRICT_READER_PROMPT,
    SECURITIZATION_VOCABULARY,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
    evaluate_unit_admissibility,
    verify_extraction_fidelity,
)
from claimlayer.v4.domains.securitization import SECURITIZATION_DOMAIN_PROFILE  # noqa: E402
from claimlayer.v4.reader import _bind_raw_proposals, _build_document_fragments  # noqa: E402
from claimlayer.v4.vocabulary import value_relation  # noqa: E402
from harness2_v4_pipeline import LoadedDocument  # noqa: E402


def _make_doc(doc_id: str, text: str, *, rank: int = 100) -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        text=text,
        attribution="authoritative" if rank >= 90 else "standard",
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
        policy=AcceptancePolicy(allowed_source_attributions=("authoritative", "standard")),
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
