"""Regression tests covering external-audit follow-up issues against 79cc1f5:
1. Single-entity multi-value predicate O(N^2) conflict blowup (50..1006 units).
2. Line-break & PDF/OCR noise false alarms (60+ clean contract pairs + diff.py).
3. Dense multi-predicate reader localization of negation/exception mutations.
4. FinanceBench v3 verify() SEC table scale headers & multi-column rows.
"""
from __future__ import annotations

import time
from typing import Any

from claimlayer.ledger import Ledger
from claimlayer.models import Assertion, Source
from claimlayer.semantic.models import SourceDocument
from claimlayer.v4 import (
    AcceptancePolicy,
    EvidenceLayer,
    PredicateSpec,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.reader import _bind_raw_proposals, _build_document_fragments
from claimlayer.v4.vocabulary import FieldSpec
from claimlayer.verifiers.diff import extract_divergent_sections
from claimlayer.verify import verify


def _dense_clause_vocab() -> Vocabulary:
    return Vocabulary(
        name="clause_test",
        version="1.0",
        predicates=(
            PredicateSpec(name="clause", value=FieldSpec(type="string"), cardinality="multi"),
        ),
    )


class _DenseClauseReader:
    """Reader that emits every non-heading line/sentence as a single-entity `clause` unit."""

    def __init__(self, negated_substrings: tuple[str, ...] = ()) -> None:
        self.negated_substrings = negated_substrings
        self.identity = {
            "provider": "test",
            "model": "dense_clause_reader",
            "prompt_version": "1",
            "prompt_digest": "d",
        }

    def propose(self, req: dict[str, Any]) -> dict[str, Any]:
        doc = req["document"]
        frags = _build_document_fragments(doc, 4000)
        frag_map = dict(frags)
        raw: list[dict[str, Any]] = []
        seen: set[str] = set()
        for fid, fobj in frags:
            for ln in fobj["quote"].split("\n"):
                s = ln.strip()
                if not s or s.startswith(("Section ", "Article ", "1. ", "2. ")) or s in seen:
                    continue
                seen.add(s)
                polarity = "affirmed"
                for marker in self.negated_substrings:
                    if marker in s:
                        polarity = "negated"
                        s_val = s.replace("may not ", "may ").replace("shall not ", "shall ")
                        break
                else:
                    s_val = s
                raw.append({
                    "entity": "Contract",
                    "predicate": "clause",
                    "value": s_val,
                    "polarity": polarity,
                    "assertion_kind": "fact",
                    "conditions": [],
                    "condition_status": "unconditional",
                    "basis": "explicit",
                    "evidence": [fid],
                    "quote": s,
                    "field_evidence": {"entity": "Contract", "value": s},
                })
        items: list[Any] = []
        rej: list[Any] = []
        _bind_raw_proposals(raw, frag_map, doc["text"], items, rej)
        return {"items": items, "rejected": rej, "calls": [], "errors": [], "fragments": len(frags), "windows": 1}


def test_multi_predicate_single_entity_scaling_and_polarity_conflict():
    vocab = _dense_clause_vocab()
    policy = AcceptancePolicy()

    for n_per_doc in (25, 503):
        lines = [
            f"Clause {i}: Obligor agrees to covenant item number {i} under Section {i % 7 + 1}."
            for i in range(n_per_doc)
        ]
        txt = "\n".join(lines)
        d1 = SourceDocument(document_id="d1", text=txt, authority_rank=100, attribution="governing")
        d2 = SourceDocument(document_id="d2", text=txt, authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader())
        layer.ingest([d1, d2])
        assert len(layer.units) >= 2 * n_per_doc

        t0 = time.perf_counter()
        reg = build_corpus_reference_register(layer, vocab)
        dt = time.perf_counter() - t0
        conflicts = [
            o
            for o in reg.obligations
            if (o.parameters or {}).get("corpus_obligation_kind") == "material_conflict"
        ]
        assert conflicts == [], f"Unexpected multi conflicts for n_per_doc={n_per_doc}: {len(conflicts)}"
        assert dt < 2.0, f"Too slow for {len(layer.units)} single-entity multi units: {dt:.3f}s"

    # Explicit opposite polarity on equal normalized value STILL produces 1 real conflict
    t1 = "Receiving Party may assign its rights.\nReceiving Party shall keep data secret."
    t2 = "Receiving Party may not assign its rights.\nReceiving Party shall keep data secret."
    d1 = SourceDocument(document_id="p1", text=t1, authority_rank=100, attribution="governing")
    d2 = SourceDocument(document_id="p2", text=t2, authority_rank=80, attribution="draft")
    l_pol = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader(negated_substrings=("may not assign",)))
    l_pol.ingest([d1, d2])
    reg_pol = build_corpus_reference_register(l_pol, vocab)
    conflicts_pol = [
        o
        for o in reg_pol.obligations
        if (o.parameters or {}).get("corpus_obligation_kind") == "material_conflict"
    ]
    assert len(conflicts_pol) == 1
    assert conflicts_pol[0].parameters["predicate"] == "clause"


def test_line_break_and_ocr_tolerance_60_clean_pairs():
    vocab = _dense_clause_vocab()
    policy = AcceptancePolicy()
    for i in range(60):
        base = (
            f"1. Confidentiality\n\n"
            f"Acme Corp. and Beta, Inc. agree that Confidential Information disclosed under Agreement {i} "
            f"by U.S. Counsel shall remain strictly confidential for 3 years. "
            f"Exceptions apply to data:\n(a) publicly available; and (b) independently developed.\n"
            f"This Agreement.\n1) Purpose. Obligations of Acme Corp. bind all Affiliates."
        )
        # Apply varied PDF/OCR noise and arbitrary mid-sentence / blank-line / abbrev wraps
        wrapped = (
            f"1. Con\ufb01dentiality\n\n"
            f"Acme Corp.\nand Beta, Inc.\nagree that Con\ufb01dential Information dis-\nclosed under Agreement {i}\n\n"
            f"by U.S.\nCounsel shall remain strictly con\ufb01dential for 3 years. "
            f"Exceptions ap-\nply to data: (a) publicly available; and (b) inde-\npendently developed.\n"
            f"This Agreement. 1) Purpose. Obliga-\ntions of Acme Corp. bind all Af\ufb01liates."
        )
        p_src = Source(id=f"p_{i}", text=base, authority_rank=100)
        s_src = Source(id=f"s_{i}", text=wrapped, authority_rank=80)
        assert extract_divergent_sections(p_src, s_src) == [], f"False diff.py divergence on pair {i}"

        d1 = SourceDocument(document_id=f"d1_{i}", text=base, authority_rank=100, attribution="gov")
        d2 = SourceDocument(document_id=f"d2_{i}", text=wrapped, authority_rank=80, attribution="draft")
        dc = collect_reference_divergences([d1, d2])
        assert dc.divergences == (), f"False section divergence on pair {i}: {dc.divergences}"

        layer = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader())
        layer.ingest([d1, d2])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        assert reg.gap_divergences == (), f"False gap divergence on pair {i}: {reg.gap_divergences}"

    # Clean wrapped prose inside a section with ONE genuine mutation elsewhere produces zero spurious wraps
    p_mix = SourceDocument(
        document_id="mix_p",
        text=(
            "1. Terms\n"
            "The Receiving Party, Inc. shall maintain all proprietary specifications in strict confidence.\n"
            "Confidentiality lasts 5 years."
        ),
        authority_rank=100,
        attribution="gov",
    )
    s_mix = SourceDocument(
        document_id="mix_s",
        text=(
            "1. Terms\n"
            "The Receiving Party, Inc.\n\nshall maintain all pro-\nprietary speci\ufb01cations in strict con\ufb01dence.\n"
            "Confidentiality lasts 3 years."
        ),
        authority_rank=80,
        attribution="draft",
    )
    l_mix = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader())
    l_mix.ingest([p_mix, s_mix])
    dc_mix = collect_reference_divergences([p_mix, s_mix])
    reg_mix = build_corpus_reference_register(l_mix, vocab, divergence_collection=dc_mix)
    assert all("proprietary" not in str(g.get("changed_text", "")) for g in reg_mix.gap_divergences)
    assert any("5:YEAR" in str(g.get("token")) or "3:YEAR" in str(g.get("token")) for g in reg_mix.gap_divergences)


def test_dense_multi_reader_localizes_negation_and_exception_changes():
    vocab = _dense_clause_vocab()
    policy = AcceptancePolicy()

    mutations: list[tuple[str, str]] = [
        ("Receiving Party may disclose data.", "Receiving Party may not disclose data."),
        ("Seller shall indemnify Buyer for losses.", "Seller shall not indemnify Buyer for losses."),
        ("Borrower may prepay the Notes.", "Borrower may prepay the Notes unless a Default exists."),
        ("License covers internal use.", "License covers internal use, except as restricted in Schedule A."),
        ("Security interest remains senior.", "Notwithstanding Section 2, security interest is subordinated."),
        ("Transfer requires consent.", "Transfer requires consent, subject to Section 4.1."),
        ("Dividends are permitted.", "Dividends are permitted, provided that Leverage stays low."),
        ("Assets are pledged to Lender.", "Assets other than Excluded Property are pledged to Lender."),
        ("Subsidiaries guarantee debt.", "Subsidiaries excluding Foreign Subs guarantee debt."),
        ("Proceeds fund operations.", "Proceeds shall be used solely for working capital."),
        ("Assignor retains rights.", "Assignor may assign only if Assignee assumes obligations."),
        ("Warranty survives closing.", "In no event shall Warranty survive closing."),
        (
            "Information stays confidential.\nFees are non-refundable.",
            "Information stays confidential.\nFees are non-refundable.\nNotwithstanding the foregoing, Regulatory disclosures are permitted.",
        ),
        (
            "Information stays confidential.\nExcept as prohibited by law, disclosure is allowed.",
            "Information stays confidential.",
        ),
    ]

    for idx, (txt_a, txt_b) in enumerate(mutations):
        da = SourceDocument(document_id=f"ma_{idx}", text=f"Section 1 Clause.\n{txt_a}", authority_rank=100, attribution="gov")
        db = SourceDocument(document_id=f"mb_{idx}", text=f"Section 1 Clause.\n{txt_b}", authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader())
        layer.ingest([da, db])
        dc = collect_reference_divergences([da, db])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        assert len(reg.gap_divergences) >= 1, f"Mutation #{idx} ({txt_a!r} -> {txt_b!r}) not localized!"

    # Also verify 12 clean OCR pairs under the dense reader emit 0 spurious section/gap warnings
    for idx in range(12):
        clean_a = f"Section 1 Obligations\nConfidential Information and obligations {idx} remain strictly pro-\ntected."
        clean_b = f"Section 1 Obligations\nCon\ufb01dential Information and obliga-\ntions {idx} remain strictly protected."
        da = SourceDocument(document_id=f"oa_{idx}", text=clean_a, authority_rank=100, attribution="gov")
        db = SourceDocument(document_id=f"ob_{idx}", text=clean_b, authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=policy, reader=_DenseClauseReader())
        layer.ingest([da, db])
        dc = collect_reference_divergences([da, db])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        assert dc.divergences == (), f"Clean OCR pair {idx} produced section divergence"
        assert reg.gap_divergences == (), f"Clean OCR pair {idx} produced gap divergence"


def test_financebench_v3_table_scale_headers_and_comparative_rows():
    # 1. Blank line + "(In millions, except per share data)" with $ on cell
    src1 = Source(
        id="10k_bs",
        text="CONSOLIDATED BALANCE SHEETS\n(In millions, except per share data)\n\nDecember 31, 2022\nLong-term debt   $12,500",
        tier="authoritative",
    )
    res1 = verify(
        [Assertion(subject="Company", predicate="long_term_debt", value="$12.5 billion", source_id="10k_bs", quote="Long-term debt   $12,500")],
        Ledger(sources=[src1]),
    )
    assert res1.status == "PASS", res1.issues

    # 2. "(Amounts in millions, except share and per share amounts)" + comparative numeric columns (2022 & 2021)
    src2 = Source(
        id="10k_is",
        text=(
            "CONSOLIDATED STATEMENTS OF INCOME\n"
            "(Amounts in millions, except share and per share amounts)\n\n"
            "Years ended December 31, 2022 and 2021\n"
            "Revenue   24,500   21,200"
        ),
        tier="authoritative",
    )
    res2_c1 = verify(
        [Assertion(subject="Company", predicate="revenue", value="$24.5 billion", source_id="10k_is", quote="Revenue   24,500   21,200")],
        Ledger(sources=[src2]),
    )
    res2_c2 = verify(
        [Assertion(subject="Company", predicate="revenue", value="$21.2 billion", source_id="10k_is", quote="Revenue   24,500   21,200")],
        Ledger(sources=[src2]),
    )
    assert res2_c1.status == "PASS", res2_c1.issues
    assert res2_c2.status == "PASS", res2_c2.issues

    # 3. "(in $ millions)" with bare year header & bare cell
    src3 = Source(
        id="10k_op",
        text="CONSOLIDATED STATEMENTS\n(in $ millions)\n\n2022\nOperating income   8,400",
        tier="authoritative",
    )
    res3 = verify(
        [Assertion(subject="Company", predicate="operating_income", value="$8.4 billion", source_id="10k_op", quote="Operating income   8,400")],
        Ledger(sources=[src3]),
    )
    assert res3.status == "PASS", res3.issues

    # 4. Negative controls: prose does not leak table scale, heading-only fails, subclause swap fails
    prose = Source(
        id="prose",
        text="The breach resulted in millions of dollars of losses. The administrative fee is $250.",
        tier="authoritative",
    )
    res_neg_prose = verify(
        [Assertion(subject="Company", predicate="fee", value="$250 million", source_id="prose", quote="The administrative fee is $250.")],
        Ledger(sources=[prose]),
    )
    assert res_neg_prose.status == "RETRY"

    res_neg_hdr = verify(
        [Assertion(subject="Company", predicate="section", value="1", source_id="10k_is", quote="Years ended")],
        Ledger(sources=[src2]),
    )
    assert res_neg_hdr.status == "RETRY"

    swap_src = Source(
        id="swap",
        text="Class A Notes pay 4.50% per annum; Class B Notes pay 5.80% per annum.",
        tier="authoritative",
    )
    res_neg_swap = verify(
        [Assertion(subject="Class A Notes", predicate="coupon_rate", value="5.80%", source_id="swap", quote="Class A Notes pay 4.50% per annum; Class B Notes pay 5.80% per annum.")],
        Ledger(sources=[swap_src]),
    )
    assert res_neg_swap.status == "RETRY"
