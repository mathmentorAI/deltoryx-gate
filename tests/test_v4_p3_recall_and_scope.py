"""Verification suite for Priority 3 (#10-#12): deleted clauses, span-overlap un-masking,
and v3 verify() local quantifier / restrictor / negation scope checks.
"""
from __future__ import annotations

from claimlayer import Assertion, Ledger, Policy, Source
from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    AcceptancePolicy,
    EvidenceLayer,
    FieldSpec,
    PredicateSpec,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domain_profile import DomainProfile


def _span_of(text: str, sub: str) -> tuple[int, int]:
    idx = text.find(sub)
    assert idx >= 0, f"Substring {sub!r} not found"
    return idx, idx + len(sub)


# ---------------------------------------------------------------------------
# P3 #10: Deleted clauses (plain sentence, numeric clause, exception clause)
# ---------------------------------------------------------------------------

def test_p3_10_deleted_clauses_sentence_numeric_and_exception():
    vocab = Vocabulary(
        "p3_deleted",
        "1.0",
        (
            PredicateSpec("governing_law", FieldSpec("string")),
            PredicateSpec("liquidity_reserve", FieldSpec("quantity", dimensions=("currency:USD",))),
            PredicateSpec("debt_cap", FieldSpec("quantity", dimensions=("currency:USD",))),
            PredicateSpec("guarantor_obligation", FieldSpec("string")),
        ),
        entities=(("Borrower", ()), ("Guarantor", ())),
    )
    profile = DomainProfile(profile_id="p3_deleted", version="1.0", vocabulary=vocab)

    def _run_pair(pri_body: str, sec_body: str, units_builder):
        anchor = "Section 99. Governing Law: This Agreement is governed by New York law."
        pri_txt = f"Section 1. Covenants: {pri_body}\n\n{anchor}"
        sec_txt = f"Section 1. Covenants: {sec_body}\n\n{anchor}"
        pri = SourceDocument("indenture.txt", pri_txt, authority_rank=100, weight=0.95)
        sec = SourceDocument("draft.txt", sec_txt, authority_rank=50, weight=0.90)
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        layer.ingest([pri, sec])
        as_, ae_ = _span_of(pri_txt, "This Agreement is governed by New York law.")
        items = [
            {
                "proposal": {
                    "entity": "Borrower",
                    "predicate": "governing_law",
                    "value": "New York law",
                    "assertion_kind": "fact",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(pri.document_id, pri.revision, as_, ae_, "This Agreement is governed by New York law.")],
            }
        ]
        items.extend(units_builder(pri, pri_txt))
        layer.ingest_units(items, attestor="p3-seed")
        div_col = collect_reference_divergences([pri, sec], domain_profile=profile)
        return build_corpus_reference_register(
            layer,
            vocab,
            divergence_collection=div_col,
            domain_profile_fingerprint=profile.fingerprint,
        )

    # 1. Deleted non-numeric obligation sentence (including "Clause 3:" prefix that previously looked like a heading)
    reg_sent = _run_pair(
        "Interest accrues quarterly on each Payment Date. Clause 3: Guarantor unconditionally guarantees all payment obligations upon an Event of Default.",
        "Interest accrues quarterly on each Payment Date.",
        lambda _pri, _txt: [],
    )
    assert reg_sent.available is True
    deleted_sent_gaps = [g for g in reg_sent.gap_divergences if g.get("token") == "DELETED_CLAUSE"]
    assert deleted_sent_gaps, f"Expected DELETED_CLAUSE for omitted sentence, got {reg_sent.gap_divergences}"
    assert any("Guarantor unconditionally guarantees" in g.get("sentence", "") for g in deleted_sent_gaps)

    # 2. Deleted exception clause
    reg_exc = _run_pair(
        "Distributions to Class C are permitted monthly. Except during a Trigger Event, excess spread flows to Certificateholder.",
        "Distributions to Class C are permitted monthly.",
        lambda _pri, _txt: [],
    )
    assert any(
        g.get("token") == "DELETED_CLAUSE" and "Except during a Trigger Event" in g.get("sentence", "")
        for g in reg_exc.gap_divergences
    )

    # 3. Deleted numeric clause even when an unrelated unit cites the full Section 1 span
    def _unrelated_unit(pri: SourceDocument, pri_txt: str):
        s_s, s_e = _span_of(pri_txt, "The Borrower shall maintain a Debt Cap of $10,000,000.")
        return [
            {
                "proposal": {
                    "entity": "Borrower",
                    "predicate": "debt_cap",
                    "value": 10000000,
                    "unit": "USD",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(pri.document_id, pri.revision, s_s, s_e, pri_txt[s_s:s_e])],
            }
        ]

    reg_num = _run_pair(
        "The Borrower shall maintain a Debt Cap of $10,000,000. The Borrower shall maintain a Liquidity Reserve of $5,000,000.",
        "The Borrower shall maintain a Debt Cap of $10,000,000.",
        _unrelated_unit,
    )
    assert any(g.get("kind") == "quantity" and "5000000" in str(g.get("token")) for g in reg_num.gap_divergences)
    assert any(g.get("token") == "DELETED_CLAUSE" for g in reg_num.gap_divergences)

    # 4. Negative / clean control: when a deleted clause IS validly covered by a matching unit on pri, 0 gaps fire
    def _matching_unit(pri: SourceDocument, pri_txt: str):
        sub = "Guarantor unconditionally guarantees all payment obligations."
        s_s, s_e = _span_of(pri_txt, sub)
        return [
            {
                "proposal": {
                    "entity": "Guarantor",
                    "predicate": "guarantor_obligation",
                    "value": "guarantees payment obligations",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(pri.document_id, pri.revision, s_s, s_e, sub)],
            }
        ]

    reg_clean = _run_pair(
        "Interest accrues quarterly. Guarantor unconditionally guarantees all payment obligations.",
        "Interest accrues quarterly.",
        _matching_unit,
    )
    assert reg_clean.gap_divergences == ()


# ---------------------------------------------------------------------------
# P3 #11: Dense 1-unit-per-sentence benchmark eliminating span-overlap masking
# ---------------------------------------------------------------------------

def test_p3_11_dense_1_unit_per_sentence_eliminates_span_overlap_masking():
    vocab = Vocabulary(
        "p3_dense_mask",
        "1.0",
        (
            PredicateSpec("cash_reserve", FieldSpec("quantity", dimensions=("currency:USD",))),
            PredicateSpec("debt_cap", FieldSpec("quantity", dimensions=("currency:USD",))),
            PredicateSpec("coupon_rate", FieldSpec("quantity", dimensions=("ratio",))),
            PredicateSpec("cure_period", FieldSpec("quantity", dimensions=("duration_calendar_days",))),
        ),
        entities=(("Issuer", ()),),
    )
    profile = DomainProfile(profile_id="p3_dense_mask", version="1.0", vocabulary=vocab)

    pri_clauses = [
        # 3 clean controls where second sentence part is identical (1 unit per sentence citing whole sentence)
        "Clause 1: Cash Reserve equals $2,000,000 and Trustee fees are paid monthly.",
        "Clause 2: Cash Reserve equals $3,000,000 and Servicing fees are paid monthly.",
        "Clause 3: Cash Reserve equals $4,000,000 and Collateral reports are delivered quarterly.",
        # 3 masked NUMERIC conflicts: extracted Cash Reserve cites whole sentence, while Debt Cap / Cure / Coupon differ unextracted!
        "Clause 4: Cash Reserve equals $2,000,000 and Debt Cap equals $10,000,000.",
        "Clause 5: Cash Reserve equals $2,500,000 and Cure Period equals 5 days.",
        "Clause 6: Cash Reserve equals $3,500,000 and Base Spread coupon equals 4.25%.",
        # 3 masked NEGATION / POLARITY flips: extracted coupon_rate conflict cites whole sentence while negation flips!
        "Clause 7: Class B coupon is 5.80% and Guarantor shall not be released prior to maturity.",
        "Clause 8: Class A coupon is 5.10% and Collateral pledge is prohibited without consent.",
        "Clause 9: Class C coupon is 6.75% and Event of Default is never waived automatically.",
    ]
    sec_clauses = [
        "Clause 1: Cash Reserve equals $2,000,000 and Trustee fees are paid monthly.",
        "Clause 2: Cash Reserve equals $3,000,000 and Servicing fees are paid monthly.",
        "Clause 3: Cash Reserve equals $4,000,000 and Collateral reports are delivered quarterly.",
        "Clause 4: Cash Reserve equals $2,000,000 and Debt Cap equals $15,000,000.",
        "Clause 5: Cash Reserve equals $2,500,000 and Cure Period equals 30 days.",
        "Clause 6: Cash Reserve equals $3,500,000 and Base Spread coupon equals 4.75%.",
        "Clause 7: Class B coupon is 5.55% and Guarantor shall be released prior to maturity.",
        "Clause 8: Class A coupon is 4.90% and Collateral pledge is permitted without consent.",
        "Clause 9: Class C coupon is 6.25% and Event of Default is waived automatically.",
    ]

    pri_txt = "Section 1. Dense Financial Schedule:\n" + "\n".join(pri_clauses)
    sec_txt = "Section 1. Dense Financial Schedule:\n" + "\n".join(sec_clauses)
    pri = SourceDocument("master.txt", pri_txt, authority_rank=100, weight=0.95)
    sec = SourceDocument("draft.txt", sec_txt, authority_rank=50, weight=0.90)

    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer.ingest([pri, sec])

    units_to_ingest = []
    # Clauses 1..6: single Cash Reserve unit citing the ENTIRE sentence on BOTH docs
    cash_vals = [2000000, 3000000, 4000000, 2000000, 2500000, 3500000]
    for i, cval in enumerate(cash_vals):
        for sdoc, s_sent in ((pri, pri_clauses[i]), (sec, sec_clauses[i])):
            ss, se = _span_of(sdoc.text, s_sent)
            units_to_ingest.append({
                "proposal": {
                    "entity": "Issuer",
                    "predicate": "cash_reserve",
                    "value": cval,
                    "unit": "USD",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(sdoc.document_id, sdoc.revision, ss, se, s_sent)],
            })
    # Clauses 7..9: extracted coupon_rate conflict citing the ENTIRE sentence on BOTH docs (trying to mask negation flips!)
    coupon_pairs = [(5.8, 5.55), (5.1, 4.9), (6.75, 6.25)]
    for offset, (p_rate, s_rate) in enumerate(coupon_pairs):
        idx = 6 + offset
        for sdoc, s_sent, rate in ((pri, pri_clauses[idx], p_rate), (sec, sec_clauses[idx], s_rate)):
            ss, se = _span_of(sdoc.text, s_sent)
            units_to_ingest.append({
                "proposal": {
                    "entity": "Issuer",
                    "predicate": "coupon_rate",
                    "value": rate,
                    "unit": "percent",
                    "assertion_kind": "fact",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(sdoc.document_id, sdoc.revision, ss, se, s_sent)],
            })

    summary = layer.ingest_units(units_to_ingest, attestor="dense-reader")
    assert summary["proposals_rejected"] == 0

    div_col = collect_reference_divergences([pri, sec], domain_profile=profile)
    reg = build_corpus_reference_register(
        layer,
        vocab,
        divergence_collection=div_col,
        domain_profile_fingerprint=profile.fingerprint,
    )
    assert reg.available is True
    gap_sentences = " || ".join(str(g.get("sentence", "")) for g in reg.gap_divergences)

    # All 3 clean sentences MUST stay free of gaps
    for clean_c in pri_clauses[:3]:
        assert clean_c not in gap_sentences

    # All 3 masked numeric conflicts MUST be detected despite overlapping Cash Reserve unit spans
    assert any("10000000" in str(g.get("token")) or "15000000" in str(g.get("token")) for g in reg.gap_divergences)
    assert any("5:DAY" in str(g.get("token")) or "30:DAY" in str(g.get("token")) for g in reg.gap_divergences)
    assert any("4.25:PERCENT" in str(g.get("token")) or "4.75:PERCENT" in str(g.get("token")) for g in reg.gap_divergences)

    # All 3 masked negation / polarity flips MUST emit textual gaps despite overlapping coupon_rate conflicts
    for idx in (6, 7, 8):
        assert any(
            g.get("kind") == "textual" and (pri_clauses[idx] in str(g.get("sentence")) or sec_clauses[idx] in str(g.get("sentence")))
            for g in reg.gap_divergences
        ), f"Negation flip in Clause {idx + 1} was masked by overlapping coupon_rate conflict!"


# ---------------------------------------------------------------------------
# P3 #12: v3 verify() quantifier / restrictor / negation local scope checks
# ---------------------------------------------------------------------------

def test_p3_12_v3_verify_local_quantifier_restrictor_and_negation_scope():
    def _verify_one(src_text: str, subject: str, predicate: str, value: str, quote: str):
        ledger = Ledger([Source(id="doc1", title="Contract", text=src_text, tier="authoritative")])
        return ledger.verify(
            assertions=[
                Assertion(
                    id="a1",
                    subject=subject,
                    predicate=predicate,
                    value=value,
                    source_id="doc1",
                    quote=quote,
                )
            ],
            policy=Policy(require_exact_quote=True, require_quantity_match=True),
        )

    # 1. Exact conditional "only if" attack: "X shall only include Y" against "X includes Y and Z only if conditions apply."
    v_only_if = _verify_one(
        "Eligible Assets includes cash and treasury bills only if conditions apply.",
        "Eligible Assets",
        "scope",
        "Eligible Assets shall only include cash",
        "Eligible Assets includes cash and treasury bills only if conditions apply.",
    )
    assert v_only_if.status == "RETRY"

    # 2. Restrictor scoped to a different clause/verb: "Borrower requires insurance and only waives reporting" vs "Borrower only requires insurance"
    v_swap = _verify_one(
        "Borrower requires insurance and only waives reporting upon closing.",
        "Borrower",
        "covenant",
        "Borrower only requires insurance",
        "Borrower requires insurance and only waives reporting upon closing.",
    )
    assert v_swap.status == "RETRY"

    # 3. Comparative bound reversal (`at least` vs `at most`, `not more than` vs `at least`)
    v_bound1 = _verify_one(
        "Notice must be delivered at least 5 business days prior to Closing.",
        "Notice",
        "notice_period",
        "at most 5 business days",
        "Notice must be delivered at least 5 business days prior to Closing.",
    )
    assert v_bound1.status == "RETRY"

    v_bound2 = _verify_one(
        "Facility borrowings shall not exceed $10,000,000 at any time.",
        "Facility",
        "borrowing_limit",
        "at least $10,000,000",
        "Facility borrowings shall not exceed $10,000,000 at any time.",
    )
    assert v_bound2.status == "RETRY"

    # 4. Universal vs partial / empty quantifier reversal
    v_univ = _verify_one(
        "Some Collateral items remain excluded from the borrowing base.",
        "Collateral",
        "inclusion",
        "all Collateral items remain in the borrowing base",
        "Some Collateral items remain excluded from the borrowing base.",
    )
    assert v_univ.status == "RETRY"

    v_empty = _verify_one(
        "Lenders are entitled to default interest upon any payment default.",
        "Lenders",
        "entitlement",
        "Lenders are never entitled to default interest",
        "Lenders are entitled to default interest upon any payment default.",
    )
    assert v_empty.status == "RETRY"

    # 5. Unrestricted universal claim against an exclusive restrictor
    v_rev_only = _verify_one(
        "Collateral solely comprises cash reserves.",
        "Collateral",
        "composition",
        "Collateral comprises all cash reserves",
        "Collateral solely comprises cash reserves.",
    )
    assert v_rev_only.status == "RETRY"

    # 6. Numeric clause negation inversion (must be checked even when quantities exist!)
    v_num_neg = _verify_one(
        "Guarantor shall not pay $10,000,000 without written Lender consent.",
        "Guarantor",
        "payment_duty",
        "Guarantor shall pay $10,000,000",
        "Guarantor shall not pay $10,000,000 without written Lender consent.",
    )
    assert v_num_neg.status == "RETRY"

    # 7. Positive controls (utility must PASS!)
    positives = [
        (
            "Class B Notes bear interest at 5.80% per annum.",
            "Class B Notes",
            "coupon_rate",
            "5.80%",
            "Class B Notes bear interest at 5.80% per annum.",
        ),
        (
            "Notice must be delivered not less than 5 business days prior to Closing.",
            "Notice",
            "notice_period",
            "at least 5 business days",
            "Notice must be delivered not less than 5 business days prior to Closing.",
        ),
        (
            "Exemption applies solely to Class A Notes.",
            "Class A Notes",
            "exemption",
            "solely to Class A Notes",
            "Exemption applies solely to Class A Notes.",
        ),
        (
            "All Senior Notes rank pari passu in right of payment.",
            "Senior Notes",
            "priority",
            "All Senior Notes rank pari passu",
            "All Senior Notes rank pari passu in right of payment.",
        ),
    ]
    for txt, subj, pred, val, q in positives:
        v_ok = _verify_one(txt, subj, pred, val, q)
        assert v_ok.status == "PASS", f"False REJECT on valid claim {val!r}: {v_ok.issues}"
