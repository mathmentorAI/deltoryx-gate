from __future__ import annotations

import json
from pathlib import Path

from claimlayer import Assertion, Ledger, Policy, Source, verify
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
from claimlayer.verifiers.diff import (
    extract_divergent_sections,
    extract_divergent_sections_with_coverage,
)


def _span_of(text: str, quote: str) -> tuple[int, int]:
    idx = text.index(quote)
    return idx, idx + len(quote)


def _run_pair_with_units(pri_txt: str, sec_txt: str, unit_builder):
    vocab = Vocabulary(
        "audit_p2_vocab",
        "1.0",
        (
            PredicateSpec("indemnity_clause", FieldSpec("string")),
            PredicateSpec("confidentiality_term", FieldSpec("string")),
            PredicateSpec("consent_requirement", FieldSpec("string")),
            PredicateSpec("payment_term", FieldSpec("string")),
            PredicateSpec("interest_rate", FieldSpec("quantity", dimensions=("ratio",))),
            PredicateSpec("fee_basis", FieldSpec("string")),
            PredicateSpec("governing_law", FieldSpec("string")),
        ),
        entities=(("Client", ()), ("Subcontractors", ()), ("Borrower", ())),
    )
    profile = DomainProfile(profile_id="audit_p2", version="1.0", vocabulary=vocab)
    anchor = "Governing Law: New York law applies."
    full_pri = f"{pri_txt.rstrip()}\n\n{anchor}"
    full_sec = f"{sec_txt.rstrip()}\n\n{anchor}"
    pri = SourceDocument("master_agreement.txt", full_pri, authority_rank=100, weight=0.95)
    sec = SourceDocument("draft_redline.txt", full_sec, authority_rank=50, weight=0.90)
    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer.ingest([pri, sec])
    a_s, a_e = _span_of(full_pri, anchor)
    items = [
        {
            "proposal": {
                "entity": "Borrower",
                "predicate": "governing_law",
                "value": "New York law",
                "assertion_kind": "fact",
            },
            "basis": "explicit",
            "evidence": [EvidenceRef(pri.document_id, pri.revision, a_s, a_e, anchor)],
        }
    ]
    extra = unit_builder(pri, pri, sec)
    if extra:
        items.extend(extra)
    layer.ingest_units(items, attestor="audit-seed")
    div_col = collect_reference_divergences([pri, sec], domain_profile=profile)
    return build_corpus_reference_register(
        layer,
        vocab,
        divergence_collection=div_col,
        domain_profile_fingerprint=profile.fingerprint,
        expected_documents={pri.document_id: pri, sec.document_id: sec},
    )


# ---------------------------------------------------------------------------
# P0 (#1): Fail-closed 100% character coverage section diff
# ---------------------------------------------------------------------------


def test_p0_100pct_coverage_and_off_section_mutations_detected():
    pri_text = (
        "Preamble: Customer shall indemnify Vendor without monetary cap.\n\n"
        "Section 1: Services. Vendor shall deliver the software.\n"
        "Section 2: Governing Law. Delaware.\n\n"
        "Trailing Schedule A: Support hours are 24/7.\n"
    )
    sec_text = (
        "Preamble: Customer shall not indemnify Vendor under any circumstances.\n\n"
        "Section 1: Services. Vendor shall deliver the software.\n"
        "Section 2: Governing Law. Delaware.\n\n"
        "Trailing Schedule A: Support hours are 24/7.\n"
    )
    report = extract_divergent_sections_with_coverage(
        Source(id="master", text=pri_text, authority_rank=100),
        Source(id="draft", text=sec_text, authority_rank=10),
        fail_on_incomplete_coverage=True,
    )
    assert report.primary_coverage == 1.0
    assert report.secondary_coverage == 1.0
    assert report.primary_uncovered_spans == ()
    assert report.secondary_uncovered_spans == ()
    keys = [d.section_key for d in report.divergences]
    assert "unsegmented_remainder" in keys


def test_p0_duplicate_headers_preserved_independently_and_asymmetric_prose_fails_closed():
    dup_pri = (
        "EXHIBIT A\nInitial Deposit: $10,000.\n"
        "EXHIBIT A\nFinal Settlement: $25,000.\n"
    )
    dup_sec = (
        "EXHIBIT A\nInitial Deposit: $10,000.\n"
        "EXHIBIT A\nFinal Settlement: $35,000.\n"
    )
    divs = extract_divergent_sections(
        Source(id="pri", text=dup_pri, authority_rank=100),
        Source(id="sec", text=dup_sec, authority_rank=10),
    )
    assert any(d.section_key.startswith("exhibit_a__occ") for d in divs)

    # Segmented primary vs unsegmented counterpart with identical copied sections + hidden extra clause
    seg_p = "Section 1: Fees. Base fee is $5,000.\nSection 2: Term. 12 months.\n"
    prose_s = (
        "Summary Memorandum\n"
        "Section 1: Fees. Base fee is $5,000.\n"
        "Section 2: Term. 12 months.\n"
        "Borrower may prepay zero-penalty at any time without notice.\n"
    )
    asym_divs = extract_divergent_sections(
        Source(id="pri", text=seg_p, authority_rank=100),
        Source(id="sec", text=prose_s, authority_rank=10),
    )
    assert len(asym_divs) >= 1
    assert any(d.section_key == "unsegmented_remainder" for d in asym_divs)


def test_p0_irregular_layouts_caps_and_alpha_lists_and_edgar_snapshot():
    p_irr = "PAYMENT OBLIGATIONS\na. Buyer shall remit $15,000 within 10 days.\n(i) Late fees accrue at 2%.\n"
    s_irr = "PAYMENT OBLIGATIONS\na. Buyer shall remit $15,000 within 10 days.\n(i) Late fees do not accrue.\n"
    report = extract_divergent_sections_with_coverage(
        Source(id="p", text=p_irr, authority_rank=100),
        Source(id="s", text=s_irr, authority_rank=10),
    )
    assert report.primary_coverage == 1.0
    assert report.secondary_coverage == 1.0
    assert len(report.divergences) >= 1

    snap_path = Path(__file__).resolve().parents[1] / "benchmarks" / "unseen_edgar_snapshot.json"
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    for c in snap["contracts"][:3]:
        src = Source(id=c["id"], text=c["text"], authority_rank=100)
        rep = extract_divergent_sections_with_coverage(src, Source(id="dup", text=c["text"], authority_rank=80))
        assert rep.primary_coverage == 1.0
        assert rep.secondary_coverage == 1.0


# ---------------------------------------------------------------------------
# P1 (#2 & #3): Structural PDF furniture exclusion & hyphen reconstruction
# ---------------------------------------------------------------------------


def test_p1_pdf_furniture_stripped_with_provenance_and_negative_controls():
    p_pdf = (
        "Confidential - Series 2026 Draft\n"
        "1\n"
        "Borrower shall maintain a cash reserve of $2,500,000 at all times under Section 4.02 and deliver quarterly compliance certificates to the Trustee.\n"
        "Page 1 of 3\n"
        "Confidential - Series 2026 Draft\n"
        "2\n"
        "Maturity of the Senior Notes occurs in December 2029 unless redeemed earlier in accordance with the optional redemption schedule.\n"
        "Page 2 of 3\n"
        "Confidential - Series 2026 Draft\n"
        "3\n"
        "No prepayment penalty applies after the second anniversary of closing.\n"
        "Page 3 of 3\n"
    )
    s_clean = (
        "Borrower shall maintain a cash reserve of $2,500,000 at all times under Section 4.02 and deliver quarterly compliance certificates to the Trustee.\n"
        "Maturity of the Senior Notes occurs in December 2029 unless redeemed earlier in accordance with the optional redemption schedule.\n"
        "No prepayment penalty applies after the second anniversary of closing.\n"
    )
    rep = extract_divergent_sections_with_coverage(
        Source(id="pdf_src", text=p_pdf, authority_rank=100),
        Source(id="clean_src", text=s_clean, authority_rank=80),
    )
    assert rep.divergences == ()
    assert len(rep.primary_excluded_furniture) >= 3

    # Negative control: single-occurrence standalone number or clause must NOT be dropped as furniture
    p_single = "Section 1: Threshold.\n5\nBorrower shall not transfer assets."
    s_single = "Section 1: Threshold.\n15\nBorrower shall not transfer assets."
    rep_neg = extract_divergent_sections_with_coverage(
        Source(id="p_neg", text=p_single, authority_rank=100),
        Source(id="s_neg", text=s_single, authority_rank=80),
    )
    assert len(rep_neg.divergences) >= 1
    assert rep_neg.primary_excluded_furniture == ()


def test_p1_linebreak_hyphen_reconstruction_preserves_compounds_and_catches_mutations():
    p_hy = (
        "Vendor grants a non-\nexclusive, royalty-\nfree license covering all docu-\n"
        "ments for a two-\nyear initial period.\n"
    )
    s_clean = "Vendor grants a non-exclusive, royalty-free license covering all documents for a two-year initial period.\n"
    rep_clean = extract_divergent_sections_with_coverage(
        Source(id="hy", text=p_hy, authority_rank=100),
        Source(id="cl", text=s_clean, authority_rank=80),
    )
    assert rep_clean.divergences == ()

    # Soft hyphen & non-breaking hyphen reflow clean quote verification
    s_soft = "Licensor provides a non\u00ad\nexclusive worldwide license for docu\u2010\nments."
    v_ok = verify(
        [
            Assertion(
                subject="Licensor",
                predicate="license_scope",
                value="non-exclusive worldwide license for documents",
                source_id="soft",
                quote="Licensor provides a non-exclusive worldwide license for documents.",
            )
        ],
        Ledger(sources=[Source(id="soft", text=s_soft, authority_rank=100)]),
        Policy(policy_id="hy_pol", min_tier_rank=10),
    )
    assert v_ok.status == "PASS"

    # Negative control: non-exclusive -> exclusive is a real mutation and must diverge
    s_mut = "Vendor grants an exclusive, royalty-free license covering all documents for a two-year initial period.\n"
    rep_mut = extract_divergent_sections_with_coverage(
        Source(id="hy", text=p_hy, authority_rank=100),
        Source(id="mut", text=s_mut, authority_rank=80),
    )
    assert len(rep_mut.divergences) >= 1


# ---------------------------------------------------------------------------
# P2 (#4 & #5): Dense reader DELETED_CLAUSE & compute-unit/basis diffs
# ---------------------------------------------------------------------------


def test_p2_dense_reader_does_not_mask_deleted_clause_and_catches_basis_diffs():
    pri_text = (
        "Client shall pay invoices on schedule.\n"
        "Subcontractors require prior written consent of Client.\n"
    )
    sec_text = "Client shall pay invoices on schedule.\n"

    def _dense_pri_only(doc, pri, sec):
        if doc.document_id != pri.document_id:
            return []
        s1, e1 = _span_of(pri.text, "Client shall pay invoices on schedule.")
        s2, e2 = _span_of(pri.text, "Subcontractors require prior written consent of Client.")
        return [
            {
                "proposal": {
                    "entity": "Client",
                    "predicate": "payment_term",
                    "value": "on schedule",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(pri.document_id, pri.revision, s1, e1, "Client shall pay invoices on schedule.")],
            },
            {
                "proposal": {
                    "entity": "Subcontractors",
                    "predicate": "consent_requirement",
                    "value": "prior written consent of Client",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(pri.document_id, pri.revision, s2, e2, "Subcontractors require prior written consent of Client.")],
            },
        ]

    reg_del = _run_pair_with_units(pri_text, sec_text, _dense_pri_only)
    assert any(g.get("token") == "DELETED_CLAUSE" and "Subcontractors" in str(g.get("sentence")) for g in reg_del.gap_divergences)

    # Compute-unit / basis modifier diffs must never collapse cleanly
    reg_days = _run_pair_with_units(
        "Borrower shall pay within 30 calendar days of invoice.",
        "Borrower shall pay within 30 business days of invoice.",
        lambda doc, pri, sec: [],
    )
    assert any(g.get("kind") == "quantity" for g in reg_days.gap_divergences)

    reg_basis = _run_pair_with_units(
        "Royalty equals 5% of net sales payable per annum.",
        "Royalty equals 5% of gross sales payable per month.",
        lambda doc, pri, sec: [],
    )
    assert any(g.get("kind") == "textual" for g in reg_basis.gap_divergences)

    # Negative control: matching days/units cleanly covered produces 0 gaps
    reg_match = _run_pair_with_units(
        "Borrower shall pay within 30 calendar days of invoice.",
        "Borrower shall pay within 30 calendar days of invoice.",
        lambda doc, pri, sec: [],
    )
    assert reg_match.gap_divergences == ()


# ---------------------------------------------------------------------------
# P3 (#6): Normalized-token table scale headers & exception clauses
# ---------------------------------------------------------------------------


def test_p3_normalized_table_headers_and_exception_clauses():
    # Parenthetical OCR-collapsed header with date tail & per-share exception
    tbl_text = (
        "CONSOLIDATED STATEMENTS OF INCOME\n"
        "($ in millions, except per share data)\n"
        "Three Months Ended August 1, 2026\n"
        "Total net revenue    $4,250\n"
        "Operating income     $1,120\n"
        "Diluted earnings per share   $2.45\n"
        "Common stock par value per share  $0.0001\n"
        "Gross margin percentage    58.4%\n"
    )
    src = Source(id="tbl_10q", text=tbl_text, authority_rank=100)
    pol = Policy(policy_id="tbl_pol", min_tier_rank=10)

    # Scaled revenue passes exact Decimal ($4,250M = $4,250,000,000); unscaled EPS/par/margin pass
    v_pass = verify(
        [
            Assertion("Co", "revenue", "$4,250,000,000", "tbl_10q", "Total net revenue    $4,250"),
            Assertion("Co", "diluted_eps", "$2.45", "tbl_10q", "Diluted earnings per share   $2.45"),
            Assertion("Co", "par_value", "$0.0001", "tbl_10q", "Common stock par value per share  $0.0001"),
            Assertion("Co", "gross_margin", "58.4%", "tbl_10q", "Gross margin percentage    58.4%"),
        ],
        Ledger(sources=[src]),
        pol,
    )
    assert v_pass.status == "PASS"

    # Reject wrong unscaled revenue ($4,250 vs $4,250,000,000) and reject wrongly-scaled EPS ($2,450,000)
    v_bad_rev = verify(
        [Assertion("Co", "revenue", "$4,250", "tbl_10q", "Total net revenue    $4,250")],
        Ledger(sources=[src]),
        pol,
    )
    assert v_bad_rev.status != "PASS"

    v_bad_eps = verify(
        [Assertion("Co", "diluted_eps", "$2,450,000", "tbl_10q", "Diluted earnings per share   $2.45")],
        Ledger(sources=[src]),
        pol,
    )
    assert v_bad_eps.status != "PASS"
