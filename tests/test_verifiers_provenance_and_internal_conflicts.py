"""Tests for Nov 16 verifier enhancements: internal quantity conflicts, heading case folding,
normalization provenance, and multi-column table year binding."""
from __future__ import annotations

from claimlayer import Assertion, Ledger, Source, verify
from claimlayer.verifiers.diff import extract_divergent_sections_with_coverage
from claimlayer.verifiers.quote_quantity import (
    canonicalize_quantity_notations_with_provenance,
    detect_internal_quantity_conflicts,
    joint_fold_heading_case,
)


def test_internal_quantity_conflicts_flag_mismatch_and_allow_matching():
    bad = "The Borrower shall cure within thirty (40) days of notice."
    good = "The Borrower shall cure within thirty (30) days of notice."

    conflicts = detect_internal_quantity_conflicts(bad)
    assert len(conflicts) == 1
    assert conflicts[0]["word_value"] == 30
    assert conflicts[0]["paren_value"] == 40
    assert detect_internal_quantity_conflicts(good) == []

    # verify() flags contradictory written-vs-parenthetical numerals even if assertion cites one side
    v_bad = verify(
        [Assertion(subject="Borrower", predicate="cure_period", value="30 days", source_id="s1", quote=bad)],
        Ledger(sources=[Source(id="s1", text=bad, tier="authoritative")]),
    )
    assert v_bad.status == "RETRY"
    assert any(iss.code == "QUANTITY_MISMATCH" for iss in v_bad.issues)

    # Matching form passes cleanly
    v_good = verify(
        [Assertion(subject="Borrower", predicate="cure_period", value="30 days", source_id="s1", quote=good)],
        Ledger(sources=[Source(id="s1", text=good, tier="authoritative")]),
    )
    assert v_good.status == "PASS"

    # extract_divergent_sections_with_coverage() always flags internal_quantity_conflict even on identical texts
    div_bad = extract_divergent_sections_with_coverage(
        Source(id="v1", text=bad, authority_rank=100),
        Source(id="v2", text=bad, authority_rank=80),
    )
    assert any(d.section_key == "internal_quantity_conflict" for d in div_bad.divergences)

    div_good = extract_divergent_sections_with_coverage(
        Source(id="v1", text=good, authority_rank=100),
        Source(id="v2", text=good, authority_rank=80),
    )
    assert div_good.divergences == ()


def test_joint_fold_heading_case_folds_headers_only_not_body_terms():
    v1_txt = "WITNESSETH:\n\n1. COVENANTS\nAn Affiliate shall maintain 10% reserves."
    v2_txt = "Witnesseth:\n\n1. Covenants\nAn Affiliate shall maintain 10% reserves."

    f1, f2, prov = joint_fold_heading_case(v1_txt, v2_txt)
    assert f1 == f2
    assert any(item["rule"] == "header_case_folding" for item in prov)

    rep_same_body = extract_divergent_sections_with_coverage(
        Source(id="v1", text=v1_txt, authority_rank=100),
        Source(id="v2", text=v2_txt, authority_rank=80),
    )
    assert rep_same_body.divergences == ()
    assert len(rep_same_body.normalization_provenance["joint"]) >= 1

    # Body defined-term case change ("Affiliate" vs "affiliate") must remain a detected divergence
    v2_body_diff = "Witnesseth:\n\n1. Covenants\nAn affiliate shall maintain 10% reserves."
    rep_body_diff = extract_divergent_sections_with_coverage(
        Source(id="v1", text=v1_txt, authority_rank=100),
        Source(id="v2", text=v2_body_diff, authority_rank=80),
    )
    assert len(rep_body_diff.divergences) >= 1


def test_normalization_provenance_records_equivalent_quantity_notations():
    txt_a = "Cure within thirty (30) days at 5 percent interest up to $1,000,000."
    txt_b = "Cure within 30 days at 5% interest up to $1000000."

    ca, prov_a = canonicalize_quantity_notations_with_provenance(txt_a)
    cb, prov_b = canonicalize_quantity_notations_with_provenance(txt_b)
    assert ca == cb
    rules_a = {r["rule"] for r in prov_a}
    assert {"word_paren_number", "percent_notation", "thousands_separator"} <= rules_a
    assert isinstance(prov_b, list)

    rep = extract_divergent_sections_with_coverage(
        Source(id="v1", text=txt_a, authority_rank=100),
        Source(id="v2", text=txt_b, authority_rank=80),
    )
    assert rep.divergences == ()
    rep_rules = {r["rule"] for r in rep.normalization_provenance["primary"]}
    assert "word_paren_number" in rep_rules


def test_table_header_years_bind_assertions_to_matching_column_period():
    table_doc = (
        "Selected Financial Data\n"
        "                  2024        2023\n"
        "Revenue        $24,500     $21,200\n"
    )
    row_quote = "Revenue        $24,500     $21,200"
    ledger = Ledger(sources=[Source(id="fin", text=table_doc, tier="authoritative")])

    # Naming 2024 with the 2024 figure ($24,500) must PASS
    res_ok = verify(
        [Assertion(subject="Revenue_2024", predicate="revenue", value="$24,500", source_id="fin", quote=row_quote)],
        ledger,
    )
    assert res_ok.status == "PASS"

    # Naming 2024 with the 2023 figure ($21,200) must fail closed
    res_wrong_col = verify(
        [Assertion(subject="Revenue_2024", predicate="revenue", value="$21,200", source_id="fin", quote=row_quote)],
        ledger,
    )
    assert res_wrong_col.status == "RETRY"
    assert any(iss.code == "QUANTITY_MISMATCH" for iss in res_wrong_col.issues)
