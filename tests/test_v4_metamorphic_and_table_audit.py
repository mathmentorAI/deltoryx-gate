"""Automated regression suite for P0-P3 layout symmetry, metamorphic invariance,
punctuation/bullet canonicalization, and row-type table scaling."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BENCH_DIR = _REPO_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from audit_p0_p3_v2_eval import (  # noqa: E402
    evaluate_p0_and_p1_metamorphic,
    evaluate_p2_bullets_and_spacing,
    evaluate_p3_disjoint_tables,
)
from claimlayer import Assertion, Ledger, Source, verify  # noqa: E402
from claimlayer.semantic.models import SourceDocument  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    AcceptancePolicy,
    EvidenceLayer,
    EvidenceRef,
    PredicateSpec,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domain_profile import DomainProfile  # noqa: E402
from claimlayer.v4.vocabulary import FieldSpec  # noqa: E402
from claimlayer.verifiers.diff import extract_divergent_sections_with_coverage  # noqa: E402


def _load_snapshot() -> dict:
    return json.loads((_BENCH_DIR / "metamorphic_edgar_snapshot.json").read_text(encoding="utf-8"))


def test_p0_symmetric_furniture_and_explicit_audit_fields():
    # Case 1: Candidate furniture appears in counterpart body (collision).
    # Symmetric filter must restore it in both versions, report exclusion_symmetric=True,
    # and yield zero false divergences.
    doc_v1 = (
        "ACME INDENTURE\n"
        "1. Covenant\n"
        "ACME INDENTURE\n"
        "The Borrower shall deliver quarterly statements within forty-five days.\n"
        "ACME INDENTURE\n"
    )
    doc_v2 = (
        "ACME INDENTURE 1. Covenant ACME INDENTURE The Borrower shall deliver quarterly "
        "statements within forty-five days. ACME INDENTURE"
    )
    report = extract_divergent_sections_with_coverage(
        Source(id="v1", text=doc_v1, authority_rank=100),
        Source(id="v2", text=doc_v2, authority_rank=80),
    )
    assert report.exclusion_symmetric is True
    assert report.exclusion_asymmetry is False
    assert report.divergences == ()
    assert abs(report.primary_coverage - 1.0) < 1e-9
    assert abs(report.secondary_coverage - 1.0) < 1e-9

    # Propagated through collect_reference_divergences -> CorpusReferenceRegister
    anchor = "\n\nSection 99: Anchor. Governing law is Delaware."
    vocab = Vocabulary(
        "sym_audit",
        "1.0",
        (PredicateSpec("governing_law", FieldSpec("string")),),
        entities=(("Party", ()),),
    )
    profile = DomainProfile("sym_audit", "1.0", vocab)
    p_doc = SourceDocument("v1.txt", doc_v1 + anchor, authority_rank=100, weight=0.95)
    s_doc = SourceDocument("v2.txt", doc_v2 + anchor, authority_rank=50, weight=0.90)
    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer.ingest([p_doc, s_doc])
    pos = p_doc.text.index("Governing law is Delaware.")
    layer.ingest_units([
        {
            "proposal": {"entity": "Party", "predicate": "governing_law", "value": "Delaware", "assertion_kind": "fact"},
            "basis": "explicit",
            "evidence": [EvidenceRef(p_doc.document_id, p_doc.revision, pos, pos + 26, "Governing law is Delaware.")],
        }
    ], attestor="sym-test")
    div_col = collect_reference_divergences([p_doc, s_doc], domain_profile=profile)
    assert div_col.exclusion_asymmetry is False
    assert isinstance(div_col.exclusion_symmetry_checks, tuple)

    reg = build_corpus_reference_register(
        layer,
        vocab,
        divergence_collection=div_col,
        domain_profile_fingerprint=profile.fingerprint,
    )
    reg_dict = reg.to_dict()
    assert reg.exclusion_asymmetry is False
    assert "exclusion_asymmetry" in reg_dict
    assert "exclusion_symmetry_checks" in reg_dict
    assert len(div_col.divergences) == 0
    assert len(reg.gap_divergences) == 0

    # Case 2: Running PDF page headers absent from plain counterpart get excluded cleanly
    # (no false content divergence) AND emit explicit exclusion_asymmetry audit flag.
    pdf_ver = "\n".join(
        f"Page {idx} of 4\n{idx}. Clause Body Text.\nItem value is {idx}00 dollars."
        for idx in range(1, 5)
    )
    plain_ver = " ".join(
        f"{idx}. Clause Body Text. Item value is {idx}00 dollars."
        for idx in range(1, 5)
    )
    asym_rep = extract_divergent_sections_with_coverage(
        Source(id="pdf", text=pdf_ver, authority_rank=100),
        Source(id="plain", text=plain_ver, authority_rank=80),
    )
    assert asym_rep.divergences == ()
    assert asym_rep.exclusion_asymmetry is True
    assert asym_rep.exclusion_symmetric is False
    assert asym_rep.exclusion_symmetry_check["asymmetry_detected"] is True
    assert len(asym_rep.exclusion_symmetry_check["primary_excluded_only"]) >= 1


def test_p1_metamorphic_edgar_suite_zero_fp_and_full_detection():
    snap = _load_snapshot()
    res = evaluate_p0_and_p1_metamorphic(snap)

    assert res["p0_symmetric_layout_clean"]["k"] == res["p0_symmetric_layout_clean"]["n"] == 10
    assert res["p1_shape_only_false_gap_rate"]["k"] == 0
    assert res["p1_shape_only_false_gap_rate"]["n"] == 90
    assert res["p1_combined_mutation_detection"]["k"] == 90
    assert res["p1_combined_mutation_detection"]["n"] == 90

    for tf_name, metrics in res["per_transform"].items():
        assert metrics["false_gap_rate"]["k"] == 0, f"False positive in {tf_name}"
        assert metrics["combined_detection_rate"]["k"] == metrics["combined_detection_rate"]["n"] == 10, (
            f"Missed mutation in {tf_name}"
        )


def test_p2_bullet_equivalence_and_punctuation_spacing():
    p2 = evaluate_p2_bullets_and_spacing()
    assert p2["bullet_style_equivalence"]["k"] == p2["bullet_style_equivalence"]["n"] == 8
    assert p2["bullet_identity_mutation_detected"]["k"] == p2["bullet_identity_mutation_detected"]["n"] == 8
    assert p2["spacing_verify_pass"]["k"] == p2["spacing_verify_pass"]["n"] == 10
    assert p2["spacing_verify_reject"]["k"] == p2["spacing_verify_reject"]["n"] == 10

    ev_clean = "Executed on December 16, 2011 for $5,000 across (30) days at 5.25% rate."
    ev_spaced = "Executed on December 16 , 2011 for $ 5,000 across ( 30 ) days at 5 . 25 % rate."
    ledger = Ledger([Source("ev", ev_clean, authority_rank=100)])
    assert verify([Assertion("Agreement", "exec_date", "2011-12-16", "ev", ev_spaced)], ledger).status == "PASS"
    assert verify([Assertion("Agreement", "fee", "$5,000", "ev", ev_spaced)], ledger).status == "PASS"
    assert verify([Assertion("Agreement", "cure", "30 days", "ev", ev_spaced)], ledger).status == "PASS"
    assert verify([Assertion("Agreement", "rate", "5.25%", "ev", ev_spaced)], ledger).status == "PASS"

    assert verify([Assertion("Agreement", "rate", "5.75%", "ev", ev_spaced)], ledger).status != "PASS"
    assert verify([Assertion("Agreement", "exec_date", "2011-12-17", "ev", ev_spaced)], ledger).status != "PASS"


def test_p3_disjoint_edgar_row_type_table_scaling():
    snap = _load_snapshot()
    p3 = evaluate_p3_disjoint_tables(snap)
    assert p3["ps_fail"] == []
    assert p3["ps_false_accept"] == []
    assert p3["mon_fail"] == []
    assert p3["per_share_unscaled_accepted"]["k"] == p3["per_share_unscaled_accepted"]["n"] == 12
    assert p3["per_share_scaled_rejected"]["k"] == p3["per_share_scaled_rejected"]["n"] == 12
    assert p3["monetary_scaled_accepted"]["k"] == p3["monetary_scaled_accepted"]["n"] == 12
