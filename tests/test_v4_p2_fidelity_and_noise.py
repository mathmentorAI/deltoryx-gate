"""Verification suite for Priority 2 (items #5-#9): tabular inheritance, KV lists,
omitted condition_status equivalence, typed versioned normalizers, and PDF/OCR noise.
"""
from __future__ import annotations

import json
import math
import re

from claimlayer.semantic.models import EvidenceRef, SourceDocument
from claimlayer.v4 import (
    AcceptancePolicy,
    EvidenceLayer,
    EvidenceUnit,
    FieldEvidence,
    FieldSpec,
    PredicateSpec,
    Vocabulary,
    verify_extraction_fidelity,
)
from claimlayer.v4.corpus_obligations import (
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domain_profile import DomainProfile
from claimlayer.v4.engine import evaluate_unit_admissibility
from claimlayer.v4.normalizers import (
    NORMALIZER_VERSION,
    normalize_date_token,
    normalize_percentage,
    normalize_separator_number,
    normalize_time_interval,
    normalize_value_with_provenance,
)
from claimlayer.v4.vocabulary import label_key


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return max(0.0, round(center - half, 4)), min(1.0, round(center + half, 4))


def _span_of(text: str, sub: str, *, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    pos = 0
    for _ in range(occurrence + 1):
        start = text.find(sub, pos)
        assert start >= 0, f"Substring {sub!r} not found"
        pos = start + 1
    return start, start + len(sub)


# ---------------------------------------------------------------------------
# P2 #5: Explicit audited tabular header inheritance (no silent inference)
# ---------------------------------------------------------------------------

def test_p2_5_tabular_header_inheritance_positive_and_negative():
    vocab = Vocabulary(
        "fin_table",
        "1.0",
        (
            PredicateSpec("revenue", FieldSpec("quantity", dimensions=("currency:USD", "currency:EUR"))),
            PredicateSpec("coupon_rate", FieldSpec("quantity", dimensions=("ratio",))),
        ),
        entities=(("Acme Holdings", ("Acme",)),),
    )
    policy = AcceptancePolicy()
    doc_text = (
        "Acme Holdings Financial Summary\n"
        "| Metric | FY 2023 ($ in thousands) | FY 2024 ($ in thousands) |\n"
        "| --- | --- | --- |\n"
        "| Revenue | 120 | 150 |\n"
    )
    doc = SourceDocument("fin_stmt.md", doc_text, authority_rank=100, weight=0.95)
    hdr_sub = "| Metric | FY 2023 ($ in thousands) | FY 2024 ($ in thousands) |"
    row_sub = "| Revenue | 120 | 150 |"
    hs, he = _span_of(doc_text, hdr_sub)
    rs, re_ = _span_of(doc_text, row_sub)
    hdr_ref = EvidenceRef(doc.document_id, doc.revision, hs, he, hdr_sub)
    row_ref = EvidenceRef(doc.document_id, doc.revision, rs, re_, row_sub)

    def _run(proposal, *, table_context=None, fe_list=()):
        layer = EvidenceLayer(vocabulary=vocab, policy=policy)
        layer.ingest(doc)
        item = {
            "proposal": proposal,
            "basis": "explicit",
            "evidence": [row_ref],
            "field_evidence": {fe.field: fe.evidence for fe in fe_list},
        }
        if table_context is not None:
            item["table_context"] = table_context
        summary = layer.ingest_units([item], attestor="table-reader", structural_reliability=0.95)
        assert summary["units_admitted"] == 1
        unit = layer.unit(summary["unit_ids"][0])
        ev = evaluate_unit_admissibility(
            unit,
            vocabulary=vocab,
            policy=policy,
            documents={doc.document_id: doc},
            stored_field_evidences=layer.field_evidences,
        )
        return layer, unit, ev

    good_prop = {
        "entity": "Acme Holdings",
        "predicate": "revenue",
        "value": 150000,
        "unit": "USD",
        "assertion_kind": "fact",
        "condition_status": "unconditional",
    }

    # 1. Negative: without table_context, silent x1000 / USD inference MUST fail
    _, _, ev_silent = _run(good_prop, table_context=None, fe_list=())
    assert not ev_silent.admissible
    assert ev_silent.issue_code == "VALUE_NOT_IN_EVIDENCE"

    # 2. Negative: table_context declared WITHOUT FieldEvidence('table_header') MUST fail
    _, _, ev_missing_hdr = _run(
        good_prop,
        table_context={"scale": 1000, "currency": "USD", "column": "FY 2024 ($ in thousands)", "period": "FY 2024"},
        fe_list=(),
    )
    assert not ev_missing_hdr.admissible
    assert ev_missing_hdr.issue_code == "MISSING_HEADER_CITATION"

    # 3. Negative: wrong x1000 scale vs header ('in millions' doc vs scale=1000)
    millions_doc_text = (
        "Acme Holdings Financial Summary\n"
        "| Metric | FY 2024 ($ in millions) |\n"
        "| --- | --- |\n"
        "| Revenue | 150 |\n"
    )
    m_doc = SourceDocument("fin_m.md", millions_doc_text, authority_rank=100, weight=0.95)
    m_hs, m_he = _span_of(millions_doc_text, "| Metric | FY 2024 ($ in millions) |")
    m_rs, m_re = _span_of(millions_doc_text, "| Revenue | 150 |")
    m_unit_prop = vocab.proposition(good_prop)
    m_unit = EvidenceUnit(
        m_unit_prop,
        (EvidenceRef(m_doc.document_id, m_doc.revision, m_rs, m_re, "| Revenue | 150 |"),),
        "explicit",
        "application",
        {"kind": "application", "attestor": "t"},
        0.95,
        proposal={**good_prop, "table_context": {"scale": 1000, "currency": "USD", "column": "FY 2024 ($ in millions)"}},
    )
    f_wrong_scale = verify_extraction_fidelity(
        millions_doc_text,
        m_unit,
        field_evidence=[FieldEvidence("table_header", EvidenceRef(m_doc.document_id, m_doc.revision, m_hs, m_he, "| Metric | FY 2024 ($ in millions) |"))],
        vocabulary=vocab,
        documents={m_doc.document_id: m_doc},
    )
    assert any(f.code == "TABULAR_SCALE_MISMATCH" for f in f_wrong_scale)

    # 4. Negative: wrong column ('120' sits in FY 2023, not declared FY 2024)
    wrong_col_prop = {**good_prop, "value": 120000}
    _, _, ev_wrong_col = _run(
        wrong_col_prop,
        table_context={"scale": 1000, "currency": "USD", "column": "FY 2024 ($ in thousands)", "period": "FY 2024"},
        fe_list=[FieldEvidence("table_header", hdr_ref)],
    )
    assert not ev_wrong_col.admissible
    assert ev_wrong_col.issue_code == "TABULAR_COLUMN_MISMATCH"

    # 5. Negative: wrong currency
    wrong_cur_prop = {**good_prop, "unit": "EUR"}
    _, _, ev_wrong_cur = _run(
        wrong_cur_prop,
        table_context={"scale": 1000, "currency": "EUR", "column": "FY 2024 ($ in thousands)"},
        fe_list=[FieldEvidence("table_header", hdr_ref)],
    )
    assert not ev_wrong_cur.admissible
    assert ev_wrong_cur.issue_code == "TABULAR_CURRENCY_MISMATCH"

    # 6. Negative: wrong period
    _, _, ev_wrong_period = _run(
        good_prop,
        table_context={"scale": 1000, "currency": "USD", "column": "FY 2024 ($ in thousands)", "period": "FY 2025"},
        fe_list=[FieldEvidence("table_header", hdr_ref)],
    )
    assert not ev_wrong_period.admissible
    assert ev_wrong_period.issue_code == "TABULAR_PERIOD_MISMATCH"

    # 7. Positive: explicit table_context + cited FieldEvidence('table_header') passes end-to-end
    layer_ok, _, ev_ok = _run(
        good_prop,
        table_context={"scale": 1000, "currency": "USD", "column": "FY 2024 ($ in thousands)", "period": "FY 2024"},
        fe_list=[FieldEvidence("table_header", hdr_ref)],
    )
    assert ev_ok.admissible, ev_ok
    ask_res = layer_ok.ask({"entity": "Acme Holdings", "predicate": "revenue", "operation": "list"})
    assert ask_res["results"][0]["status"] == "accepted"
    assert ask_res["results"][0]["value"] == {"value": "150000", "unit": "USD"}
    assert ask_res["results"][0]["unit"] == "USD"


# ---------------------------------------------------------------------------
# P2 #6: Comma and semicolon separated lists inside JSON / KV records
# ---------------------------------------------------------------------------

def test_p2_6_kv_json_comma_and_semicolon_lists():
    vocab = Vocabulary(
        "kv_lists",
        "1.0",
        (
            PredicateSpec("categories", FieldSpec("string"), aliases=("allowed categories",)),
            PredicateSpec("excluded_categories", FieldSpec("string")),
        ),
        entities=(("Acme Corp", ("Acme",)),),
    )
    json_text = json.dumps(
        {
            "entity": "Acme Corp",
            "categories": "A, B, C",
            "excluded_categories": "D; E",
        }
    )
    doc = SourceDocument("rec.json", json_text, authority_rank=100, weight=0.95)
    ref = EvidenceRef(doc.document_id, doc.revision, 0, len(json_text), json_text)

    for ok_item in ("A", "B", "C"):
        prop = vocab.proposition(
            {
                "entity": "Acme Corp",
                "predicate": "categories",
                "value": ok_item,
                "assertion_kind": "fact",
                "condition_status": "unconditional",
            }
        )
        u = EvidenceUnit(prop, (ref,), "explicit", "application", {"kind": "application", "attestor": "kv"}, 0.95)
        findings = verify_extraction_fidelity(json_text, u, vocabulary=vocab, documents={doc.document_id: doc})
        assert not findings, f"Unexpected findings for listed item {ok_item!r}: {findings}"

    # Semicolon list KV format
    kv_text = 'entity: "Acme Corp", categories: "Alpha; Beta; Gamma", excluded_categories: "Delta; Epsilon"'
    doc_kv = SourceDocument("rec.kv", kv_text, authority_rank=100, weight=0.95)
    ref_kv = EvidenceRef(doc_kv.document_id, doc_kv.revision, 0, len(kv_text), kv_text)
    prop_beta = vocab.proposition(
        {
            "entity": "Acme Corp",
            "predicate": "categories",
            "value": "Beta",
            "assertion_kind": "fact",
            "condition_status": "unconditional",
        }
    )
    u_beta = EvidenceUnit(prop_beta, (ref_kv,), "explicit", "application", {"kind": "application", "attestor": "kv"}, 0.95)
    assert not verify_extraction_fidelity(kv_text, u_beta, vocabulary=vocab, documents={doc_kv.document_id: doc_kv})

    # Negative: absent item 'F'
    prop_f = vocab.proposition(
        {
            "entity": "Acme Corp",
            "predicate": "categories",
            "value": "F",
            "assertion_kind": "fact",
            "condition_status": "unconditional",
        }
    )
    u_f = EvidenceUnit(prop_f, (ref,), "explicit", "application", {"kind": "application", "attestor": "kv"}, 0.95)
    assert verify_extraction_fidelity(json_text, u_f, vocabulary=vocab, documents={doc.document_id: doc})

    # Negative: item from DIFFERENT KV key ('D' belongs to excluded_categories, not categories)
    prop_d = vocab.proposition(
        {
            "entity": "Acme Corp",
            "predicate": "categories",
            "value": "D",
            "assertion_kind": "fact",
            "condition_status": "unconditional",
        }
    )
    u_d = EvidenceUnit(prop_d, (ref,), "explicit", "application", {"kind": "application", "attestor": "kv"}, 0.95)
    f_other_key = verify_extraction_fidelity(json_text, u_d, vocabulary=vocab, documents={doc.document_id: doc})
    assert any(f.code == "RELATIONAL_ASSOCIATION_MISMATCH" for f in f_other_key)


# ---------------------------------------------------------------------------
# P2 #7: Explicit condition_status context equivalence rule when omitted
# ---------------------------------------------------------------------------

def test_p2_7_omitted_condition_status_context_rule():
    vocab = Vocabulary(
        "cond_equiv",
        "1.0",
        (PredicateSpec("payment_term", FieldSpec("quantity", dimensions=("duration_calendar_days",))),),
        require_explicit_condition_status=False,
    )
    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    text = "Acme payment term is 30 days."
    doc = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
    layer.ingest(doc)
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_term",
                    "value": 30,
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
            }
        ],
        attestor="reader",
        structural_reliability=0.95,
    )

    # 1. Query omitting condition_status against explicit 'unconditional' unit MUST NOT auto-support
    omitted_chk = layer.check_claim(
        {
            "entity": "Acme",
            "predicate": "payment_term",
            "value": 30,
            "unit": "calendar_days",
            "assertion_kind": "requirement",
        },
        validity="ignore",
    )
    assert omitted_chk["status"] == "not_supported"
    assert omitted_chk["relation"] == "equal"  # comparable value without automatic 'supported'
    assert {"path": "/condition_status", "claim": "unknown_condition", "ledger": "unconditional"} in omitted_chk["differences"]

    # 2. Explicit matching 'unconditional' returns 'supported'
    exact_chk = layer.check_claim(
        {
            "entity": "Acme",
            "predicate": "payment_term",
            "value": 30,
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
        validity="ignore",
    )
    assert exact_chk["status"] == "supported"

    # 3. Reverse direction: ledger omitted condition_status vs explicit 'unconditional' query
    layer_om = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer_om.ingest(doc)
    layer_om.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "payment_term",
                    "value": 30,
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
            }
        ],
        attestor="reader",
        structural_reliability=0.95,
    )
    rev_chk = layer_om.check_claim(
        {
            "entity": "Acme",
            "predicate": "payment_term",
            "value": 30,
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
        validity="ignore",
    )
    assert rev_chk["status"] == "not_supported"
    assert {"path": "/condition_status", "claim": "unconditional", "ledger": "unknown_condition"} in rev_chk["differences"]


# ---------------------------------------------------------------------------
# P2 #8: Typed versioned normalizers & provenance without quote mutation
# ---------------------------------------------------------------------------

def test_p2_8_typed_versioned_normalizers_symmetric_and_provenance():
    assert NORMALIZER_VERSION == "v4-norm-1"

    # Time intervals: '16:30-21:0' / '4:30 PM–9 PM' <-> '16:30-21:00'
    t_ocr, trace1 = normalize_value_with_provenance("time_interval", "16:30-21:0")
    t_pm, trace2 = normalize_value_with_provenance("time_interval", "4:30 PM–9 PM")
    assert t_ocr == "16:30-21:00"
    assert t_pm == "16:30-21:00"
    assert trace1.version == NORMALIZER_VERSION
    assert trace1.raw == "16:30-21:0"
    assert trace2.raw == "4:30 PM–9 PM"

    # Dates, separator numbers, percentages
    assert normalize_date_token("31 de enero de 2027") == "2027-01-31"
    assert normalize_date_token("31/01/2027") == "2027-01-31"
    assert normalize_date_token("January 31, 2027") == "2027-01-31"
    assert normalize_separator_number("1,250.50") == "1250.5"
    assert normalize_separator_number("1.250,50") == "1250.5"
    assert normalize_percentage("5,80 %") == "5.8%"
    assert normalize_percentage("5.80%") == "5.8%"

    # End-to-end EvidenceLayer + verify_extraction_fidelity without mutating quoted text
    vocab = Vocabulary(
        "norm_domain",
        "1.0",
        (
            PredicateSpec("service_window", FieldSpec("string")),
            PredicateSpec("effective_date", FieldSpec("date")),
            PredicateSpec("coupon_rate", FieldSpec("quantity", dimensions=("ratio",))),
        ),
        entities=(("Acme Corp", ("Acme",)),),
    )
    doc_text = (
        "Acme Corp service window is 4:30 PM–9 PM daily. "
        "Acme Corp effective date is 31 de enero de 2027. "
        "Acme Corp coupon rate is 5,80 %."
    )
    doc = SourceDocument("sched.txt", doc_text, authority_rank=100, weight=0.95)
    s_win, e_win = _span_of(doc_text, "Acme Corp service window is 4:30 PM–9 PM daily.")
    s_dat, e_dat = _span_of(doc_text, "Acme Corp effective date is 31 de enero de 2027.")
    s_cpn, e_cpn = _span_of(doc_text, "Acme Corp coupon rate is 5,80 %.")

    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer.ingest(doc)
    assert layer.configuration["normalizer_version"] == NORMALIZER_VERSION

    res = layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme Corp",
                    "predicate": "service_window",
                    "value": "16:30-21:00",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, s_win, e_win, doc_text[s_win:e_win])],
            },
            {
                "proposal": {
                    "entity": "Acme Corp",
                    "predicate": "effective_date",
                    "value": "2027-01-31",
                    "assertion_kind": "fact",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, s_dat, e_dat, doc_text[s_dat:e_dat])],
            },
            {
                "proposal": {
                    "entity": "Acme Corp",
                    "predicate": "coupon_rate",
                    "value": "5,80",
                    "unit": "%",
                    "assertion_kind": "fact",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, s_cpn, e_cpn, doc_text[s_cpn:e_cpn])],
            },
        ],
        attestor="reader",
        structural_reliability=0.95,
    )
    assert res["units_admitted"] == 3
    for uid in res["unit_ids"]:
        u = layer.unit(uid)
        ev = evaluate_unit_admissibility(u, vocabulary=vocab, policy=AcceptancePolicy(), documents={doc.document_id: doc})
        assert ev.admissible, ev
        assert ev.normalizer_version == NORMALIZER_VERSION
        # Quote MUST remain verbatim unmutated source substring
        assert doc_text[u.evidence[0].start : u.evidence[0].end] == u.evidence[0].quote

    # Semantic equivalence on check_claim for '16:30-21:0' against stored '4:30 PM–9 PM' <-> '16:30-21:00'
    chk_match = layer.check_claim(
        {
            "entity": "Acme Corp",
            "predicate": "service_window",
            "value": normalize_time_interval("16:30-21:0"),
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
        validity="ignore",
    )
    assert chk_match["status"] == "supported"
    assert chk_match["normalizer_version"] == NORMALIZER_VERSION

    # Bad window rejected cleanly
    bad_win = EvidenceUnit(
        vocab.proposition(
            {
                "entity": "Acme Corp",
                "predicate": "service_window",
                "value": "16:30-20:00",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            }
        ),
        (EvidenceRef(doc.document_id, doc.revision, s_win, e_win, doc_text[s_win:e_win]),),
        "explicit",
        "application",
        {"kind": "application", "attestor": "r"},
        0.95,
    )
    assert any(
        f.code == "VALUE_NOT_IN_EVIDENCE"
        for f in verify_extraction_fidelity(doc_text, bad_win, vocabulary=vocab, documents={doc.document_id: doc})
    )


# ---------------------------------------------------------------------------
# P2 #9: PDF/OCR noise normalization benchmark (>= 50 clean noisy pairs)
# ---------------------------------------------------------------------------

def _legacy_naive_diff_has_gap(clean_text: str, noisy_text: str) -> bool:
    """Measure what pre-P2#9 diff (`_norm_sentence_eq` + `_DIFF_WORD_RE`) reported without OCR normalization."""
    clean_words = re.findall(r"[A-Za-z0-9]+", clean_text.lower())
    noisy_words = re.findall(r"[A-Za-z0-9]+", noisy_text.lower())
    return clean_words != noisy_words


def test_p2_9_pdf_ocr_noise_benchmark_50_plus_clean_pairs_and_planted_diffs():
    templates = [
        ("The final financing obligation shall reflect full cashflow schedules at 5.80%.",
         "The \ufb01nal \ufb01nancing obliga-\ntion shall re\ufb02ect full cash\ufb02ow schedules at 5.80%."),
        ('The "Senior Facility" allows a cure period of 30 calendar days under Section 4.',
         "\u201cSenior Facility\u201d\u2014which Wait no: \u201cSenior Facility\u201d" if False else 'The \u201cSenior Facility\u201d allows a cure pe-\nriod of 30 calendar days under Section 4.'),
        ("Effective interest distributions require formal confirmation prior to settlement.",
         "E\ufb00ective interest distributions require formal con\ufb01r-\nmation prior\u00ad to settlement."),
        ("The Issuer shall deliver official officer certificates on a quarterly basis.",
         "The Issuer shall deliver o\ufb03cial o\ufb03cer certi\ufb01-\ncates on a quarterly basis."),
        ("Subordinated tranches remain strictly first-loss until principal shortfall is cured.",
         "Subordinated tranches remain strictly \ufb01rst\u2013loss until prin-\ncipal shortfall\n is cured."),
    ]
    clean_pairs: list[tuple[str, str]] = []
    for idx in range(55):
        base_clean, base_noisy = templates[idx % len(templates)]
        tag = f"Clause {idx + 1}: "
        # Inject extra intra-paragraph soft wrap on noisy counterpart
        clean_sent = tag + base_clean
        noisy_sent = tag + base_noisy.replace(" days ", " days\n").replace(" prior", "\nprior")
        clean_pairs.append((clean_sent, noisy_sent))

    assert len(clean_pairs) >= 50

    vocab = Vocabulary(
        "ocr_bench",
        "1.0",
        (
            PredicateSpec("coupon_rate", FieldSpec("quantity", dimensions=("ratio",))),
            PredicateSpec("governing_law", FieldSpec("string")),
        ),
        entities=(("Issuer", ()),),
    )
    profile = DomainProfile(profile_id="ocr_bench", version="1.0", vocabulary=vocab)

    def _run_ocr_register(c_body: str, n_body: str, prefix: str):
        anchor_sent = "The Issuer obligation is governed by Delaware law."
        c_full = f"{c_body}\n\nSection 2. Governing Law: {anchor_sent}"
        n_full = f"{n_body}\n\nSection 2. Governing Law: {anchor_sent}"
        d_clean = SourceDocument(f"{prefix}_clean.txt", c_full, authority_rank=100, weight=0.95)
        d_noisy = SourceDocument(f"{prefix}_noisy.pdf", n_full, authority_rank=50, weight=0.90)
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        layer.ingest([d_clean, d_noisy])
        as_, ae_ = _span_of(c_full, anchor_sent)
        layer.ingest_units(
            [
                {
                    "proposal": {
                        "entity": "Issuer",
                        "predicate": "governing_law",
                        "value": "Delaware",
                        "assertion_kind": "fact",
                    },
                    "basis": "explicit",
                    "evidence": [EvidenceRef(d_clean.document_id, d_clean.revision, as_, ae_, anchor_sent)],
                }
            ],
            attestor="ocr_bench_seed",
        )
        div_col = collect_reference_divergences([d_clean, d_noisy], domain_profile=profile)
        return build_corpus_reference_register(
            layer,
            vocab,
            divergence_collection=div_col,
            domain_profile_fingerprint=profile.fingerprint,
        )

    naive_false_gaps = 0
    post_false_gaps = 0

    for idx, (c_txt, n_txt) in enumerate(clean_pairs):
        if _legacy_naive_diff_has_gap(c_txt, n_txt):
            naive_false_gaps += 1
        reg = _run_ocr_register(c_txt, n_txt, f"clean_{idx}")
        assert reg.available is True
        if reg.gap_divergences:
            post_false_gaps += 1

    # Baseline (naive tokenization) flagged all clean pairs; post-fix eliminates 100% false OCR gaps
    assert naive_false_gaps == len(clean_pairs)
    assert post_false_gaps == 0
    ci_before = _wilson_ci(naive_false_gaps, len(clean_pairs))
    ci_after = _wilson_ci(post_false_gaps, len(clean_pairs))
    assert ci_after[1] < ci_before[0]

    # Planted real differences inside noisy OCR pairs MUST still be detected (0% miss rate)
    planted_misses = 0
    for idx in range(10):
        c_txt = f"Clause {idx + 1}: The final financing obligation bears interest at 5.80% per annum."
        n_txt = f"Clause {idx + 1}: The \ufb01nal \ufb01nancing obliga-\ntion bears interest at 5.55% per annum."
        reg = _run_ocr_register(c_txt, n_txt, f"planted_{idx}")
        assert reg.available is True
        if not reg.gap_divergences:
            planted_misses += 1
    assert planted_misses == 0
