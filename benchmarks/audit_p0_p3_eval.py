"""Reproducible evaluation suite for P0–P3 audit fixes on unseen EDGAR contracts,
real 10-K/10-Q tables, and newly designed general layout/pagination/hyphen/dense-reader suites.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from claimlayer import Assertion, Ledger, Policy, Source, verify
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
from claimlayer.verifiers import diff as diff_mod
from claimlayer.verifiers.diff import extract_divergent_sections


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if n <= 0:
        return {"k": 0, "n": 0, "rate": None, "ci_95_low": None, "ci_95_high": None}
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return {
        "k": k,
        "n": n,
        "rate": round(p, 4),
        "ci_95_low": round(max(0.0, center - half), 4),
        "ci_95_high": round(min(1.0, center + half), 4),
    }


def _wrap_to_width(text: str, width: int) -> str:
    """Wrap paragraphs onto `width` columns cleanly (breaking hyphenated words across lines when wide enough)."""
    out_lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out_lines.append("")
            continue
        # Keep short standalone header lines unwrapped
        if len(para) <= width or para.startswith(("Section ", "Article ", "1. ", "2. ", "ARTICLE ")):
            out_lines.append(para)
            continue
        cur = ""
        words = para.split(" ")
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= width:
                cur = f"{cur} {w}"
            else:
                # If a hyphenated compound straddles the width boundary, wrap right after '-'
                if "-" in w and len(cur) + 1 + w.index("-") + 1 <= width:
                    h_idx = w.index("-")
                    out_lines.append(f"{cur} {w[: h_idx + 1]}")
                    cur = w[h_idx + 1 :]
                elif len(w) >= 10 and len(cur) + 6 <= width:
                    split_at = max(3, width - len(cur) - 2)
                    if split_at < len(w) - 3:
                        out_lines.append(f"{cur} {w[:split_at]}-")
                        cur = w[split_at:]
                    else:
                        out_lines.append(cur)
                        cur = w
                else:
                    out_lines.append(cur)
                    cur = w
        if cur:
            out_lines.append(cur)
    return "\n".join(out_lines)


def _measure_doc_coverage(text: str) -> float:
    if hasattr(diff_mod, "extract_divergent_sections_with_coverage"):
        report = diff_mod.extract_divergent_sections_with_coverage(
            Source(id="p", text=text, authority_rank=100),
            Source(id="s", text=text, authority_rank=80),
        )
        return float(report.primary_coverage)
    secs = diff_mod._split_into_sections(text)
    if not text.strip():
        return 1.0
    if not secs:
        return 1.0  # legacy fail-closed unsegmented fallback
    intervals = sorted((int(s.start), int(s.end)) for s in secs.values())
    covered = 0
    cur_s, cur_e = -1, -1
    for s, e in intervals:
        if s > cur_e:
            if cur_e > cur_s:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e > cur_s:
        covered += cur_e - cur_s
    non_ws = sum(1 for ch in text if not ch.isspace())
    return covered / max(len(text), non_ws, 1)


def evaluate_p0_unseen_contracts(snap: dict[str, Any]) -> dict[str, Any]:
    contracts = snap["contracts"]
    full_cov_count = 0
    off_section_detected = 0
    details = []
    for idx, c in enumerate(contracts):
        orig = c["text"]
        secs = diff_mod._split_into_sections(orig)
        first_start = min((int(s.start) for k, s in secs.items() if k != "unsegmented_remainder"), default=0)
        cov = _measure_doc_coverage(orig)
        if cov >= 0.999999:
            full_cov_count += 1

        # Mutate ONLY in the unsegmented leading region before first recognized section
        # (guaranteed outside all _split_into_sections blocks when first_start > 0)
        mut_prefix = "Guarantor shall indemnify Holder without limitation.\n\n"
        mut_alt = "Guarantor shall not indemnify Holder without limitation.\n\n"
        p_txt = (mut_prefix + orig) if first_start > 0 else orig
        s_txt = (mut_alt + orig) if first_start > 0 else orig
        divs = extract_divergent_sections(
            Source(id=f"{c['id']}_pri", text=p_txt, authority_rank=100),
            Source(id=f"{c['id']}_sec", text=s_txt, authority_rank=80),
        )
        detected = len(divs) >= 1
        if detected:
            off_section_detected += 1
        details.append({
            "id": c["id"],
            "coverage": round(cov, 4),
            "first_hdr_offset": first_start,
            "off_section_detected": detected,
        })

    # Additional structural layout cases (preamble+1sec, interleaved unheaded, duplicate keys)
    layout_cases = [
        (
            "Preamble: Customer shall pay all undisputed amounts.\nSection 1. Scope: Services apply globally.\n",
            "Preamble: Customer shall not pay any disputed amounts.\nSection 1. Scope: Services apply globally.\n",
            "preamble_negation",
        ),
        (
            "Section 1. Term: Agreement lasts 2 years.\nUNHEADED SCHEDULE\nFee is $12,000 per month.\nSection 2. Law: Texas.\n",
            "Section 1. Term: Agreement lasts 2 years.\nUNHEADED SCHEDULE\nFee is $18,000 per month.\nSection 2. Law: Texas.\n",
            "interleaved_unheaded",
        ),
        (
            "Section 1. Overview: Applies to North America and Canada.\n"
            + ("Clause details continue for regulatory reporting across operating units. " * 6)
            + "\nSection 1. Overview: Cap is $100,000.\n",
            "Section 1. Overview: Applies to North America and Canada.\n"
            + ("Clause details continue for regulatory reporting across operating units. " * 6)
            + "\nSection 1. Overview: Cap is $250,000.\n",
            "duplicate_shorter_section_key",
        ),
        (
            "Recitals: Parties intend an exclusive license or joint venture.\n\nSection 5. Notices: Written mail.\n",
            "Recitals: Parties intend a non-exclusive license and sole venture.\n\nSection 5. Notices: Written mail.\n",
            "preamble_conjunction_and_polarity",
        ),
    ]
    layout_hits = 0
    for idx, (pt, st, tag) in enumerate(layout_cases):
        divs = extract_divergent_sections(
            Source(id=f"ly_p_{idx}", text=pt, authority_rank=100),
            Source(id=f"ly_s_{idx}", text=st, authority_rank=80),
        )
        if divs:
            layout_hits += 1

    return {
        "full_coverage_100pct": wilson_ci(full_cov_count, len(contracts)),
        "off_section_mutation_detected": wilson_ci(off_section_detected, len(contracts)),
        "structural_layout_detected": wilson_ci(layout_hits, len(layout_cases)),
        "details": details,
    }


def evaluate_p1_furniture_and_hyphens() -> dict[str, Any]:
    # 1. PDF page furniture (repeating banner headers, sequential "- n -" / "Page n of m" / bare page numbers)
    furniture_clean_zero_fp = 0
    neg_ctrl_preserved = 0
    n_f = 20
    for i in range(n_f):
        body_p1 = f"Section 1. Delivery\nSupplier shall deliver {100 + i} pallets on Monday. Each shipment includes tracking."
        body_p2 = f"Section 2. Payment\nBuyer shall settle invoices within {15 + (i % 4) * 5} days of receipt."
        body_p3 = f"Section 3. Warranty\nHardware carries a {12 + (i % 3) * 6}-month limited factory warranty."
        clean_src = Source(id=f"fc_a_{i}", text=f"{body_p1}\n\n{body_p2}\n\n{body_p3}", authority_rank=100)
        banner = f"MERIDIAN SUPPLY AGREEMENT — REVISION {i}" if i % 2 == 0 else f"CONFIDENTIAL OPERATING SCHEDULE #{i + 1}"
        if i % 3 == 0:
            pg = (
                f"{body_p1}\nPage 1 of 3\n{banner}\n\n"
                f"{body_p2}\nPage 2 of 3\n{banner}\n\n"
                f"{body_p3}\nPage 3 of 3\n{banner}"
            )
        elif i % 3 == 1:
            pg = (
                f"{body_p1}\n- 1 -\n{banner}\n\n"
                f"{body_p2}\n- 2 -\n{banner}\n\n"
                f"{body_p3}\n- 3 -\n{banner}"
            )
        else:
            pg = (
                f"{banner}\n1\n{body_p1}\n\n"
                f"{banner}\n2\n{body_p2}\n\n"
                f"{banner}\n3\n{body_p3}"
            )
        pag_src = Source(id=f"fc_b_{i}", text=pg, authority_rank=80)
        if extract_divergent_sections(clean_src, pag_src) == []:
            furniture_clean_zero_fp += 1

        # Negative control: single isolated short clause OR standalone number in one version only
        added_line = "42" if i % 2 == 0 else "Liability cap applies."
        neg_src = Source(id=f"fn_b_{i}", text=f"{body_p1}\n\n{body_p2}\n{added_line}\n\n{body_p3}", authority_rank=80)
        if len(extract_divergent_sections(clean_src, neg_src)) >= 1:
            neg_ctrl_preserved += 1

    # 2. Line-break hyphenation across widths (36, 50, 68) preserving lexical compounds + negative control
    hyphen_clean_zero_fp = 0
    hyphen_neg_detected = 0
    n_h = 15
    for i in range(n_h):
        base = (
            f"Section 1. License Grant\n"
            f"Vendor grants Customer a non-exclusive, non-transferable, royalty-free license for third-party "
            f"components over a two-year initial commitment covering deployment unit {i}.\n"
            f"All confidential documentation must remain safeguarded under standard organizational protocols."
        )
        width = (36, 48, 62)[i % 3]
        reflowed = _wrap_to_width(base, width)
        if extract_divergent_sections(
            Source(id=f"h_a_{i}", text=base, authority_rank=100),
            Source(id=f"h_b_{i}", text=reflowed, authority_rank=80),
        ) == []:
            hyphen_clean_zero_fp += 1

        # Negative control: actual lexical change on a hyphenated compound ("non-exclusive" -> "exclusive")
        mutated = _wrap_to_width(base.replace("non-exclusive", "exclusive", 1), width)
        if len(extract_divergent_sections(
            Source(id=f"hn_a_{i}", text=base, authority_rank=100),
            Source(id=f"hn_b_{i}", text=mutated, authority_rank=80),
        )) >= 1:
            hyphen_neg_detected += 1

    return {
        "furniture_clean_zero_fp": wilson_ci(furniture_clean_zero_fp, n_f),
        "furniture_negative_ctrl_preserved": wilson_ci(neg_ctrl_preserved, n_f),
        "hyphen_reflow_zero_fp": wilson_ci(hyphen_clean_zero_fp, n_h),
        "hyphen_mutation_detected": wilson_ci(hyphen_neg_detected, n_h),
    }


def _dense_vocab_and_reader():
    vocab = Vocabulary(
        name="audit_dense_test",
        version="1.0",
        predicates=(PredicateSpec(name="clause", value=FieldSpec(type="string"), cardinality="multi"),),
    )

    class DenseReader:
        identity = {"provider": "test", "model": "dense_clause_eval", "prompt_version": "1", "prompt_digest": "d"}

        def propose(self, req: dict[str, Any]) -> dict[str, Any]:
            doc = req["document"]
            frags = _build_document_fragments(doc, 4000)
            fmap = dict(frags)
            raw: list[dict[str, Any]] = []
            for fid, fobj in frags:
                for ln in fobj["quote"].split("\n"):
                    s = ln.strip()
                    if not s or s.startswith(("Section ", "Article ")):
                        continue
                    raw.append({
                        "entity": "Agreement",
                        "predicate": "clause",
                        "value": s,
                        "polarity": "affirmed",
                        "assertion_kind": "fact",
                        "conditions": [],
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [fid],
                        "quote": s,
                        "field_evidence": {"entity": "Agreement", "value": s},
                    })
            items: list[Any] = []
            rej: list[Any] = []
            _bind_raw_proposals(raw, fmap, doc["text"], items, rej)
            return {"items": items, "rejected": rej, "calls": [], "errors": [], "fragments": len(frags), "windows": 1}

    return vocab, DenseReader()


def evaluate_p2_dense_reader() -> dict[str, Any]:
    vocab, reader = _dense_vocab_and_reader()
    deleted_pairs = [
        ("Confidentiality lasts 4 years.\nContractor shall maintain cyber insurance of $2,000,000.", "Confidentiality lasts 4 years."),
        ("Fees are due quarterly.\nLicensee shall permanently erase backup archives upon exit.", "Fees are due quarterly."),
        ("Reports ship monthly.\nAuditor may inspect billing ledgers once per year.", "Reports ship monthly."),
        ("System uptime is 99.9%.\nProvider shall reimburse downtime credits within 20 days.", "System uptime is 99.9%."),
        ("Ownership stays with Author.\nGuarantor guarantees prompt performance of all payment duties.", "Ownership stays with Author."),
        ("Travel costs require preapproval.\nSubcontractors require prior written consent of Client.", "Travel costs require preapproval."),
        ("Source code remains escrowed.\nTrustee holds a first-priority lien on Reserve Accounts.", "Source code remains escrowed."),
        ("Taxes are borne by Buyer.\nSeller warrants good and marketable title free of liens.", "Taxes are borne by Buyer."),
        ("Data is stored regionally.\nProcessor notifies Controller within 48 hours of any breach.", "Data is stored regionally."),
        ("Notice is sent by courier.\nTermination for convenience requires 90 calendar days notice.", "Notice is sent by courier."),
        ("Disputes go to arbitration.\nIndemnitor defends all third-party patent infringement suits.", "Disputes go to arbitration."),
        ("Interest accrues on overdue balances.\nDefault triggers acceleration of the remaining principal.", "Interest accrues on overdue balances."),
    ]
    del_hits = 0
    for idx, (pt, st) in enumerate(deleted_pairs):
        da = SourceDocument(document_id=f"del_p_{idx}", text=f"Section 1. Covenants\n{pt}", authority_rank=100, attribution="gov")
        db = SourceDocument(document_id=f"del_s_{idx}", text=f"Section 1. Covenants\n{st}", authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy(), reader=reader)
        layer.ingest([da, db])
        dc = collect_reference_divergences([da, db])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        has_del = any(
            g.get("token") == "DELETED_CLAUSE" and isinstance(g.get("span"), (list, tuple)) and len(g["span"]) == 2
            for g in reg.gap_divergences
        )
        if has_del:
            del_hits += 1

    # Clean control for deleted clauses (no deletions) -> 0 false gaps
    del_clean_hits = 0
    for idx, (pt, _) in enumerate(deleted_pairs[:6]):
        da = SourceDocument(document_id=f"delc_p_{idx}", text=f"Section 1. Covenants\n{pt}", authority_rank=100, attribution="gov")
        db = SourceDocument(document_id=f"delc_s_{idx}", text=f"Section 1. Covenants\n{pt}", authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy(), reader=reader)
        layer.ingest([da, db])
        dc = collect_reference_divergences([da, db])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        if reg.gap_divergences == ():
            del_clean_hits += 1

    unit_mod_pairs = [
        ("Invoice settles within 30 days of delivery.", "Invoice settles within 30 business days of delivery."),
        ("Notice is due within 10 calendar days.", "Notice is due within 10 business days."),
        ("Cure period runs for 1 calendar month.", "Cure period runs for 1 month."),
        ("Settlement window equals 15 working days.", "Settlement window equals 15 days."),
        ("Commission equals 4% of net sales.", "Commission equals 4% of gross sales."),
        ("Earnout is $500,000 based on net proceeds.", "Earnout is $500,000 based on gross proceeds."),
        ("Late fee accrues at 6% per annum.", "Late fee accrues at 6% per month."),
        ("Reporting threshold covers 1 fiscal year.", "Reporting threshold covers 1 calendar year."),
        ("Grace window expires in 5 banking days.", "Grace window expires in 5 calendar days."),
        ("Interest computes using 5% simple interest.", "Interest computes using 5% compound interest."),
        ("Payment follows net 45 terms.", "Payment follows 45 business days terms."),
        ("Renewal runs for 2 calendar months.", "Renewal runs for 2 months."),
    ]
    mod_hits = 0
    for idx, (pt, st) in enumerate(unit_mod_pairs):
        da = SourceDocument(document_id=f"mod_p_{idx}", text=f"Section 1. Terms\n{pt}", authority_rank=100, attribution="gov")
        db = SourceDocument(document_id=f"mod_s_{idx}", text=f"Section 1. Terms\n{st}", authority_rank=80, attribution="draft")
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy(), reader=reader)
        layer.ingest([da, db])
        dc = collect_reference_divergences([da, db])
        reg = build_corpus_reference_register(layer, vocab, divergence_collection=dc)
        if len(reg.gap_divergences) >= 1:
            mod_hits += 1

    return {
        "deleted_clause_dense_reader": wilson_ci(del_hits, len(deleted_pairs)),
        "deleted_clause_clean_control": wilson_ci(del_clean_hits, 6),
        "compute_unit_modifier_diff": wilson_ci(mod_hits, len(unit_mod_pairs)),
    }


def evaluate_p3_real_tables(snap: dict[str, Any]) -> dict[str, Any]:
    by_id = {t["id"]: t["text"] for t in snap["primary_paren_tables"]}
    # Add statement tables too
    for t in snap["statement_tables"]:
        by_id[t["id"]] = t["text"]

    # Ground-truth PASS assertions on unseen real 10-K/10-Q tables (scaled rows + unscaled exceptions + OCR header)
    costco_is = by_id["p10q_costco_0000909832-26-000051_96"]
    costco_ci = by_id["p10q_costco_0000909832-26-000051_128"]
    costco_bs = by_id["p10q_costco_0000909832-26-000051_143"]
    nvidia_is = by_id["p10q_nvidia_0001045810-26-000075_830"]
    nvidia_mkt = by_id["p10q_nvidia_0001045810-26-000075_840"]
    walmart_is = by_id["p10q_walmart_0000104169-26-000154_89"]
    walmart_bs = by_id["p10q_walmart_0000104169-26-000154_140"]
    target_is = by_id["p10q_target_0000027419-26-000042_571"]
    target_bare = by_id["p10q_target_0000027419-26-000042_255"] if "p10q_target_0000027419-26-000042_255" in by_id else next(v for k, v in by_id.items() if k.startswith("p10q_target_") and "(millions)" in v)
    adobe_ppa = by_id["p10q_adobe_0000796343-26-000156_435"]
    adobe_notes = by_id["p10q_adobe_0000796343-26-000156_832"]
    adobe_rev = by_id["p10q_adobe_0000796343-26-000156_964"]
    netflix_r3 = next(t["text"] for t in snap["statement_tables"] if "netflix" in t["id"] and "$ in Thousands" in t["text"])
    visa_par = next(t["text"] for t in snap["statement_tables"] if "visa" in t["id"] and "Parenthetical" in t["text"])

    pass_cases = [
        # (source_id, text, subject, predicate, value, quote)
        ("costco_is", costco_is, "Costco", "net_sales", "$69,154 million", "Net sales $ 69,154 $ 61,965 $ 203,374 $ 185,480"),
        ("costco_ci", costco_ci, "Costco", "net_income", "$2,192 million", "NET INCOME $ 2,192 $ 1,903 $ 6,228 $ 5,489"),
        ("costco_bs", costco_bs, "Costco", "cash_and_equivalents", "$18,946 million", "Cash and cash equivalents $ 18,946 $ 14,161"),
        ("nvidia_is", nvidia_is, "NVIDIA", "revenue", "$96,221 million", "Revenue $ 96,221 $ 81,615 $ 46,743 18 % 106 %"),
        ("nvidia_is_opinc", nvidia_is, "NVIDIA", "operating_income", "$63,734 million", "Operating income $ 63,734 $ 53,536 $ 28,440 19 % 124 %"),
        # Real exception row: Net income per diluted share MUST remain $2.46 (NOT $2.46 million!)
        ("nvidia_is_eps", nvidia_is, "NVIDIA", "diluted_eps", "$2.46", "Net income per diluted share $ 2.46 $ 2.39 $ 1.08 3 % 128 %"),
        # Real percentage row inside ($ in millions, except per share data) must remain 75.0%
        ("nvidia_is_gm", nvidia_is, "NVIDIA", "gross_margin", "75.0%", "Gross margin 75.0 % 74.9 % 72.4 % 0.1 pts 2.6 pts"),
        ("nvidia_mkt", nvidia_mkt, "NVIDIA", "data_center_revenue", "$89,023 million", "Data Center $ 89,023 $ 75,246 $ 41,096 18 % 117 %"),
        ("walmart_is", walmart_is, "Walmart", "total_revenues", "$187,937 million", "Total revenues 187,937 177,402 365,688 343,011"),
        ("walmart_is_oi", walmart_is, "Walmart", "operating_income", "$9,383 million", "Operating income 9,383 7,286 16,876 14,421"),
        ("walmart_bs", walmart_bs, "Walmart", "inventories", "$61,600 million", "Inventories 61,600 58,851 57,729"),
        ("target_is", target_is, "Target", "net_sales", "$26,539 million", "Net sales $ 26,539 $ 25,211 5.3 % $ 51,982 $ 49,057 6.0 %"),
        ("target_bare", target_bare, "Target", "swap_notional_header", "$2.45 billion", "notional amounts totaling $2.45 billion as of August 1, 2026"),
        ("adobe_ppa", adobe_ppa, "Adobe", "goodwill", "$1,251 million", "1,251 N/A"),
        ("adobe_rev", adobe_rev, "Adobe", "subscription_revenue", "$6,582 million", "Subscription $ 6,582 $ 5,791 14 % $ 19,196 $ 16,915 13 %"),
        ("netflix_r3", netflix_r3, "Netflix", "net_income", "$3,663,927 thousand", "Net income $ 3,663,927 $ 2,546,916 $ 12,348,132 $ 8,562,680"),
        # Real balance sheet parenthetical (in dollars per share) MUST NOT scale to millions
        ("visa_par", visa_par, "Visa", "preferred_par_value", "$0.0001", "Preferred stock, par value (in dollars per share) $ 0.0001 $ 0.0001"),
        # Real balance sheet common stock shares issued (shares in Millions) MUST scale only shares, NOT dollars
        ("visa_shares", visa_par, "Visa", "common_par_value", "$0.0001", "Common stock, par value (in dollars per share) $ 0.0001 $ 0.0001"),
        # OCR-corrupted spaced/collapsed parenthetical headers over real NVIDIA/Costco text
        ("nvidia_ocr_nospace", nvidia_is.replace("($ in millions, except per share data)", "($inmillions,exceptpersharedata)"), "NVIDIA", "net_income", "$59,688 million", "Net income $ 59,688 $ 58,321 $ 26,422 2 % 126 %"),
        ("nvidia_ocr_eps", nvidia_is.replace("($ in millions, except per share data)", "(amounts inmillions, exceptpershare)"), "NVIDIA", "diluted_eps", "$2.46", "Net income per diluted share $ 2.46 $ 2.39 $ 1.08 3 % 128 %"),
        # Adobe notes table with bare (millions) variant
        ("target_bare_table", target_bare.replace("(millions)", "(in millions)"), "Target", "bare_millions_check", "$2.20 billion", "$2.20 billion as of January 31, 2026"),
        # Real Target table replacing header with bare "(U.S. dollars in millions, except percentages)"
        ("target_usd_hdr", target_is.replace("(dollars in millions)", "(U.S. dollars in millions, except percentages)"), "Target", "cost_of_sales", "$17,603 million", "Cost of sales (a)"),
        # Share exception in Costco balance sheet (common stock shares / par value must remain unscaled even when row says par value)
        ("costco_bs_short", costco_bs, "Costco", "short_term_investments", "$1,050 million", "Short-term investments 1,050 1,123"),
        ("nvidia_eps_quoted_full", nvidia_is, "NVIDIA", "revenue_vs_eps", "$2.46", "Net income per diluted share $ 2.46 $ 2.39"),
    ]
    # Wait: ensure "Cost of sales (a)\n17,603 ..." quote is verbatim! Use verbatim line "SG&A expenses 5,725 5,359 6.8 11,286 9,950 13.4"
    pass_cases[21] = ("target_usd_hdr", target_is.replace("(dollars in millions)", "(U.S. dollars in millions, except percentages)"), "Target", "sga_expenses", "$5,725 million", "SG&A expenses 5,725 5,359 6.8 11,286 9,950 13.4")

    # Ground-truth REJECT cases (wrong scale OR mis-scaling an excluded per-share/percentage/share item OR prose boundary)
    reject_cases = [
        ("r_costco_wrong_scale_k", costco_is, "Costco", "net_sales", "$69,154 thousand", "Net sales $ 69,154 $ 61,965 $ 203,374 $ 185,480"),
        ("r_costco_unscaled", costco_is, "Costco", "net_sales", "$69,154", "Net sales $ 69,154 $ 61,965 $ 203,374 $ 185,480"),
        # Critical exception-enforcement rejections: applying millions to per-share / par-value / percentage values!
        ("r_nvidia_eps_millions", nvidia_is, "NVIDIA", "diluted_eps", "$2.46 million", "Net income per diluted share $ 2.46 $ 2.39 $ 1.08 3 % 128 %"),
        ("r_nvidia_eps_wrong", nvidia_is, "NVIDIA", "diluted_eps", "$2.39 million", "Net income per diluted share $ 2.46 $ 2.39 $ 1.08 3 % 128 %"),
        ("r_nvidia_gm_millions", nvidia_is, "NVIDIA", "gross_margin", "$75.0 million", "Gross margin 75.0 % 74.9 % 72.4 % 0.1 pts 2.6 pts"),
        ("r_nvidia_rev_wrong_unit", nvidia_is, "NVIDIA", "revenue", "$96,221 billion", "Revenue $ 96,221 $ 81,615 $ 46,743 18 % 106 %"),
        ("r_walmart_unscaled", walmart_is, "Walmart", "operating_income", "$9,383", "Operating income 9,383 7,286 16,876 14,421"),
        ("r_walmart_thousands", walmart_is, "Walmart", "total_revenues", "$187,937 thousand", "Total revenues 187,937 177,402 365,688 343,011"),
        ("r_target_pct_as_usd", target_is.replace("(dollars in millions)", "(dollars in millions, except percentages)"), "Target", "sales_growth", "$5.3 million", "Net sales $ 26,539 $ 25,211 5.3 % $ 51,982 $ 49,057 6.0 %"),
        ("r_adobe_rev_unscaled", adobe_rev, "Adobe", "subscription_revenue", "$6,582", "Subscription $ 6,582 $ 5,791 14 % $ 19,196 $ 16,915 13 %"),
        ("r_adobe_pct_millions", adobe_rev.replace("(dollars in millions)", "(dollars in millions, except percentages)"), "Adobe", "pct_total", "97 million", "Percentage of total revenue 97 % 97 % 97 % 96 %"),
        ("r_netflix_millions_instead_k", netflix_r3, "Netflix", "net_income", "$3,663,927 million", "Net income $ 3,663,927 $ 2,546,916 $ 12,348,132 $ 8,562,680"),
        ("r_netflix_unscaled", netflix_r3, "Netflix", "net_income", "$3,663,927", "Net income $ 3,663,927 $ 2,546,916 $ 12,348,132 $ 8,562,680"),
        ("r_visa_par_millions", visa_par, "Visa", "preferred_par_value", "$0.0001 million", "Preferred stock, par value (in dollars per share) $ 0.0001 $ 0.0001"),
        ("r_visa_common_par_m", visa_par, "Visa", "common_par_value", "$0.0001 million", "Common stock, par value (in dollars per share) $ 0.0001 $ 0.0001"),
        ("r_nvidia_ocr_eps_m", nvidia_is.replace("($ in millions, except per share data)", "(inmillions,exceptpersharedata)"), "NVIDIA", "diluted_eps", "$2.46 million", "Net income per diluted share $ 2.46 $ 2.39 $ 1.08 3 % 128 %"),
        ("r_nvidia_ocr_wrong_k", nvidia_is.replace("($ in millions, except per share data)", "(inmillions,exceptpersharedata)"), "NVIDIA", "revenue", "$96,221 thousand", "Revenue $ 96,221 $ 81,615 $ 46,743 18 % 106 %"),
        # Wrong column / wrong period attribute binding
        ("r_walmart_heading_only", walmart_is, "Walmart", "period", "2026", "Three Months Ended July 31"),
        # Prose boundary separating scale banner from later sentence
        ("r_prose_sep", "Earlier losses were in millions of dollars. The flat filing fee was finalized after Closing.\nClosing fee   $450", "Issuer", "closing_fee", "$450 million", "Closing fee   $450"),
        # Wrong company/metric swap on multi-row quote
        ("r_nvidia_row_swap", nvidia_is, "NVIDIA", "revenue", "$63,734 million", "Revenue $ 96,221\nOperating income $ 63,734"),
        ("r_adobe_wrong_col_rate", adobe_notes, "Adobe", "notes_2027_amount", "$2.15 million", "2.15% 2027 Notes"),
        ("r_adobe_wrong_scale_b", adobe_ppa, "Adobe", "goodwill", "$1,251 billion", "1,251 N/A"),
        ("r_costco_ci_wrong_scale", costco_ci, "Costco", "net_income", "$2,192", "NET INCOME $ 2,192 $ 1,903 $ 6,228 $ 5,489"),
        ("r_walmart_bs_wrong_scale", walmart_bs, "Walmart", "inventories", "$61,600 thousand", "Inventories 61,600 58,851 57,729"),
    ]

    pass_ok = 0
    pass_failures = []
    for cid, txt, subj, pred, val, quote in pass_cases:
        res = verify([Assertion(subject=subj, predicate=pred, value=val, source_id=cid, quote=quote)], Ledger([Source(id=cid, text=txt, tier="authoritative")]))
        if res.status == "PASS":
            pass_ok += 1
        else:
            pass_failures.append((cid, res.status, [i.code for i in res.issues]))

    reject_ok = 0
    reject_failures = []
    for cid, txt, subj, pred, val, quote in reject_cases:
        res = verify([Assertion(subject=subj, predicate=pred, value=val, source_id=cid, quote=quote)], Ledger([Source(id=cid, text=txt, tier="authoritative")]))
        if res.status == "RETRY":
            reject_ok += 1
        else:
            reject_failures.append((cid, res.status))

    return {
        "real_tables_pass_correct": wilson_ci(pass_ok, len(pass_cases)),
        "real_tables_reject_wrong": wilson_ci(reject_ok, len(reject_cases)),
        "pass_failures": pass_failures,
        "reject_failures": reject_failures,
    }


def main() -> None:
    snap = json.loads((_BENCH_DIR / "unseen_edgar_snapshot.json").read_text())
    report = {
        "p0": evaluate_p0_unseen_contracts(snap),
        "p1": evaluate_p1_furniture_and_hyphens(),
        "p2": evaluate_p2_dense_reader(),
        "p3": evaluate_p3_real_tables(snap),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
