#!/usr/bin/env python3
"""Matched-denominator P0 evaluation suite comparing before/after and v1/v2 on identical inputs.

Can target any repository checkout root (e.g. baseline commit worktree 4c818bd vs HEAD) via
``--repo-root`` so every metric uses the EXACT same fixtures, corpora, and denominators, and
reports both rejection/detection rates AND positive utility (PASS / supported / ACCEPT) with
two-sided 95% Wilson confidence intervals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if n <= 0:
        return {"k": 0, "n": 0, "rate": None, "ci_95_low": None, "ci_95_high": None}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return {
        "k": k,
        "n": n,
        "rate": round(p, 4),
        "ci_95_low": round(max(0.0, center - half), 4),
        "ci_95_high": round(min(1.0, center + half), 4),
    }


def _find_span(text: str, sub: str) -> tuple[int, int]:
    idx = text.find(sub)
    if idx < 0:
        raise ValueError(f"Substring {sub!r} not found in text")
    return idx, idx + len(sub)


def _eval_v1_v2_matched(repo_root: Path) -> dict[str, Any]:
    """Run v1 and v2 recorded readers under target checkout and emit matched-denominator stats."""
    benchmarks_dir = repo_root / "benchmarks"
    harness_dir = repo_root / "examples" / "harness2_securitization"
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(benchmarks_dir))
    sys.path.insert(0, str(harness_dir))

    from claimlayer.v4 import EvidenceLayer, RecordedProposalReader, build_corpus_reference_register
    from measure_reader_recall import (
        _CORPUS_SPECS,
        _collect_reference_divergence_collection,
        _conflict_found_in_register,
        _count_sentence_checks,
    )

    ann = json.loads((benchmarks_dir / "conflict_annotations.json").read_text(encoding="utf-8"))
    corpora_ann = ann["corpora"]
    versions = {
        "v1": harness_dir / "recorded_reads",
        "v2": harness_dir / "recorded_reads_v2",
    }

    per_ver: dict[str, Any] = {}
    for ver_name, rec_dir in versions.items():
        per_corpus: dict[str, Any] = {}
        for case_id, spec in _CORPUS_SPECS.items():
            short_name, builder, vocab, pol, dprof = spec
            c_ann = corpora_ann[case_id]
            corpus = builder()
            reader = RecordedProposalReader.from_file(rec_dir / f"{short_name}.json")
            layer = EvidenceLayer(vocabulary=vocab, policy=pol, reader=reader)
            sdocs = list(corpus.v4_sources.values())
            layer.ingest(sdocs)

            div_coll = _collect_reference_divergence_collection(corpus, domain_profile=dprof)
            divs = list(div_coll.divergences)
            sent_checks = _count_sentence_checks(corpus, divs)
            reg = build_corpus_reference_register(
                layer,
                vocab,
                divergence_collection=div_coll,
                expected_documents=corpus.v4_sources,
            )

            conflicts = list(c_ann.get("conflicts") or ())
            default_pri = str(c_ann.get("primary_source_id") or "")
            default_sec = str(c_ann.get("secondary_source_id") or "")
            found = sum(
                1 for cf in conflicts if _conflict_found_in_register(cf, corpus, reg, default_pri, default_sec)
            )
            sent_all: set[tuple[str, str, str, str]] = set()
            sent_qty: set[tuple[str, str, str, str]] = set()
            sent_txt: set[tuple[str, str, str, str]] = set()
            gaps_by_kind: dict[str, int] = {"quantity": 0, "textual": 0, "section": 0}
            for g in reg.gap_divergences:
                gk = str(g.get("kind") or "quantity")
                gaps_by_kind[gk] = gaps_by_kind.get(gk, 0) + 1
                skey = (
                    str(g.get("primary_source_id", "")),
                    str(g.get("secondary_source_id", "")),
                    str(g.get("section_key", "")),
                    str(g.get("sentence") or ""),
                )
                sent_all.add(skey)
                if gk == "quantity":
                    sent_qty.add(skey)
                elif gk == "textual":
                    sent_txt.add(skey)

            rr_prov = dict(getattr(reg, "reread_provenance", {}) or {})
            per_corpus[short_name] = {
                "clean_in_this_version": found == len(conflicts),
                "annotated_conflicts": len(conflicts),
                "conflicts_found": found,
                "divergent_sections": len(divs),
                "sentence_checks": sent_checks,
                "initial_gaps": int(rr_prov.get("initial_gap_count", len(reg.gap_divergences))),
                "closed_by_reread": int(rr_prov.get("closed_gap_count", 0)),
                "final_gaps": len(reg.gap_divergences),
                "gaps_by_kind": gaps_by_kind,
                "sentences_with_any_gap": min(len(sent_all), sent_checks),
                "sentences_with_qty_gap": min(len(sent_qty), sent_checks),
                "sentences_with_txt_gap": min(len(sent_txt), sent_checks),
                "escalates": len(reg.gap_divergences) > 0,
            }

        # Common clean subset across BOTH v1 and v2: {"apex", "ridgeline"}
        common_clean = ("apex", "ridgeline")
        all_ids = tuple(per_corpus.keys())

        def _agg(ids: tuple[str, ...]) -> dict[str, Any]:
            sc = sum(per_corpus[cid]["sentence_checks"] for cid in ids)
            dc = sum(per_corpus[cid]["divergent_sections"] for cid in ids)
            ac = sum(per_corpus[cid]["annotated_conflicts"] for cid in ids)
            fc = sum(per_corpus[cid]["conflicts_found"] for cid in ids)
            ig = sum(per_corpus[cid]["initial_gaps"] for cid in ids)
            cg = sum(per_corpus[cid]["closed_by_reread"] for cid in ids)
            fg = sum(per_corpus[cid]["final_gaps"] for cid in ids)
            esc = sum(1 for cid in ids if per_corpus[cid]["escalates"])
            sa = sum(per_corpus[cid]["sentences_with_any_gap"] for cid in ids)
            sq = sum(per_corpus[cid]["sentences_with_qty_gap"] for cid in ids)
            st = sum(per_corpus[cid]["sentences_with_txt_gap"] for cid in ids)
            return {
                "corpora_ids": list(ids),
                "corpora_count": len(ids),
                "divergent_sections": dc,
                "sentence_checks": sc,
                "recall": _wilson_ci(fc, ac),
                "initial_gaps": ig,
                "closed_by_reread": cg,
                "final_gaps": fg,
                "gaps_by_kind": {
                    k: sum(per_corpus[cid]["gaps_by_kind"][k] for cid in ids)
                    for k in ("quantity", "textual", "section")
                },
                "corpus_escalation_rate": _wilson_ci(esc, len(ids)),
                "sentence_any_gap_rate": _wilson_ci(sa, sc),
                "sentence_qty_gap_rate": _wilson_ci(sq, sc),
                "sentence_txt_gap_rate": _wilson_ci(st, sc),
            }

        per_ver[ver_name] = {
            "per_corpus": per_corpus,
            "matched_all_4_corpora": _agg(all_ids),
            "matched_common_clean_2_corpora": _agg(common_clean),
        }

    return per_ver


def _eval_p1_scaling_matched() -> dict[str, Any]:
    from claimlayer.semantic.models import SourceDocument
    from claimlayer.v4 import (
        AcceptancePolicy,
        EvidenceLayer,
        EvidenceRef,
        FieldSpec,
        PredicateSpec,
        Vocabulary,
        build_corpus_reference_register,
    )

    results: list[dict[str, Any]] = []
    for n_units in (50, 100, 198):
        half = n_units // 2
        entities = tuple((f"Tranche {i:04d}", ()) for i in range(half))
        vocab = Vocabulary(
            "p1_bench",
            "1.0",
            (PredicateSpec("fee_amount", FieldSpec("quantity", dimensions=("currency:USD",))),),
            entities=entities,
        )
        lines = [f"Line {i}: Tranche {i:04d} fee is ${1000 + i}.00 USD." for i in range(half)]
        txt = "\n".join(lines)
        d1 = SourceDocument("doc1", txt, authority_rank=100, weight=0.95)
        d2 = SourceDocument("doc2", txt, authority_rank=50, weight=0.90)
        layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        layer.ingest([d1, d2])
        units = []
        for i in range(half):
            sent = lines[i]
            for sdoc in (d1, d2):
                s, e = _find_span(sdoc.text, sent)
                units.append({
                    "proposal": {
                        "entity": f"Tranche {i:04d}",
                        "predicate": "fee_amount",
                        "value": 1000 + i,
                        "unit": "USD",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "basis": "explicit",
                    "evidence": [EvidenceRef(sdoc.document_id, sdoc.revision, s, e, sent)],
                })
        layer.ingest_units(units, attestor="scale-bench")
        t0 = time.perf_counter()
        _ = build_corpus_reference_register(layer, vocab, expected_documents={"doc1": d1, "doc2": d2})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results.append({"units": len(layer.units), "ms": round(elapsed_ms, 2)})

    t_lo = results[0]["ms"]
    t_hi = results[-1]["ms"]
    n_lo = results[0]["units"]
    n_hi = results[-1]["units"]
    alpha = math.log(max(t_hi, 1e-6) / max(t_lo, 1e-6)) / math.log(n_hi / n_lo)
    return {"points": results, "exponent_50_to_198": round(alpha, 3)}


def _eval_p2_battery_matched() -> dict[str, Any]:
    from claimlayer.semantic.models import SourceDocument
    from claimlayer.v4 import (
        AcceptancePolicy,
        EvidenceLayer,
        EvidenceRef,
        FieldSpec,
        PredicateSpec,
        Vocabulary,
    )
    from claimlayer.v4.corpus_obligations import build_corpus_reference_register, collect_reference_divergences
    from claimlayer.v4.domain_profile import DomainProfile
    from claimlayer.v4.engine import evaluate_unit_admissibility

    vocab = Vocabulary(
        "p2_matched",
        "1.0",
        (
            PredicateSpec("fee_rate", FieldSpec("quantity", dimensions=("currency:USD",)), conditions=(("event_of_default", FieldSpec("boolean")),)),
            PredicateSpec("closing_date", FieldSpec("date")),
            PredicateSpec("governing_law", FieldSpec("enum", enum=("new-york", "delaware"))),
            PredicateSpec("auto_renew", FieldSpec("boolean")),
            PredicateSpec("reserve_req", FieldSpec("quantity", dimensions=("currency:USD",))),
        ),
        entities=(("Facility", ()), ("Class A", ()), ("Class B", ())),
    )

    # 1. Table + KV list + Normalizers utility (positive) vs attack (negative)
    doc_txt = (
        "Schedule 1 Fee Grid\n"
        "| Tranche | Closing Fee ($ in thousands) |\n"
        "| --- | --- |\n"
        "| Class A | 25 |\n"
        "| Class B | 15 |\n\n"
        "Facility Summary: Closing Date: March 15, 2026; Auto Renew: yes; Reserve Requirement: $500,000 USD.\n"
        "Facility Summary: Governing Law: delaware; Collateral Cap: $2,000,000 USD.\n"
    )
    sdoc = SourceDocument("grid.txt", doc_txt, authority_rank=100, weight=0.95)
    hdr_sub = "| Tranche | Closing Fee ($ in thousands) |"
    row_a_sub = "| Class A | 25 |"
    row_b_sub = "| Class B | 15 |"
    hdr_s, hdr_e = _find_span(doc_txt, hdr_sub)
    row_a_s, row_a_e = _find_span(doc_txt, row_a_sub)
    row_b_s, row_b_e = _find_span(doc_txt, row_b_sub)
    kv_sub = "Facility Summary: Closing Date: March 15, 2026; Auto Renew: yes; Reserve Requirement: $500,000 USD."
    kv_s, kv_e = _find_span(doc_txt, kv_sub)
    kv2_sub = "Facility Summary: Governing Law: delaware; Collateral Cap: $2,000,000 USD."
    kv2_s, kv2_e = _find_span(doc_txt, kv2_sub)

    pos_props = [
        {
            "proposal": {
                "entity": "Class A",
                "predicate": "fee_rate",
                "value": 25000,
                "unit": "USD",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "table_context": {"scale": 1000, "currency": "USD", "column": "Closing Fee ($ in thousands)"},
            },
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, row_a_s, row_a_e, row_a_sub)],
            "field_evidence": {"table_header": EvidenceRef("grid.txt", sdoc.revision, hdr_s, hdr_e, hdr_sub)},
        },
        {
            "proposal": {"entity": "Facility", "predicate": "governing_law", "value": "delaware", "assertion_kind": "fact", "condition_status": "unconditional"},
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, kv2_s, kv2_e, kv2_sub)],
        },
        {
            "proposal": {"entity": "Facility", "predicate": "closing_date", "value": "2026-03-15", "assertion_kind": "fact", "condition_status": "unconditional"},
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, kv_s, kv_e, kv_sub)],
        },
        {
            "proposal": {"entity": "Facility", "predicate": "auto_renew", "value": True, "assertion_kind": "fact", "condition_status": "unconditional"},
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, kv_s, kv_e, kv_sub)],
        },
    ]
    neg_props = [
        # Wrong row/tranche mixup on table: Class B cited with Class A's fee (25,000)
        {
            "proposal": {
                "entity": "Class B",
                "predicate": "fee_rate",
                "value": 25000,
                "unit": "USD",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
                "table_context": {"scale": 1000, "currency": "USD", "column": "Closing Fee ($ in thousands)"},
            },
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, row_b_s, row_b_e, row_b_sub)],
            "field_evidence": {"table_header": EvidenceRef("grid.txt", sdoc.revision, hdr_s, hdr_e, hdr_sub)},
        },
        # Wrong boolean value: False when yes
        {
            "proposal": {"entity": "Facility", "predicate": "auto_renew", "value": False, "assertion_kind": "fact", "condition_status": "unconditional"},
            "basis": "explicit",
            "evidence": [EvidenceRef("grid.txt", sdoc.revision, kv_s, kv_e, kv_sub)],
        },
    ]

    def _is_supported_admissible(doc_obj: SourceDocument, item: dict[str, Any]) -> bool:
        ly = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        ly.ingest([doc_obj])
        res = ly.ingest_units([item], attestor="p2-eval")
        if res.get("units_admitted", 0) != 1:
            return False
        unit = ly.unit(res["unit_ids"][0])
        ev = evaluate_unit_admissibility(
            unit,
            vocabulary=vocab,
            policy=AcceptancePolicy(),
            documents={doc_obj.document_id: doc_obj},
            stored_field_evidences=ly.field_evidences,
        )
        if not ev.admissible:
            return False
        claim_prop = {k: v for k, v in item["proposal"].items() if k != "table_context"}
        chk = ly.check_claim(claim_prop)
        return chk.get("status") == "supported"

    pos_ok = sum(1 for p in pos_props if _is_supported_admissible(sdoc, p))
    neg_rej = sum(1 for p in neg_props if not _is_supported_admissible(sdoc, p))

    # Condition omission test (P2 #7)
    cond_txt = "Section 4: Facility fee is $10,000 USD if an Event of Default has occurred."
    cdoc = SourceDocument("cond.txt", cond_txt, authority_rank=100, weight=0.95)
    cs, ce = _find_span(cond_txt, cond_txt)
    cond_pos = {
        "proposal": {
            "entity": "Facility",
            "predicate": "fee_rate",
            "value": 10000,
            "unit": "USD",
            "assertion_kind": "requirement",
            "conditions": [{"field": "event_of_default", "operator": "eq", "value": True}],
            "condition_status": "conditional",
        },
        "basis": "explicit",
        "evidence": [EvidenceRef("cond.txt", cdoc.revision, cs, ce, cond_txt)],
    }
    cond_neg = {
        "proposal": {
            "entity": "Facility",
            "predicate": "fee_rate",
            "value": 10000,
            "unit": "USD",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
        "basis": "explicit",
        "evidence": [EvidenceRef("cond.txt", cdoc.revision, cs, ce, cond_txt)],
    }
    if _is_supported_admissible(cdoc, cond_pos):
        pos_ok += 1
    if not _is_supported_admissible(cdoc, cond_neg):
        neg_rej += 1
    ly_om = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    ly_om.ingest([sdoc])
    om_unit = {
        "proposal": {"entity": "Facility", "predicate": "governing_law", "value": "delaware", "assertion_kind": "fact", "condition_status": "unconditional"},
        "basis": "explicit",
        "evidence": [EvidenceRef("grid.txt", sdoc.revision, kv2_s, kv2_e, kv2_sub)],
    }
    ly_om.ingest_units([om_unit], attestor="p2-om")
    if ly_om.check_claim({"entity": "Facility", "predicate": "governing_law", "value": "delaware", "assertion_kind": "fact"}, validity="ignore").get("status") != "supported":
        neg_rej += 1

    # OCR / PDF diff noise vs true change (P2 #9)
    ocr_anchor = "Reserve Requirement: $500,000 USD.\n"
    ocr_pairs = [
        (ocr_anchor + "Clause 1: Collateral must remain\nsecured at all times.\nPage 1 of 4\n", ocr_anchor + "Clause 1: Collateral must remain secured at all times.", False),
        (ocr_anchor + "Clause 2: Borrower may pre-\npay the principal balance.", ocr_anchor + "Clause 2: Borrower may prepay the principal balance.", False),
        (ocr_anchor + "Clause 3: Guarantee—enforceable in full.\n", ocr_anchor + "Clause 3: Guarantee - enforceable in full.", False),
        (ocr_anchor + "Clause 4: Collateral must remain secured.", ocr_anchor + "Clause 4: Collateral must not remain secured.", True),
    ]
    dprof = DomainProfile(profile_id="p2_ocr", version="1.0", vocabulary=vocab)
    ocr_clean_ok = 0
    ocr_dirty_ok = 0
    for idx, (t1, t2, expect_gap) in enumerate(ocr_pairs):
        d1 = SourceDocument("a.txt", t1, authority_rank=100, weight=0.95)
        d2 = SourceDocument("b.txt", t2, authority_rank=50, weight=0.90)
        ly = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        ly.ingest([d1, d2])
        anchor_s1, anchor_e1 = _find_span(t1, "Reserve Requirement: $500,000 USD.")
        anchor_s2, anchor_e2 = _find_span(t2, "Reserve Requirement: $500,000 USD.")
        ly.ingest_units(
            [
                {
                    "proposal": {"entity": "Facility", "predicate": "reserve_req", "value": 500000, "unit": "USD", "assertion_kind": "requirement", "condition_status": "unconditional"},
                    "basis": "explicit",
                    "evidence": [EvidenceRef("a.txt", d1.revision, anchor_s1, anchor_e1, "Reserve Requirement: $500,000 USD.")],
                },
                {
                    "proposal": {"entity": "Facility", "predicate": "reserve_req", "value": 500000, "unit": "USD", "assertion_kind": "requirement", "condition_status": "unconditional"},
                    "basis": "explicit",
                    "evidence": [EvidenceRef("b.txt", d2.revision, anchor_s2, anchor_e2, "Reserve Requirement: $500,000 USD.")],
                },
            ],
            attestor="ocr-anchor",
        )
        dc = collect_reference_divergences([d1, d2], domain_profile=dprof)
        rg = build_corpus_reference_register(ly, vocab, divergence_collection=dc, expected_documents={"a.txt": d1, "b.txt": d2})
        has_gap = len(rg.gap_divergences) > 0
        if not expect_gap and not has_gap:
            ocr_clean_ok += 1
        elif expect_gap and has_gap:
            ocr_dirty_ok += 1

    return {
        "utility_admitted_and_supported_positive": _wilson_ci(pos_ok, len(pos_props) + 1),
        "adversarial_rejected_fidelity_and_condition": _wilson_ci(neg_rej, len(neg_props) + 2),
        "ocr_clean_no_false_gap": _wilson_ci(ocr_clean_ok, 3),
        "ocr_genuine_change_detected": _wilson_ci(ocr_dirty_ok, 1),
    }


def _eval_p3_battery_matched() -> dict[str, Any]:
    from claimlayer import Assertion, Ledger, Policy, Source
    from claimlayer.semantic.models import SourceDocument
    from claimlayer.v4 import (
        AcceptancePolicy,
        EvidenceLayer,
        EvidenceRef,
        FieldSpec,
        PredicateSpec,
        Vocabulary,
        build_corpus_reference_register,
        collect_reference_divergences,
    )
    from claimlayer.v4.domain_profile import DomainProfile

    # P3 #12: v3 verify() local quantifier/restrictor/negation scope (8 attacks + 6 positives)
    attacks = [
        ("Eligible Assets includes cash and treasury bills only if conditions apply.", "Eligible Assets", "scope", "Eligible Assets shall only include cash"),
        ("Borrower requires insurance and only waives reporting upon closing.", "Borrower", "covenant", "Borrower only requires insurance"),
        ("Notice must be delivered at least 5 business days prior to Closing.", "Notice", "notice_period", "at most 5 business days"),
        ("Facility borrowings shall not exceed $10,000,000 at any time.", "Facility", "borrowing_limit", "at least $10,000,000"),
        ("Some Collateral items remain excluded from the borrowing base.", "Collateral", "inclusion", "all Collateral items remain in the borrowing base"),
        ("Lenders are entitled to default interest upon any payment default.", "Lenders", "entitlement", "Lenders are never entitled to default interest"),
        ("Collateral solely comprises cash reserves.", "Collateral", "composition", "Collateral comprises all cash reserves"),
        ("Guarantor shall not pay $10,000,000 without written Lender consent.", "Guarantor", "payment_duty", "Guarantor shall pay $10,000,000"),
    ]
    positives = [
        ("Class B Notes bear interest at 5.80% per annum.", "Class B Notes", "coupon_rate", "5.80%"),
        ("Notice must be delivered not less than 5 business days prior to Closing.", "Notice", "notice_period", "at least 5 business days"),
        ("Exemption applies solely to Class A Notes.", "Class A Notes", "exemption", "solely to Class A Notes"),
        ("All Senior Notes rank pari passu in right of payment.", "Senior Notes", "priority", "All Senior Notes rank pari passu"),
        ("Guarantor shall pay $10,000,000 upon acceleration.", "Guarantor", "payment_duty", "Guarantor shall pay $10,000,000"),
        ("Reserve Account balance must equal $2,500,000 at Closing.", "Reserve Account", "balance", "$2,500,000"),
    ]

    def _run_v3(txt: str, subj: str, pred: str, val: str) -> str:
        ldg = Ledger([Source(id="doc1", title="Contract", text=txt, tier="authoritative")])
        v = ldg.verify(
            assertions=[Assertion(id="a1", subject=subj, predicate=pred, value=val, source_id="doc1", quote=txt)],
            policy=Policy(require_exact_quote=True, require_quantity_match=True),
        )
        return v.status

    v3_atk_rej = sum(1 for args in attacks if _run_v3(*args) == "RETRY")
    v3_pos_pass = sum(1 for args in positives if _run_v3(*args) == "PASS")

    # P3 #10: Deleted clauses (2 deleted + 1 identical control)
    vocab = Vocabulary(
        "p3_del",
        "1.0",
        (PredicateSpec("cash_reserve", FieldSpec("quantity", dimensions=("currency:USD",))),),
        entities=(("Issuer", ()),),
    )
    dprof = DomainProfile(profile_id="p3_del", version="1.0", vocabulary=vocab)
    del_cases = [
        ("Section 1: Waterfall.\nCash Reserve is $2,000,000.\nServicing fee shall not exceed $50,000.\nGuarantor remains liable upon acceleration.", "Section 1: Waterfall.\nCash Reserve is $2,000,000.", 2000000, "Cash Reserve is $2,000,000.", True),
        ("Section 2: Covenants.\nCash Reserve is $3,000,000.\nCollateral shall never be released without consent.", "Section 2: Covenants.\nCash Reserve is $3,000,000.", 3000000, "Cash Reserve is $3,000,000.", True),
        ("Section 3: Clean.\nCash Reserve is $4,000,000.\nTrustee fee is paid monthly.", "Section 3: Clean.\nCash Reserve is $4,000,000.\nTrustee fee is paid monthly.", 4000000, "Cash Reserve is $4,000,000.", False),
    ]
    del_tp = 0
    del_tn = 0
    for ptxt, stxt, cval, anchor_sent, expect_gap in del_cases:
        dp = SourceDocument("p.txt", ptxt, authority_rank=100, weight=0.95)
        ds = SourceDocument("s.txt", stxt, authority_rank=50, weight=0.90)
        ly = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
        ly.ingest([dp, ds])
        ps1, pe1 = _find_span(ptxt, anchor_sent)
        ss1, se1 = _find_span(stxt, anchor_sent)
        ly.ingest_units(
            [
                {
                    "proposal": {"entity": "Issuer", "predicate": "cash_reserve", "value": cval, "unit": "USD", "assertion_kind": "requirement", "condition_status": "unconditional"},
                    "basis": "explicit",
                    "evidence": [EvidenceRef("p.txt", dp.revision, ps1, pe1, anchor_sent)],
                },
                {
                    "proposal": {"entity": "Issuer", "predicate": "cash_reserve", "value": cval, "unit": "USD", "assertion_kind": "requirement", "condition_status": "unconditional"},
                    "basis": "explicit",
                    "evidence": [EvidenceRef("s.txt", ds.revision, ss1, se1, anchor_sent)],
                },
            ],
            attestor="p3-del",
        )
        dc = collect_reference_divergences([dp, ds], domain_profile=dprof)
        rg = build_corpus_reference_register(ly, vocab, divergence_collection=dc, expected_documents={"p.txt": dp, "s.txt": ds})
        has_gap = len(rg.gap_divergences) > 0
        if expect_gap and has_gap:
            del_tp += 1
        elif not expect_gap and not has_gap:
            del_tn += 1

    # P3 #11: Dense 1-unit-per-sentence overlap masking (2 masked unextracted conflicts vs 1 clean shared sentence)
    p11_p = (
        "Section 1:\n"
        "Clause A: Cash Reserve equals $2,000,000 and Debt Cap equals $10,000,000.\n"
        "Clause B: Cash Reserve equals $3,000,000 and Guarantor shall not be released.\n"
        "Clause C: Cash Reserve equals $4,000,000 and Trustee fees are paid monthly."
    )
    p11_s = (
        "Section 1:\n"
        "Clause A: Cash Reserve equals $2,000,000 and Debt Cap equals $15,000,000.\n"
        "Clause B: Cash Reserve equals $3,000,000 and Guarantor shall be released.\n"
        "Clause C: Cash Reserve equals $4,000,000 and Trustee fees are paid monthly."
    )
    d1 = SourceDocument("m1.txt", p11_p, authority_rank=100, weight=0.95)
    d2 = SourceDocument("m2.txt", p11_s, authority_rank=50, weight=0.90)
    ly11 = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    ly11.ingest([d1, d2])
    c_lines_1 = p11_p.splitlines()[1:]
    c_lines_2 = p11_s.splitlines()[1:]
    u11 = []
    for l1, l2, val in zip(c_lines_1, c_lines_2, (2000000, 3000000, 4000000)):
        for sd, ln in ((d1, l1), (d2, l2)):
            ss, se = _find_span(sd.text, ln)
            u11.append({
                "proposal": {"entity": "Issuer", "predicate": "cash_reserve", "value": val, "unit": "USD", "assertion_kind": "requirement"},
                "basis": "explicit",
                "evidence": [EvidenceRef(sd.document_id, sd.revision, ss, se, ln)],
            })
    ly11.ingest_units(u11, attestor="p3-mask")
    dc11 = collect_reference_divergences([d1, d2], domain_profile=dprof)
    rg11 = build_corpus_reference_register(ly11, vocab, divergence_collection=dc11, expected_documents={"m1.txt": d1, "m2.txt": d2})
    gap_sents = " || ".join(str(g.get("sentence") or "") for g in rg11.gap_divergences)
    mask_detected = int("Clause A" in gap_sents) + int("Clause B" in gap_sents)
    mask_clean_ok = int("Clause C" not in gap_sents)

    return {
        "v3_scope_negation_true_reject": _wilson_ci(v3_atk_rej, len(attacks)),
        "v3_positive_utility_PASS": _wilson_ci(v3_pos_pass, len(positives)),
        "p3_deleted_clause_true_detect": _wilson_ci(del_tp, 2),
        "p3_deleted_clause_clean_no_false_gap": _wilson_ci(del_tn, 1),
        "p3_overlap_masking_true_detect": _wilson_ci(mask_detected, 2),
        "p3_overlap_clean_no_false_gap": _wilson_ci(mask_clean_ok, 1),
    }


def _eval_heldout_utility(repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root / "benchmarks"))
    from heldout_multidomain_eval import run_heldout_evaluation

    rep = run_heldout_evaluation(repetitions=1)
    arm = rep["summary_by_arm"]["v3_v4"]
    fa = int(arm["false_accepts"])
    sd = int(arm["seeded_total"])
    fr = int(arm["false_rejects"])
    cl = int(arm["clean_total"])
    return {
        "v3_v4_true_reject_seeded": _wilson_ci(sd - fa, sd),
        "v3_v4_false_accept_seeded": _wilson_ci(fa, sd),
        "v3_v4_clean_accept_utility": _wilson_ci(cl - fr, cl),
        "v3_v4_false_reject_clean": _wilson_ci(fr, cl),
        "v3_v4_sla_passed": bool(arm["sla_passed"]),
    }


def run_all(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    return {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "p0_1_v1_vs_v2_matched": _eval_v1_v2_matched(repo_root),
        "p1_scaling_matched": _eval_p1_scaling_matched(),
        "p2_fidelity_and_noise_matched": _eval_p2_battery_matched(),
        "p3_recall_and_scope_matched": _eval_p3_battery_matched(),
        "p0_3_heldout_multidomain": _eval_heldout_utility(repo_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run matched-denominator P0/P1/P2/P3 evaluation.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    out = run_all(args.repo_root)
    txt = json.dumps(out, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
