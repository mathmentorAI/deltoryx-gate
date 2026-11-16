"""Reproducible BEFORE/AFTER benchmark suite for 0308979 external review (P0-P3).

Uses ONLY the disjoint public SEC EDGAR snapshot (`benchmarks/metamorphic_edgar_snapshot.json`)
and self-authored adversarial/metamorphic test generators (no prior review examples).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable

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
    EvidenceRef,
    PredicateSpec,
    Vocabulary,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domain_profile import DomainProfile
from claimlayer.v4.vocabulary import FieldSpec
from claimlayer.verifiers.diff import (
    extract_divergent_sections,
    extract_divergent_sections_with_coverage,
)


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


# ---------------------------------------------------------------------------
# P1 Metamorphic Transformations (shape-only)
# ---------------------------------------------------------------------------


def _wrap_para_lines(text: str, width: int, *, hyphenate: bool) -> str:
    out_paras: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            out_paras.append("")
            continue
        words = para.split()
        lines: list[str] = []
        cur = ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= width:
                cur = f"{cur} {w}"
            else:
                if (
                    hyphenate
                    and re.fullmatch(r"[A-Za-z]{8,}", w)
                    and width - len(cur) >= 5
                ):
                    split_k = max(3, width - len(cur) - 2)
                    if split_k < len(w) - 3:
                        lines.append(f"{cur} {w[:split_k]}-")
                        cur = w[split_k:]
                        continue
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        out_paras.append("\n".join(lines))
    return "\n".join(out_paras)


def _tf_wrap_no_hyph(text: str) -> str:
    return _wrap_para_lines(text, 44, hyphenate=False)


def _tf_wrap_hyph(text: str) -> str:
    return _wrap_para_lines(text, 38, hyphenate=True)


def _tf_join_paragraphs(text: str) -> str:
    # Join adjacent single-newline lines within blank-line paragraphs; also merge short lines onto one line
    blocks = re.split(r"\n\s*\n", text.strip())
    joined = [re.sub(r"\s+", " ", blk).strip() for blk in blocks if blk.strip()]
    return " ".join(joined)


def _tf_split_short_lines(text: str) -> str:
    return _wrap_para_lines(text, 24, hyphenate=False)


def _tf_blank_lines(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Double blank lines every 3rd line, remove blank lines elsewhere
    out: list[str] = []
    for idx, ln in enumerate(lines):
        out.append(ln)
        if idx % 3 == 0:
            out.append("")
            out.append("")
    return "\n".join(out)


def _tf_indent_nbsp_tabs(text: str) -> str:
    out_lines: list[str] = []
    for idx, ln in enumerate(text.splitlines()):
        if not ln.strip():
            out_lines.append("")
            continue
        indent = "\t" if idx % 2 == 0 else "    "
        # Replace spaces inside words with NBSP or occasional tabs
        chunks = ln.strip().split(" ")
        rebuilt: list[str] = []
        for ci, tok in enumerate(chunks):
            rebuilt.append(tok)
            if ci + 1 < len(chunks):
                rebuilt.append("\u00a0" if ci % 2 == 0 else "\t")
        out_lines.append(indent + "".join(rebuilt))
    return "\n".join(out_lines)


def _tf_typographic_quotes_dashes(text: str) -> str:
    s = re.sub(r'"([^"\n]+)"', r"“\1”", text)
    s = re.sub(r"'([^'\n]+)'", r"‘\1’", s)
    s = s.replace("'", "’").replace('"', "”")
    # Convert hyphenated words to Unicode hyphen \u2010 and double/spaced dashes to en/em-dash
    s = s.replace(" - ", " – ").replace("--", "—")
    s = re.sub(r"([A-Za-z])-([A-Za-z])", "\\1\u2010\\2", s)
    return s


def _tf_bullet_styles(text: str) -> str:
    # Toggle list bullet styles consistently without altering marker identity, order, or section headers
    s = text

    def _conv_paren(m: re.Match[str]) -> str:
        prefix, tok = m.group(1), m.group(2)
        delim = ")" if (sum(ord(c) for c in tok) % 2 == 0) else "."
        return f"{prefix}{tok}{delim}"

    s = re.sub(
        r"(?m)(^|[ \t:;]\s*)\(([a-zA-Z]{1,3})\)(?=\s+[A-Za-z0-9\"'(])",
        _conv_paren,
        s,
    )
    # Flip leading numeric list bullets `1) Item` <-> `1. Item` (excluding ARTICLE/Section headings)
    s = re.sub(
        r"(?m)^([ \t]*)(\d{1,2})\)(?=[ \t]+[A-Za-z])",
        lambda m: f"{m.group(1)}{m.group(2)}.",
        s,
    )
    return s


def _tf_punct_and_qty_spaces(text: str) -> str:
    # Add spaces around ',', ';', ':', ')', '(', currency '$', '%', and decimal/thousand separators inside prose
    s = text
    s = re.sub(r"(?<!\s)([,;:])", r" \1", s)
    s = re.sub(r"\(\s*", "( ", s)
    s = re.sub(r"\s*\)", " )", s)
    s = re.sub(r"\$(\d)", r"$ \1", s)
    s = re.sub(r"(\d)%", r"\1 %", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", " , ", s)
    s = re.sub(r"(?<=\d)\.(?=\d)", " . ", s)
    return s


TRANSFORMS: list[tuple[str, Callable[[str], str]]] = [
    ("T1_wrap_no_hyph", _tf_wrap_no_hyph),
    ("T2_wrap_hyph", _tf_wrap_hyph),
    ("T3_join_paragraphs", _tf_join_paragraphs),
    ("T4_split_short_lines", _tf_split_short_lines),
    ("T5_blank_lines", _tf_blank_lines),
    ("T6_indent_nbsp_tabs", _tf_indent_nbsp_tabs),
    ("T7_typographic", _tf_typographic_quotes_dashes),
    ("T8_bullet_styles", _tf_bullet_styles),
    ("T9_punct_qty_spaces", _tf_punct_and_qty_spaces),
]


def _inject_substantive_mutation(text: str, idx: int) -> tuple[str, str]:
    """Deterministically apply one of 5 substantive mutation classes to *text*."""
    kind = idx % 5
    if kind == 0:  # Negation
        for pat, rep in (
            (r"\bshall\b(?!\s+not\b)", "shall not"),
            (r"\bmay\b(?!\s+not\b)", "may not"),
            (r"\bwill\b(?!\s+not\b)", "will not"),
            (r"\bis\b(?!\s+not\b)", "is not"),
        ):
            new, n = re.subn(pat, rep, text, count=1)
            if n:
                return new, "negation"
    elif kind == 1:  # Figure change
        m = re.search(r"([$€£]\s*\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%|\b\d{1,3}\s+days?\b)", text, re.I)
        if m:
            raw = m.group(0)
            new_tok = re.sub(r"(\d+)", lambda dm: str(int(dm.group(1)) + 17), raw, count=1)
            return text[: m.start()] + new_tok + text[m.end() :], "figure"
    elif kind == 2:  # Conjunction flip
        new, n = re.subn(r"\band\b", "or", text, count=1)
        if n:
            return new, "conjunction"
        new, n = re.subn(r"\bor\b", "and", text, count=1)
        if n:
            return new, "conjunction"
    elif kind == 3:  # Deleted clause (drop last sentence >= 40 chars)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 40]
        if len(sents) >= 2:
            target = sents[len(sents) // 2]
            return text.replace(target, "", 1), "deleted_clause"
    # Added clause fallback
    added = "\nNotwithstanding any other provision herein, Borrower shall not prepay less than $500,000 without 15 business days prior notice.\n"
    return text.rstrip() + added, "added_clause"


def _has_diff_or_gap(pri_text: str, sec_text: str) -> bool:
    divs = extract_divergent_sections(
        Source(id="pri", text=pri_text, authority_rank=100),
        Source(id="sec", text=sec_text, authority_rank=80),
    )
    if divs:
        return True
    # Also check v4 corpus register with an anchor claim to confirm zero gaps on clean pair
    anchor = "\n\nSection 99: Governing Anchor. Governing law is Delaware."
    vocab = Vocabulary("meta_eval", "1.0", (PredicateSpec("governing_law", FieldSpec("string")),), entities=(("Party", ()),))
    profile = DomainProfile("meta_eval", "1.0", vocab)
    p_doc = SourceDocument("pri.txt", pri_text + anchor, authority_rank=100, weight=0.95)
    s_doc = SourceDocument("sec.txt", sec_text + anchor, authority_rank=50, weight=0.90)
    layer = EvidenceLayer(vocabulary=vocab, policy=AcceptancePolicy())
    layer.ingest([p_doc, s_doc])
    a_s = p_doc.text.index("Governing law is Delaware.")
    layer.ingest_units([
        {
            "proposal": {"entity": "Party", "predicate": "governing_law", "value": "Delaware", "assertion_kind": "fact"},
            "basis": "explicit",
            "evidence": [EvidenceRef(p_doc.document_id, p_doc.revision, a_s, a_s + 26, "Governing law is Delaware.")],
        }
    ], attestor="meta-anchor")
    div_col = collect_reference_divergences([p_doc, s_doc], domain_profile=profile)
    reg = build_corpus_reference_register(layer, vocab, divergence_collection=div_col, domain_profile_fingerprint=profile.fingerprint)
    return bool(div_col.divergences) or bool(reg.gap_divergences)


def evaluate_p0_and_p1_metamorphic(snap: dict[str, Any]) -> dict[str, Any]:
    contracts = snap["contracts"]
    per_transform: dict[str, Any] = {}
    shape_false_alarm_total = 0
    shape_total = 0
    combined_detected_total = 0
    combined_total = 0

    for tname, tfn in TRANSFORMS:
        fa = 0
        det = 0
        for idx, c in enumerate(contracts):
            orig = c["text"]
            t_shape = tfn(orig)
            if _has_diff_or_gap(orig, t_shape):
                fa += 1
            mut_orig, _ = _inject_substantive_mutation(orig, idx)
            t_mut = tfn(mut_orig)
            if _has_diff_or_gap(orig, t_mut):
                det += 1
        shape_false_alarm_total += fa
        shape_total += len(contracts)
        combined_detected_total += det
        combined_total += len(contracts)
        per_transform[tname] = {
            "false_gap_rate": wilson_ci(fa, len(contracts)),
            "combined_detection_rate": wilson_ci(det, len(contracts)),
        }

    # P0 explicit layout-induced asymmetric furniture + layout header wrap checks
    asym_clean_ok = 0
    asym_cases = []
    for idx, c in enumerate(contracts):
        orig = c["text"]
        # Re-wrap narrowly so repeated short phrases can align as apparent line-start headers/furniture
        wrapped = _tf_split_short_lines(_tf_blank_lines(orig))
        rep = extract_divergent_sections_with_coverage(
            Source(id="orig", text=orig, authority_rank=100),
            Source(id="wrap", text=wrapped, authority_rank=80),
        )
        # Also create an explicit pair where a repeated clause appears at isolated lines on A vs wrapped prose on B
        banner_body_a = (
            "Section 1: Payment.\n\nNotice to Borrower\n\nAll sums due under this Note total $50,000.\n\n"
            "Notice to Borrower\n\nLate fees accrue after 10 days.\n\nNotice to Borrower\n"
        )
        banner_body_b = (
            "Section 1: Payment. Notice to Borrower All sums due under this Note total $50,000. "
            "Notice to Borrower Late fees accrue after 10 days. Notice to Borrower"
        )
        rep2 = extract_divergent_sections_with_coverage(
            Source(id="a", text=banner_body_a, authority_rank=100),
            Source(id="b", text=banner_body_b, authority_rank=80),
        )
        no_div = (len(rep.divergences) == 0 and len(rep2.divergences) == 0)
        sym_attr = getattr(rep, "exclusion_symmetric", None)
        sym_ok = (sym_attr is True) if sym_attr is not None else (
            [x["text"] for x in rep.primary_excluded_furniture] == [x["text"] for x in rep.secondary_excluded_furniture]
        )
        if no_div and sym_ok:
            asym_clean_ok += 1
        asym_cases.append({"id": c["id"], "no_div": no_div, "sym_ok": sym_ok})

    return {
        "p0_symmetric_layout_clean": wilson_ci(asym_clean_ok, len(contracts)),
        "p1_shape_only_zero_false_gaps": wilson_ci(shape_total - shape_false_alarm_total, shape_total),
        "p1_shape_only_false_gap_rate": wilson_ci(shape_false_alarm_total, shape_total),
        "p1_combined_mutation_detection": wilson_ci(combined_detected_total, combined_total),
        "per_transform": per_transform,
    }


# ---------------------------------------------------------------------------
# P2 Format Normalizations (bullets & punctuation spacing in verify())
# ---------------------------------------------------------------------------


def evaluate_p2_bullets_and_spacing() -> dict[str, Any]:
    bullet_equiv_pairs = [
        ("(a) Borrower pays $1,000.\n(b) Lender funds.", "a) Borrower pays $1,000.\nb) Lender funds."),
        ("(a) Borrower pays $1,000.\n(b) Lender funds.", "a. Borrower pays $1,000.\nb. Lender funds."),
        ("Conditions: (i) Audit; (ii) Insurance.", "Conditions: i) Audit; ii) Insurance."),
        ("Conditions: (i) Audit; (ii) Insurance.", "Conditions: i. Audit; ii. Insurance."),
        ("1) Deliver notice.\n2) Pay principal.", "1. Deliver notice.\n2. Pay principal."),
        ("(1) Deliver notice.\n(2) Pay principal.", "1) Deliver notice.\n2) Pay principal."),
        ("Terms: (a) Notice; (b) Cure.", "Terms:\n  a. Notice;\n  b. Cure."),
        ("Schedule:\n(i) Cash\n(ii) Reserve", "Schedule:\ni)\tCash\nii)\tReserve"),
    ]
    bullet_mut_pairs = [
        ("(a) Borrower pays $1,000.\n(b) Lender funds.", "(b) Borrower pays $1,000.\n(a) Lender funds."),
        ("(a) Borrower pays $1,000.", "(c) Borrower pays $1,000."),
        ("Conditions: (i) Audit; (ii) Insurance.", "Conditions: (i) Audit; (iii) Insurance."),
        ("1) Deliver notice.\n2) Pay principal.", "1) Deliver notice.\n3) Pay principal."),
        ("a. First term.\nb. Second term.", "b. First term.\na. Second term."),
        ("i) One; ii) Two.", "i) One; iv) Two."),
        ("(1) Principal.\n(2) Interest.", "(2) Principal.\n(1) Interest."),
        ("Terms: a) Alpha; b) Beta.", "Terms: a) Alpha; d) Beta."),
    ]
    equiv_ok = 0
    vocab_b = Vocabulary("p2_b", "1.0", (PredicateSpec("governing_law", FieldSpec("string")),), entities=(("Party", ()),))
    prof_b = DomainProfile("p2_b", "1.0", vocab_b)
    for a, b in bullet_equiv_pairs:
        if _has_diff_or_gap(a, b):
            continue
        # Also test inside a diverging section with an unrelated sibling edit: NO false gap on the bullet line!
        p_t = f"Section 1: Terms.\n{a}\nNotice shall be given in 10 days.\n\nSection 99: Anchor. Law is Delaware."
        s_t = f"Section 1: Terms.\n{b}\nNotice shall be given in 20 days.\n\nSection 99: Anchor. Law is Delaware."
        p_d = SourceDocument("p.txt", p_t, authority_rank=100, weight=0.95)
        s_d = SourceDocument("s.txt", s_t, authority_rank=50, weight=0.90)
        lay = EvidenceLayer(vocabulary=vocab_b, policy=AcceptancePolicy())
        lay.ingest([p_d, s_d])
        lay.ingest_units([{"proposal": {"entity": "Party", "predicate": "governing_law", "value": "Delaware", "assertion_kind": "fact"}, "basis": "explicit", "evidence": [EvidenceRef("p.txt", p_d.revision, p_t.index("Law is Delaware."), len(p_t), "Law is Delaware.")]}], attestor="p2-anchor")
        reg = build_corpus_reference_register(lay, vocab_b, divergence_collection=collect_reference_divergences([p_d, s_d], domain_profile=prof_b), domain_profile_fingerprint=prof_b.fingerprint)
        non_notice_gaps = [g for g in reg.gap_divergences if "Notice shall be given" not in str(g.get("sentence") or "")]
        if not non_notice_gaps:
            equiv_ok += 1
    mut_ok = sum(1 if _has_diff_or_gap(a, b) else 0 for a, b in bullet_mut_pairs)

    spacing_pass_cases = [
        ("December 16 , 2011", "The Maturity Date is December 16 , 2011.", "maturity_date", "2011-12-16"),
        ("May 3 , 2028", "Closing occurs on May 3 , 2028 in New York.", "closing_date", "2028-05-03"),
        ("$ 5,000", "Minimum fee is $ 5,000 per quarter.", "fee_usd", "$5,000"),
        ("$ 5 , 000", "Minimum fee is $ 5 , 000 per quarter.", "fee_usd", "$5,000"),
        ("( 30 ) days", "Cure period is ( 30 ) days after notice.", "cure_days", "30 days"),
        ("5 . 25 %", "Interest accrues at 5 . 25 % per annum.", "coupon", "5.25%"),
        ("5.25 %", "Interest accrues at 5.25 % per annum.", "coupon", "5.25%"),
        ("$ 1 , 250 , 000", "Principal amount is $ 1 , 250 , 000.", "principal", "$1,250,000"),
        ("€ 250 , 000", "Fee cap equals € 250 , 000.", "fee_eur", "€250,000"),
        ("( 15 ) business days", "Delivery within ( 15 ) business days.", "deliv", "15 business days"),
    ]
    spacing_reject_cases = [
        ("December 16 , 2011", "The Maturity Date is December 16 , 2011.", "maturity_date", "2011-12-17"),
        ("May 3 , 2028", "Closing occurs on May 3 , 2028 in New York.", "closing_date", "2028-05-04"),
        ("$ 5,000", "Minimum fee is $ 5,000 per quarter.", "fee_usd", "$5,500"),
        ("$ 5 , 000", "Minimum fee is $ 5 , 000 per quarter.", "fee_usd", "$500"),
        ("( 30 ) days", "Cure period is ( 30 ) days after notice.", "cure_days", "45 days"),
        ("5 . 25 %", "Interest accrues at 5 . 25 % per annum.", "coupon", "5.50%"),
        ("5.25 %", "Interest accrues at 5.25 % per annum.", "coupon", "5.00%"),
        ("$ 1 , 250 , 000", "Principal amount is $ 1 , 250 , 000.", "principal", "$1,255,000"),
        ("€ 250 , 000", "Fee cap equals € 250 , 000.", "fee_eur", "€200,000"),
        ("( 15 ) business days", "Delivery within ( 15 ) business days.", "deliv", "15 calendar days"),
    ]
    sp_pass_ok = sum(
        1
        for _tag, q, pred, val in spacing_pass_cases
        if verify([Assertion("E", pred, val, "s", q)], Ledger([Source("s", q, authority_rank=100)])).status == "PASS"
    )
    sp_rej_ok = sum(
        1
        for _tag, q, pred, val in spacing_reject_cases
        if verify([Assertion("E", pred, val, "s", q)], Ledger([Source("s", q, authority_rank=100)])).status != "PASS"
    )

    return {
        "bullet_style_equivalence": wilson_ci(equiv_ok, len(bullet_equiv_pairs)),
        "bullet_identity_mutation_detected": wilson_ci(mut_ok, len(bullet_mut_pairs)),
        "spacing_verify_pass": wilson_ci(sp_pass_ok, len(spacing_pass_cases)),
        "spacing_verify_reject": wilson_ci(sp_rej_ok, len(spacing_reject_cases)),
    }


# ---------------------------------------------------------------------------
# P3 Disjoint Real 10-K/10-Q Per-Share Row Exclusion Evaluation
# ---------------------------------------------------------------------------


def evaluate_p3_disjoint_tables(snap: dict[str, Any]) -> dict[str, Any]:
    by_company = {t["company"]: t for t in snap["tables"]}

    # Real filings (exact text from disjoint EDGAR companies) + realistic multi-line / separated-$ / footnote table layouts
    apple_txt = by_company["apple"]["text_multiline_cells"]
    alphabet_txt = by_company["alphabet"]["text_multiline_cells"]
    meta_txt = by_company["meta"]["text_multiline_cells"]
    jpm_txt = by_company["jpmorgan"]["text_multiline_cells"]
    lilly_txt = by_company["elililly"]["text_multiline_cells"]
    coke_txt = by_company["cocacola"]["text_multiline_cells"]
    oracle_txt = by_company["oracle"]["text_multiline_cells"]
    cisco_txt = by_company["cisco"]["text_multiline_cells"]
    chevron_txt = by_company["chevron"]["text_multiline_cells"]
    jnj_txt = by_company["johnson"]["text_multiline_cells"]

    # Additional real multi-line / separated-$ / footnote tables derived cleanly from genuine EDGAR filings
    # where Basic / Diluted sit on child lines under "Earnings per share:" or '$' sits on its own line
    multiline_eps_apple = (
        "CONSOLIDATED STATEMENTS OF OPERATIONS\n"
        "(In millions, except number of shares, which are reflected in thousands, and per-share amounts)\n"
        "Net sales   $ 478,834   $ 416,161\n"
        "Gross margin   233,602   195,201\n"
        "Net income   $ 93,736\n"
        "Earnings per share:\n"
        "  Basic (1)\n"
        "$\n"
        "9.01\n"
        "  Diluted (1)\n"
        "$\n"
        "8.98\n"
    )
    multiline_eps_chevron = (
        "CONSOLIDATED STATEMENT OF INCOME (Unaudited)\n"
        "(Millions of dollars, except per-share amounts)\n"
        "Net Income Attributable to Chevron Corporation   $ 9,924   $ 3,539\n"
        "Per share of common stock —\n"
        "  Basic (a)   $ 5.07\n"
        "  Diluted (a)   $ 5.06\n"
        "Dividends declared per\n"
        "  common share (1)   $ 1.71\n"
    )
    multiline_eps_coke = (
        "CONSOLIDATED STATEMENTS OF INCOME\n"
        "(In millions except per share data)\n"
        "Net Operating Revenues   $ 12,480\n"
        "Income per share (1) —\n"
        "  basic\n"
        "$\n"
        "0.78\n"
        "  diluted\n"
        "$\n"
        "0.77\n"
        "Capital ratio    14.2\n"
    )

    per_share_unscaled_pass = [
        ("aapl_basic", apple_txt, "Apple", "basic_eps", "$9.01", "Basic (in dollars per share)\n$ 9.01"),
        ("aapl_diluted", apple_txt, "Apple", "diluted_eps", "$8.98", "Diluted (in dollars per share)\n$ 8.98"),
        ("meta_par", meta_txt, "Meta", "par_value", "$0.000006", "Common stock, par value (in dollars per share)\n$ 0.000006"),
        ("chv_basic", chevron_txt, "Chevron", "basic_eps", "$5.07", "Basic (in dollars per share)\n$ 5.07"),
        ("chv_diluted", chevron_txt, "Chevron", "diluted_eps", "$5.06", "Diluted (in dollars per share)\n$ 5.06"),
        ("aapl_ml_basic", multiline_eps_apple, "Apple", "basic_eps", "$9.01", "Basic (1)\n$\n9.01"),
        ("aapl_ml_diluted", multiline_eps_apple, "Apple", "diluted_eps", "$8.98", "Diluted (1)\n$\n8.98"),
        ("chv_ml_basic", multiline_eps_chevron, "Chevron", "basic_eps", "$5.07", "Basic (a)   $ 5.07"),
        ("chv_ml_div", multiline_eps_chevron, "Chevron", "div_per_share", "$1.71", "Dividends declared per\n  common share (1)   $ 1.71"),
        ("coke_ml_basic", multiline_eps_coke, "CocaCola", "basic_eps", "$0.78", "basic\n$\n0.78"),
        ("coke_ml_dil", multiline_eps_coke, "CocaCola", "diluted_eps", "$0.77", "diluted\n$\n0.77"),
        ("coke_ratio", multiline_eps_coke, "CocaCola", "capital_ratio", "14.2", "Capital ratio    14.2"),
    ]
    per_share_scaled_reject = [
        ("r_aapl_basic_m", apple_txt, "Apple", "basic_eps", "$9.01 million", "Basic (in dollars per share)\n$ 9.01"),
        ("r_aapl_dil_k", apple_txt, "Apple", "diluted_eps", "$8.98 thousand", "Diluted (in dollars per share)\n$ 8.98"),
        ("r_meta_par_m", meta_txt, "Meta", "par_value", "$0.000006 million", "Common stock, par value (in dollars per share)\n$ 0.000006"),
        ("r_chv_basic_m", chevron_txt, "Chevron", "basic_eps", "$5.07 million", "Basic (in dollars per share)\n$ 5.07"),
        ("r_chv_dil_m", chevron_txt, "Chevron", "diluted_eps", "$5.06 million", "Diluted (in dollars per share)\n$ 5.06"),
        ("r_aapl_ml_b_m", multiline_eps_apple, "Apple", "basic_eps", "$9.01 million", "Basic (1)\n$\n9.01"),
        ("r_aapl_ml_d_m", multiline_eps_apple, "Apple", "diluted_eps", "$8.98 million", "Diluted (1)\n$\n8.98"),
        ("r_chv_ml_b_m", multiline_eps_chevron, "Chevron", "basic_eps", "$5.07 million", "Basic (a)   $ 5.07"),
        ("r_chv_ml_div_m", multiline_eps_chevron, "Chevron", "div_per_share", "$1.71 million", "Dividends declared per\n  common share (1)   $ 1.71"),
        ("r_coke_ml_b_m", multiline_eps_coke, "CocaCola", "basic_eps", "$0.78 million", "basic\n$\n0.78"),
        ("r_coke_ml_d_m", multiline_eps_coke, "CocaCola", "diluted_eps", "$0.77 million", "diluted\n$\n0.77"),
        ("r_coke_ratio_m", multiline_eps_coke, "CocaCola", "capital_ratio", "$14.2 million", "Capital ratio    14.2"),
    ]
    monetary_scaled_pass = [
        ("m_aapl_sales", apple_txt, "Apple", "net_sales", "$478,834 million", "Net sales\n$ 478,834"),
        ("m_aapl_gm", apple_txt, "Apple", "gross_margin", "$233,602 million", "Gross margin\n233,602"),
        ("m_goog_cash", alphabet_txt, "Alphabet", "cash", "$56,334 million", "Cash and cash equivalents\n$ 56,334"),
        ("m_goog_gw", alphabet_txt, "Alphabet", "goodwill", "$58,117 million", "Goodwill\n58,117"),
        ("m_jpm_ib", jpm_txt, "JPMorgan", "ib_fees", "$2,960 million", "Investment banking fees\n$ 2,960"),
        ("m_lly_rev", lilly_txt, "EliLilly", "revenue", "$24,397 million", "Revenue\n$ 24,397"),
        ("m_jnj_eq", jnj_txt, "JNJ", "equity", "$81,529 million", "Total shareholders’ equity\n81,529"),
        ("m_jnj_assets", jnj_txt, "JNJ", "assets", "$197,425 million", "Total liabilities and shareholders’ equity\n$ 197,425"),
        ("m_chv_ni", chevron_txt, "Chevron", "net_income", "$9,924 million", "Net Income (Loss) Attributable to Chevron Corporation\n$ 9,924"),
        ("m_ml_aapl_ni", multiline_eps_apple, "Apple", "net_income", "$93,736 million", "Net income   $ 93,736"),
        ("m_ml_chv_ni", multiline_eps_chevron, "Chevron", "net_income", "$9,924 million", "Net Income Attributable to Chevron Corporation   $ 9,924   $ 3,539"),
        ("m_ml_coke_rev", multiline_eps_coke, "CocaCola", "revenue", "$12,480 million", "Net Operating Revenues   $ 12,480"),
    ]

    ps_ok = sum(
        1
        for cid, txt, s, p, v, q in per_share_unscaled_pass
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status == "PASS"
    )
    ps_rej = sum(
        1
        for cid, txt, s, p, v, q in per_share_scaled_reject
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status == "RETRY"
    )
    mon_ok = sum(
        1
        for cid, txt, s, p, v, q in monetary_scaled_pass
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status == "PASS"
    )
    # Collect exact failure ids for diagnostics
    ps_fail = [
        (cid, verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status)
        for cid, txt, s, p, v, q in per_share_unscaled_pass
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status != "PASS"
    ]
    ps_false_accept = [
        (cid, verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status)
        for cid, txt, s, p, v, q in per_share_scaled_reject
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status != "RETRY"
    ]
    mon_fail = [
        (cid, verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status)
        for cid, txt, s, p, v, q in monetary_scaled_pass
        if verify([Assertion(s, p, v, cid, q)], Ledger([Source(cid, txt, authority_rank=100)])).status != "PASS"
    ]

    return {
        "per_share_unscaled_accepted": wilson_ci(ps_ok, len(per_share_unscaled_pass)),
        "per_share_scaled_rejected": wilson_ci(ps_rej, len(per_share_scaled_reject)),
        "monetary_scaled_accepted": wilson_ci(mon_ok, len(monetary_scaled_pass)),
        "ps_fail": ps_fail,
        "ps_false_accept": ps_false_accept,
        "mon_fail": mon_fail,
    }


def main() -> None:
    snap = json.loads((_BENCH_DIR / "metamorphic_edgar_snapshot.json").read_text())
    report = {
        "p0_p1": evaluate_p0_and_p1_metamorphic(snap),
        "p2": evaluate_p2_bullets_and_spacing(),
        "p3": evaluate_p3_disjoint_tables(snap),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
