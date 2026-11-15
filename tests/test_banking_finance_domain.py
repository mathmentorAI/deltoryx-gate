#!/usr/bin/env python3
"""Tests for decoupled DomainAdapter and banking-finance domain contract (Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HARNESS_DIR = _ROOT / "examples" / "harness2_securitization"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from claimlayer import Assertion  # noqa: E402
from claimlayer.v4.domains import (  # noqa: E402
    BANKING_FINANCE_VOCABULARY,
    resolve_banking_governing_role,
)
from harness2_v4_pipeline import (  # noqa: E402
    _build_corpus_objects,
    evaluate_domain_candidate,
    get_domain_adapter,
    run_securitization_repair_loop,
)


def _build_synthetic_banking_corpus():
    docs = [
        (
            "D1",
            "lender-markup-term-sheet.docx",
            "counterparty_markup",
            "standard",
            60,
            False,
            1,
            "Counterparty Markup: Lender proposes Applicable Margin of 5.25% per annum and Maximum Net Leverage of 4.0x.",
        ),
        (
            "D2",
            "negotiation-playbook.docx",
            "client_playbook_policy",
            "authoritative",
            95,
            True,
            10,
            "Client Negotiation Playbook: Maximum acceptable Applicable Margin is 4.75% per annum; target Maximum Net Leverage covenant is 4.5x.",
        ),
        (
            "D3",
            "original-term-sheet.docx",
            "baseline_term_sheet",
            "authoritative",
            90,
            True,
            20,
            "Baseline Term Sheet: Senior Secured Term Loan Facility Commitment is $200,000,000 at an Applicable Margin of 4.50% per annum.",
        ),
        (
            "D4",
            "partner-instructions-email.eml",
            "client_instructions_scope",
            "standard",
            70,
            False,
            30,
            "Partner Instructions: Focus review on Applicable Margin, leverage covenant headroom, and Term Loan Commitment.",
        ),
        (
            "D5",
            "ridgeline-financial-summary.xlsx",
            "quantitative_financial_model",
            "authoritative",
            100,
            True,
            40,
            "Financial Summary: Ridgeline Consolidated Adjusted EBITDA is $50,000,000 for LTM Q4.",
        ),
    ]
    corpus = _build_corpus_objects(
        case_id="banking_finance_synthetic",
        task_id="banking-finance/analyze-counterparty-markup",
        dataset_role="synthetic_unit_test",
        docs_cfg=docs,
        common_sha256="banking_synthetic_sha",
        common_chars=600,
        total_lines=50,
    )
    from claimlayer.v4.domains.banking_finance import BANKING_FINANCE_ACCEPTANCE_POLICY, BANKING_FINANCE_VOCABULARY
    from claimlayer.v4.engine import EvidenceLayer
    bank_layer = EvidenceLayer(vocabulary=BANKING_FINANCE_VOCABULARY, policy=BANKING_FINANCE_ACCEPTANCE_POLICY)
    fac_src = corpus.v4_sources["original-term-sheet.docx"]
    fac_q = "Senior Secured Term Loan Facility Commitment is $200,000,000"
    fac_s = fac_src.text.index(fac_q)
    bank_layer.ingest([fac_src])
    bank_layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Term Loan",
                    "predicate": "facility_amount",
                    "value": {"value": 200000000, "unit": "USD"},
                    "assertion_kind": "fact",
                    "condition_status": "unconditional",
                },
                "evidence": [
                    {
                        "document_id": fac_src.document_id,
                        "revision": fac_src.revision,
                        "start": fac_s,
                        "end": fac_s + len(fac_q),
                        "quote": fac_q,
                    }
                ],
            }
        ],
        attestor="corpus_reference_builder",
    )
    from dataclasses import replace
    return replace(corpus, evidence_layer=bank_layer)


def test_banking_context_authority_resolution_and_adapter_selection():
    assert resolve_banking_governing_role("financial_metrics") == ("quantitative_financial_model",)
    assert resolve_banking_governing_role("negotiation_policy") == ("client_playbook_policy",)
    assert resolve_banking_governing_role("baseline_terms") == ("baseline_term_sheet",)
    assert resolve_banking_governing_role("review_scope") == ("client_instructions_scope",)

    corpus = _build_synthetic_banking_corpus()
    adapter = get_domain_adapter(corpus)
    assert adapter.domain_id == "banking_finance"
    assert adapter.vocabulary.name == BANKING_FINANCE_VOCABULARY.name
    assert adapter.resolve_governing_doc_fn(corpus, "negotiation_policy").filename == "negotiation-playbook.docx"
    assert adapter.resolve_governing_doc_fn(corpus, "financial_metrics").filename == "ridgeline-financial-summary.xlsx"
    assert adapter.resolve_governing_doc_fn(corpus, "review_scope").filename == "partner-instructions-email.eml"


def test_banking_adapter_blocks_counterparty_markup_authority_inversion():
    corpus = _build_synthetic_banking_corpus()
    bad_candidate = {
        "report_text": "Lender markup governs over negotiation playbook; Applicable Margin is 5.25%.",
        "assertions": [
            {
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": "5.25%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": "lender-markup-term-sheet.docx",
                "quote": "Lender proposes Applicable Margin of 5.25% per annum",
            }
        ],
    }
    res = evaluate_domain_candidate(bad_candidate, corpus, gate_mode="v3_only", require_assertions=True)
    assert not res.accepted
    assert "AUTHORITY_INVERSION" in {i["code"] for i in res.issues}


def test_banking_adapter_accepts_context_grounded_v3_v4_and_repair_loop():
    corpus = _build_synthetic_banking_corpus()
    ebitda_src = corpus.v4_sources["ridgeline-financial-summary.xlsx"]
    ebitda_q = "Ridgeline Consolidated Adjusted EBITDA is $50,000,000 for LTM Q4."
    ebitda_s = ebitda_src.text.index(ebitda_q)
    fac_src = corpus.v4_sources["original-term-sheet.docx"]
    fac_q = "Senior Secured Term Loan Facility Commitment is $200,000,000"
    fac_s = fac_src.text.index(fac_q)
    valid_graph = {
        "claims": {
            "c_ebitda": {
                "claim_id": "c_ebitda",
                "kind": "extraction",
                "subject": "Ridgeline",
                "predicate": "ebitda",
                "value": 50000000,
                "unit": "USD",
                "evidence": [
                    {
                        "document_id": "ridgeline-financial-summary.xlsx",
                        "quote": ebitda_q,
                        "revision": ebitda_src.revision,
                        "start": ebitda_s,
                        "end": ebitda_s + len(ebitda_q),
                    }
                ],
            },
            "c_fac": {
                "claim_id": "c_fac",
                "kind": "extraction",
                "subject": "Term Loan",
                "predicate": "facility_amount",
                "value": 200000000,
                "unit": "USD",
                "evidence": [
                    {
                        "document_id": "original-term-sheet.docx",
                        "quote": fac_q,
                        "revision": fac_src.revision,
                        "start": fac_s,
                        "end": fac_s + len(fac_q),
                    }
                ],
            },
        },
        "steps": [],
        "conclusion_id": "c_fac",
    }

    bad_initial = {
        "report_text": "Lender markup governs over negotiation playbook; leverage is 4.5%.",
        "assertions": [
            {
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": "5.25%",
                "unit": "PERCENT",
                "role": "governing",
                "source_id": "lender-markup-term-sheet.docx",
                "quote": "Lender proposes Applicable Margin of 5.25% per annum",
            }
        ],
        "graph": valid_graph,
    }
    good_repaired = {
        "report_text": "According to the text, the Term Loan is 200000000 USD.",
        "assertions": [
            {
                "subject": "Term Loan",
                "predicate": "facility_amount",
                "value": "$200,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "original-term-sheet.docx",
                "quote": "Senior Secured Term Loan Facility Commitment is $200,000,000",
            },
            {
                "subject": "Ridgeline",
                "predicate": "ebitda",
                "value": "$50,000,000",
                "unit": "USD",
                "role": "governing",
                "source_id": "ridgeline-financial-summary.xlsx",
                "quote": ebitda_q,
            },
            {
                "subject": "Term Loan",
                "predicate": "interest_margin",
                "value": "5.25%",
                "unit": "PERCENT",
                "role": "subordinate_discrepancy",
                "source_id": "lender-markup-term-sheet.docx",
                "quote": "Lender proposes Applicable Margin of 5.25% per annum",
            },
            {
                "subject": "Partner Scope",
                "predicate": "review_scope",
                "value": "Applicable Margin, leverage covenant headroom, and Term Loan Commitment",
                "role": "governing",
                "source_id": "partner-instructions-email.eml",
                "quote": "Focus review on Applicable Margin, leverage covenant headroom, and Term Loan Commitment.",
            },
        ],
        "graph": valid_graph,
        "metadata": {"has_material_discrepancies": True},
    }

    loop_res = run_securitization_repair_loop(
        corpus=corpus,
        initial_candidate=bad_initial,
        repair_fn=lambda _cand, _fb, _it: good_repaired,
        gate_mode="v3_v4",
        max_iterations=2,
    )
    assert loop_res["accepted"] is True
    assert loop_res["final_disposition"] == "ACCEPT"
    assert loop_res["iterations_used"] == 2
    assert loop_res["operational_authorization"]["status"] == "PASS"
