"""Comprehensive adversarial, edge-case, and formalization-gap regression test suite.

Covers all findings from the external technical & AI Control audit:
  - A1–A13 + extraction traps (false PASS prevention in quote_quantity.py)
  - B1–B13 (false RETRY/ESCALATE prevention on real-world legal/financial formats)
  - C1–C11 (specification gaming, key aliasing, omission, duplicate IDs, MCP trust boundary)
  - D1–D6 (fail-closed section diffing, Roman/Spanish/bare headers, cross-references, renumbering)
  - E1–E5 (RFC 9162 Merkle domain separation, full verdict envelope binding, HMAC & ledger binding)
  - F1–F3, L1, N1, W1–W3 (multi-conflict triage, metrics honesty, Unicode keys, SQLite sync, policy sealing)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from claimlayer import (
    Assertion,
    FactRecord,
    Ledger,
    MCPVerificationServer,
    Policy,
    Source,
    build_comparison_policy,
    verify,
    verify_receipt,
)
from claimlayer.mcp_server import MCPSession
from claimlayer.models import normalize_entity_key, normalize_key
from claimlayer.receipt import compute_merkle_root
from examples.harness2_securitization.replay_om_vs_indenture import build_securitization_case
from examples.harness2_securitization.run_harness2_ab import stage_self_contained_harness


# =============================================================================
# 1. FALSE NEGATIVES FIXED (A1–A13 & EXTRACTION TRAPS)
# =============================================================================


def test_a1_to_a13_false_negatives_blocked() -> None:
    # A1: Minimal out-of-context quote ("30" taken from "Section 30")
    s_a1 = Source(id="s1", text="Section 30 sets general definitions for the deal.", tier="authoritative")
    v_a1 = verify(
        [Assertion(subject="Deal", predicate="term", value="30 days", source_id="s1", quote="30")],
        Ledger([s_a1]),
    )
    assert v_a1.status == "RETRY"

    # A2: Quote lacks the figure, even though "45 days" appears elsewhere in the source
    s_a2 = Source(
        id="s2",
        text="Payment terms are 45 days. All invoices must be submitted electronically.",
        tier="authoritative",
    )
    v_a2 = verify(
        [
            Assertion(
                subject="ACME",
                predicate="payment_terms",
                value="45 days",
                source_id="s2",
                quote="All invoices must be submitted electronically.",
            )
        ],
        Ledger([s_a2]),
    )
    assert v_a2.status == "RETRY"
    assert v_a2.issues[0].code == "QUANTITY_MISMATCH"

    # A3: Date appears elsewhere in source (closing date), not in the cited maturity quote
    s_a3 = Source(
        id="s3",
        text="Closing date is 2026-01-15. Maturity date is governed by Schedule B.",
        tier="authoritative",
    )
    v_a3 = verify(
        [
            Assertion(
                subject="Notes",
                predicate="maturity_date",
                value="2026-01-15",
                source_id="s3",
                quote="Maturity date is governed by Schedule B.",
            )
        ],
        Ledger([s_a3]),
    )
    assert v_a3.status == "RETRY"
    assert v_a3.issues[0].code == "DATE_MISMATCH"

    # A4: Truncated leading negation ("In no event shall the Servicer be terminated without cause")
    s_a4 = Source(
        id="s4",
        text="In no event shall the Servicer be terminated without cause.",
        tier="authoritative",
    )
    v_a4_trunc = verify(
        [
            Assertion(
                subject="Servicer",
                predicate="termination_without_cause",
                value="permitted",
                source_id="s4",
                quote="the Servicer be terminated without cause",
            )
        ],
        Ledger([s_a4]),
    )
    assert v_a4_trunc.status == "RETRY"

    v_a4_full = verify(
        [
            Assertion(
                subject="Servicer",
                predicate="termination_without_cause",
                value="permitted",
                source_id="s4",
                quote="In no event shall the Servicer be terminated without cause.",
            )
        ],
        Ledger([s_a4]),
    )
    assert v_a4_full.status == "RETRY"

    # A5 & A6: Compound word numbers ("twenty-five (25) days" != 5 days; "one hundred twenty (120) days" != 20 days)
    s_a5 = Source(
        id="s5",
        text="Cure period is twenty-five (25) days. Survival period is one hundred twenty (120) days.",
        tier="authoritative",
    )
    v_a5 = verify(
        [
            Assertion(
                subject="Deal",
                predicate="cure_period",
                value="5 days",
                source_id="s5",
                quote="Cure period is twenty-five (25) days.",
            )
        ],
        Ledger([s_a5]),
    )
    assert v_a5.status == "RETRY"
    assert v_a5.issues[0].code == "QUANTITY_MISMATCH"

    v_a6 = verify(
        [
            Assertion(
                subject="Deal",
                predicate="survival",
                value="20 days",
                source_id="s5",
                quote="Survival period is one hundred twenty (120) days.",
            )
        ],
        Ledger([s_a5]),
    )
    assert v_a6.status == "RETRY"
    assert v_a6.issues[0].code == "QUANTITY_MISMATCH"

    # A7: "5 calendar days" must NOT pass against "5 business days"
    s_a7 = Source(id="s7", text="Notice must be given within 5 business days.", tier="authoritative")
    v_a7 = verify(
        [
            Assertion(
                subject="Deal",
                predicate="notice",
                value="5 calendar days",
                source_id="s7",
                quote="Notice must be given within 5 business days.",
            )
        ],
        Ledger([s_a7]),
    )
    assert v_a7.status == "RETRY"
    assert v_a7.issues[0].code == "UNIT_MISMATCH"

    # A8: Cross-attribution in multi-entity quote (Class A = 5.10%, Class C = 6.75%; claiming Class C = 5.10%)
    s_a8 = Source(
        id="s8",
        text="The Class A Notes bear interest at 5.10% per annum, and Class C Notes bear interest at 6.75% per annum.",
        tier="authoritative",
    )
    v_a8 = verify(
        [
            Assertion(
                subject="Class C",
                predicate="interest_rate",
                value="5.10%",
                source_id="s8",
                quote="The Class A Notes bear interest at 5.10% per annum, and Class C Notes bear interest at 6.75% per annum.",
            )
        ],
        Ledger([s_a8]),
    )
    assert v_a8.status == "RETRY"
    assert v_a8.issues[0].code == "QUANTITY_MISMATCH"

    # A9: Exact Decimal equality ($50,000,000 must NOT match $50,000,049)
    s_a9 = Source(id="s9", text="Principal amount is $50,000,049.", tier="authoritative")
    v_a9 = verify(
        [
            Assertion(
                subject="Notes",
                predicate="principal",
                value="$50,000,000",
                source_id="s9",
                quote="Principal amount is $50,000,049.",
            )
        ],
        Ledger([s_a9]),
    )
    assert v_a9.status == "RETRY"
    assert v_a9.issues[0].code == "QUANTITY_MISMATCH"

    # A10: "5%" must NOT match "5,5% anual"
    s_a10 = Source(id="s10", text="El interés aplicable es del 5,5% anual.", tier="authoritative")
    v_a10 = verify(
        [
            Assertion(
                subject="Prestamo",
                predicate="interes",
                value="5%",
                source_id="s10",
                quote="El interés aplicable es del 5,5% anual.",
            )
        ],
        Ledger([s_a10]),
    )
    assert v_a10.status == "RETRY"
    assert v_a10.issues[0].code == "QUANTITY_MISMATCH"

    # A11: Dimensionless "30" must NOT match "30%"
    s_a11 = Source(id="s11", text="Concentration limit shall not exceed 30%.", tier="authoritative")
    v_a11 = verify(
        [
            Assertion(
                subject="Pool",
                predicate="limit",
                value="30",
                source_id="s11",
                quote="Concentration limit shall not exceed 30%.",
            )
        ],
        Ledger([s_a11]),
    )
    assert v_a11.status == "RETRY"
    assert v_a11.issues[0].code == "UNIT_MISMATCH"

    # A12: Sign preservation ("-5%" must NOT pass as "5%")
    s_a12 = Source(id="s12", text="Operating margin was -5% for the period.", tier="authoritative")
    v_a12 = verify(
        [
            Assertion(
                subject="Co",
                predicate="margin",
                value="5%",
                source_id="s12",
                quote="Operating margin was -5% for the period.",
            )
        ],
        Ledger([s_a12]),
    )
    assert v_a12.status == "RETRY"
    assert v_a12.issues[0].code == "QUANTITY_MISMATCH"

    # A13: Table scale header ("in thousands": $125 means $125,000, not $125)
    s_a13 = Source(
        id="s13",
        text="Table 1 (in thousands): Annual servicing fee is $125.",
        tier="authoritative",
    )
    v_a13_unscaled = verify(
        [
            Assertion(
                subject="Servicer",
                predicate="fee",
                value="$125",
                source_id="s13",
                quote="Table 1 (in thousands): Annual servicing fee is $125.",
            )
        ],
        Ledger([s_a13]),
    )
    assert v_a13_unscaled.status == "RETRY"

    v_a13_scaled = verify(
        [
            Assertion(
                subject="Servicer",
                predicate="fee",
                value="$125,000",
                source_id="s13",
                quote="Table 1 (in thousands): Annual servicing fee is $125.",
            )
        ],
        Ledger([s_a13]),
    )
    assert v_a13_scaled.status == "PASS"

    # Extra extraction traps: "3M EURIBOR", "Class 2 B Notes", "5 mil millones"
    s_extra = Source(
        id="s_extra",
        text=(
            "Class 2 B Notes bear interest at 3M EURIBOR plus 200 bps. "
            "El límite total es de 5 mil millones de euros."
        ),
        tier="authoritative",
    )
    v_euribor_trap = verify(
        [
            Assertion(
                subject="Notes",
                predicate="rate",
                value="3,000,000 EUR",
                source_id="s_extra",
                quote="Class 2 B Notes bear interest at 3M EURIBOR plus 200 bps.",
            )
        ],
        Ledger([s_extra]),
    )
    assert v_euribor_trap.status == "RETRY"

    v_mil_millones_trap = verify(
        [
            Assertion(
                subject="Deal",
                predicate="limit",
                value="5,000 EUR",
                source_id="s_extra",
                quote="El límite total es de 5 mil millones de euros.",
            )
        ],
        Ledger([s_extra]),
    )
    assert v_mil_millones_trap.status == "RETRY"

    v_mil_millones_ok = verify(
        [
            Assertion(
                subject="Deal",
                predicate="limit",
                value="5,000,000,000 EUR",
                source_id="s_extra",
                quote="El límite total es de 5 mil millones de euros.",
            )
        ],
        Ledger([s_extra]),
    )
    assert v_mil_millones_ok.status == "PASS"


# =============================================================================
# 2. FALSE POSITIVES FIXED (B1–B13)
# =============================================================================


def test_b1_to_b13_false_positives_pass() -> None:
    s = Source(
        id="doc",
        tier="authoritative",
        text=(
            "Interest is 5 per cent. per annum (or 5 por ciento). "
            "Facility limit is EUR 500,000,000. "
            "Maturity is January 15, 2026. "
            "Cure notice is between 30 and 60 days. "
            "The ﬁnancial obli-\ngation is non\u2011recourse\u00ad. "
            "El precio es de 1.250.000 euros. "
            "Spread is 150 basis points. "
            "Supermajority requires 66 2/3% of holders. "
            "Grace period is two (2) weeks. "
            "If an interest default occurs, cure period is 5 business days. "
            "When a covenant default occurs, cure period is 30 days."
        ),
    )
    ledger = Ledger([s])

    # B1–B9
    assertions = [
        Assertion(subject="X", predicate="b1_en", value="5%", source_id="doc", quote="Interest is 5 per cent. per annum"),
        Assertion(subject="X", predicate="b1_es", value="5%", source_id="doc", quote="Interest is 5 per cent. per annum (or 5 por ciento)."),
        Assertion(subject="X", predicate="b2", value="500,000,000 EUR", source_id="doc", quote="Facility limit is EUR 500,000,000."),
        Assertion(subject="X", predicate="b3", value="2026-01-15", source_id="doc", quote="Maturity is January 15, 2026."),
        Assertion(subject="X", predicate="b4", value="30 days", source_id="doc", quote="Cure notice is between 30 and 60 days."),
        Assertion(subject="X", predicate="b5", value="non-recourse", source_id="doc", quote="The financial obligation is non-recourse."),
        Assertion(subject="X", predicate="b6", value="1250000 EUR", source_id="doc", quote="El precio es de 1.250.000 euros."),
        Assertion(subject="X", predicate="b7", value="150 bps", source_id="doc", quote="Spread is 150 basis points."),
        Assertion(subject="X", predicate="b8", value="66.67%", source_id="doc", quote="Supermajority requires 66 2/3% of holders."),
        Assertion(subject="X", predicate="b9", value="14 days", source_id="doc", quote="Grace period is two (2) weeks."),
        # B11: Two conditional cure periods in the same source
        Assertion(subject="X", predicate="cure_period", value="5 business days", source_id="doc", quote="If an interest default occurs, cure period is 5 business days."),
        Assertion(subject="X", predicate="cure_period", value="30 days", source_id="doc", quote="When a covenant default occurs, cure period is 30 days."),
    ]
    v = verify(assertions, ledger)
    assert v.status == "PASS", f"Unexpected issues: {v.issues}"

    # B12 & B13: Peer sources with equivalent jurisdictional strings ("the laws of the State of New York" == "New York")
    # and commensurable units ("1.50%" == "150 bps")
    s_peer1 = Source(
        id="p1",
        tier="authoritative",
        text="Governing law is the laws of the State of New York. Fee is 1.50%.",
    )
    s_peer2 = Source(
        id="p2",
        tier="authoritative",
        text="Governing law is New York. Fee is 150 bps.",
    )
    peer_ledger = Ledger(
        sources=[s_peer1, s_peer2],
        facts=[
            FactRecord(subject="Deal", predicate="governing_law", value="New York", source_id="p2", quote="Governing law is New York."),
            FactRecord(subject="Deal", predicate="fee", value="150 bps", source_id="p2", quote="Fee is 150 bps."),
        ],
    )
    v_peer = verify(
        [
            Assertion(
                subject="Deal",
                predicate="governing_law",
                value="the laws of the State of New York",
                source_id="p1",
                quote="Governing law is the laws of the State of New York.",
            ),
            Assertion(
                subject="Deal",
                predicate="fee",
                value="1.50%",
                source_id="p1",
                quote="Fee is 1.50%.",
            ),
        ],
        peer_ledger,
    )
    assert v_peer.status == "PASS", f"Unexpected peer conflict: {v_peer.issues}"


# =============================================================================
# 3. SPECIFICATION GAMING & TRUST BOUNDARY VECTORS (C1–C11)
# =============================================================================


def test_c1_to_c11_specification_gaming_blocked() -> None:
    # C1a, C1b (payment_term), C1c (ACME Corp): Key aliasing on superseded source
    msa = Source(id="msa", text="ACME shall pay invoices within 45 days.", tier="standard")
    amend = Source(
        id="amend",
        text="Section 4 is amended: ACME shall pay invoices within 30 days.",
        tier="authoritative",
        supersedes=["msa"],
    )
    l_c1 = Ledger(
        sources=[msa, amend],
        facts=[
            FactRecord(
                subject="ACME",
                predicate="payment_terms",
                value="30 days",
                source_id="amend",
                quote="ACME shall pay invoices within 30 days.",
            )
        ],
    )
    for subj, pred in [("ACME", "payment_terms"), ("ACME", "payment_term"), ("ACME Corp", "payment_terms")]:
        v_c1 = verify(
            [Assertion(subject=subj, predicate=pred, value="45 days", source_id="msa", quote="ACME shall pay invoices within 45 days.")],
            l_c1,
        )
        assert v_c1.status == "RETRY", f"Failed on ({subj}, {pred})"
        assert v_c1.issues[0].code == "SOURCE_SUPERSEDED"

    # C2: Omission of conflicting peer authoritative source when NO FactRecords are preloaded
    s_hi1 = Source(id="hi1", text="Class C Note interest rate shall be 6.75% per annum.", tier="authoritative")
    s_hi2 = Source(id="hi2", text="Class C Note interest rate shall be 6.50% per annum.", tier="authoritative")
    l_c2 = Ledger(sources=[s_hi1, s_hi2])  # Zero FactRecords!
    v_c2 = verify(
        [
            Assertion(
                subject="Class_C_Notes",
                predicate="interest_rate",
                value="6.75%",
                source_id="hi1",
                quote="Class C Note interest rate shall be 6.75% per annum.",
            )
        ],
        l_c2,
    )
    assert v_c2.status == "ESCALATE"
    assert v_c2.issues[0].code == "UNRESOLVED_PEER_CONFLICT"

    # C3a & C3b: Two conflicting assertions in same batch, both given id="x"
    v_c3b = verify(
        [
            Assertion(
                id="x",
                subject="Class_C_Notes",
                predicate="interest_rate",
                value="6.75%",
                source_id="hi1",
                quote="Class C Note interest rate shall be 6.75% per annum.",
            ),
            Assertion(
                id="x",
                subject="Class_C_Notes",
                predicate="interest_rate",
                value="6.50%",
                source_id="hi2",
                quote="Class C Note interest rate shall be 6.50% per annum.",
            ),
        ],
        Ledger(sources=[s_hi1, s_hi2]),
    )
    assert v_c3b.status == "ESCALATE"

    # C4: Vacuous coverage ("reviewed") on required_predicates
    v_c4 = verify(
        [
            Assertion(
                subject="Class_C_Notes",
                predicate="section_2_04",
                value="reviewed",
                source_id="hi1",
                quote="Class C Note interest rate shall be 6.75% per annum.",
            )
        ],
        Ledger(sources=[s_hi1]),
        Policy(required_predicates=["section_2_04"]),
    )
    assert v_c4.status == "RETRY"

    # C5a, C5b, C5c: MCP trust boundary
    mcp_ledger = Ledger(sources=[s_hi1])
    mcp = MCPVerificationServer(mcp_ledger)
    sess = MCPSession(initialize_sent=True, initialized=True)

    # C5a: Agent passes policy with require_exact_quote=False
    resp_c5a = mcp.handle_envelope(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "verify_proposal",
                "arguments": {
                    "policy": {"require_exact_quote": False},
                    "assertions": [
                        {
                            "subject": "Class_C_Notes",
                            "predicate": "interest_rate",
                            "value": "99%",
                            "source_id": "hi1",
                            "quote": "invented quote",
                        }
                    ],
                },
            },
        },
        sess,
    )
    assert "error" in resp_c5a[0]

    # C5b: Agent attempts to register a fake authoritative source over an already-populated authoritative ledger
    resp_c5b = mcp.handle_envelope(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "register_source",
                "arguments": {
                    "id": "amendment_fake",
                    "text": "Class C Note interest rate shall be 0.00%.",
                    "tier": "authoritative",
                },
            },
        },
        sess,
    )
    assert "error" in resp_c5b[0]

    # C5c: Agent attempts to overwrite existing source 'hi1'
    resp_c5c = mcp.handle_envelope(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "register_source",
                "arguments": {
                    "id": "hi1",
                    "text": "Modified text.",
                    "tier": "standard",
                },
            },
        },
        sess,
    )
    assert "error" in resp_c5c[0]

    # C6: Confidence validation (NaN, 7.0, "high") and source reliability cap
    for bad_conf in [float("nan"), float("inf"), 7.0, -0.5, "high", True]:
        with pytest.raises(ValueError):
            Assertion(
                subject="X",
                predicate="y",
                value="10%",
                source_id="hi1",
                quote="quote",
                confidence=bad_conf,  # type: ignore[arg-type]
            )

    low_rel_src = Source(
        id="low_rel",
        text="Estimated reserve is 10%.",
        tier="standard",
        metadata={"reliability": 0.6},
    )
    v_c6_cap = verify(
        [
            Assertion(
                subject="Deal",
                predicate="reserve",
                value="10%",
                source_id="low_rel",
                quote="Estimated reserve is 10%.",
                confidence=1.0,  # Agent tries to inflate confidence above source reliability (0.6)
            )
        ],
        Ledger([low_rel_src]),
        Policy(min_support=0.70),
    )
    assert v_c6_cap.status == "ESCALATE"
    assert v_c6_cap.issues[0].code == "LOW_EVIDENCE_SUPPORT"

    # C7b & C7c: Turn 2 of securitization replay WITHOUT preloaded FactRecords (C7b)
    # and with renamed subject "Class_C_Notes" (C7c)
    full_ledger, indenture, om = build_securitization_case()
    factless_ledger = Ledger(sources=[indenture, om])
    pol_sec, _ = build_comparison_policy(indenture, om, min_tier_rank=50)

    for test_ledger, subj in [(factless_ledger, "Deal_Waterfall"), (full_ledger, "Class_C_Notes")]:
        v_c7 = verify(
            [
                Assertion(
                    subject=subj,
                    predicate="section_2_04",
                    value="6.50%",
                    source_id="draft_offering_memorandum",
                    quote="Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum.",
                ),
                Assertion(
                    subject="Deal_Waterfall",
                    predicate="section_5_04",
                    value="Step 13",
                    source_id="executed_indenture_2026",
                    quote="Step 13 Reserve Account Replenishment up to the Required Reserve Amount (junior to Class B and Class C principal).",
                ),
                Assertion(
                    subject="Deal_Waterfall",
                    predicate="section_7_01",
                    value="5 business days",
                    source_id="executed_indenture_2026",
                    quote="Failure to pay interest continuing for 5 business days constitutes an Event of Default.",
                ),
                Assertion(
                    subject="Deal_Waterfall",
                    predicate="section_9_01",
                    value="10%",
                    source_id="executed_indenture_2026",
                    quote="Servicer may redeem the Notes when Pool Balance falls to 10% of Initial Pool Balance.",
                ),
            ],
            test_ledger,
            pol_sec,
        )
        assert v_c7.status == "RETRY"
        assert any(i.code == "HIGHER_TIER_OVERRIDE" for i in v_c7.issues)

    # C8 & C8b: Human resolution must NOT override a newer superseding amendment or an expired source
    s_exp = Source(
        id="indenture_old",
        text="Class C rate is 6.75% per annum.",
        tier="authoritative",
        valid_until="2020-12-31",
    )
    l_c8 = Ledger(sources=[s_exp])
    l_c8.resolve_escalation(
        subject="Class_C",
        predicate="rate",
        selected_value="6.75%",
        resolver="counsel",
    )
    # C8b: Source expired in 2020 -> RETRY (SOURCE_EXPIRED) despite human resolution
    v_c8b = verify(
        [Assertion(subject="Class_C", predicate="rate", value="6.75%", source_id="indenture_old", quote="Class C rate is 6.75% per annum.")],
        l_c8,
        as_of="2026-10-30",
    )
    assert v_c8b.status == "RETRY"
    assert v_c8b.issues[0].code == "SOURCE_EXPIRED"

    # C8: Active source superseded by a later amendment to 7.00%
    s_active = Source(id="ind_2025", text="Class C rate is 6.75% per annum.", tier="authoritative")
    s_amend_2026 = Source(
        id="amend_2026",
        text="Class C rate is amended to 7.00% per annum.",
        tier="authoritative",
        supersedes=["ind_2025"],
    )
    l_c8_amend = Ledger(
        sources=[s_active, s_amend_2026],
        facts=[FactRecord(subject="Class_C", predicate="rate", value="7.00%", source_id="amend_2026", quote="Class C rate is amended to 7.00% per annum.")],
    )
    l_c8_amend.resolve_escalation(subject="Class_C", predicate="rate", selected_value="6.75%", resolver="counsel")
    v_c8_old = verify(
        [Assertion(subject="Class_C", predicate="rate", value="6.75%", source_id="ind_2025", quote="Class C rate is 6.75% per annum.")],
        l_c8_amend,
    )
    assert v_c8_old.status == "RETRY"
    assert v_c8_old.issues[0].code == "SOURCE_SUPERSEDED"

    v_c8_new = verify(
        [Assertion(subject="Class_C", predicate="rate", value="7.00%", source_id="amend_2026", quote="Class C rate is amended to 7.00% per annum.")],
        l_c8_amend,
    )
    assert v_c8_new.status == "PASS"

    # C10: Default policy rejects tier="draft" (rank=10 < min_tier_rank=20)
    s_draft = Source(id="email", text="Let us offer a 25% discount.", tier="draft")
    v_c10 = verify(
        [Assertion(subject="Deal", predicate="discount", value="25%", source_id="email", quote="Let us offer a 25% discount.")],
        Ledger([s_draft]),
    )
    assert v_c10.status == "RETRY"
    assert v_c10.issues[0].code == "INSUFFICIENT_AUTHORITY_TIER"

    # C11a & C11b: Numeric bound enforced even when predicate is renamed from refund_amount to refund_total
    s_ref = Source(id="ref_doc", text="Approved customer refund is $9,000.", tier="authoritative")
    pol_ref = Policy(numeric_bounds={"refund_amount": (None, 5_000.0)})
    for pred_name in ("refund_amount", "refund_total"):
        v_c11 = verify(
            [Assertion(subject="Order_1", predicate=pred_name, value="$9,000", source_id="ref_doc", quote="Approved customer refund is $9,000.")],
            Ledger([s_ref]),
            pol_ref,
        )
        assert v_c11.status == "ESCALATE"
        assert v_c11.issues[0].code == "POLICY_THRESHOLD_EXCEEDED"


# =============================================================================
# 4. SECTION DIFFING & COMPARISON POLICY (D1–D6)
# =============================================================================


def test_d1_to_d6_diff_verifiers() -> None:
    # D1 & D6: Roman numerals ("ARTICLE IV"), bare numbers ("4.01 Events of Default"), Spanish ("Cláusula 4"), and indentation
    p_doc = Source(
        id="prim",
        tier="authoritative",
        text=(
            "  ARTICLE IV: Covenants. Leverage ratio shall not exceed 3.5x.\n"
            "4.01 Events of Default. Cure period is 5 business days.\n"
            "Cláusula 4: Jurisdicción. El plazo es de 10 días.\n"
        ),
    )
    s_doc = Source(
        id="sec",
        tier="standard",
        text=(
            "  ARTICLE IV: Covenants. Leverage ratio shall not exceed 4.5x.\n"
            "4.01 Events of Default. Cure period is 30 days.\n"
            "Cláusula 4: Jurisdicción. El plazo es de 20 días.\n"
            "Section 9.99: Extra Provision. Bonus fee is $50,000.\n"
        ),
    )
    pol, divs = build_comparison_policy(p_doc, s_doc)
    div_keys = {d.section_key for d in divs}
    # D1, D6, and D2 (Section 9.99 added only in secondary!)
    assert "article_iv" in div_keys
    assert "section_4_01" in div_keys
    assert "cláusula_4" in div_keys
    assert "section_9_99" in div_keys
    assert len(pol.required_predicates) == 4

    # D1 fail-closed on unsegmented documents
    u1 = Source(id="u1", text="The interest rate is 6.75% per annum.", tier="authoritative")
    u2 = Source(id="u2", text="The interest rate is 6.50% per annum.", tier="standard")
    pol_u, divs_u = build_comparison_policy(u1, u2)
    assert pol_u.required_predicates == ("document_body",)
    assert len(divs_u) == 1

    # D3: Cross-reference "Section 7.01 above" at line start must NOT overwrite real Section 7.01
    p_d3 = Source(
        id="p_d3",
        tier="authoritative",
        text=(
            "Section 7.01: Events of Default. Failure to pay interest for 5 business days is a default.\n"
            "Section 8.01: Remedies. Upon an event under\n"
            "Section 7.01 above, the Trustee may accelerate the Notes.\n"
        ),
    )
    s_d3 = Source(
        id="s_d3",
        tier="standard",
        text=(
            "Section 7.01: Events of Default. Failure to pay interest for 30 days is a default.\n"
            "Section 8.01: Remedies. Upon an event under\n"
            "Section 7.01 above, the Trustee may accelerate the Notes.\n"
        ),
    )
    _, divs_d3 = build_comparison_policy(p_d3, s_d3)
    assert [d.section_key for d in divs_d3] == ["section_7_01"]

    # D4: Renumbered section with identical title and content must NOT be flagged as missing
    p_d4 = Source(
        id="p_d4",
        tier="authoritative",
        text="Section 4.01: Governing Law. This Agreement is governed by New York law.",
    )
    s_d4 = Source(
        id="s_d4",
        tier="standard",
        text="Section 5.01: Governing Law. This Agreement is governed by New York law.",
    )
    _, divs_d4 = build_comparison_policy(p_d4, s_d4)
    assert divs_d4 == []


# =============================================================================
# 5. CRYPTOGRAPHIC RECEIPT & MERKLE INTEGRITY (E1–E5)
# =============================================================================


def test_e1_to_e5_receipt_and_merkle_integrity() -> None:
    s = Source(id="s1", text="Payment terms are 30 days.", tier="authoritative")
    ledger = Ledger([s], signing_key="operator-hmac-secret-2026")
    pol = Policy(policy_id="strict_pol")

    # E1: A RETRY verdict rewritten as PASS cannot validate
    v_retry = ledger.verify(
        [Assertion(subject="ACME", predicate="terms", value="99 days", source_id="s1", quote="Payment terms are 30 days.")],
        pol,
    )
    assert v_retry.status == "RETRY"
    forged = v_retry.to_dict()
    forged["status"] = "PASS"
    forged["allowed_to_proceed"] = True
    forged["issues"] = []
    assert verify_receipt(forged) is False

    # Valid PASS verdict with HMAC key
    v_pass = ledger.verify(
        [Assertion(subject="ACME", predicate="terms", value="30 days", source_id="s1", quote="Payment terms are 30 days.")],
        pol,
    )
    assert v_pass.status == "PASS"
    assert v_pass.receipt.hmac_authenticated is True
    assert verify_receipt(v_pass, ledger=ledger, policy=pol, signing_key="operator-hmac-secret-2026") is True
    assert verify_receipt(v_pass, signing_key="wrong-key") is False

    # E2: Tampering with allowed_to_proceed or agent_id invalidates receipt immediately
    e2_dict = v_retry.to_dict()
    e2_dict["allowed_to_proceed"] = True
    assert verify_receipt(e2_dict, signing_key="operator-hmac-secret-2026") is False

    e2_agent = v_pass.to_dict()
    e2_agent["agent_id"] = "impersonated_agent"
    assert verify_receipt(e2_agent, signing_key="operator-hmac-secret-2026") is False

    # E3 & E5: Mutating policy, sources, facts, or resolutions invalidates ledger-bound receipt verification
    assert verify_receipt(v_pass, ledger=ledger, policy=Policy(policy_id="different_pol")) is False
    ledger.add_fact(FactRecord(subject="ACME", predicate="terms", value="30 days", source_id="s1", quote="Payment terms are 30 days."))
    assert verify_receipt(v_pass, ledger=ledger, policy=pol) is False

    # E4: CVE-2012-2459 odd-leaf duplication check: [a, b, c] != [a, b, c, c]
    assert compute_merkle_root(["aa", "bb", "cc"]) != compute_merkle_root(["aa", "bb", "cc", "cc"])


# =============================================================================
# 6. LEDGER, CONCURRENCY, NOISY-OR & HARNESS GATE (F1–F3, L1, N1, W1–W3)
# =============================================================================


def test_f1_f2_f3_l1_n1_and_w1_to_w3(tmp_path: Path) -> None:
    # F1: Resolving 1 of 2 conflicts in a single verdict keeps verdict OPEN until the 2nd is also resolved
    s1 = Source(id="s1", text="Rate is 6.75%. Cap is $10M.", tier="authoritative")
    s2 = Source(id="s2", text="Rate is 6.50%. Cap is $5M.", tier="authoritative")
    l_f1 = Ledger(
        sources=[s1, s2],
        facts=[
            FactRecord(subject="Deal", predicate="rate", value="6.50%", source_id="s2", quote="Rate is 6.50%."),
            FactRecord(subject="Deal", predicate="cap", value="5000000 USD", source_id="s2", quote="Cap is $5M."),
        ],
    )
    v_multi = l_f1.verify(
        [
            Assertion(subject="Deal", predicate="rate", value="6.75%", source_id="s1", quote="Rate is 6.75%."),
            Assertion(subject="Deal", predicate="cap", value="$10M", source_id="s1", quote="Cap is $10M."),
        ],
        record=True,
    )
    assert v_multi.status == "ESCALATE"
    assert l_f1.get_metrics()["open_escalations"] == 1

    # Resolve only the first conflict ('rate') -> verdict must remain OPEN!
    l_f1.resolve_escalation(
        verdict_id=v_multi.verdict_id,
        subject="Deal",
        predicate="rate",
        selected_value="6.75%",
    )
    assert l_f1.get_metrics()["open_escalations"] == 1

    # Resolve the second conflict ('cap') -> now verdict transitions to RESOLVED!
    l_f1.resolve_escalation(
        verdict_id=v_multi.verdict_id,
        subject="Deal",
        predicate="cap",
        selected_value="$10M",
    )
    assert l_f1.get_metrics()["open_escalations"] == 0

    # F2: 9 failed RETRYs + 1 ESCALATE (with 0 PASS) must report 0.0% human_supervision_saved_pct
    l_f2 = Ledger(sources=[s1, s2], facts=[FactRecord(subject="Deal", predicate="rate", value="6.50%", source_id="s2", quote="Rate is 6.50%.")])
    for _ in range(9):
        l_f2.verify(
            [Assertion(subject="Deal", predicate="rate", value="99%", source_id="s1", quote="Rate is 6.75%.")],
            record=True,
        )
    l_f2.verify(
        [Assertion(subject="Deal", predicate="rate", value="6.75%", source_id="s1", quote="Rate is 6.75%.")],
        record=True,
    )
    assert l_f2.get_metrics()["human_supervision_saved_pct"] == 0.0

    # F3: Unicode keys (Cyrillic, CJK) and entity article normalization ("the Servicer" == "Servicer")
    assert normalize_key("Газпром") == "газпром"
    assert normalize_key("トヨタ") == "トヨタ"
    assert normalize_entity_key("the Servicer") == normalize_entity_key("Servicer")
    assert normalize_entity_key("ACME Corp.") == normalize_entity_key("ACME")

    # N1: Two copies of the same document (identical sha256) with reliability=0.6 each do NOT sum to 0.84
    dup1 = Source(id="dup1", text="Reserve target is 10%.", tier="standard", metadata={"reliability": 0.6})
    dup2 = Source(id="dup2", text="Reserve target is 10%.", tier="standard", metadata={"reliability": 0.6})
    l_n1 = Ledger(
        sources=[dup1, dup2],
        facts=[FactRecord(subject="Deal", predicate="reserve", value="10%", source_id="dup2", quote="Reserve target is 10%.")],
    )
    v_n1 = l_n1.verify(
        [Assertion(subject="Deal", predicate="reserve", value="10%", source_id="dup1", quote="Reserve target is 10%.")],
        Policy(min_support=0.70),
    )
    assert v_n1.status == "ESCALATE"
    assert v_n1.confidence == 0.6

    # L1: Two Ledger instances on the same SQLite file see each other's writes
    db_file = tmp_path / "shared.sqlite"
    l_writer = Ledger(db_path=db_file)
    l_reader = Ledger(db_path=db_file)
    l_writer.add_source(Source(id="shared_src", text="Limit is 10 days.", tier="authoritative"))
    assert l_reader.get_source("shared_src") is not None

    # W1, W2, W3: Harness gate tier classification and .claimlayer/policy.json tamper detection
    staged = stage_self_contained_harness(tmp_path / "harness")
    sandbox = tmp_path / "sandbox"
    docs_dir = sandbox / "documents"
    docs_dir.mkdir(parents=True)
    import shutil

    shutil.copytree(staged / "scripts", sandbox / "scripts")
    (docs_dir / "executed_indenture.txt").write_text(
        "Section 2.04: Class C Notes bear interest at 6.75% per annum.\n", encoding="utf-8"
    )
    (docs_dir / "draft_offering-memorandum.txt").write_text(
        "Section 2.04: Class C Notes bear interest at 6.50% per annum.\n", encoding="utf-8"
    )
    (docs_dir / "side_letter_unsigned_v0.txt").write_text(
        "Section 2.04: Informal side letter draft.\n", encoding="utf-8"
    )

    prep = subprocess.run(
        [sys.executable, "scripts/claimlayer_gate.py", "prepare", "documents"],
        cwd=sandbox,
        capture_output=True,
        text=True,
        check=True,
    )
    prep_data = json.loads(prep.stdout)
    tiers = {s["id"]: s["tier"] for s in prep_data["sources_registered"]}
    # W1: side_letter_unsigned_v0.txt must be classified as 'draft', NOT 'authoritative'
    assert tiers["side_letter_unsigned_v0.txt"] == "draft"

    # W2: Proposal with fabricated quote returns exit code 2 (RETRY)
    prop_file = sandbox / "proposal.json"
    prop_file.write_text(
        json.dumps(
            {
                "assertions": [
                    {
                        "subject": "Class_C_Notes",
                        "predicate": "section_2_04",
                        "value": "6.75%",
                        "source_id": "executed_indenture.txt",
                        "quote": "Fabricated quote not in document.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    res_w2 = subprocess.run(
        [sys.executable, "scripts/claimlayer_gate.py", "verify", "proposal.json"],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert res_w2.returncode == 2

    # W3: Agent edits .claimlayer/policy.json to disable require_exact_quote -> still blocked (exit code 2, POLICY_TAMPERED)
    pol_path = sandbox / ".claimlayer" / "policy.json"
    tampered_pol = json.loads(pol_path.read_text(encoding="utf-8"))
    tampered_pol["require_exact_quote"] = False
    pol_path.write_text(json.dumps(tampered_pol), encoding="utf-8")

    res_w3 = subprocess.run(
        [sys.executable, "scripts/claimlayer_gate.py", "verify", "proposal.json"],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert res_w3.returncode == 2
    w3_out = json.loads(res_w3.stdout)
    assert w3_out["status"] == "RETRY"
    assert w3_out["issues"][0]["code"] == "POLICY_TAMPERED"
