"""Regression tests for the architectural & security audit remediation."""
from __future__ import annotations

from decimal import Decimal
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claimlayer import Assertion, FactRecord, Ledger, Policy, Source, verify
from claimlayer.mcp_server import MCPSession, MCPVerificationServer
from claimlayer.v4 import (
    AcceptancePolicy,
    B2B_CONTRACTS_VOCABULARY,
    B2B_DEFAULT_ACCEPTANCE_POLICY,
    ClauseRelation,
    EvidenceLayer,
    EvidenceRef,
    FieldEvidence,
    FieldSpec,
    LLMProposalReader,
    PredicateSpec,
    STANDARD_UNITS,
    SourceDocument,
    Unit,
    Vocabulary,
    authorize_operation,
    resolve_scope_and_precedence,
    verify_authorization_verdict,
    verify_extraction_fidelity,
)
from claimlayer.verifiers.quote_quantity import extract_quantities


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _init_mcp_session(*, allow_source_registration: bool = True) -> tuple[MCPVerificationServer, MCPSession]:
    server = MCPVerificationServer(Ledger(), allow_source_registration=allow_source_registration)
    session = MCPSession()
    server.handle_envelope(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}},
        session,
    )
    server.handle_envelope({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
    return server, session


def _call_mcp(server: MCPVerificationServer, session: MCPSession, name: str, arguments: dict) -> dict:
    return server.handle_envelope(
        {"jsonrpc": "2.0", "id": 99, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        session,
    )[0]


def _b2b_layer(reader=None, *, policy: AcceptancePolicy = B2B_DEFAULT_ACCEPTANCE_POLICY) -> EvidenceLayer:
    return EvidenceLayer(reader=reader, vocabulary=B2B_CONTRACTS_VOCABULARY, policy=policy)


# ---------------------------------------------------------------------------
# 1. Finite float validation in Policy, AcceptancePolicy, and MCP server
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False, -0.1, 1.1])
def test_policy_and_assertion_reject_nonfinite_or_bool_thresholds(bad_val):
    with pytest.raises(ValueError):
        Policy(min_support=bad_val)
    with pytest.raises(ValueError):
        Policy(confidence_threshold=bad_val)
    with pytest.raises(ValueError):
        Policy(conflict_escalation_threshold=bad_val)
    with pytest.raises(ValueError):
        Assertion(subject="Acme", predicate="fee", value="10", source_id="s1", quote="fee 10", confidence=bad_val)
    with pytest.raises(ValueError):
        FactRecord(subject="Acme", predicate="fee", value="10", source_id="s1", confidence=bad_val)


@pytest.mark.parametrize(
    "bounds",
    [
        {"fee": (float("nan"), 10)},
        {"fee": (0, float("inf"))},
        {"fee": (float("-inf"), 10)},
        {"fee": (True, 10)},
        {"fee": (20, 10)},
    ],
)
def test_policy_rejects_nonfinite_bool_or_inverted_numeric_bounds(bounds):
    with pytest.raises(ValueError):
        Policy(numeric_bounds=bounds)


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf"), True, False])
def test_v4_acceptance_policy_rejects_nonfinite_and_bool(bad_val):
    with pytest.raises(ValueError):
        AcceptancePolicy(tau=bad_val)
    with pytest.raises(ValueError):
        AcceptancePolicy(delta=bad_val)
    with pytest.raises(ValueError):
        AcceptancePolicy(path_discount=bad_val)


@pytest.mark.parametrize("bad_str", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_mcp_rejects_nonfinite_policy_and_assertion_numbers_with_invalid_params(bad_str):
    server, session = _init_mcp_session()
    reg = _call_mcp(
        server,
        session,
        "register_source",
        {"id": "doc1", "text": "Notice period is 30 days.", "tier": "authoritative"},
    )
    assert reg["result"]["structuredContent"]["registered"] is True

    base_assertion = {
        "subject": "Acme",
        "predicate": "notice_period",
        "value": "30 days",
        "source_id": "doc1",
        "quote": "Notice period is 30 days.",
    }

    for field in ("min_support", "confidence_threshold", "conflict_escalation_threshold"):
        resp = _call_mcp(
            server,
            session,
            "verify_proposal",
            {"agent_id": "a1", "assertions": [base_assertion], "policy": {field: bad_str}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    for bad_bool in (True, False):
        resp = _call_mcp(
            server,
            session,
            "verify_proposal",
            {"agent_id": "a1", "assertions": [base_assertion], "policy": {"min_support": bad_bool}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    resp_bounds = _call_mcp(
        server,
        session,
        "verify_proposal",
        {"agent_id": "a1", "assertions": [base_assertion], "policy": {"numeric_bounds": {"notice_period": [bad_str, 60]}}},
    )
    assert "error" in resp_bounds
    assert resp_bounds["error"]["code"] == -32602

    resp_conf = _call_mcp(
        server,
        session,
        "verify_proposal",
        {"agent_id": "a1", "assertions": [{**base_assertion, "confidence": bad_str}]},
    )
    assert "error" in resp_conf
    assert resp_conf["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# 2 & 3. Mandatory value grounding & token-bounded matching (no 2-in-23 bypass)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("proposed_days", ["777", "2"])
def test_v4_rejects_ungrounded_or_substring_value_even_without_field_evidence(proposed_days):
    text = "The cure period is 23 days."
    doc = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)

    unit = {
        "proposal": {
            "entity": "Acme",
            "predicate": "cure_period",
            "value": proposed_days,
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
        "basis": "explicit",
        "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
    }
    summary = layer.ingest_units([unit], attestor="reader-a", structural_reliability=1.0)
    assert summary["units_admitted"] == 1
    admitted_unit = layer.units[0]

    # Direct verify_extraction_fidelity check without field_evidence
    findings_no_fe = verify_extraction_fidelity(text, admitted_unit)
    assert "VALUE_NOT_IN_EVIDENCE" in {f.code for f in findings_no_fe}

    # Direct verify_extraction_fidelity check with explicit FieldEvidence
    fe = [
        FieldEvidence("entity", EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)),
        FieldEvidence("value", EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)),
    ]
    findings_with_fe = verify_extraction_fidelity(text, admitted_unit, field_evidence=fe, require_field_evidence=("value",))
    assert "VALUE_NOT_IN_EVIDENCE" in {f.code for f in findings_with_fe}

    # End-to-end authorize_operation blocks without field_evidence
    verdict = authorize_operation(layer, entity="Acme", required_predicates=("cure_period",))
    assert verdict.status == "RETRY"
    assert verdict.allowed is False
    assert "VALUE_NOT_IN_EVIDENCE" in {issue.code for issue in verdict.issues}


def test_v4_accepts_token_bounded_exact_and_commensurable_values():
    text = "The cure period for Acme is 23 days."
    doc = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)

    fe = [
        FieldEvidence("entity", EvidenceRef(doc.document_id, doc.revision, 20, 24, "Acme")),
        FieldEvidence("value", EvidenceRef(doc.document_id, doc.revision, 28, 35, "23 days")),
    ]
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "23",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
                "field_evidence": fe,
            }
        ],
        attestor="reader-a",
        structural_reliability=1.0,
    )
    verdict = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("cure_period",),
        require_field_evidence=("entity", "value"),
    )
    assert verdict.status == "PASS"
    assert verdict.allowed is True


# ---------------------------------------------------------------------------
# 4. Unassessed structural_reliability=None -> ESCALATE (never 1.0)
# ---------------------------------------------------------------------------

def test_unassessed_structural_reliability_escalates_and_never_defaults_to_one():
    text = "The cure period for Acme is 30 days."
    doc = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
            }
        ],
        attestor="uncalibrated-reader",
        structural_reliability=None,
    )

    resolved = resolve_scope_and_precedence(
        layer.units,
        entity="Acme",
        predicate="cure_period",
        vocabulary=B2B_CONTRACTS_VOCABULARY,
        documents=layer.documents,
    )
    assert resolved["status"] == "UNASSESSED"
    assert resolved.get("support") is None

    verdict = authorize_operation(layer, entity="Acme", required_predicates=("cure_period",))
    assert verdict.status == "ESCALATE"
    assert verdict.allowed is False
    assert verdict.requires_human is True
    assert "UNASSESSED_READER_RELIABILITY" in {i.code for i in verdict.issues}


# ---------------------------------------------------------------------------
# 5. Layer-native metadata, stale revisions, dependence deduplication, digests
# ---------------------------------------------------------------------------

def test_authorize_reads_layer_authority_weight_field_evidence_and_unread_status():
    t_primary = "Section 5: The cure period for Acme is 30 days."
    t_secondary = "Annex A: The cure period for Acme is 15 days."
    t_unread = "Side Letter: Pending review."
    d_primary = SourceDocument("msa.txt", t_primary, authority_rank=90, weight=0.9, dependence_group="grp-a")
    d_secondary = SourceDocument("annex.txt", t_secondary, authority_rank=20, weight=0.9, dependence_group="grp-b")
    d_unread = SourceDocument("side.txt", t_unread, authority_rank=80, weight=0.9)

    layer = _b2b_layer()
    layer.ingest([d_primary, d_secondary, d_unread])
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d_primary.document_id, d_primary.revision, 0, len(t_primary), t_primary)],
                "field_evidence": [
                    FieldEvidence("entity", EvidenceRef(d_primary.document_id, d_primary.revision, 31, 35, "Acme")),
                    FieldEvidence("value", EvidenceRef(d_primary.document_id, d_primary.revision, 39, 46, "30 days")),
                ],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "15",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d_secondary.document_id, d_secondary.revision, 0, len(t_secondary), t_secondary)],
                "field_evidence": [
                    FieldEvidence("entity", EvidenceRef(d_secondary.document_id, d_secondary.revision, 29, 33, "Acme")),
                    FieldEvidence("value", EvidenceRef(d_secondary.document_id, d_secondary.revision, 37, 44, "15 days")),
                ],
            },
        ],
        attestor="reader",
        structural_reliability=1.0,
    )

    # d_unread was never read -> authorize_operation detects UNREAD_DOCUMENTS automatically from layer
    verdict_unread = authorize_operation(
        layer,
        entity="Acme",
        required_predicates=("cure_period",),
        require_field_evidence=("entity", "value"),
    )
    assert verdict_unread.status == "RETRY"
    assert "UNREAD_DOCUMENTS" in {i.code for i in verdict_unread.issues}

    # Now ingest a clean layer with only d_primary + d_secondary (no unread docs),
    # and bind operation_id + payload -> higher authority_rank on layer wins without source_tiers arg.
    layer_clean = _b2b_layer()
    layer_clean.ingest([d_primary, d_secondary])
    layer_clean.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d_primary.document_id, d_primary.revision, 0, len(t_primary), t_primary)],
                "field_evidence": [
                    FieldEvidence("entity", EvidenceRef(d_primary.document_id, d_primary.revision, 31, 35, "Acme")),
                    FieldEvidence("value", EvidenceRef(d_primary.document_id, d_primary.revision, 39, 46, "30 days")),
                ],
            },
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "15",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d_secondary.document_id, d_secondary.revision, 0, len(t_secondary), t_secondary)],
                "field_evidence": [
                    FieldEvidence("entity", EvidenceRef(d_secondary.document_id, d_secondary.revision, 29, 33, "Acme")),
                    FieldEvidence("value", EvidenceRef(d_secondary.document_id, d_secondary.revision, 37, 44, "15 days")),
                ],
            },
        ],
        attestor="reader",
        structural_reliability=1.0,
    )
    verdict_ok = authorize_operation(
        layer_clean,
        entity="Acme",
        required_predicates=("cure_period",),
        require_field_evidence=("entity", "value"),
        operation_id="op-execute-1",
        payload={"action": "update_crm", "cure_days": 30},
    )
    assert verdict_ok.status == "PASS"
    assert verdict_ok.allowed is True
    assert verdict_ok.resolved_values["cure_period"]["value"] == "30"
    assert verdict_ok.operation_id == "op-execute-1"
    assert len(verdict_ok.payload_digest or "") == 64
    assert len(verdict_ok.snapshot_digest or "") == 64

    # Repeated identical calls yield identical snapshot_digest and payload_digest.
    verdict_repeat = authorize_operation(
        layer_clean,
        entity="Acme",
        required_predicates=("cure_period",),
        require_field_evidence=("entity", "value"),
        operation_id="op-execute-1",
        payload={"action": "update_crm", "cure_days": 30},
    )
    assert verdict_repeat.payload_digest == verdict_ok.payload_digest
    assert verdict_repeat.snapshot_digest == verdict_ok.snapshot_digest

    # Mutating payload changes both payload_digest and bound snapshot_digest.
    verdict_other_payload = authorize_operation(
        layer_clean,
        entity="Acme",
        required_predicates=("cure_period",),
        require_field_evidence=("entity", "value"),
        operation_id="op-execute-1",
        payload={"action": "update_crm", "cure_days": 999},
    )
    assert verdict_other_payload.payload_digest != verdict_ok.payload_digest
    assert verdict_other_payload.snapshot_digest != verdict_ok.snapshot_digest


def test_dependence_group_deduplicates_noisy_or_support():
    policy = AcceptancePolicy(tau=0.80, delta=0.20, allowed_assertion_kinds=("requirement", "fact", "definition"))
    t1 = "Source 1: The cure period for Acme is 30 days."
    t2 = "Source 2: The cure period for Acme is 30 days."

    # Same dependence_group -> support stays at max(0.6, 0.6) = 0.6 < tau=0.8 -> ESCALATE
    same_grp = _b2b_layer(policy=policy)
    d1 = SourceDocument("s1.txt", t1, authority_rank=100, weight=0.6, dependence_group="shared-feed")
    d2 = SourceDocument("s2.txt", t2, authority_rank=100, weight=0.6, dependence_group="shared-feed")
    same_grp.ingest([d1, d2])
    same_grp.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d.document_id, d.revision, 0, len(d.text), d.text)],
            }
            for d in (d1, d2)
        ],
        attestor="reader",
        structural_reliability=1.0,
    )
    v_same = authorize_operation(same_grp, entity="Acme", required_predicates=("cure_period",))
    assert v_same.status == "ESCALATE"

    # Distinct dependence_groups -> 1 - 0.4 * 0.4 = 0.84 >= tau=0.8 -> PASS
    diff_grp = _b2b_layer(policy=policy)
    d3 = SourceDocument("s3.txt", t1, authority_rank=100, weight=0.6, dependence_group="feed-a")
    d4 = SourceDocument("s4.txt", t2, authority_rank=100, weight=0.6, dependence_group="feed-b")
    diff_grp.ingest([d3, d4])
    diff_grp.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d.document_id, d.revision, 0, len(d.text), d.text)],
            }
            for d in (d3, d4)
        ],
        attestor="reader",
        structural_reliability=1.0,
    )
    v_diff = authorize_operation(diff_grp, entity="Acme", required_predicates=("cure_period",))
    assert v_diff.status == "PASS"


def test_stale_revision_and_read_errors_emit_structured_retry_without_crash():
    t_v1 = "The cure period for Acme is 30 days."
    t_v2 = "The cure period for Acme is 45 days."
    d_v1 = SourceDocument("msa.txt", t_v1, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(d_v1)
    layer.ingest_units(
        [
            {
                "proposal": {
                    "entity": "Acme",
                    "predicate": "cure_period",
                    "value": "30",
                    "unit": "calendar_days",
                    "assertion_kind": "requirement",
                    "condition_status": "unconditional",
                },
                "basis": "explicit",
                "evidence": [EvidenceRef(d_v1.document_id, d_v1.revision, 0, len(t_v1), t_v1)],
            }
        ],
        attestor="reader",
        structural_reliability=1.0,
    )

    # Updating the document revision makes previously ingested units stale -> structured RETRY (no ValueError)
    d_v2 = SourceDocument("msa.txt", t_v2, authority_rank=100, weight=0.95)
    layer.ingest(d_v2)
    verdict = authorize_operation(layer, entity="Acme", required_predicates=("cure_period",))
    assert verdict.status == "RETRY"
    assert verdict.allowed is False
    assert "STALE_EVIDENCE_REVISION" in {i.code for i in verdict.issues}

    # Failing reader records read_errors directly on layer -> detected even without ingest_summary
    class _BrokenReader:
        identity = {"model": "broken", "prompt_sha256": "0"}

        def propose(self, request):
            raise RuntimeError("upstream parse timeout")

    layer_err = _b2b_layer(reader=_BrokenReader())
    layer_err.ingest(SourceDocument("bad.txt", t_v1, authority_rank=100, weight=0.95))
    verdict_err = authorize_operation(layer_err, entity="Acme", required_predicates=("cure_period",))
    assert verdict_err.status == "RETRY"
    assert "READ_ERRORS_PRESENT" in {i.code for i in verdict_err.issues}


# ---------------------------------------------------------------------------
# 6. numeric_bounds unit normalization in both v3 and v4 ("2 weeks" == "14 days")
# ---------------------------------------------------------------------------

def test_numeric_bounds_unit_normalization_in_v3_and_v4():
    # v3: 2 weeks == 14 days against both days-based and weeks-based bounds
    src_days = Source(id="s_days", text="The notice window is 14 days.", tier="authoritative")
    src_weeks = Source(id="s_weeks", text="The notice window is 2 weeks.", tier="authoritative")
    a_days = Assertion(subject="Acme", predicate="notice_window", value="14 days", source_id="s_days", quote=src_days.text)
    a_weeks = Assertion(subject="Acme", predicate="notice_window", value="2 weeks", source_id="s_weeks", quote=src_weeks.text)

    for bound_key, bound_range, expected_status in [
        ("notice_window:days", (1, 10), "ESCALATE"),
        ("notice_window:days", (1, 14), "PASS"),
        ("notice_window:weeks", (1, 1), "ESCALATE"),
        ("notice_window:weeks", (1, 2), "PASS"),
    ]:
        pol = Policy(numeric_bounds={bound_key: bound_range})
        v_days = verify([a_days], Ledger(sources=[src_days]), pol)
        v_weeks = verify([a_weeks], Ledger(sources=[src_weeks]), pol)
        assert v_days.status == expected_status
        assert v_weeks.status == expected_status
        assert [i.code for i in v_days.issues] == [i.code for i in v_weeks.issues]

    # v4: {"max": 2, "unit": "weeks"} vs {"max": 14, "unit": "calendar_days"} behave identically
    for text, should_pass in [("The cure period is 14 days.", True), ("The cure period is 15 days.", False)]:
        val = "14" if should_pass else "15"
        doc = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
        layer = _b2b_layer()
        layer.ingest(doc)
        layer.ingest_units(
            [
                {
                    "proposal": {
                        "entity": "Acme",
                        "predicate": "cure_period",
                        "value": val,
                        "unit": "calendar_days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                    },
                    "basis": "explicit",
                    "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(text), text)],
                }
            ],
            attestor="reader",
            structural_reliability=1.0,
        )
        v_bound_weeks = authorize_operation(
            layer,
            entity="Acme",
            required_predicates=("cure_period",),
            numeric_bounds={"cure_period": {"max": 2, "unit": "weeks"}},
        )
        v_bound_days = authorize_operation(
            layer,
            entity="Acme",
            required_predicates=("cure_period",),
            numeric_bounds={"cure_period": {"max": 14, "unit": "calendar_days"}},
        )
        assert v_bound_weeks.status == v_bound_days.status
        assert v_bound_weeks.status == ("PASS" if should_pass else "ESCALATE")
        if not should_pass:
            assert "POLICY_BOUND_ABOVE_MAX" in {i.code for i in v_bound_weeks.issues}


# ---------------------------------------------------------------------------
# 7. Word-fraction + unit composition & PYTHONHASHSEED determinism
# ---------------------------------------------------------------------------

def test_word_fractions_compose_with_following_unit_unless_standalone():
    for phrase in ("one half year", "one half of a year"):
        qs = extract_quantities(f"The renewal window is {phrase}.")
        assert len(qs) == 1
        assert qs[0].magnitude == Decimal("0.5")
        assert qs[0].unit == "YEARS"

    standalone = extract_quantities("It covers one half of the portfolio.")
    assert any(q.magnitude == Decimal("50") and q.unit == "PERCENT" for q in standalone)


def test_multi_quantity_subject_binding_invariant_across_hashseeds():
    script = """
import json
from claimlayer import Assertion, Ledger, Source, verify

text = "Section 4 limits: Alpha Corp cap is $1,000,000 and Beta LLC cap is $2,500,000."
ledger = Ledger(sources=[Source(id="doc", text=text, tier="authoritative")])
v1 = verify([Assertion(subject="Alpha Corp", predicate="liability_cap", value="$1,000,000", source_id="doc", quote=text)], ledger)
v2 = verify([Assertion(subject="Beta LLC", predicate="liability_cap", value="$1,000,000", source_id="doc", quote=text)], ledger)
print(json.dumps({
    "v1_status": v1.status,
    "v1_codes": [i.code for i in v1.issues],
    "v2_status": v2.status,
    "v2_codes": [i.code for i in v2.issues],
}, sort_keys=True))
"""
    outputs = []
    for seed in ("1", "2"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(REPO_ROOT / "src")
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
        outputs.append(json.loads(proc.stdout))

    assert outputs[0] == outputs[1]
    assert outputs[0]["v1_status"] == "PASS"
    assert outputs[0]["v2_status"] == "RETRY"


# ---------------------------------------------------------------------------
# 8. b2b_live_eval benchmark methodology fixes
# ---------------------------------------------------------------------------

def test_b2b_live_eval_methodology_invariants():
    import benchmarks.b2b_live_eval as b2b_eval

    report = b2b_eval.run_benchmark(live=False)
    assert report["mode"] == "dry-run"
    assert report["reader_a_dev_measured_reliability"] == pytest.approx(0.9)
    assert report["layer1_gold_reasoning"]["dev"]["accuracy"] == 1.0
    assert report["layer1_gold_reasoning"]["test"]["accuracy"] == 1.0
    assert report["layer2_dual_reader_extraction"]["test"]["shared_error_rate"] > 0.0
    assert report["layer3_risk_coverage"]["blind_test_metrics"]["selective_risk"] == 0.0
    assert all(c["values_match"] for c in report["layer3_risk_coverage"]["blind_test_cases"])

    # False PASS includes value mismatch even when expected_allowed == actual_allowed == True
    rc_wrong_val = b2b_eval._risk_coverage_at(
        [
            {"expected_allowed": True, "actual_allowed": True, "values_match": True},
            {"expected_allowed": True, "actual_allowed": True, "values_match": False},
        ]
    )
    assert rc_wrong_val["accepted"] == 2
    assert rc_wrong_val["false_pass"] == 1
    assert rc_wrong_val["selective_risk"] == pytest.approx(0.5)

    # Single-pass extraction: every LLMProposalReader.propose call occurs inside
    # _CapturingReader (dev + test once per reader); the 3-point sweep and blind test
    # execute strictly via _ReplayReader with zero additional LLM calls.
    llm_calls = 0
    cap_calls = 0
    replay_calls = 0
    orig_llm_propose = LLMProposalReader.propose
    orig_cap_propose = b2b_eval._CapturingReader.propose
    orig_replay_propose = b2b_eval._ReplayReader.propose

    def _count_llm(self, request):
        nonlocal llm_calls
        llm_calls += 1
        return orig_llm_propose(self, request)

    def _count_cap(self, request):
        nonlocal cap_calls
        cap_calls += 1
        return orig_cap_propose(self, request)

    def _count_replay(self, request):
        nonlocal replay_calls
        replay_calls += 1
        return orig_replay_propose(self, request)

    LLMProposalReader.propose = _count_llm
    b2b_eval._CapturingReader.propose = _count_cap
    b2b_eval._ReplayReader.propose = _count_replay
    try:
        b2b_eval.run_benchmark(live=False)
    finally:
        LLMProposalReader.propose = orig_llm_propose
        b2b_eval._CapturingReader.propose = orig_cap_propose
        b2b_eval._ReplayReader.propose = orig_replay_propose

    assert cap_calls > 0
    assert llm_calls == cap_calls
    assert replay_calls > 0

    # Semantic alignment detects omissions as disagreements even if the other reader missed it
    gold = [
        {"case_id": "c1", "entity": "Acme", "predicate": "cure_period", "scope": {}, "document_id": "d1", "value": "30", "unit": "calendar_days", "polarity": "positive"},
        {"case_id": "c1", "entity": "Acme", "predicate": "payment_terms", "scope": {}, "document_id": "d1", "value": "45", "unit": "calendar_days", "polarity": "positive"},
    ]
    rec_a = [gold[0], gold[1]]
    rec_b = [gold[0]]  # payment_terms omitted by reader B
    aligned = b2b_eval._align_dual_reader_cases(gold, rec_a, rec_b)
    assert len(aligned) == 2
    assert b2b_eval.evaluate_dual_readers(aligned)["disagreement_rate"] > 0.0


# ---------------------------------------------------------------------------
# 9. Finding A: strict unit compatibility in v4 numeric fidelity
# ---------------------------------------------------------------------------

def _ingest_single_unit(
    layer: EvidenceLayer,
    doc: SourceDocument,
    proposal: dict,
    *,
    basis: str = "explicit",
    structural_reliability: float = 1.0,
) -> dict:
    return layer.ingest_units(
        [
            {
                "proposal": proposal,
                "basis": basis,
                "evidence": [EvidenceRef(doc.document_id, doc.revision, 0, len(doc.text), doc.text)],
            }
        ],
        attestor="reader-a",
        structural_reliability=structural_reliability,
    )


@pytest.mark.parametrize(
    ("doc_text", "proposed_unit", "proposed_val"),
    [
        ("Orion filing charge is 23 USD.", "calendar_days", "23"),
        ("Orion cure period is 23 business days.", "calendar_days", "23"),
    ],
)
def test_v4_rejects_incompatible_units_on_same_magnitude(doc_text, proposed_unit, proposed_val):
    doc = SourceDocument("msa.txt", doc_text, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)
    summary = _ingest_single_unit(
        layer,
        doc,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": proposed_val,
            "unit": proposed_unit,
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )
    assert summary["units_admitted"] == 1
    findings = verify_extraction_fidelity(doc_text, layer.units[0])
    assert "VALUE_NOT_IN_EVIDENCE" in {f.code for f in findings}

    verdict = authorize_operation(layer, entity="Orion", required_predicates=("cure_period",))
    assert verdict.status == "RETRY"
    assert verdict.allowed is False
    assert verdict.resolved_values == {}
    assert "VALUE_NOT_IN_EVIDENCE" in {i.code for i in verdict.issues}


@pytest.mark.parametrize(
    ("doc_text", "proposed_unit", "proposed_val"),
    [
        ("Orion cure period is 23 business days.", "business_days", "23"),
        ("Orion cure period is 23 days.", "calendar_days", "23"),
        ("Orion cure period is 2 weeks.", "calendar_days", "14"),
    ],
)
def test_v4_accepts_compatible_and_convertible_units(doc_text, proposed_unit, proposed_val):
    doc = SourceDocument("msa.txt", doc_text, authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)
    _ingest_single_unit(
        layer,
        doc,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": proposed_val,
            "unit": proposed_unit,
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )
    findings = verify_extraction_fidelity(doc_text, layer.units[0])
    assert findings == []

    verdict = authorize_operation(layer, entity="Orion", required_predicates=("cure_period",))
    assert verdict.status == "PASS"
    assert verdict.allowed is True
    assert verdict.resolved_values == {"cure_period": {"value": proposed_val, "unit": proposed_unit}}


def test_v4_custom_vocabulary_unit_fidelity_and_rejection():
    custom_units = STANDARD_UNITS.extend(
        [
            Unit("pallet", "package_count", "1", ("pallet", "pallets")),
            Unit("crate", "package_count", "2", ("crate", "crates")),
            Unit("parcel", "parcel_count", "1", ("parcel", "parcels")),
        ]
    )
    custom_vocab = Vocabulary(
        "logistics",
        "1",
        (PredicateSpec("delivery_qty", FieldSpec("quantity", dimensions=("package_count",))),),
        units=custom_units,
    )

    doc_ok = SourceDocument("ship.txt", "Deliver 4 crates to Orion.", authority_rank=100, weight=0.95)
    layer_ok = EvidenceLayer(vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer_ok.ingest(doc_ok)
    _ingest_single_unit(
        layer_ok,
        doc_ok,
        {
            "entity": "Orion",
            "predicate": "delivery_qty",
            "value": "8",
            "unit": "pallet",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )
    assert verify_extraction_fidelity(doc_ok.text, layer_ok.units[0], vocabulary=custom_vocab) == []
    verdict_ok = authorize_operation(layer_ok, entity="Orion", required_predicates=("delivery_qty",))
    assert verdict_ok.status == "PASS"
    assert verdict_ok.allowed is True

    doc_bad = SourceDocument("ship_bad.txt", "Deliver 4 parcels to Orion.", authority_rank=100, weight=0.95)
    layer_bad = EvidenceLayer(vocabulary=custom_vocab, policy=B2B_DEFAULT_ACCEPTANCE_POLICY)
    layer_bad.ingest(doc_bad)
    _ingest_single_unit(
        layer_bad,
        doc_bad,
        {
            "entity": "Orion",
            "predicate": "delivery_qty",
            "value": "4",
            "unit": "pallet",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )
    bad_findings = verify_extraction_fidelity(doc_bad.text, layer_bad.units[0], vocabulary=custom_vocab)
    assert "VALUE_NOT_IN_EVIDENCE" in {f.code for f in bad_findings}
    verdict_bad = authorize_operation(layer_bad, entity="Orion", required_predicates=("delivery_qty",))
    assert verdict_bad.status == "RETRY"
    assert verdict_bad.allowed is False
    assert verdict_bad.resolved_values == {}


# ---------------------------------------------------------------------------
# 10. Finding B: unified admissibility across fidelity and resolution
# ---------------------------------------------------------------------------

def test_v4_excluded_assertion_kind_source_attribution_and_temporal_validity_never_bypass_fidelity():
    text = "Orion cure period is 23 days."

    # 1) Excluded assertion_kind=("fact",) with allowed_assertion_kinds=("requirement",)
    policy_req = AcceptancePolicy(tau=0.70, delta=0.20, allowed_assertion_kinds=("requirement",))
    doc1 = SourceDocument("msa.txt", text, authority_rank=100, weight=0.95)
    layer1 = _b2b_layer(policy=policy_req)
    layer1.ingest(doc1)
    _ingest_single_unit(
        layer1,
        doc1,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": "777",
            "unit": "calendar_days",
            "assertion_kind": "fact",
            "condition_status": "unconditional",
        },
    )
    verdict1 = authorize_operation(layer1, entity="Orion", required_predicates=("cure_period",))
    assert verdict1.status == "RETRY"
    assert verdict1.allowed is False
    assert verdict1.resolved_values == {}
    assert "MISSING_REQUIRED_PREDICATE" in {i.code for i in verdict1.issues}

    # 2) Excluded source attribution
    policy_attr = AcceptancePolicy(tau=0.70, delta=0.20, allowed_source_attributions=("controlling_instrument",))
    doc2 = SourceDocument("note.txt", text, authority_rank=100, weight=0.95, attribution="informational")
    layer2 = _b2b_layer(policy=policy_attr)
    layer2.ingest(doc2)
    _ingest_single_unit(
        layer2,
        doc2,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": "777",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )
    verdict2 = authorize_operation(layer2, entity="Orion", required_predicates=("cure_period",))
    assert verdict2.status == "RETRY"
    assert verdict2.allowed is False
    assert verdict2.resolved_values == {}

    # 3) Expired temporal validity relative to as_of
    doc3 = SourceDocument("msa_old.txt", text, authority_rank=100, weight=0.95)
    layer3 = _b2b_layer()
    layer3.ingest(doc3)
    _ingest_single_unit(
        layer3,
        doc3,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": "777",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "valid_from": "2024-01-01",
            "valid_until": "2024-12-31",
        },
    )
    verdict3 = authorize_operation(
        layer3,
        entity="Orion",
        required_predicates=("cure_period",),
        as_of="2025-06-01",
    )
    assert verdict3.status == "RETRY"
    assert verdict3.allowed is False
    assert verdict3.resolved_values == {}

    # 4) Positive control: admitted assertion_kind, attribution, and active window pass cleanly
    policy_both = AcceptancePolicy(
        tau=0.70,
        delta=0.20,
        allowed_assertion_kinds=("requirement",),
        allowed_source_attributions=("controlling_instrument",),
    )
    doc4 = SourceDocument("msa_valid.txt", text, authority_rank=100, weight=0.95, attribution="controlling_instrument")
    layer4 = _b2b_layer(policy=policy_both)
    layer4.ingest(doc4)
    _ingest_single_unit(
        layer4,
        doc4,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": "23",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "valid_from": "2025-01-01",
            "valid_until": "2025-12-31",
        },
    )
    verdict4 = authorize_operation(
        layer4,
        entity="Orion",
        required_predicates=("cure_period",),
        as_of="2025-06-01",
    )
    assert verdict4.status == "PASS"
    assert verdict4.allowed is True
    assert verdict4.resolved_values == {"cure_period": {"value": "23", "unit": "calendar_days"}}


# ---------------------------------------------------------------------------
# 11. Finding C: complete snapshot_digest over inputs and document revisions
# ---------------------------------------------------------------------------

def test_v4_snapshot_digest_binds_as_of_bounds_relations_tiers_and_doc_revisions():
    def _build_and_authorize(
        *,
        doc_text: str = "Orion cure period is 23 days.",
        as_of: str = "2025-01-01",
        bounds: dict | None = None,
        relations: tuple[ClauseRelation, ...] = (),
        tiers: dict | None = None,
    ):
        doc = SourceDocument("msa.txt", doc_text, authority_rank=100, weight=0.95)
        layer = _b2b_layer()
        layer.ingest(doc)
        _ingest_single_unit(
            layer,
            doc,
            {
                "entity": "Orion",
                "predicate": "cure_period",
                "value": "23",
                "unit": "calendar_days",
                "assertion_kind": "requirement",
                "condition_status": "unconditional",
            },
        )
        return authorize_operation(
            layer,
            entity="Orion",
            required_predicates=("cure_period",),
            as_of=as_of,
            numeric_bounds=bounds if bounds is not None else {"cure_period": (None, 100.0)},
            clause_relations=relations,
            source_tiers=tiers if tiers is not None else {"msa.txt": 100},
        )

    v_base = _build_and_authorize()
    v_repeat = _build_and_authorize()
    assert v_base.status == "PASS"
    assert v_base.snapshot_digest == v_repeat.snapshot_digest

    variants = [
        ("as_of", _build_and_authorize(as_of="2026-01-01")),
        ("numeric_bounds", _build_and_authorize(bounds={"cure_period": (None, 200.0)})),
        ("clause_relations", _build_and_authorize(relations=(ClauseRelation("amends_clause", "2", "1"),))),
        ("source_tiers", _build_and_authorize(tiers={"msa.txt": 80})),
        ("doc_revision", _build_and_authorize(doc_text="Section 1: Orion cure period is 23 days.")),
    ]
    seen_digests = {v_base.snapshot_digest}
    for label, v_other in variants:
        assert v_other.status == "PASS", f"Expected PASS for {label}"
        assert v_other.resolved_values == v_base.resolved_values
        assert v_other.snapshot_digest not in seen_digests, f"snapshot_digest collision on {label}"
        seen_digests.add(v_other.snapshot_digest)


# ---------------------------------------------------------------------------
# 12. Finding D: payload_digest verification & HMAC-SHA256 verdict verification
# ---------------------------------------------------------------------------

def test_v4_payload_digest_mismatch_and_hmac_verification():
    from dataclasses import replace as dc_replace

    doc = SourceDocument("msa.txt", "Orion cure period is 23 days.", authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)
    _ingest_single_unit(
        layer,
        doc,
        {
            "entity": "Orion",
            "predicate": "cure_period",
            "value": "23",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
        },
    )

    v_mismatch = authorize_operation(
        layer,
        entity="Orion",
        required_predicates=("cure_period",),
        operation_id="op-mismatch",
        payload={"days": 23},
        payload_digest="FAKE",
    )
    assert v_mismatch.status == "ESCALATE"
    assert v_mismatch.allowed is False
    assert v_mismatch.resolved_values == {}
    assert v_mismatch.payload_digest != "FAKE"
    assert "PAYLOAD_DIGEST_MISMATCH" in {i.code for i in v_mismatch.issues}

    v_invalid = authorize_operation(
        layer,
        entity="Orion",
        required_predicates=("cure_period",),
        operation_id="op-invalid",
        payload=None,
        payload_digest="FAKE",
    )
    assert v_invalid.status == "ESCALATE"
    assert v_invalid.allowed is False
    assert v_invalid.resolved_values == {}
    assert v_invalid.payload_digest is None
    assert "INVALID_PAYLOAD_DIGEST" in {i.code for i in v_invalid.issues}

    key = b"secret-signing-key-v4"
    v_signed = authorize_operation(
        layer,
        entity="Orion",
        required_predicates=("cure_period",),
        operation_id="op-1",
        payload={"days": 23},
        signing_key=key,
    )
    assert v_signed.status == "PASS"
    assert v_signed.allowed is True
    assert v_signed.signature is not None
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=layer,
            operation_id="op-1",
            payload={"days": 23},
            signing_key=key,
        )
        is True
    )

    # Mutations must fail closed
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=layer,
            operation_id="op-1",
            payload={"days": 24},
            signing_key=key,
        )
        is False
    )
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=layer,
            operation_id="op-2",
            payload={"days": 23},
            signing_key=key,
        )
        is False
    )
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=layer,
            operation_id="op-1",
            payload={"days": 23},
            signing_key=b"wrong-key",
        )
        is False
    )
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=layer,
            operation_id="op-1",
            payload={"days": 23},
            signing_key=None,
            allow_unkeyed=False,
        )
        is False
    )
    v_forged = dc_replace(v_signed, signature="f" * 64)
    assert (
        verify_authorization_verdict(
            v_forged,
            layer=layer,
            operation_id="op-1",
            payload={"days": 23},
            signing_key=key,
        )
        is False
    )

    mutated_layer = _b2b_layer()
    mutated_doc = SourceDocument("msa.txt", "Orion cure period is 99 days.", authority_rank=100, weight=0.95)
    mutated_layer.ingest(mutated_doc)
    assert (
        verify_authorization_verdict(
            v_signed,
            layer=mutated_layer,
            operation_id="op-1",
            payload={"days": 23},
            signing_key=key,
        )
        is False
    )


# ---------------------------------------------------------------------------
# 13. Finding E: word fraction + percent must not multiply by 100
# ---------------------------------------------------------------------------

def test_v3_word_fractions_followed_by_percent_do_not_double_scale():
    q1 = extract_quantities("one half percent")
    assert len(q1) == 1
    assert q1[0].magnitude == Decimal("0.5")
    assert q1[0].unit == "PERCENT"

    q2 = extract_quantities("one-half of a percent")
    assert len(q2) == 1
    assert q2[0].magnitude == Decimal("0.5")
    assert q2[0].unit == "PERCENT"

    q3 = extract_quantities("one quarter of one percent")
    assert len(q3) == 1
    assert q3[0].magnitude == Decimal("0.25")
    assert q3[0].unit == "PERCENT"

    # End-to-end v3 verification
    src = Source(id="s1", text="The servicing fee is one half percent.", tier="authoritative")
    ledger = Ledger(sources=[src])
    ok = verify(
        [Assertion(subject="Acme", predicate="servicing_fee", value="0.5%", source_id="s1", quote=src.text)],
        ledger,
        Policy(min_support=0.5),
    )
    bad = verify(
        [Assertion(subject="Acme", predicate="servicing_fee", value="50%", source_id="s1", quote=src.text)],
        ledger,
        Policy(min_support=0.5),
    )
    assert ok.status == "PASS"
    assert bad.status == "RETRY"

    # Positive control: standalone fraction without explicit percent/unit remains a percentage ratio
    standalone = extract_quantities("one half of the portfolio")
    assert len(standalone) == 1
    assert standalone[0].magnitude == Decimal("50")
    assert standalone[0].unit == "PERCENT"


# ---------------------------------------------------------------------------
# 14. b2b_live_eval static oracle, record fields, and Reader B independence
# ---------------------------------------------------------------------------

def test_b2b_live_eval_static_oracle_record_fields_and_reader_b_independence():
    from dataclasses import replace as dc_replace
    import benchmarks.b2b_live_eval as b2b_eval

    dev_cases = b2b_eval._build_dev_cases()
    test_cases = b2b_eval._build_test_cases()
    for c in dev_cases + test_cases:
        if c.expected_allowed:
            assert len(c.expected_resolved_values) > 0, f"{c.case_id} missing static expected_resolved_values"

    # Verify _canonical_unit_record preserves assertion_kind and temporal validity fields
    doc = SourceDocument("d.txt", "Acme cure period is 30 days.", authority_rank=100, weight=0.95)
    layer = _b2b_layer()
    layer.ingest(doc)
    _ingest_single_unit(
        layer,
        doc,
        {
            "entity": "Acme",
            "predicate": "cure_period",
            "value": "30",
            "unit": "calendar_days",
            "assertion_kind": "requirement",
            "condition_status": "unconditional",
            "valid_from": "2025-01-01",
            "valid_until": "2025-12-31",
        },
    )
    rec = b2b_eval._canonical_unit_record("c1", layer.units[0])
    assert rec["assertion_kind"] == "requirement"
    assert rec["valid_from"] == "2025-01-01"
    assert rec["valid_until"] == "2025-12-31"

    # Inject hallucinated Reader B proposal on dev; Reader A reliability & sample size must not change
    base_report = b2b_eval.run_benchmark(live=False)
    orig_build_dev = b2b_eval._build_dev_cases

    def _dev_with_reader_b_hallucination():
        cases = orig_build_dev()
        out = []
        for c in cases:
            if c.case_id == "dev_01_clean_pass":
                mut_b = {k: list(v) for k, v in c.reader_b_output.items()}
                mut_b["dev1_msa"].append(
                    {
                        "entity": "ApexCorp",
                        "predicate": "cure_period",
                        "value": 999,
                        "unit": "days",
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "quote": "ApexCorp standard payment terms are 30 days.",
                        "field_evidence": {"entity": "ApexCorp", "value": "30 days"},
                    }
                )
                out.append(dc_replace(c, reader_b_output=mut_b))
            else:
                out.append(c)
        return out

    b2b_eval._build_dev_cases = _dev_with_reader_b_hallucination
    try:
        mut_report = b2b_eval.run_benchmark(live=False)
    finally:
        b2b_eval._build_dev_cases = orig_build_dev

    assert mut_report["reader_a_dev_measured_reliability"] == pytest.approx(
        base_report["reader_a_dev_measured_reliability"]
    )
    assert mut_report["reader_a_dev_sample_size"] == base_report["reader_a_dev_sample_size"] == 10
    assert (
        mut_report["layer2_dual_reader_extraction"]["dev"]["disagreement_rate"]
        > base_report["layer2_dual_reader_extraction"]["dev"]["disagreement_rate"]
    )
    assert "calibration_note" in mut_report
