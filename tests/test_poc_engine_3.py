"""Offline unit tests for POC-3 (JustificationGraph, DeclaredRules, relations, monotonicity)."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_POC_DIR = os.path.join(_REPO_ROOT, "case_studies", "poc_engine_3")
if _POC_DIR not in sys.path:
    sys.path.insert(0, _POC_DIR)

import poc_domain as pd  # noqa: E402
from claimlayer.semantic.models import EvidenceRef  # noqa: E402
from claimlayer.v4.evaluation import evaluate_response  # noqa: E402
from claimlayer.v4.justification import (  # noqa: E402
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    VerificationObligation,
)


def test_preregistration_complete_and_valid() -> None:
    prereg = pd.load_preregistration()
    assert len(prereg["claims"]) == 9
    assert prereg["reference_date"] == "2026-11-20"
    assert [c["claim_id"] for c in prereg["claims"]] == [
        "C1a", "C1b", "C1c", "C2a", "C2b", "C2c", "C3a", "C3b", "C3c",
    ]
    allowed = set(prereg["labels"])
    for c in prereg["claims"]:
        assert c["expected_label"] in allowed
        assert set(c["source_ids"]) <= set(prereg["sources"].keys())


def test_mode_a_all_nine_claims_correct_and_traceable() -> None:
    results = pd.run_mode_a_all(use_default_registry=False)
    by_id = {r["claim_id"]: r for r in results}
    assert len(by_id) == 9
    assert all(r["correct"] for r in results), [r for r in results if not r["correct"]]

    # Verify exact epistemic kinds and trace contents
    assert by_id["C1a"]["obtained_label"] == "DERIVADA"
    assert by_id["C1a"]["effective_epistemic_kind"] == "derivation"
    assert "R1_persona_mayor" in by_id["C1a"]["trace"]["rules_applied"]

    assert by_id["C1b"]["obtained_label"] == "CONTRADICHA"
    assert any(
        "TEMPORAL_DERIVATION_CONTRADICTED" in c["reasons"]
        for st in by_id["C1b"]["trace"]["inference_steps"]
        for c in st["checks"]
    )

    assert by_id["C1c"]["obtained_label"] == "SIN EVIDENCIA"
    assert by_id["C2a"]["obtained_label"] == "DERIVADA"
    assert set(by_id["C2a"]["trace"]["rules_applied"]) == {
        "R2a_prop_implica_control",
        "R2_controla_transitiva",
    }
    assert by_id["C2b"]["obtained_label"] == "SIN EVIDENCIA"
    assert by_id["C2c"]["obtained_label"] == "SIN EVIDENCIA"

    assert by_id["C3a"]["obtained_label"] == "HIPÓTESIS"
    assert by_id["C3a"]["effective_epistemic_kind"] == "hypothesis"
    assert by_id["C3b"]["obtained_label"] == "DERIVADA"
    assert by_id["C3b"]["effective_epistemic_kind"] == "derivation"
    assert by_id["C3c"]["obtained_label"] == "CONTRADICHA"


def test_negative_invariants_binding_direction_grounding_and_raw_engine_limits() -> None:
    docs = pd.build_poc_sources()
    prof = pd.build_poc_domain_profile()
    reg = pd.build_poc_verifier_registry()

    # 1. Inverted variable binding (Beta -> Alfa from Alfa -> Beta ownership) MUST fail R2a_prop_implica_control
    ref_s2 = EvidenceRef(document_id="S2", revision="rev-0", start=0, end=len(docs["S2"].text), quote=docs["S2"].text)
    g_bad_r2a = JustificationGraph(
        claims={
            "e": ClaimNode(claim_id="e", kind="extraction", subject="Alfa S.A.", predicate="propietaria_100", value="Beta S.L.", evidence=(ref_s2,)),
            "c": ClaimNode(claim_id="c", kind="derivation", subject="Beta S.L.", predicate="controla", value="Alfa S.A."),
        },
        steps=(
            InferenceStep(step_id="s", operation="rule_application", premises=("e",), conclusion_id="c", rule_id="R2a_prop_implica_control", rule_version="1.0"),
        ),
        conclusion_id="c",
    )
    res_bad_r2a = evaluate_response(g_bad_r2a, evidence={"S2": docs["S2"]}, domain_profile=prof, verifier_registry=reg)
    assert res_bad_r2a.assessment == "refuted"
    assert any("VARIABLE_BINDING_MISMATCH" in c.reasons for c in res_bad_r2a.completed_checks)

    # 2. Inverted endpoint in native RelationVerifier (Alfa->Beta->Gamma claiming Gamma->Alfa) MUST be refuted
    ref_s3 = EvidenceRef(document_id="S3", revision="rev-0", start=0, end=len(docs["S3"].text), quote=docs["S3"].text)
    g_bad_rel = JustificationGraph(
        claims={
            "d1": ClaimNode(claim_id="d1", kind="extraction", subject="Alfa S.A.", predicate="controla", value="Beta S.L.", evidence=(ref_s2,)),
            "d2": ClaimNode(claim_id="d2", kind="extraction", subject="Beta S.L.", predicate="controla", value="Gamma S.L.", evidence=(ref_s3,)),
            "cg": ClaimNode(claim_id="cg", kind="derivation", subject="Gamma S.L.", predicate="controla", value="Alfa S.A."),
        },
        steps=(
            InferenceStep(
                step_id="sr",
                operation="relation_composition",
                premises=("d1", "d2"),
                conclusion_id="cg",
                rule_id="R2_controla_transitiva",
                rule_version="1.0",
                obligations=(VerificationObligation(obligation_id="ob_rel", family="relations"),),
            ),
        ),
        conclusion_id="cg",
    )
    res_bad_rel = evaluate_response(g_bad_rel, evidence={"S2": docs["S2"], "S3": docs["S3"]}, domain_profile=prof, verifier_registry=reg)
    assert res_bad_rel.assessment == "refuted"
    assert any("RELATION_ENDPOINT_MISMATCH" in c.reasons for c in res_bad_rel.completed_checks)

    # 3. Grounding rejects reversed entities / fabricated quote / Spanish non-matching conditional
    assert not pd.anchor_premise_in_docs(
        {"doc_id": "S2", "subject": "Beta S.L.", "predicate": "propietaria_100", "value": "Alfa S.A.", "verbatim_quote": docs["S2"].text},
        {"S2": docs["S2"]},
    )["grounded"]
    assert not pd.anchor_premise_in_docs(
        {"doc_id": "S4", "subject": "Contrato", "predicate": "preaviso_no_renovacion_enviado", "value": True, "verbatim_quote": docs["S4"].text},
        {"S4": docs["S4"]},
    )["grounded"]
    assert pd.parse_spanish_dmy_date("05/12/2026") == "2026-12-05"

    # 4. Raw DEFAULT_VERIFIER_REGISTRY without POC verifier adapters exposes exact engine limitations on C1a and C3b
    raw_by_id = {r["claim_id"]: r for r in pd.run_mode_a_all(use_default_registry=True)}
    assert raw_by_id["C1a"]["assessment"] == "inconclusive"
    assert raw_by_id["C3b"]["assessment"] == "refuted"
