"""Unit-profile isolation: STANDARD_UNITS keeps its historical fingerprint; Spanish labels
live in an explicit profile that the derivation planner selects when it applies."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from claimlayer.semantic.models import SourceDocument, digest
from claimlayer.v4 import (
    SPANISH_UNIT_ALIASES,
    SPANISH_UNITS,
    STANDARD_UNITS,
    UNIT_PROFILES,
    DomainProfile,
    OperationCatalog,
    PlannerCallResult,
    UnmappedProposal,
    detect_text_language,
    resolve_unit_profile,
    verify_claim,
    verify_plan,
)

# Historical values. Do NOT update these to make a test pass: a change here is a
# compatibility break for every frozen vocabulary / preregistration built on these units.
HISTORICAL_STANDARD_UNITS_DIGEST = "8fb557d46f66b4904114c9c1b18b3377c9224170f253462027e5ec434e1703d2"
# Units in force when the live 14/14 derivation benchmark ran (2026-11-20/21).
BENCHMARK_SPANISH_UNITS_DIGEST = "edcc833daa7c074c05e1e2ae5d2916670d40460e6695f136ab185e80ad9bc17f"


def test_standard_units_keep_historical_fingerprint_and_no_spanish_labels() -> None:
    assert digest(STANDARD_UNITS.describe()) == HISTORICAL_STANDARD_UNITS_DIGEST
    for labels in SPANISH_UNIT_ALIASES.values():
        for label in labels:
            with pytest.raises(UnmappedProposal):
                STANDARD_UNITS.lookup(label)


def test_spanish_profile_only_adds_labels() -> None:
    assert digest(SPANISH_UNITS.describe()) == BENCHMARK_SPANISH_UNITS_DIGEST
    std = {u["name"]: u for u in STANDARD_UNITS.describe()}
    es = {u["name"]: u for u in SPANISH_UNITS.describe()}
    assert std.keys() == es.keys()
    for name, u in std.items():
        assert (es[name]["dimension"], es[name]["factor"]) == (u["dimension"], u["factor"])
        assert tuple(es[name]["aliases"][: len(u["aliases"])]) == tuple(u["aliases"])
    assert SPANISH_UNITS.lookup("días").name == "calendar_days"
    assert SPANISH_UNITS.lookup("dólares").name == "USD"
    assert SPANISH_UNITS.lookup("dollars").name == "USD"  # English labels still recognised
    assert UNIT_PROFILES == {"standard": STANDARD_UNITS, "es": SPANISH_UNITS}


def test_with_aliases_is_non_mutating_and_rejects_unknown_units() -> None:
    before = digest(STANDARD_UNITS.describe())
    STANDARD_UNITS.with_aliases({"USD": ("pavos",)})
    assert digest(STANDARD_UNITS.describe()) == before
    with pytest.raises(ValueError):
        STANDARD_UNITS.with_aliases({"no_such_unit": ("x",)})


def test_language_detection_and_profile_resolution() -> None:
    assert detect_text_language("La cita fue el miércoles.", "Hoy es viernes y quedé con ella dos días antes.") == "es"
    assert detect_text_language("The meeting was on Wednesday.", "Today is Friday and we met two days before.") == "en"
    assert resolve_unit_profile("auto", "Quedan 750 dólares.", ["Techo: 2.400 dólares."])[0] == "es"
    assert resolve_unit_profile("auto", "The remaining margin is 750 USD.", ["Cap: 2,400 USD."])[0] == "standard"
    assert resolve_unit_profile("standard", "Quedan 750 dólares.", [])[0] == "standard"
    assert resolve_unit_profile("es", "The margin is 750 USD.", [])[0] == "es"
    with pytest.raises(ValueError):
        resolve_unit_profile("fr", "x", [])


class _CapturingPlanner:
    def __init__(self, proposal: dict[str, Any]) -> None:
        self.proposal = proposal
        self.seen_unit_digests: list[str] = []

    @property
    def identity(self) -> Mapping[str, Any]:
        return {"provider": "test_double", "model": "capturing-planner"}

    def plan(self, *, sources: Sequence[SourceDocument], claim: str, catalog: OperationCatalog,
             domain_profile: DomainProfile, repair_feedback: Any = None) -> PlannerCallResult:
        self.seen_unit_digests.append(digest(domain_profile.effective_unit_registry().describe()))
        return PlannerCallResult(ok=True, proposal=self.proposal, latency_ms=1.0,
                                 usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})


_DOCS = {
    "contrato": "Techo presupuestario de la Fase Beta: 2.400 dólares.",
    "factura": "Importe facturado en la Fase Beta: 1.650 dólares.",
}
_CLAIM = "Quedan 750 dólares disponibles en la Fase Beta."


def _usd_plan() -> dict[str, Any]:
    def prem(cid: str, doc: str, event: str, value: int) -> dict[str, Any]:
        return {"claim_id": cid, "document_id": doc, "quote": _DOCS[doc], "entity": "Fase Beta",
                "event": event, "predicate": event.replace(" ", "_"), "value": value,
                "unit": "dólares", "direction": "neutral", "polarity": 1}
    return {
        "claim_target": {"subject": "Fase Beta", "predicate": "saldo_disponible", "value": 750,
                         "unit": "dólares", "polarity": 1},
        "premises": [prem("p_cap", "contrato", "techo presupuestario", 2400),
                     prem("p_inv", "factura", "importe facturado", 1650)],
        "steps": [{"step_id": "s1", "operation": "sub", "premises": ["p_cap", "p_inv"], "conclusion_id": "c1"}],
        "derived_conclusion_id": "c1",
        "missing_premises": [], "assumptions": [], "suggested_rules": [],
    }


def test_planner_receives_spanish_profile_for_spanish_input_and_it_matters() -> None:
    planner = _CapturingPlanner(_usd_plan())
    r = verify_claim(_DOCS, _CLAIM, planner=planner)
    assert planner.seen_unit_digests == [BENCHMARK_SPANISH_UNITS_DIGEST]
    assert r.telemetry.unit_profile == "es"
    assert r.telemetry.unit_registry_fingerprint == BENCHMARK_SPANISH_UNITS_DIGEST
    assert r.epistemic_verdict == "derived"
    assert r.computed_conclusion is not None and r.computed_conclusion["value"] == 750
    assert r.computed_conclusion["unit"] == "USD"

    # Same proposal under the historical STANDARD profile: 'dólares' is not a known label,
    # so the derivation must not be accepted.
    planner_std = _CapturingPlanner(_usd_plan())
    r_std = verify_claim(_DOCS, _CLAIM, planner=planner_std, unit_profile="standard")
    # The unit defect triggers the bounded repair retry; every attempt uses the same profile.
    assert planner_std.seen_unit_digests and set(planner_std.seen_unit_digests) == {HISTORICAL_STANDARD_UNITS_DIGEST}
    assert r_std.telemetry.unit_profile == "standard"
    assert r_std.epistemic_verdict != "derived"
    assert r_std.operational_routing.status != "PASS"


def test_explicit_domain_profile_wins_and_offline_verify_plan_records_profile() -> None:
    explicit = DomainProfile(profile_id="acme", unit_registry=STANDARD_UNITS)
    planner = _CapturingPlanner(_usd_plan())
    r = verify_claim(_DOCS, _CLAIM, planner=planner, domain_profile=explicit)
    assert planner.seen_unit_digests and set(planner.seen_unit_digests) == {HISTORICAL_STANDARD_UNITS_DIGEST}
    assert r.telemetry.unit_profile == "domain_profile:acme"

    r_off = verify_plan(_DOCS, _CLAIM, _usd_plan())
    assert r_off.telemetry.unit_profile == "es"
    assert r_off.epistemic_verdict == "derived"
    assert r_off.to_dict()["telemetry"]["unit_registry_fingerprint"] == BENCHMARK_SPANISH_UNITS_DIGEST
