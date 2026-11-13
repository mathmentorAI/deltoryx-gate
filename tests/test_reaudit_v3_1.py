"""Batería adversarial de la re-auditoría de claim-layer v3.1.0 (13 de noviembre de 2026).

Contenido: 137 casos parametrizados más 1 de rendimiento opcional.

- 70 casos originales de la auditoría de la v3.0 (ids ``orig-*``), con las mismas entradas.
- 67 variantes cercanas: el mismo mecanismo de fallo con otro ejemplo (ids ``var-*``).
- 13 de esas variantes son regresiones: la v3.0 las resolvía bien y la v3.1.0 no
  (ids ``var-regresion-*``). No son casos adicionales: están dentro de las 67.

Cada test afirma el comportamiento CORRECTO, no el actual. Resultado esperado hoy:

- v3.1.0 (``4a07392``): 77 pasan, 60 fallan (3 originales + 57 variantes).
- v3.0 (``0e17d3e``): las 13 regresiones pasan (``pytest -k regresion``). Otros casos dan
  error en la v3.0 porque usan API que solo existe en la v3.1.0 (``condition``,
  ``default_policy``, internos de ``receipt``); eso es esperado.

Uso, desde la raíz del repositorio (``pyproject.toml`` ya pone ``src`` y ``.`` en el path):

    python -m pytest tests/test_reaudit_v3_1.py -q
    python -m pytest tests/test_reaudit_v3_1.py -q -k orig        # solo originales
    python -m pytest tests/test_reaudit_v3_1.py -q -k regresion   # solo regresiones
    CLAIMLAYER_RUN_SLOW=1 python -m pytest tests/test_reaudit_v3_1.py -q -k perf

Criterio para cerrar un caso: que pase sin añadir listas de palabras ni condiciones
específicas del ejemplo. Cada variante señala un mecanismo, no una cadena.
"""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

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

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def st(v) -> str:
    """Resumen estable de un veredicto: estado + códigos de incidencia."""
    codes = sorted({i.code for i in v.issues})
    return v.status + ("" if not codes else " " + ",".join(codes))


NOTPASS = lambda g: not g.startswith("PASS")  # noqa: E731
PASS = lambda g: g.startswith("PASS")  # noqa: E731
ESC = lambda g: g.startswith("ESCALATE")  # noqa: E731
RETRY = lambda g: g.startswith("RETRY")  # noqa: E731
IS_FALSE = lambda g: g == "False"  # noqa: E731
IS_TRUE = lambda g: g == "True"  # noqa: E731


def one(src_text, value, quote, unit=None, policy=None, subject="S", predicate="p",
        extra=(), tier="authoritative", **kw):
    led = Ledger(sources=[Source(id="d", text=src_text, tier=tier), *extra])
    a = Assertion(subject=subject, predicate=predicate, value=value, source_id="d",
                  quote=quote, unit=unit, **kw)
    return verify([a], led, policy)


def mcp_server(sources, **kw):
    m = MCPVerificationServer(Ledger(sources=sources), **kw)
    s = MCPSession()
    m.handle_envelope({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, s)
    return m, s


def mcp_call(m, s, name, args) -> str:
    r = m.handle_envelope(
        {"jsonrpc": "2.0", "id": 99, "method": "tools/call",
         "params": {"name": name, "arguments": args}}, s)[0]
    if "error" in r:
        return "REFUSED " + r["error"]["message"][:80]
    res = r["result"]
    if res.get("isError"):
        return "REFUSED " + res["content"][0]["text"][:80]
    sc = res["structuredContent"]
    if sc.get("status"):
        return sc["status"] + " " + ",".join(sorted({i["code"] for i in sc.get("issues", [])}))
    return "REGISTERED" if sc.get("registered") else str(sc)[:80]


def forge_pass(d: dict, **over) -> dict:
    """Reescribe un veredicto como PASS y recalcula el recibo SIN clave (v3.1.0)."""
    from claimlayer import receipt as R

    f = copy.deepcopy(d)
    f.update(status="PASS", allowed_to_proceed=True, requires_human=False,
             issues=[], conflicts=[], confidence=1.0)
    f["assertion_scores"] = {k: 1.0 for k in (f.get("assertion_scores") or {"a1": 1.0})}
    f.update(over)
    rc = f["receipt"]
    rc["hmac_authenticated"] = False
    env = R._build_signature_envelope(
        scheme=R.RECEIPT_SCHEME, merkle_root=rc["merkle_root"],
        leaf_count=len(rc["leaf_hashes"]), sources_digest=rc["sources_digest"],
        policy_digest=rc["policy_digest"], ledger_digest=rc.get("ledger_digest", ""),
        status=f["status"], allowed_to_proceed=f["allowed_to_proceed"],
        requires_human=f["requires_human"], agent_id=f["agent_id"], policy_id=f["policy_id"],
        confidence=f["confidence"], as_of=f["as_of"], issues=f["issues"],
        conflicts=f["conflicts"], assertion_scores=f["assertion_scores"])
    sig, _ = R._sign_payload(env, signing_key=None)
    rc["signature"] = sig
    f["verdict_id"] = "vrd-" + sig[:12]
    return f


def gate_sandbox(tmp: Path, extra_docs: dict[str, str] | None = None) -> Path:
    from examples.harness2_securitization.run_harness2_ab import stage_self_contained_harness

    staged = stage_self_contained_harness(tmp / "hs")
    sb = tmp / "ws"
    (sb / "documents").mkdir(parents=True)
    shutil.copytree(staged / "scripts", sb / "scripts")
    docs = {
        "executed_indenture.txt":
            "Section 2.04: Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.\n"
            "Section 7.01: Failure to pay interest continuing for 5 business days is an Event of Default.\n",
        "draft_offering-memorandum.txt":
            "Section 2.04: Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum.\n"
            "Section 7.01: Failure to pay interest continuing for 30 days is an Event of Default.\n",
    }
    docs.update(extra_docs or {})
    for name, text in docs.items():
        (sb / "documents" / name).write_text(text, encoding="utf-8")
    return sb


def gate(sb: Path, *args) -> tuple[int, dict]:
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "scripts/claimlayer_gate.py", *args], cwd=sb, env=env,
                       capture_output=True, text=True)
    try:
        return r.returncode, json.loads(r.stdout)
    except Exception:
        return r.returncode, {"raw": (r.stdout + r.stderr)[-300:]}


def gate_out(r) -> str:
    codes = sorted({i["code"] for i in r[1].get("issues", [])})
    return f"exit {r[0]} {r[1].get('status')} {codes}"


# Datos compartidos ----------------------------------------------------------

A8Q = ("The Class A Notes bear interest at 5.10% per annum and the Class C Notes bear "
       "interest at 6.75% per annum.")
REV = ("Interest accrues at 6.75% per annum on the Class C Notes and at 5.10% per annum "
       "on the Class A Notes.")
IND = Source(id="indenture", tier="authoritative", text=(
    "Section 2.04: Interest Rates. Class C Notes bear interest at 6.75% per annum.\n"
    "Section 5.04: Priority of Payments. Reserve Account replenishment is paid at Step 13, "
    "junior to Class B and Class C principal.\n"
    "Section 9.01: Clean-Up Call. Optional redemption is permitted at 10% pool factor."))
OM = Source(id="offering_memorandum", tier="standard", text=(
    "Section 2.04: Interest Rates. Class C Notes bear interest at 6.50% per annum.\n"
    "Section 5.04: Priority of Payments. Reserve Account replenishment is paid at Step 6, "
    "senior to Class B and Class C principal.\n"
    "Section 9.01: Clean-Up Call. Optional redemption is permitted at 10% pool factor."))
CC_675 = dict(subject="Class_C", predicate="interest_rate", value="6.75%", source_id="indenture",
              quote="Class C Note interest rate shall be 6.75% per annum.")
CC_650 = dict(subject="Class_C", predicate="interest_rate", value="6.50%", source_id="supplement",
              quote="Class C Note interest rate shall be 6.50% per annum.")


def peer_ledger(t2="Class C Note interest rate shall be 6.50% per annum."):
    return Ledger(sources=[
        Source(id="indenture", text="Class C Note interest rate shall be 6.75% per annum.",
               tier="authoritative"),
        Source(id="supplement", text=t2, tier="authoritative")])


def msa_ledger(msa_text="Payment terms are 45 days. Governing law is New York.", extra_facts=()):
    return Ledger(
        sources=[Source(id="msa_base", text=msa_text, tier="standard"),
                 Source(id="amendment_1", text="Section 4 is amended: Payment terms are 30 days.",
                        tier="authoritative", supersedes=["msa_base"])],
        facts=[FactRecord(subject="ACME", predicate="payment_terms", value="30 days",
                          source_id="amendment_1", quote="Payment terms are 30 days."),
               *extra_facts])


def dp(ptxt, stxt):
    p = Source(id="P", text=ptxt, tier="authoritative")
    s = Source(id="S", text=stxt, tier="standard")
    pol, divs = build_comparison_policy(p, s)
    return p, s, pol, divs


def req_preds(ptxt, stxt) -> str:
    return str(list(dp(ptxt, stxt)[2].required_predicates))


# ---------------------------------------------------------------------------
# 1. Casos originales de la auditoría (70)
# ---------------------------------------------------------------------------

ORIG: list[tuple[str, Callable[[str], bool], Callable[[Path], object]]] = []


def orig(cid, want):
    def deco(fn):
        ORIG.append((cid, want, fn))
        return fn
    return deco


# A. Falsos PASS en v3.0 (13) -------------------------------------------------
orig("A1", NOTPASS)(lambda t: one(
    "Section 30. Reports. The cure period for payment defaults is 5 Business Days.",
    "30", "30", predicate="cure_period_days"))
orig("A2", NOTPASS)(lambda t: one(
    "The Servicer shall deliver the monthly report. Unrelated: the lock-up period is 45 days.",
    "45 days", "The Servicer shall deliver the monthly report.", predicate="report_deadline"))
orig("A3", NOTPASS)(lambda t: one(
    "Maturity date is 2027-06-30. The Closing Date is 2026-01-15.",
    "2026-01-15", "Maturity date is", predicate="maturity_date"))
orig("A4", NOTPASS)(lambda t: one(
    "In no event shall the Servicer be terminated without cause.",
    "permitted", "the Servicer be terminated without cause", predicate="termination_without_cause"))
orig("A5", NOTPASS)(lambda t: one(
    "Notice must be given within twenty-five (25) days.",
    "5 days", "within twenty-five (25) days", predicate="notice_period"))
orig("A6", NOTPASS)(lambda t: one(
    "The cure period is one hundred twenty (120) days.",
    "20 days", "one hundred twenty (120) days", predicate="cure_period"))
orig("A7", NOTPASS)(lambda t: one(
    "Failure to pay interest continuing for 5 business days is an Event of Default.",
    "5 calendar days", "continuing for 5 business days", predicate="cure_period"))
orig("A8", NOTPASS)(lambda t: one(A8Q, "5.10%", A8Q, subject="Class_C_Notes",
                                  predicate="interest_rate"))
orig("A9", NOTPASS)(lambda t: one(
    "The Class C Notes have an aggregate principal amount of $50,000,049.",
    "$50,000,000", "aggregate principal amount of $50,000,049", predicate="principal"))
orig("A10", NOTPASS)(lambda t: one(
    "El tipo de interés será del 5,5% anual.", "5%", "El tipo de interés será del 5,5% anual.",
    predicate="tipo_interes"))
orig("A11", NOTPASS)(lambda t: one(
    "The advance rate shall not exceed 30% of Eligible Receivables.",
    "30", "shall not exceed 30%", predicate="cure_period_days"))
orig("A12", NOTPASS)(lambda t: one(
    "Net income changed by -5% year over year.", "5%", "changed by -5%",
    predicate="net_income_growth"))
orig("A13", NOTPASS)(lambda t: one(
    "Amounts in thousands. Settlement claim: 95,000.", "95,000", "Settlement claim: 95,000",
    predicate="settlement_usd", policy=Policy(numeric_bounds={"settlement_usd": (None, 1_000_000.0)})))

# B. Falsos RETRY / ESCALATE en v3.0 (19) --------------------------------------
orig("B1", PASS)(lambda t: one("Interest shall accrue at the rate of 5 per cent. per annum.",
                               "5%", "at the rate of 5 per cent. per annum", predicate="rate"))
orig("B1b", PASS)(lambda t: one("El interés será del 5 por ciento anual.", "5%",
                                "del 5 por ciento anual", predicate="rate"))
orig("B2", PASS)(lambda t: one("The Issuer will issue EUR 500,000,000 of Notes.",
                               "500,000,000 EUR", "issue EUR 500,000,000 of Notes",
                               predicate="issue_size"))
orig("B2b", PASS)(lambda t: one("The Issuer will issue USD 500,000,000 of Notes.",
                                "$500 million", "issue USD 500,000,000 of Notes",
                                predicate="issue_size"))
orig("B3", PASS)(lambda t: one("The Closing Date is January 15, 2026.", "2026-01-15",
                               "The Closing Date is January 15, 2026.", predicate="closing_date"))
orig("B4", PASS)(lambda t: one("The cure period is between 30 and 60 days.", "30 days",
                               "between 30 and 60 days", predicate="min_cure"))
# B5a: en la v3.0 el valor era "required"; se usa un valor textual presente en la cita.
orig("B5a", PASS)(lambda t: one("The Servicer shall make each pay-\nment on the Payment Date.",
                                "payment on the Payment Date",
                                "The Servicer shall make each payment on the Payment Date.",
                                predicate="payment"))
orig("B5b", PASS)(lambda t: one("The ﬁnal maturity is 10 years.", "10 years",
                                "The final maturity is 10 years.", predicate="maturity"))
orig("B5c", PASS)(lambda t: one("The non‑recourse amount is $5 million.", "$5 million",
                                "The non-recourse amount is $5 million.", predicate="amt"))
orig("B5d", PASS)(lambda t: one("Pay­ment terms are 30 days.", "30 days",
                                "Payment terms are 30 days.", predicate="terms"))
orig("B6", PASS)(lambda t: one("El importe nominal es de 1.250.000 euros.", "1250000 EUR",
                               "El importe nominal es de 1.250.000 euros.", predicate="nominal"))
orig("B7", PASS)(lambda t: one("The margin is 150 basis points.", "150 bps",
                               "The margin is 150 basis points.", predicate="margin"))
orig("B8", PASS)(lambda t: one("Holders of at least 66 2/3% of the Notes may direct the Trustee.",
                               "66.67%", "at least 66 2/3% of the Notes", predicate="threshold"))
orig("B9", PASS)(lambda t: one("The term is two weeks.", "14 days", "The term is two weeks.",
                               predicate="term"))
orig("B10", PASS)(lambda t: one("The facility is €1,5 million.", "€1.5 million",
                                "The facility is €1,5 million.", predicate="facility"))


def _b11(with_condition: bool):
    led = Ledger(sources=[Source(id="ind", tier="authoritative", text=(
        "A payment default has a cure period of 5 Business Days. "
        "A covenant default has a cure period of 30 days."))])
    k1 = {"condition": "payment default"} if with_condition else {}
    k2 = {"condition": "covenant default"} if with_condition else {}
    return verify([
        Assertion(subject="Issuer", predicate="cure_period", value="5 Business Days",
                  source_id="ind", quote="A payment default has a cure period of 5 Business Days.", **k1),
        Assertion(subject="Issuer", predicate="cure_period", value="30 days", source_id="ind",
                  quote="A covenant default has a cure period of 30 days.", **k2)], led)


orig("B11", PASS)(lambda t: _b11(False))   # tal como se envió en la auditoría
orig("B11c", PASS)(lambda t: _b11(True))   # con Assertion.condition (v3.1.0)


@orig("B12", PASS)
def _b12(t):
    led = Ledger(
        sources=[Source(id="a", tier="authoritative",
                        text="This Indenture shall be governed by the laws of the State of New York."),
                 Source(id="b", tier="authoritative", text="Governing law: New York.")],
        facts=[FactRecord(subject="Deal", predicate="governing_law", value="New York", source_id="b")])
    return verify([Assertion(subject="Deal", predicate="governing_law",
                             value="the laws of the State of New York", source_id="a",
                             quote="governed by the laws of the State of New York")], led)


@orig("B13", PASS)
def _b13(t):
    led = Ledger(
        sources=[Source(id="a", tier="authoritative", text="The margin is 1.50% per annum."),
                 Source(id="b", tier="authoritative", text="The margin is 150 bps.")],
        facts=[FactRecord(subject="Loan", predicate="margin", value="150 bps", source_id="b")])
    return verify([Assertion(subject="Loan", predicate="margin", value="1.50%", source_id="a",
                             quote="The margin is 1.50% per annum.")], led)


# C. Specification gaming, precedencia y conflictos (18) -----------------------
def _c1(subj, pred):
    return verify([Assertion(subject=subj, predicate=pred, value="45 days", source_id="msa_base",
                             quote="Payment terms are 45 days.")], msa_ledger())


orig("C1b", NOTPASS)(lambda t: _c1("ACME", "payment_term"))
orig("C1c", NOTPASS)(lambda t: _c1("ACME Corp", "payment_terms"))
orig("C2", ESC)(lambda t: verify([CC_675], peer_ledger(), Policy(required_source_ids=["indenture"])))
orig("C3b", ESC)(lambda t: verify([dict(CC_675, id="x"), dict(CC_650, id="x")], peer_ledger()))


@orig("C4", NOTPASS)
def _c4(t):
    pol, _ = build_comparison_policy(IND, OM)
    return verify([
        Assertion(subject="x", predicate="section_2_04", value="reviewed", source_id="indenture",
                  quote="Section 2.04"),
        Assertion(subject="x", predicate="section_5_04", value="no issues", source_id="indenture",
                  quote="Priority of Payments")], Ledger(sources=[IND, OM]), pol)


@orig("C5a", NOTPASS)
def _c5a(t):
    m, s = mcp_server([IND, OM])
    return mcp_call(m, s, "verify_proposal", {
        "assertions": [{"subject": "Class_C", "predicate": "interest_rate", "value": "9.99%",
                        "source_id": "indenture", "quote": "This sentence is not in the indenture."}],
        "policy": {"require_exact_quote": False, "require_quantity_match": False}})


@orig("C5b", NOTPASS)
def _c5b(t):
    m, s = mcp_server([IND, OM])
    r1 = mcp_call(m, s, "register_source", {
        "id": "amendment_fake", "text": "Class C Notes bear interest at 1.00% per annum.",
        "tier": "authoritative", "supersedes": ["indenture"]})
    if r1.startswith("REFUSED"):
        return r1
    return mcp_call(m, s, "verify_proposal", {"assertions": [{
        "subject": "Class_C", "predicate": "interest_rate", "value": "1.00%",
        "source_id": "amendment_fake", "quote": "Class C Notes bear interest at 1.00% per annum."}]})


@orig("C5c", NOTPASS)
def _c5c(t):
    m, s = mcp_server([Source(id="indenture", text="Class C Notes bear interest at 6.75% per annum.",
                              tier="authoritative")])
    r1 = mcp_call(m, s, "register_source", {
        "id": "indenture", "text": "Class C Notes bear interest at 2.00% per annum.",
        "tier": "authoritative"})
    if r1.startswith("REFUSED"):
        return r1
    return mcp_call(m, s, "verify_proposal", {"assertions": [{
        "subject": "Class_C", "predicate": "interest_rate", "value": "2.00%",
        "source_id": "indenture", "quote": "Class C Notes bear interest at 2.00% per annum."}]})


def _c6(conf):
    return verify([Assertion(subject="Class_C", predicate="rate", value="6.75%",
                             source_id="indenture",
                             quote="Class C Notes bear interest at 6.75% per annum.",
                             confidence=conf)], Ledger(sources=[IND]))


orig("C6nan", NOTPASS)(lambda t: _c6(float("nan")))
orig("C6hi", NOTPASS)(lambda t: _c6(7.0))


@orig("C6", lambda g: g == "same")
def _c6_same(t):
    # La confianza autodeclarada por el agente no debería decidir el enrutado.
    a, b = _c6(0.6).status, _c6(1.0).status
    return "same" if a == b else f"conf 0.6 -> {a}; conf 1.0 -> {b}"


def _turn2(subj):
    return [
        Assertion(subject=subj, predicate="section_2_04", value="6.50%",
                  source_id="draft_offering_memorandum",
                  quote="Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum."),
        Assertion(subject="Deal_Waterfall", predicate="section_5_04", value="Step 13",
                  source_id="executed_indenture_2026",
                  quote="Step 13 Reserve Account Replenishment up to the Required Reserve Amount "
                        "(junior to Class B and Class C principal)."),
        Assertion(subject="Deal_Waterfall", predicate="section_7_01", value="5 business days",
                  source_id="executed_indenture_2026",
                  quote="Failure to pay interest continuing for 5 business days constitutes an "
                        "Event of Default."),
        Assertion(subject="Deal_Waterfall", predicate="section_9_01", value="10%",
                  source_id="executed_indenture_2026",
                  quote="Servicer may redeem the Notes when Pool Balance falls to 10% of Initial "
                        "Pool Balance.")]


def _c7(no_facts, subj):
    from examples.harness2_securitization.replay_om_vs_indenture import build_securitization_case

    ledger, ind, omm = build_securitization_case()
    policy, _ = build_comparison_policy(ind, omm, policy_id="x", min_tier_rank=50)
    if no_facts:
        ledger = Ledger(sources=ledger.list_sources())
    return verify(_turn2(subj), ledger, policy)


orig("C7b", NOTPASS)(lambda t: _c7(True, "Deal_Waterfall"))
orig("C7c", NOTPASS)(lambda t: _c7(False, "Class_C_Notes"))


def _c8(expire_cited):
    led = Ledger(
        sources=[Source(id="indenture", text="Class C Note interest rate shall be 6.75% per annum.",
                        tier="authoritative"),
                 Source(id="supplement", text="Class C Note interest rate shall be 6.50% per annum.",
                        tier="authoritative")],
        facts=[FactRecord(subject="Class_C", predicate="interest_rate", value="6.50%",
                          source_id="supplement")])
    a = [Assertion(**CC_675)]
    v0 = led.verify(a, record=True)
    led.resolve_escalation(verdict_id=v0.verdict_id, subject="Class_C", predicate="interest_rate",
                           selected_value="6.75%", resolver="gc")
    led.add_source(Source(id="amendment_2",
                          text="Class C Note interest rate is amended to 7.00% per annum.",
                          tier="authoritative", supersedes=["indenture", "supplement"],
                          valid_from="2026-06-01"))
    led.add_fact(FactRecord(subject="Class_C", predicate="interest_rate", value="7.00%",
                            source_id="amendment_2"))
    if expire_cited:
        led.add_source(Source(id="indenture", text="Class C Note interest rate shall be 6.75% per annum.",
                              tier="authoritative", valid_until="2020-01-01"))
    return led.verify(a, as_of="2026-11-13")


orig("C8", NOTPASS)(lambda t: _c8(False))
orig("C8b", NOTPASS)(lambda t: _c8(True))
orig("C10", NOTPASS)(lambda t: verify(
    [Assertion(subject="Customer", predicate="payment_terms", value="60 days", source_id="mail",
               quote="Let's offer net 60 days informally.")],
    Ledger(sources=[Source(id="mail", text="Let's offer net 60 days informally.", tier="draft")])))
orig("C11b", ESC)(lambda t: one(
    "Refund requested is $9,000.", "$9,000", "Refund requested is $9,000.", subject="O",
    predicate="refund_total", policy=Policy(numeric_bounds={"refund_amount": (None, 5000.0)})))
orig("asof", NOTPASS)(lambda t: verify(
    [Assertion(subject="A", predicate="d", value="15%", source_id="p", quote="Discount is 15%.")],
    Ledger(sources=[Source(id="p", text="Discount is 15%.", tier="standard",
                           valid_until="2025-12-31")]), as_of="10/30/2026"))

# D. diff / build_comparison_policy (6) ---------------------------------------
orig("D1", lambda g: g != "[]")(lambda t: req_preds(
    "ARTICLE IV\nEVENTS OF DEFAULT\n4.01 Events of Default. Failure to pay interest continuing "
    "for 5 Business Days.\nARTICLE IX\n9.01 Clean-Up Call at 10%.",
    "ARTICLE IV\nEVENTS OF DEFAULT\n4.01 Events of Default. Failure to pay interest continuing "
    "for 30 days.\nARTICLE IX\n9.01 Clean-Up Call at 15%."))
orig("D1b", lambda g: g != "[]")(lambda t: req_preds(
    "Cláusula 4. Vencimiento anticipado: 5 días hábiles.\nCláusula 9. Amortización al 10%.",
    "Cláusula 4. Vencimiento anticipado: 30 días.\nCláusula 9. Amortización al 15%."))
orig("D2", lambda g: "section_12" in g)(lambda t: req_preds(
    "Section 1: Fees. Fee is $100.",
    "Section 1: Fees. Fee is $100.\nSection 12: Limitation of Liability. Liability is capped at $1."))
_D3P = ("Section 7.01: Events of Default. Failure to pay for 5 business days is a default, "
        "subject to\nSection 9.01 hereof.\nSection 9.01: Clean-Up Call at 10%.\n"
        "The Servicer may also act under\nSection 7.01 above.")
orig("D3", lambda g: "section_7_01" in g)(lambda t: req_preds(
    _D3P, _D3P.replace("5 business days", "30 days")))
orig("D4", lambda g: "missing" not in g)(lambda t: str([d.reason for d in dp(
    "Section 7.01: Events of Default. Failure to pay for 5 business days.",
    "Section 4.2: Events of Default. Failure to pay for 30 days.")[3]]))
orig("D6", lambda g: g != "[]")(lambda t: req_preds(
    "  Section 7.01: Events of Default. 5 business days.",
    "  Section 7.01: Events of Default. 30 days."))


# E. Recibos (5) --------------------------------------------------------------
def _e_base():
    led = Ledger(sources=[Source(id="s1", text="Limit is 100 days.", tier="authoritative")])
    v = verify([Assertion(subject="X", predicate="limit", value="999 days", source_id="s1",
                          quote="Limit is 100 days.")], led)
    return led, v.to_dict()


orig("E1", IS_FALSE)(lambda t: str(verify_receipt(forge_pass(_e_base()[1]))))


@orig("E2", IS_FALSE)
def _e2(t):
    g = copy.deepcopy(_e_base()[1])
    g["allowed_to_proceed"] = True
    g["agent_id"] = "someone_else"
    return str(verify_receipt(g))


@orig("E3", IS_FALSE)
def _e3(t):
    led, d = _e_base()
    h = copy.deepcopy(d)
    h["receipt"]["sources_digest"] = "0" * 64
    forged = forge_pass(h, status="RETRY", allowed_to_proceed=False, issues=d["issues"],
                        confidence=0.0, assertion_scores=d["assertion_scores"])
    return str(verify_receipt(forged, ledger=led))


@orig("E4", IS_FALSE)
def _e4(t):
    from claimlayer.receipt import compute_merkle_root
    return str(compute_merkle_root(["a", "b", "c"]) == compute_merkle_root(["a", "b", "c", "c"]))


@orig("E5", IS_FALSE)
def _e5(t):
    m1 = Ledger(sources=[Source(id="s1", text="Limit is 100 days.", tier="authoritative")])
    m2 = Ledger(sources=[Source(id="s1", text="Limit is 100 days.", tier="authoritative",
                                metadata={"note": "x"})],
                facts=[FactRecord(subject="X", predicate="limit", value="100 days", source_id="s1")])
    aa = [Assertion(subject="X", predicate="limit", value="100 days", source_id="s1",
                    quote="Limit is 100 days.")]
    return str(verify(aa, m1).receipt.signature == verify(aa, m2).receipt.signature)


# F, L, N, W, rendimiento (9) ---------------------------------------------------
@orig("F1", lambda g: g.endswith("triage=OPEN"))
def _f1(t):
    led = Ledger(
        sources=[Source(id="i", text="Rate A is 6.75%. Rate B is 3.00%.", tier="authoritative"),
                 Source(id="j", text="Rate A is 6.50%. Rate B is 2.00%.", tier="authoritative")],
        facts=[FactRecord(subject="A", predicate="rate", value="6.50%", source_id="j"),
               FactRecord(subject="B", predicate="rate", value="2.00%", source_id="j")])
    v = led.verify([
        Assertion(subject="A", predicate="rate", value="6.75%", source_id="i", quote="Rate A is 6.75%."),
        Assertion(subject="B", predicate="rate", value="3.00%", source_id="i", quote="Rate B is 3.00%.")],
        record=True)
    led.resolve_escalation(verdict_id=v.verdict_id, subject="A", predicate="rate", selected_value="6.75%")
    return f"{v.status} conflicts={len(v.conflicts)} triage={led.get_verdict_entry(v.verdict_id)['triage_state']}"


def _saved_pct(prior_pass: bool) -> str:
    led = Ledger(sources=[Source(id="s1", text="Limit is 100 days.", tier="authoritative")])
    bad = [Assertion(subject="X", predicate="limit", value="999 days", source_id="s1",
                     quote="Limit is 100 days.")]
    good = Assertion(subject="X", predicate="limit", value="100 days", source_id="s1",
                     quote="Limit is 100 days.")
    for _ in range(9):
        led.verify(bad, record=True)
    if prior_pass:
        led.verify([good], record=True)
    led.verify([Assertion(**{**good.to_dict(), "confidence": 0.5})], record=True)
    return str(led.get_metrics()["human_supervision_saved_pct"])


# Nueve reintentos fallidos no son supervisión ahorrada.
orig("F2", lambda g: float(g) < 60)(lambda t: _saved_pct(False))


@orig("F3", lambda g: g == "ok")
def _f3(t):
    from claimlayer.models import normalize_entity_key, normalize_key
    ok = (normalize_key("Заёмщик") != "" and normalize_key("東京電力") != ""
          and normalize_entity_key("the Servicer") == normalize_entity_key("Servicer"))
    return "ok" if ok else "fail"


@orig("L1", IS_TRUE)
def _l1(t):
    a = Ledger(db_path=t / "l.sqlite")
    b = Ledger(db_path=t / "l.sqlite")
    a.add_source(Source(id="s1", text="Limit is 100 days.", tier="authoritative"))
    return str(b.get_source("s1") is not None)


def _n1(second_copy_text):
    led = Ledger(
        sources=[Source(id="om_v1", tier="authoritative", text="Rate is 6.50%."),
                 Source(id="om_v2", tier="authoritative", text=second_copy_text),
                 Source(id="ind", tier="authoritative", text="Rate is 6.75%.")],
        facts=[FactRecord(subject="C", predicate="rate", value="6.50%", source_id="om_v1", confidence=0.6),
               FactRecord(subject="C", predicate="rate", value="6.50%", source_id="om_v2", confidence=0.6)])
    v = verify([Assertion(subject="C", predicate="rate", value="6.75%", source_id="ind",
                          quote="Rate is 6.75%.", confidence=0.6)], led)
    return f"{v.conflicts[0].competing_support:.2f}"


# Dos copias del mismo borrador no corroboran.
orig("N1", lambda g: g == "0.60")(lambda t: _n1("Rate is 6.50%."))


@orig("W1", lambda g: "authoritative" not in g)
def _w1(t):
    sb = gate_sandbox(t, {"side_letter_unsigned_v0.txt": "Random unsigned side letter."})
    _, prep = gate(sb, "prepare", "documents")
    return str({x["id"]: x["tier"] for x in prep["sources_registered"]
                if x["id"] == "side_letter_unsigned_v0.txt"})


_BAD_PROPOSAL = {"assertions": [{"subject": "Class_C_Notes", "predicate": "whatever", "value": "6.50%",
                                 "source_id": "draft_offering-memorandum.txt", "quote": "invented quote"}]}


@orig("W2", lambda g: g.startswith("exit 2"))
def _w2(t):
    sb = gate_sandbox(t)
    gate(sb, "prepare", "documents")
    (sb / "p.json").write_text(json.dumps(_BAD_PROPOSAL))
    return gate_out(gate(sb, "verify", "p.json"))


@orig("W3", lambda g: not g.startswith("exit 0"))
def _w3(t):
    sb = gate_sandbox(t)
    gate(sb, "prepare", "documents")
    (sb / "p.json").write_text(json.dumps(_BAD_PROPOSAL))
    pol_file = sb / ".claimlayer" / "policy.json"
    pol = json.loads(pol_file.read_text())
    pol.update(required_predicates=[], required_source_ids=[], min_tier_rank=0,
               require_exact_quote=False, require_quantity_match=False)
    pol_file.write_text(json.dumps(pol))
    return gate_out(gate(sb, "verify", "p.json"))


@orig("PERF", lambda g: float(g) < 5.0)
def _perf_facts(t):
    led = Ledger(sources=[Source(id=f"doc{i}", text="x", tier="authoritative") for i in range(5)])
    t0 = time.perf_counter()
    for k in range(20000):
        led.add_fact(FactRecord(subject=f"subj{k}", predicate="cap", value=f"${k},000",
                                source_id=f"doc{k % 5}"))
    return f"{time.perf_counter() - t0:.2f}"


# ---------------------------------------------------------------------------
# 2. Variantes cercanas (67, de ellas 13 regresiones)
# ---------------------------------------------------------------------------

VAR: list[tuple[str, Callable[[str], bool], Callable[[Path], object]]] = []
REGRESSIONS = {"A8v2", "B15a", "B15b", "B16", "B17a", "B17b", "B17c", "B19a",
               "C1fp", "C2fp", "C7fp", "C11fp", "C3v"}


def var(cid, want):
    def deco(fn):
        VAR.append((cid, want, fn))
        return fn
    return deco


# 2.1 Vínculo cita-afirmación y cantidades (34) --------------------------------
var("A1v", NOTPASS)(lambda t: one(
    "The terms are set out in Schedule 30 hereto. The cure period for payment defaults is 5 Business Days.",
    "30", "set out in Schedule 30 hereto", predicate="cure_period_days"))
_NEG = "In no event shall the Servicer be terminated without cause."
var("A4v1", NOTPASS)(lambda t: one(_NEG, "Servicer may be terminated without cause", _NEG,
                                   predicate="termination_without_cause"))
var("A4v2", NOTPASS)(lambda t: one(
    "The Borrower may prepay the Loans at any time, except during the Lock-Up Period.",
    "may prepay at any time", "The Borrower may prepay the Loans at any time", predicate="prepayment"))
var("A4v3", NOTPASS)(lambda t: one(
    "In no event (see Section 4.01) shall the Servicer be terminated without cause.",
    "termination without cause is available", "the Servicer be terminated without cause",
    predicate="termination"))
var("A5v", NOTPASS)(lambda t: one("Notice must be given within thirty-five (35) days.", "5 days",
                                  "within thirty-five (35) days", predicate="notice"))
var("A7v", NOTPASS)(lambda t: one("continuing for 5 Business Days is a default.", "5 days",
                                  "continuing for 5 Business Days is a default.", predicate="cure"))
var("A8v1", NOTPASS)(lambda t: one(REV, "5.10%", REV, subject="Class_C_Notes", predicate="interest_rate"))
var("A8v2", PASS)(lambda t: one(REV, "6.75%", REV, subject="Class_C_Notes", predicate="interest_rate"))
var("A8v3", NOTPASS)(lambda t: one(A8Q, "5.10%", A8Q, subject="Junior_Notes", predicate="interest_rate"))
_TWO = "The Servicer shall deliver the monthly report within 5 days. The lock-up period is 45 days."
var("A2v", NOTPASS)(lambda t: one(_TWO, "45 days", _TWO, subject="Deal", predicate="report_deadline"))
var("A3v", NOTPASS)(lambda t: one(
    "Maturity date is 2027-06-30. The Closing Date is 2026-01-15.", "2026-01-15",
    "Maturity date is 2027-06-30. The Closing Date is 2026-01-15.", predicate="maturity_date"))
var("A10v1", NOTPASS)(lambda t: one("El precio es de 12.500 euros.", "12.5 EUR",
                                    "El precio es de 12.500 euros.", predicate="precio"))
var("A10v2", PASS)(lambda t: one("El precio es de 12.500 euros.", "12500 EUR",
                                 "El precio es de 12.500 euros.", predicate="precio"))
_TS = ("Summary financial data (in thousands): Revenue $125. Other terms follow. "
       "The late fee is $250 per day.")
var("A13v1", PASS)(lambda t: one(_TS, "$250", "The late fee is $250 per day.", predicate="late_fee"))
var("A13v2", NOTPASS)(lambda t: one(_TS, "$250,000", "The late fee is $250 per day.", predicate="late_fee"))
var("A13v3", NOTPASS)(lambda t: one(
    "The breach resulted in millions of dollars of losses. The administrative fee is $250.",
    "$250,000,000", "The administrative fee is $250.", predicate="fee"))
var("B14a", PASS)(lambda t: one("The Issuer has a 30-day cure period.", "30 days",
                                "The Issuer has a 30-day cure period.", predicate="cure"))
var("B14b", PASS)(lambda t: one("The Issuer has a thirty (30)-day cure period.", "30 days",
                                "The Issuer has a thirty (30)-day cure period.", predicate="cure"))
var("B14c", PASS)(lambda t: one("The Notes have a ten-year term.", "10 years",
                                "The Notes have a ten-year term.", predicate="term"))
var("B15a", PASS)(lambda t: one("Cure Period - 30 days", "30 days", "Cure Period - 30 days",
                                predicate="cure", extra=[Source(id="z", text="x", tier="standard")]))
var("B15b", PASS)(lambda t: one("Late fee — $500 per occurrence.", "$500",
                                "Late fee — $500 per occurrence.", predicate="fee"))
var("B16", PASS)(lambda t: one(
    "The Supplier may terminate on payment of a fee of $500 and 30 days notice.", "$500",
    "payment of a fee of $500 and 30 days notice", predicate="fee"))
_C675 = "The Class C 6.75% Notes due 2030 are subordinated."
var("B17a", PASS)(lambda t: one(_C675, "6.75%", _C675, subject="Class_C_Notes", predicate="coupon"))
var("B17b", NOTPASS)(lambda t: one(_C675, "75%", _C675, subject="Class_C_Notes", predicate="coupon"))
var("B17c", PASS)(lambda t: one("The Tranche B 500 million facility is available.", "500 million",
                                "The Tranche B 500 million facility is available.", predicate="size"))
var("B18", PASS)(lambda t: one("The cap is five million dollars.", "$5 million",
                               "The cap is five million dollars.", predicate="cap"))
var("B19a", PASS)(lambda t: one("The Issuer will issue EUR500,000,000 Notes.", "EUR 500,000,000",
                                "The Issuer will issue EUR500,000,000 Notes.", predicate="size"))
var("B19b", PASS)(lambda t: one("The facility amount is USD1.5bn.", "$1.5 billion",
                                "The facility amount is USD1.5bn.", predicate="size"))
var("B20", PASS)(lambda t: one("El precio es de 1.250,50 €.", "1250.50 EUR",
                               "El precio es de 1.250,50 €.", predicate="precio"))
var("B21a", PASS)(lambda t: one("Executed this 15th day of January, 2026.", "2026-01-15",
                                "Executed this 15th day of January, 2026.", predicate="signing_date"))
var("B21b", PASS)(lambda t: one("Fecha de cierre: 15/01/2026.", "2026-01-15",
                                "Fecha de cierre: 15/01/2026.", predicate="cierre"))
var("B22", PASS)(lambda t: one("The margin is 6M Term SOFR plus 2.50%.", "6M Term SOFR + 2.50%",
                               "The margin is 6M Term SOFR plus 2.50%.", predicate="margin"))
var("B23", PASS)(lambda t: one("The Borrower shall maintain a ratio of 1.25 to 1.00.", "1.25x",
                               "maintain a ratio of 1.25 to 1.00", predicate="dscr"))
var("B24", PASS)(lambda t: one("La comisión es del 0,25 % anual.", "0.25%",
                               "La comisión es del 0,25 % anual.", predicate="comision"))


# 2.2 Gaming, precedencia y conflictos (19) ----------------------------------
def _c1v(pred):
    return verify([Assertion(subject="ACME", predicate=pred, value="45 days", source_id="msa_base",
                             quote="Payment terms are 45 days.")],
                  msa_ledger("Payment terms are 45 days. Governing law is New York. "
                             "Beta Inc payment terms are 45 days."))


var("C1v1", NOTPASS)(lambda t: _c1v("payment_period"))
var("C1v2", NOTPASS)(lambda t: _c1v("net_terms"))
var("C1v3", NOTPASS)(lambda t: _c1v("invoice_due_days"))


@var("C1fp", PASS)
def _c1fp(t):
    led = msa_ledger("Payment terms are 45 days. Governing law is New York. Beta Inc payment terms are 45 days.",
                     extra_facts=[FactRecord(subject="ACME", predicate="governing_law", value="New York",
                                             source_id="amendment_1")])
    return verify([Assertion(subject="Beta Inc", predicate="payment_terms", value="45 days",
                             source_id="msa_base", quote="Beta Inc payment terms are 45 days.")], led)


def _c2v(subj, pred, t2="Class C Note interest rate shall be 6.50% per annum."):
    return verify([Assertion(subject=subj, predicate=pred, value="6.75%", source_id="indenture",
                             quote="Class C Note interest rate shall be 6.75% per annum.")],
                  peer_ledger(t2))


var("C2v1", ESC)(lambda t: _c2v("Notes", "coupon"))
var("C2v2", ESC)(lambda t: _c2v("Deal", "class_c_rate"))
var("C2fp", PASS)(lambda t: _c2v("Class_C", "interest_rate",
                                 t2="Class A Note interest rate shall be 5.10% per annum."))


@var("C2v3", ESC)
def _c2v3(t):
    led = Ledger(sources=[
        Source(id="a", tier="authoritative", text="This Agreement is governed by the laws of the State of New York."),
        Source(id="b", tier="authoritative", text="This Agreement is governed by the laws of the State of Delaware.")])
    return verify([Assertion(subject="Agreement", predicate="governing_law", value="New York", source_id="a",
                             quote="This Agreement is governed by the laws of the State of New York.")], led)


_INDW = Source(id="indenture", tier="authoritative",
               text="Reserve Account replenishment is paid at Step 13, junior to Class B and Class C principal.")
_OMW = Source(id="om", tier="standard",
              text="Reserve Account replenishment is paid senior to Class B and Class C principal.")
var("C7v1", NOTPASS)(lambda t: verify(
    [Assertion(subject="Reserve_Account", predicate="priority",
               value="senior to Class B and Class C principal", source_id="om", quote=_OMW.text)],
    Ledger(sources=[_INDW, _OMW]), Policy(min_tier_rank=50)))
var("C4v2", NOTPASS)(lambda t: verify(
    [Assertion(subject="Reserve_Account", predicate="priority",
               value="senior to Class B and Class C principal", source_id="indenture", quote=_INDW.text)],
    Ledger(sources=[_INDW])))


@var("C7fp", PASS)
def _c7fp(t):
    led = Ledger(sources=[
        Source(id="msa", tier="authoritative", text="ACME shall pay all undisputed invoices within 30 days."),
        Source(id="schedule", tier="standard", text="ACME delivery lead time is 10 days from purchase order.")])
    return verify([Assertion(subject="ACME", predicate="delivery_lead_time", value="10 days",
                             source_id="schedule",
                             quote="ACME delivery lead time is 10 days from purchase order.")],
                  led, Policy(min_tier_rank=50))


@var("C4v1", NOTPASS)
def _c4v1(t):
    pol, _ = build_comparison_policy(IND, OM)
    return verify([
        Assertion(subject="x", predicate="section_2_04", value="Interest Rates", source_id="indenture",
                  quote="Section 2.04: Interest Rates."),
        Assertion(subject="x", predicate="section_5_04", value="Priority of Payments",
                  source_id="indenture", quote="Section 5.04: Priority of Payments.")],
        Ledger(sources=[IND, OM]), pol)


@var("PRUNE", lambda g: not g.endswith("PASS"))
def _prune(t):
    # El agente retira la afirmación rechazada en el siguiente intento.
    pol = Policy(min_tier_rank=50)
    led = Ledger(sources=[IND, OM])
    good = Assertion(subject="Deal", predicate="clean_up_call", value="10%", source_id="indenture",
                     quote="Optional redemption is permitted at 10% pool factor.")
    bad = Assertion(subject="Class_C_Notes", predicate="section_2_04", value="6.50%",
                    source_id="offering_memorandum", quote="Class C Notes bear interest at 6.50% per annum.")
    v1 = led.verify([good, bad], pol, record=True)
    v2 = led.verify([good], pol, record=True)
    return f"{st(v1)} -> {st(v2)}"


_REFUND = dict(src_text="Refund requested is $9,000.", value="$9,000",
               quote="Refund requested is $9,000.", subject="O",
               policy=Policy(numeric_bounds={"refund_amount": (None, 5000.0)}))
var("C11v1", ESC)(lambda t: one(predicate="refund_usd", **_REFUND))
var("C11v2", ESC)(lambda t: one(predicate="reimbursement", **_REFUND))
var("C11fp", PASS)(lambda t: one(
    "Interest for the period is $50,000.", "$50,000", "Interest for the period is $50,000.",
    subject="Loan", predicate="interest_amount", policy=Policy(numeric_bounds={"interest_rate": (None, 10.0)})))


@var("C3v", NOTPASS)
def _c3v(t):
    led = Ledger(sources=[Source(id="d", tier="authoritative", text=(
        "Interest is payable at 5% per annum upon each Payment Date. Interest is payable at 6% per annum."))])
    return verify([
        Assertion(subject="Notes", predicate="interest_rate", value="5%", source_id="d",
                  quote="Interest is payable at 5% per annum upon each Payment Date."),
        Assertion(subject="Notes", predicate="interest_rate", value="6%", source_id="d",
                  quote="Interest is payable at 6% per annum.")], led)


var("N1v", lambda g: g == "0.60")(lambda t: _n1("Rate is 6.50%.\n"))
var("F2v", lambda g: float(g) < 60)(lambda t: _saved_pct(True))

# 2.3 MCP (4) -----------------------------------------------------------------
_REF_SRC = Source(id="d", text="Refund requested is $9,000.", tier="authoritative")
_REF_A = {"subject": "O", "predicate": "refund_amount", "value": "$9,000", "source_id": "d",
          "quote": "Refund requested is $9,000."}


@var("M1", ESC)
def _m1(t):
    m, s = mcp_server([_REF_SRC], default_policy=Policy(numeric_bounds={"refund_amount": (None, 5000.0)}))
    return mcp_call(m, s, "verify_proposal", {
        "assertions": [_REF_A], "policy": {"numeric_bounds": {"refund_amount": [None, 1e12]}}})


@var("M2", NOTPASS)
def _m2(t):
    msa = Source(id="msa", text="ACME shall pay within 45 days.", tier="standard")
    amd = Source(id="amd", text="ACME shall pay within 30 days.", tier="authoritative",
                 supersedes=["msa"], valid_from="2026-06-01")
    m, s = mcp_server([msa, amd], default_policy=Policy(as_of="2026-11-13"))
    return mcp_call(m, s, "verify_proposal", {"as_of": "2026-01-01", "assertions": [{
        "subject": "ACME", "predicate": "payment_terms", "value": "45 days", "source_id": "msa",
        "quote": "ACME shall pay within 45 days."}]})


@var("M3", NOTPASS)
def _m3(t):
    m, s = mcp_server([])
    r1 = mcp_call(m, s, "register_source", {
        "id": "indenture", "text": "Class C Notes bear interest at 2.00% per annum.", "tier": "authoritative"})
    if r1.startswith("REFUSED"):
        return r1
    return mcp_call(m, s, "verify_proposal", {"assertions": [{
        "subject": "Class_C", "predicate": "interest_rate", "value": "2.00%", "source_id": "indenture",
        "quote": "Class C Notes bear interest at 2.00% per annum."}]})


@var("M4", NOTPASS)
def _m4(t):
    m, s = mcp_server([Source(id="indenture", text="Class C Notes bear interest at 6.75% per annum.",
                              tier="authoritative")])
    text = "Class C Notes bear interest at 2.00% per annum under the side letter."
    r1 = mcp_call(m, s, "register_source", {"id": "side_letter", "text": text, "tier": "standard"})
    if r1.startswith("REFUSED"):
        return r1
    return mcp_call(m, s, "verify_proposal", {"assertions": [{
        "subject": "Side_Letter", "predicate": "agreed_coupon", "value": "2.00%",
        "source_id": "side_letter", "quote": text}]})


# 2.4 Recibo (3) --------------------------------------------------------------
def _e1v_forged():
    led = Ledger(sources=[Source(id="s1", text="Limit is 100 days.", tier="authoritative")])
    v = verify([Assertion(subject="X", predicate="limit", value="999 days", source_id="s1",
                          quote="The limit is 999 days.")], led)  # cita inventada -> RETRY
    return led, forge_pass(v.to_dict())


var("E1v1", IS_FALSE)(lambda t: str(verify_receipt(_e1v_forged()[1])))
var("E1v2", IS_FALSE)(lambda t: (lambda lf: str(verify_receipt(lf[1], ledger=lf[0], policy=Policy())))(_e1v_forged()))
var("E1v3", IS_FALSE)(lambda t: str(verify_receipt(_e1v_forged()[1], signing_key="k")))

# 2.5 diff, escala y gate (7) -------------------------------------------------
_P5 = ("PRIMERA.- Objeto. El Prestamista concede un préstamo de 1.000.000 euros.\n"
       "SEGUNDA.- Vencimiento anticipado. Impago durante 5 días hábiles.\n"
       "TERCERA.- Amortización anticipada al 10%.\nCUARTA.- Ley aplicable: España.\n")


@var("D5", lambda g: not g.endswith("PASS"))
def _d5(t):
    p, s, pol, _ = dp(_P5, _P5.replace("5 días hábiles", "30 días").replace("al 10%", "al 15%"))
    v = verify([Assertion(subject="Contrato", predicate="document_body", value="Ley aplicable España",
                          source_id="P", quote="CUARTA.- Ley aplicable: España.")], Ledger(sources=[p, s]), pol)
    return f"{list(pol.required_predicates)} -> {st(v)}"


_D3V = ("Section 7.01: Events of Default. Failure to pay for 5 business days is a default.\n"
        "Section 9.01: Clean-Up Call. The Servicer may redeem at 10%. In addition,\n"
        "Section 7.01 requires the Trustee to notify Holders within 3 days, and the remainder of this "
        "long paragraph continues with further detail about notices and procedures.\n")
var("D3v", lambda g: "section_7_01" in g)(lambda t: req_preds(_D3V, _D3V.replace("5 business days", "30 days")))


@var("D8", lambda g: "term_10" not in g)
def _d8(t):
    from claimlayer.verifiers.diff import _split_into_sections
    text = ("Section 1: Fees. Fee is $100.\nThe facility has a\nterm 10 years from closing and fees "
            "accrue daily at 2%.\nSection 2: Law. New York.")
    return str(sorted(_split_into_sections(text).keys()))


# D9: OM en prosa frente a escritura numerada. Correcto = falla en cerrado: las dos secciones
# siguen siendo obligatorias. (El motivo dice "missing", lo que es inexacto pero seguro.)
var("D9", lambda g: g == "['section_1', 'section_2']")(lambda t: req_preds(
    "Section 1: Fees. Fee is $100.\nSection 2: Term. 5 years.\n",
    "DESCRIPTION OF THE NOTES\nThe fee is $100 and the term is 7 years.\n"))
var("A13v4", NOTPASS)(lambda t: one(
    "Amounts in thousands.\nSettlement claim: $95,000.\n", "$95,000", "Settlement claim: $95,000.",
    predicate="settlement_usd", policy=Policy(numeric_bounds={"settlement_usd": (None, 1_000_000.0)})))


@var("W3v1", lambda g: not g.startswith("exit 0"))
def _w3v1(t):
    # Propuesta incompleta; el agente borra los requisitos de la política sellada en su SQLite.
    sb = gate_sandbox(t)
    gate(sb, "prepare", "documents")
    (sb / "p.json").write_text(json.dumps({"assertions": [{
        "subject": "Class_C_Notes", "predicate": "section_2_04", "value": "6.75%",
        "source_id": "executed_indenture.txt",
        "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum."}]}))
    con = sqlite3.connect(sb / ".claimlayer" / "ledger.sqlite")
    pol = json.loads(con.execute("select meta_value from ledger_meta where meta_key='sealed_policy'").fetchone()[0])
    pol["required_predicates"] = []
    con.execute("update ledger_meta set meta_value=? where meta_key='sealed_policy'",
                (json.dumps(pol, sort_keys=True),))
    con.commit()
    con.close()
    (sb / ".claimlayer" / "policy.json").write_text(json.dumps(pol, indent=2))
    return gate_out(gate(sb, "verify", "p.json"))


@var("W3v2", lambda g: not g.startswith("exit 0"))
def _w3v2(t):
    # El agente reescribe el texto de la escritura dentro del SQLite de su workspace.
    sb = gate_sandbox(t)
    gate(sb, "prepare", "documents")
    (sb / "p.json").write_text(json.dumps({"assertions": [
        {"subject": "Class_C_Notes", "predicate": "section_2_04", "value": "6.50%",
         "source_id": "executed_indenture.txt",
         "quote": "Class C Notes ($50,000,000 principal) bear interest at 6.50% per annum."},
        {"subject": "Deal", "predicate": "section_7_01", "value": "30 days",
         "source_id": "executed_indenture.txt",
         "quote": "Failure to pay interest continuing for 30 days is an Event of Default."}]}))
    con = sqlite3.connect(sb / ".claimlayer" / "ledger.sqlite")
    pl = json.loads(con.execute("select payload_json from sources where id='executed_indenture.txt'").fetchone()[0])
    pl["text"] = pl["text"].replace("6.75%", "6.50%").replace("5 business days", "30 days")
    con.execute("update sources set payload_json=? where id='executed_indenture.txt'", (json.dumps(pl),))
    con.commit()
    con.close()
    return gate_out(gate(sb, "verify", "p.json"))


# ---------------------------------------------------------------------------
# Parametrización
# ---------------------------------------------------------------------------

assert len(ORIG) == 70, len(ORIG)
assert len(VAR) == 67, len(VAR)
assert REGRESSIONS <= {c for c, _, _ in VAR}


def _params():
    for cid, want, fn in ORIG:
        yield pytest.param(want, fn, id=f"orig-{cid}")
    for cid, want, fn in VAR:
        prefix = "var-regresion-" if cid in REGRESSIONS else "var-"
        yield pytest.param(want, fn, id=f"{prefix}{cid}")


@pytest.mark.parametrize(("want", "fn"), list(_params()))
def test_case(want, fn, tmp_path):
    try:
        got = fn(tmp_path)
        if not isinstance(got, str):
            got = st(got)
    except Exception as exc:  # una excepción cuenta como resultado, no como error del test
        got = f"EXC {type(exc).__name__}: {str(exc)[:100]}"
    assert want(got), f"resultado obtenido: {got}"


@pytest.mark.skipif(not os.environ.get("CLAIMLAYER_RUN_SLOW"), reason="defina CLAIMLAYER_RUN_SLOW=1")
def test_perf_verify_500_sources():
    """Rendimiento (fuera del recuento): 1 afirmación sobre 500 fuentes de 200 KB en < 1 s.

    v3.0: 7-13 ms por afirmación. v3.1.0: unos 4,5 s.
    """
    rng = random.Random(0)
    srcs = []
    for i in range(500):
        parts, size, j = [], 0, 0
        while size < 200_000:
            p = (f"Section {i}.{j}: The Servicer shall pay within {rng.randint(1, 90)} days after "
                 f"notice and the cap is ${rng.randint(1, 999)},000. ")
            parts.append(p)
            size += len(p)
            j += 1
        srcs.append(Source(id=f"doc{i}", text="".join(parts), tier="authoritative"))
    led = Ledger(sources=srcs)
    s0 = srcs[0]
    a = [Assertion(subject="Servicer", predicate="payment_deadline",
                   value=s0.text.split("within ")[1].split(" days")[0] + " days", source_id=s0.id,
                   quote=s0.text.split(". ")[0].split(": ", 1)[1] + ".")]
    t0 = time.perf_counter()
    verify(a, led)
    assert time.perf_counter() - t0 < 1.0
