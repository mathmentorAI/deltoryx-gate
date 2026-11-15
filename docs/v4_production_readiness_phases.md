# Plan de Preparación para Producción (Fases 1–4)

Documentación de los cambios implementados en las 4 fases de endurecimiento y preparación para producción de `claimlayer` (v3/v4) y del plano de control FastAPI (`api/`).

---

## Fase 1 — Cierre de brechas de verificación semántica y de citas

- **`claim-layer/src/claimlayer/v4/correspondence.py`**:
  - Normalización numérica (`_normalize_numeric_str`) y mejoras de correspondencia de respuesta.
- **`claim-layer/src/claimlayer/v4/domains/securitization.py`**:
  - Verificación estructurada de cascadas de pagos (*waterfall*): `WaterfallStepSpec`, `WaterfallComparisonResult`, `verify_waterfall_steps`, normalización de beneficiarios/tipos de pago/rangos (`_norm_beneficiary`, `_norm_payment_type`, `_norm_position_range`, `_slot_key`, `_cond_signature`) y grafo de precedencia con cierre transitivo (`_normalize_target_ref`, `_build_precedence_graph`, `_transitive_closure`, `_parse_slot_str`).
- **`claim-layer/src/claimlayer/v4/justification.py`**, **`claim-layer/src/claimlayer/v4/repair.py`**, **`claim-layer/src/claimlayer/v4/verifiers.py`**, **`claim-layer/src/claimlayer/v4/vocabulary.py`**, **`claim-layer/src/claimlayer/verifiers/quote_quantity.py`**:
  - Endurecimiento de verificadores, justificación, ciclo de reparación y vocabulario/extracción de cantidades y citas.

---

## Fase 2 — Generalización multidominio (`banking_finance`)

- **`claim-layer/src/claimlayer/v4/domains/banking_finance.py`**:
  - Nuevo contrato/adaptador de dominio para banca y financiación (`resolve_banking_governing_role`, reglas de jerarquía normativa y precedencia documental frente a borradores o *markups* de contraparte).
- **`claim-layer/src/claimlayer/v4/domains/__init__.py`**:
  - Registro y selección de adaptadores por dominio (`securitization` y `banking_finance`).
- **`claim-layer/tests/test_banking_finance_domain.py`**:
  - Tests unitarios e integración para resolución de autoridad contextual, bloqueo de inversión de autoridad y bucle v3/v4 con reparación (`test_banking_context_authority_resolution_and_adapter_selection`, `test_banking_adapter_blocks_counterparty_markup_authority_inversion`, `test_banking_adapter_accepts_context_grounded_v3_v4_and_repair_loop`).

---

## Fase 3 — Suite de evaluación *held-out* multidominio

- **`claim-layer/benchmarks/heldout_multidomain_eval.py`**:
  - Corpus sintéticos *held-out* para titulización (`_build_meridian_securitization_corpus`) y banca (`_build_apex_banking_corpus`), constructor de escenarios (`_build_heldout_scenarios`), ejecutor (`run_heldout_evaluation`) y CLI (`main`).
- **`claim-layer/tests/test_heldout_eval_suite.py`**:
  - Validación automática de SLAs v3/v4 y generación de hoja ciega (`test_heldout_multidomain_eval_passes_v3_v4_sla_and_writes_blind_sheet`).

---

## Fase 4 — Plano de control FastAPI, persistencia SQLite, telemetría y resiliencia LLM

- **`api/audit_store.py`**:
  - Clase `AuditReceiptStore` sobre SQLite transaccional (tabla `v4_audit_receipts`) con `record_receipt`, `get_receipt`, `list_receipts` y `get_telemetry`.
- **`api/main.py`**:
  - Integración del flujo v4 con `_AUDIT_STORE`, `_V4_CACHE`, `V4ReviewIn` y `_domain_contract`.
  - Nuevos endpoints:
    - `POST /api/v1/v4/review` (`v4_review_endpoint`)
    - `GET /api/v1/v4/receipts` (`list_v4_receipts`)
    - `GET /api/v1/v4/receipts/{receipt_id}` (`get_v4_receipt`)
    - `GET /api/v1/v4/telemetry` (`get_v4_telemetry`)
- **`claim-layer/examples/harness2_securitization/run_three_arm_eval.py`**:
  - Reintentos con *exponential backoff* (`max_retries=3`, `backoff_base_s=1.0`) en `Client.call` ante `HTTPError` (`429`, `500`, `502`, `503`, `504`), `URLError` y `TimeoutError`.
  - Guardia de presupuesto mediante `--max-total-tokens`.
- **Tests añadidos**:
  - `tests/test_v4_production_api.py` (`test_v4_review_persists_digest_receipts_cache_and_telemetry`).
  - `claim-layer/tests/test_three_arm_eval_runner.py` (`test_client_retries_transient_http_errors_with_backoff`).

---

## Verificación de la suite completa

```bash
PYTHONPATH=claim-layer/src:claim-layer:. .venv/bin/pytest claim-layer/tests/ tests/ -q
```

Resultado tras completar la Fase 4: **`395 passed, 1 skipped in 7.21s`**.
