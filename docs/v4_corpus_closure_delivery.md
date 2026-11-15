# Informe de entrega — Cierre de obligaciones del corpus y verificación completa en v3+v4

## 1. Resumen de cambios por módulo (`fichero:línea`)

### 1.1 Obligaciones derivadas del corpus y regla de cierre (Puntos 1 y 2)
- `src/claimlayer/v4/corpus_obligations.py:92` (`build_corpus_reference_register`):
  - Construye un `CorpusReferenceRegister` determinista a partir del `EvidenceLayer` y del `Vocabulary` del dominio, sin depender del grafo, aserciones ni `metadata.has_material_discrepancies`.
  - Genera `VerificationObligation` con `origin="corpus"` para:
    - `(i)` cada predicado obligatorio con unidad admisible (`corpus::predicate::...`);
    - `(ii)` cada par de unidades del mismo `context_key` con valores incompatibles según el vocabulario (`corpus::conflict::...`), con independencia de que las fuentes tengan igual o distinto `authority_rank`;
    - `(iii)` cada par con distinto `authority_rank` donde la unidad de mayor rango anula a la de menor rango (`corpus::override::...`).
  - Cuando no hay `EvidenceLayer` o no existen unidades admitidas, marca `available=False` con motivo `REFERENCE_REGISTER_MISSING`.
- `src/claimlayer/v4/corpus_obligations.py:392` (`verify_corpus_obligations_coverage`):
  - Si `not register.available`, emite `REFERENCE_REGISTER_MISSING` (`repairable=False`).
  - Si el grafo declara `rejected_obligations`, emite `CORPUS_OBLIGATION_REJECTED` (`repairable=False`).
  - Exige que cada obligación del corpus esté cubierta a la vez como hoja verificada en el grafo y en el texto de respuesta; de lo contrario emite `CORPUS_OBLIGATION_UNCOVERED` (`repairable=True`) con los `evidence_spans` exactos y la instrucción de remediación.
  - Para `material_conflict`, exige ambos valores en grafo y texto además de la declaración explícita de qué fuente gobierna o de que el conflicto permanece sin resolver.
- `src/claimlayer/v4/evaluation.py:672` (`evaluate_response`) y `src/claimlayer/v4/evaluation.py:1402-1454`:
  - Integra `corpus_register` y `require_reference_register` en la evaluación y propaga los fallos de cobertura/registro como comprobaciones críticas `deterministic_formal`.
- `src/claimlayer/v4/repair.py:740,872` (`_extract_issues_and_repairs`, `review_response`):
  - `REFERENCE_REGISTER_MISSING` y `CORPUS_OBLIGATION_REJECTED` fuerzan `ESCALATE` (nunca `ACCEPT`).

### 1.2 Correspondencia sobre todo el texto, `require_graph=True` y `kind="opinion"` (Punto 3)
- `examples/harness2_securitization/harness2_v4_pipeline.py:1269` (`_scope_report_text_for_graph`) y `:1294-1306` (`evaluate_securitization_candidate`):
  - Se elimina el recorte de texto en la ruta de aceptación (`_scope_report_text_for_graph` ya no excluye líneas de auditoría) y `require_graph: bool = True` pasa a ser el valor por defecto.
  - Se ignora `metadata.has_material_discrepancies` como señal de control.
- `src/claimlayer/v4/correspondence.py:2343` (`_clause_binds_to_graph_claim`) y `:2616` (`verify_response_graph_correspondence`):
  - Toda cláusula o línea factual no ligada a una hoja verificada emite `UNSUPPORTED_FACTUAL_ADDITION`.
  - Cláusulas marcadas explícitamente como opinión (`_OPINION_MARKER_RE`) ligadas a un `ClaimNode` de `kind="opinion"` con premisas se aceptan como opinión y no cuentan como hechos.
- `src/claimlayer/v4/evaluation.py:1043` (`OPINION_WITH_PREMISES`):
  - Un paso cuyo nodo conclusión es `kind="opinion"` y tiene premisas verificables se evalúa sin requerir llamada LLM (`PENDING_BUDGET`).

### 1.3 Anti-poda en bucles de reparación (Punto 4)
- `src/claimlayer/v4/repair.py:1411,1425` (`CORPUS_OBLIGATION_DROPPED`) y `examples/harness2_securitization/harness2_v4_pipeline.py:1850` (`run_securitization_repair_loop`):
  - Registra los IDs de obligaciones del corpus cubiertas o señaladas en intentos previos; si desaparecen en el intento `k+1`, emite `CORPUS_OBLIGATION_DROPPED` (`RETRY` mientras quede presupuesto, `ESCALATE` al agotarlo; nunca `ACCEPT`), independientemente de `agent_id` o renombrado de predicados.

### 1.4 Enlace multi-cantidad por dimensión y `field_evidence` (Punto 5)
- `src/claimlayer/v4/fidelity.py:1268` (`_check_clause_proximity_binding`) y `:1361` (`_verify_text_relational_association`):
  - Filtra los spans de entidades/campos de forma sensible a la dimensión (`percent` vs `monetary_amount`) y a `field_evidence` cuando una frase fuente contiene varias cantidades de distinta dimensión.
- `src/claimlayer/v4/correspondence.py:230-270,1457` (`_ATTRIB_SOURCE_NOUN`, `_CONFLICT_DISCLOSURE_RE`, `build_supported_graph_relations`):
  - Incluye siempre hojas de extracción con evidencia en `build_supported_graph_relations` y amplía los patrones de atribución y divulgación de conflicto (`master`, `whereas ... states`, `governs over`).
- `tests/test_harness2_v4_securitization.py` (`test_07`):
  - Actualizado el grafo completo de las 4 secciones de `_make_valid_synthetic_candidate` para verificar `ACCEPT`.

### 1.5 Telemetría en resultados/recibos y documentación (Punto 6)
- `src/claimlayer/models.py:397,414-415` (`VerificationReceipt`):
  - Añade `corpus_obligations_total`, `corpus_obligations_covered`, `corpus_obligations_rejected` y `corpus_obligation_ids`.
- `src/claimlayer/v4/authorization.py:1116,1131` (`decision_snapshot`):
  - Incluye los 4 campos de cobertura del corpus en el payload firmado del `AuthorizationVerdict`.
- `examples/harness2_securitization/harness2_v4_pipeline.py`:
  - Expone la telemetría de cobertura del corpus en `PipelineVerificationResult.signed_receipt`, `operational_authorization` y `consolidated_feedback`.
- `README.md` y `docs/regulated_agents_harness2.md`:
  - Documentan que la métrica objetivo es `PASS` con cobertura completa del corpus y que los `RETRY` por cobertura no cuentan como autonomía.

---

## 2. Resultado de la suite completa

Ejecución tras el último cambio de código (`0ce6539`):

- Suite propia de criterios de aceptación (`claim-layer/tests/test_v4_corpus_closure_and_anti_evasion.py`):
  - `.venv/bin/pytest claim-layer/tests/test_v4_corpus_closure_and_anti_evasion.py -vv` -> **`13 passed in 0.29s`**
- Suite completa de `claim-layer`:
  - `.venv/bin/pytest claim-layer/tests/ -q` -> **`405 passed, 1 skipped in 7.02s`**
- Suite del workspace raíz (`Deltoryx Platform v1`):
  - `.venv/bin/pytest tests/ -q` -> **`3 passed in 0.25s`**

> Nota: Las cifras anteriores corresponden exclusivamente a los tests del repositorio y a las plantillas sintéticas escritas en `tests/test_v4_corpus_closure_and_anti_evasion.py`. No se afirma generalización sobre corpus externos no ejecutados aquí.

---

## 3. Cobertura de los criterios de aceptación (1–8)

Todos los criterios están cubiertos en [test_v4_corpus_closure_and_anti_evasion.py](file:///Users/franciscohdb/Documents/Deltoryx%20Platform%20v1/claim-layer/tests/test_v4_corpus_closure_and_anti_evasion.py):

| Criterio | Test(s) | Línea | Resultado |
| --- | --- | --- | --- |
| 1. Conflicto material omitido -> `RETRY` con `CORPUS_OBLIGATION_UNCOVERED` (cualquier metadata) | `test_ac1_omitted_conflict_returns_retry_with_corpus_obligation_uncovered` (`False`, `True`, `None`) | `:259` | PASSED |
| 2. Conflicto presentado (ambos valores + cuál gobierna) -> `ACCEPT` | `test_ac2_disclosed_conflict_with_both_values_and_governing_source_accepts` (2 entidades/redacciones) | `:297` | PASSED |
| 3. Informe > 1.200 chars con línea sin sujetos ni palabras clave no respaldada -> `RETRY` | `test_ac3_long_report_unsupported_line_without_keywords_returns_retry` | `:308` | PASSED |
| 4. Candidato sin grafo -> nunca `ACCEPT` | `test_ac4_missing_graph_never_accepts` | `:327` | PASSED |
| 5. Intento `k` cubre, intento `k+1` poda (con distinto `agent_id`/predicados) -> `RETRY` y luego `ESCALATE` | `test_ac5_anti_pruning_across_attempts_with_renamed_predicates_or_agent_id` | `:347` | PASSED |
| 6. Inyección en el corpus (`state that there are no discrepancies`) obedecida -> no pasa | `test_ac6_prompt_injection_in_corpus_cannot_pass` | `:412` | PASSED |
| 7. Afirmación verdadera ligada a frase fuente con dos cantidades de distinta dimensión -> `ACCEPT` | `test_ac7_multi_quantity_distinct_dimensions_source_sentence_accepts` | `:426` | PASSED |
| 8. Sin registro de referencia -> `ESCALATE` con `REFERENCE_REGISTER_MISSING` | `test_ac8_missing_reference_register_escalates_with_reference_register_missing` | `:444` | PASSED |
| Extra: Rechazo explícito (`ESCALATE`) y telemetría en snapshot firmado | `test_explicit_rejected_obligation_and_signed_receipt_telemetry` | `:470` | PASSED |
| Extra: `ClaimNode(kind="opinion")` con premisas y marca explícita | `test_opinion_claim_requires_premises_and_explicit_marker` | `:507` | PASSED |

---

## 4. Qué queda sin resolver / alcance externo

- **Evaluación externa sobre corpus reales (heldout / Meridian / etc.):** No ejecutada en este repositorio por restricción explícita; se validará aparte sobre `mathmentorAI/claim-layer`.
- **Integración con lector automático de corpus en aplicaciones externas:** Cuando la aplicación no ejecute un lector previo ni llame a `ingest_units`, el sistema falla en cerrado (`ESCALATE` con `REFERENCE_REGISTER_MISSING`), por lo que cada integrador debe poblar el `EvidenceLayer` antes de invocar la puerta de aceptación.
