# Contrato de Diseño y Garantías: Obligaciones Impuestas por el Motor y Ciclo de Evaluación-Reparación (`claimlayer.v4`)

Este documento define la arquitectura y el contrato formal del ciclo de **evaluación, feedback estructurado, reparación y reverificación** de respuestas generadas por LLM sobre el motor graduado de `claimlayer.v4`.

## 1. Principio Rector: Una Sola Ruta de Evaluación

No existe un verificador paralelo para respuestas en lenguaje natural. Tanto las comprobaciones sobre grafos (`evaluate_response`) como el ciclo de revisión y reparación (`review_response` / `RepairOrchestrator`) ejecutan **el mismo núcleo**:
- `JustificationGraph` + clausura epistémica (`effective_epistemic_kind`)
- **Generador de Obligaciones Impuestas por el Motor** (`OBLIGATION_GENERATOR_VERSION = "1.0.0"`)
- Registro de 9 familias verificadoras (`DEFAULT_VERIFIER_REGISTRY`), con `UnitRegistry`, `FieldSpec`, fidelidad relacional y `evaluate_unit_admissibility()`
- Descomposición ortogonal `(evidence_support_index, conflict_index, uncertainty_index)` y `verification_coverage`
- Calibración estricta (`calibrated_correctness_probability = None` salvo calibración empírica explícita fuera de muestra)
- Vinculación criptográfica mediante `response_digest`, `assessment_digest` y firma HMAC opcional en autorización (`authorize()`).

---

## 2. Obligaciones Propuestas vs. Impuestas por el Motor vs. Pendientes

En `claimlayer.v4`, un modelo generativo puede proponer un grafo de justificación, pero **no controla qué debe comprobarse**:

1. **Obligaciones propuestas (`proposed_obligations`)**: las que acompaña cada `InferenceStep` al recibirse el grafo.
2. **Obligaciones impuestas por el motor (`mandatory_obligations`)**: generadas de forma determinista y versionada (`OBLIGATION_GENERATOR_VERSION`) a partir de:
   - El tipo semántico inferido de la conclusión (incluyendo unidades y `FieldSpec` del `DomainProfile`).
   - La pregunta y el contexto de evaluación (`as_of`, texto de respuesta).
   - La precisión de los datos en premisas (fechas completas `YYYY-MM-DD` frente a `YYYY` / `YYYY-MM`, intervalos o cantidades).
   - Las dependencias, condiciones y supuestos de las premisas ancestrales.
   - Evidencia contradictoria o múltiples fuentes en el contexto.
3. **Obligaciones pendientes (`pending_obligations`)**: obligaciones cuya verificación devuelve `unknown` o no pudo ejecutarse por presupuesto o falta de datos/correspondencia.

### Anti-evasión por etiquetado genérico
Si una conclusión usa etiquetas genéricas (`"number"`, `"value"`, `"result"`, `"amount"`, `"total"` o predicado vacío) u operaciones genéricas (`"add"`, `"arithmetic"`), el motor inspecciona premisas, unidades, fechas y la pregunta para imponer igualmente las familias pertinentes. Si el tipo semántico relevante sigue siendo incierto (`semantic_type_uncertain`), registra explícitamente `AMBIGUOUS_OR_GENERIC_SEMANTIC_TYPE` como incertidumbre pendiente.

---

## 3. Las 6 Familias de Obligaciones Impuestas

| Familia obligatoria | Verificador v4 | Qué comprueba y cuándo rechaza / abstiene |
|---|---|---|
| `age_and_duration` | `TemporalVerifier` | Deriva intervalos reales desde fechas o edades ancla con granularidad (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`). Rechaza valores fuera de intervalo (`TEMPORAL_DERIVATION_CONTRADICTED`, incluso con `"aproximadamente"`). Abstiene ante valores exactos con fechas incompletas (`UNJUSTIFIED_TEMPORAL_PRECISION`) o aproximaciones que ocultan intervalos amplios (`APPROXIMATION_HIDES_WIDE_UNCERTAINTY`). |
| `quantities` | `QuantityVerifier` / `ArithmeticVerifier` | Impone compatibilidad dimensional vía `UnitRegistry`, detección de unidades desconocidas, y trazabilidad de redondeo/tolerancia (`ROUNDING_OR_TOLERANCE_UNSPECIFIED`). |
| `documental_attribution` | `DocumentAuthorityVerifier` | Cuando las fuentes son declaraciones documentales no contrastadas externamente (`source_statement`, `memo`, `allegation`) y se evalúa una respuesta candidata, exige atribución explícita al texto aportado sin presentarla como verdad histórica externa contrastada (`SOURCE_CONDITIONAL_CONCLUSION`). |
| `conditions_and_exceptions` | `ExplicitLogicVerifier` | Comprueba que toda condición o salvedad en las premisas ancestrales se conserva en la conclusión (`DROPPED_REQUIRED_CONDITION`). |
| `aggregations` | `AggregationVerifier` | Impide presentar un subtotal o población incompleta como total sin calificar (`SUBTOTAL_PRESENTED_AS_TOTAL`, `UNVERIFIED_POPULATION_COMPLETENESS`) y exige deduplicación por `event_id` y unidades conmensurables. |
| `material_conflict` | `DocumentAuthorityVerifier` | Impide presentar como inequívoca una respuesta cuando existe evidencia contradictoria material activa sin revelar el conflicto (`UNDISCLOSED_MATERIAL_CONFLICT`). |

---

## 4. Vinculación Texto ↔ Grafo y Caducidad por Digest

El sistema distingue tres niveles explícitos (`representation_mode`):
1. `graph_only`: evaluación de un grafo suministrado sin texto de respuesta (`response_correspondence_verified = False`).
2. `supplied_graph_with_response`: grafo suministrado junto con `candidate_response` y `question`, donde se verifica además la correspondencia texto ↔ grafo.
3. `adapter_extracted_graph`: grafo propuesto dinámicamente por un adaptador o LLM desde la respuesta y evidencia.

Todo `ResponseAssessment` y `ReviewResult` con respuesta calcula `response_digest = digest({"question": ..., "candidate_response": ...})`. Cualquier edición posterior de la respuesta invalida la aprobación previa (`verify_review_freshness() -> False`).

---

## 5. Verificación Semántica de Reparaciones (No Palabras Sueltas)

La correspondencia texto ↔ grafo (`verify_response_graph_correspondence`) analiza la estructura de cláusulas y magnitudes de toda la respuesta para detectar:
- **Contradicción con su propia salvedad** (`CONTRADICTORY_PRECISION_QUALIFICATION`, p. ej. *"Aproximadamente 106 años; por tanto, sabemos que tiene exactamente 106"*).
- **Contradicción de atribución documental** (`CONTRADICTORY_SOURCE_ATTRIBUTION`, p. ej. afirmar *"según el texto"* y luego *"está demostrado históricamente como hecho verificado"*).
- **Cambio de sujeto, cálculo o unidades** durante un retry (`SUBJECT_CHANGED_IN_REPAIR`, `CALCULATION_ALTERED_OR_INCONSISTENT`, `UNIT_ALTERED_IN_REPAIR`).
- **Supresión de condiciones relevantes o conflictos materiales** (`CONDITION_DROPPED_IN_RESPONSE`, `CONFLICT_HIDDEN_IN_RESPONSE`).
- **Adición factual sin respaldo** en evidencia (`UNSUPPORTED_FACTUAL_ADDITION`).
- **Evasión** (eliminar la afirmación cuestionada dejando la pregunta sin respuesta -> `QUESTION_LEFT_UNANSWERED_EVASION`) frente a eliminación legítima de afirmaciones secundarias innecesarias.

---

## 6. Feedback Estructurado, Ciclo Acotado y Separación de Autorización

- `ReviewResult` devuelve `disposition` (`ACCEPT`, `RETRY`, `ESCALATE`, `ABSTAIN`), `response_digest`, `assessment_digest`, `verified_elements`, `issues` (con `code`, `claim_id`/`step_id`, `evidence_refs`, `reason`, `required_change`, `acceptance_condition`, `severity`), `required_repairs` y `unresolved_obligations`.
- `RepairOrchestrator` ejecuta un ciclo acotado por `RepairBudget` (`max_attempts`, `VerificationBudget`), detecta respuestas duplicadas o estancamiento, reutiliza `VerificationCache` solo si las dependencias no cambian, conserva el historial completo y **nunca acepta por agotamiento de intentos**.
- Aceptar una respuesta explicativa bien acotada (`disposition == "ACCEPT"`, por ejemplo *"Según el texto, tendría aproximadamente 106 años..."*) **no implica** autorizar una operación ejecutiva (`authorize(assessment, ...)` puede seguir devolviendo `ESCALATE` si la operación exige precisión exacta o verdad contrastada externamente).

## 7. Verificación Real de Correspondencia Texto-Grafo y Perturbaciones

La correspondencia texto-grafo en `claimlayer.v4.correspondence` (`verify_response_graph_correspondence`) no depende solo de coincidencias léxicas superficiales ni frases literales:

1. **Verificación aritmética mediante AST (`verify_response_equations`, `EquationCheckResult`)**:
   - Extrae ecuaciones en línea y cadenas de igualdades (`=`), limpia prefijos/unidades y desenvuelve paréntesis externos, evaluando expresiones mediante `ast.parse(..., mode="eval")` y aritmética exacta `Decimal` (sin `eval`).
   - Rechaza igualdades aritméticamente falsas (`FALSE_EQUATION_IN_RESPONSE`, p. ej. `13 + (2026 − 1933) = 13`) e igualdades ciertas pero desconectadas de los operandos/resultados del paso en el grafo (`UNGROUNDED_EQUATION_IN_RESPONSE`, p. ej. `50 + 50 = 100`).
   - Deja expresiones simbólicas o no parseables como pendientes (`UNVERIFIED_EXPRESSION_IN_RESPONSE`, `status="unknown"`), impidiendo `ACCEPT` y manteniendo `response_correspondence_verified=False`.

2. **Comparación estructurada de condiciones y polaridad (`StructuredCondition`)**:
   - Clasifica marcadores de excepción (`salvo`, `excepto`, `a menos que`), prerrequisitos positivos (`si`, `siempre que`, y prerrequisitos negados como `solo si no`), y anulaciones concesivas (`incluso en caso de`, `aunque`).
   - Rechaza inversiones de polaridad (`CONDITION_POLARITY_INVERTED`, p. ej. *"incluso en caso de disputa activa"* frente a *"salvo en caso de disputa activa"*), omisiones (`CONDITION_DROPPED_IN_RESPONSE`), y sustituciones por condiciones no soportadas por evidencia (`UNSUPPORTED_CONDITION_SUBSTITUTION`, p. ej. *"si hace sol"*).
   - Menciones ambiguas sin marcador de polaridad claro quedan en estado pendiente (`CONDITION_CORRESPONDENCE_UNRESOLVED`, `status="unknown"`) salvo resolución explícita mediante `SemanticEvaluatorConfig` (`open_semantics`).

3. **Descomposición por cláusulas, polaridad de conclusión y afirmaciones no numéricas (`ResponseClauseClaim`)**:
   - Divide sub-cláusulas discursivas y separa fórmulas, descargos/limitaciones de fecha (`no se puede determinar la edad exacta`), condiciones y afirmaciones.
   - Detecta inversión de polaridad sobre la conclusión sin quedar enmascarada por descargos de limitación (`CONCLUSION_POLARITY_INVERTED`) y detecta afirmaciones fácticas no numéricas introducidas fuera del léxico verificado del documento/pregunta/grafo (`UNSUPPORTED_FACTUAL_ADDITION`, con delegación opcional a `open_semantics` -> `RESPONSE_CORRESPONDENCE_PENDING_SEMANTIC`).

4. **Patrones estructurales composicionales y evaluación por perturbaciones**:
   - Acepta paráfrasis legítimas de atribución documental, estimación/aproximación (`estimación`, `unos ...`, `en torno a`, `rondaría`) y granularidad de calendario (`faltan el día y el mes para precisar su edad`).
   - `claim-layer/benchmarks/v4_repair_cycle_demo.py` (`run_perturbation_benchmark`) y `tests/test_v4_repair_cycle.py` (`test_17`–`test_21`) miden `false_accept_rate` sobre perturbaciones adversariales y `false_reject_rate` sobre paráfrasis válidas.
