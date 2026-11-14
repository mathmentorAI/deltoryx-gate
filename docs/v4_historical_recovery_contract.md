# Fase 0 — Inventario Histórico y Contrato de Garantías (`ba81728` y `d70bf3f` → ClaimLayer v4)

Este documento fija el contrato formal antes de implementar cualquier código de recuperación en ClaimLayer v4 / Deltoryx Gate. Los módulos históricos en `ba81728` y `d70bf3f` se utilizan **únicamente como referencia conceptual** y nunca como dependencia en tiempo de ejecución ni como rollback de las garantías vigentes en `claimlayer.v4` (`UnitRegistry`, `FieldSpec`, fidelidad relacional, `evaluate_unit_admissibility()`, transacciones SQLite, snapshots HMAC y protocolo de ejecución en 3 fases).

---

## 1. `core/inference_engine.py` (`ba81728`, ampliado en `d70bf3f`)

- **Capacidades a recuperar:**
  - Traza auditable completa (`resolve_claims_detailed()`) con desglose por grupo candidato, soporte acumulado, oposición de grupos perdedores y motivo terminal.
  - Filtrado temporal activo/inactivo (`_is_active`) y separación de conflictos simultáneos frente a evolución temporal secuencial mediante intervalos de Allen (`allen_interval_precedes`).
  - Comparación trivaluada (`compare_claims_three_valued`: `'compatible' | 'incompatible' | 'unknown'`) e índice acotado de conflicto $Q_g = \frac{S_g^2}{S_g + O_g}$.
  - Desduplicación por clave de dependencia (`_dependence_key`) antes de la fusión Noisy-OR.
- **Antipatrones a evitar:**
  - Presentar pesos heurísticos de tabla o puntuaciones resultantes como probabilidades calibradas de verdad (`projected_probability`).
  - Normalización libre basada en listas de predicados comodín (`_MULTIVALUED_PREDICATES`, `_CATCHALL_PREDICATES`) sin tipos ni dimensiones `UnitRegistry`.
- **1. Entradas requeridas:** Lista de proposiciones tipadas (`ClaimNode`), marca temporal opcional `as_of`, pesos por familia/origen en `DomainProfile`, claves de grupo de dependencia y vocabulario tipado (`Vocabulary` / `UnitRegistry`).
- **2. Garantía exacta:** Calcula de forma determinista qué candidatos cuentan con soporte agrupado tras desduplicar fuentes dependientes, separa intervalos temporales disjuntos de contradicciones simultáneas y reporta `evidence_support_index`, `conflict_index` y `uncertainty_index`. No demuestra verdad empírica externa del texto ni produce una probabilidad calibrada sin calibrador empírico independiente.
- **3. Supuestos:** Las claves de dependencia/origen identifican adecuadamente fuentes derivadas entre sí; dos afirmaciones con `FieldSpec` y `UnitRegistry` compatibles pueden compararse deterministamente dentro de la misma ventana temporal declarada.
- **4. Condiciones de abstención:**
  - Comparación en `'unknown'` (granularidad de fecha insuficiente o intervalos solapados).
  - Ninguna afirmación activa dentro de `as_of` (`unsupported` / `inconclusive`).
  - Margen insuficiente ($S_g - O_g < \delta$) o conflicto elevado ($C > \text{max\_conflict\_index}$).
- **5. Pruebas disponibles y huecos detectados:** Existían `tests/test_inference_engine.py`, `tests/test_evidence_value_polarity.py`, `tests/test_multi_document_conflicts.py`. Huecos históricos: comparación numérica sin dimensiones `UnitRegistry`, uso de strings libres sin `Vocabulary`, y ausencia de bloqueo frente a fiabilidad no evaluada (`structural_reliability=None`) o errores de extracción relacional.

---

## 2. `core/graph_traverser.py` (`d70bf3f`)

- **Capacidades a recuperar:**
  - Recorrido multi-salto explícito (`MultihopStep`, `MultihopPath`) y traza paso a paso.
  - Composición en serie mediante T-norma de Gödel (eslabón más débil a lo largo del camino) con descuento acotado por salto $\gamma^{k-1}$.
  - Prevención de ciclos por camino (`path_visited`) y búsqueda acotada por profundidad y caminos (`max_depth`, `max_paths`).
- **Antipatrones a evitar:**
  - Fusión mediante Noisy-OR sobre caminos no disjuntos que comparten el mismo documento, fuente o `dependence_group` (inflación por redundancia).
  - Emparejamiento laxo por subcadenas de entidad/predicado (`pred_norm in c_pred_norm`) sin autorización explícita del esquema.
- **1. Entradas requeridas:** Grafo acíclico de justificación (`JustificationGraph`), diccionario de reglas de composición autorizadas (`RelationCompositionRule` en `DomainProfile`), límites explícitos de exploración (`VerificationBudget`).
- **2. Garantía exacta:** Solo admite composiciones $(r_1, r_2) \to r_3$ autorizadas en el perfil de dominio con coincidencia exacta de entidad puente, solapamiento temporal activo y sin ciclos; acota el soporte transmitido por el mínimo de sus premisas descontado por tipo de inferencia y nunca duplica soporte por caminos correlacionados compartiendo linaje.
- **3. Supuestos:** La composición autorizada preserva validez dentro de la intersección temporal de sus aristas y las identidades de entidad ya están canonizadas mediante `Vocabulary.entity()`.
- **4. Condiciones de abstención:**
  - Composición no declarada ni transitiva autorizada en `DomainProfile.relation_rules` (`UNKNOWN: UNAUTHORIZED_RELATION_COMPOSITION`).
  - Exceso de presupuesto (`max_depth`, `max_paths` o `max_steps` -> `PENDING_BUDGET`).
  - Intervalos temporales disjuntos en aristas consecutivas o entidad puente inexistente (`refuted`).
- **5. Pruebas disponibles y huecos detectados:** Probado parcialmente en `tests/test_sprint5_end_to_end_unification.py`. Huecos históricos: permitía que dos caminos provenientes de la misma fuente subyacente inflaran la confianza por Noisy-OR y admitía cadenas heterogéneas sin reglas explícitas en perfil.

---

## 3. `core/aggregation_engine.py` (`d70bf3f`)

- **Capacidades a recuperar:**
  - Operaciones agregadas (`sum`, `count`, `average`, `min`, `max`) con comprobación explícita de compatibilidad de unidades, identidad de evento (`event_id`) y alcance poblacional.
- **Antipatrones a evitar:**
  - Conversión/extracción heurística de unidades mediante regex sobre texto libre (`_extract_claim_unit`).
  - Descarte silencioso de unidades minoritarias por mayoría (`_filter_compatible_units`) para emitir una suma aparentemente segura sobre una población incompleta.
- **1. Entradas requeridas:** Conjunto de `ClaimNode` homogéneos en dominio/población, operación de agregación declarada, `UnitRegistry` y metadatos explícitos de completitud poblacional (`population_complete` / `expected_population_ids`) e identidad de evento (`event_id`).
- **2. Garantía exacta:** Calcula la aritmética exacta en `Decimal` (tras normalización con `UnitRegistry`) solo cuando todas las unidades son dimensionalmente compatibles, no existe ambigüedad de eventos duplicados y la población está acreditada como completa.
- **3. Supuestos:** Los elementos incluidos representan eventos o individuos correctamente identificados por `event_id` cuando pueden repetirse en varias fuentes.
- **4. Condiciones de abstención:**
  - Unidades incompatibles entre miembros de la agregación -> `refuted` (`AGGREGATION_INCOMPATIBLE_UNITS`).
  - Dos menciones idénticas sin `event_id` que aclare si son el mismo evento o dos distintos -> `unknown` (`AMBIGUOUS_EVENT_DUPLICATION`).
  - Población completa no acreditada -> `unknown` (`UNVERIFIED_POPULATION_COMPLETENESS`).
- **5. Pruebas disponibles y huecos detectados:** Probado en `tests/test_sprint2_improvements.py` y `test_audit_edge_cases.py`. Huecos históricos: `_filter_compatible_units` ignoraba unidades incompatibles devolviendo una suma parcial y la desduplicación por texto heurístico dependía de tokens de fecha/importe en lugar de `event_id` explícito y contratos `FieldSpec`.

---

## 4. `core/epistemic_judge.py` (`d70bf3f`)

- **Capacidades a recuperar:**
  - Separación estructural entre hechos/extracciones, derivaciones formales, alegaciones de parte e hipótesis/predicciones/recomendaciones.
  - Detección de ciclos en DAGs de premisas (`_has_premise_cycle`).
  - Clausura epistémica transitiva: el rango de una conclusión está acotado por el mínimo rango de sus premisas.
- **Antipatrones a evitar:**
  - Tratar listas de expresiones léxicas (`_PROVEN_JUDICIAL_CUES`, `_ALLEGATION_TERMS`) como prueba suficiente de verdad factual o validez deductiva.
  - Permitir que una hipótesis o recomendación gane estatus de hecho (`extraction`) por acumulación de peso o menciones.
- **1. Entradas requeridas:** `ClaimNode` con clase epistémica obligatoria (`CLAIM_KINDS = ('extraction', 'derivation', 'hypothesis', 'prediction', 'recommendation')`), DAG de `InferenceStep` y roles de atestación declarados en `DomainProfile`.
- **2. Garantía exacta:** Garantiza invariablemente la acotación epistémica `effective_epistemic_kind`: ninguna conclusión derivada de una `hypothesis`, `prediction` o `recommendation` puede alcanzar rango `extraction` ni `derivation` probada, y ninguna referencia circular puede validarse.
- **3. Supuestos:** Las fuentes tienen su rol de atestación (`attestor`) asignado por la aplicación o la política de dominio (no autodeclarado por el modelo generativo).
- **4. Condiciones de abstención:**
  - Ciclo detectado en dependencias (`CYCLIC_JUSTIFICATION_GRAPH`).
  - Premisa ausente o con `structural_reliability=None` en ruta de autorización (`UNASSESSED_READER_RELIABILITY`).
  - Uso de `hypothesis` / `prediction` / `recommendation` cuando la operación exige base factual/deductiva (`NON_FACTUAL_EPISTEMIC_KIND_FOR_OPERATION`).
- **5. Pruebas disponibles y huecos detectados:** Probado en `tests/test_audit_edge_cases.py` y `tests/test_sprint5_end_to_end_unification.py`. Huecos históricos: clasificación mediante substring matching multilingüe que degradaba o promocionaba claims por cues superficiales y permitía reglas auto-declaradas sin un registro versionado.

---

## 5. `core/epistemic_certificate.py` (`d70bf3f`)

- **Capacidades a recuperar:**
  - Descomposición ortogonal en tres índices sobre el símplex: soporte (`evidence_support_index` / $b$), conflicto (`conflict_index` / $d$) e incertidumbre residual (`uncertainty_index` / $u$).
  - Distinción entre "falta de evidencia" (alta incertidumbre) y "contradicción fuerte" (alto conflicto).
  - Resúmenes criptográficos deterministas e integridad de snapshots mediante digest/HMAC.
- **Antipatrones a evitar:**
  - Etiquetar $b + a \cdot u$ (`projected_probability`) como probabilidad real o calibrada.
  - Equiparar una firma SHA-256 / Merkle root o HMAC con corrección semántica o validez factual de los datos firmados.
- **1. Entradas requeridas:** Índices agregados $(S, O)$ desduplicados por linaje, cobertura de verificación (`verification_coverage`), pasos críticos pendientes y calibrador empírico opcional (`EmpiricalCalibration.calibration_bins`).
- **2. Garantía exacta:** Separa explícitamente soporte evidencial, conflicto entre alternativas, ignorancia residual y cobertura de obligaciones verificables; vincula criptográficamente el veredicto de autorización al snapshot exacto del ledger y a la clave HMAC solo para integridad y no repudio. Devuelve siempre `calibrated_correctness_probability=None` salvo calibración empírica verificada fuera de muestra.
- **3. Supuestos:** Los índices $S$ y $O$ reflejan pesos de política/fiabilidad sobre el `DomainProfile`, no frecuencias poblacionales universales.
- **4. Condiciones de abstención:**
  - Cobertura incompleta o pasos críticos pendientes impiden reportar `supported` en evaluación o `PASS` en autorización (`uncertainty_index` elevado, `assessment="partially_supported"` o `"inconclusive"`).
  - Error formal (`refuted`) anula cualquier autorización con independencia de firma o pesos.
- **5. Pruebas disponibles y huecos detectados:** Probado en `tests/test_sprint3_improvements.py` y `tests/test_sprint5_end_to_end_unification.py`. Huecos históricos: `generate_epistemic_certificate` devolvía `projected_probability` sin calibrador real y no incluía cobertura de obligaciones ni vinculación transaccional con estado de ejecución (`IN_FLIGHT` / `UNCERTAIN`).

---

## Matriz de Compatibilidad y Migración en `claimlayer.v4`

| Módulo histórico (`ba81728` / `d70bf3f`) | Qué se recupera en `claimlayer.v4` | Nuevo módulo / API v4 destino | Garantía preservada de v4 |
|---|---|---|---|
| `core/inference_engine.py` | Traza detallada, intervalos de Allen, conflicto acotado, desduplicación por linaje | `v4/evaluation.py` + `v4/verifiers.py` (`TemporalVerifier`) | Comparación tipada via `FieldSpec`, `UnitRegistry` y fechas ISO |
| `core/graph_traverser.py` | Caminos multi-salto, T-norma de Gödel, descuento por salto, prevención de ciclos | `v4/justification.py` + `v4/evaluation.py` + `v4/verifiers.py` (`RelationVerifier`) | Reglas de composición declaradas en `DomainProfile`, cero inflación por linaje compartido |
| `core/aggregation_engine.py` | `sum`, `count`, `average`, `min`, `max` con control de población y eventos | `v4/verifiers.py` (`AggregationVerifier`) | Conversión exacta `UnitRegistry`, bloqueo ante unidades incompatibles o población incompleta |
| `core/epistemic_judge.py` | Clases epistémicas, cierre transitivo en DAG, exclusión de alegaciones no verificadas | `v4/justification.py` (`CLAIM_KINDS`, `effective_epistemic_kind`) + `v4/domain_profile.py` | Ninguna ponderación supera un fallo formal; la fiabilidad no la fija el LLM |
| `core/epistemic_certificate.py` | Descomposición soporte / conflicto / incertidumbre y digest/HMAC de auditoría | `v4/evaluation.py` (`ResponseAssessment`) + `v4/authorization.py` (`authorize()`) | `calibrated_correctness_probability=None` por defecto; firma HMAC vinculada al ledger SQLite |
