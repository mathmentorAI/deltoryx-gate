# PR-MCI · Claude Code · evaluación de 20 afirmaciones con el motor de derivación v4

**Pregunta.** ¿Puede el motor de inferencia de Deltoryx Gate verificar afirmaciones que Claude Code escribió en sus PRs, cuando la conclusión exige combinar varias piezas del diff y ninguna fuente la enuncia literalmente?

**Respuesta corta (ejecución en vivo del 21-nov-2026).** Todavía no. 0 de los 12 casos que exigían inferencia se resolvieron con una derivación válida. Hubo **0 falsas aceptaciones** (0/11), pero **1 refutación incorrecta** (1/16) y **1 PASS con justificación defectuosa**. 16 de 20 casos acabaron `unresolved`: 10 por defectos detectados por el verificador y 6 por salidas truncadas del planificador. El motor falla de forma segura (no da PASS a lo que no debe), pero con el catálogo actual no aporta inferencia defendible sobre código.

## 1. Estudio original y unidad de análisis

- **Corpus:** PR-MCI (Gong, Pinna, Bian, Zhang, MSR'26), commit `ac6bbf4deab044d86389a6e178441292306bedcf`, copia local en `Evidence Intelligence/caso_prmci_llm/PR-MCI`.
- **Población:** `Claude_Code`, 72 PRs en `validation/validation_results.csv` (59 aligned, 10 partial, 3 misaligned). Protocolo y resultados: `case_studies/pr_mci_claude_code/README.md`. Construcción de casos: `parse.py`. Métricas: `run.py`, `eval_v2vocab.py`, `check_claims.py`.
- **Afirmación evaluada en PR-MCI:** la descripción completa del PR (escrita por el agente Claude Code) frente a su parche. La respuesta correcta es la etiqueta de consenso de dos anotadores humanos (aligned/partial/misaligned). El verificador original disponía del cuerpo del PR y del parche del corpus.
- **Particiones:** los 72 PRs de Claude son la **fase 1 (desarrollo)**, usada para diseñar `rules_v2` y el vocabulario `code_change` v2.0. La partición reservada (fase 2, n=528) no contiene PRs de Claude. **Exposición previa: total**, lo que no favorece al motor nuevo, que no se desarrolló con PRs.
- **Unidad aquí:** una afirmación literal por PR (20 PRs distintos). PR-MCI no etiqueta afirmaciones individuales, así que los veredictos por afirmación son una **extensión**. En 7 casos (`provenance: replica`) la afirmación es la que los anotadores señalaron como causa de la etiqueta.

### Hallazgo sobre el corpus (base frente a head)

El parche de PR-MCI **no es el diff neto base→head**: concatena parches por commit, con estados intermedios. En `2975165913` (misaligned) y `3146561196` (partial), las notas de los anotadores justifican la etiqueta con archivos (12 y 7) que **no están en el diff neto** del PR. En esta evaluación la evidencia es el diff neto de GitHub (`pulls/{n}/files`, en caché en `github_cache/`), con `base_sha` y `head_sha` registrados. Esas dos etiquetas se marcan como `not_transferable`.

## 2. Selección y congelación (antes de la primera llamada)

- `select_and_fetch.py`: se incluyen todos los PRs no alineados que cumplen 1–25 archivos, churn ≤ 1500 y parche neto ≤ 60 kB (9 PRs), y 11 alineados en orden `random.Random(20261121)`, saltando los que no superan la comprobación posterior.
- **Prueba de dificultad, no muestra representativa:** 11 de 20 alineados frente al 82 % del corpus. Resultados no extrapolables a tasas del corpus.
- **Manifiestos** (`frozen/`, commit `52d5527`):
  - `inputs_manifest.json` `710b8b28…`: afirmación y fuentes. Documentos neutros `diff_summary` y `file_NN`; sin cuerpo del PR, sin etiquetas ni nombres que revelen la respuesta.
  - `expected_manifest.json` `fd0382c9…`: veredicto, alternativas aceptables, evidencia mínima, razonamiento aceptable y situación.
- **Situaciones:** S1 (se sostiene enlazando ≥2 hechos): 9 · S2 (contraejemplo concreto): 4 · S3 (solo se sostiene una versión más débil): 7.
- **Veredictos esperados:** 9 `derived`, 4 `refuted`, 7 `insufficient_evidence`. 12 casos marcados como «exigen inferencia».

## 3. Camino de producto ejecutado

- **Entrada:** `POST /api/v1/v4/claims/verify` en `uvicorn api.main:app` (plataforma `4fd6ed9`, rama `feat/v4-claims-verify-api`).
- **Motor:** claim-layer `b7295f7` (`feat/v4-derivation-verifier`), sin cambios locales en `api/` ni `src/`.
- **Planificador:** `nvidia/nemotron-3-ultra-550b-a55b`, temperatura 0, `max_tokens` 2048, thinking desactivado, 2 reintentos de transporte, 1 reintento de reparación. Prompt `v4-derivation-planner-2` (`ba80052a…`). Catálogo `deltoryx_v4_core` 1.0.0 (`58d83b0d…`). Perfil de unidades `auto`, que eligió `standard` porque las afirmaciones están en inglés.
- **Sin cambios durante la medición:** ningún operador, adaptador, regla ni ajuste de prompt. Configuración idéntica para los 20 casos.

## 4. Resultados por caso

| Caso | PR @ head | Afirmación (literal) | Esperado | Observado | Ruta | Evidencia decisiva | Operación comprobada | Justificación |
|---|---|---|---|---|---|---|---|---|
| C01 | `2886171819` @ `7b58015ee861` | The fix ensures this_order is always at least 1 | derivada (PASS) | sin resolver (unsupported_operation_or_rule) | ESCALATE | + this_order = max(1, min(...)) partido en 3 líneas del diff | identity (propuesta); sugerida max_lower_bound | abstención del verificador |
| C02 | `2889875255` @ `6738cfb15981` | Integration with threat modeling, attack tree, mitigations, and tes… | refutada (RETRY) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C03 | `2964743634` @ `00ec1b658263` | Add TTY environment check to `cargo insta review` | derivada (PASS) | derivada (verified_derivation) | PASS | c1 = identity(p1) con p1 = línea +CHANGELOG (texto del autor); p2 = hunk de cli.rs con … | identity x2 (sin cálculo) | ⚠️ etiqueta OK, justificación defectuosa |
| C04 | `2975165913` @ `7c98b89f2eab` | Updated all links pointing to /notebooks/Commands/ to now point to … | insuficiente (ESCALATE) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C05 | `2986431376` @ `af12d91e9e14` | Fix heading hierarchy to follow consistent H1 -> H2 -> H3 structure | insuficiente (ESCALATE) | sin resolver (unsupported_operation_or_rule) | ESCALATE | 7 pares -/+ de cabeceras (### -> ##, #### -> ###) | identity x7; sugerida heading_hierarchy_consistency | abstención del verificador |
| C06 | `3037457814` @ `291dea32a330` | This PR adds the Language Server (https://github.com/isaacphi/mcp-l… | derivada (PASS) | sin resolver (premise_grounding_defect) | RETRY | + - [isaacphi/mcp-language-server](...) en docs/code-analysis--quality.md | identity x3 | abstención del verificador |
| C07 | `3047699666` @ `d6e0df6d133b` | Added test file `frontend/packages/db/supabase/tests/database/03-or… | derivada (PASS) | sin resolver (premise_grounding_defect) | RETRY | SELECT plan(8) + ruta del archivo añadido | identity x2 (sin eq/recuento) | abstención del verificador |
| C08 | `3069923749` @ `ee87963235a0` | Adds fields for `cargo-insta` version and the output of `cargo inst… | derivada (PASS) | sin resolver (premise_grounding_defect) | RETRY | + id: cargo-insta-version / cargo-insta-command-version | identity x2 | abstención del verificador |
| C09 | `3119417980` @ `7c14c023d687` | Add includeStackTrace option to reduce LLM token usage by 80-90% | insuficiente (ESCALATE) | sin resolver (premise_grounding_defect) | RETRY | intentó usar las cadenas '⚠️ ... save 80-90% tokens' escritas por el autor | identity | abstención del verificador |
| C10 | `3132739442` @ `a7797dc5ccbe` | Consolidate algorithmic constants into comprehensive constants header | insuficiente (ESCALATE) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C11 | `3135266974` @ `d9e86877a77b` | Implements full `Learner` protocol (`partial_fit`, `sample`, `predi… | derivada (PASS) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C12 | `3146561196` @ `dc77327c65ad` | Build will fail if versions are inconsistent or package-lock.json i… | insuficiente (ESCALATE) | sin resolver (malformed_proposal) | RETRY | — | identity x3 | abstención del verificador |
| C13 | `3152317896` @ `f4ee968ed9cc` | Implemented `porto create-merchant` command for creating merchant a… | refutada (RETRY) | sin resolver (unsupported_operation_or_rule) | ESCALATE | .command('create-account [alias: ca]', ...) en src/cli/index.ts | identity x3; sugeridas command_alias_equivalence, account_type_classification | abstención del verificador |
| C14 | `3196885081` @ `56c48436d5f1` | mcp/: Server registry & orchestration (207 lines total) | derivada (PASS) | sin resolver (malformed_proposal) | RETRY | @@ -0,0 +1,77 @@ y +1,132 (sin salto final) | identity (sin add) | abstención del verificador |
| C15 | `3197644488` @ `b52399fca463` | docs: add release notes for v0.0.4 | derivada (PASS) | refutada (verified_refutation) | RETRY | Changed files: - docs/release-notes/v0.0.4.md (added) | identity | ❌ refutación incorrecta |
| C16 | `3204772289` @ `2bde5d2d8a50` | Added GUI Controller functionality to LangConnect for managing MCP … | refutada (RETRY) | insuficiente (corpus_insufficient_evidence) | ESCALATE | Files changed: 0 / Lines added: 0 / Lines deleted: 0 | ninguna (missing_premises) | ✅ correcta |
| C17 | `3206405036` @ `7f9f9949d2f9` | Added comprehensive test for undefined opts scenario | insuficiente (ESCALATE) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C18 | `3214876564` @ `9fab87c771ca` | Uses SHA-256 hash of plan content for consistent keys | refutada (RETRY) | insuficiente (corpus_insufficient_evidence) | ESCALATE | + const memoryKey = `action_${responseAction}_${uuidv4()}`; ausencia de sha/hash | ninguna (missing_premises) | ✅ correcta |
| C19 | `3245466752` @ `2bbe1568cabf` | This PR fixes the output to include the server name as the root key | derivada (PASS) | sin resolver (malformed_proposal) | RETRY | — | — | error operativo (salida truncada) |
| C20 | `3246950577` @ `15be1f698ea5` | This resolves an issue where running `pdm export --lockfile pylock.… | insuficiente (ESCALATE) | sin resolver (premise_grounding_defect) | RETRY | — | identity x3 | abstención del verificador |

## 5. Métricas (denominadores explícitos)

**Matriz de confusión de veredictos (filas = esperado, columnas = observado):**

| Esperado \ Observado | derived | refuted | insufficient_evidence | unresolved | Total |
|---|---:|---:|---:|---:|---:|
| derived | 1 | 1 | 0 | 7 | 9 |
| refuted | 0 | 0 | 2 | 2 | 4 |
| insufficient_evidence | 0 | 0 | 0 | 7 | 7 |

| Métrica | Valor |
|---|---|
| Veredicto en el conjunto aceptable | 3/20 |
| Correctamente verificado (veredicto aceptable y justificación válida) | 2/20 (C16, C18; ambos abstenciones aceptables) |
| Acierto de etiqueta con justificación defectuosa | 1/20 (C03) |
| Falsas aceptaciones (PASS sobre referencia refutada o insuficiente) | **0/11** |
| Refutaciones incorrectas (refuted sobre referencia no refutada) | **1/16** (C15) |
| Abstenciones justificadas / abstenciones totales | 2/18 |
| Casos resolubles (derived o refuted esperado) que acabaron en abstención o unresolved | 11/13 |
| Abstenciones del verificador (defecto de premisa, de objetivo u operación no soportada) | 10/20 |
| Errores operativos (salida del planificador truncada, resultado final) | 6/20 |
| Errores del proveedor | 0/20 |
| Coincidencia de ruta PASS/RETRY/ESCALATE | 3/20 (2 casuales) |
| Intentos con salida malformada | 15/36 |
| Casos con inferencia resueltos con derivación válida | **0/12** |

**Qué verificó Deltoryx y qué aceptó de Nemotron.** En los resultados finales se admitieron 12 premisas. Deltoryx comprobó la cita literal, el documento y la revisión de las 12. El **valor** de las 12 (siempre booleano o texto, nunca numérico) lo asignó la interpretación de Nemotron. La única operación ejecutada fue `identity` (3 veces), que reafirma una premisa sin calcular nada. No se ejecutó ninguna operación aritmética ni de comparación, ni siquiera en C07 (8 tests) o C14 (207 líneas), donde la derivación era numérica.

**Coste** (n=20; mediana [mínimo–máximo]):

| | Mediana | Rango |
|---|---:|---|
| Tokens por afirmación | 14 965 | 3 214 – 46 560 |
| Latencia de planificación | 27,8 s | 4,6 – 55,9 s |
| Tiempo del verificador | 7,5 ms | 0 – 53 ms |
| Extremo a extremo (cliente HTTP) | 27,8 s | 4,6 – 55,9 s |

36 llamadas en total; 16 de los 20 casos usaron el reintento de reparación. 343 771 tokens en total. Ejecución completa de 11 min (incluye el evaluador anterior). 0 errores del proveedor.

## 6. Dónde está la frontera (hallazgos centrales)

1. **`identity` convierte una interpretación en una «derivación».** En C03 la conclusión es `identity(p1)` sobre la línea del CHANGELOG, texto escrito por el autor. En C15 es `identity` sobre «added = true». Deltoryx comprueba que la cita existe, pero el significado lo pone Nemotron. Es el límite que el experimento debía hacer visible.
2. **Comparador sin tipos.** C15 refuta «docs: add release notes for v0.0.4» porque compara `True` con `'v0.0.4'`. Valores incomparables deberían dar `unresolved`, no `verified_refutation`.
3. **La correspondencia por cláusula, pensada para prosa, bloquea la inferencia sobre código.** C06, C07 y C08 fallan porque la ruta del archivo (cabecera) y el contenido (línea del hunk) están en cláusulas distintas, cuando ese enlace es justamente la inferencia necesaria. En C03, a la inversa, admitió una entidad que no estaba en la cita.
4. **Las citas multilínea del diff no son literales.** El formato del diff (prefijos `+` e indentación) hace que Nemotron recomponga expresiones partidas en varias líneas (C01, C11, C13, C20), y la comprobación literal las rechaza.
5. **Truncamiento.** Con `max_tokens` = 2048, 6 casos acaban sin propuesta. Es un límite de configuración, no de evidencia.
6. **Comprobaciones del objetivo demasiado rígidas.** C14 (207, 'lines') y C12 se rechazan por desajuste de `claim_target`; en C14 el objetivo era correcto, pero 'lines' no es una unidad declarada.
7. **Base frente a head.** No hay ninguna regla que distinga líneas `-` de líneas `+`. En esta ejecución ninguna premisa usó una línea solo de base para una conclusión sobre head (C05 citó pares `-`/`+` de forma legítima), así que el riesgo existe pero no se materializó.
8. **Contraejemplo encontrado pero no formalizable.** En C18 Nemotron identificó el `uuidv4` frente a SHA-256 y en C16 el diff vacío, pero el motor no tiene forma de convertirlos en refutación. Son intentos fallidos de refutar, no refutaciones demostradas.
9. **Operaciones que pidió Nemotron:** `max_lower_bound`, `heading_hierarchy_consistency`, `command_alias_equivalence`, `account_type_classification`. Quedaron registradas como limitaciones y no se ejecutaron. La de «equivalencia de alias» (C13) habría acercado create-account a create-merchant: aceptarla sin validar habría sido peligroso.
10. **Límite de la API:** solo se devuelve la propuesta del último intento. Del primer intento quedan el veredicto y los defectos (`attempt_traces`), pero no la propuesta original.

## 7. Ejemplos

**Inferencia que aportó valor:** no se observó ninguna derivación válida. El caso más cercano es C18: Nemotron citó correctamente `const memoryKey = \`action_${responseAction}_${uuidv4()}\`` y la ausencia de hash. El motor lo dejó en `insufficient_evidence` / ESCALATE: una abstención correcta, pero sin refutación.

**Fallo más relevante para la defendibilidad (C15, PR `3197644488` @ `b52399fca463`):**
- Fuente: `Changed files: - docs/release-notes/v0.0.4.md (added, +55/-0)`.
- Propuesta: `claim_target.value = "v0.0.4"`, premisa `value = true`, paso `identity(p1)`.
- Motor: `True ≠ 'v0.0.4'` → `refuted`, `verified_refutation`, RETRY.
- Referencia: `derived`. Si este resultado se usara como bloqueo, rechazaría un PR correcto con un recibo de «refutación verificada».

**PASS con justificación defectuosa (C03):** la afirmación es cierta y el código lo muestra (`if !term.is_term() { return Err(...) }`), pero la conclusión formal es `identity` sobre la línea +CHANGELOG del autor. Si el CHANGELOG hubiera mentido, el motor habría dado el mismo PASS.

## 8. Comparación con los evaluadores anteriores (misma afirmación y misma evidencia)

- **`gate_v4_vocab2`** (Nemotron en vivo, vocabulario v2.0): 19/20 afirmaciones `unverifiable` (`describes_behavior`) y 1 `supported` (C07, `adds_test`: sostiene que el test existe, no los 8 casos). Veredicto de PR: 19 CONSISTENT y 1 INCONSISTENT (C16, `empty_diff`). Como abstención por afirmación es honesto. Usado como bloqueo de PR, daría CONSISTENT en 10 de las 11 referencias refutadas o insuficientes.
- **`rules_v2_postreview_code_config`:** su contrato es un cuerpo de PR completo. Con una sola frase dispara `P_placeholder` en 8/20 casos, así que **la comparación no es válida y no se publica ninguna cifra de mejora**.
- **Caso a caso frente a `gate_v4_vocab2`:** el motor nuevo obtiene la etiqueta correcta en C03 (pero con justificación defectuosa), pierde C07 (el anterior dio `supported` y el nuevo `unresolved`) e **introduce** el error C15. En C16 y C18 ambos se abstienen; el nuevo enruta a ESCALATE con una explicación del motivo. Sin mejora neta demostrable.

## 9. Reproducir

```bash
# 1. Selección y caché (la caché incluida evita llamadas a GitHub)
PRMCI_DIR=".../PR-MCI" python3 claim-layer/case_studies/pr_mci_claude_derivation/select_and_fetch.py
# 2. Manifiestos (ya congelados; el script no sobrescribe)
PRMCI_DIR=".../PR-MCI" python3 claim-layer/case_studies/pr_mci_claude_derivation/build_manifests.py
# 3. Ejecución en vivo (NVIDIA_API_KEY en el entorno o en .env)
.venv/bin/python claim-layer/case_studies/pr_mci_claude_derivation/run_eval.py
# 4. Puntuación
python3 claim-layer/case_studies/pr_mci_claude_derivation/score.py
```

Artefactos: `results/live_results.json` (respuestas completas, propuestas y grafos), `results/old_eval_cache.jsonl`, `results/justification_review.json` (revisión manual) y `results/scored.json`. Ninguno contiene credenciales.

## 10. Respuesta a la pregunta de decisión

**¿Qué hallazgos sobre los PRs de Claude puede defender hoy Deltoryx?** Solo dos, y ambos son abstenciones correctas: C16 (el PR no cambia nada) y C18 (no hay hash en el diff), las dos sin PASS y enviadas a revisión. Ninguna afirmación del estudio quedó derivada ni refutada con una cadena que Deltoryx pudiera reproducir. Sí puede defender que **no aceptó ninguna afirmación falsa o sin respaldo** (0/11).

**¿Qué depende del juicio de Nemotron?** Todo lo que pasó de la cita al significado. Las 12 premisas admitidas tienen el texto comprobado por Deltoryx y el valor puesto por Nemotron. Las 3 operaciones ejecutadas fueron `identity`.

**¿Dónde habría sido peligroso aceptar el resultado?**
- C15: refutación «verificada» de una afirmación cierta, por un fallo del comparador.
- C03: PASS correcto por casualidad, sostenido por texto del autor.
- C13: si se hubiera aceptado la regla `command_alias_equivalence` propuesta por Nemotron sin validar, create-account habría pasado por create-merchant.
- Además, como bloqueo de PR, `gate_v4_vocab2` habría marcado CONSISTENT 10 de las 11 afirmaciones no respaldadas.

---

# Segunda medición (21-nov-2026, 19:08 UTC): motor de reconstrucción con operaciones propuestas

**Configuración.**
- Código: plataforma `b8108d5` y claim-layer `fba49d2`, sin cambios locales.
- Prompt `v4-derivation-planner-3`: Nemotron puede proponer operaciones (`proposed_operations`).
- Política de `PASS` mínimo B y `max_tokens` 8192.
- Mismos manifiestos congelados (`710b8b28…` / `fd0382c9…`) y el mismo expected.
- Resultados: `results/live_results_v3.json`. Revisión manual: `results/justification_review_v3.json`. Métricas: `results/scored_v3.json` (`python3 score.py --live results/live_results_v3.json --review results/justification_review_v3.json --out results/scored_v3.json`).

## Comparación con la primera medición

| Métrica (denominador) | v1 (prompt v2, 2048 tokens) | v2 (prompt v3, reconstrucción) |
|---|---|---|
| Correctamente verificado (20) | 2 | 2 |
| Etiqueta correcta con justificación defectuosa (20) | 1, **con PASS** (C03) | 1, **ESCALATE por grado E** (C03) |
| Falsas aceptaciones (11) | 0 | 0 |
| Refutaciones incorrectas (16) | 1 (C15, bool frente a texto) | 1 (C15, texto frente a texto, **grado A**) |
| Inferencias resueltas con derivación válida (12) | 0 | 0 |
| Rutas PASS (20) | 1 | **0** |
| Errores operativos por salida truncada (20) | 6 | **0** |
| Abstenciones del verificador (20) | 10 | 16 |
| Intentos con salida malformada | 15/36 | **3/35** |
| Casos con operaciones propuestas por Nemotron (20) | — | 1 (C05, 2 afirmadas, 0 expresiones) |
| Operaciones ejecutadas | identity ×3 | identity ×4 |
| Tokens por afirmación, mediana [rango] | 14 965 [3 214–46 560] | 14 278 [3 706–47 173] |
| Planificación, mediana [rango] | 27,8 s [4,6–55,9] | 26,2 s [2,8–169,9] |
| Verificador, mediana [rango] | 7,5 ms [0–53] | 20,3 ms [0,6–81] |

La matriz de confusión de veredictos es idéntica en ambas mediciones (derived: 1 derived, 1 refuted, 7 unresolved; refuted: 2 insufficient, 2 unresolved; insufficient: 7 unresolved). Lo que cambia es la ruta: ya no hay ningún `PASS`, y los 6 casos que antes perdían la respuesta por truncamiento ahora llegan a un defecto de premisa diagnosticable. Los picos de latencia de C04 (170 s) y C10 (119 s) se deben a respuestas mucho más largas con el nuevo presupuesto de tokens.

## Hallazgos nuevos de la segunda medición

1. **Un grado A no garantiza que se compare el mismo atributo (C15).** «added» (estado del archivo) frente a «v0.0.4» (versión) son del mismo tipo, texto, y la premisa es literal, así que sale una refutación verificada con grado A. Falta exigir que la conclusión calculada y el objetivo de la afirmación se refieran al mismo atributo antes de refutar.
2. **Evidencia circular del autor (C14).** Nemotron citó `PR_TEMPLATE.md`, un archivo añadido por el autor en el mismo PR que repite la descripción. La premisa era «literal», lo que habría dado grado A. Solo la comprobación de agregación del núcleo («total») evitó un `PASS`. La protección contra usar la afirmación como premisa solo detecta copias exactas, no reformulaciones del autor dentro de las fuentes.
3. **Nemotron casi no propone operaciones.** Lo hizo en 1 de 20 casos (C05), sin expresiones ejecutables, aunque disponía de `max`, `min` y `add` (C01 y C14 los necesitaban). Además, las bloqueó la prioridad heredada de `suggested_rules`.
4. **Regresión en el benchmark temporal con el prompt v3: 13/14** (`benchmarks/v4_derivation_planner_live_results_v3.json`). NT3 sale `unsupported_operation_or_rule` en lugar de `insufficient_evidence` porque Nemotron añadió una `suggested_rule` que no usa, y esta tiene prioridad sobre la premisa ausente. Ruta ESCALATE en ambos casos y 0 falsas aceptaciones.
5. **Siguen sin resolverse** las citas multilínea de diffs (C01, C05, C13), la correspondencia por cláusula entre la ruta del archivo y la línea (C06, C07, C08) y los objetivos reescritos como condición (C12).

## Respuesta actualizada a la pregunta de decisión

Deltoryx sigue pudiendo defender las mismas dos abstenciones (C16, C18) y **0 aceptaciones indebidas**. Ahora además ninguna conclusión que dependa de una interpretación de Nemotron llega a `PASS`. Lo que **no** se puede defender todavía es que un grado A signifique «la conclusión responde a la afirmación»: C15 y C14 muestran dos vías (atributo distinto y evidencia circular del autor) por las que una cadena con todos sus eslabones literales sigue siendo inválida.

---

# Tercera medición, M3 (etapa 1: vinculación, evidencia que reformula la afirmación, recuentos reproducibles)

**Configuración.**
- Revisión ejecutada: plataforma `48d936c` y claim-layer `94e3d4f`, sin cambios locales.
- Prompt `v4-derivation-planner-3`; catálogo con `count_lines`.
- PR: 21-nov 20:41 UTC (`results/live_results_stage1.json`). Temporal: 20:43 UTC (`benchmarks/v4_derivation_planner_live_results_stage1.json`).
- Mismos manifiestos congelados. Revisión manual: `results/justification_review_stage1.json`. Métricas: `results/scored_stage1.json`.

| Métrica (denominador) | M1 | M2 | **M3** |
|---|---|---|---|
| Inferencias resueltas con derivación válida (12) | 0 | 0 | **0** |
| Falsas aceptaciones (11) | 0 | 0 | **0** |
| Refutaciones incorrectas (16) | 1 | 1 (grado A) | **0** |
| Correctamente verificados (20) | 2 | 2 | 2 |
| Abstenciones justificadas / abstenciones totales | 2/18 | 2/18 | 2/18 |
| Abstenciones del verificador (20) | 10 | 16 | 17 |
| Errores operativos (20) | 6 | 0 | 0 |
| Rutas PASS (20) | 1 | 0 | 0 |
| Intentos malformados | 15/36 | 3/35 | 6/37 |
| Tokens totales · por inferencia válida | 343 771 · indefinido | 378 908 · indefinido | 404 179 · **indefinido (0 inferencias válidas)** |
| Extremo a extremo total · por inferencia válida | 602,8 s · indefinido | 811,0 s · indefinido | 913,3 s · **indefinido** |
| Planificación por intento, mediana [rango] | — | — | ver `scored_stage1.json` → `per_attempt` |
| Benchmark temporal | 14/14 (v2) | 13/14 | **14/14** (1 reintento) |

Los tiempos y tokens de cada intento (primer intento y reintento) están en `scored_*.json → per_attempt.by_case`, para que el coste incurrido sea visible aunque el caso no se resuelva.

## Cambios por caso M2 → M3

| Caso | M2 | M3 | Qué cambió |
|---|---|---|---|
| C15 | refuted / RETRY, grado A (incorrecta) | unresolved / RETRY, `conclusion_binding_defect` | Corregido: el atributo de la premisa («file status») no es el afirmado. |
| C14 | unresolved (agregación tras citar `PR_TEMPLATE.md`) | unresolved, citas no literales | Nemotron propone `add(77, 132)` sobre las cabeceras de hunk; las citas multilínea no son literales. Riesgo detectado: 209 ≠ 207 habría refutado una afirmación cierta (convenio `wc -l`). |
| C11 | unresolved | unresolved | Primer uso de `count_lines`, sobre una premisa con valor lista y cita no literal. |
| C05 | unresolved (bloqueado por `suggested_rules`) | unresolved, defecto de premisa | Ya no lo bloquea la regla sugerida sin usar; primera operación propuesta con expresión ejecutable, sobre valores interpretados («H2»). |
| C17 | unresolved, defecto de premisa | unresolved, objetivo malformado | Variación del planificador. |
| C03 | derived / ESCALATE (E) | igual | Sin PASS. |
| C16, C18 | insufficient / ESCALATE | igual | Abstenciones correctas. |
| Resto (C01, C02, C04, C06–C10, C12, C13, C19, C20) | unresolved | unresolved | Defectos de anclaje: citas multilínea, correspondencia entre cláusulas, valores interpretados. |

**Benchmark temporal.** NT3 vuelve a «evidencia insuficiente» porque las reglas sugeridas sin usar ya no bloquean: 14/14 y 0 falsas aceptaciones.

**Diagnóstico de por qué Nemotron no produce derivaciones válidas.**
1. **Copia bloques de código o diff reformateados** (indentación, prefijos) y la cita deja de ser literal (C01, C11, C13, C14).
2. **Asigna valores interpretados** («H2», `true`, listas) en lugar de valores que aparecen en la cita.
3. **Enlaza la entidad con el contenido** cuando la ruta del archivo está en la cabecera y el valor en otra línea, y la comprobación de cláusula lo rechaza (C06–C08, C19).

El análisis manual de la etapa 2 (`stage2_manual/`) muestra que, con citas correctas, el contrato actual ya expresa y verifica derivaciones de este tipo.
