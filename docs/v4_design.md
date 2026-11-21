# ClaimLayer v4 (4.0.0a1, experimental)

24 de octubre de 2026. Motor nuevo en `src/claimlayer/v4/`. No modifica la v3 (`claimlayer.ClaimLayer`) ni la v2 (`claimlayer.legacy`), que siguen disponibles para comparar.

## Principio

### Dirección de producto acordada el 20 de noviembre de 2026

El objetivo es que la capacidad de verificar afirmaciones crezca al ritmo de su generación. El usuario entrega fuentes y una afirmación nueva. Nemotron propone las premisas, sus referencias y una composición de operaciones para comprobarla. Deltoryx contrasta cada premisa con la fuente y su contexto, ejecuta los pasos admitidos y compara la conclusión obtenida con la afirmación original. La explicación debe permitir reproducir el cálculo e identificar qué supuesto cambiaría el resultado.

La unidad que el modelo construye para cada consulta es una derivación en `JustificationGraph`. El operador controla el significado y la versión de las operaciones ejecutables. Una regla nueva sugerida por el modelo se registra como supuesto pendiente de justificación. El resultado distingue deducción exacta bajo premisas explícitas, soporte evidencial y probabilidad calibrada cuando exista calibración aplicable. La decisión epistémica y el enrutamiento operativo se comunican por separado.

**Primer hito de aceptación.** Todos los casos deben pasar por la misma entrada pública de fuentes y afirmación, con el mismo constructor de propuestas y sin un grafo preparado para cada caso.

| Fuente | Afirmación | Resultado esperado |
| --- | --- | --- |
| Hoy es viernes y quedé con ella dos días antes. | La cita fue el miércoles. | Derivada. El evento se ancla al viernes y el desplazamiento es de dos días hacia atrás. |
| La misma fuente | La cita fue el jueves. | Refutada mediante el mismo cálculo, que devuelve miércoles. |
| Quedé con ella dos días antes. | La cita fue el miércoles. | Indeterminada. Falta el día de referencia. |
| Hoy es viernes. Su cumpleaños fue el martes y quedé con ella dos días antes de su cumpleaños. | La cita fue el domingo. | Derivada usando el cumpleaños como ancla, incluso al cruzar el límite semanal. |

La implementación se evaluará también con paráfrasis reservadas y propuestas con el ancla o el signo alterados. Se registrarán afirmaciones correctamente verificadas por minuto, cobertura, aceptaciones erróneas, coste de inferencia y minutos de revisión humana. El tiempo incluirá extracción, construcción del grafo, comprobación y reintentos. Los benchmarks por dominio aportan evidencia de progreso cuando ejercitan este recorrido.

### Dirección actualizada el 21 de noviembre de 2026: reconstrucción trazable (Evidence Intelligence)

Una IA produce un resultado a partir de unas fuentes. Deltoryx Gate lo trata como afirmación a justificar y pide a Nemotron que lo **reconstruya** desde las fuentes con una cadena de operaciones explícitas. **Nemotron puede proponer las operaciones que necesite**; el producto es la reconstrucción trazable y su grado de fiabilidad. Esto sustituye el criterio anterior «solo operaciones del catálogo de la aplicación». Se mantiene una frontera: Deltoryx no ejecuta código generado por el modelo. Una operación propuesta es un dato: o bien una expresión en un lenguaje cerrado que interpreta y re-ejecuta Deltoryx, o bien una operación sin definición ejecutable cuyo resultado queda registrado como afirmación del modelo.

| Grado | Eslabón | Quién garantiza el significado |
| --- | --- | --- |
| A | Premisa con valor comprobado en la cita + operación del núcleo | Texto fuente y código de Deltoryx |
| B | Regla declarada por la aplicación | El operador |
| C | Operación propuesta por Nemotron como expresión sobre primitivas, ejecutada por Deltoryx | Deltoryx la ejecuta; Nemotron eligió la composición |
| D | Operación propuesta sin definición ejecutable (resultado afirmado por el modelo) | Solo Nemotron, registrada |
| E | Premisa cuyo valor la cita no fija (interpretación del modelo) | Solo Nemotron |

El grado de una conclusión es el de su eslabón más débil. La aplicación fija el grado mínimo para `PASS` (`ReliabilityPolicy`, por defecto B; en la API, `DELTORYX_PASS_MIN_GRADE`). Por debajo de ese mínimo el dictamen se conserva y la ruta pasa a `ESCALATE` con el eslabón débil identificado. Las operaciones propuestas pasan también por las comprobaciones obligatorias del núcleo (por ejemplo, la completitud de población en agregaciones).

**Implementado** (`v4/reconstruction.py`, prompt `v4-derivation-planner-3`; `v4-derivation-planner-2` se conserva para reproducir el 14/14):
- `proposed_operations` en la propuesta del planificador;
- intérprete cerrado con lista blanca de primitivas y límites de profundidad y tamaño;
- clasificación de premisas en literales o interpretadas;
- bloque `reliability` en la biblioteca, el CLI y la API;
- comparación con tipos (los valores incomparables dan `unresolved / incomparable_conclusion`, nunca `verified_refutation`);
- presupuesto de salida del planificador configurable (8192 por defecto en la API).

**Motivación medida.** La primera medición sobre 20 afirmaciones reales de PRs de Claude Code (`case_studies/pr_mci_claude_derivation`) dio 0/12 inferencias resueltas con una derivación válida, una refutación incorrecta por comparación sin tipos y un `PASS` sostenido por `identity` sobre texto del autor. En el replay de esas propuestas con el motor nuevo no hay ningún `PASS` indebido: C03 queda en grado E (`ESCALATE`) y C15 en `incomparable_conclusion`. Las 14 propuestas del benchmark temporal mantienen su dictamen con grado A.

**Pendiente:**
- segunda medición en vivo con el prompt v3;
- promoción de operaciones (de D a C y de C a B) mediante aprobación, tests o uso acumulado;
- operaciones de dominio para código: base frente a head, llamadas, ejecución de tests;
- extractor de números que no reconoce una cifra seguida de punto final («es 9.»);
- evaluación independiente y prueba bajo carga.

### Evidencia del primer hito (21 de noviembre de 2026): viabilidad de extremo a extremo

**Qué se ha demostrado.** El recorrido completo funciona con un modelo real. El consumidor entrega fuentes y una afirmación, Nemotron propone una derivación, Deltoryx comprueba cada premisa y ejecuta las operaciones, y la respuesta separa el dictamen sobre la afirmación de la decisión operativa. Todos los casos entraron por la misma interfaz, con el mismo prompt (`v4-derivation-planner-2`, digest `ba80052a…`) y el mismo catálogo de operaciones. Ningún caso usó un grafo preparado ni una regla elegida por caso.

**Benchmark en vivo: 14/14.** Modelo `nvidia/nemotron-3-ultra-550b-a55b` (temperatura 0, `max_tokens` 2048, sin razonamiento extendido). Datos completos, con propuesta original, grafo admitido, resultado recalculado y motivos, en `benchmarks/v4_derivation_planner_live_results.json`.

| Caso | Afirmación | Esperado | Observado | Conclusión calculada | Decisión |
| --- | --- | --- | --- | --- | --- |
| T1 | La cita fue el miércoles. | derivada | derivada | miércoles | PASS |
| T2 | La cita fue el jueves. | refutada | refutada | miércoles | RETRY |
| T3 | La cita fue el miércoles. (sin ancla) | evidencia insuficiente | evidencia insuficiente | — | ESCALATE |
| T4 | La cita fue el domingo. | derivada | derivada | domingo | PASS |
| T5 | La cita fue el miércoles. (ancla cumpleaños) | refutada | refutada | domingo | RETRY |
| NT1 | Margen restante 350 EUR | derivada | derivada | 350 EUR | PASS |
| NT2 | Margen restante 450 EUR | refutada | refutada | 350 EUR | RETRY |
| NT3 | Margen restante 350 EUR (falta el gasto) | evidencia insuficiente | evidencia insuficiente | — | ESCALATE |
| H1 | La reunión tuvo lugar el martes. | derivada | derivada | martes | PASS |
| H2 | El encuentro ocurrió el lunes. (sin ancla) | evidencia insuficiente | evidencia insuficiente | — | ESCALATE |
| H3 | El acuerdo se firmó el sábado. (dos anclas) | derivada | derivada | sábado | PASS |
| H4 | El acuerdo se firmó el lunes. (dos anclas) | refutada | refutada | sábado | RETRY |
| H5 | La entrega venció el viernes. | refutada | refutada | sábado | RETRY |
| H6 | Quedan 750 USD en la Fase Beta. | derivada | derivada | 750 USD | PASS |

T1–T5 y NT1–NT3 son los casos de aceptación (8/8, ejecutados el 20 de noviembre). H1–H6 son paráfrasis reservadas (6/6, ejecutadas el 21 de noviembre). Se declararon antes de ejecutarlas, con el prompt ya fijado, siguiendo los ejes de variación descritos en `paraphrase_selection_protocol`. No hubo aceptaciones erróneas, errores del proveedor ni reintentos. Coste medio por afirmación: 1 llamada, unos 3.480 tokens, unos 6,7 s de planificación y unos 10 ms de verificación. Rendimiento en serie: unas 8,9 afirmaciones verificadas por minuto.

**API de la plataforma en vivo: 4/4.** `POST /api/v1/v4/claims/verify` se ejecutó sobre uvicorn real, por HTTP, con Nemotron real, en los commits limpios `fcca663` (plataforma) y `4f52c54` (claim-layer). Script: `scripts/live_claims_api_check.py`. Respuestas: `evidence/v4_claims_api_live_run.json` en el repositorio de la plataforma.

| Caso | Dictamen | Decisión | Justificación devuelta |
| --- | --- | --- | --- |
| H3, dos anclas, la referencia es el lanzamiento | derivada (sábado) | PASS | ancla «el lanzamiento oficial ocurrió el lunes» `doc_1[27:66]`; desplazamiento −2 de `doc_1[68:126]`; `weekday_shift = sábado` |
| H4, la misma fuente, se afirma lunes | refutada (calcula sábado) | RETRY | el mismo cálculo; el diagnóstico señala que «Ayer era miércoles» − 2 daría lunes, pero es un ancla incorrecta |
| T3, referencia sin ancla | evidencia insuficiente | ESCALATE | falta la premisa `anchor_weekday`; no se calcula nada |
| H6, resta entre documentos | derivada (750 USD) | PASS | 2.400 USD de `contrato[0:48]`, 1.650 USD de `factura[0:45]`; `sub = 750 USD`, cifra que no aparece en ninguna fuente |

En los cuatro casos `calibrated_correctness_probability` es `null` y `calibration_status` es `uncalibrated`. La garantía que se declara es una deducción exacta bajo premisas explícitas, cuya fidelidad se comprueba contra citas literales. No se trata de una probabilidad.

**Qué no demuestra.** Con 14 casos se comprueba que la arquitectura funciona de extremo a extremo con un modelo real, no que escale ni que generalice. Las paráfrasis reservadas las redactó la misma sesión que fijó el prompt: son reservadas respecto a la ejecución, pero no las eligió alguien independiente. El rendimiento medido es en serie, con un único proveedor, y el coste está dominado por la latencia de planificación.

**Trabajo futuro explícito.**
1. *Evaluación independiente.* Un conjunto de afirmaciones redactado y etiquetado por terceros que no hayan visto el prompt ni el catálogo, con más dominios y operaciones, y que mida cobertura, aceptaciones erróneas y minutos de revisión humana.
2. *Prueba bajo carga.* Ejecución concurrente para medir el rendimiento sostenido (afirmaciones verificadas por minuto), la latencia p95/p99, los límites del proveedor, el coste por afirmación y el comportamiento ante fallos y reintentos. Es la prueba de que la verificación puede crecer al ritmo de la generación.
3. *Calibración.* Calibración empírica sobre la evaluación independiente para poder emitir `calibrated_correctness_probability` donde sea aplicable.

**Perfiles de unidades.** `STANDARD_UNITS` conserva su huella histórica (`8fb557d4…`), de la que dependen los estudios congelados (PR-MCI). Las etiquetas en español viven en el perfil explícito `SPANISH_UNITS` (`edcc833d…`, idéntico a las unidades vigentes durante el benchmark). `verify_claim`, la CLI y la API aceptan `unit_profile` = `auto` | `standard` | `es`, y la elección queda registrada en la telemetría de cada resultado.

**El modelo propone; la política determinista decide.** Es la regla del paper de marzo ("los LLM no son responsables del razonamiento central") con los contratos formales del manuscrito revisado del 19 de octubre. Ningún modelo emite `supported`/`contradicted`: esa decisión la toma código sobre el registro.

Motivo: el diagnóstico del 23 de octubre (`verifier_failure_diagnosis.md`) mostró que en v3 el modelo leía bien la fuente y aun así aprobaba afirmaciones con un campo cambiado, y que el validador aceptaba `supported` siempre que el modelo marcase las casillas como verdaderas. Además, el vocabulario improvisado de los casos hacía imposible una comparación exacta (`percent_per_annum` frente a `percent per annum`).

## Qué toma de cada versión

| Pieza | Origen | En v4 |
|---|---|---|
| `SourceDocument`, `EvidenceRef`, almacén append-only con checksums | v3 | Reutilizados sin cambios |
| Citas por IDs de fragmento ligadas por código a offsets exactos | v3 (a3) | Reutilizado en el lector |
| Autoridad declarada por la aplicación (`authority_rank`), nunca por el nombre del archivo | v3 | Mantenido; constantes `TIER_PRIMARY/ANNEX/EXTERNAL` = 100/50/10 de v2 |
| Tabla epistemológica (`INFERENCE_WEIGHTS`, `BLOCKED_INFERENCE_TYPES`) | v2 | Importada de `core.interfaces` (una sola fuente); el lector declara la base, el peso lo pone la tabla |
| Noisy-OR entre grupos de dependencia, máximo dentro del grupo | paper §5.2 | `_support` |
| Q = S²/(S+O) y regla de margen S−O ≥ δ | paper §5.3 | `_decide`; estado `unresolved_conflict` si no se cumple |
| Comparación de tres valores (compatible/incompatible/desconocido) | paper §4.2 | Cardinalidad declarada en el vocabulario; dimensiones distintas → `unresolved_comparison` |
| Derivaciones solo con regla instalada y versionada | v2.0.1 / v3 | `derive`; regla `identity` incluida |
| Caminos: candidatos heterogéneos, prueba solo con regla transitiva declarada | v2 (DependencyGraph) + paper §6.2 | `find_paths` (candidatos) y `derive_paths` (TransitiveRule) |
| Puntuación de camino R(π) = γʰ·min s | v2 / paper §6.2 | Ejemplo del paper 0,676875 reproducido en test |
| Agregación con identidad de evento antes de sumar | v3 / paper §7 | Conflicto dentro de un evento se resuelve por autoridad/soporte o se abstiene |
| **Vocabulario de dominio** (predicados tipados, unidades con dimensión, condiciones declaradas) | nuevo | `Vocabulary`; las propuestas que no encajan se rechazan con motivo |
| **Conversión exacta de unidades** solo si está declarada | nuevo | `UnitRegistry`; días naturales, hábiles y meses son dimensiones distintas |
| **Respuestas condicionales** etiquetadas en vez de descartadas | nuevo | `conditions='report'` (por defecto) o `'discharge'` |
| **`check_claim`**: contraste determinista de una afirmación externa con el registro | nuevo | `supported / contradicted / not_supported / not_representable / insufficient_evidence` |

## Índice de soporte

Para cada unidad: a = min(span, structure, applicability, basis).

- span = 1,0 cuando la cita está ligada exactamente a la revisión vigente (si no, la unidad no entra).
- structure = fiabilidad estructural del productor. **Debe medirse**: si el lector no la declara, las unidades quedan `unassessed` y no pueden aceptarse (paper §5.1). Para registros tipados de la aplicación se declara explícitamente.
- applicability = `weight` del documento, fijado por la aplicación.
- basis = peso de la tabla v2 para la base declarada (`explicit` = 1,0).

Es un índice de política, no una probabilidad calibrada. τ = 0,5 y δ = 0,1 son marcadores de posición (`thresholds_validated: false`): hay que elegirlos con datos de desarrollo y fijarlos antes de evaluar.

## El lector

`LLMProposalReader` recibe el documento troceado en fragmentos con ID, el vocabulario y las bases admisibles. No recibe la afirmación a comprobar, ni metadatos de autoridad, ni se le pide un veredicto. El prompt es genérico: no contiene reglas derivadas de casos concretos del benchmark (a diferencia de los parches de los prompts de v3). Las respuestas brutas se guardan; las lecturas idénticas se reutilizan de caché.

## Benchmark de contratos de la librería

`benchmarks/library_contracts_v4/run.py`, sin LLM ni red. Mismos registros tipados traducidos a la API de cada versión.

| Bloque | Casos | v2 | v3 | v4 |
|---|---:|---:|---:|---:|
| A — 20 contratos funcionales compartidos (`functional_parity`, sin cambios) | 20 | 17 | 20 | 20 |
| B — contratos del paper revisado | 11 | 7 | 5 | 11 |

Bloque B por caso (✓ cumple):

| Contrato | v2 | v3 | v4 |
|---|---|---|---|
| P01 copias dependientes no corroboran | ✗ cuenta 2 copias | ✓ | ✓ |
| P02 soporte igual → sin respuesta asentada | ✓ | ✓ | ✓ |
| P03 afirmación explícita pesa más que inferencia débil | ✓ | ✓* | ✓ |
| P03b igual con orden invertido | ✓ | ✗ | ✓ |
| P04 base bloqueada (narrativa) excluida | ✓ | ✗ | ✓ |
| P05 autoridad por metadatos, no por nombre de archivo | ✗ el nombre "annex" la degrada | ✓ | ✓ |
| P06 valores múltiples compatibles | ✓ | ✓ | ✓ |
| P07 etiquetas de unidad canonicalizadas | ✓ | ✗ las trata como unidades distintas | ✓ |
| P08 conversión declarada antes de sumar (1 año + 6 meses) | ✗ devuelve 1 | ✗ se abstiene | ✓ 18 meses |
| P09 la condición acompaña a la respuesta | ✗ no la representa | ✗ se abstiene | ✓ |
| P10 ranking de camino γʰ·min s = 0,676875 | ✓ | ✗ no existe | ✓ |

\* En P03 la v3 acierta por desempate de orden de valores, no por pesos: P03b lo confirma.

**Cautelas.** El bloque B lo escribí a partir del paper a la vez que diseñaba la v4: que la v4 cumpla 11/11 muestra conformidad con esos contratos, no superioridad general. Tras la primera ejecución hubo dos enmiendas documentadas en `protocol.json`: nombres de una letra en P10 (la v2 no los indexa, artefacto ajeno al contrato; tolerancia 1e-4 porque la v2 redondea a 4 decimales) y el caso P03b. La primera ejecución está intacta en `benchmarks/library_contracts_v4_initial_results`. Son contratos del motor sobre datos tipados: no miden extracción ni calidad de respuesta extremo a extremo.

## Lo que no está probado ni incluido

- **Calidad de extracción con un modelo real.** Es ahora el riesgo principal: todos los errores semánticos se trasladan a la lectura, que es medible por separado (precisión/cobertura por predicado). No se ha hecho ninguna llamada a modelos.
- Umbrales τ/δ y fiabilidad estructural del lector: sin calibrar.
- Interpretación de preguntas en lenguaje natural: v4 solo acepta consultas estructuradas.
- No portado de v2: opiniones Jøsang/κ, certificado Merkle v2, consolidación automática de alias. Los alias de entidad son explícitos en el vocabulario.
- La importación por defecto `claimlayer.ClaimLayer` sigue siendo v3 y `pyproject.toml` sigue en 3.0.0a11: promover v4 es una decisión pendiente.

## Siguiente paso recomendado

1. Escribir un vocabulario para las tareas LAB y un conjunto **nuevo** de casos etiquetados antes de ejecutar (los 72 actuales quedan como regresión conocida).
2. Ejecutar el lector con el modelo real sobre esos documentos, medir `structural_reliability` por predicado y fijar τ/δ con un subconjunto de desarrollo.
3. Evaluar en el conjunto reservado: cobertura de positivos, negativos rechazados por mecanismo y coste.

## Uso

```python
from claimlayer.v4 import EvidenceLayer, Vocabulary, PredicateSpec, FieldSpec, LLMProposalReader, OpenAICompatibleModel

vocab = Vocabulary('credit', '1', (PredicateSpec('sofr_floor', FieldSpec('quantity', dimensions=('annual_rate',))),))
reader = LLMProposalReader(OpenAICompatibleModel(base_url=..., model=..., api_key=...), structural_reliability=0.9)
layer = EvidenceLayer('evidence_v4.db', vocabulary=vocab, reader=reader)
layer.ingest(SourceDocument('markup.docx', text, authority_rank=100, weight=0.9))
layer.ask({'entity': 'Lender markup', 'predicate': 'sofr_floor'}, assertion_kinds=('proposal',))
layer.check_claim({...})  # nunca llama al modelo
```

Ejemplo sin red: `PYTHONPATH=src python examples/v4_quickstart.py`. Tests: `tests/test_v4_*.py` (52).

## Verificación

- Suite completa en el equipo del usuario (Python 3.10.12, pytest 9.1.1): **317 aprobados, 1 fallo previo y ajeno a v4** (`test_semantic_harness_gate.py` importa el módulo externo `parse_doc`, que no está en esa ruta). En el contenedor de desarrollo (Python 3.13): 312 aprobados sin esos dos ficheros.
- El benchmark de contratos se ejecutó en ambos entornos con resultados idénticos (A: 17/20/20; B: 7/5/11). Los artefactos de `benchmarks/library_contracts_v4_results` son los del equipo del usuario.
- En una carpeta que no permite borrar archivos, SQLite no puede gestionar su diario: ejecutar con `--out` a un directorio local y copiar después (`python benchmarks/library_contracts_v4/run.py --out /tmp/lc`). Los tests usan directorios temporales y no tienen este problema.

## Hito v4: Contrato de lector estricto, evaluación en 3 capas y segundo dominio (Titulización)

Sobre la arquitectura v4 se ha implementado el siguiente bloque:

1. **Contrato de lector estricto y fidelidad (`src/claimlayer/v4/reader.py`, `src/claimlayer/v4/fidelity.py`)**:
   - `STRICT_READER_PROMPT` (opt-in mediante `LLMProposalReader(model, prompt=STRICT_READER_PROMPT)`) exige `condition_status` explícito y citas `field_evidence` por campo, resueltas a spans exactos (con fallback exacto para tokens cortos como nombres de entidad o cantidades).
   - `verify_extraction_fidelity()` añade `DIRECTION_OF_CHANGE_MISMATCH` (paridad de signo ante aumentos/reducciones) y `ENTITY_NOT_IN_FIELD_EVIDENCE` (grounding de la entidad en su span), además de `render_canonical_claim(unit)`.
2. **Benchmark de 3 capas con soporte `--dry-run` y `--live` (`benchmarks/b2b_live_eval.py`)**:
   - Separa `DEV_CASES` (6 casos de calibración) y `TEST_CASES` (6 casos ciegos) midiendo por separado: Capa 1 (`gold_reasoning`), Capa 2 (`dual_reader_extraction` con `evaluate_dual_readers`) y Capa 3 (`risk_coverage` calibrando `tau`/`delta` en `dev` y midiendo `selective_risk` vs `automation_coverage` en `test`).
   - En `--dry-run` (dobles deterministas, sin llamadas a APIs externas): Capa 1 obtiene `6/6` (`1.0`) en `dev` y `6/6` (`1.0`) en `test`; en Capa 3 (`test` ciego), `automation_coverage = 0.5`, `selective_risk = 0.0` (`3` aceptados, `3` abstenidos, `0` falsos `PASS`), bloqueando el caso de error correlacionado mediante `UNCAPTURED_CONDITION` y `POLARITY_MISMATCH`. El modo `--live` queda listo para ejecutarse con dos `OpenAICompatibleModel` reales (`--endpoint-a`, `--model-a`, `--api-key-a`, `--endpoint-b`, `--model-b`, `--api-key-b`).
3. **Frontera de confianza (`tests/test_v4_milestone_phases.py`)**:
   - Verificado que `FieldEvidence` manipulado y `unread_documents` bloquean `authorize_operation`, que `AuthorizationVerdict` es inmutable (`FrozenInstanceError`), y que `verify_receipt(..., allow_unkeyed=False)` rechaza veredictos sin firma HMAC, con clave incorrecta o manipulados.
4. **Segundo dominio acotado: Titulización (`src/claimlayer/v4/domains/securitization.py`)**:
   - Define `SECURITIZATION_VOCABULARY` (8 predicados: `coupon_rate`, `clean_up_call_threshold`, `cure_period`, `waterfall_priority`, `reporting_frequency`, `note_principal`, `delinquency_trigger`, `event_of_default`; `scope_keys=("tranche", "clause_id", "reporting_period", "event_id")`; `require_explicit_condition_status=True`) y `SECURITIZATION_ACCEPTANCE_POLICY`, reutilizando sin modificaciones el motor de fidelidad y autorización.

### Verificación reproducible

```bash
PYTHONPATH=claim-layer/src:claim-layer:src .venv/bin/pytest claim-layer/tests/ tests/test_api_and_triage.py -q
# -> 266 passed, 1 skipped

PYTHONPATH=claim-layer/src:claim-layer .venv/bin/python claim-layer/benchmarks/b2b_live_eval.py --dry-run
```
