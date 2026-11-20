# Diagnóstico de Atribución de Errores Fase 3 (`n = 120`, AIDev-Dev Originals)

> [!IMPORTANT]
> **Nota metodológica (`silver: Claude`):** Las etiquetas de referencia de los 120 ítems son **etiquetas «plata»** generadas de forma ciega por Claude (modelo distinto de Nemotron y sin acceso a `blind_key.json` ni a las decisiones del gate), **no anotación humana final**. Antes de usar estas cifras como referencia pública, conviene auditar con una persona al menos los 43 ítems prioritarios listados en `REVISION_HUMANA_prioritaria.md`.

## Punto 2 — Techo del Motor de Inferencia (`JustificationGraph` + Reglas Declaradas + Invariantes Epistémicos)

- **Exactitud mapeada 4 clases (`n = 120`):** **119/120 (99.2%)**, IC 95 % `[0.9543, 0.9985]`
- **Exactitud sin `out_of_scope_meta` (`n = 113`):** **112/113 (99.1%)**, IC 95 % `[0.9516, 0.9984]`
- **Monotonicidad epistémica (`use_declared_rules=False`):** `0` ascendidas a `derived_supported` (invariante estricto verificado).
- **Único fallo del motor (`DIAG-041`):** afirmación negativa universal de comportamiento sin cambios sobre un PR de 170 ficheros; el motor solo comprueba contradicciones puntuales declaradas.

### Matriz de Confusión del Motor (`silver` -> `engine`)
```json
{
  "contradicted": {
    "no_premises_in_diff": 1
  },
  "no_premises": {
    "no_premises_in_diff": 6
  },
  "out_of_scope_meta": {
    "unverifiable": 7
  },
  "supported": {
    "derived_supported": 41,
    "hypothesis_with_verified_premises": 65
  }
}
```

## Punto 3 — Rigidez del Verificador Literal (`code_config` + Guardia de Pertinencia)

| Medida | ANTES (`3808d82`) | DESPUÉS (fixes genéricos) | IC 95 % DESPUÉS |
|---|---:|---:|---:|
| Premisas plata aceptadas (`n = 210`) | 120/210 (57.1%) | 133/210 (63.3%) | `[0.5663, 0.6956]` |
| Premisas plata rechazadas (`n = 210`) | 90/210 (42.9%) | 77/210 (36.7%) | `[0.3044, 0.4337]` |
| Afirmaciones `supported` aceptadas (`n = 106`) | 76/106 (71.7%) | 82/106 (77.4%) | `[0.6852, 0.8429]` |
| Exactitud oráculo + verificador (`n = 120`) | 89/120 (74.2%) | 95/120 (79.2%) | `[0.7105, 0.8547]` |

- **Anti-laxitud en `N2_hard` (verificador DESPUÉS):** fuga estricta (`derived_supported`) **4/109 (3.67%)**, fuga amplia (`>=1` premisa verificada) **14/109 (12.84%)**, tasa de captura **88/109 (80.73%)**, `ESCALATE` = `1`.

### Desglose de causas de rechazo por premisa (`n = 210`, ANTES vs DESPUÉS)

| Causa de rechazo | ANTES (`3808d82`) | DESPUÉS |
|---|---:|---:|
| `target_not_in_premise_line` | 26 | 28 |
| `same_hunk_sibling_changed_line_target` | 23 | 24 |
| `code_config_literal_strictness` | 12 | 0 |
| `cross_category_op:configures_on_src` | 8 | 9 |
| `insufficient_overlap_or_short_line` | 7 | 7 |
| `cross_category_op:documents_on_src` | 3 | 3 |
| `same_hunk_context_target` | 3 | 0 |
| `cross_category_op:documents_on_test` | 2 | 2 |
| `cross_category_op:updates_dependency_on_src` | 2 | 2 |
| `test_path_pattern_miss` | 2 | 0 |
| `cross_category_op:configures_on_test` | 1 | 1 |
| `cross_category_op:updates_dependency_on_docs` | 1 | 1 |

## Punto 4 — Calidad del Lector, Recuperación por Hunk `@@` y Comparativa NIM

| Configuración del Lector | Visibilidad `>=1` línea plata (`n=106`) | Recall línea exacta (`n=210` / `n=106`) | Citas no ancladas | `op` exacta (`n=120`) | E2E `n=120` (Verif DESPUÉS) |
|---|---:|---:|---:|---:|---:|
| Baseline Nemotron Pass 2 (caché) | 83.0% | 38.6% / 54.7% | 11/167 | 66.7% | 71/120 (59.2%) |
| Baseline Nemotron Pass 2+3 (caché `3808d82`) | 83.0% | 42.4% / 59.4% | 16/221 | 66.7% | 75/120 (62.5%) |
| 4(a) `nvidia/nemotron-3-ultra-550b-a55b` + Retrieval `@@` | 95.3% | 64.3% / 83.0% | 9/278 | 78.3% | 105/120 (87.5%) |
| 4(b) `poolside/laguna-xs-2.1` + Retrieval `@@` | 95.3% | 57.6% / 72.6% | 21/274 | 67.5% | 86/120 (71.7%) |

- **Telemetría 4(a) (`nvidia/nemotron-3-ultra-550b-a55b`):** `363800` tokens (`332240` in / `31560` out), latencia media `7.023s` (p50 `5.205s`, p95 `14.494s`), coste ref. `USD 0.25615` (`$0` free tier dev).
- **Telemetría 4(b) (`poolside/laguna-xs-2.1`):** `352871` tokens (`320240` in / `32631` out), latencia media `2.713s` (p50 `2.43s`, p95 `5.762s`), coste ref. `USD 0.09015` (`$0` free tier dev).
- **Nota sobre sustitución en 4(b):** minimaxai/minimax-m3.1-flash se probó inicialmente en 4(b): completó 79/120 llamadas (0 errores de parseo, 0/166 citas no ancladas, 70/120 exactitud incluyendo los 41 fallos API y 62/106 en supported), pero sufrió HTTP 429 Too Many Requests persistente en NIM (41/120 fallos incluso con workers=1 y min_interval=4.5s), por lo que 4(b) se completó al 120/120 con poolside/laguna-xs-2.1.

## Punto 5 — Tabla Final de Atribución del Error (`silver: Claude`, `n = 120`)

En el pipeline base `3808d82`, acertó **66/120** y falló **54/120**. Descomposición mutuamente excluyente de esos **54** errores:

| Fuente del error | Errores | % sobre fallos (`n=54`) | IC 95 % (fallos) | % sobre total (`n=120`) | IC 95 % (`n=120`) |
|---|---:|---:|---:|---:|---:|
| Solo Lector (Pass 2/3 no propuso premisa válida o falló `out_of_scope_meta`) | 32 | 59.3% | `[0.4597, 0.7132]` | 26.7% | `[0.1957, 0.3521]` |
| Solo Verificador (rigidez de `code_config` / guardia de pertinencia / categoría) | 17 | 31.5% | `[0.2068, 0.4474]` | 14.2% | `[0.0904, 0.2152]` |
| Conjunta Lector + Verificador (ambos fallan por separado) | 4 | 7.4% | `[0.0292, 0.1755]` | 3.3% | `[0.013, 0.0826]` |
| Solo Motor de Inferencia (falla aun con premisas oráculo) | 1 | 1.8% | `[0.0033, 0.0977]` | 0.8% | `[0.0015, 0.0457]` |

### Cascada completa de exactitud (`n = 120` y `supported n = 106`)

| Etapa | Exactitud `n=120` | IC 95 % (`n=120`) | Recall `supported n=106` | IC 95 % (`n=106`) |
|---|---:|---:|---:|---:|
| 1. Engine Ceiling (Oracle -> Engine) | 119/120 (99.2%) | `[0.9543, 0.9985]` | 106/106 (100.0%) | `[0.965, 1.0]` |
| 2. Oracle -> Verifier BEFORE (3808d82) -> Engine | 89/120 (74.2%) | `[0.6567, 0.8116]` | 76/106 (71.7%) | `[0.6248, 0.794]` |
| 3. Oracle -> Verifier AFTER (v3 fixes) -> Engine | 95/120 (79.2%) | `[0.7105, 0.8547]` | 82/106 (77.4%) | `[0.6852, 0.8429]` |
| 4. Baseline Nemotron (Pass 2/3) + Verifier BEFORE | 66/120 (55.0%) | `[0.4608, 0.6361]` | 60/106 (56.6%) | `[0.471, 0.6564]` |
| 5. Baseline Nemotron (Pass 2/3) + Verifier AFTER | 75/120 (62.5%) | `[0.5358, 0.7065]` | 69/106 (65.1%) | `[0.5564, 0.735]` |
| 6a. 4(a) Nemotron + Retrieval + Verifier BEFORE | 92/120 (76.7%) | `[0.6834, 0.8333]` | 85/106 (80.2%) | `[0.716, 0.8666]` |
| 6b. 4(a) Nemotron + Retrieval + Verifier AFTER | 105/120 (87.5%) | `[0.804, 0.9228]` | 98/106 (92.5%) | `[0.8581, 0.9613]` |
| 7a. 4(b) Poolside Laguna XS 2.1 + Retrieval + Verifier BEFORE | 73/120 (60.8%) | `[0.5189, 0.691]` | 68/106 (64.1%) | `[0.5467, 0.7264]` |
| 7b. 4(b) Poolside Laguna XS 2.1 + Retrieval + Verifier AFTER | 86/120 (71.7%) | `[0.6303, 0.7896]` | 81/106 (76.4%) | `[0.675, 0.8348]` |

### Matices cualitativos de las etiquetas plata (Claude)

- **Auto-corrección intra-PR:** DIAG-039, DIAG-045, DIAG-105, DIAG-120 — hunks posteriores corrigen líneas anteriores del mismo PR; un verificador plano sobre todo el diff puede detectar contradicciones locales superadas al final.
- **Renombrados con hunk vacío:** DIAG-091, DIAG-075, DIAG-089 — el parche unificado no incluye líneas `+`/`-`, solo la ruta renombrada.
- **Diff con merge contaminado / contradicción discutida:** DIAG-041 — único caso anotado como `contradicted` por la anotación plata.
- **Revisión humana pendiente:** `43` ítems prioritarios marcados en `REVISION_HUMANA_prioritaria.md` antes de publicar resultados.
- **Efecto de recuperación multi-línea en 4(a)/4(b):** el lector por afirmación puede proponer hasta 3 líneas del diff (incluyendo la definición `+def`/declaración completa), logrando en `supported` superar al oráculo de 1 línea mínima cuando la cita plata única omitía la cabecera o línea definitoria verificable.
- **Compromiso `no_premises -> out_of_scope_meta` (`unverifiable`):** en afirmaciones sin premisas en el diff (`n = 6`), el prompt por afirmación tiende a clasificar algunas como `out_of_scope_meta` en lugar de intentar asociarles código y acabar en `no_premises_in_diff`.
