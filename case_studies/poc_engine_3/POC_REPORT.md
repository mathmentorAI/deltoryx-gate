# POC del Motor de Inferencia v4: 3 Casos (9 Afirmaciones)

## Configuración y Prerregistro
- **SHA-256 `PREREGISTRATION.json`:** `49c65c4dae63d5220e54325b9677fb8142b2e19d1d45c90a6189937daa91191a`
- **Fecha de referencia:** `2026-11-20`
- **Modelo NIM (Modos B y C):** `nvidia/nemotron-3-ultra-550b-a55b` (`temperature=0`, `enable_thinking=False`, `max_tokens=800`, `3` ejecuciones independientes por modo).
- **Restricción cumplida:** Ningún archivo de `src/claimlayer/` ni de Phase 3 (`pr_mci/`, `pr_mci_phase3/`) ha sido modificado; los adaptadores y reglas de dominio residen exclusivamente en `case_studies/poc_engine_3/poc_domain.py`.

## 1. Tabla Comparativa: 9 Afirmaciones × 3 Modos

| ID | Afirmación | Fuentes | Esperado | Modo A (Motor solo) | Modo B (Lector NIM + Motor, 3 corridas) | Modo C (LLM directo, 3 corridas) |
|---|---|---|---|---|---|---|
| `C1a` | Buffett es una persona mayor | `S1` | **DERIVADA** | DERIVADA (OK) | DERIVADA (OK, 3/3 igual, 3/3 ok) | DERIVADA (OK, 3/3 igual, 3/3 ok) |
| `C1b` | Buffett tiene 100 años | `S1` | **CONTRADICHA** | CONTRADICHA (OK) | CONTRADICHA (OK, 3/3 igual, 3/3 ok) | CONTRADICHA (OK, 3/3 igual, 3/3 ok) |
| `C1c` | Buffett es vegetariano | `S1` | **SIN EVIDENCIA** | SIN EVIDENCIA (OK) | SIN EVIDENCIA (OK, 3/3 igual, 3/3 ok) | SIN EVIDENCIA (OK, 3/3 igual, 3/3 ok) |
| `C2a` | Alfa controla Gamma | `S2+S3` | **DERIVADA** | DERIVADA (OK) | DERIVADA (OK, 3/3 igual, 3/3 ok) | DERIVADA (OK, 3/3 igual, 3/3 ok) |
| `C2b` | Gamma controla Alfa | `S2+S3` | **SIN EVIDENCIA** | SIN EVIDENCIA (OK) | SIN EVIDENCIA (OK, 3/3 igual, 3/3 ok) | CONTRADICHA (FALLO, 3/3 igual, 0/3 ok) |
| `C2c` | Alfa controla Gamma | `S2` | **SIN EVIDENCIA** | SIN EVIDENCIA (OK) | SIN EVIDENCIA (OK, 3/3 igual, 3/3 ok) | SIN EVIDENCIA (OK, 3/3 igual, 3/3 ok) |
| `C3a` | El contrato se renovará | `S4` | **HIPÓTESIS** | HIPÓTESIS (OK) | HIPÓTESIS (OK, 3/3 igual, 3/3 ok) | HIPÓTESIS (OK, 3/3 igual, 3/3 ok) |
| `C3b` | El contrato se renovará | `S4+S5` | **DERIVADA** | DERIVADA (OK) | DERIVADA (OK, 3/3 igual, 3/3 ok) | HIPÓTESIS (FALLO, 3/3 igual, 0/3 ok) |
| `C3c` | El contrato se renovará | `S4+S6` | **CONTRADICHA** | CONTRADICHA (OK) | CONTRADICHA (OK, 3/3 igual, 3/3 ok) | CONTRADICHA (OK, 3/3 igual, 3/3 ok) |

- **Totales acierto (mayoría / 3 de 3 exactas):** Modo A = **9/9**; Modo B = **9/9** (3/3 en `9/9`); Modo C = **7/9** (3/3 en `7/9`).

## 2. Rastro del `JustificationGraph` en Modo A (9 Afirmaciones)

### `C1a` — «Buffett es una persona mayor» (`S1`)
- **Resultado:** `DERIVADA` (esperado `DERIVADA`) | `assessment=supported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** `R1_persona_mayor`
- **Nodos (premisas e intermedios):**
  - `C1a::d_age` (`derivation`): `(Warren Buffett, edad, 96` `years`) | _nodo derivado / sin fuente_
  - `C1a::d_ge65` (`derivation`): `(Warren Buffett, edad_al_menos_65, True`) | _nodo derivado / sin fuente_
  - `C1a::p_birth` (`extraction`): `(Warren Buffett, birth_date, 1930-08-30`) | fuente `S1`: «Warren Buffett nació el 30 de agosto de 1930.»
- **Pasos de inferencia y verificadores:**
  - `C1a::s_age` (`temporal_age`): `['C1a::p_birth'] -> C1a::d_age` | Checks: temporal=verified ['TEMPORAL_DERIVATION_VERIFIED']
  - `C1a::s_cmp_65` (`ge`): `['C1a::d_age'] -> C1a::d_ge65` | Checks: comparison=verified ['COMPARISON_VERIFIED']; quantities=verified ['UNIT_COMPATIBILITY_VERIFIED']
  - `C1a::s_r1` (`rule_application` con regla `R1_persona_mayor@1.0`): `['C1a::d_ge65'] -> C1a::conclusion` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::age_and_duration=not_applicable ['BOOLEAN_DOWNSTREAM_RULE_CONCLUSION']; mandatory::quantities=verified ['UNITS_VALID_IN_REGISTRY']; mandatory::documental_attribution=verified ['DOCUMENT_LINKAGE_ATTRIBUTED']

### `C1b` — «Buffett tiene 100 años» (`S1`)
- **Resultado:** `CONTRADICHA` (esperado `CONTRADICHA`) | `assessment=refuted` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** _ninguna (sin camino/regla aplicable)_
- **Nodos (premisas e intermedios):**
  - `C1b::p_birth` (`extraction`): `(Warren Buffett, birth_date, 1930-08-30`) | fuente `S1`: «Warren Buffett nació el 30 de agosto de 1930.»
- **Pasos de inferencia y verificadores:**
  - `C1b::s_age` (`temporal_age`): `['C1b::p_birth'] -> C1b::conclusion` | Checks: temporal=refuted ['TEMPORAL_DERIVATION_CONTRADICTED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::age_and_duration=refuted ['TEMPORAL_DERIVATION_CONTRADICTED']; mandatory::quantities=verified ['UNITS_VALID_IN_REGISTRY']; mandatory::documental_attribution=verified ['EXTERNAL_TRUTH_CORROBORATED']

### `C1c` — «Buffett es vegetariano» (`S1`)
- **Resultado:** `SIN EVIDENCIA` (esperado `SIN EVIDENCIA`) | `assessment=unsupported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** _ninguna (sin camino/regla aplicable)_
- **Nodos (premisas e intermedios):**
  - _Ninguna premisa aplicable extraída para esta afirmación._
- **Pasos de inferencia y verificadores:**
  - _Sin pasos de inferencia conectados a la conclusión._
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::documental_attribution=verified ['DOCUMENT_LINKAGE_ATTRIBUTED']

### `C2a` — «Alfa controla Gamma» (`S2+S3`)
- **Resultado:** `DERIVADA` (esperado `DERIVADA`) | `assessment=supported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** `R2_controla_transitiva`, `R2a_prop_implica_control`
- **Nodos (premisas e intermedios):**
  - `C2a::ctrl_0` (`derivation`): `(Alfa S.A., controla, Beta S.L.`) | _nodo derivado / sin fuente_
  - `C2a::ctrl_1` (`derivation`): `(Beta S.L., controla, Gamma S.L.`) | _nodo derivado / sin fuente_
  - `C2a::e_0` (`extraction`): `(Alfa S.A., propietaria_100, Beta S.L.`) | fuente `S2`: «Alfa S.A. es propietaria del 100 % de Beta S.L.»
  - `C2a::e_1` (`extraction`): `(Beta S.L., propietaria_100, Gamma S.L.`) | fuente `S3`: «Beta S.L. es propietaria del 100 % de Gamma S.L.»
- **Pasos de inferencia y verificadores:**
  - `C2a::s_r2a_0` (`rule_application` con regla `R2a_prop_implica_control@1.0`): `['C2a::e_0'] -> C2a::ctrl_0` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
  - `C2a::s_r2a_1` (`rule_application` con regla `R2a_prop_implica_control@1.0`): `['C2a::e_1'] -> C2a::ctrl_1` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
  - `C2a::s_r2_trans` (`relation_composition` con regla `R2_controla_transitiva@1.0`): `['C2a::ctrl_0', 'C2a::ctrl_1'] -> C2a::conclusion` | Checks: relations=verified ['RELATION_COMPOSITION_VERIFIED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::documental_attribution=verified ['DOCUMENT_LINKAGE_ATTRIBUTED']

### `C2b` — «Gamma controla Alfa» (`S2+S3`)
- **Resultado:** `SIN EVIDENCIA` (esperado `SIN EVIDENCIA`) | `assessment=unsupported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** _ninguna (sin camino/regla aplicable)_
- **Nodos (premisas e intermedios):**
  - `C2b::e_0` (`extraction`): `(Alfa S.A., propietaria_100, Beta S.L.`) | fuente `S2`: «Alfa S.A. es propietaria del 100 % de Beta S.L.»
  - `C2b::e_1` (`extraction`): `(Beta S.L., propietaria_100, Gamma S.L.`) | fuente `S3`: «Beta S.L. es propietaria del 100 % de Gamma S.L.»
- **Pasos de inferencia y verificadores:**
  - _Sin pasos de inferencia conectados a la conclusión._
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::documental_attribution=verified ['DOCUMENT_LINKAGE_ATTRIBUTED']

### `C2c` — «Alfa controla Gamma» (`S2`)
- **Resultado:** `SIN EVIDENCIA` (esperado `SIN EVIDENCIA`) | `assessment=unsupported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** _ninguna (sin camino/regla aplicable)_
- **Nodos (premisas e intermedios):**
  - `C2c::e_0` (`extraction`): `(Alfa S.A., propietaria_100, Beta S.L.`) | fuente `S2`: «Alfa S.A. es propietaria del 100 % de Beta S.L.»
- **Pasos de inferencia y verificadores:**
  - _Sin pasos de inferencia conectados a la conclusión._
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::documental_attribution=verified ['DOCUMENT_LINKAGE_ATTRIBUTED']

### `C3a` — «El contrato se renovará» (`S4`)
- **Resultado:** `HIPÓTESIS` (esperado `HIPÓTESIS`) | `assessment=unsupported` | `effective_epistemic_kind=hypothesis` (declarado `derivation`)
- **Reglas declaradas aplicadas:** `R3_renovacion_automatica`
- **Nodos (premisas e intermedios):**
  - `C3a::p_clausula` (`extraction`): `(Contrato, clausula_renovacion_automatica, True`) | `condition_status=conditional` | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
  - `C3a::p_hyp_sin_preaviso` (`hypothesis`): `(Contrato, sin_preaviso_en_plazo, True`) | _nodo derivado / sin fuente_
  - `C3a::p_vencimiento` (`extraction`): `(Contrato, fecha_vencimiento, 2026-12-31`) | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
- **Pasos de inferencia y verificadores:**
  - `C3a::s_r3_hyp` (`rule_application` con regla `R3_renovacion_automatica@1.0`): `['C3a::p_clausula', 'C3a::p_hyp_sin_preaviso'] -> C3a::conclusion` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::documental_attribution=verified ['EXTERNAL_TRUTH_CORROBORATED']; mandatory::conditions_and_exceptions=verified ['CONDITIONS_PRESERVED_VERIFIED']

### `C3b` — «El contrato se renovará» (`S4+S5`)
- **Resultado:** `DERIVADA` (esperado `DERIVADA`) | `assessment=supported` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** `R3_renovacion_automatica`, `R3_sin_preaviso_en_plazo`
- **Nodos (premisas e intermedios):**
  - `C3b::d_dias_rest` (`derivation`): `(Contrato, dias_hasta_vencimiento, 26` `days`) | _nodo derivado / sin fuente_
  - `C3b::d_plazo_venc` (`derivation`): `(Contrato, plazo_preaviso_vencido, True`) | _nodo derivado / sin fuente_
  - `C3b::d_sin_en_plazo` (`derivation`): `(Contrato, sin_preaviso_en_plazo, True`) | _nodo derivado / sin fuente_
  - `C3b::p_clausula` (`extraction`): `(Contrato, clausula_renovacion_automatica, True`) | `condition_status=conditional` | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
  - `C3b::p_sin_fecha` (`extraction`): `(Contrato, fecha_registro_sin_preaviso, 2026-12-05`) | fuente `S5`: «Sin comunicaciones de preaviso registradas hasta el 05/12/2026.»
  - `C3b::p_sin_reg` (`extraction`): `(Contrato, sin_preaviso_registrado, True`) | fuente `S5`: «Sin comunicaciones de preaviso registradas hasta el 05/12/2026.»
  - `C3b::p_vencimiento` (`extraction`): `(Contrato, fecha_vencimiento, 2026-12-31`) | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
- **Pasos de inferencia y verificadores:**
  - `C3b::s_dias_rest` (`temporal_duration`): `['C3b::p_sin_fecha', 'C3b::p_vencimiento'] -> C3b::d_dias_rest` | Checks: temporal=verified ['TEMPORAL_DERIVATION_VERIFIED']
  - `C3b::s_lt30` (`lt`): `['C3b::d_dias_rest'] -> C3b::d_plazo_venc` | Checks: comparison=verified ['COMPARISON_VERIFIED']; quantities=verified ['UNIT_COMPATIBILITY_VERIFIED']
  - `C3b::s_sin` (`rule_application` con regla `R3_sin_preaviso_en_plazo@1.0`): `['C3b::p_sin_reg', 'C3b::d_plazo_venc'] -> C3b::d_sin_en_plazo` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
  - `C3b::s_r3` (`rule_application` con regla `R3_renovacion_automatica@1.0`): `['C3b::p_clausula', 'C3b::d_sin_en_plazo'] -> C3b::conclusion` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::quantities=verified ['UNITS_VALID_IN_REGISTRY']; mandatory::documental_attribution=verified ['EXTERNAL_TRUTH_CORROBORATED']; mandatory::conditions_and_exceptions=verified ['CONDITION_DISCHARGED_BY_VERIFIED_PREMISE']

### `C3c` — «El contrato se renovará» (`S4+S6`)
- **Resultado:** `CONTRADICHA` (esperado `CONTRADICHA`) | `assessment=refuted` | `effective_epistemic_kind=derivation` (declarado `derivation`)
- **Reglas declaradas aplicadas:** `R3_exclusion_preaviso`, `R3_preaviso_en_plazo`
- **Nodos (premisas e intermedios):**
  - `C3c::d_dias_ant` (`derivation`): `(Contrato, dias_antelacion_preaviso, 46` `days`) | _nodo derivado / sin fuente_
  - `C3c::d_en_plazo` (`derivation`): `(Contrato, preaviso_en_plazo, True`) | _nodo derivado / sin fuente_
  - `C3c::d_ge30` (`derivation`): `(Contrato, antelacion_al_menos_30_dias, True`) | _nodo derivado / sin fuente_
  - `C3c::p_clausula` (`extraction`): `(Contrato, clausula_renovacion_automatica, True`) | `condition_status=conditional` | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
  - `C3c::p_preaviso` (`extraction`): `(Contrato, preaviso_no_renovacion_enviado, True`) | fuente `S6`: «Preaviso de no renovación enviado el 15/11/2026.»
  - `C3c::p_preaviso_fecha` (`extraction`): `(Contrato, fecha_preaviso, 2026-11-15`) | fuente `S6`: «Preaviso de no renovación enviado el 15/11/2026.»
  - `C3c::p_vencimiento` (`extraction`): `(Contrato, fecha_vencimiento, 2026-12-31`) | fuente `S4`: «El contrato vence el 31/12/2026. Se renueva automáticamente salvo preaviso por escrito con al menos 30 días de antelación.»
- **Pasos de inferencia y verificadores:**
  - `C3c::s_dias_ant` (`temporal_duration`): `['C3c::p_preaviso_fecha', 'C3c::p_vencimiento'] -> C3c::d_dias_ant` | Checks: temporal=verified ['TEMPORAL_DERIVATION_VERIFIED']
  - `C3c::s_ge30` (`ge`): `['C3c::d_dias_ant'] -> C3c::d_ge30` | Checks: comparison=verified ['COMPARISON_VERIFIED']; quantities=verified ['UNIT_COMPATIBILITY_VERIFIED']
  - `C3c::s_plazo` (`rule_application` con regla `R3_preaviso_en_plazo@1.0`): `['C3c::p_preaviso', 'C3c::d_ge30'] -> C3c::d_en_plazo` | Checks: explicit_logic=verified ['EXPLICIT_RULE_VERIFIED']
  - `C3c::s_excl` (`rule_application` con regla `R3_exclusion_preaviso@1.0`): `['C3c::p_clausula', 'C3c::d_en_plazo'] -> C3c::conclusion` | Checks: explicit_logic=refuted ['EXCLUSION_RULE_TRIGGERED']
- **Obligaciones obligatorias (`mandatory::conclusion`):** mandatory::quantities=verified ['UNITS_VALID_IN_REGISTRY']; mandatory::documental_attribution=verified ['EXTERNAL_TRUTH_CORROBORATED']; mandatory::conditions_and_exceptions=verified ['CONDITION_RESOLVED_BY_TIMELY_NOTICE']

## 3. Detalle por Corrida en Modo B y Modo C

### Modo B (Extractor Nemotron + Motor v4)
- Corrida `1` `C1a`: **DERIVADA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `1` `C1b`: **CONTRADICHA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `1` `C1c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `1` `C2a`: **DERIVADA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `1` `C2b`: **SIN EVIDENCIA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `1` `C2c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `1` `C3a`: **HIPÓTESIS** (`OK`) | ancladas=2, rechazadas=0
- Corrida `1` `C3b`: **DERIVADA** (`OK`) | ancladas=4, rechazadas=0
- Corrida `1` `C3c`: **CONTRADICHA** (`OK`) | ancladas=4, rechazadas=0
- Corrida `2` `C1a`: **DERIVADA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `2` `C1b`: **CONTRADICHA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `2` `C1c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `2` `C2a`: **DERIVADA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `2` `C2b`: **SIN EVIDENCIA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `2` `C2c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `2` `C3a`: **HIPÓTESIS** (`OK`) | ancladas=2, rechazadas=0
- Corrida `2` `C3b`: **DERIVADA** (`OK`) | ancladas=4, rechazadas=0
- Corrida `2` `C3c`: **CONTRADICHA** (`OK`) | ancladas=4, rechazadas=0
- Corrida `3` `C1a`: **DERIVADA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `3` `C1b`: **CONTRADICHA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `3` `C1c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `3` `C2a`: **DERIVADA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `3` `C2b`: **SIN EVIDENCIA** (`OK`) | ancladas=2, rechazadas=0
- Corrida `3` `C2c`: **SIN EVIDENCIA** (`OK`) | ancladas=1, rechazadas=0
- Corrida `3` `C3a`: **HIPÓTESIS** (`OK`) | ancladas=2, rechazadas=0
- Corrida `3` `C3b`: **DERIVADA** (`OK`) | ancladas=4, rechazadas=0
- Corrida `3` `C3c`: **CONTRADICHA** (`OK`) | ancladas=4, rechazadas=0

### Modo C (Nemotron directo)
- Corrida `1` `C1a`: **DERIVADA** (`OK`) | «Warren Buffett nació el 30/08/1930, por lo que el 20/11/2026 tiene 96 años (2026-1930=96, y su cumpleaños de 2026 ya pasó en agosto). Según la regla declarada, persona_mayor(x) si »
- Corrida `1` `C1b`: **CONTRADICHA** (`OK`) | «Según la fuente S1, Warren Buffett nació el 30/08/1930. A la fecha de referencia 2026-11-20, su edad es 96 años (cumplió 96 el 30/08/2026). La afirmación de que tiene 100 años es f»
- Corrida `1` `C1c`: **SIN EVIDENCIA** (`OK`) | «La única fuente proporcionada indica la fecha de nacimiento de Warren Buffett, pero no contiene ninguna información sobre su dieta o hábitos alimenticios. La regla declarada sobre »
- Corrida `1` `C2a`: **DERIVADA** (`OK`) | «De S2 y la regla propietaria_100 => controla se deduce que Alfa controla Beta. De S3 y la misma regla se deduce que Beta controla Gamma. Aplicando la transitividad declarada (contr»
- Corrida `1` `C2b`: **CONTRADICHA** (`FALLO`) | «Las fuentes establecen que Alfa S.A. controla Beta S.L. (propietaria_100) y Beta S.L. controla Gamma S.L. (propietaria_100). Por la regla de transitividad, Alfa controla Gamma, no »
- Corrida `1` `C2c`: **SIN EVIDENCIA** (`OK`) | «La fuente solo indica que Alfa S.A. es propietaria del 100% de Beta S.L., lo que según la regla implica que Alfa controla a Beta. No hay información sobre Gamma ni sobre la relació»
- Corrida `1` `C3a`: **HIPÓTESIS** (`OK`) | «La cláusula establece renovación automática salvo preaviso escrito con ≥30 días de antelación al 31/12/2026. A la fecha de referencia (20/11/2026) faltan 41 días para el vencimient»
- Corrida `1` `C3b`: **HIPÓTESIS** (`FALLO`) | «La cláusula S4 establece renovación automática salvo preaviso escrito con ≥30 días de antelación al 31/12/2026 (fecha límite: 01/12/2026). La fuente S5 solo confirma ausencia de pr»
- Corrida `1` `C3c`: **CONTRADICHA** (`OK`) | «La fuente S6 indica que se envió un preaviso de no renovación el 15/11/2026, lo cual cumple con el requisito de la cláusula en S4 de notificar por escrito con al menos 30 días de a»
- Corrida `2` `C1a`: **DERIVADA** (`OK`) | «Warren Buffett nació el 30/08/1930, por lo que el 20/11/2026 tiene 96 años (2026-1930=96, y su cumpleaños de 2026 ya pasó en agosto). Según la regla declarada, persona_mayor(x) si »
- Corrida `2` `C1b`: **CONTRADICHA** (`OK`) | «Según la fuente S1, Warren Buffett nació el 30/08/1930. A la fecha de referencia 2026-11-20, su edad es 96 años (cumplió 96 el 30/08/2026). La afirmación de que tiene 100 años es f»
- Corrida `2` `C1c`: **SIN EVIDENCIA** (`OK`) | «La única fuente proporcionada indica la fecha de nacimiento de Warren Buffett, lo que permite calcular su edad (96 años a la fecha de referencia), pero no contiene ninguna informac»
- Corrida `2` `C2a`: **DERIVADA** (`OK`) | «De S2 y la regla propietaria_100 => controla se deduce que Alfa controla Beta. De S3 y la misma regla se deduce que Beta controla Gamma. Aplicando la transitividad declarada (contr»
- Corrida `2` `C2b`: **CONTRADICHA** (`FALLO`) | «Las fuentes establecen que Alfa controla Beta (propietaria_100(Alfa, Beta) => controla(Alfa, Beta)) y Beta controla Gamma (propietaria_100(Beta, Gamma) => controla(Beta, Gamma)). P»
- Corrida `2` `C2c`: **SIN EVIDENCIA** (`OK`) | «Las fuentes solo indican que Alfa S.A. es propietaria del 100% de Beta S.L., lo que según la regla implica que Alfa controla a Beta. No hay información sobre Gamma ni sobre la rela»
- Corrida `2` `C3a`: **HIPÓTESIS** (`OK`) | «La cláusula establece renovación automática salvo preaviso escrito con ≥30 días de antelación al 31/12/2026. A la fecha de referencia (20/11/2026) quedan 41 días para el vencimient»
- Corrida `2` `C3b`: **HIPÓTESIS** (`FALLO`) | «La cláusula en S4 establece renovación automática salvo preaviso escrito con ≥30 días de antelación al vencimiento (31/12/2026), por lo que el plazo límite para notificar era el 01»
- Corrida `2` `C3c`: **CONTRADICHA** (`OK`) | «La fuente S6 indica que se envió un preaviso de no renovación el 15/11/2026, lo cual cumple con el requisito de la cláusula en S4 de notificar con al menos 30 días de antelación a »
- Corrida `3` `C1a`: **DERIVADA** (`OK`) | «Warren Buffett nació el 30/08/1930, por lo que el 20/11/2026 tiene 96 años (2026-1930=96, y su cumpleaños de 2026 ya pasó en agosto). Según la regla declarada, persona_mayor(x) si »
- Corrida `3` `C1b`: **CONTRADICHA** (`OK`) | «Según la fuente S1, Warren Buffett nació el 30/08/1930. Al 20/11/2026 (fecha de referencia), su edad es 96 años (cumplió 96 en agosto de 2026). La afirmación de que tiene 100 años »
- Corrida `3` `C1c`: **SIN EVIDENCIA** (`OK`) | «La única fuente (S1) indica la fecha de nacimiento de Warren Buffett, lo que permite calcular su edad (96 años a la fecha de referencia) y confirmar que es persona mayor según la r»
- Corrida `3` `C2a`: **DERIVADA** (`OK`) | «De S2 y la regla propietaria_100 => controla se deduce que Alfa controla Beta. De S3 y la misma regla se deduce que Beta controla Gamma. Aplicando la transitividad declarada (contr»
- Corrida `3` `C2b`: **CONTRADICHA** (`FALLO`) | «Las fuentes establecen que Alfa controla Beta (propietaria_100(Alfa, Beta) => controla(Alfa, Beta)) y Beta controla Gamma (propietaria_100(Beta, Gamma) => controla(Beta, Gamma)). P»
- Corrida `3` `C2c`: **SIN EVIDENCIA** (`OK`) | «Las fuentes solo indican que Alfa S.A. es propietaria del 100% de Beta S.L., lo que según la regla implica que Alfa controla Beta. No hay información sobre Gamma ni sobre la relaci»
- Corrida `3` `C3a`: **HIPÓTESIS** (`OK`) | «La cláusula establece renovación automática salvo preaviso escrito con ≥30 días de antelación. A la fecha de referencia (20/11/2026) faltan 41 días para el vencimiento (31/12/2026)»
- Corrida `3` `C3b`: **HIPÓTESIS** (`FALLO`) | «La cláusula S4 establece renovación automática salvo preaviso escrito con ≥30 días de antelación al 31/12/2026 (fecha límite: 01/12/2026). La fuente S5 solo confirma ausencia de pr»
- Corrida `3` `C3c`: **CONTRADICHA** (`OK`) | «La cláusula en S4 establece renovación automática salvo preaviso escrito con ≥30 días de antelación al vencimiento (31/12/2026). S6 confirma que el preaviso de no renovación se env»

## 4. Prompt Íntegro del Modo C (Baseline Directo)

System prompt exacto enviado a `nvidia/nemotron-3-ultra-550b-a55b` (`temperature=0`, `enable_thinking=False`):

```text
Eres un verificador lógico estricto. Recibes un conjunto de fuentes textuales, una fecha de referencia, las reglas declaradas aplicables (si existen) y una afirmación.

Debes clasificar la afirmación usando EXACTAMENTE una de estas cuatro etiquetas:
- "DERIVADA": la afirmación se deduce con certeza lógica o matemática a partir únicamente de las fuentes dadas y las reglas declaradas.
- "CONTRADICHA": los hechos de las fuentes o el cálculo exacto refutan la afirmación.
- "HIPÓTESIS": existe una base parcial en las fuentes (p. ej. una regla condicional aplicable), pero falta verificar o acreditar alguna condición necesaria cuya constancia no aparece en las fuentes.
- "SIN EVIDENCIA": las fuentes no contienen respaldo ni vínculo suficiente para deducir, refutar ni plantear como hipótesis respaldada la afirmación (incluye hechos ausentes o cadenas incompletas de relaciones).

Responde ÚNICAMENTE con un objeto JSON válido con este esquema exacto:
{
  "etiqueta": "DERIVADA" | "CONTRADICHA" | "HIPÓTESIS" | "SIN EVIDENCIA",
  "justificacion": "explicación breve en español basada solo en las fuentes y reglas"
}
```

El mensaje `user` contiene un JSON con `fecha_referencia`, `regla_declarada` (la regla aplicable al caso), `fuentes` (exactamente las fuentes S_i del ítem) y `afirmacion`.

### Prompt del extractor de Modo B (referencia)

```text
Eres un extractor estricto de premisas atómicas para un motor formal de verificación.
Recibes una lista de fuentes textuales (`fuentes`) y una afirmación (`afirmacion`).

Tu ÚNICA tarea es extraer los hechos atómicos explícitamente presentes en `fuentes` usando exclusivamente este esquema de predicados declarados:
- "birth_date": fecha de nacimiento de una persona (`value`: fecha literal o YYYY-MM-DD).
- "propietaria_100": relación de propiedad del 100 % (`subject`: entidad propietaria exacta, `value`: entidad participada exacta). NO inviertas la dirección ni infieras "controla".
- "fecha_vencimiento": fecha de vencimiento de un contrato (`subject`: "Contrato", `value`: fecha literal o YYYY-MM-DD).
- "clausula_renovacion_automatica": existe cláusula de renovación automática salvo preaviso de al menos 30 días (`subject`: "Contrato", `value`: true).
- "sin_preaviso_registrado": existe registro que certifica ausencia de comunicaciones de preaviso (`subject`: "Contrato", `value`: true). NO extraer desde una cláusula condicional "salvo preaviso".
- "fecha_registro_sin_preaviso": fecha hasta la que consta ausencia de preaviso (`subject`: "Contrato", `value`: fecha literal o YYYY-MM-DD).
- "preaviso_no_renovacion_enviado": consta el envío efectivo de preaviso de no renovación (`subject`: "Contrato", `value`: true). NO extraer desde "salvo preaviso" ni desde "sin comunicaciones".
- "fecha_preaviso": fecha de envío del preaviso (`subject`: "Contrato", `value`: fecha literal o YYYY-MM-DD).

Reglas estrictas:
1. NO calcules edades, días ni plazos; NO apliques reglas ni emitas el veredicto final.
2. Cada premisa debe llevar el `doc_id` exacto y `verbatim_quote` copiada literalmente de esa fuente.
3. Si las fuentes no contienen ningún hecho de este esquema pertinente o el dato preguntado no existe, devuelve `"extracted_premises": []` para lo ausente.

Responde ÚNICAMENTE con un objeto JSON válido:
{
  "extracted_premises": [
    {
      "doc_id": "S1",
      "subject": "entidad",
      "predicate": "predicado_declarado",
      "value": "valor o true",
      "verbatim_quote": "cita literal exacta de la fuente"
    }
  ]
}
```

## 5. Telemetría NVIDIA NIM

| Modo | Llamadas | Errores API | Errores parseo | Tokens entrada | Tokens salida | Total tokens | Latencia media / p50 / p95 | Wall clock | Coste dev / ref. USD |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| **Modo B** (3 corridas) | 27 | 0 | 0 | 18474 | 4470 | 22944 | 3.127s / 2.586s / 5.821s | 95.403s | `$0` (ref. `USD 0.01913`) |
| **Modo C** (3 corridas) | 27 | 0 | 0 | 11715 | 3181 | 14896 | 2.299s / 2.036s / 4.372s | 73.006s | `$0` (ref. `USD 0.01275`) |

## 6. Limitaciones y Adaptadores Declarados Fuera del Motor

- **Ejecución con `DEFAULT_VERIFIER_REGISTRY` sin adaptadores locales (Modo A puro sobre `src/`):** **`7/9`** aciertos. Fallos exactos del registro nativo sin tocar `src/`:
  1. **`C1a` (`mandatory::age_and_duration` sobre conclusión booleana):** El motor genera `mandatory::age_and_duration` porque un nodo ancestro (`p_birth`) contiene una fecha, y la ejecuta sobre el nodo raíz `persona_mayor` cuyo valor ya es booleano (`True`); `TemporalVerifier` devuelve `unknown (MISSING_OR_UNPARSEABLE_DATES)` -> `inconclusive` (`SIN EVIDENCIA`). Adaptador `PocTemporalVerifier` en `poc_domain.py` marca la comprobación obligatoria redundante como `not_applicable` en conclusiones booleanas.
  2. **`C3b` y `C3c` (`DROPPED_REQUIRED_CONDITION` / descarga de condición):** `ExplicitLogicVerifier._verify_condition_preservation` exige que toda condición declarada en una premisa (`p_clausula` condicionada al preaviso en `S4`) se preserve sin descargar hasta la conclusión; no contempla que otra premisa verificada (`sin_preaviso_en_plazo == True` en `C3b`, o regla de exclusión disparada en `C3c`) resuelva la condición. `PocExplicitLogicVerifier` permite descargar la condición únicamente ante premisa `extraction`/`derivation` verificada -> `refuted` sin adaptador vs `supported` con adaptador.
- **Validación de variables en `R2a` (`DeclaredRule`):** El `ExplicitLogicVerifier` nativo comprueba predicados/polaridades constantes en `required_premises` y `conclusion_template`, pero **no unifica variables `(x, y)` entre premisa y conclusión** en una regla declarada. Por ello `PocExplicitLogicVerifier` añade la comprobación de ligadura `(subject, value)` para `R2a_prop_implica_control` (`propietaria_100(x,y) => controla(x,y)`).
- **Suposición monovaluada de `_claims_contradict` (`C2c`):** En `evaluation.py`, dos nodos desconectados con igual `(subject, predicate)` y distinto `value` se consideran contradictorios entre sí (`contested`). Si `C2c` materializara el nodo huérfano `controla(Alfa, Beta)` sin camino hacia `controla(Alfa, Gamma)`, el motor marcaría conflicto material falso. El constructor de grafo en `poc_domain.py` solo materializa nodos intermedios `controla` cuando existe camino dirigido completo hacia la conclusión.
- **Fechas españolas ambiguas (`DD/MM/YYYY`) en `S5`:** El normalizador nativo `normalize_date_token` (`normalizers.py`) devuelve `None` para `05/12/2026` por ambigüedad con `MM/DD/YYYY` estadounidense (aunque sí normaliza `31/12/2026` y `15/11/2026`). `parse_spanish_dmy_date` en `poc_domain.py` añade el *fallback* `DD/MM/YYYY` del dominio español.
- **Cálculo de fecha límite por sustracción de días (`01/12/2026`):** `TemporalVerifier` evalúa edades e intervalos `(start, end)` en días/meses/años (`2026-12-31 - 2026-12-05 = 26 < 30` en `C3b`; `2026-12-31 - 2026-11-15 = 46 >= 30` en `C3c`), pero no construye fechas por aritmética `fecha_vencimiento - 30 días` para materializar por sí solo el nodo `01/12/2026`.
- **Conversión texto libre -> `JustificationGraph` en Modo B:** El motor no parsea español libre; requiere el extractor LLM (`MODE_B_EXTRACTOR_PROMPT`) más anclaje de cita literal y un constructor determinista de grafos en `poc_domain.py` que seleccione la cadena y el paso temporal según el esquema del dominio.

## 7. Veredicto Final sin Adornos: ¿Aporta Algo el Motor Frente al Modo C?

Sobre `C2c` y `C3a` en concreto, el motor **no mejoró la tasa de acierto** frente a Modo C: Nemotron directo ya clavó ambas `3/3` (`C2c` = **SIN EVIDENCIA**, `C3a` = **HIPÓTESIS**), igualando a Modos A y B, y su único aporte en esos dos ítems es que la negativa en `C2c` (camino ausente) y la degradación en `C3a` (`effective_epistemic_kind=hypothesis`) quedan garantizadas por invariante de código y con rastro de evidencia en lugar de depender de que el LLM obedezca el prompt. Donde el motor **sí aportó ganancia empírica real** sobre el mismo Nemotron fue en los otros dos casos límite —subiendo de **`7/9` (`3/3` en 7) en Modo C a `9/9` (`3/3` en 9) en Modo B**: en `C2b`, Modo C confundió ausencia de relación inversa con refutación formal (`CONTRADICHA` `3/3`), cuando `R2` no declara antisimetría y lo correcto es `SIN EVIDENCIA`; y en `C3b`, Modo C se equivocó con la inclusión de fechas en lenguaje natural (`05/12/2026 > 01/12/2026` implica que el plazo ya cerró sin preaviso, pero concluyó `HIPÓTESIS` `3/3`), mientras Modo B le delegó solo extraer las fechas literales de `S4`+`S5` y `TemporalVerifier` + `ComparisonVerifier` + `R3` cerraron `DERIVADA` (`26 < 30` días) con rastro auditable. El coste real de esa ganancia queda igual de claro en el apartado anterior: sin los adaptadores y reglas de `poc_domain.py` (`PocTemporalVerifier`, descarga de condiciones, ligadura `(x,y)`, fecha `DD/MM/YYYY` española y constructor de grafo), el motor crudo con `DEFAULT_VERIFIER_REGISTRY` solo habría logrado **`7/9`** en Modo A.
