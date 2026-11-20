# Guía de Anotación Humana — Diagnóstico de Atribución de Errores Fase 3 (`n = 120`)

## 1. Propósito y Cegamiento
Este paquete contiene **120 afirmaciones conductuales** muestreadas de forma estratificada de los 200 PRs originales de `AIDev-Dev` (baseline congelado `phase3-v2.0`, excluyendo las 50 afirmaciones auditadas anteriormente).

- **Cegamiento estricto:** La plantilla y el visor NO muestran el agente de código (`Claude_Code`, `Copilot`, `Cursor`, `Devin`, `OpenAI_Codex`), ni el estado del modelo (`no_premises_in_diff`, `hypothesis_with_verified_premises`, `derived_supported`), ni la operación/premisas propuestas por Nemotron, ni el estrato del ítem.
- **No abrir `blind_key.json`** antes de completar y validar los 120 ítems.

---

## 2. Cómo Abrir el Visor de Diff Interactivo
Tienes tres formas de trabajar sobre exactamente los mismos 120 ítems (`DIAG-001` .. `DIAG-120`):

1. **Visor HTML interactivo (recomendado, 100 % offline en navegador):**
   Abre directamente en Safari/Chrome/Firefox:
   `~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/diagnostic_120/diff_viewer.html`
   - Busca dentro de todo el diff del PR con la barra de búsqueda o filtra por fichero en el panel lateral.
   - Haz **clic en cualquier línea del diff** para añadir/quitar `{file, chunk_id, diff_line_no, literal_quote}` en `gold_premises` (amarillo) o en `contradicting_premises` (rojo, cambiando el modo de clic).
   - El progreso se autoguarda en `localStorage` del navegador. Usa **Importar JSONL** para reanudar desde un fichero guardado y **Exportar JSONL** (`annotations_completed.jsonl`) o **Exportar CSV** (`annotations_completed.csv`) al terminar.

2. **Hojas Markdown individuales + plantilla JSONL/CSV:**
   - Hojas legibles con líneas numeradas (`L0001 | +...`): `sheets/DIAG-001.md` .. `sheets/DIAG-120.md`.
   - Plantillas en blanco: `annotations_template.jsonl` y `annotations_template.csv`.

3. **Visor de terminal (CLI):**
   ```bash
   cd "claim-layer/case_studies/pr_mci_phase3"
   ../../../.venv/bin/python diagnostic_pack.py view DIAG-001 --search <término>
   ```

---

## 3. Campos a Rellenar por Afirmación

### A. `annotation_status` (obligatorio)
- **`supported`**: El diff contiene al menos una línea concreta que respalda esta afirmación. Requiere **`>= 1` premisa en `gold_premises`**.
- **`no_premises`**: Es una afirmación sobre el código/comportamiento que afirma algo sustantivo, pero **no existe respaldo en el diff** para ella (alucinación, componente omitido o cambio inexistente) sin contradicción explícita en sentido contrario, o es demasiado ambigua sin rastro en el patch. Debe tener **`gold_premises` vacío**.
- **`contradicted`**: El diff **contradice directamente** lo que dice la afirmación (por ejemplo: afirma añadir cuando el diff elimina; afirma añadir tests pero hay 0 cambios de tests; indica versión `2.0` cuando el manifiesto pone `1.4`; afirma solo cambios de docs mientras edita código fuente). Requiere `diff_contradicts_claim = true` y conviene marcar las líneas contradictorias en `contradicting_premises`.
- **`out_of_scope_meta`**: No es una afirmación técnica sobre los cambios del PR, sino texto meta/plantilla/pie automático (`"Co-Authored-By"`, `"Generated with..."`, checklist vacío `- [ ]`, enlace de issue `#123`, encabezado genérico sin contenido). Debe tener **`gold_premises` vacío**.

### B. `diff_contradicts_claim` (obligatorio: `true` / `false`)
- Marca `true` si el diff aporta evidencia directa que refuta o contradice la afirmación (aunque otra parte pudiera tocar el área; obligatoriamente `true` cuando `annotation_status = "contradicted"`). Marca `false` en caso contrario.

### C. `gold_operation` (obligatorio, 1 de las 8 operaciones)
1. `implements`: Implementa, añade o modifica lógica, función, clase, método o símbolo de código fuente.
2. `adds_test_for`: Añade o modifica tests unitarios, de integración, E2E o snapshots/fixtures de test.
3. `removes_support_for`: Elimina, borra o retira soporte, ruta o símbolo existente.
4. `configures`: Configura ajustes, workflows CI/CD, YAML/JSON/TOML, variables/flags o esquema de config.
5. `updates_dependency`: Añade, actualiza o fija versiones de paquetes/dependencias en manifiestos o lockfiles.
6. `documents`: Añade o actualiza documentación (`README`, `.md`, `docs/`, docstrings o comentarios explicativos).
7. `other_behavior`: Afirmación conductual sustantiva y verificable que no encaja limpiamente en las 6 anteriores (p. ej. refactor transversal o descripción de flujo general).
8. `out_of_scope_meta`: Texto meta, boilerplate de plantilla o enlace sin afirmación sobre el código.

### D. `gold_target` (texto corto o vacío/`null`)
El símbolo exacto, ruta/nombre de fichero, clave/flag de configuración o nombre/versión de paquete citado como objetivo principal de la afirmación (p. ej., `compute_retry_backoff`, `tsconfig.json`, `lodash 4.17.21`). Déjalo vacío si la afirmación es genérica sin identificador concreto.

### E. `gold_premises` / `contradicting_premises`
Lista de líneas exactas seleccionadas del diff (con `{file, chunk_id, diff_line_no, literal_quote}`). Selecciona **las líneas mínimas suficientes** que prueban (o refutan) la afirmación:
- Para `implements`: prioriza la línea `+` de declaración/definición del símbolo o las líneas `+` donde se implementa la conducta descrita (no hace falta marcar archivos enteros).
- Para `adds_test_for`: marca la(s) línea(s) `+` del test que cita/ejercita ese comportamiento.
- Para `removes_support_for`: marca la(s) línea(s) `-` clave eliminada(s).
- Para `configures` / `updates_dependency` / `documents`: marca la línea exacta de config/manifiesto/doc.

### F. `confidence` y `notes`
- `confidence` (obligatorio): `high`, `medium`, o `low`.
- `notes` (opcional): breve aclaración útil (especialmente si `confidence = "low"`, `contradicted`, o casos límite).

---

## 4. Validación antes de Entregar
Guarda tu fichero exportado fuera de git (por ejemplo en `~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/diagnostic_120/annotations_completed.jsonl`) y ejecuta el validador determinista:

```bash
cd "claim-layer/case_studies/pr_mci_phase3"
../../../.venv/bin/python diagnostic_pack.py validate \
  --annotations "~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/diagnostic_120/annotations_completed.jsonl" \
  --pack-dir "~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/diagnostic_120"
```

Cuando el validador devuelva `"valid": true` (`120 / 120`, citas verificadas verbatim en el diff), avisa indicando la ruta del fichero para ejecutar los **puntos 2–5** del diagnóstico.
