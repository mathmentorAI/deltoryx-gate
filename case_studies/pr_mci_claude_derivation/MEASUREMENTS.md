# Registro de mediciones (PR-MCI Claude + benchmark temporal)

Cada medición registra por separado: **revisión ejecutada** (código que produjo los resultados), **revisión de resultados** (commit de claim-layer donde se guardan) y **puntero del submódulo** en la plataforma en el momento de registrar los resultados. Las mediciones anteriores se conservan intactas como referencia.

| Id | Fecha (UTC) | Revisión ejecutada (plataforma / claim-layer) | Prompt | Archivos de resultados | Revisión de resultados | Puntero del submódulo |
|---|---|---|---|---|---|---|
| M1 | 2026-11-21 12:19 | `4fd6ed9` / `b7295f7` | v4-derivation-planner-2 | `results/live_results.json` | `9a3d06b` | `0386998` |
| M2 | 2026-11-21 19:08 (PR) · 18:59 (temporal) | `b8108d5` / `fba49d2` | v4-derivation-planner-3 | `results/live_results_v3.json`, `benchmarks/v4_derivation_planner_live_results_v3.json` | `0100433` | `fba49d2` |
| M3 (etapa 1) | 2026-11-21 20:41 (PR) · 20:43 (temporal) | `48d936c` / `94e3d4f` | v4-derivation-planner-3 + catálogo con `count_lines` | `results/live_results_stage1.json`, `benchmarks/v4_derivation_planner_live_results_stage1.json` | `8e83632` | `94e3d4f` |
| M4 (etapa 2) | 2026-11-22 07:38 (PR) · 07:40 (temporal) · 07:43 (transferencia) | `a51acf3` / `dc3c6a8` | v4-derivation-planner-4 (rangos de líneas, lados del diff, convenio de recuento) | regresión: `results/live_results_stage2.json`, `benchmarks/v4_derivation_planner_live_results_stage2.json`; transferencia: `../pr_mci_transfer_sample/results/live_results_stage2.json` | `6b4bc12` | `dc3c6a8` |
| M5 (etapa 3) | pendiente | se rellena tras la ejecución | v4-derivation-planner-5 (presencia tipada, recuento con alcance, definiciones de recuento) | regresión: `results/live_results_stage3.json`, `benchmarks/v4_derivation_planner_live_results_stage3.json`, `../pr_mci_transfer_sample/results/live_results_stage3.json`; transferencia 2: `../pr_mci_transfer_sample_2/results/live_results_stage3.json` | pendiente | pendiente |

Notas:
- M5 va precedida de una medición del verificador sin modelo (`stage3_replay/`): planes de M4 con el verificador de M4 (`dc3c6a8`) y el de la etapa 3 (`9836ed1`).
- Errata de M4: la revisión etiquetó C14 como «veredicto aceptable»; el veredicto registrado fue `unresolved`, que no es aceptable. Las métricas calculadas de M4 son correctas. Los archivos de M4 no se modifican.
