# Registro de mediciones (PR-MCI Claude + benchmark temporal)

Cada medición registra por separado: **revisión ejecutada** (código que produjo los resultados), **revisión de resultados** (commit de claim-layer donde se guardan) y **puntero del submódulo** en la plataforma en el momento de registrar los resultados. Las mediciones anteriores se conservan intactas como referencia.

| Id | Fecha (UTC) | Revisión ejecutada (plataforma / claim-layer) | Prompt | Archivos de resultados | Revisión de resultados | Puntero del submódulo |
|---|---|---|---|---|---|---|
| M1 | 2026-11-21 12:19 | `4fd6ed9` / `b7295f7` | v4-derivation-planner-2 | `results/live_results.json` | `9a3d06b` | `0386998` |
| M2 | 2026-11-21 19:08 (PR) · 18:59 (temporal) | `b8108d5` / `fba49d2` | v4-derivation-planner-3 | `results/live_results_v3.json`, `benchmarks/v4_derivation_planner_live_results_v3.json` | `0100433` | `fba49d2` |
| M3 (etapa 1) | 2026-11-21 20:41 (PR) · 20:43 (temporal) | `48d936c` / `94e3d4f` | v4-derivation-planner-3 + catálogo con `count_lines` | `results/live_results_stage1.json`, `benchmarks/v4_derivation_planner_live_results_stage1.json` | `8e83632` | `94e3d4f` |
| M4 (etapa 2) | pendiente | se rellena tras la ejecución | v4-derivation-planner-4 (rangos de líneas, lados del diff, convenio de recuento) | regresión: `results/live_results_stage2.json`, `benchmarks/v4_derivation_planner_live_results_stage2.json`; transferencia: `../pr_mci_transfer_sample/results/live_results_stage2.json` | pendiente | pendiente |
