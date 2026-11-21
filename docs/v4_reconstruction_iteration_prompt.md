# Prompt de iteración: reconstrucción trazable con operaciones propuestas por Nemotron

Trabaja en el repositorio local de Deltoryx Gate (plataforma `feat/v4-claims-verify-api`, submódulo `claim-layer` en `feat/v4-derivation-verifier`). Conserva los commits existentes y la primera medición de PR-MCI (`case_studies/pr_mci_claude_derivation`, commit `9a3d06b`) sin modificarla. No hagas reset, rebase, push ni cambios en `main`.

## Dirección de producto (Evidence Intelligence)

Una IA genera un resultado a partir de unas fuentes. Deltoryx Gate toma ese resultado como afirmación a justificar y pide a Nemotron que lo reconstruya desde las fuentes mediante una cadena de operaciones explícitas. **Nemotron puede proponer las operaciones que necesite**; ninguna operación es una caja negra. El producto es la reconstrucción trazable, y la fiabilidad de la conclusión depende del tipo de paso más débil que haya hecho falta. Una operación propuesta no bloquea la reconstrucción: queda registrada y rebaja el grado de fiabilidad.

Esto sustituye el criterio anterior «solo operaciones del catálogo de la aplicación». Se mantiene una frontera: Deltoryx nunca ejecuta código generado por el modelo. Las operaciones propuestas se expresan como **datos** en un lenguaje de expresiones cerrado que interpreta Deltoryx, o quedan como afirmaciones del modelo sin ejecutar.

## Escala de fiabilidad (eslabón más débil)

| Grado | Paso | Quién garantiza el significado |
| --- | --- | --- |
| A | Premisa con valor comprobado literalmente en la cita + operación del núcleo | Texto fuente y código de Deltoryx |
| B | Regla de dominio declarada por la aplicación | El operador |
| C | Operación propuesta por Nemotron como expresión sobre primitivas del núcleo, ejecutada por Deltoryx | Deltoryx la ejecuta; Nemotron eligió la composición |
| D | Operación propuesta por Nemotron sin definición ejecutable; su resultado es el que afirma el modelo | Solo Nemotron (registrada y auditable) |
| E | Premisa cuyo valor no se puede comprobar en la cita (interpretación del modelo) | Nadie más que el modelo |

El grado de una conclusión es el mínimo de los grados de las premisas y pasos de los que depende. La aplicación fija qué grado mínimo permite `PASS` (por defecto, B). Por debajo de ese mínimo, el dictamen epistémico se conserva pero el enrutamiento pasa a `ESCALATE` con el eslabón débil identificado.

## Entregables

1. **Operaciones propuestas.** El planificador puede devolver `proposed_operations` (nombre, descripción, entradas, `expression` u omisión, tipo y unidad de salida) y usarlas en `steps`. Deltoryx valida el esquema, interpreta las expresiones con un intérprete cerrado (sin acceso a atributos, E/S, bucles ni llamadas fuera de la lista blanca, con límites de profundidad y tamaño) y re-ejecuta cada paso en la verificación formal. Una operación sin expresión exige `proposed_value` en el paso y queda como grado D.
2. **Fidelidad de premisas.** Clasifica cada premisa admitida como `literal` (valor comprobado contra la cita por normalizadores deterministas) o `interpreted` (por ejemplo, booleanos o etiquetas que no aparecen en la cita). Una derivación por `identity` sobre una premisa interpretada es grado E.
3. **Grado y traza.** El resultado público (biblioteca, CLI, API) incluye un bloque `reliability`: grado, eslabón más débil, grado de cada premisa y paso, operaciones propuestas con su definición y política aplicada.
4. **Correcciones generales detectadas en la primera medición** (sin adaptadores por caso):
   - Comparación con tipos: una conclusión de tipo incomparable con la afirmación (por ejemplo, booleano frente a texto) produce `unresolved / incomparable_conclusion` reparable, nunca `verified_refutation`.
   - Presupuesto de salida del planificador configurable en la aplicación, con un valor por defecto en la API suficiente para evidencias grandes; se registra en la procedencia.
5. **Prompt versionado.** Nueva versión del prompt de planificación que explique `proposed_operations`, el lenguaje de expresiones y la preferencia por premisas literales. Conserva la versión anterior para reproducir la evidencia 14/14.
6. **Documentación** de la nueva dirección en `docs/v4_design.md`.

## Criterios de aceptación

- Pruebas deterministas: intérprete seguro (rechaza nodos desconocidos, exceso de profundidad, tipos inválidos); operación compuesta ejecutada y re-verificada (grado C); operación afirmada (grado D) que da `derived` pero `ESCALATE` con la política por defecto y `PASS` si la aplicación baja el umbral; `identity` sobre premisa interpretada → grado E.
- Las 14 propuestas reales del benchmark temporal, reproducidas por la API, siguen dando el mismo dictamen y quedan en grado A con `PASS`.
- Las 20 propuestas reales de la primera medición PR-MCI, reproducidas offline con el motor nuevo: ningún `PASS` nuevo sobre referencias refutadas o insuficientes; C15 deja de ser una refutación verificada; C03 queda por debajo del umbral de `PASS`. Esto es un replay, no una medición en vivo.
- Suite de regresión completa sin fallos.
- Segunda medición en vivo, con los mismos manifiestos congelados y sin tocar el expected: 20 casos PR-MCI y 14 casos temporales con el nuevo prompt, guardados en archivos nuevos. Registra grado, operaciones propuestas y su tipo, tokens y latencias. Si el proveedor no es accesible desde el entorno, deja el comando preparado y marca la medición como pendiente.

## Fuera de alcance en esta iteración

Promoción automática de operaciones de D a C o de C a B (aprobación, tests o uso acumulado), operaciones de dominio para código (líneas de base frente a head, búsqueda de llamadas, ejecución de tests aislada) y evaluación independiente. Quedan como trabajo siguiente explícito.
