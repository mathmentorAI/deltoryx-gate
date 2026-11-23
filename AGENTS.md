# Evaluaciones de Deltoryx Gate y Deltoryx Code

Cada prueba nueva del producto debe ejecutar v4 con una LLM real que construya el motor de inferencia específico de cada caso. La entrada es la afirmación original y sus fuentes. La LLM propone las premisas, sus referencias y las operaciones que las conectan con la conclusión. El verificador de Deltoryx contrasta las premisas con las fuentes, comprueba las operaciones y ejecuta la derivación admitida.

Usa `POST /api/v1/v4/claims/verify` o `claimlayer.v4.planner.verify_claim` con el modelo configurado. En una comparación con otra herramienta, aplica este recorrido a cada hallazgo y también a cada control modificado. Registra el modelo, la propuesta recibida, el resultado de la verificación y su justificación. Una propuesta defectuosa de la LLM cuenta como fallo del recorrido y debe constar en los resultados.

Queda descartado ejecutar evaluaciones de producto con reglas estáticas, verificadores aislados, planes escritos a mano, propuestas grabadas, cachés de respuestas o modelos simulados. Esto también se aplica a los pilotos, las comparativas y los modos por defecto de sus ejecutables. Si el proveedor está inaccesible, detén la prueba, identifica los casos pendientes y registra el bloqueo de infraestructura por separado. No ejecutes una alternativa ni atribuyas al producto un resultado para esos casos.

Los resultados históricos deben interpretarse según la versión y el recorrido que realmente se ejecutaron.

## Única versión de producto

El desarrollo continúa exclusivamente sobre Deltoryx Gate 4.0.0 y la arquitectura de seis capas con contrato `gate-layers-1`. Las entradas son `claimlayer.v4.planner.verify_claim`, `POST /api/v1/v4/claims/verify` y `deltoryx-gate verify-claim`. `deltoryx_gate` y `claimlayer` importan la misma implementación.

No se deben restaurar motores, rutas, interfaces, ejemplos ni ejecutables de versiones anteriores, ni introducir un selector para utilizarlos. Los auxiliares internos de extracción y cálculo forman parte del núcleo actual. Las comprobaciones de ingeniería de contratos y empaquetado se informan separadas de las evaluaciones de producto con LLM real.
