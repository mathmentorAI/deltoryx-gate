# Desarrollo de Gate 4.0.0

El trabajo continúa exclusivamente sobre las [seis capas](gate_layers_architecture.md) y el contrato `gate-layers-1`. Estas prioridades describen mejoras de la implementación actual, sin introducir otro motor ni otra ruta de producto.

1. **Interpretación de la tarea.** Mejorar la cobertura de cláusulas, la selección de sujetos y la localización del bloque relevante. La LLM responde de formular la pregunta adecuada; el núcleo conserva su responsabilidad formal.
2. **Fuentes y premisas.** Conservar estructura, revisiones, coordenadas, unidades y procedencia de normalización. Representar expresamente las referencias ambiguas y el contexto de tablas.
3. **Derivaciones y decisiones.** Mejorar la construcción de planes completos y ejecutables, manteniendo independientes los contratos de verificación y la política de aplicación. Toda autorización debe quedar vinculada a contexto, operación, payload y política vigente.
4. **Evaluación independiente.** Medir utilidad para revisión, errores materiales, abstenciones, intentos y esfuerzo humano sobre casos reservados y adjudicados independientemente. Las comparaciones de producto recorren las seis capas con una LLM real por afirmación y control; las etiquetas esperadas permanecen fuera del contexto del modelo.

La validación de ingeniería se informa por separado. Los resultados disponibles no demuestran aún una mejora de priorización de los comentarios OCR. Véase el [contrato de integración](gate_layers_api.md) para las garantías y límites actuales.
