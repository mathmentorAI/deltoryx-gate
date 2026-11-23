# Deltoryx Gate 4.0.0

Biblioteca de verificación por seis capas. Contrato público único: `gate-layers-1`.

La LLM interpreta una afirmación y construye una derivación nueva a partir de sus fuentes. El núcleo contrasta referencias, tipos, unidades y operaciones. La política de aplicación determina los permisos; el comportamiento predeterminado es informativo y no autoriza acciones.

```bash
python -m pip install -e '.[dev]'
export NVIDIA_API_KEY='configurar en el entorno'
deltoryx-gate verify-claim --source 'doc_1=Hoy es viernes y quedamos dos días antes.' --claim 'La cita fue el miércoles.'
```

```python
import os
from deltoryx_gate import build_derivation_planner_model, verify_claim

model = build_derivation_planner_model(os.environ["NVIDIA_API_KEY"])
result = verify_claim({"doc_1": "Texto original de la fuente"}, "Afirmación original", model=model)
receipt = result.to_dict()
print(receipt["verification"]["payload"])
print(receipt["decision"]["payload"])  # Consultar authorized; INFORM no autoriza.
```

`deltoryx_gate`, `claimlayer` y `claimlayer.v4` apuntan a la misma implementación. El módulo `claimlayer.v4.planner` conserva la entrada definida en el protocolo de evaluación del proyecto. No hay selector de motor ni compatibilidad con otros contratos de producto.

- [Arquitectura y responsabilidades](docs/gate_layers_architecture.md).
- [Contrato e integración](docs/gate_layers_api.md).
- [Licencia](LICENSE).

Las pruebas de `tests/` son comprobaciones de ingeniería, no una medida de calidad de la LLM. Las evaluaciones de producto se ejecutan con proveedor real desde la plataforma mediante `scripts/evaluate_gate_layers.py`, guardando cada propuesta y recibo. Las seis capas y sus auxiliares viven bajo `src/claimlayer/v4/`; el transporte y los contratos internos de documentos están en `src/claimlayer/semantic/`.
