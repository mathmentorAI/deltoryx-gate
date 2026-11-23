# Gate: contrato e integración

Contrato único `gate-layers-1`, biblioteca `deltoryx-gate` 4.0.0.

## Uso en Python

```python
import os
from claimlayer.v4 import build_derivation_planner_model, verify_claim

model = build_derivation_planner_model(os.environ["NVIDIA_API_KEY"])
result = verify_claim(
    {"doc_1": "Hoy es viernes y quedé con ella dos días antes."},
    "La cita fue el miércoles.",
    model=model,
)

for goal in result.verification.payload["goals"]:
    print(goal["target"], goal["formal_comparison"], goal["conclusion"])
print(result.decision.payload)  # Información; sin política de acción, no autoriza.
receipt = result.to_dict()
```

El ejemplo solicita una derivación nueva; no suministra un grafo ni promete que el modelo produzca una propuesta válida.

## Entrada HTTP

```json
{
  "claim": "Afirmación original del comentario",
  "context": {"path": "src/example.ts", "reviewed_revision": "SHA revisado", "completeness": "partial"},
  "sources": [
    {"document_id": "file_01", "text": "Parche original completo entregado", "path": "src/example.ts", "reviewed_revision": "SHA revisado", "kind": "unified_diff", "completeness": "partial"}
  ],
  "unit_profile": "auto"
}
```

`context` y los metadatos son opcionales. También se acepta `as_of`. El cliente no puede suministrar grafos, reglas ni políticas. `operation`, cuando se solicita evaluar una acción, contiene `operation_id` y `payload`; no configura la política del servidor.

Las fuentes conservan la identidad del documento y su revisión de contenido. `reviewed_revision` describe la revisión externa, no sustituye el hash interno. Los rangos que propone el modelo se refieren al texto del documento entregado, desde la línea 1.

## Respuesta

La respuesta contiene `contract_version`, `context`, `task`, `premises`, `plan`, `verification`, `decision` y `trace`. La API añade `receipt_id` y `created_at`. Cada artefacto tiene `payload`, procedencia, versión, identidad, digest y padres. En un fallo temprano, los artefactos que aún no existen son `null`.

- `task.payload.goals`: objetivos interpretados por la LLM.
- `premises.payload.admitted` y `.rejected`: premisas admitidas y rechazadas.
- `verification.payload.goals`: estado formal, comparación, conclusión y pasos por objetivo.
- `verification.payload.evidence_checks` y `.source_issues`: comprobaciones independientes contra las fuentes.
- `decision.payload`: disposición, autorización, alcance y razones de política.
- `trace.attempts`: propuestas crudas, reparaciones y fallos, junto al modelo y consumo.

Los adaptadores personalizados de planificación implementan `plan(context=ContextBundle, catalog=..., domain_profile=..., repair_feedback=..., frozen_task=...)` y devuelven `PlannerCallResult`. El adaptador devuelve el error de infraestructura de forma explícita; no sustituye el proveedor por propuestas grabadas. Para la integración habitual basta con pasar `model=`; no se necesita escribir un adaptador.

`formal_comparison=satisfied` habla del objetivo estructurado y de sus premisas explícitas. No significa que el núcleo haya certificado la traducción del texto original. La comparación no certifica la equivalencia semántica con la afirmación original.

HTTP 200 significa que se procesó la petición; puede contener objetivos indeterminados o rechazados. HTTP 422 señala entrada inválida. HTTP 503 señala infraestructura o configuración inaccesible y conserva un recibo parcial cuando el contexto pudo construirse. El fallo no se convierte en una afirmación sobre el corpus.

## Política explícita de acción

La conducta predeterminada es informativa. Para evaluar una autorización desde Python:

```python
from claimlayer.semantic.models import digest
from claimlayer.v4 import ActionPolicy, decision_is_current
from claimlayer.v4.context import build_context

# La aplicación determina la tarea y operación autorizables.
context = build_context(sources, claim, context=metadata)
operation = {"operation_id": "app:operation", "payload": action_payload}
policy = ActionPolicy(
    policy_id="app-policy", version="1", operation_id=operation["operation_id"],
    payload_digest=digest(action_payload), context_digest=context.content_digest,
    accept_model_interpretation=True, pass_min_grade="B",
)
result = verify_claim(sources, claim, context=metadata, model=model,
                      operation=operation, action_policy=policy)
allowed = result.decision.payload["authorized"] and decision_is_current(
    result.decision, result.context, result.verification, operation, policy
)
```

Este ejemplo de integración no ejecuta una acción externa. La aplicación asume explícitamente la interpretación del modelo y define la política. El núcleo conserva su garantía formal.

En HTTP, el servidor puede leer los mismos campos de política desde un JSON indicado por `DELTORYX_ACTION_POLICY_FILE`. El archivo y las credenciales son configuración de aplicación; nunca entradas controladas por el modelo. Cambiar contexto, payload o versión de política invalida la autorización anterior.

## CLI y ejecutables de evaluación

```bash
export NVIDIA_API_KEY='configurar fuera de los recibos'
deltoryx-gate verify-claim --source 'doc_1=Texto de la fuente' --claim 'Afirmación'
```

La CLI admite también `--sources-json` con metadatos de fuentes. Códigos: 0 para `INFORM` o `PASS`, 2 para `RETRY` y 3 para `ESCALATE`. Un código 0 informativo no autoriza una acción: consultar siempre `decision.payload.authorized` y la política vigente.

Desde la plataforma:

```bash
.venv/bin/python scripts/evaluate_gate_layers.py --suite all
.venv/bin/python scripts/evaluate_gate_layers.py --suite ocr
.venv/bin/python scripts/evaluate_gate_layers.py --suite temporal
.venv/bin/python scripts/live_claims_api_check.py
```

Los tres primeros ejecutan respectivamente 26, 12 y 14 casos reales con NVIDIA y crean directorios nuevos. El cuarto inicia el servidor y realiza cuatro llamadas reales por HTTP. Los resultados de transporte y de contenido se registran por separado.

## Servicio y persistencia

La API se inicia desde la plataforma con `python -m uvicorn api.main:app`. Configurar `NVIDIA_API_KEY` en el entorno del servicio. Opcionalmente: `DELTORYX_PLANNER_MODEL`, `DELTORYX_PLANNER_BASE_URL`, `DELTORYX_PLANNER_TIMEOUT` y `DELTORYX_PLANNER_MAX_TOKENS`. El proveedor predeterminado es NVIDIA y no hay fallback a otro motor.

El almacén SQLite acepta solo `gate-layers-1`, en la tabla `gate_receipts`. `CLAIMLAYER_AUDIT_DB` configura su ruta; por defecto es `.build/gate_audit.db`. Los artefactos originales no se reescriben al indexarlos. `GET /api/v1/v4/receipts`, `GET /api/v1/v4/receipts/{receipt_id}` y `GET /api/v1/v4/telemetry` consultan únicamente este contrato. `GET /health` declara la versión del servicio y del contrato.

Los recibos incluyen fuentes y respuestas crudas del modelo. No se guardan credenciales ni cabeceras de autorización. Las evaluaciones reales crean un directorio nuevo por ejecución; las etiquetas de referencia de `evaluation/inputs.json` solo se consultan después de la inferencia.
