"""Traceable reconstruction support: model-proposed operations, premise fidelity and reliability grades.

Evidence Intelligence direction: an AI produces a result; Deltoryx asks the planner to rebuild it
from the sources with explicit operations. The planner MAY propose operations it needs. Deltoryx
never executes model-written code: a proposed operation is either

* an expression in a closed, data-only language interpreted here (grade C, re-executed by Deltoryx), or
* a declared-but-unexecutable operation whose result is the value the model asserts (grade D).

The reliability of a conclusion is its weakest link (premises and steps it depends on):

  A  premise value verified literally in its quote + application core operation
  B  application-declared rule
  C  model-proposed operation expressed over core primitives, executed by Deltoryx
  D  model-proposed operation without executable definition (model-asserted result)
  E  premise whose value cannot be checked against its quote (model interpretation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping, Sequence

from ..semantic.models import digest
from .justification import ClaimNode
from .operations import OperationCatalog, OperationOutput, OperationSpec
from .vocabulary import UnitRegistry, UnmappedProposal, label_key

GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E")
GRADE_LABELS: dict[str, str] = {
    "A": "literal_premises_and_core_operations",
    "B": "application_declared_rule",
    "C": "model_composed_operation_executed_by_deltoryx",
    "D": "model_asserted_operation_not_executed",
    "E": "model_interpreted_premise",
}
GRADE_GUARANTEES: dict[str, str] = {
    "A": "Texto fuente comprobado y operaciones del núcleo de Deltoryx.",
    "B": "Regla declarada y versionada por la aplicación.",
    "C": "Composición propuesta por Nemotron, interpretada y re-ejecutada por Deltoryx.",
    "D": "Resultado afirmado por Nemotron mediante una operación propuesta que Deltoryx no puede ejecutar.",
    "E": "Valor de premisa asignado por la interpretación de Nemotron; la cita existe pero no fija ese valor.",
}


def weakest(grades: Sequence[str]) -> str:
    return max(grades, key=GRADES.index) if grades else "A"


def grade_at_least(grade: str, minimum: str) -> bool:
    return GRADES.index(grade) <= GRADES.index(minimum)


@dataclass(frozen=True)
class ReliabilityPolicy:
    """Application policy: minimum grade that may route to PASS (default B)."""

    pass_min_grade: str = "B"

    def __post_init__(self) -> None:
        if self.pass_min_grade not in GRADES:
            raise ValueError(f"pass_min_grade must be one of {GRADES}.")

    def describe(self) -> dict[str, Any]:
        return {"pass_min_grade": self.pass_min_grade}


DEFAULT_RELIABILITY_POLICY = ReliabilityPolicy()


# ---------------------------------------------------------------------------
# Premise fidelity
# ---------------------------------------------------------------------------

def premise_fidelity(value: Any) -> str:
    """'literal' when grounding verified the value against the quote with deterministic
    normalizers (numbers, weekdays, dates, text whose content words occur in the clause);
    'interpreted' otherwise (booleans, empty/structured values) because grounding accepts
    them without checking the quote."""
    if isinstance(value, bool) or value is None:
        return "interpreted"
    if isinstance(value, (int, float, Decimal)):
        return "literal"
    if isinstance(value, str):
        return "literal" if re.search(r"[0-9A-Za-zÀ-ÿ]{2,}|\d", value) else "interpreted"
    return "interpreted"


# ---------------------------------------------------------------------------
# Closed expression language (data, never code)
# ---------------------------------------------------------------------------

MAX_EXPR_DEPTH = 24
MAX_EXPR_NODES = 200
MAX_STR_LEN = 10_000

EXPRESSION_PRIMITIVES: dict[str, dict[str, Any]] = {
    "add": {"arity": "2+", "doc": "suma numérica exacta"},
    "sub": {"arity": 2, "doc": "a - b"},
    "mul": {"arity": "2+", "doc": "producto numérico exacto"},
    "div": {"arity": 2, "doc": "a / b (b != 0)"},
    "neg": {"arity": 1, "doc": "-a"},
    "abs": {"arity": 1, "doc": "|a|"},
    "min": {"arity": "2+", "doc": "mínimo numérico"},
    "max": {"arity": "2+", "doc": "máximo numérico"},
    "eq": {"arity": 2, "doc": "igualdad (números exactos, texto sin mayúsculas/acentos)"},
    "ne": {"arity": 2, "doc": "desigualdad"},
    "lt": {"arity": 2, "doc": "a < b"}, "le": {"arity": 2, "doc": "a <= b"},
    "gt": {"arity": 2, "doc": "a > b"}, "ge": {"arity": 2, "doc": "a >= b"},
    "and": {"arity": "2+", "doc": "conjunción booleana"},
    "or": {"arity": "2+", "doc": "disyunción booleana"},
    "not": {"arity": 1, "doc": "negación booleana"},
    "if": {"arity": 3, "doc": "if(cond, a, b)"},
    "contains": {"arity": 2, "doc": "texto a contiene texto b (sin mayúsculas/acentos)"},
    "starts_with": {"arity": 2, "doc": "texto a empieza por b"},
    "length": {"arity": 1, "doc": "longitud de un texto"},
    "lower": {"arity": 1, "doc": "texto en minúsculas"},
}


class ExpressionError(ValueError):
    pass


def _num(v: Any) -> Decimal:
    if isinstance(v, bool):
        raise ExpressionError("boolean used where a number is required")
    try:
        d = Decimal(str(v).strip().replace(",", "")) if isinstance(v, str) else Decimal(str(v))
    except (InvalidOperation, ValueError) as exc:
        raise ExpressionError(f"not a number: {v!r}") from exc
    if not d.is_finite():
        raise ExpressionError("non-finite number")
    return d


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise ExpressionError(f"not a boolean: {v!r}")
    return v


def _text(v: Any) -> str:
    if not isinstance(v, str):
        raise ExpressionError(f"not a text: {v!r}")
    return v


def _out(d: Any) -> Any:
    if isinstance(d, Decimal):
        if d == d.to_integral_value():
            return int(d)
        return float(d)
    return d


def validate_expression(expr: Any, inputs: Sequence[str]) -> int:
    """Structural validation; returns node count. Raises ExpressionError."""
    count = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal count
        count += 1
        if count > MAX_EXPR_NODES:
            raise ExpressionError("expression too large")
        if depth > MAX_EXPR_DEPTH:
            raise ExpressionError("expression too deep")
        if not isinstance(node, Mapping) or len(node) == 0:
            raise ExpressionError(f"node must be a non-empty object: {node!r}")
        keys = set(node)
        if keys == {"input"}:
            if node["input"] not in inputs:
                raise ExpressionError(f"unknown input {node['input']!r}")
            return
        if keys == {"const"}:
            c = node["const"]
            if not isinstance(c, (bool, int, float, str)) or (isinstance(c, str) and len(c) > MAX_STR_LEN):
                raise ExpressionError("const must be a scalar")
            return
        if keys == {"op", "args"}:
            op = node["op"]
            spec = EXPRESSION_PRIMITIVES.get(op) if isinstance(op, str) else None
            if spec is None:
                raise ExpressionError(f"primitive not allowed: {op!r}")
            args = node["args"]
            if not isinstance(args, list):
                raise ExpressionError("args must be a list")
            ar = spec["arity"]
            if (isinstance(ar, int) and len(args) != ar) or (ar == "2+" and len(args) < 2):
                raise ExpressionError(f"wrong arity for {op}")
            for a in args:
                walk(a, depth + 1)
            return
        raise ExpressionError(f"unknown node shape {sorted(keys)}")

    walk(expr, 0)
    return count


def evaluate_expression(expr: Mapping[str, Any], env: Mapping[str, Any]) -> Any:
    """Interpret a validated expression over input values. Pure; no I/O; bounded."""

    def ev(node: Mapping[str, Any]) -> Any:
        if "input" in node:
            return env[node["input"]]
        if "const" in node:
            return node["const"]
        op, args = node["op"], node["args"]
        if op == "if":
            return ev(args[1]) if _bool(ev(args[0])) else ev(args[2])
        if op == "and":
            return all(_bool(ev(a)) for a in args)
        if op == "or":
            return any(_bool(ev(a)) for a in args)
        vals = [ev(a) for a in args]
        if op == "add":
            return sum((_num(v) for v in vals), Decimal(0))
        if op == "sub":
            return _num(vals[0]) - _num(vals[1])
        if op == "mul":
            r = Decimal(1)
            for v in vals:
                r *= _num(v)
            return r
        if op == "div":
            d = _num(vals[1])
            if d == 0:
                raise ExpressionError("division by zero")
            return _num(vals[0]) / d
        if op == "neg":
            return -_num(vals[0])
        if op == "abs":
            return abs(_num(vals[0]))
        if op == "min":
            return min(_num(v) for v in vals)
        if op == "max":
            return max(_num(v) for v in vals)
        if op in ("eq", "ne"):
            a, b = vals
            if isinstance(a, bool) or isinstance(b, bool):
                same = isinstance(a, bool) and isinstance(b, bool) and a is b
            elif isinstance(a, str) and isinstance(b, str):
                same = label_key(a) == label_key(b)
            else:
                same = _num(a) == _num(b)
            return same if op == "eq" else not same
        if op in ("lt", "le", "gt", "ge"):
            a, b = _num(vals[0]), _num(vals[1])
            return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[op]
        if op == "not":
            return not _bool(vals[0])
        if op == "contains":
            return label_key(_text(vals[1])) in label_key(_text(vals[0]))
        if op == "starts_with":
            return label_key(_text(vals[0])).startswith(label_key(_text(vals[1])))
        if op == "length":
            return len(_text(vals[0]))
        if op == "lower":
            return _text(vals[0]).lower()
        raise ExpressionError(f"primitive not allowed: {op!r}")

    return _out(ev(expr))


# ---------------------------------------------------------------------------
# Proposed operations
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,63}$")
_INPUT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
PROPOSED_FAMILY = "explicit_logic"


@dataclass(frozen=True)
class ProposedOperation:
    name: str
    description: str
    inputs: tuple[str, ...]
    expression: dict[str, Any] | None
    output_type: str
    output_unit: str | None
    grade: str  # 'C' (executable) or 'D' (model-asserted)
    node_count: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "description": self.description, "inputs": list(self.inputs),
            "expression": self.expression, "output_type": self.output_type, "output_unit": self.output_unit,
            "grade": self.grade, "grade_label": GRADE_LABELS[self.grade],
            "definition_digest": digest({"name": self.name, "inputs": list(self.inputs), "expression": self.expression,
                                         "output_unit": self.output_unit}),
        }


def parse_proposed_operations(
    raw: Any,
    *,
    core_catalog: OperationCatalog,
    registry: UnitRegistry,
) -> tuple[list[ProposedOperation], list[dict[str, Any]]]:
    """Validate planner-proposed operations. Returns (operations, defects)."""
    ops: list[ProposedOperation] = []
    defects: list[dict[str, Any]] = []
    if raw in (None, []):
        return ops, defects
    if not isinstance(raw, list):
        return ops, [{"code": "MALFORMED_PROPOSED_OPERATIONS", "message": "proposed_operations must be a list."}]
    seen: set[str] = set()
    for idx, po in enumerate(raw):
        if not isinstance(po, Mapping):
            defects.append({"code": "MALFORMED_PROPOSED_OPERATION", "message": f"proposed_operations[{idx}] must be an object."})
            continue
        name = str(po.get("name") or "").strip()
        if not _NAME_RE.match(name):
            defects.append({"code": "INVALID_PROPOSED_OPERATION_NAME", "message": f"Invalid operation name {name!r}."})
            continue
        if core_catalog.has(name) or name in seen:
            defects.append({"code": "PROPOSED_OPERATION_SHADOWS_EXISTING", "message": f"'{name}' already exists; core operations cannot be redefined."})
            continue
        inputs = po.get("inputs") or []
        if not isinstance(inputs, list) or not all(isinstance(i, str) and _INPUT_RE.match(i) for i in inputs) or len(set(inputs)) != len(inputs):
            defects.append({"code": "INVALID_PROPOSED_OPERATION_INPUTS", "message": f"'{name}': inputs must be distinct identifiers."})
            continue
        unit = po.get("output_unit")
        if unit not in (None, "", "inherit"):
            try:
                registry.lookup(str(unit))
            except UnmappedProposal:
                defects.append({"code": "UNKNOWN_UNIT:" + str(unit), "message": f"'{name}': output_unit {unit!r} not declared."})
                continue
        expr = po.get("expression")
        node_count = 0
        if expr is not None:
            try:
                node_count = validate_expression(expr, inputs)
            except ExpressionError as exc:
                defects.append({"code": "INVALID_PROPOSED_EXPRESSION", "message": f"'{name}': {exc}"})
                continue
        seen.add(name)
        ops.append(ProposedOperation(
            name=name, description=str(po.get("description") or "")[:1000], inputs=tuple(inputs),
            expression=dict(expr) if isinstance(expr, Mapping) else None,
            output_type=str(po.get("output_type") or "value"), output_unit=(None if unit in (None, "") else str(unit)),
            grade="C" if expr is not None else "D", node_count=node_count,
        ))
    return ops, defects


def _make_executor(op: ProposedOperation):
    def _exec(premises: Sequence[ClaimNode], parameters: Mapping[str, Any], profile: Any) -> OperationOutput:
        if op.expression is None:
            if "__asserted_value" not in parameters or parameters.get("__asserted_value") is None:
                return OperationOutput(ok=False, error_code=f"MODEL_ASSERTED_OPERATION_WITHOUT_VALUE:{op.name}",
                                       error_kind="undeclared_rule")
            return OperationOutput(
                ok=True, value=parameters["__asserted_value"], unit=parameters.get("__asserted_unit"),
                details={"proposed_operation": op.name, "grade": "D", "model_asserted": True},
                assumptions=(f"model_asserted_operation:{op.name}",),
            )
        if len(premises) != len(op.inputs):
            return OperationOutput(ok=False, error_code=f"PROPOSED_OPERATION_ARITY:{op.name}", error_kind="formal_error")
        env = {name: p.value for name, p in zip(op.inputs, premises)}
        try:
            value = evaluate_expression(op.expression, env)
        except ExpressionError as exc:
            return OperationOutput(ok=False, error_code=f"PROPOSED_EXPRESSION_ERROR:{exc}", error_kind="formal_error")
        unit = op.output_unit
        if unit == "inherit":
            units = {p.unit for p in premises}
            unit = units.pop() if len(units) == 1 else None
        return OperationOutput(ok=True, value=value, unit=unit,
                               details={"proposed_operation": op.name, "grade": "C", "inputs": env},
                               assumptions=(f"model_composed_operation:{op.name}",))
    return _exec


def proposed_operation_specs(ops: Sequence[ProposedOperation]) -> list[OperationSpec]:
    return [
        OperationSpec(
            name=op.name, version="proposed-" + op.describe()["definition_digest"][:12], family=PROPOSED_FAMILY,
            description=f"[propuesta por el planificador, grado {op.grade}] {op.description}",
            inputs=tuple({"name": i} for i in op.inputs), output_type=op.output_type, executor=_make_executor(op),
        )
        for op in ops
    ]


# ---------------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------------

QUOTE_CONSUMING_OPS = frozenset({"count_lines"})

_STOP = frozenset("""a al and as at by con de del el en es for from in is la las los of on or para por que se the to un una y this that""".split())


def _fold(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold()


def content_tokens(text: Any) -> set[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9_.]*", _fold(str(text or "")))
    return {t.strip(".") for t in toks if len(t.strip(".")) >= 2 and t.strip(".") not in _STOP}


def _ordered_tokens(text: Any) -> list[str]:
    toks = re.findall(r"[a-z0-9][a-z0-9_.]*", _fold(str(text or "")))
    return [x.strip(".") for x in toks if len(x.strip(".")) >= 2 and x.strip(".") not in _STOP]


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def is_claim_restatement(quote: str, claim: str, *, min_tokens: int = 4, threshold: float = 0.8) -> bool:
    """A quoted span that reproduces the claim's wording in order (>= threshold of the claim's
    content tokens as an ordered subsequence) accredits only that someone asserted it, not the
    fact. Ordered matching avoids flagging code that merely shares vocabulary with the claim."""
    ct = _ordered_tokens(claim)
    if len(ct) < min_tokens:
        return False
    qt = _ordered_tokens(quote)[:4000]
    # A quote carrying a different number/version is a potential counterexample, not a restatement.
    digit_tokens = {x for x in ct if any(ch.isdigit() for ch in x)}
    if digit_tokens and not digit_tokens <= set(qt):
        return False
    return _lcs_len(ct, qt) / len(ct) >= threshold


def step_grade(operation: str, proposed: Mapping[str, ProposedOperation]) -> str:
    if operation in proposed:
        return proposed[operation].grade
    if operation == "rule_application":
        return "B"
    return "A"


def compute_reliability(
    *,
    conclusion_id: str,
    premises: Mapping[str, Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    proposed: Mapping[str, ProposedOperation],
    policy: ReliabilityPolicy,
) -> dict[str, Any]:
    by_conc = {str(s["conclusion_id"]): s for s in steps}
    elements: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        if node_id in premises:
            consumers = {str(s["operation"]) for s in steps if node_id in (s.get("premises") or ())}
            fid = ("literal_quote" if consumers and consumers <= QUOTE_CONSUMING_OPS
                   else premise_fidelity(premises[node_id].get("value")))
            elements.append({"kind": "premise", "id": node_id, "fidelity": fid, "grade": "A" if fid in ("literal", "literal_quote") else "E",
                             "value": premises[node_id].get("value"),
                             "document_id": (premises[node_id].get("evidence") or {}).get("document_id")})
            return
        st = by_conc.get(node_id)
        if st is None:
            return
        op = str(st["operation"])
        elements.append({"kind": "step", "id": str(st.get("step_id")), "operation": op, "grade": step_grade(op, proposed)})
        for p in st.get("premises") or ():
            visit(str(p))

    visit(conclusion_id)
    grade = weakest([e["grade"] for e in elements])
    weakest_links = [e for e in elements if e["grade"] == grade] if grade != "A" else []
    return {
        "grade": grade,
        "grade_label": GRADE_LABELS[grade],
        "guarantee": GRADE_GUARANTEES[grade],
        "weakest_links": weakest_links,
        "elements": elements,
        "proposed_operations": [op.describe() for op in proposed.values()],
        "policy": policy.describe(),
        "pass_allowed_by_grade": grade_at_least(grade, policy.pass_min_grade),
    }
