"""Frozen evaluation of the `code_change` v2.0 vocabulary with Gate v4 + Nemotron.

Separates literally checkable code claims (files, identifiers, dependencies, tests,
scalar counts) from behavioral/feature claims routed explicitly to `describes_behavior`
(status `unverifiable`, excluded from unsupported rate and INCONSISTENT triggers).

Security & method rules:
- Checks the frozen CODE_CHANGE_VOCAB_V2 fingerprint at startup.
- Never stores or logs API keys, PR bodies, or diff text snippets.
- Uses official `claimlayer.v4.triage.verify_draft()` directly (with inherited
  `closed_entities=True` via `autovocab.with_entities`).
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
import re
import sys
import threading
import time

csv.field_size_limit(10**9)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parse  # noqa: E402
from claimlayer.semantic.models import canonical  # noqa: E402
from claimlayer.v4 import EvidenceLayer, EvidenceRef, SourceDocument  # noqa: E402
from claimlayer.v4.reader import LLMProposalReader  # noqa: E402
from claimlayer.v4.triage import verify_draft  # noqa: E402
from claimlayer.v4.vocabulary import FieldSpec, PredicateSpec, Vocabulary  # noqa: E402

ENTITY = "pull request"
EXPECTED_VOCAB_FINGERPRINT = "620ebd71c7dda399cb8b90e5ac038a2a2cf510782a86202b79b75c8d84900d2c"

CODE_CHANGE_VOCAB_V2 = Vocabulary(
    name="code_change",
    version="2.0",
    closed_entities=True,
    entities=((ENTITY, ("PR", "this PR", "the PR", "the pull request", "this pull request")),),
    require_explicit_condition_status=False,
    predicates=(
        PredicateSpec(
            "changes_file",
            FieldSpec(
                "string",
                description=(
                    'Exact repository file path or filename changed by the pull request '
                    '(e.g. "src/auth.py" or "package.json"). Never use feature phrases or class names. '
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "adds_identifier",
            FieldSpec(
                "string",
                description=(
                    "Exact syntactic code or config identifier added in the diff "
                    "(function, class, method, type, constant, config key, or CLI flag; "
                    'never an English feature phrase). Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "removes_identifier",
            FieldSpec(
                "string",
                description=(
                    "Exact syntactic code or config identifier removed in the diff "
                    "(function, class, method, type, constant, config key, or CLI flag; "
                    'never an English feature phrase). Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "modifies_identifier",
            FieldSpec(
                "string",
                description=(
                    "Exact existing syntactic code or config identifier modified in the diff "
                    '(never an English feature phrase). Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "changes_dependency",
            FieldSpec.record(
                description=(
                    "Package dependency added, removed, or version-updated in a manifest or lockfile. "
                    'Omit "version" if no version is stated (never use null). Always set entity to "pull request".'
                ),
                required=("package", "action"),
                package=FieldSpec("string"),
                version=FieldSpec("string"),
                action=FieldSpec("enum", enum=("add", "remove", "update")),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "adds_test",
            FieldSpec(
                "string",
                description=(
                    "Exact test file path or test function/case identifier added in the diff. "
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "removes_test",
            FieldSpec(
                "string",
                description=(
                    "Exact test file path or test function/case identifier removed in the diff. "
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "files_changed_count",
            FieldSpec(
                "number",
                description=(
                    "Exact integer number of files changed if explicitly stated in the PR text. "
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="single",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "lines_added_count",
            FieldSpec(
                "number",
                description=(
                    "Exact integer number of lines added if explicitly stated in the PR text. "
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="single",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "lines_deleted_count",
            FieldSpec(
                "number",
                description=(
                    "Exact integer number of lines deleted if explicitly stated in the PR text. "
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="single",
            assertion_kinds=("fact",),
        ),
        PredicateSpec(
            "describes_behavior",
            FieldSpec(
                "string",
                description=(
                    "High-level feature, bugfix, refactoring, or behavioral description that is NOT a literal "
                    'file path, identifier, dependency, test name, or count (e.g. "adds token refresh mechanism"). '
                    'Always set entity to "pull request".'
                ),
            ),
            cardinality="multi",
            assertion_kinds=("fact", "proposal"),
        ),
    ),
)

assert CODE_CHANGE_VOCAB_V2.fingerprint == EXPECTED_VOCAB_FINGERPRINT, (
    f"Vocabulary fingerprint drift: {CODE_CHANGE_VOCAB_V2.fingerprint} != {EXPECTED_VOCAB_FINGERPRINT}"
)

UNVERIFIABLE_PREDICATES = frozenset({"describes_behavior"})

_COMMENT_LINE_RE = re.compile(r"^\s*(?://|/\*|\*|<!--|#\s)")
_DEF_RE = re.compile(
    r"\b(?:export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|type|interface|enum)|"
    r"def|class|function|fn|func(?:\s*\([^)]*\))?|struct|interface|enum|type|const|let|var|module|trait|impl)\s+"
    r"([A-Za-z_][\w]*)"
)
_TYPED_DECL_RE = re.compile(
    r"^\s*(?:public|private|protected|internal|static|async|override|virtual|final|open|suspend)\s+"
    r"(?:(?:public|private|protected|internal|static|async|override|virtual|final|open|suspend)\s+)*"
    r"(?:[\w<>?\[\],.*&]+\s+)*([A-Za-z_][\w]*)\s*\("
)
_METHOD_HEAD_RE = re.compile(
    r"^\s*(?:async\s+)?([A-Za-z_][\w]*)\s*\([^)]*\)\s*(?::\s*[\w<>|\[\]?., ]+)?\s*\{"
)
_KEY_RE = re.compile(r'^\s*(?:-\s+)?"?([A-Za-z_][\w.-]{1,60})"?\s*[:=]')
_CONST_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{2,})\s*[=:,]")
_FLAG_RE = re.compile(r"(?<![A-Za-z0-9-])(--[a-z][a-z0-9-]{1,40})\b")
_QUALIFIED_RE = re.compile(r"\b([A-Za-z_][\w]*(?:(?:\.|::|->)[A-Za-z_][\w]*)+)\b")
_HUNK_RE = re.compile(r"^@@[^@]*@@\s*(.*)$")

_TEST_FILE_RE = re.compile(
    r"(?i)(?:^|/)(?:tests?|specs?|__tests__)/|(?:_test|\.test|\.spec)\.[A-Za-z0-9]+$|"
    r"Tests?\.(?:swift|java|kt|cs|go|py|ts|tsx|js|jsx|rs|rb)$"
)
_TEST_FN_RE = re.compile(r"\b(?:def\s+(test_\w+)|func\s+(Test\w+)|fn\s+(test_\w+))")
_TEST_JS_RE = re.compile(r"\b(?:it|test|describe)\(\s*['\"]([^'\"]{3,80})['\"]")

_DEP_JSON_RE = re.compile(r'^\s*"([@A-Za-z0-9_./-]{2,80})"\s*:\s*"([~^>=<*!v]*\d+[^"\s]{0,30})"')
_DEP_PIP_RE = re.compile(r"^\s*([A-Za-z0-9_.-]{2,60})(?:\[[^\]]+\])?\s*(?:==|>=|<=|~=|!=)\s*([0-9][A-Za-z0-9.*+-]{0,30})")
_DEP_TOML_RE = re.compile(
    r'^\s*([A-Za-z0-9_-]{2,60})\s*=\s*(?:"([~^>=<*]*\d+[^"\s]{0,30})"|\{[^}]*version\s*=\s*"([^"]+)")'
)
_DEP_GO_RE = re.compile(r"^\s*([A-Za-z0-9.-]+\.[A-Za-z0-9./_-]+)\s+(v\d+\.\d+\.\d+[^\s]*)")
_DEP_GEM_RE = re.compile(r"""^\s*gem\s+['"]([A-Za-z0-9_.-]+)['"]\s*(?:,\s*['"]([^'"]+)['"])?""")

_STOPWORDS = frozenset({
    "if", "else", "elif", "for", "while", "return", "switch", "case", "default",
    "break", "continue", "try", "catch", "finally", "throw", "throws", "new",
    "import", "from", "export", "true", "false", "null", "none", "nil",
    "this", "self", "super", "void", "int", "str", "bool", "float", "string",
    "number", "boolean", "object", "any", "the", "and", "or", "not", "in",
    "is", "to", "of", "with", "as", "by", "on", "at", "be", "it", "us", "me",
    "that", "are", "was", "were", "has", "have", "can", "will", "do", "type",
    "name", "value", "data", "result", "results", "error", "status", "note",
    "summary", "description", "details", "example", "http", "https",
})


def _extract_line_identifiers(line: str) -> list[str]:
    if _COMMENT_LINE_RE.match(line):
        return []
    found: list[str] = []
    found.extend(_DEF_RE.findall(line))
    found.extend(_TYPED_DECL_RE.findall(line))
    found.extend(_METHOD_HEAD_RE.findall(line))
    if m := _KEY_RE.match(line):
        found.append(m.group(1))
    if m := _CONST_RE.match(line):
        found.append(m.group(1))
    found.extend(_FLAG_RE.findall(line))
    for q in _QUALIFIED_RE.findall(line):
        found.append(q)
        last = re.split(r"\.|::|->", q)[-1]
        if len(last) >= 3:
            found.append(last)
    out: list[str] = []
    for sym in found:
        if sym.lower() in _STOPWORDS or len(sym) < 2:
            continue
        out.append(sym)
    return out


def _extract_dep(line: str) -> tuple[str, str | None] | None:
    if m := _DEP_JSON_RE.match(line):
        pkg, ver = m.group(1), m.group(2)
        if pkg.lower() not in {"version", "node", "npm", "python"}:
            return pkg, ver
    if m := _DEP_PIP_RE.match(line):
        return m.group(1), m.group(2)
    if m := _DEP_TOML_RE.match(line):
        return m.group(1), (m.group(2) or m.group(3))
    if m := _DEP_GO_RE.match(line):
        return m.group(1), m.group(2)
    if m := _DEP_GEM_RE.match(line):
        return m.group(1), m.group(2)
    return None


def _extract_tests(line: str) -> list[str]:
    out: list[str] = []
    for m in _TEST_FN_RE.finditer(line):
        tok = next(g for g in m.groups() if g)
        out.append(tok)
    for m in _TEST_JS_RE.finditer(line):
        out.append(m.group(1))
    return out


def _path_variants(path_str: str) -> list[str]:
    norm = path_str.strip().lstrip("./")
    if not norm:
        return []
    parts = norm.split("/")
    return ["/".join(parts[i:]) for i in range(len(parts))]


def diff_facts_v2(
    pr: dict,
    *,
    max_syms: int | None = None,
    max_tests: int | None = None,
) -> dict:
    patch = pr["patch"] or ""
    files = pr["files"] or []
    added, removed, touched = [], [], []
    dep_plus, dep_minus = {}, {}
    tests_plus, tests_minus = [], []

    for raw in patch.splitlines():
        if raw.startswith("+++") or raw.startswith("---"):
            continue
        if raw.startswith("+"):
            line = raw[1:]
            added.extend(_extract_line_identifiers(line))
            tests_plus.extend(_extract_tests(line))
            if dep := _extract_dep(line):
                dep_plus[dep[0]] = dep[1]
        elif raw.startswith("-"):
            line = raw[1:]
            removed.extend(_extract_line_identifiers(line))
            tests_minus.extend(_extract_tests(line))
            if dep := _extract_dep(line):
                dep_minus[dep[0]] = dep[1]
        elif m := _HUNK_RE.match(raw):
            if m.group(1):
                touched.extend(_extract_line_identifiers(m.group(1)))
        elif raw.startswith(" "):
            touched.extend(_extract_line_identifiers(raw[1:]))

    a_uniq = list(dict.fromkeys(added))
    r_uniq = list(dict.fromkeys(removed))
    a_set, r_set = set(a_uniq), set(r_uniq)
    mod_syms = list(
        dict.fromkeys(
            [x for x in a_uniq if x in r_set]
            + [x for x in dict.fromkeys(touched) if x not in a_set and x not in r_set]
        )
    )
    add_syms = [x for x in a_uniq if x not in r_set]
    rem_syms = [x for x in r_uniq if x not in a_set]

    file_vals: list[str] = []
    test_files: list[str] = []
    for f in files:
        for v in _path_variants(f):
            file_vals.append(v)
        if _TEST_FILE_RE.search(f):
            for v in _path_variants(f):
                test_files.append(v)
    file_vals = list(dict.fromkeys(file_vals))

    tp_uniq = list(dict.fromkeys(tests_plus))
    tm_uniq = list(dict.fromkeys(tests_minus))
    tp_set, tm_set = set(tp_uniq), set(tm_uniq)
    add_tests = list(dict.fromkeys(test_files + [t for t in tp_uniq if t not in tm_set]))
    rem_tests = list(dict.fromkeys([t for t in tm_uniq if t not in tp_set]))

    deps: list[dict] = []
    for pkg, ver in dep_plus.items():
        if pkg in dep_minus:
            if dep_minus[pkg] != ver:
                deps.append({"package": pkg, **({"version": ver} if ver else {}), "action": "update"})
                deps.append({"package": pkg, "action": "update"})
        else:
            deps.append({"package": pkg, **({"version": ver} if ver else {}), "action": "add"})
            if ver:
                deps.append({"package": pkg, "action": "add"})
    for pkg, ver in dep_minus.items():
        if pkg not in dep_plus:
            deps.append({"package": pkg, **({"version": ver} if ver else {}), "action": "remove"})
            if ver:
                deps.append({"package": pkg, "action": "remove"})

    truncated_fields: dict[str, dict[str, int]] = {}

    def _apply_cap(name: str, seq: list[str], cap: int | None) -> list[str]:
        if cap is None:
            return seq
        if cap < 0:
            raise ValueError(f"Truncation cap for {name} must be non-negative, got {cap}")
        if len(seq) > cap:
            truncated_fields[name] = {"kept": cap, "total": len(seq)}
            return seq[:cap]
        return seq

    add_syms_out = _apply_cap("add_syms", add_syms, max_syms)
    rem_syms_out = _apply_cap("rem_syms", rem_syms, max_syms)
    mod_syms_out = _apply_cap("mod_syms", mod_syms, max_syms)
    add_tests_out = _apply_cap("add_tests", add_tests, max_tests)
    rem_tests_out = _apply_cap("rem_tests", rem_tests, max_tests)

    return {
        "files_raw": list(files),
        "file_vals": file_vals,
        "add_syms": add_syms_out,
        "rem_syms": rem_syms_out,
        "mod_syms": mod_syms_out,
        "deps": deps,
        "add_tests": add_tests_out,
        "rem_tests": rem_tests_out,
        "files_count": len(files),
        "adds_count": pr.get("adds"),
        "dels_count": pr.get("dels"),
        "truncated": bool(truncated_fields),
        "truncated_fields": truncated_fields,
    }


def build_ledger_v2(
    pr: dict,
    *,
    max_syms: int | None = None,
    max_tests: int | None = None,
):
    facts = diff_facts_v2(pr, max_syms=max_syms, max_tests=max_tests)
    entries: list[tuple[str, object, str]] = []
    seen_keys: set[tuple[str, str]] = set()

    def add_entry(pred: str, val: object, sentence: str) -> None:
        ck = (pred, canonical(val))
        if ck in seen_keys:
            return
        seen_keys.add(ck)
        entries.append((pred, val, sentence))

    for fv in facts["file_vals"]:
        add_entry("changes_file", fv, f"The {ENTITY} changes file {fv} .")
    for pred, vals, verb in (
        ("adds_identifier", facts["add_syms"], "adds identifier"),
        ("removes_identifier", facts["rem_syms"], "removes identifier"),
        ("modifies_identifier", facts["mod_syms"], "modifies identifier"),
        ("adds_test", facts["add_tests"], "adds test"),
        ("removes_test", facts["rem_tests"], "removes test"),
    ):
        for v in vals:
            add_entry(pred, v, f"The {ENTITY} {verb} {v} .")
    for dep in facts["deps"]:
        ver_part = f" version {dep['version']}" if "version" in dep else ""
        add_entry(
            "changes_dependency",
            dep,
            f"The {ENTITY} changes dependency {dep['package']}{ver_part} action {dep['action']} .",
        )
    add_entry("files_changed_count", facts["files_count"], f"The {ENTITY} files changed count is {facts['files_count']} total.")
    if facts["adds_count"] is not None:
        add_entry("lines_added_count", facts["adds_count"], f"The {ENTITY} lines added count is {facts['adds_count']} total.")
    if facts["dels_count"] is not None:
        add_entry("lines_deleted_count", facts["dels_count"], f"The {ENTITY} lines deleted count is {facts['dels_count']} total.")

    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for _, _, sentence in entries:
        text_parts.append(sentence)
        spans.append((cursor, cursor + len(sentence)))
        cursor += len(sentence) + 1
    text = "\n".join(text_parts) or f"The {ENTITY} changes no files."

    doc = SourceDocument(document_id=f"diff_{pr['pr_id']}", text=text, authority_rank=100, weight=1.0)
    layer = EvidenceLayer(None, vocabulary=CODE_CHANGE_VOCAB_V2)
    layer.ingest(doc)
    items = [
        {
            "proposal": {"entity": ENTITY, "predicate": pred, "value": val, "assertion_kind": "fact"},
            "evidence": [EvidenceRef(doc.document_id, doc.revision, st, en, sentence)],
        }
        for (pred, val, sentence), (st, en) in zip(entries, spans)
    ]
    adm = layer.ingest_units(items, attestor="app") if items else {"units_admitted": 0, "rejection_reasons": {}}
    summary_facts = {
        "files": facts["files_count"],
        "added_ids": len(facts["add_syms"]),
        "removed_ids": len(facts["rem_syms"]),
        "modified_ids": len(facts["mod_syms"]),
        "deps": len(facts["deps"]),
        "added_tests": len(facts["add_tests"]),
        "truncated": facts["truncated"],
        "truncated_fields": facts["truncated_fields"],
    }
    return layer, adm, summary_facts


def verdict_v2(rep: dict, facts: dict):
    dr = rep.get("draft_reading") or {}
    if dr.get("read_errors"):
        return "ERROR", ["read_error"], {}, {}, []

    by_uid: dict[str, dict] = {}
    for seg in rep["segments"]:
        for c in seg["claims"]:
            by_uid.setdefault(c["unit_id"], c)
    for c in rep.get("unlocated_claims", []):
        by_uid.setdefault(c["unit_id"], c)

    ver_counts = collections.Counter()
    pred_status = collections.Counter()
    unverifiable_count = 0
    unit_statuses: list[tuple[str, str]] = []

    for uid in sorted(by_uid):
        c = by_uid[uid]
        pred = (c.get("proposition") or {}).get("predicate", "")
        if pred in UNVERIFIABLE_PREDICATES:
            unverifiable_count += 1
            pred_status[f"{pred}:unverifiable"] += 1
            unit_statuses.append((uid, f"{pred}:unverifiable"))
        else:
            st = c["status"]
            ver_counts[st] += 1
            pred_status[f"{pred}:{st}"] += 1
            unit_statuses.append((uid, f"{pred}:{st}"))

    supported = ver_counts.get("supported", 0)
    contradicted = ver_counts.get("contradicted", 0) + ver_counts.get("unresolved_conflict", 0)
    escalated = ver_counts.get("error_escalated", 0)
    unsup = sum(v for k, v in ver_counts.items() if k != "supported")
    tot_ver = supported + unsup

    reasons = []
    if facts.get("truncated"):
        reasons.append("diff_facts_truncated")
    if contradicted >= 1:
        reasons.append(f"contradicted {contradicted}")
    if escalated >= 1:
        reasons.append(f"error_escalated {escalated}")
    if unsup >= 2 and tot_ver > 0 and unsup / tot_ver >= 0.5:
        reasons.append(f"unsupported {unsup}/{tot_ver}")
    if facts["files"] == 0 and (tot_ver + unverifiable_count) > 0:
        reasons.append("empty_diff")

    all_counts = dict(sorted(ver_counts.items()))
    if unverifiable_count:
        all_counts["unverifiable"] = unverifiable_count
    if facts.get("truncated"):
        verdict = "ESCALATE"
    elif reasons:
        verdict = "INCONSISTENT"
    else:
        verdict = "CONSISTENT"
    return (
        verdict,
        reasons,
        all_counts,
        dict(sorted(pred_status.items())),
        unit_statuses,
    )


class CachedModelV2:
    def __init__(self, model, path: str):
        self.model = model
        self.path = path
        self.cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._identity = model.identity if model else {"provider": "replay"}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    self.cache[r["key"]] = r["result"]
                    self._identity = r.get("identity", self._identity)

    @property
    def identity(self):
        return self._identity

    def _make_key(self, system: str, payload: dict) -> str:
        ident = self.model.identity if self.model is not None else self._identity
        blob = json.dumps({"identity": ident, "system": system, "payload": payload}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, system, payload):
        key = self._make_key(system, payload)
        with self._lock:
            if key in self.cache:
                return self.cache[key]
        if self.model is None:
            raise KeyError("replay: missing call " + key[:12])
        res = self.model.complete(system, payload)
        with self._lock:
            if key not in self.cache:
                self.cache[key] = res
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {"key": key, "identity": self.model.identity, "result": res},
                                ensure_ascii=False,
                                default=str,
                            )
                            + "\n"
                        )
        return res


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def compute_metrics(rows, pred_fn, pos_fn) -> dict:
    tp = sum(1 for r in rows if pos_fn(r) and pred_fn(r))
    fp = sum(1 for r in rows if not pos_fn(r) and pred_fn(r))
    fn = sum(1 for r in rows if pos_fn(r) and not pred_fn(r))
    tn = len(rows) - tp - fp - fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "n": len(rows),
        "pos": tp + fn,
        "neg": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "precision_ci95": wilson(tp, tp + fp),
        "recall": round(rec, 4),
        "recall_ci95": wilson(tp, tp + fn),
        "f1": round(f1, 4),
        "fp_rate": round(fpr, 4),
        "fp_rate_ci95": wilson(fp, fp + tn),
    }


def evaluate_pr(
    row_meta: dict,
    pr: dict,
    reader: LLMProposalReader,
    *,
    max_syms: int | None = None,
    max_tests: int | None = None,
) -> dict:
    ledger, adm, facts = build_ledger_v2(pr, max_syms=max_syms, max_tests=max_tests)
    draft = (pr["title"] + "\n\n" + pr["body"]).strip()
    unit_statuses: list[tuple[str, str]] = []
    try:
        rep = verify_draft(ledger, draft, reader=reader, draft_id=f"pr_{pr['pr_id']}", guard=False)
        v, reasons, st, pred_st, unit_statuses = verdict_v2(rep, facts)
        dr = rep.get("draft_reading") or {}
        rej = dict(sorted((dr.get("rejection_reasons") or {}).items()))
        err = None if v != "ERROR" else str(dr.get("read_errors"))[:300]
    except Exception as exc:
        v, reasons, st, pred_st, rej, err = "ERROR", [], {}, {}, {}, f"{type(exc).__name__}: {exc}"[:300]
    lab = (row_meta["consensus_label"] or row_meta["annotator1_label"]).lower()
    receipt_payload = {
        "pr_id": pr["pr_id"],
        "gate": v,
        "reasons": reasons,
        "claim_status": st,
        "pred_status": pred_st,
        "ledger_units": adm.get("units_admitted"),
        "draft_rejected": rej,
        "unit_statuses": unit_statuses,
        "vocabulary_fingerprint": EXPECTED_VOCAB_FINGERPRINT,
    }
    if facts.get("truncated"):
        receipt_payload["truncated"] = True
        receipt_payload["truncated_fields"] = facts["truncated_fields"]
    receipt_sha256 = hashlib.sha256(
        json.dumps(receipt_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "pr_id": pr["pr_id"],
        "agent": row_meta["agent"],
        "url": pr["url"],
        "label": lab,
        "binary_label": row_meta["binary_label"],
        "paper_heuristic": row_meta["predicted_heuristic"],
        "gate": v,
        "reasons": reasons,
        "claim_status": st,
        "pred_status": pred_st,
        "facts": facts,
        "ledger_units": adm.get("units_admitted"),
        "ledger_rejected": dict(sorted((adm.get("rejection_reasons") or {}).items())),
        "draft_rejected": rej,
        "receipt_sha256": receipt_sha256,
        "error": err,
    }


def summarize_block(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["gate"] != "ERROR"]
    pos_any = lambda x: x["label"] != "aligned"
    pos_mis = lambda x: x["binary_label"] == "misaligned"
    g = lambda x: x["gate"] == "INCONSISTENT"
    h = lambda x: x["paper_heuristic"] == "misaligned"
    return {
        "n": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "escalated": sum(1 for r in ok if r["gate"] == "ESCALATE"),
        "truncated": sum(1 for r in ok if (r.get("facts") or {}).get("truncated")),
        "labels": dict(sorted(collections.Counter(r["label"] for r in rows).items())),
        "gate_v4_vocab2": {
            "partial_o_misaligned": compute_metrics(ok, g, pos_any),
            "misaligned": compute_metrics(ok, g, pos_mis),
        },
        "paper_heuristic": {
            "partial_o_misaligned": compute_metrics(ok, h, pos_any),
            "misaligned": compute_metrics(ok, h, pos_mis),
        },
        "claim_status": dict(sorted(sum((collections.Counter(r["claim_status"]) for r in ok), collections.Counter()).items())),
        "pred_status": dict(sorted(sum((collections.Counter(r["pred_status"]) for r in ok), collections.Counter()).items())),
        "draft_rejected": dict(sorted(sum((collections.Counter(r["draft_rejected"]) for r in ok), collections.Counter()).items())),
    }


def run_evaluation(
    *,
    phase: str,
    prmci_dir: str,
    model,
    limit: int = 0,
    workers: int = 1,
    legacy_section: bool = False,
    gate_commit: str = "unknown",
) -> dict:
    assert CODE_CHANGE_VOCAB_V2.fingerprint == EXPECTED_VOCAB_FINGERPRINT
    reader = LLMProposalReader(model=model)
    with open(f"{prmci_dir}/validation/validation_results.csv", encoding="utf-8-sig") as fh:
        val_all = list(csv.DictReader(fh))

    if phase == "phase1":
        target_agents = ["Claude_Code"]
    elif phase == "phase2":
        target_agents = ["OpenAI_Codex", "Copilot", "Devin", "Cursor"]
    else:
        target_agents = ["Claude_Code", "OpenAI_Codex", "Copilot", "Devin", "Cursor"]

    prs_by_id = {}
    agent_by_pid = {}
    for ag in target_agents:
        for p in parse.load(ag, prmci_dir=prmci_dir, legacy_section=legacy_section):
            prs_by_id[p["pr_id"]] = p
            agent_by_pid[p["pr_id"]] = ag

    work_rows = []
    for r in val_all:
        if r["pr_id"] in prs_by_id:
            r_copy = dict(r)
            r_copy["agent"] = agent_by_pid[r["pr_id"]]
            work_rows.append(r_copy)
    if limit:
        work_rows = work_rows[:limit]

    results_by_idx = {}
    if workers <= 1:
        for idx, r in enumerate(work_rows):
            results_by_idx[idx] = evaluate_pr(r, prs_by_id[r["pr_id"]], reader)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(evaluate_pr, r, prs_by_id[r["pr_id"]], reader): idx for idx, r in enumerate(work_rows)}
            for fut in as_completed(futs):
                results_by_idx[futs[fut]] = fut.result()

    ordered_rows = [results_by_idx[i] for i in range(len(work_rows))]
    summary = summarize_block(ordered_rows)
    by_agent = {
        ag: summarize_block([x for x in ordered_rows if x["agent"] == ag])
        for ag in target_agents
        if any(x["agent"] == ag for x in ordered_rows)
    }
    return {
        "gate_commit": gate_commit,
        "legacy_section": legacy_section,
        "vocabulary": {
            "name": CODE_CHANGE_VOCAB_V2.name,
            "version": CODE_CHANGE_VOCAB_V2.version,
            "fingerprint": CODE_CHANGE_VOCAB_V2.fingerprint,
        },
        "reader": getattr(model, "identity", {}),
        "summary": summary,
        "by_agent": by_agent,
        "rows": ordered_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["phase1", "phase2", "both"], default="phase1")
    ap.add_argument("--prmci-dir", default=os.environ.get("PRMCI_DIR", parse.SRC))
    ap.add_argument("--reader", choices=["llm", "replay"], default="replay")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--legacy-section", action="store_true")
    args = ap.parse_args()

    if args.reader == "llm":
        from claimlayer.semantic.adapters import OpenAICompatibleModel

        key = os.environ.get("NVIDIA_API_KEY") or sys.exit("Missing NVIDIA_API_KEY")
        base = OpenAICompatibleModel(
            base_url=args.base_url,
            model=args.model,
            api_key=key,
            timeout=180,
            retries=3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        model = CachedModelV2(base, args.cache)
    else:
        model = CachedModelV2(None, args.cache)

    payload = run_evaluation(
        phase=args.phase,
        prmci_dir=args.prmci_dir,
        model=model,
        limit=args.limit,
        workers=args.workers,
        legacy_section=args.legacy_section,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False, sort_keys=True, default=str)
    print(json.dumps({"summary": payload["summary"]}, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
