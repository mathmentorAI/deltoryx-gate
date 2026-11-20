"""Bounded domain contract and v4 inference verification for coding-agent PR diffs.

Implements:
- Complete, uncapped typed diff ledger with deterministic hunk-to-file attribution
  (exact on AIDev commit records, 3-tier fail-closed reconstruction on PR-MCI patches)
  and fail-closed `truncated -> ESCALATE` handling when explicit caps are exceeded.
- Strict EpistemicProfile (`code-diff-strict@1.0`) admitting only `explicit`,
  `syntactic`, and `deductive` bases.
- Versioned DomainProfile (`code_change_inference@1.0.0`) with 6 anti-laxity
  DeclaredRule / RelationCompositionRule behavioral rules plus 3 I5 structural rules.
- Anti-N2 premise relevance guard with frozen `CODE_STOPWORDS` and repository-token
  exclusion.
- Native `JustificationGraph` + `evaluate_response()` execution (`max_semantic_calls=0`)
  enforcing `verify(..., evidence_profile="code_config")` on all premise quotes and
  strict epistemic monotonicity (`hypothesis` never ascends to `derived_supported`).

Does NOT modify `verify()`, `check_claim()`, `DEFAULT_VERIFIER_REGISTRY`, or existing
verifiers and evidence profiles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

from ...ledger import Ledger
from ...models import Assertion, Source
from ...semantic.models import EvidenceRef, SourceDocument, canonical, digest
from ...verifiers.code_evidence import _has_boundary_aligned_match
from ...verify import verify
from ..domain_profile import (
    DeclaredRule,
    DomainProfile,
    PolicyConfiguration,
    RelationCompositionRule,
)
from ..epistemics import BLOCKED_INFERENCE_TYPES, EXPLICIT, EpistemicProfile
from ..evaluation import VerificationBudget, evaluate_response
from ..justification import (
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    VerificationObligation,
    effective_epistemic_kind,
)
from ..verifiers import (
    DEFAULT_VERIFIER_REGISTRY,
    ExplicitLogicVerifier,
    VerifierContext,
    VerifierRegistry,
    VerifierResult,
)
from ..vocabulary import FieldSpec, PredicateSpec, Vocabulary

# ---------------------------------------------------------------------------
# Frozen domain constants & fingerprints
# ---------------------------------------------------------------------------

PR_ENTITY = "pull request"

CLAIM_STATES: tuple[str, ...] = (
    "literal_supported",
    "literal_unsupported",
    "derived_supported",
    "hypothesis_with_verified_premises",
    "no_premises_in_diff",
    "premise_contradicted",
    "unverifiable",
)

PR_VERDICTS: tuple[str, ...] = ("CONSISTENT", "REVIEW", "INCONSISTENT", "ESCALATE")

BEHAVIOR_OPERATIONS: tuple[str, ...] = (
    "implements",
    "adds_test_for",
    "removes_support_for",
    "configures",
    "updates_dependency",
    "documents",
    "other_behavior",
    "out_of_scope_meta",
)

OPERATION_TO_RULE: dict[str, tuple[str, str]] = {
    "implements": ("code.implements_symbol", "1.0"),
    "adds_test_for": ("code.adds_test_for", "1.0"),
    "removes_support_for": ("code.removes_support_for", "1.0"),
    "configures": ("code.configures", "1.0"),
    "updates_dependency": ("code.updates_dependency", "1.0"),
    "documents": ("code.documents", "1.0"),
}

# Frozen BEFORE touching any AIDev data or running any Phase 3 calibration.
CODE_STOPWORDS: frozenset[str] = frozenset({
    "add", "added", "adding", "adds", "all", "allow", "also", "and", "any", "api", "app",
    "apply", "are", "arg", "args", "argument", "arguments", "attr", "base", "basic", "been",
    "behavior", "behaviour", "better", "bool", "boolean", "bug", "bugs", "build", "bump",
    "but", "call", "called", "calls", "can", "case", "cases", "change", "changed", "changes",
    "changing", "check", "checked", "checks", "chore", "class", "classes", "clean", "cleanup",
    "client", "code", "common", "compat", "compatible", "complete", "component", "components",
    "conf", "config", "configs", "configuration", "configure", "configured", "configures",
    "const", "context", "convert", "converted", "core", "correct", "correctly", "could",
    "create", "created", "creates", "creating", "current", "data", "default", "defaults",
    "def", "define", "defined", "defines", "delete", "deleted", "deletes", "dep", "deps",
    "depend", "dependency", "dependencies", "desc", "description", "detail", "details",
    "diff", "disable", "disabled", "doc", "docs", "document", "documentation", "documented",
    "documents", "does", "done", "each", "edit", "edited", "elif", "else", "enable",
    "enabled", "enables", "end", "ensure", "ensures", "entry", "entries", "enum", "env",
    "error", "errors", "etc", "event", "events", "example", "examples", "exception", "exist",
    "existing", "export", "extend", "extended", "extends", "extra", "fail", "failed",
    "fails", "failure", "false", "feat", "feature", "features", "field", "fields", "file",
    "files", "filter", "final", "fix", "fixed", "fixes", "fixing", "flag", "flags", "float",
    "flow", "for", "format", "from", "full", "func", "function", "functions", "gen",
    "general", "generate", "generated", "get", "gets", "git", "group", "handle", "handled",
    "handler", "handlers", "handles", "handling", "has", "have", "helper", "helpers", "http",
    "https", "impl", "implement", "implemented", "implementing", "implements", "implementation",
    "import", "imports", "improve", "improved", "improves", "improvement", "include",
    "included", "includes", "index", "info", "init", "initialize", "input", "inputs",
    "int", "interface", "internal", "into", "introduce", "introduced", "introduces", "invalid",
    "issue", "issues", "item", "items", "json", "key", "keys", "kind", "label", "large",
    "len", "let", "level", "lib", "line", "lines", "list", "load", "local", "lock", "log",
    "logging", "logic", "main", "make", "manage", "manager", "map", "match", "max", "may",
    "message", "meta", "method", "methods", "migrate", "min", "missing", "mock", "mode",
    "model", "models", "mod", "modified", "modifies", "modify", "modifying", "module",
    "modules", "more", "move", "moved", "moves", "msg", "multiple", "must", "name", "names",
    "needed", "new", "nil", "node", "nodes", "none", "not", "note", "notes", "null", "num",
    "number", "obj", "object", "objects", "old", "one", "only", "opt", "option", "options",
    "order", "other", "out", "output", "outputs", "override", "package", "packages", "page",
    "param", "params", "parameter", "parameters", "parse", "parser", "part", "pass", "patch",
    "path", "paths", "pattern", "payload", "per", "perform", "pkg", "place", "plan",
    "plugin", "point", "post", "pr", "prevent", "print", "process", "project", "prop",
    "properties", "property", "provide", "provided", "provides", "pull", "put", "query",
    "range", "raw", "read", "readme", "record", "records", "refactor", "refactored",
    "refactoring", "ref", "refs", "register", "related", "remove", "removed", "removes",
    "removing", "rename", "renamed", "renames", "replace", "replaced", "replaces", "repo",
    "report", "req", "request", "requests", "require", "required", "requires", "res",
    "reset", "resolve", "resolved", "resource", "response", "responses", "result", "results",
    "return", "returns", "review", "root", "route", "routes", "row", "rows", "rule", "rules",
    "run", "runner", "running", "runs", "same", "save", "schema", "scope", "script",
    "scripts", "section", "sections", "select", "self", "send", "server", "service",
    "services", "session", "set", "sets", "setting", "settings", "setup", "should", "show",
    "simple", "single", "size", "skip", "some", "source", "sources", "spec", "specs", "src",
    "start", "state", "states", "static", "status", "step", "steps", "str", "string",
    "strings", "struct", "style", "sub", "summary", "super", "support", "supported",
    "supporting", "supports", "switch", "sym", "symbol", "symbols", "sync", "system",
    "table", "target", "task", "tasks", "temp", "template", "test", "tested", "testing",
    "tests", "text", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "throw", "throws", "time", "timeout", "title", "tmp", "todo",
    "token", "tokens", "toml", "tool", "tools", "total", "track", "trigger", "true", "try",
    "two", "txt", "type", "types", "ui", "unit", "units", "unknown", "update", "updated",
    "updates", "updating", "upgrade", "url", "usage", "use", "used", "user", "users",
    "uses", "using", "util", "utils", "utility", "utilities", "val", "valid", "validate",
    "validation", "value", "values", "var", "vars", "version", "versions", "view", "void",
    "when", "where", "which", "while", "will", "with", "without", "work", "workflow",
    "workflows", "working", "works", "wrapper", "write", "written", "wrong", "yaml", "yml",
})

CODE_STOPWORDS_SHA256: str = digest(sorted(CODE_STOPWORDS))

_CODE_STRICT_WEIGHTS: dict[str, float] = {
    EXPLICIT: 1.0,
    "syntactic": 1.0,
    "deductive": 1.0,
}

_CODE_BLOCKED_BASES: frozenset[str] = frozenset(
    set(BLOCKED_INFERENCE_TYPES)
    | {
        "common_sense",
        "temporal",
        "inductive",
        "causal",
        "elimination",
        "occam",
        "abductive_inv",
        "dialectical",
        "multi_hop",
        "default",
        "authority",
        "transductive",
        "probabilistic",
        "analogical",
        "pragmatic",
        "intentionality",
        "attribution",
        "semantic",
        "fixed_point",
        "spatial",
        "bayesian",
        "evolutionary",
    }
)

CODE_DIFF_EPISTEMIC_PROFILE = EpistemicProfile(
    name="code-diff-strict",
    version="1.0",
    weights=_CODE_STRICT_WEIGHTS,
    blocked=_CODE_BLOCKED_BASES,
)

# ---------------------------------------------------------------------------
# Regexes & file categorization
# ---------------------------------------------------------------------------

_COMMENT_LINE_RE = re.compile(r"^\s*(?://|/\*|\*|\*/|<!--|#(?!!)|\"\"\"|''')")
_DOCSTRING_OR_COMMENT_RE = re.compile(r"^\s*(?://|/\*|\*|<!--|#\s|\"\"\"|'''|///|//!)")
_DEF_RE = re.compile(
    r"\b(?:export\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|type|interface|enum)|"
    r"async\s+def|def|class|function|fn|pub\s+(?:async\s+)?fn|func(?:\s*\([^)]*\))?|"
    r"struct|interface|enum|type|const|let|var|module|trait|impl)\s+([A-Za-z_][\w]*)"
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
    r"(?i)(?:^|/)(?:tests?|specs?|__tests__)/|(?:_test|_tests|\.test|\.spec)\.[A-Za-z0-9]+$|"
    r"Tests?\.(?:swift|java|kt|cs|go|py|ts|tsx|js|jsx|rs|rb)$"
)
_TEST_FN_RE = re.compile(r"\b(?:def\s+(test_\w+)|func\s+(Test\w+)|fn\s+(test_\w+))")
_TEST_JS_RE = re.compile(r"\b(?:it|test|describe)\(\s*['\"]([^'\"]{3,80})['\"]")
_DOCS_FILE_RE = re.compile(
    r"(?i)(?:^|/)(?:docs?|documentation)/|(?:^|/)readme(?:\.[^/]+)?$|(?:^|/)changelog(?:\.[^/]+)?$|"
    r"\.(?:md|mdx|rst|adoc|txt)$"
)
_LOCK_GEN_RE = re.compile(
    r"(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|uv\.lock|"
    r"composer\.lock|Gemfile\.lock|\.snap$|\.min\.js$)$",
    re.IGNORECASE,
)
_CONFIG_FILE_RE = re.compile(
    r"(?i)(?:^|/)(?:\.github/workflows|ci|config|\.vscode|\.devcontainer)/|"
    r"(?:^|/)(?:pyproject\.toml|package\.json|requirements[^/]*\.txt|environment\.ya?ml|setup\.cfg|"
    r"setup\.py|tox\.ini|pom\.xml|build\.gradle(?:\.kts)?|gradle\.properties|composer\.json|"
    r"Cargo\.toml|go\.mod|Gemfile|Makefile|Dockerfile[^/]*|tsconfig[^/]*\.json|\.env[^/]*|"
    r"\.gitignore|\.eslintrc[^/]*|\.prettierrc[^/]*)|\.(?:ya?ml|json|toml|ini|cfg|conf|properties)$"
)
_CODE_EXT_RE = re.compile(
    r"\.(?:py|js|jsx|ts|tsx|mjs|cjs|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|hpp|cs|sh|zig|sql|vue|svelte)$",
    re.IGNORECASE,
)

_DEP_JSON_RE = re.compile(r'^\s*"([@A-Za-z0-9_./-]{2,80})"\s*:\s*"([~^>=<*!v]*\d+[^"\s]{0,30})"')
_DEP_PIP_RE = re.compile(r"^\s*([A-Za-z0-9_.-]{2,60})(?:\[[^\]]+\])?\s*(?:==|>=|<=|~=|!=)\s*([0-9][A-Za-z0-9.*+-]{0,30})")
_DEP_TOML_RE = re.compile(
    r'^\s*([A-Za-z0-9_-]{2,60})\s*=\s*(?:"([~^>=<*]*\d+[^"\s]{0,30})"|\{[^}]*version\s*=\s*"([^"]+)")'
)
_DEP_GO_RE = re.compile(r"^\s*([A-Za-z0-9.-]+\.[A-Za-z0-9./_-]+)\s+(v\d+\.\d+\.\d+[^\s]*)")
_DEP_GEM_RE = re.compile(r"""^\s*gem\s+['"]([A-Za-z0-9_.-]+)['"]\s*(?:,\s*['"]([^'"]+)['"])?""")


def classify_file_category(path: str) -> str:
    """Return one of `'test'`, `'docs'`, `'config'`, or `'src'`."""
    p = path.strip().replace("\\", "/")
    if _TEST_FILE_RE.search(p):
        return "test"
    if _DOCS_FILE_RE.search(p):
        return "docs"
    if _CONFIG_FILE_RE.search(p):
        return "config"
    return "src"


def split_code_tokens(text: str) -> list[str]:
    """Tokenize identifiers/paths into lowercase words plus compound tokens (order-preserving)."""
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    # Extract dotted/slashed/colon compounds and regular identifiers
    raw_parts = re.findall(r"[A-Za-z0-9_./:-]+", text)
    for raw in raw_parts:
        clean = raw.strip(".:/-_")
        if len(clean) >= 2:
            low = clean.lower()
            if low not in seen:
                seen.add(low)
                out.append(low)
        # Split on punctuation then camelCase
        for seg in re.split(r"[./:_-]+", clean):
            if not seg:
                continue
            seg_low = seg.lower()
            if len(seg_low) >= 2 and seg_low not in seen:
                seen.add(seg_low)
                out.append(seg_low)
            camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", seg)
            camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", camel)
            for sub in camel.split():
                sub_low = sub.lower()
                if len(sub_low) >= 2 and sub_low not in seen:
                    seen.add(sub_low)
                    out.append(sub_low)
    return out


def repo_identifier_tokens(repo_name_or_url: str | None) -> frozenset[str]:
    """Extract lowercase project/owner tokens from a repository identifier or URL."""
    if not repo_name_or_url:
        return frozenset()
    m = re.search(r"github\.com/([^/]+/[^/]+)", repo_name_or_url)
    raw = m.group(1) if m else repo_name_or_url
    raw = raw.split("/pull/")[0].strip("/")
    toks = {t for t in split_code_tokens(raw) if len(t) >= 2}
    return frozenset(toks)


def specific_tokens(
    text: str,
    *,
    repo_tokens: frozenset[str] = frozenset(),
    min_len: int = 3,
) -> list[str]:
    """Return non-stopword, non-repo-name tokens useful for specific diff matching."""
    out: list[str] = []
    for tok in split_code_tokens(text):
        if len(tok) < min_len:
            continue
        if tok in CODE_STOPWORDS or tok in repo_tokens:
            continue
        out.append(tok)
    return out


_STEM_SUFFIXES: tuple[str, ...] = (
    "ation",
    "tion",
    "sion",
    "ment",
    "ing",
    "ied",
    "ies",
    "ed",
    "es",
    "er",
    "ly",
    "s",
)


def _stem_token(tok: str, *, repo_tokens: frozenset[str] = frozenset()) -> str:
    """Conservatively stem a non-stopword, non-repo token of length >= 5 to a stem >= 4 chars."""
    t = (tok or "").strip().lower()
    if len(t) < 5 or t in CODE_STOPWORDS or t in repo_tokens:
        return t
    for suf in _STEM_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            cand = t[: -len(suf)]
            if cand not in CODE_STOPWORDS and cand not in repo_tokens:
                return cand
            break
    return t


def _enclosing_hunk_header(doc_text: str, pos: int) -> str:
    """Return the enclosing `@@ ... @@ <header>` context label preceding character offset *pos*."""
    if not doc_text or pos <= 0:
        return ""
    prefix = doc_text[:pos]
    for ln in reversed(prefix.splitlines()):
        m = _HUNK_RE.match(ln)
        if m:
            return m.group(1).strip()
    return ""


def is_compound_identifier(token: str) -> bool:
    """Return True if *token* looks like a concrete multi-part symbol, path, or CLI flag."""
    t = token.strip("`\"'")
    if len(t) < 4:
        return False
    if any(sep in t for sep in ("_", "/", "::", "->", "--")):
        return True
    if "." in t and not t.endswith(".") and re.search(r"[A-Za-z]\.[A-Za-z]", t):
        return True
    return bool(re.search(r"[a-z][A-Z]", t))


def _norm_camel_snake(token: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token.strip())
    return re.sub(r"[\W]+", "_", s).strip("_").lower()


def _matches_target_symbol(target: str, text: str) -> bool:
    """Strict word-boundary / snake-camel match for *target* in *text*."""
    tgt = (target or "").strip().strip("`\"'")
    if not tgt or not text:
        return False
    core = re.sub(r"\(\)$", "", tgt)
    if _has_boundary_aligned_match(text, tgt) or (core != tgt and _has_boundary_aligned_match(text, core)):
        return True
    if _has_boundary_aligned_match(text.lower(), core.lower()):
        return True
    norm_t = _norm_camel_snake(core)
    if len(norm_t) >= 3:
        for cand in re.findall(r"[A-Za-z0-9_./:-]{3,}", text):
            if _norm_camel_snake(cand) == norm_t:
                return True
            if "/" in cand and _norm_camel_snake(cand.rsplit("/", 1)[-1].rsplit(".", 1)[0]) == norm_t:
                return True
    return False


# ---------------------------------------------------------------------------
# Diff extraction & deterministic 3-tier hunk-to-file attribution
# ---------------------------------------------------------------------------

def _path_variants(path_str: str) -> list[str]:
    norm = path_str.strip().lstrip("./")
    if not norm:
        return []
    parts = norm.split("/")
    return ["/".join(parts[i:]) for i in range(len(parts))]


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
        if sym.lower() in CODE_STOPWORDS or len(sym) < 2:
            continue
        out.append(sym)
    return out


def _extract_definitions(line: str) -> list[str]:
    if _COMMENT_LINE_RE.match(line):
        return []
    found: list[str] = []
    found.extend(_DEF_RE.findall(line))
    found.extend(_TYPED_DECL_RE.findall(line))
    found.extend(_METHOD_HEAD_RE.findall(line))
    return [s for s in dict.fromkeys(found) if len(s) >= 2 and s.lower() not in CODE_STOPWORDS]


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


def _chunk_syntax_compatible(chunk: str, filename: str) -> bool:
    """Conservative negative compatibility check between a patch chunk and candidate file path."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename.rsplit("/", 1)[-1] else ""
    cat = classify_file_category(filename)
    plus_lines = [ln[1:] for ln in chunk.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    if not plus_lines:
        return True
    has_py_def = any(re.match(r"^\s*(?:async\s+)?def\s+\w+\(", ln) for ln in plus_lines)
    has_md_head = sum(1 for ln in plus_lines if re.match(r"^#{1,4}\s+\S", ln)) >= 2
    if cat == "docs" and has_py_def:
        return False
    if ext == ".json" and has_py_def:
        return False
    if ext in {".py", ".rs", ".go", ".java"} and has_md_head and not any("def " in ln or "fn " in ln or "func " in ln for ln in plus_lines):
        return False
    return True


def _match_chunk_to_file(chunk: str, files: Sequence[str]) -> str | None:
    """Try unambiguous file attribution when positional 1-to-1 mapping does not hold."""
    if not files:
        return None
    # Explicit git diff header inside chunk if present
    for pat in (r"(?m)^\+\+\+\s+(?:b/)?(.+)$", r"(?m)^---\s+(?:a/)?(.+)$", r"(?m)^diff --git a/(\S+) b/(\S+)$"):
        m = re.search(pat, chunk)
        if m:
            cand = m.group(m.lastindex or 1).strip()
            if cand != "/dev/null":
                hits = [f for f in files if f == cand or f.endswith("/" + cand)]
                if len(hits) == 1:
                    return hits[0]

    # Try unique stem / extension hints in chunk lines
    compat = [f for f in files if _chunk_syntax_compatible(chunk, f)]
    if len(compat) == 1:
        return compat[0]

    chunk_toks = set(split_code_tokens(chunk[:1500]))
    scored: list[str] = []
    for f in compat:
        stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if len(stem) >= 4 and stem not in CODE_STOPWORDS and stem in chunk_toks:
            scored.append(f)
    if len(scored) == 1:
        return scored[0]
    return None


def _build_chunks_from_pr(pr: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build diff hunk chunks with explicit attribution status."""
    raw_fp = pr.get("file_patches")
    if isinstance(raw_fp, Sequence) and raw_fp:
        chunks: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw_fp):
            fname = str(entry.get("filename") or "").strip() or None
            patch = str(entry.get("patch") or "")
            chunks.append(
                {
                    "chunk_id": f"diff_{pr.get('pr_id', 'unknown')}_{idx:04d}",
                    "file": fname,
                    "category": classify_file_category(fname) if fname else "unknown",
                    "attribution_mode": "exact_aidev" if fname else "unattributed",
                    "patch": patch,
                    "status": str(entry.get("status") or ""),
                }
            )
        return chunks

    files = list(pr.get("files") or ())
    patch = str(pr.get("patch") or "")
    if not patch.strip():
        return []
    raw_chunks = [c for c in patch.split("\n---FILE---\n") if c.strip()]
    pos_ok = len(raw_chunks) == len(files) and all(
        _chunk_syntax_compatible(c, f) for c, f in zip(raw_chunks, files)
    )
    chunks_out: list[dict[str, Any]] = []
    for idx, c in enumerate(raw_chunks):
        if pos_ok:
            fname = files[idx]
            mode = "positional_1to1"
        else:
            fname = _match_chunk_to_file(c, files)
            mode = "unambiguous_match" if fname else "unattributed"
        chunks_out.append(
            {
                "chunk_id": f"diff_{pr.get('pr_id', 'unknown')}_{idx:04d}",
                "file": fname,
                "category": classify_file_category(fname) if fname else "unknown",
                "attribution_mode": mode,
                "patch": c,
                "status": "",
            }
        )
    return chunks_out


@dataclass(frozen=True)
class DiffFactRecord:
    """Single literal diff fact bound to its raw SourceDocument span."""

    fact_id: str
    kind: str
    value: str
    file: str | None
    category: str
    attribution_mode: str
    document_id: str
    revision: str
    start: int
    end: int
    quote: str
    line_sign: str = "+"  # '+' | '-' | 'context' | 'meta'
    extra: dict[str, Any] = field(default_factory=dict)

    def to_evidence_ref(self) -> EvidenceRef:
        return EvidenceRef(
            document_id=self.document_id,
            revision=self.revision,
            start=self.start,
            end=self.end,
            quote=self.quote,
        )


@dataclass
class TypedDiffLedger:
    """Complete uncapped diff register with literal spans and attribution metadata."""

    pr_id: str
    files: tuple[str, ...]
    file_categories: dict[str, str]
    category_counts: dict[str, int]
    chunks: tuple[dict[str, Any], ...]
    raw_documents: dict[str, SourceDocument]
    facts: tuple[DiffFactRecord, ...]
    add_syms: tuple[str, ...]
    rem_syms: tuple[str, ...]
    mod_syms: tuple[str, ...]
    add_defs_by_file: dict[str, frozenset[str]]
    rem_defs_by_file: dict[str, frozenset[str]]
    all_defs_added: frozenset[str]
    all_defs_removed: frozenset[str]
    test_cites_by_file: dict[str, frozenset[str]]
    add_tests: tuple[str, ...]
    rem_tests: tuple[str, ...]
    config_keys_by_file: dict[str, frozenset[str]]
    doc_terms_by_file: dict[str, frozenset[str]]
    added_docstring_lines: tuple[tuple[str | None, str, str, int, int], ...]
    dependencies: tuple[dict[str, Any], ...]
    files_count: int
    adds_count: int | None
    dels_count: int | None
    truncated: bool
    truncated_fields: dict[str, dict[str, int]]
    ledger_digest: str
    summary_text: str

    def check_literal_quote(self, value: str, quote: str, *, document_id: str | None = None) -> tuple[bool, str | None, tuple[str, ...]]:
        """Run untouched `verify(..., evidence_profile='code_config')` against raw diff text."""
        q_norm = (quote or "").strip()
        if not q_norm or not value:
            return False, None, ("EMPTY_QUOTE_OR_VALUE",)

        cand_docs: list[SourceDocument] = []
        if document_id and document_id in self.raw_documents:
            cand_docs.append(self.raw_documents[document_id])
        else:
            for did, doc in sorted(self.raw_documents.items()):
                if q_norm in doc.text or q_norm.lstrip("+- ") in doc.text:
                    cand_docs.append(doc)
        if not cand_docs:
            return False, None, ("QUOTE_NOT_IN_DIFF",)

        last_codes: tuple[str, ...] = ()
        for doc in cand_docs:
            ledger = Ledger(sources=[Source(id=doc.document_id, text=doc.text, tier="authoritative", evidence_profile="code_config")])
            res = verify(
                [Assertion(subject=PR_ENTITY, predicate="diff_evidence", value=str(value), source_id=doc.document_id, quote=q_norm)],
                ledger,
            )
            if res.status == "PASS":
                return True, doc.document_id, ()
            last_codes = tuple(iss.code for iss in res.issues)
        return False, None, last_codes or ("CODE_CONFIG_VERIFY_FAILED",)

    def locate_ref(self, quote: str, *, preferred_doc_id: str | None = None) -> EvidenceRef | None:
        q = (quote or "").strip()
        if not q:
            return None
        docs_order: list[SourceDocument] = []
        if preferred_doc_id and preferred_doc_id in self.raw_documents:
            docs_order.append(self.raw_documents[preferred_doc_id])
        for did in sorted(self.raw_documents):
            if did != preferred_doc_id:
                docs_order.append(self.raw_documents[did])
        q_norm_ws = re.sub(r"[ \t]+", " ", q)
        for doc in docs_order:
            txt = doc.text
            idx = txt.find(q)
            if idx >= 0:
                if "\n" not in q:
                    # Expand partial single-line match to its complete enclosing diff line
                    # so leading signs (+/-) and guards (if/when) stay intact.
                    ls = txt.rfind("\n", 0, idx)
                    ls = 0 if ls < 0 else ls + 1
                    le = txt.find("\n", idx + len(q))
                    le = len(txt) if le < 0 else le
                    if le > ls and txt[ls:le].strip():
                        return EvidenceRef(doc.document_id, doc.revision, ls, le, txt[ls:le])
                return EvidenceRef(doc.document_id, doc.revision, idx, idx + len(q), txt[idx : idx + len(q)])
            if "\n" not in q:
                pos = 0
                for raw_ln in txt.splitlines(keepends=True):
                    stripped_ln = raw_ln.rstrip("\r\n")
                    ln_ws = re.sub(r"[ \t]+", " ", stripped_ln.strip())
                    if q_norm_ws and (q_norm_ws == ln_ws or q_norm_ws in ln_ws):
                        return EvidenceRef(doc.document_id, doc.revision, pos, pos + len(stripped_ln), stripped_ln)
                    pos += len(raw_ln)
        # Strict same-sign fallback when leading diff indent varies after '+' or '-'
        if "\n" not in q and q[0] in ("+", "-"):
            sign = q[0]
            q_body_ws = re.sub(r"[ \t]+", " ", q[1:].strip())
            if len(q_body_ws) >= 4:
                for doc in docs_order:
                    pos = 0
                    for raw_ln in doc.text.splitlines(keepends=True):
                        stripped_ln = raw_ln.rstrip("\r\n")
                        if stripped_ln.startswith(sign):
                            ln_body_ws = re.sub(r"[ \t]+", " ", stripped_ln[1:].strip())
                            if q_body_ws == ln_body_ws:
                                return EvidenceRef(doc.document_id, doc.revision, pos, pos + len(stripped_ln), stripped_ln)
                        pos += len(raw_ln)
        return None


def build_typed_diff_ledger(
    pr: Mapping[str, Any],
    *,
    max_syms: int | None = None,
    max_tests: int | None = None,
) -> TypedDiffLedger:
    """Build an uncapped, attributed, fail-closed TypedDiffLedger from a PR dict."""
    pr_id = str(pr.get("pr_id") or "unknown")
    files = list(pr.get("files") or ())
    chunks = _build_chunks_from_pr(pr)
    if not files and chunks:
        files = list(dict.fromkeys(c["file"] for c in chunks if c["file"]))

    file_cats: dict[str, str] = {f: classify_file_category(f) for f in files}
    cat_counts = {
        "src": sum(1 for f in files if file_cats[f] == "src" and not _LOCK_GEN_RE.search(f)),
        "test": sum(1 for f in files if file_cats[f] == "test"),
        "config": sum(1 for f in files if file_cats[f] == "config" and not _LOCK_GEN_RE.search(f)),
        "docs": sum(1 for f in files if file_cats[f] == "docs"),
    }

    raw_docs: dict[str, SourceDocument] = {}
    facts: list[DiffFactRecord] = []
    added_all: list[str] = []
    removed_all: list[str] = []
    touched_all: list[str] = []
    add_defs_by_file: dict[str, set[str]] = {}
    rem_defs_by_file: dict[str, set[str]] = {}
    all_defs_add: set[str] = set()
    all_defs_rem: set[str] = set()
    test_cites_by_file: dict[str, set[str]] = {}
    tests_plus: list[str] = []
    tests_minus: list[str] = []
    config_keys_by_file: dict[str, set[str]] = {}
    doc_terms_by_file: dict[str, set[str]] = {}
    added_docstrings: list[tuple[str | None, str, str, int, int]] = []
    dep_plus: dict[str, tuple[str | None, str | None, str, str, int, int]] = {}
    dep_minus: dict[str, tuple[str | None, str | None, str, str, int, int]] = {}

    fact_seq = 0

    def _next_id() -> str:
        nonlocal fact_seq
        fact_seq += 1
        return f"f_{fact_seq:05d}"

    for chunk in chunks:
        cid = chunk["chunk_id"]
        patch_txt = chunk["patch"]
        fname = chunk["file"]
        cat = chunk["category"]
        mode = chunk["attribution_mode"]
        if not patch_txt or not patch_txt.strip():
            continue
        doc = SourceDocument(document_id=cid, text=patch_txt, authority_rank=100, weight=1.0)
        raw_docs[cid] = doc

        offset = 0
        for line in patch_txt.splitlines(keepends=True):
            line_stripped = line.rstrip("\r\n")
            l_start = offset
            l_end = offset + len(line_stripped)
            offset += len(line)
            if not line_stripped or line_stripped.startswith("+++") or line_stripped.startswith("---"):
                continue

            if line_stripped.startswith("+"):
                body = line_stripped[1:]
                defs = _extract_definitions(body)
                syms = _extract_line_identifiers(body)
                t_fns = _extract_tests(body)
                added_all.extend(syms)
                tests_plus.extend(t_fns)
                for dsym in defs:
                    all_defs_add.add(dsym)
                    if fname and cat == "src":
                        add_defs_by_file.setdefault(fname, set()).add(dsym)
                        facts.append(
                            DiffFactRecord(
                                fact_id=_next_id(),
                                kind="added_definition",
                                value=dsym,
                                file=fname,
                                category=cat,
                                attribution_mode=mode,
                                document_id=cid,
                                revision=doc.revision,
                                start=l_start,
                                end=l_end,
                                quote=line_stripped,
                                line_sign="+",
                            )
                        )
                if fname and cat == "test":
                    t_tokens = set(syms) | set(split_code_tokens(body))
                    test_cites_by_file.setdefault(fname, set()).update(t_tokens)
                elif cat == "unknown" and t_fns:
                    for tf in t_fns:
                        tests_plus.append(tf)

                if fname and cat == "config" and not _LOCK_GEN_RE.search(fname):
                    if km := _KEY_RE.match(body):
                        config_keys_by_file.setdefault(fname, set()).add(km.group(1))
                    for fl in _FLAG_RE.findall(body):
                        config_keys_by_file.setdefault(fname, set()).add(fl)
                if fname and cat == "docs":
                    doc_terms_by_file.setdefault(fname, set()).update(split_code_tokens(body))
                elif _DOCSTRING_OR_COMMENT_RE.match(body) and len(body.strip()) >= 8:
                    added_docstrings.append((fname, line_stripped, cid, l_start, l_end))

                if dep := _extract_dep(body):
                    dep_plus[dep[0]] = (dep[1], fname, cid, line_stripped, l_start, l_end)

                if defs or t_fns or (fname and cat in ("config", "docs")) or _KEY_RE.match(body):
                    facts.append(
                        DiffFactRecord(
                            fact_id=_next_id(),
                            kind="added_line",
                            value=body.strip()[:240],
                            file=fname,
                            category=cat,
                            attribution_mode=mode,
                            document_id=cid,
                            revision=doc.revision,
                            start=l_start,
                            end=l_end,
                            quote=line_stripped,
                            line_sign="+",
                        )
                    )

            elif line_stripped.startswith("-"):
                body = line_stripped[1:]
                defs = _extract_definitions(body)
                syms = _extract_line_identifiers(body)
                t_fns = _extract_tests(body)
                removed_all.extend(syms)
                tests_minus.extend(t_fns)
                all_defs_rem.update(defs)
                if fname:
                    rem_defs_by_file.setdefault(fname, set()).update(defs)
                if fname and cat == "config" and not _LOCK_GEN_RE.search(fname):
                    if km := _KEY_RE.match(body):
                        config_keys_by_file.setdefault(fname, set()).add(km.group(1))
                if dep := _extract_dep(body):
                    dep_minus[dep[0]] = (dep[1], fname, cid, line_stripped, l_start, l_end)
                if defs or t_fns or _KEY_RE.match(body):
                    facts.append(
                        DiffFactRecord(
                            fact_id=_next_id(),
                            kind="removed_line",
                            value=body.strip()[:240],
                            file=fname,
                            category=cat,
                            attribution_mode=mode,
                            document_id=cid,
                            revision=doc.revision,
                            start=l_start,
                            end=l_end,
                            quote=line_stripped,
                            line_sign="-",
                        )
                    )
            elif m := _HUNK_RE.match(line_stripped):
                if m.group(1):
                    touched_all.extend(_extract_line_identifiers(m.group(1)))
            elif line_stripped.startswith(" "):
                touched_all.extend(_extract_line_identifiers(line_stripped[1:]))

    a_uniq = list(dict.fromkeys(added_all))
    r_uniq = list(dict.fromkeys(removed_all))
    a_set, r_set = set(a_uniq), set(r_uniq)
    mod_syms = list(
        dict.fromkeys([x for x in a_uniq if x in r_set] + [x for x in dict.fromkeys(touched_all) if x not in a_set and x not in r_set])
    )
    add_syms = [x for x in a_uniq if x not in r_set]
    rem_syms = [x for x in r_uniq if x not in a_set]

    test_files: list[str] = []
    for f in files:
        if file_cats[f] == "test":
            test_files.extend(_path_variants(f))
    tp_uniq = list(dict.fromkeys(tests_plus))
    tm_uniq = list(dict.fromkeys(tests_minus))
    tp_set, tm_set = set(tp_uniq), set(tm_uniq)
    add_tests = list(dict.fromkeys(test_files + [t for t in tp_uniq if t not in tm_set]))
    rem_tests = list(dict.fromkeys([t for t in tm_uniq if t not in tp_set]))

    dependencies: list[dict[str, Any]] = []
    for pkg, (ver, fn, cid, ln, st, en) in dep_plus.items():
        if pkg in dep_minus:
            if dep_minus[pkg][0] != ver:
                dependencies.append({"package": pkg, **({"version": ver} if ver else {}), "action": "update", "file": fn, "quote": ln, "document_id": cid})
        else:
            dependencies.append({"package": pkg, **({"version": ver} if ver else {}), "action": "add", "file": fn, "quote": ln, "document_id": cid})
    for pkg, (ver, fn, cid, ln, st, en) in dep_minus.items():
        if pkg not in dep_plus:
            dependencies.append({"package": pkg, **({"version": ver} if ver else {}), "action": "remove", "file": fn, "quote": ln, "document_id": cid})

    truncated_fields: dict[str, dict[str, int]] = {}

    def _apply_cap(name: str, seq: list[str], cap: int | None) -> tuple[str, ...]:
        if cap is None:
            return tuple(seq)
        if cap < 0:
            raise ValueError(f"Truncation cap for {name} must be >= 0, got {cap}")
        if len(seq) > cap:
            truncated_fields[name] = {"kept": cap, "total": len(seq)}
            return tuple(seq[:cap])
        return tuple(seq)

    add_syms_t = _apply_cap("add_syms", add_syms, max_syms)
    rem_syms_t = _apply_cap("rem_syms", rem_syms, max_syms)
    mod_syms_t = _apply_cap("mod_syms", mod_syms, max_syms)
    add_tests_t = _apply_cap("add_tests", add_tests, max_tests)
    rem_tests_t = _apply_cap("rem_tests", rem_tests, max_tests)

    meta_lines = [
        f"The {PR_ENTITY} changed {len(files)} files (src={cat_counts['src']}, test={cat_counts['test']}, config={cat_counts['config']}, docs={cat_counts['docs']}).",
        f"The {PR_ENTITY} added_lines={pr.get('adds')} deleted_lines={pr.get('dels')}.",
        *[f"The {PR_ENTITY} changed file {f} category={file_cats[f]}." for f in files],
    ]
    summary_text = "\n".join(meta_lines)
    raw_docs[f"diff_meta_{pr_id}"] = SourceDocument(document_id=f"diff_meta_{pr_id}", text=summary_text, authority_rank=100, weight=1.0)

    ledger_digest = digest({
        "pr_id": pr_id,
        "files": files,
        "add_syms": list(add_syms_t),
        "rem_syms": list(rem_syms_t),
        "mod_syms": list(mod_syms_t),
        "add_tests": list(add_tests_t),
        "rem_tests": list(rem_tests_t),
        "deps": [{k: v for k, v in d.items() if k in ("package", "version", "action", "file")} for d in dependencies],
        "cat_counts": cat_counts,
        "truncated": bool(truncated_fields),
        "truncated_fields": truncated_fields,
        "chunks": [digest(c["patch"]) for c in chunks],
    })

    return TypedDiffLedger(
        pr_id=pr_id,
        files=tuple(files),
        file_categories=file_cats,
        category_counts=cat_counts,
        chunks=tuple(chunks),
        raw_documents=raw_docs,
        facts=tuple(facts),
        add_syms=add_syms_t,
        rem_syms=rem_syms_t,
        mod_syms=mod_syms_t,
        add_defs_by_file={k: frozenset(v) for k, v in add_defs_by_file.items()},
        rem_defs_by_file={k: frozenset(v) for k, v in rem_defs_by_file.items()},
        all_defs_added=frozenset(all_defs_add),
        all_defs_removed=frozenset(all_defs_rem),
        test_cites_by_file={k: frozenset(v) for k, v in test_cites_by_file.items()},
        add_tests=add_tests_t,
        rem_tests=rem_tests_t,
        config_keys_by_file={k: frozenset(v) for k, v in config_keys_by_file.items()},
        doc_terms_by_file={k: frozenset(v) for k, v in doc_terms_by_file.items()},
        added_docstring_lines=tuple(added_docstrings),
        dependencies=tuple(dependencies),
        files_count=len(files),
        adds_count=pr.get("adds"),
        dels_count=pr.get("dels"),
        truncated=bool(truncated_fields),
        truncated_fields=truncated_fields,
        ledger_digest=ledger_digest,
        summary_text=summary_text,
    )


# ---------------------------------------------------------------------------
# Versioned Vocabulary, DeclaredRules, RelationCompositionRules & DomainProfile
# ---------------------------------------------------------------------------

CODE_CHANGE_INFERENCE_VOCABULARY = Vocabulary(
    name="code_change_inference",
    version="1.0.0",
    closed_entities=True,
    entities=((PR_ENTITY, ("PR", "this PR", "the PR", "the pull request", "this pull request")),),
    require_explicit_condition_status=False,
    predicates=(
        PredicateSpec("implements", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("adds_test_for", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("removes_support_for", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("configures", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("updates_dependency", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("documents", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal")),
        PredicateSpec("other_behavior", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact", "proposal", "hypothesis")),
        PredicateSpec("i5_structural_signal", FieldSpec("string"), cardinality="multi", assertion_kinds=("fact",)),
    ),
)

DECLARED_CODE_RULES: dict[str, DeclaredRule] = {
    "code.implements_symbol": DeclaredRule(
        rule_id="code.implements_symbol",
        version="1.0",
        kind="conjunction",
        description="A behavior claim claiming implementation is derived only if a non-comment '+' definition in a src file matches target.",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "implements"},
        parameters={"require_src_file": True, "require_target_definition": True},
    ),
    "code.adds_test_for": DeclaredRule(
        rule_id="code.adds_test_for",
        version="1.0",
        kind="conjunction",
        description="Adding tests for X requires an added/modified test file or test function whose '+' lines literally cite X.",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "adds_test_for"},
        parameters={"require_test_artifact": True, "require_target_cited_in_test": True},
    ),
    "code.removes_support_for": DeclaredRule(
        rule_id="code.removes_support_for",
        version="1.0",
        kind="conjunction",
        description="Removing support for X requires X removed in '-' lines with net_removed=True (absent from '+' lines).",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "removes_support_for"},
        parameters={"require_net_removed": True},
    ),
    "code.configures": DeclaredRule(
        rule_id="code.configures",
        version="1.0",
        kind="conjunction",
        description="Configuration claim requires a non-lockfile config file modifying key/flag K matching target.",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "configures"},
        parameters={"require_config_file": True, "exclude_lockfiles": True},
    ),
    "code.updates_dependency": DeclaredRule(
        rule_id="code.updates_dependency",
        version="1.0",
        kind="implication",
        description="Dependency claim requires matching manifest/lockfile package change and exact version if stated.",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "updates_dependency"},
        parameters={"require_exact_version_if_stated": True},
    ),
    "code.documents": DeclaredRule(
        rule_id="code.documents",
        version="1.0",
        kind="conjunction",
        description="Documentation claim requires a docs-file change matching target, or a '+' docstring literally citing the exact target symbol.",
        required_premises=({"predicate": "diff_premise", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "documents"},
        parameters={"docstring_requires_literal_target": True},
    ),
    "code.doc_only_code_contradiction": DeclaredRule(
        rule_id="code.doc_only_code_contradiction",
        version="1.0",
        kind="exclusion",
        description="Checked 'Documentation update' only checkbox contradicts non-doc code file edits in the diff.",
        required_premises=({"predicate": "i5_structural_signal", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "i5_structural_signal", "polarity": -1},
        parameters={"structural_kind": "D_doc_only_but_code"},
    ),
    "code.placeholder_for_large_diff": DeclaredRule(
        rule_id="code.placeholder_for_large_diff",
        version="1.0",
        kind="threshold",
        description="Placeholder description (<8 content words and short title) covering >=100 lines of non-doc code churn.",
        required_premises=({"predicate": "i5_structural_signal", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "i5_structural_signal"},
        parameters={"structural_kind": "P_placeholder_large_diff"},
    ),
    "code.severe_scope_understatement": DeclaredRule(
        rule_id="code.severe_scope_understatement",
        version="1.0",
        kind="threshold",
        description="Explicit file list (>=2 enumerated files) leaving >=75% of significant changed files unmentioned.",
        required_premises=({"predicate": "i5_structural_signal", "condition_status": "unconditional"},),
        conclusion_template={"predicate": "i5_structural_signal"},
        parameters={"structural_kind": "U_severe_understated_files"},
    ),
}

RELATION_CODE_RULES: dict[str, RelationCompositionRule] = {
    "rel.adds_test_for": RelationCompositionRule(
        rule_id="rel.adds_test_for",
        version="1.0",
        left_predicate="pr_adds_test_file",
        right_predicate="test_cites_symbol",
        composed_predicate="adds_test_for",
        require_temporal_overlap=False,
        require_exact_bridge=True,
    ),
    "rel.configures": RelationCompositionRule(
        rule_id="rel.configures",
        version="1.0",
        left_predicate="pr_changes_config_file",
        right_predicate="config_sets_key",
        composed_predicate="configures",
        require_temporal_overlap=False,
        require_exact_bridge=True,
    ),
    "rel.documents": RelationCompositionRule(
        rule_id="rel.documents",
        version="1.0",
        left_predicate="pr_changes_doc_file",
        right_predicate="doc_mentions_term",
        composed_predicate="documents",
        require_temporal_overlap=False,
        require_exact_bridge=True,
    ),
}

CODE_CHANGE_INFERENCE_PROFILE = DomainProfile(
    profile_id="code_change_inference",
    version="1.0.0",
    vocabulary=CODE_CHANGE_INFERENCE_VOCABULARY,
    epistemic_profile=CODE_DIFF_EPISTEMIC_PROFILE,
    policy=PolicyConfiguration(
        tau=0.70,
        delta=0.15,
        allowed_claim_kinds_for_evaluation=("extraction", "derivation", "hypothesis"),
        allowed_claim_kinds_for_authorization=("extraction", "derivation"),
        block_critical_unverified_in_authorization=True,
        require_deterministic_critical_steps=True,
        allow_weight_override_of_formal_error=False,
    ),
    declared_rules=DECLARED_CODE_RULES,
    relation_rules=RELATION_CODE_RULES,
)

# ---------------------------------------------------------------------------
# Anti-N2 Relevance Guard & CodeDiffRuleVerifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelevanceConfig:
    """Parameters governing anti-N2 premise relevance filtering."""

    min_specific_anchors: int = 2
    require_target_match_when_present: bool = True
    require_category_compat: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerdictThresholds:
    """Thresholds mapping PR-level claim states to CONSISTENT / REVIEW / INCONSISTENT."""

    k_phantom: int = 1
    theta_phantom: float = 0.50
    singleton_phantom_inconsistent: bool = True
    k_lit: int = 2
    theta_lit: float = 0.50
    review_on_single_unsupported_or_phantom: bool = True
    review_on_all_unruled_hypothesis: bool = True
    include_i5: bool = True
    i5_placeholder_max_words: int = 8
    i5_placeholder_max_title_words: int = 3
    i5_placeholder_min_churn: int = 100
    i5_understated_min_enumerated: int = 2
    i5_understated_min_share: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _operation_category_compatible(operation: str, premise_file: str | None, premise_quote: str, ledger: TypedDiffLedger) -> bool:
    if not premise_file:
        # Unattributed chunks cannot satisfy file-category operations
        if operation in ("configures", "documents", "adds_test_for"):
            return bool(operation == "adds_test_for" and _extract_tests(premise_quote.lstrip("+- ")))
        return True
    cat = ledger.file_categories.get(premise_file, classify_file_category(premise_file))
    if operation == "adds_test_for":
        return cat == "test" or bool(_extract_tests(premise_quote.lstrip("+- ")))
    if operation == "configures":
        return cat == "config" and not _LOCK_GEN_RE.search(premise_file)
    if operation == "documents":
        return cat == "docs" or bool(_DOCSTRING_OR_COMMENT_RE.match(premise_quote.lstrip("+- ")))
    if operation == "updates_dependency":
        return bool(_extract_dep(premise_quote.lstrip("+- "))) or cat == "config"
    return True


def check_premise_relevance(
    *,
    claim_text: str,
    operation: str,
    target: str | None,
    premise_file: str | None,
    premise_symbol: str | None,
    premise_quote: str,
    ledger: TypedDiffLedger,
    repo_tokens: frozenset[str] = frozenset(),
    config: RelevanceConfig = RelevanceConfig(),
    hunk_header: str | None = None,
) -> tuple[bool, str]:
    """Deterministic anti-N2 guard rejecting generic same-repo or stopword-only matches."""
    if config.require_category_compat and not _operation_category_compatible(
        operation, premise_file, premise_quote, ledger
    ):
        return False, "INCOMPATIBLE_OPERATION_CATEGORY"

    prem_blob = " ".join(
        x for x in (premise_file or "", hunk_header or "", premise_symbol or "", premise_quote or "") if x
    )
    prem_specific_list = specific_tokens(prem_blob, repo_tokens=repo_tokens, min_len=3)
    prem_specific = set(prem_specific_list)
    prem_stems = {_stem_token(t, repo_tokens=repo_tokens) for t in prem_specific_list}

    tgt_clean = (target or "").strip().strip("`\"'")
    tgt_spec = specific_tokens(tgt_clean, repo_tokens=repo_tokens, min_len=3)
    if config.require_target_match_when_present and tgt_spec:
        # Concrete target present: must match target symbol/path OR share specific tokens/stems or compound identifier
        if _matches_target_symbol(tgt_clean, prem_blob):
            return True, "TARGET_EXACT_MATCH"
        tgt_compounds = [tok for tok in re.findall(r"[A-Za-z0-9_./:-]+", tgt_clean) if is_compound_identifier(tok)]
        if any(_matches_target_symbol(tc, prem_blob) for tc in tgt_compounds):
            return True, "TARGET_COMPOUND_MATCH"
        tgt_stems = {_stem_token(t, repo_tokens=repo_tokens) for t in tgt_spec}
        tgt_atomic_stems = {
            _stem_token(t, repo_tokens=repo_tokens)
            for t in tgt_spec
            if not is_compound_identifier(t)
        } or tgt_stems
        shared_tokens = set(tgt_spec) & prem_specific
        shared_stems = tgt_stems & prem_stems
        shared_anchors = shared_tokens | shared_stems
        if len(tgt_atomic_stems) == 1 and (shared_anchors & (tgt_atomic_stems | set(tgt_spec))):
            return True, "TARGET_SPECIFIC_TOKEN_MATCH"
        if len(tgt_atomic_stems) >= 2 and len(shared_anchors) >= min(len(tgt_atomic_stems), config.min_specific_anchors):
            return True, "TARGET_MULTI_SPECIFIC_MATCH"
        return False, "TARGET_NOT_IN_PREMISE"

    # Generic / missing target: rely on non-stopword, non-repo specific anchors in claim_text
    claim_compounds = [
        tok
        for tok in re.findall(r"`([^`]{3,80})`|([A-Za-z0-9_./:-]{4,})", claim_text or "")
        for tok in (tok if isinstance(tok, tuple) else (tok,))
        if tok and is_compound_identifier(tok) and tok.lower() not in repo_tokens and tok.lower() not in CODE_STOPWORDS
    ]
    if any(_matches_target_symbol(comp, prem_blob) for comp in claim_compounds):
        return True, "CLAIM_COMPOUND_ANCHOR_MATCH"

    claim_spec = specific_tokens(claim_text or "", repo_tokens=repo_tokens, min_len=4)
    claim_stems = {_stem_token(t, repo_tokens=repo_tokens) for t in claim_spec}
    overlap = (set(claim_spec) & prem_specific) | (claim_stems & prem_stems)
    if len(overlap) >= config.min_specific_anchors:
        return True, "MULTI_SPECIFIC_ANCHOR_MATCH"
    return False, "INSUFFICIENT_SPECIFIC_OVERLAP"


def _detect_rule_satisfaction(
    *,
    operation: str,
    target: str | None,
    verified_premises: Sequence[Mapping[str, Any]],
    ledger: TypedDiffLedger,
) -> tuple[bool, str, str]:
    """Deterministic semantic check for the 6 declared diff inference rules."""
    tgt = (target or "").strip().strip("`\"'")
    if operation not in OPERATION_TO_RULE:
        return False, "UNDECLARED_OPERATION", "unsupported_op"
    if not verified_premises:
        return False, "NO_VERIFIED_PREMISES", "refuted"

    if operation == "implements":
        if not tgt or tgt.lower() in CODE_STOPWORDS or len(tgt) < 3:
            return False, "TARGET_TOO_GENERIC_FOR_IMPLEMENTS", "unknown"
        for p in verified_premises:
            fn = p.get("file")
            q = p.get("quote", "")
            if not fn or ledger.file_categories.get(fn) != "src":
                continue
            if not q.lstrip().startswith("+"):
                continue
            body = q.lstrip()[1:]
            defs = _extract_definitions(body)
            if any(_matches_target_symbol(tgt, d) for d in defs) or (
                any(_matches_target_symbol(tgt, d) for d in ledger.add_defs_by_file.get(fn, ()))
                and _matches_target_symbol(tgt, q)
            ):
                return True, "IMPLEMENTS_SYMBOL_RULE_VERIFIED", "verified"
        return False, "NO_SRC_PLUS_DEFINITION_MATCHING_TARGET", "unknown"

    if operation == "adds_test_for":
        if not tgt or tgt.lower() in CODE_STOPWORDS or len(tgt) < 3:
            return False, "TARGET_TOO_GENERIC_FOR_TEST_RULE", "unknown"
        if ledger.category_counts["test"] == 0 and not ledger.add_tests:
            return False, "NO_TESTS_ADDED_IN_DIFF", "refuted"
        for p in verified_premises:
            fn = p.get("file")
            q = p.get("quote", "")
            is_test_src = bool(fn and ledger.file_categories.get(fn) == "test")
            is_test_fn = bool(_extract_tests(q.lstrip("+- ")))
            if (is_test_src or is_test_fn) and q.lstrip().startswith("+") and _matches_target_symbol(tgt, q):
                return True, "ADDS_TEST_FOR_RULE_VERIFIED", "verified"
        return False, "ADDED_TEST_DOES_NOT_CITE_TARGET", "unknown"

    if operation == "removes_support_for":
        if not tgt or tgt.lower() in CODE_STOPWORDS or len(tgt) < 3:
            return False, "TARGET_TOO_GENERIC_FOR_REMOVAL", "unknown"
        in_plus_only = any(_matches_target_symbol(tgt, s) for s in ledger.add_syms)
        in_rem = any(_matches_target_symbol(tgt, s) for s in ledger.rem_syms) or any(
            _matches_target_symbol(tgt, d) for d in ledger.all_defs_removed
        )
        if in_plus_only and not in_rem:
            return False, "TARGET_ONLY_ADDED_NEVER_REMOVED", "refuted"
        for p in verified_premises:
            q = p.get("quote", "")
            if q.lstrip().startswith("-") and _matches_target_symbol(tgt, q) and not in_plus_only:
                return True, "REMOVES_SUPPORT_FOR_RULE_VERIFIED", "verified"
        return False, "NO_NET_REMOVAL_PREMISE_MATCHING_TARGET", "unknown"

    if operation == "configures":
        if not tgt or tgt.lower() in CODE_STOPWORDS or len(tgt) < 2:
            return False, "TARGET_TOO_GENERIC_FOR_CONFIG", "unknown"
        for p in verified_premises:
            fn = p.get("file")
            q = p.get("quote", "")
            if not fn or ledger.file_categories.get(fn) != "config" or _LOCK_GEN_RE.search(fn):
                continue
            body = q.lstrip("+- ")
            keys = _KEY_RE.findall(body) + _FLAG_RE.findall(body)
            if _matches_target_symbol(tgt, fn) or any(_matches_target_symbol(tgt, k) for k in keys) or _matches_target_symbol(tgt, body):
                return True, "CONFIGURES_RULE_VERIFIED", "verified"
        return False, "NO_CONFIG_KEY_PREMISE_MATCHING_TARGET", "unknown"

    if operation == "updates_dependency":
        if not tgt or len(tgt) < 2:
            return False, "TARGET_TOO_GENERIC_FOR_DEPENDENCY", "unknown"
        ver_m = re.search(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", tgt)
        pkg_clean = re.sub(r"\bv?\d+\.\d+(?:\.\d+)?\b", "", tgt).strip() or tgt
        for dep in ledger.dependencies:
            if _matches_target_symbol(dep["package"], pkg_clean) or dep["package"].lower() == pkg_clean.lower():
                if ver_m and dep.get("version"):
                    dep_ver = dep["version"].lstrip("^~>=<!v")
                    if dep_ver != ver_m.group(1):
                        return False, "DEPENDENCY_VERSION_MISMATCH", "refuted"
                return True, "UPDATES_DEPENDENCY_RULE_VERIFIED", "verified"
        for p in verified_premises:
            q = p.get("quote", "")
            dep = _extract_dep(q.lstrip("+- "))
            if dep and (dep[0].lower() == pkg_clean.lower() or _matches_target_symbol(pkg_clean, dep[0])):
                return True, "UPDATES_DEPENDENCY_RULE_VERIFIED", "verified"
        return False, "DEPENDENCY_NOT_MATCHED_IN_MANIFEST", "unknown"

    if operation == "documents":
        if not tgt or tgt.lower() in CODE_STOPWORDS or len(tgt) < 2:
            return False, "TARGET_TOO_GENERIC_FOR_DOCUMENTS", "unknown"
        for p in verified_premises:
            fn = p.get("file")
            q = p.get("quote", "")
            if fn and ledger.file_categories.get(fn) == "docs" and q.lstrip().startswith("+"):
                if _matches_target_symbol(tgt, fn) or _matches_target_symbol(tgt, q):
                    return True, "DOCUMENTS_DOCFILE_RULE_VERIFIED", "verified"
            # Requirement #4: docstring/comment in non-doc file ONLY counts if it literally cites exact target symbol!
            body = q.lstrip("+- ")
            if _DOCSTRING_OR_COMMENT_RE.match(body) and q.lstrip().startswith("+") and _matches_target_symbol(tgt, body):
                return True, "DOCUMENTS_LITERAL_DOCSTRING_RULE_VERIFIED", "verified"
        return False, "NO_DOC_FILE_OR_LITERAL_DOCSTRING_TARGET", "unknown"

    return False, "UNHANDLED_OPERATION", "unknown"


class CodeDiffRuleVerifier:
    """Dedicated `explicit_logic` verifier enforcing CODE_CHANGE_INFERENCE_PROFILE rules."""

    family = "explicit_logic"
    verifier_id = "code_diff_explicit_logic"
    version = "1.0.0"
    guarantee_kind = "deterministic_formal"

    def __init__(self, ledger: TypedDiffLedger, *, use_declared_rules: bool = True) -> None:
        self._ledger = ledger
        self._use_declared_rules = use_declared_rules
        self._base = ExplicitLogicVerifier()

    def verify(self, obligation: VerificationObligation, context: VerifierContext) -> VerifierResult:
        params = {**context.step.parameters, **obligation.parameters}
        if not self._use_declared_rules:
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                reasons=("DECLARED_RULES_DISABLED_BY_ABLATION",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        rule_id = context.step.rule_id or params.get("rule_id")
        rule_ver = context.step.rule_version or params.get("rule_version")
        rule = context.domain_profile.declared_rules.get(str(rule_id)) if rule_id else None
        if rule is None or (rule_ver is not None and rule.version != str(rule_ver)):
            return VerifierResult(
                obligation_id=obligation.obligation_id,
                step_id=context.step.step_id,
                family=self.family,
                status="unknown",
                guarantee_kind=self.guarantee_kind,
                evidence={"rule_id": rule_id, "requested_version": rule_ver},
                reasons=("UNDECLARED_OR_MISMATCHED_RULE",),
                verifier_id=self.verifier_id,
                verifier_version=self.version,
            )

        # First enforce declared-rule structural preconditions & template match via v4 ExplicitLogicVerifier
        base_res = self._base.verify(obligation, context)
        if base_res.status != "verified":
            return base_res

        op = str(params.get("operation") or context.step.operation)
        target = params.get("target")
        prem_dicts = list(params.get("verified_premises") or ())
        ok, code, status = _detect_rule_satisfaction(
            operation=op,
            target=target,
            verified_premises=prem_dicts,
            ledger=self._ledger,
        )
        return VerifierResult(
            obligation_id=obligation.obligation_id,
            step_id=context.step.step_id,
            family=self.family,
            status="verified" if ok else status,
            guarantee_kind=self.guarantee_kind,
            evidence={"rule_id": rule.rule_id, "rule_version": rule.version, "operation": op, "target": target},
            reasons=(code,),
            verifier_id=self.verifier_id,
            verifier_version=self.version,
        )


def build_code_diff_verifier_registry(ledger: TypedDiffLedger, *, use_declared_rules: bool = True) -> VerifierRegistry:
    """Create a fresh isolated VerifierRegistry without modifying DEFAULT_VERIFIER_REGISTRY."""
    reg = VerifierRegistry()
    for fam in DEFAULT_VERIFIER_REGISTRY.families():
        if fam == "explicit_logic":
            reg.register(CodeDiffRuleVerifier(ledger, use_declared_rules=use_declared_rules))
        else:
            reg.register(DEFAULT_VERIFIER_REGISTRY.get(fam))
    return reg


# ---------------------------------------------------------------------------
# Behavior Claim JustificationGraph Builder & Evaluator
# ---------------------------------------------------------------------------

def _detect_hard_factual_contradiction(
    *,
    claim_text: str,
    operation: str,
    target: str | None,
    ledger: TypedDiffLedger,
) -> tuple[bool, str | None]:
    """Detect direct factual contradictions verifiable against the diff register (not mere absence)."""
    txt = claim_text or ""
    tgt = (target or "").strip().strip("`\"'")
    # 1. Dependency version contradiction (e.g., claims X 2.0.0 when diff updates/adds X 1.8.0)
    if operation == "updates_dependency" or any(d["package"].lower() in txt.lower() for d in ledger.dependencies):
        for dep in ledger.dependencies:
            pkg = dep["package"].lower()
            if pkg in txt.lower() and dep.get("version"):
                dep_ver = dep["version"].lstrip("^~>=<!v")
                claimed_vers = re.findall(r"\bv?(\d+\.\d+(?:\.\d+)?)\b", txt)
                if claimed_vers and dep_ver not in claimed_vers:
                    return True, f"dependency_version_mismatch:{dep['package']}:claimed={claimed_vers}:diff={dep_ver}"

    # 2. Explicit added-vs-removed polarity flip for concrete target symbol
    if operation == "removes_support_for" and tgt and len(tgt) >= 3:
        in_add = any(_matches_target_symbol(tgt, s) for s in ledger.add_syms) or any(
            _matches_target_symbol(tgt, d) for d in ledger.all_defs_added
        )
        in_rem = any(_matches_target_symbol(tgt, s) for s in ledger.rem_syms) or any(
            _matches_target_symbol(tgt, d) for d in ledger.all_defs_removed
        )
        if in_add and not in_rem:
            return True, f"polarity_flip_claimed_remove_but_only_added:{tgt}"

    # 3. Explicit unit/integration test claim when diff changes zero test files and zero test functions
    if operation == "adds_test_for" and re.search(r"\b(?:unit\s+tests?|integration\s+tests?|e2e\s+tests?|test\s+suite|test\s+cases?)\b", txt, re.I):
        if ledger.category_counts["test"] == 0 and not ledger.add_tests:
            return True, "claimed_tests_but_zero_tests_in_diff"

    # 4. Explicit documentation file (README / CHANGELOG / docs/) claim when diff changes zero doc files
    if operation == "documents" and re.search(r"\b(?:README(?:\.md)?|CHANGELOG(?:\.md)?|docs?\s+directory|documentation\s+file)\b", txt, re.I):
        if ledger.category_counts["docs"] == 0:
            return True, "claimed_doc_file_but_zero_docs_in_diff"

    # 5. Explicit file count stated inside behavior sentence contradicting ledger
    if m := re.search(r"\b(?:modifies|changes|touches|updates)\s+(\d+)\s+files?\b", txt, re.I):
        n_claimed = int(m.group(1))
        if n_claimed != ledger.files_count and n_claimed > 0:
            return True, f"files_count_mismatch:claimed={n_claimed}:diff={ledger.files_count}"

    return False, None


def verify_behavior_graph(
    *,
    claim_id: str,
    claim_text: str,
    operation: str,
    target: str | None,
    proposed_premises: Sequence[Mapping[str, Any]],
    ledger: TypedDiffLedger,
    repo_tokens: frozenset[str] = frozenset(),
    relevance_config: RelevanceConfig = RelevanceConfig(),
    use_declared_rules: bool = True,
    reread_attempted: bool = False,
) -> dict[str, Any]:
    """Verify one behavioral PR claim via JustificationGraph + code_config + DeclaredRules."""
    op = operation if operation in BEHAVIOR_OPERATIONS else "other_behavior"
    tgt = target.strip() if isinstance(target, str) and target.strip() else None

    if op == "out_of_scope_meta":
        rec = _build_claim_receipt(
            claim_id=claim_id,
            claim_class="behavior",
            operation=op,
            target=tgt,
            premises_verified=(),
            premises_rejected=(),
            rule_id=None,
            rule_version=None,
            eff_kind="hypothesis",
            reread_attempted=reread_attempted,
            status="unverifiable",
        )
        return {
            "claim_id": claim_id,
            "class": "behavior",
            "operation": op,
            "target": tgt,
            "status": "unverifiable",
            "effective_epistemic_kind": "hypothesis",
            "rule_id": None,
            "rule_version": None,
            "verified_premises": [],
            "rejected_premises": [],
            "contradiction_reason": None,
            "reread_attempted": reread_attempted,
            "claim_receipt_sha256": rec,
        }

    # Check hard factual contradiction first
    contra_hit, contra_reason = _detect_hard_factual_contradiction(
        claim_text=claim_text, operation=op, target=tgt, ledger=ledger
    )

    verified_premises: list[dict[str, Any]] = []
    rejected_premises: list[dict[str, Any]] = []

    for idx, prem in enumerate(proposed_premises or ()):
        if not isinstance(prem, Mapping):
            rejected_premises.append({"reason": "malformed_premise"})
            continue
        p_file = str(prem.get("file") or "").strip() or None
        p_sym = str(prem.get("symbol_or_key") or "").strip() or None
        p_quote = str(prem.get("literal_quote") or "").strip()
        p_doc_id = str(prem.get("fact_id") or prem.get("hunk_id") or "").strip() or None
        p_basis = str(prem.get("basis") or EXPLICIT).strip().lower()

        if not CODE_DIFF_EPISTEMIC_PROFILE.admissible(p_basis):
            rejected_premises.append({"reason": f"blocked_epistemic_basis:{p_basis}", "quote": p_quote[:120]})
            continue

        # Locate span first so sign-strict whitespace/indent recovery runs before code_config verify(),
        # and verify ONLY the verbatim diff line against raw diff text (never ungrounded p_sym as assertion.value).
        pref_id = p_doc_id if p_doc_id in ledger.raw_documents else None
        ref = ledger.locate_ref(p_quote, preferred_doc_id=pref_id)
        if ref is None:
            rejected_premises.append({"reason": "span_not_located", "quote": p_quote[:120]})
            continue
        resolved_quote = ref.quote if "\n" not in p_quote else p_quote

        ok_cc, matched_doc_id, cc_codes = ledger.check_literal_quote(
            resolved_quote, resolved_quote, document_id=ref.document_id
        )
        if not ok_cc or matched_doc_id is None:
            rejected_premises.append({"reason": "code_config_rejected:" + ",".join(cc_codes), "quote": p_quote[:120]})
            continue

        # Recover true attribution from ledger chunk if reader omitted or used basename shorthand
        chunk_meta = next((c for c in ledger.chunks if c["chunk_id"] == matched_doc_id), None)
        resolved_file = chunk_meta["file"] if chunk_meta and chunk_meta.get("file") else p_file
        if chunk_meta and chunk_meta.get("file") and p_file:
            actual_l = chunk_meta["file"].lstrip("./").lower()
            claimed_l = p_file.lstrip("./").lower()
            if actual_l != claimed_l and not actual_l.endswith("/" + claimed_l):
                rejected_premises.append({"reason": "wrong_file_attribution", "claimed_file": p_file, "actual_file": chunk_meta["file"]})
                continue

        # Prevent LLM symbol_or_key spoofing from bypassing relevance when symbol is absent from quote/file
        verified_sym = (
            p_sym
            if (p_sym and (p_sym.lower() in resolved_quote.lower() or (resolved_file and p_sym.lower() in resolved_file.lower())))
            else None
        )
        hunk_hdr = (
            _enclosing_hunk_header(ledger.raw_documents[matched_doc_id].text, ref.start)
            if matched_doc_id in ledger.raw_documents
            else ""
        )

        # Check anti-N2 relevance guard
        rel_ok, rel_reason = check_premise_relevance(
            claim_text=claim_text,
            operation=op,
            target=tgt,
            premise_file=resolved_file,
            premise_symbol=verified_sym,
            premise_quote=resolved_quote,
            ledger=ledger,
            repo_tokens=repo_tokens,
            config=relevance_config,
            hunk_header=hunk_hdr,
        )
        if not rel_ok:
            rejected_premises.append({"reason": f"rejected_irrelevant_premise:{rel_reason}", "file": resolved_file, "quote": resolved_quote[:120]})
            continue

        verified_premises.append(
            {
                "node_id": f"{claim_id}::p{idx}",
                "file": resolved_file,
                "symbol_or_key": verified_sym,
                "quote": resolved_quote,
                "basis": p_basis,
                "ref": ref,
                "relevance_reason": rel_reason,
            }
        )

    if contra_hit:
        rec = _build_claim_receipt(
            claim_id=claim_id,
            claim_class="behavior",
            operation=op,
            target=tgt,
            premises_verified=verified_premises,
            premises_rejected=rejected_premises,
            rule_id=None,
            rule_version=None,
            eff_kind="hypothesis",
            reread_attempted=reread_attempted,
            status="premise_contradicted",
        )
        return {
            "claim_id": claim_id,
            "class": "behavior",
            "operation": op,
            "target": tgt,
            "status": "premise_contradicted",
            "effective_epistemic_kind": "hypothesis",
            "rule_id": None,
            "rule_version": None,
            "verified_premises": [{"file": p["file"], "symbol_or_key": p["symbol_or_key"], "relevance_reason": p["relevance_reason"]} for p in verified_premises],
            "rejected_premises": rejected_premises,
            "contradiction_reason": contra_reason,
            "reread_attempted": reread_attempted,
            "claim_receipt_sha256": rec,
        }

    # Ungrounded/rejected reader quotes leave zero verified diff premises -> no_premises_in_diff
    # (premise_contradicted is reserved strictly for hard diff contradictions and refuted rules).
    if not verified_premises:
        rec = _build_claim_receipt(
            claim_id=claim_id,
            claim_class="behavior",
            operation=op,
            target=tgt,
            premises_verified=(),
            premises_rejected=rejected_premises,
            rule_id=None,
            rule_version=None,
            eff_kind="hypothesis",
            reread_attempted=reread_attempted,
            status="no_premises_in_diff",
        )
        return {
            "claim_id": claim_id,
            "class": "behavior",
            "operation": op,
            "target": tgt,
            "status": "no_premises_in_diff",
            "effective_epistemic_kind": "hypothesis",
            "rule_id": None,
            "rule_version": None,
            "verified_premises": [],
            "rejected_premises": rejected_premises,
            "contradiction_reason": None,
            "reread_attempted": reread_attempted,
            "claim_receipt_sha256": rec,
        }

    # Build native v4 JustificationGraph and evaluate via evaluate_response()
    rule_pair = OPERATION_TO_RULE.get(op)
    rule_id, rule_ver = rule_pair if (use_declared_rules and rule_pair is not None) else (None, None)
    conc_pred = op if any(p.name == op for p in CODE_CHANGE_INFERENCE_VOCABULARY.predicates) else "other_behavior"
    conc_kind = "derivation" if (rule_id is not None) else "hypothesis"

    claims_nodes: dict[str, ClaimNode] = {}
    for p in verified_premises:
        claims_nodes[p["node_id"]] = ClaimNode(
            claim_id=p["node_id"],
            kind="extraction",
            subject=PR_ENTITY,
            predicate="diff_premise",
            value=p["quote"],
            text=p["quote"],
            assertion_kind="fact",
            evidence=(p["ref"],),
            condition_status="unconditional",
            structural_reliability=1.0,
        )

    conc_id = f"{claim_id}::conclusion"
    claims_nodes[conc_id] = ClaimNode(
        claim_id=conc_id,
        kind=conc_kind,
        subject=PR_ENTITY,
        predicate=conc_pred,
        value=tgt or claim_text[:120],
        text=claim_text,
        assertion_kind="proposal" if conc_kind == "hypothesis" else "fact",
        condition_status="unconditional",
    )

    premise_summaries = [{"file": p["file"], "symbol_or_key": p["symbol_or_key"], "quote": p["quote"]} for p in verified_premises]
    extra_steps: list[InferenceStep] = []
    step_premises: list[str] = [p["node_id"] for p in verified_premises]
    if rule_id is not None:
        # Attach native 2-hop RelationCompositionRule bridge edges when a file-attributed premise backs a relational rule
        rel_spec_map = {
            "adds_test_for": ("pr_adds_test_file", "test_cites_symbol", "rel.adds_test_for", "test"),
            "configures": ("pr_changes_config_file", "config_sets_key", "rel.configures", "config"),
            "documents": ("pr_changes_doc_file", "doc_mentions_term", "rel.documents", "docs"),
        }
        bridge_premise = None
        if op in rel_spec_map:
            req_cat = rel_spec_map[op][3]
            bridge_premise = next(
                (
                    p
                    for p in verified_premises
                    if p.get("file")
                    and ledger.file_categories.get(p["file"]) == req_cat
                    and not (req_cat == "config" and _LOCK_GEN_RE.search(p["file"]))
                ),
                None,
            )
        can_use_rel = bridge_premise is not None
        obs: list[VerificationObligation] = [
            VerificationObligation(
                obligation_id=f"{claim_id}::ob_rule",
                family="explicit_logic",
                description=f"Verify {rule_id}@{rule_ver}",
                critical=True,
                parameters={
                    "rule_id": rule_id,
                    "rule_version": rule_ver,
                    "operation": op,
                    "target": tgt,
                    "verified_premises": premise_summaries,
                },
            )
        ]
        if op in rel_spec_map and can_use_rel and bridge_premise is not None and tgt:
            l_pred, r_pred, rel_rule_id, _ = rel_spec_map[op]
            b_file = bridge_premise["file"]
            f_ref = ledger.locate_ref(b_file) or bridge_premise["ref"]
            e0_id = f"{claim_id}::rel_left"
            e1_id = f"{claim_id}::rel_right"
            claims_nodes[e0_id] = ClaimNode(
                claim_id=e0_id,
                kind="extraction",
                subject=PR_ENTITY,
                predicate=l_pred,
                value=b_file,
                text=b_file,
                assertion_kind="fact",
                evidence=(f_ref,),
                condition_status="unconditional",
                structural_reliability=1.0,
            )
            claims_nodes[e1_id] = ClaimNode(
                claim_id=e1_id,
                kind="extraction",
                subject=b_file,
                predicate=r_pred,
                value=tgt,
                text=bridge_premise["quote"],
                assertion_kind="fact",
                evidence=(bridge_premise["ref"],),
                condition_status="unconditional",
                structural_reliability=1.0,
            )
            extra_steps.append(
                InferenceStep(
                    step_id=f"{claim_id}::rel_step",
                    premises=(e0_id, e1_id),
                    conclusion_id=conc_id,
                    operation="relation_composition",
                    rule_id=rel_rule_id,
                    rule_version="1.0",
                    obligations=(
                        VerificationObligation(
                            obligation_id=f"{claim_id}::ob_rel",
                            family="relations",
                            description=f"Verify {rel_rule_id}@1.0",
                            critical=True,
                            parameters={"rule_id": rel_rule_id},
                        ),
                    ),
                )
            )
            # Align conclusion value with composed target entity/token
            claims_nodes[conc_id] = ClaimNode(
                claim_id=conc_id,
                kind=conc_kind,
                subject=PR_ENTITY,
                predicate=conc_pred,
                value=tgt,
                text=claim_text,
                assertion_kind="fact",
                condition_status="unconditional",
            )
        step = InferenceStep(
            step_id=f"{claim_id}::step",
            premises=tuple(step_premises),
            conclusion_id=conc_id,
            operation="rule_application",
            rule_id=rule_id,
            rule_version=rule_ver,
            obligations=tuple(obs),
        )
    else:
        step = InferenceStep(
            step_id=f"{claim_id}::step",
            premises=tuple(step_premises),
            conclusion_id=conc_id,
            operation=op,
            obligations=(
                VerificationObligation(
                    obligation_id=f"{claim_id}::ob_unruled",
                    family="explicit_logic",
                    description="Unruled hypothesis step (no DeclaredRule)",
                    critical=True,
                    parameters={"operation": op, "target": tgt},
                ),
            ),
        )

    graph = JustificationGraph(claims=claims_nodes, steps=(step, *extra_steps), conclusion_id=conc_id)
    registry = build_code_diff_verifier_registry(ledger, use_declared_rules=use_declared_rules)
    assessment = evaluate_response(
        graph,
        evidence=ledger.raw_documents,
        domain_profile=CODE_CHANGE_INFERENCE_PROFILE,
        verifier_registry=registry,
        budget=VerificationBudget(max_steps=32, max_depth=4, max_semantic_calls=0),
    )

    eff_kind = assessment.effective_epistemic_kind
    if assessment.assessment == "refuted":
        final_status = "premise_contradicted"
        refuted_reasons = [",".join(c.reasons) for c in assessment.completed_checks if c.status == "refuted"]
        contra_out = refuted_reasons[0] if refuted_reasons else "rule_precondition_refuted"
    elif assessment.assessment == "supported" and eff_kind == "derivation" and rule_id is not None:
        final_status = "derived_supported"
        contra_out = None
    else:
        # Enforces invariant: verified premises without valid declared rule stay hypothesis_with_verified_premises
        eff_kind = effective_epistemic_kind("hypothesis", [c.kind for c in claims_nodes.values() if c.claim_id != conc_id])
        final_status = "hypothesis_with_verified_premises"
        contra_out = None

    rec = _build_claim_receipt(
        claim_id=claim_id,
        claim_class="behavior",
        operation=op,
        target=tgt,
        premises_verified=verified_premises,
        premises_rejected=rejected_premises,
        rule_id=rule_id if final_status == "derived_supported" else None,
        rule_version=rule_ver if final_status == "derived_supported" else None,
        eff_kind=eff_kind,
        reread_attempted=reread_attempted,
        status=final_status,
    )
    return {
        "claim_id": claim_id,
        "class": "behavior",
        "operation": op,
        "target": tgt,
        "status": final_status,
        "effective_epistemic_kind": eff_kind,
        "assessment": assessment.assessment,
        "assessment_digest": assessment.assessment_digest,
        "rule_id": rule_id if final_status == "derived_supported" else None,
        "rule_version": rule_ver if final_status == "derived_supported" else None,
        "verified_premises": [{"file": p["file"], "symbol_or_key": p["symbol_or_key"], "relevance_reason": p["relevance_reason"]} for p in verified_premises],
        "rejected_premises": rejected_premises,
        "contradiction_reason": contra_out,
        "reread_attempted": reread_attempted,
        "claim_receipt_sha256": rec,
    }


def _build_claim_receipt(
    *,
    claim_id: str,
    claim_class: str,
    operation: str | None,
    target: str | None,
    premises_verified: Sequence[Mapping[str, Any]],
    premises_rejected: Sequence[Mapping[str, Any]],
    rule_id: str | None,
    rule_version: str | None,
    eff_kind: str,
    reread_attempted: bool,
    status: str,
) -> str:
    return digest(
        canonical(
            {
                "claim_id": claim_id,
                "class": claim_class,
                "operation": operation,
                "target": target,
                "verified_premises": [
                    {"file": p.get("file"), "symbol_or_key": p.get("symbol_or_key"), "quote": p.get("quote")}
                    for p in premises_verified
                ],
                "rejected_reasons": [r.get("reason") for r in premises_rejected],
                "rule_id": rule_id,
                "rule_version": rule_version,
                "effective_epistemic_kind": eff_kind,
                "reread_attempted": reread_attempted,
                "status": status,
            }
        )
    )


# ---------------------------------------------------------------------------
# I5 Structural Rules & PR Verdict Engine
# ---------------------------------------------------------------------------

def evaluate_i5_structural_signals(
    pr: Mapping[str, Any],
    ledger: TypedDiffLedger,
    *,
    thresholds: VerdictThresholds = VerdictThresholds(),
) -> list[dict[str, Any]]:
    """Evaluate auditable I5 structural signals, each backed by explicit ledger counts."""
    signals: list[dict[str, Any]] = []
    title = str(pr.get("title") or "")
    body = str(pr.get("body") or "")
    full_text = f"{title}\n\n{body}" if title else body
    files = list(ledger.files)

    # 1. D_doc_only_but_code
    checked = re.findall(r"(?m)^\s*-\s*\[[xX]\]\s*(.+)$", body)
    types = [c for c in checked if re.search(r"(bug fix|new feature|breaking change|documentation update)", c, re.I)]
    code_non_doc = [f for f in files if _CODE_EXT_RE.search(f) and ledger.file_categories.get(f) != "docs"]
    if types and all(re.search(r"documentation", c, re.I) for c in types) and code_non_doc:
        signals.append(
            {
                "signal": "D_doc_only_but_code",
                "rule_id": "code.doc_only_code_contradiction",
                "rule_version": "1.0",
                "count": len(code_non_doc),
            }
        )

    # 2. P_placeholder_large_diff
    stripped = re.sub(r"(?m)^\s*-\s*\[[ xX]\].*$", " ", re.sub(r"(🤖 Generated with.*|Co-Authored-By:.*|<!--.*?-->)", " ", body, flags=re.I | re.S))
    body_words = len(re.findall(r"[A-Za-z]{3,}", re.sub(r"(?m)^#+.*$", " ", stripped)))
    title_words = len(re.findall(r"[A-Za-z]{3,}", title))
    churn = (ledger.adds_count or 0) + (ledger.dels_count or 0)
    if body_words < thresholds.i5_placeholder_max_words and title_words < thresholds.i5_placeholder_max_title_words and churn >= thresholds.i5_placeholder_min_churn:
        signals.append(
            {
                "signal": "P_placeholder_large_diff",
                "rule_id": "code.placeholder_for_large_diff",
                "rule_version": "1.0",
                "body_words": body_words,
                "churn": churn,
            }
        )

    # 3. U_severe_understated_files
    # Detect explicit file path mentions in description body
    mentioned_paths = re.findall(
        r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|json|md|yml|yaml|toml|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|sh|sql))(?![\w])",
        full_text,
        re.I,
    )
    enumerated = list(dict.fromkeys(re.sub(r"^\./+", "", p).lower() for p in mentioned_paths if not p.lower().startswith(("http", "www."))))
    sig_files = [f for f in files if not _LOCK_GEN_RE.search(f)]
    if len(enumerated) >= thresholds.i5_understated_min_enumerated and len(sig_files) >= 4:
        unmentioned = [
            f
            for f in sig_files
            if f.lower() not in enumerated
            and f.rsplit("/", 1)[-1].lower() not in full_text.lower()
            and not any(f.lower().endswith("/" + ep) for ep in enumerated if "/" in ep)
        ]
        share = len(unmentioned) / max(1, len(sig_files))
        if share >= thresholds.i5_understated_min_share:
            signals.append(
                {
                    "signal": "U_severe_understated_files",
                    "rule_id": "code.severe_scope_understatement",
                    "rule_version": "1.0",
                    "share": round(share, 4),
                    "unmentioned_count": len(unmentioned),
                    "total_sig_files": len(sig_files),
                }
            )
    return signals


def compute_pr_verdict(
    *,
    pr: Mapping[str, Any],
    ledger: TypedDiffLedger,
    literal_claims: Sequence[Mapping[str, Any]],
    behavior_results: Sequence[Mapping[str, Any]],
    draft_rejected: Mapping[str, int],
    reader_rejected: Sequence[Mapping[str, Any]] = (),
    error_escalated_count: int = 0,
    read_errors: Sequence[Any] = (),
    thresholds: VerdictThresholds = VerdictThresholds(),
    prompt_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Compute final PR verdict in {CONSISTENT, REVIEW, INCONSISTENT, ESCALATE} with SHA-256 receipt."""
    state_counts = {st: 0 for st in CLAIM_STATES}
    for lc in literal_claims:
        st = lc["status"]
        if st in state_counts:
            state_counts[st] += 1
    for br in behavior_results:
        st = br["status"]
        if st in state_counts:
            state_counts[st] += 1

    # Fail-closed ESCALATE precedes everything (never CONSISTENT)
    if ledger.truncated or error_escalated_count >= 1 or read_errors:
        esc_reasons = []
        if ledger.truncated:
            esc_reasons.append("diff_facts_truncated")
        if error_escalated_count >= 1:
            esc_reasons.append(f"error_escalated {error_escalated_count}")
        if read_errors:
            esc_reasons.append(f"read_errors {len(read_errors)}")
        pr_receipt = _build_pr_receipt(
            pr_id=ledger.pr_id,
            verdict="ESCALATE",
            reasons=esc_reasons,
            literal_claims=literal_claims,
            behavior_results=behavior_results,
            i5_signals=(),
            reader_rejected=reader_rejected,
            ledger=ledger,
            prompt_digests=prompt_digests,
        )
        return {
            "verdict": "ESCALATE",
            "reasons": esc_reasons,
            "state_counts": state_counts,
            "i5_signals": [],
            "reader_rejected": list(reader_rejected),
            "pr_receipt_sha256": pr_receipt,
        }

    tot_lit = state_counts["literal_supported"] + state_counts["literal_unsupported"]
    tot_beh_checkable = (
        state_counts["derived_supported"]
        + state_counts["hypothesis_with_verified_premises"]
        + state_counts["no_premises_in_diff"]
    )
    tot_checkable = tot_lit + tot_beh_checkable + state_counts["premise_contradicted"]
    i5_signals = evaluate_i5_structural_signals(pr, ledger, thresholds=thresholds) if thresholds.include_i5 else []

    inc_reasons: list[str] = []
    # I1: empty diff with checkable claims
    if ledger.files_count == 0 and (tot_checkable > 0 or state_counts["unverifiable"] > 0):
        inc_reasons.append("I1_empty_diff")
    # I2: direct factual/premise contradiction
    if state_counts["premise_contradicted"] >= 1:
        inc_reasons.append(f"I2_premise_contradicted:{state_counts['premise_contradicted']}")
    # I3: phantom behavioral claims (no_premises_in_diff even after targeted reread)
    n_phan = state_counts["no_premises_in_diff"]
    if n_phan >= thresholds.k_phantom and tot_beh_checkable > 0:
        frac_phan = n_phan / tot_beh_checkable
        if frac_phan >= thresholds.theta_phantom and (
            tot_beh_checkable >= 2
            or (thresholds.singleton_phantom_inconsistent and state_counts["literal_supported"] == 0)
        ):
            inc_reasons.append(f"I3_phantom_behavior:{n_phan}/{tot_beh_checkable}")
    # I4: literal unsupported concentration
    n_lunsup = state_counts["literal_unsupported"]
    if n_lunsup >= thresholds.k_lit and tot_lit > 0 and (n_lunsup / tot_lit) >= thresholds.theta_lit:
        inc_reasons.append(f"I4_literal_unsupported:{n_lunsup}/{tot_lit}")
    # I5: auditable structural contradictions (toggleable for Ablation (e))
    for sig in i5_signals:
        inc_reasons.append(f"I5_{sig['signal']}")

    if inc_reasons:
        verdict = "INCONSISTENT"
        reasons = inc_reasons
    else:
        rev_reasons: list[str] = []
        if thresholds.review_on_single_unsupported_or_phantom:
            if n_phan >= 1:
                rev_reasons.append(f"REVIEW_partial_phantom:{n_phan}/{tot_beh_checkable}")
            if n_lunsup >= 1:
                rev_reasons.append(f"REVIEW_partial_literal_unsupported:{n_lunsup}/{tot_lit}")
        if sum(draft_rejected.values()) > 0 and (state_counts["literal_supported"] + state_counts["derived_supported"] == 0):
            rev_reasons.append("REVIEW_draft_rejected_no_support")
        if (
            thresholds.review_on_all_unruled_hypothesis
            and tot_beh_checkable >= 2
            and state_counts["hypothesis_with_verified_premises"] == tot_beh_checkable
            and state_counts["literal_supported"] == 0
        ):
            rev_reasons.append("REVIEW_all_hypothesis_without_rules")
        if rev_reasons:
            verdict = "REVIEW"
            reasons = rev_reasons
        else:
            verdict = "CONSISTENT"
            reasons = []

    pr_receipt = _build_pr_receipt(
        pr_id=ledger.pr_id,
        verdict=verdict,
        reasons=reasons,
        literal_claims=literal_claims,
        behavior_results=behavior_results,
        i5_signals=i5_signals,
        reader_rejected=reader_rejected,
        ledger=ledger,
        prompt_digests=prompt_digests,
    )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "state_counts": state_counts,
        "i5_signals": i5_signals,
        "reader_rejected": list(reader_rejected),
        "pr_receipt_sha256": pr_receipt,
    }


def _build_pr_receipt(
    *,
    pr_id: str,
    verdict: str,
    reasons: Sequence[str],
    literal_claims: Sequence[Mapping[str, Any]],
    behavior_results: Sequence[Mapping[str, Any]],
    i5_signals: Sequence[Mapping[str, Any]],
    reader_rejected: Sequence[Mapping[str, Any]] = (),
    ledger: TypedDiffLedger,
    prompt_digests: Mapping[str, str] | None,
) -> str:
    return digest(
        canonical(
            {
                "pr_id": pr_id,
                "verdict": verdict,
                "reasons": list(reasons),
                "literal_receipts": [lc.get("claim_receipt_sha256") for lc in literal_claims],
                "behavior_receipts": [br.get("claim_receipt_sha256") for br in behavior_results],
                "i5_signals": list(i5_signals),
                "reader_rejected": list(reader_rejected),
                "ledger_digest": ledger.ledger_digest,
                "truncated": ledger.truncated,
                "truncated_fields": ledger.truncated_fields,
                "domain_profile_fingerprint": CODE_CHANGE_INFERENCE_PROFILE.fingerprint(),
                "stopwords_sha256": CODE_STOPWORDS_SHA256,
                "prompt_digests": dict(sorted((prompt_digests or {}).items())),
            }
        )
    )


__all__ = [
    "BEHAVIOR_OPERATIONS",
    "CLAIM_STATES",
    "CODE_CHANGE_INFERENCE_PROFILE",
    "CODE_CHANGE_INFERENCE_VOCABULARY",
    "CODE_DIFF_EPISTEMIC_PROFILE",
    "CODE_STOPWORDS",
    "CODE_STOPWORDS_SHA256",
    "CodeDiffRuleVerifier",
    "DECLARED_CODE_RULES",
    "DiffFactRecord",
    "OPERATION_TO_RULE",
    "PR_ENTITY",
    "PR_VERDICTS",
    "RELATION_CODE_RULES",
    "RelevanceConfig",
    "TypedDiffLedger",
    "VerdictThresholds",
    "build_code_diff_verifier_registry",
    "build_typed_diff_ledger",
    "check_premise_relevance",
    "classify_file_category",
    "compute_pr_verdict",
    "evaluate_i5_structural_signals",
    "is_compound_identifier",
    "repo_identifier_tokens",
    "specific_tokens",
    "split_code_tokens",
    "verify_behavior_graph",
]
