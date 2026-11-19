"""Verificador determinista de afirmaciones de la descripción de un PR contra su diff.

Diseño (fijado ANTES de evaluar, a partir de la taxonomía de PR-MCI, sin mirar las notas de los PRs evaluados):
- Se extraen solo afirmaciones comprobables de la descripción (no se interpreta prosa libre):
  F  ficheros/rutas mencionados en contexto de cambio  -> deben estar en la lista de ficheros cambiados
  I  identificadores entre `comillas invertidas` en contexto de cambio -> deben aparecer en líneas +/- del diff
  E  la descripción afirma cambios y el diff está vacío -> contradicción (Phantom Changes)
  D  casilla "[x] Documentation update" como único tipo marcado y el diff toca código -> contradicción (Intent mismatch)
  P  descripción de relleno (sin contenido propio tras quitar plantilla, pie de Claude y casillas) -> Placeholder
  U  la descripción enumera ficheros y la mayor parte del churn está en ficheros no mencionados -> Scope Understated
- Cada afirmación soportada se valida con deltoryx-gate verify(): Assertion con la cita literal de la línea del diff que la
  respalda; PASS = soportada con evidencia literal. Sin cita posible -> 'unverified' (el gate falla cerrado).
- Veredicto del PR (umbrales fijados a priori): INCONSISTENT si E o D o P, o si hay >=2 afirmaciones sin evidencia y son
  >=50 % de las afirmaciones F+I, o si U >= 0.5 con >=2 ficheros enumerados. En otro caso CONSISTENT.
"""
from __future__ import annotations
import hashlib, json, re

from claimlayer import Assertion, Ledger, Source, verify

CHANGE_VERBS = r"(add|added|adds|create|created|creates|implement|implemented|implements|update|updated|updates|modify|modified|modifies|change|changed|changes|remove|removed|removes|delete|deleted|deletes|rename|renamed|renames|refactor|refactored|refactors|fix|fixed|fixes|replace|replaced|replaces|move|moved|moves|introduce|introduced|introduces|extend|extended|extends|bump|bumped|upgrade|upgraded|migrate|migrated|convert|converted|enable|enabled|disable|disabled|new)"
CHANGE_HEADINGS = r"(changes|changes made|summary|what changed|modifications|implementation|files changed|key changes|what's changed|details)"
FOOTER = re.compile(r"(🤖 Generated with.*|Co-Authored-By:.*|<!--.*?-->)", re.I | re.S)
PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|json|md|mdx|yml|yaml|toml|cfg|ini|txt|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|hpp|cs|sh|zig|sql|html|css|scss|vue|svelte|lock|env|example|dockerfile|gradle|xml|proto|tf))(?![\w])", re.I)
TICK_RE = re.compile(r"`([^`\n]{2,80})`")
LOCK_GEN = re.compile(r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|uv\.lock|\.snap$|\.min\.js$)", re.I)
CODE_EXT = re.compile(r"\.(py|js|jsx|ts|tsx|mjs|cjs|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|hpp|cs|sh|zig|sql|vue|svelte)$", re.I)


def strip_template(body: str) -> str:
    b = FOOTER.sub(" ", body)
    b = re.sub(r"(?m)^\s*-\s*\[[ xX]\].*$", " ", b)          # casillas
    return b


def content_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", re.sub(r"(?m)^#+.*$", " ", text)))


def change_context_units(body: str):
    """Frases/viñetas que afirman cambios: contienen un verbo de cambio o cuelgan de un encabezado de cambios."""
    units, under = [], False
    for line in strip_template(body).splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^#{1,6}\s", s) or re.match(r"^\*\*[^*]+\*\*:?$", s):
            under = bool(re.search(CHANGE_HEADINGS, s, re.I))
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z`])", s):
            if under and re.match(r"^[-*•]|\d+\.", s) or re.search(r"\b" + CHANGE_VERBS + r"\b", sent, re.I):
                units.append(sent)
    return units


def diff_lines(patch: str):
    plus = [l[1:] for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++")]
    minus = [l[1:] for l in patch.splitlines() if l.startswith("-") and not l.startswith("---")]
    return plus, minus


def gate_check(value: str, quote: str, source_text: str, sid: str) -> str:
    v = verify([Assertion(subject="pull_request", predicate="contains_change", value=value, source_id=sid, quote=quote)],
               Ledger(sources=[Source(id=sid, text=source_text, tier="authoritative")]))
    return v.status


def check_pr(pr: dict) -> dict:
    body, patch, files = pr["body"], pr["patch"] or "", pr["files"]
    plus, minus = diff_lines(patch)
    changed_text = "\n".join(plus + minus)
    file_list_text = "\n".join(files)
    basenames = {f.rsplit("/", 1)[-1].lower() for f in files}
    claims = []
    units = change_context_units(body)
    seen = set()
    for u in units:
        for m in PATH_RE.finditer(u):
            p = m.group(1).strip("./")
            key = ("F", p.lower())
            if key in seen or p.lower().startswith(("http", "www.")):
                continue
            seen.add(key)
            hit = next((f for f in files if f.lower() == p.lower() or f.lower().endswith("/" + p.lower()) or f.rsplit("/", 1)[-1].lower() == p.rsplit("/", 1)[-1].lower()), None)
            status = "supported" if hit and gate_check(hit.rsplit("/", 1)[-1], hit, file_list_text, "changed_files") == "PASS" else "unverified"
            claims.append({"kind": "F", "value": p, "context": u[:160], "evidence": hit, "status": status})
        for m in TICK_RE.finditer(u):
            tok = m.group(1).strip()
            if PATH_RE.fullmatch(tok) or len(tok) < 3 or " " in tok and len(tok.split()) > 4:
                continue
            key = ("I", tok)
            if key in seen:
                continue
            seen.add(key)
            ev = next((l for l in plus + minus if tok in l), None)
            if ev is not None:
                q = ev.strip()[:300]
                status = "supported" if gate_check(tok, q, changed_text, "diff_changed_lines") == "PASS" else "unverified"
            else:
                status = "unverified"
            claims.append({"kind": "I", "value": tok, "context": u[:160], "evidence": (ev or "")[:160] or None, "status": status})
    signals = {}
    n_lines = len(plus) + len(minus)
    if units and (not files or n_lines == 0):
        signals["E_empty_diff"] = True
    checked = re.findall(r"(?m)^\s*-\s*\[[xX]\]\s*(.+)$", body)
    types = [c for c in checked if re.search(r"(bug fix|new feature|breaking change|documentation update)", c, re.I)]
    if types and all(re.search(r"documentation", c, re.I) for c in types) and any(CODE_EXT.search(f) for f in files):
        signals["D_doc_only_but_code"] = [f for f in files if CODE_EXT.search(f)][:5]
    if content_words(strip_template(body)) < 8:
        signals["P_placeholder"] = True
    enumerated = [c for c in claims if c["kind"] == "F"]
    if len(enumerated) >= 2 and files:
        mentioned = {c["evidence"] for c in enumerated if c["evidence"]}
        sig = [f for f in files if not LOCK_GEN.search(f)]
        unmentioned = [f for f in sig if f not in mentioned and f.rsplit("/", 1)[-1].lower() not in body.lower()]
        share = len(unmentioned) / max(1, len(sig))
        if share >= 0.5:
            signals["U_understated_files"] = {"share": round(share, 2), "examples": unmentioned[:5]}
    fi = [c for c in claims if c["kind"] in ("F", "I")]
    unsup = [c for c in fi if c["status"] != "supported"]
    if len(unsup) >= 2 and len(unsup) / max(1, len(fi)) >= 0.5:
        signals["phantom_claims"] = len(unsup)
    verdict = "INCONSISTENT" if signals else "CONSISTENT"
    rec = {"pr_id": pr["pr_id"], "url": pr["url"], "verdict": verdict, "signals": signals,
           "claims": claims, "n_claims": len(fi), "n_supported": len(fi) - len(unsup),
           "files_changed": len(files), "lines_changed": n_lines}
    rec["receipt_sha256"] = hashlib.sha256(json.dumps({k: rec[k] for k in ("pr_id", "verdict", "signals", "claims")}, sort_keys=True, ensure_ascii=False).encode()
                                           + hashlib.sha256((body + "\x00" + patch + "\x00" + "|".join(files)).encode()).digest()).hexdigest()
    return rec
