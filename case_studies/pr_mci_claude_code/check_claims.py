"""Deterministic PR description claim checker (v2 + post-review bug-fixed mode).

Provides:
- `check_pr(pr, *, evidence_profile="legal_prose")`: original v2 checker (kept intact).
- `check_pr_corrected(pr, *, evidence_profile="legal_prose")`: post-review bug-fixed
  variant applying only Task 1 bug fixes (word-boundary identifier match, non-collapsing
  directory paths vs basenames, centered long-line quotes, consistent D diagnostic filter).
"""
from __future__ import annotations

import hashlib
import json
import re

from claimlayer import Assertion, Ledger, Source, verify

CHANGE_VERBS = r"(add|added|adds|create|created|creates|implement|implemented|implements|update|updated|updates|modify|modified|modifies|change|changed|changes|remove|removed|removes|delete|deleted|deletes|rename|renamed|renames|refactor|refactored|refactors|fix|fixed|fixes|replace|replaced|replaces|move|moved|moves|introduce|introduced|introduces|extend|extended|extends|bump|bumped|upgrade|upgraded|migrate|migrated|convert|converted|enable|enabled|disable|disabled|new)"
CHANGE_HEADINGS = r"(changes|changes made|summary|what changed|modifications|implementation|files changed|key changes|what's changed|details)"
FOOTER = re.compile(r"(🤖 Generated with.*|Co-Authored-By:.*|<!--.*?-->)", re.I | re.S)
PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|js|jsx|ts|tsx|mjs|cjs|json|md|mdx|yml|yaml|toml|cfg|ini|txt|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|hpp|cs|sh|zig|sql|html|css|scss|vue|svelte|lock|env|example|dockerfile|gradle|xml|proto|tf))(?![\w])", re.I)
TICK_RE = re.compile(r"`([^`\n]{2,80})`")
LOCK_GEN = re.compile(r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|uv\.lock|\.snap$|\.min\.js$)", re.I)
CODE_EXT = re.compile(r"\.(py|js|jsx|ts|tsx|mjs|cjs|rs|go|java|kt|swift|rb|php|c|h|cc|cpp|hpp|cs|sh|zig|sql|vue|svelte)$", re.I)

VERSION = "v2"
CODE_TOKEN = re.compile(r"^[A-Za-z_][\w]*(?:(?:\.|::|->)[A-Za-z_][\w]*)*(?:\(\))?$")


def strip_template(body: str) -> str:
    b = FOOTER.sub(" ", body)
    b = re.sub(r"(?m)^\s*-\s*\[[ xX]\].*$", " ", b)
    return b


def content_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]{3,}", re.sub(r"(?m)^#+.*$", " ", text)))


def change_context_units(body: str):
    """Sentences/bullets asserting changes (contain a change verb or appear under a changes heading)."""
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


def gate_check(value: str, quote: str, source_text: str, sid: str, *, evidence_profile: str = "legal_prose") -> str:
    v = verify(
        [Assertion(subject="pull_request", predicate="contains_change", value=value, source_id=sid, quote=quote)],
        Ledger(sources=[Source(id=sid, text=source_text, tier="authoritative", evidence_profile=evidence_profile)]),
    )
    return v.status


def _has_identifier(line: str, sym: str) -> re.Match[str] | None:
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(sym) + r"(?![A-Za-z0-9_])", line)


def _centered_quote(line: str, sym: str, fallback: str, max_len: int = 300) -> str:
    s = line.strip()
    if len(s) <= max_len:
        return s
    m = _has_identifier(s, sym) or _has_identifier(s, fallback)
    if not m:
        return s[:max_len]
    start = max(0, m.start() - 100)
    end = min(len(s), start + max_len)
    if start > 0 and s[start - 1].isalnum():
        ws = s.find(" ", start, m.start())
        if ws != -1 and ws + 1 <= m.start():
            start = ws + 1
    if end < len(s) and s[end].isalnum():
        ws = s.rfind(" ", m.end(), end)
        if ws != -1 and ws > m.end():
            end = ws
    return s[start:end].strip() or s[max(0, m.start() - 100) : min(len(s), max(0, m.start() - 100) + max_len)]


def _check_pr_impl(pr: dict, *, corrected: bool = False, evidence_profile: str = "legal_prose") -> dict:
    body = (pr.get("title", "") + "\n\n" + pr["body"]) if pr.get("title") else pr["body"]
    patch, files = pr["patch"] or "", pr["files"]
    plus, minus = diff_lines(patch)
    context = [l[1:] for l in patch.splitlines() if l.startswith(" ")] + [l for l in patch.splitlines() if l.startswith("@@")]
    changed_text = patch
    file_list_text = "\n".join(files)
    claims = []
    units = change_context_units(body)
    seen = set()
    for u in units:
        for m in PATH_RE.finditer(u):
            p = re.sub(r"^\./+", "", m.group(1)) if corrected else m.group(1).strip("./")
            if "/" not in p and re.fullmatch(r"[A-Z][\w]*\.js", p):
                continue
            key = ("F", p.lower())
            if key in seen or p.lower().startswith(("http", "www.")):
                continue
            seen.add(key)
            pl = p.lower()
            if corrected:
                if "/" in pl:
                    hit = next((f for f in files if f.lower() == pl or f.lower().endswith("/" + pl)), None)
                else:
                    hit = next((f for f in files if f.rsplit("/", 1)[-1].lower() == pl), None)
            else:
                hit = next(
                    (
                        f
                        for f in files
                        if f.lower() == pl
                        or f.lower().endswith("/" + pl)
                        or f.rsplit("/", 1)[-1].lower() == p.rsplit("/", 1)[-1].lower()
                    ),
                    None,
                )
            val_f = hit.rsplit("/", 1)[-1] if hit else ""
            status = (
                "supported"
                if hit and gate_check(val_f, hit, file_list_text, "changed_files", evidence_profile=evidence_profile) == "PASS"
                else "unverified"
            )
            claims.append({"kind": "F", "value": p, "context": u[:160], "evidence": hit, "status": status})
        for m in TICK_RE.finditer(u):
            tok = m.group(1).strip()
            if PATH_RE.fullmatch(tok) or len(tok) < 3 or not CODE_TOKEN.match(tok):
                continue
            key = ("I", tok)
            if key in seen:
                continue
            seen.add(key)
            core = re.sub(r"\(\)$", "", tok)
            last = re.split(r"\.|::|->", core)[-1]
            if corrected:
                ev = next((l for l in plus + minus if _has_identifier(l, core)), None) or next(
                    (l for l in context if _has_identifier(l, core) or (len(last) > 3 and _has_identifier(l, last))),
                    None,
                )
            else:
                ev = next((l for l in plus + minus if core in l), None) or next(
                    (l for l in context if core in l or (len(last) > 3 and last in l)),
                    None,
                )
            snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", last).lower()
            stem_hit = next(
                (f for f in files if f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower() in (snake, last.lower())), None
            ) if len(last) > 3 else None
            if ev is None and stem_hit:
                status = (
                    "supported"
                    if gate_check(
                        stem_hit.rsplit("/", 1)[-1],
                        stem_hit,
                        file_list_text,
                        "changed_files",
                        evidence_profile=evidence_profile,
                    )
                    == "PASS"
                    else "unverified"
                )
                claims.append({"kind": "I", "value": tok, "context": u[:160], "evidence": "fichero " + stem_hit, "status": status})
                continue
            if ev is not None:
                q = _centered_quote(ev, core, last) if corrected else ev.strip()[:300]
                status = (
                    "supported"
                    if gate_check(tok, q, changed_text, "diff_changed_lines", evidence_profile=evidence_profile) == "PASS"
                    else "unverified"
                )
            else:
                status = "unverified"
            claims.append({"kind": "I", "value": tok, "context": u[:160], "evidence": (ev or "")[:160] or None, "status": status})
    signals = {}
    n_lines = len(plus) + len(minus)
    if units and (not files or n_lines == 0):
        signals["E_empty_diff"] = True
    checked = re.findall(r"(?m)^\s*-\s*\[[xX]\]\s*(.+)$", body)
    types = [c for c in checked if re.search(r"(bug fix|new feature|breaking change|documentation update)", c, re.I)]
    code_non_doc = [f for f in files if CODE_EXT.search(f) and not re.match(r"(?i)^(docs?|documentation)/", f)]
    if types and all(re.search(r"documentation", c, re.I) for c in types) and code_non_doc:
        signals["D_doc_only_but_code"] = (code_non_doc[:5] if corrected else [f for f in files if CODE_EXT.search(f)][:5])
    title_words = content_words(pr.get("title", ""))
    if content_words(strip_template(pr["body"])) < 8 and title_words < 3:
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
    ver = "v2_postreview" if corrected else VERSION
    rec = {
        "pr_id": pr["pr_id"],
        "url": pr["url"],
        "verdict": verdict,
        "signals": signals,
        "claims": claims,
        "n_claims": len(fi),
        "n_supported": len(fi) - len(unsup),
        "files_changed": len(files),
        "lines_changed": n_lines,
        "version": ver,
        "evidence_profile": evidence_profile,
    }
    rec["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {k: rec[k] for k in ("pr_id", "verdict", "signals", "claims", "evidence_profile")},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
        + hashlib.sha256((body + "\x00" + patch + "\x00" + "|".join(files)).encode()).digest()
    ).hexdigest()
    return rec


def check_pr(pr: dict, *, evidence_profile: str = "legal_prose") -> dict:
    """Original v2 checker (preserved intact; pass corrected=True or call check_pr_corrected for Task 1 fixes)."""
    return _check_pr_impl(pr, corrected=False, evidence_profile=evidence_profile)


def check_pr_corrected(pr: dict, *, evidence_profile: str = "legal_prose") -> dict:
    """Post-review bug-fixed v2 checker with Task 1 fixes applied."""
    return _check_pr_impl(pr, corrected=True, evidence_profile=evidence_profile)
