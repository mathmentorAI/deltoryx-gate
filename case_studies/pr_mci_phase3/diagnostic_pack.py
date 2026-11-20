"""Build, view, and validate the human-annotation pack for Phase 3 dev error attribution (`n=120`).

Implements Point 1 ONLY of the error-attribution diagnostic on AIDev-Dev originals
(frozen baseline `phase3-v2.0`, commit `3808d82`):
- Stratified deterministic sampling (`60 no_premises_in_diff`,
  `40 hypothesis_with_verified_premises`, `20 derived_supported`), strictly excluding
  previously audited claims.
- Zero live NIM calls, zero model/LLM pre-annotations, and zero third-party PR text or
  diffs inside git (`diagnostic_120/` lives strictly in external AIDev_phase3 storage).
- Self-contained offline browser diff viewer (`diff_viewer.html` + lazy-loaded
  `pr_diffs/pr_<pr_id>.js`), standalone Markdown sheets (`sheets/DIAG-XXX.md`),
  blank JSONL/CSV annotation templates, terminal `view` command, and strict `validate`.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_phase3 as p3  # noqa: E402
from claimlayer.semantic.models import canonical, digest  # noqa: E402
from claimlayer.v4.domains.code_change_inference import (  # noqa: E402
    BEHAVIOR_OPERATIONS,
    RelevanceConfig,
    VerdictThresholds,
    _LOCK_GEN_RE,
    build_typed_diff_ledger,
    repo_identifier_tokens,
    verify_behavior_graph,
)

DIAGNOSTIC_VERSION = "diagnostic-120-v1.0"
DIAGNOSTIC_SEED = 20261121
DEFAULT_STRATA: dict[str, int] = {
    "no_premises_in_diff": 60,
    "hypothesis_with_verified_premises": 40,
    "derived_supported": 20,
}

VALID_ANNOTATION_STATUSES: tuple[str, ...] = (
    "supported",
    "no_premises",
    "contradicted",
    "out_of_scope_meta",
)
VALID_CONFIDENCES: tuple[str, ...] = ("high", "medium", "low")
VALID_OPERATIONS: tuple[str, ...] = BEHAVIOR_OPERATIONS

OPERATION_DESCRIPTIONS_ES: dict[str, str] = {
    "implements": "Implementa / modifica / añade función, clase, método, símbolo o lógica",
    "adds_test_for": "Añade o actualiza tests para verificar un símbolo o comportamiento",
    "removes_support_for": "Elimina / quita / desactiva soporte o código existente",
    "configures": "Configura ajustes, CI/workflows, YAML/JSON/TOML, flags o constantes",
    "updates_dependency": "Añade, actualiza o elimina dependencias / versiones en manifiestos o lockfiles",
    "documents": "Documenta en README/docs o docstrings/comentarios",
    "other_behavior": "Otra afirmación conductual verificable del código/diff que no encaja arriba",
    "out_of_scope_meta": "Meta/fuera de alcance (plantilla PR sin contenido, pie generado, checklist genérico, enlace a issue)",
}

DEFAULT_AIDEV_DIR = os.path.expanduser(
    "~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3"
)
DEFAULT_OUT_DIR = os.path.join(DEFAULT_AIDEV_DIR, "diagnostic_120")
DEFAULT_PREREG = os.path.join(_THIS_DIR, "PREREGISTRATION.json")
DEFAULT_MANIFEST_OUT = os.path.join(_THIS_DIR, "diagnostic_120_manifest.json")

_CSV_COLUMNS: tuple[str, ...] = (
    "item_id",
    "claim_id",
    "pr_id",
    "repo",
    "pr_url",
    "pr_title",
    "claim_text",
    "diff_files_count",
    "sheet_file",
    "annotation_status",
    "diff_contradicts_claim",
    "gold_operation",
    "gold_target",
    "gold_premises_json",
    "contradicting_premises_json",
    "confidence",
    "notes",
)


def safe_js_json(obj: Any) -> str:
    """Serialize *obj* to JSON safely embeddable inside JavaScript `<script>` or `.js` files."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return (
        raw.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def load_excluded_claim_ids(path: str | None) -> set[str]:
    """Load `claim_id` exclusions from a JSON list of strings or dicts (`{'claim_id': ...}`)."""
    if not path:
        return set()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, Mapping):
        if isinstance(data.get("excluded_claim_ids"), list):
            data = data["excluded_claim_ids"]
        elif isinstance(data.get("items"), list):
            data = data["items"]
        else:
            raise ValueError(f"Unsupported exclusion JSON mapping keys in {path}")
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    out: set[str] = set()
    for ent in data:
        if isinstance(ent, str) and ent.strip():
            out.add(ent.strip())
        elif isinstance(ent, Mapping) and str(ent.get("claim_id") or "").strip():
            out.add(str(ent["claim_id"]).strip())
        else:
            raise ValueError(f"Invalid exclusion entry in {path}: {ent!r}")
    return out


def load_frozen_prereg_config(prereg_path: str) -> tuple[RelevanceConfig, VerdictThresholds, dict[str, Any]]:
    """Read frozen `RelevanceConfig`, `VerdictThresholds`, and metadata from PREREGISTRATION.json."""
    with open(prereg_path, encoding="utf-8") as fh:
        prereg = json.load(fh)
    fc = prereg.get("frozen_config") or {}
    rel_cfg = RelevanceConfig(**(fc.get("relevance_config") or {}))
    th_cfg = VerdictThresholds(**(fc.get("thresholds") or {}))
    baseline_commit = str(prereg.get("code_library_commit") or "").strip()
    if not baseline_commit:
        try:
            baseline_commit = subprocess.check_output(
                ["git", "-C", _REPO_ROOT, "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            baseline_commit = ""
    meta = {
        "preregistration_version": str(prereg.get("preregistration_version") or ""),
        "baseline_commit": baseline_commit,
    }
    return rel_cfg, th_cfg, meta


def _normalize_pr_dict(pr: Mapping[str, Any]) -> dict[str, Any]:
    pr_dict = dict(pr)
    if "patch" not in pr_dict:
        pr_dict["patch"] = "\n---FILE---\n".join(
            (fp.get("patch") or "")
            for fp in pr_dict.get("file_patches", ())
            if (fp.get("patch") or "").strip()
        )
    if "files" not in pr_dict or pr_dict["files"] is None:
        pr_dict["files"] = [
            str(fp.get("filename"))
            for fp in pr_dict.get("file_patches", ())
            if fp.get("filename")
        ]
    return pr_dict


def extract_numbered_pr_diff(pr: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a PR's full attributed diff into file/chunk blocks with 1-based line numbers."""
    pr_dict = _normalize_pr_dict(pr)
    pr_id = str(pr_dict.get("pr_id") or "")
    typed_ledger = build_typed_diff_ledger(pr_dict)
    chunks_out: list[dict[str, Any]] = []
    for ch in typed_ledger.chunks:
        fname = str(ch.get("file") or "<unattributed>")
        patch_txt = str(ch.get("patch") or "")
        raw_lines = patch_txt.splitlines()
        numbered: list[dict[str, Any]] = []
        adds = 0
        dels = 0
        for ln_no, ln_txt in enumerate(raw_lines, start=1):
            if ln_txt.startswith("+") and not ln_txt.startswith("+++"):
                kind = "add"
                adds += 1
            elif ln_txt.startswith("-") and not ln_txt.startswith("---"):
                kind = "del"
                dels += 1
            elif ln_txt.startswith("@@"):
                kind = "hunk"
            else:
                kind = "ctx"
            numbered.append({"line_no": ln_no, "kind": kind, "text": ln_txt})
        chunks_out.append(
            {
                "chunk_id": str(ch.get("chunk_id") or ""),
                "file": fname,
                "category": str(ch.get("category") or "src"),
                "is_lock_or_generated": bool(fname != "<unattributed>" and _LOCK_GEN_RE.search(fname)),
                "adds": adds,
                "dels": dels,
                "line_count": len(numbered),
                "lines": numbered,
            }
        )
    return {
        "pr_id": pr_id,
        "repo": str(pr_dict.get("repo") or ""),
        "number": pr_dict.get("number"),
        "pr_url": str(pr_dict.get("url") or ""),
        "pr_title": str(pr_dict.get("title") or "").strip(),
        "pr_body": str(pr_dict.get("body") or "").strip(),
        "files": list(typed_ledger.files),
        "diff_files_count": typed_ledger.files_count,
        "adds_count": pr_dict.get("adds"),
        "dels_count": pr_dict.get("dels"),
        "chunks": chunks_out,
    }


def sample_diagnostic_claims(
    prs: Sequence[Mapping[str, Any]],
    *,
    pass1_model: Any,
    pass2_model: Any,
    relevance_config: RelevanceConfig = RelevanceConfig(),
    thresholds: VerdictThresholds = VerdictThresholds(),
    excluded_claim_ids: set[str] | frozenset[str] = frozenset(),
    strata_quotas: Mapping[str, int] | None = None,
    seed: int = DIAGNOSTIC_SEED,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, dict[str, Any]]]:
    """Replay cached Pass 1/2/3 offline on *prs* and draw a blind stratified claim sample.

    Returns `(ordered_items, pool_sizes_by_stratum, pr_diffs_by_id)`.
    """
    quotas = dict(strata_quotas or DEFAULT_STRATA)
    pools: dict[str, list[dict[str, Any]]] = {st: [] for st in quotas}
    pr_by_id: dict[str, dict[str, Any]] = {}
    candidate_records_by_pr: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    for pr in prs:
        pr_dict = _normalize_pr_dict(pr)
        pr_id = str(pr_dict.get("pr_id") or "")
        pr_by_id[pr_id] = pr_dict
        rep = p3.evaluate_pr_phase3(
            pr_dict,
            pass1_model=pass1_model,
            pass2_model=pass2_model,
            relevance_config=relevance_config,
            thresholds=thresholds,
            modes=("d_full",),
            include_claim_text_in_memory=True,
        )
        summaries_by_cid = {
            b["claim_id"]: b for b in rep.get("behavior_claim_summaries", ())
        }
        proposals = rep.get("_in_memory_beh_proposals") or ()
        if not proposals:
            continue

        typed_ledger = build_typed_diff_ledger(pr_dict)
        repo_toks = repo_identifier_tokens(pr_dict.get("repo") or pr_dict.get("url"))

        for prop in proposals:
            cid = str(prop["claim_id"])
            if cid in excluded_claim_ids:
                continue
            summ = summaries_by_cid.get(cid)
            if not summ:
                continue
            st = str(summ.get("status") or "")
            if st not in pools:
                continue

            ver_full = verify_behavior_graph(
                claim_id=cid,
                claim_text=str(prop["claim_text"]),
                operation=str(prop["operation"]),
                target=prop.get("target"),
                proposed_premises=prop["merged_premises"],
                ledger=typed_ledger,
                repo_tokens=repo_toks,
                relevance_config=relevance_config,
                use_declared_rules=True,
                reread_attempted=bool(prop.get("reread_attempted")),
            )
            item_rec = {
                "claim_id": cid,
                "pr_id": pr_id,
                "agent": str(pr_dict.get("agent") or ""),
                "repo": str(pr_dict.get("repo") or ""),
                "pr_url": str(pr_dict.get("url") or ""),
                "pr_title": str(pr_dict.get("title") or "").strip(),
                "claim_text": str(prop["claim_text"]),
                "stratum": st,
                "baseline_d_full_status": st,
                "baseline_operation": str(summ.get("operation") or ""),
                "baseline_target": summ.get("target"),
                "baseline_rule_id": summ.get("rule_id"),
                "baseline_reread_attempted": bool(summ.get("reread_attempted")),
                "baseline_p2_premises": list(prop.get("p2_premises") or ()),
                "baseline_merged_premises": list(prop.get("merged_premises") or ()),
                "baseline_verified_premises": list(ver_full.get("verified_premises") or ()),
                "baseline_rejected_premises": list(ver_full.get("rejected_premises") or ()),
            }
            pools[st].append(item_rec)
            candidate_records_by_pr[pr_id].append(item_rec)

    pool_sizes: dict[str, int] = {}
    sampled: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for st, quota in quotas.items():
        pool = sorted(pools[st], key=lambda x: x["claim_id"])
        pool_sizes[st] = len(pool)
        if len(pool) < quota:
            raise ValueError(
                f"Stratum {st!r} pool size {len(pool)} is smaller than requested quota {quota}."
            )
        sampled.extend(rng.sample(pool, quota))

    rng_order = random.Random(seed + 1)
    rng_order.shuffle(sampled)

    ordered: list[dict[str, Any]] = []
    needed_pr_ids: set[str] = set()
    for seq, rec in enumerate(sampled, start=1):
        item_id = f"DIAG-{seq:03d}"
        needed_pr_ids.add(rec["pr_id"])
        ordered.append(
            {
                **rec,
                "item_id": item_id,
                "sheet_file": f"sheets/{item_id}.md",
            }
        )

    pr_diffs_by_id = {
        pid: extract_numbered_pr_diff(pr_by_id[pid])
        for pid in sorted(needed_pr_ids)
    }
    return ordered, pool_sizes, pr_diffs_by_id


def _render_markdown_sheet(item: Mapping[str, Any], pr_diff: Mapping[str, Any]) -> str:
    lines: list[str] = [
        f"# Diagnostic Item {item['item_id']}",
        "",
        f"- **PR ID:** `{item['pr_id']}`",
        f"- **Repository:** `{pr_diff.get('repo', '')}`",
        f"- **URL:** {pr_diff.get('pr_url', '')}",
        f"- **Changed Files:** {pr_diff.get('diff_files_count', 0)} (+{pr_diff.get('adds_count', 0)} / -{pr_diff.get('dels_count', 0)})",
        "",
        "## PR Title",
        "",
        str(pr_diff.get("pr_title") or ""),
        "",
        "## Afirmación a Evaluar (`claim_text`)",
        "",
        f"> {item['claim_text']}",
        "",
        "## PR Description",
        "",
        str(pr_diff.get("pr_body") or "_(empty body)_"),
        "",
        "## Changed Files",
        "",
    ]
    for fn in pr_diff.get("files", ()):
        lines.append(f"- `{fn}`")
    lines.extend(["", "## Numbered Diff Hunks", ""])
    for ch in pr_diff.get("chunks", ()):
        lines.append(
            f"### `{ch['file']}` — `{ch['chunk_id']}` (+{ch['adds']} / -{ch['dels']})"
        )
        lines.append("```diff")
        for ln in ch.get("lines", ()):
            lines.append(f"L{ln['line_no']:04d} | {ln['text']}")
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_diff_viewer_html(*, manifest_digest: str, total_items: int) -> str:
    """Return 100% self-contained, DOM-safe (`textContent` only) offline diff viewer HTML."""
    op_meta = [
        {"id": op, "desc": OPERATION_DESCRIPTIONS_ES.get(op, op)}
        for op in VALID_OPERATIONS
    ]
    op_meta_json = safe_js_json(op_meta)
    statuses_json = safe_js_json(
        [
            {"id": "supported", "label": "supported — El diff respalda esta afirmación (requiere >= 1 premisa)"},
            {"id": "no_premises", "label": "no_premises — No existe respaldo literal suficiente en el diff"},
            {"id": "contradicted", "label": "contradicted — El diff contradice directamente la afirmación"},
            {"id": "out_of_scope_meta", "label": "out_of_scope_meta — Meta / boilerplate PR / checklist vacío / enlace"},
        ]
    )
    confs_json = safe_js_json(list(VALID_CONFIDENCES))
    storage_key = f"deltoryx_diag120_{manifest_digest[:16]}"

    html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Diagnóstico Fase 3 (n=__TOTAL__) — Visor de Diff y Anotación</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f6f8fa;
    --panel: #ffffff;
    --panel-2: #f0f3f6;
    --border: #d0d7de;
    --fg: #1f2328;
    --muted: #656d76;
    --accent: #0969da;
    --add-bg: #dafbe1;
    --add-fg: #116329;
    --del-bg: #ffebe9;
    --del-fg: #82071e;
    --hunk-bg: #ddf4ff;
    --hunk-fg: #0550ae;
    --sel-sup: #fff3b0;
    --sel-sup-b: #d4a72c;
    --sel-con: #ffdce0;
    --sel-con-b: #cf222e;
    --hit-bg: #fff8c5;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1c2128;
      --border: #30363d;
      --fg: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --add-bg: #12261e;
      --add-fg: #7ee787;
      --del-bg: #2c1619;
      --del-fg: #ffa198;
      --hunk-bg: #13233a;
      --hunk-fg: #79c0ff;
      --sel-sup: #3b2e00;
      --sel-sup-b: #e3b341;
      --sel-con: #49161d;
      --sel-con-b: #f85149;
      --hit-bg: #4b3900;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--fg); line-height: 1.45; }
  header.topbar {
    position: sticky; top: 0; z-index: 20;
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 8px 14px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between;
  }
  .topbar-group { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  button, select, input, textarea {
    font: inherit; color: inherit; background: var(--panel); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 9px;
  }
  button { cursor: pointer; background: var(--panel-2); font-weight: 500; }
  button:hover { border-color: var(--accent); }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button:focus-visible, select:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 12px; border: 1px solid var(--border); background: var(--panel-2); }
  .badge.ok { background: var(--add-bg); color: var(--add-fg); }
  .badge.warn { background: var(--sel-sup); border-color: var(--sel-sup-b); }
  .layout { display: grid; grid-template-columns: 270px 1fr; min-height: calc(100vh - 52px); }
  @media (max-width: 980px) { .layout { grid-template-columns: 1fr; } }
  aside.sidebar {
    border-right: 1px solid var(--border); background: var(--panel); padding: 10px;
    max-height: calc(100vh - 52px); overflow-y: auto; position: sticky; top: 52px;
  }
  .item-list { list-style: none; padding: 0; margin: 8px 0 0; display: flex; flex-direction: column; gap: 4px; }
  .item-btn {
    width: 100%; text-align: left; padding: 6px 8px; border-radius: 6px; font-size: 12.5px;
    display: flex; justify-content: space-between; align-items: center; gap: 6px; background: transparent;
  }
  .item-btn.active { background: var(--hunk-bg); border-color: var(--accent); font-weight: 600; }
  .item-btn .sub { display: block; font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 175px; }
  main.workspace { padding: 14px; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
  .claim-box { font-size: 16px; font-weight: 600; padding: 10px 12px; border-left: 4px solid var(--accent); background: var(--panel-2); border-radius: 4px; }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
  label.fld { display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; font-weight: 600; }
  label.fld.inline { flex-direction: row; align-items: center; gap: 8px; padding-top: 22px; }
  .prem-list { display: flex; flex-direction: column; gap: 4px; margin-top: 6px; }
  .prem-chip {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;
    padding: 5px 8px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    border: 1px solid var(--sel-sup-b); background: var(--sel-sup);
  }
  .prem-chip.contra { border-color: var(--sel-con-b); background: var(--sel-con); }
  .diff-shell { display: grid; grid-template-columns: 240px 1fr; gap: 10px; min-width: 0; }
  @media (max-width: 1100px) { .diff-shell { grid-template-columns: 1fr; } }
  .file-nav { border: 1px solid var(--border); border-radius: 6px; background: var(--panel); padding: 8px; max-height: 70vh; overflow-y: auto; }
  .file-link { width: 100%; text-align: left; font-size: 12px; padding: 4px 6px; margin-bottom: 3px; word-break: break-all; background: transparent; }
  .diff-hunks { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  details.chunk { border: 1px solid var(--border); border-radius: 6px; background: var(--panel); content-visibility: auto; contain-intrinsic-size: auto 220px; }
  details.chunk > summary {
    padding: 7px 10px; cursor: pointer; background: var(--panel-2); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px; display: flex; justify-content: space-between; gap: 8px;
  }
  .diff-lines { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-x: auto; }
  .dline { display: grid; grid-template-columns: 54px 1fr; padding: 1px 6px; cursor: pointer; white-space: pre; border-bottom: 1px solid transparent; }
  .dline:hover { filter: brightness(0.95); }
  .dline .ln { color: var(--muted); user-select: none; text-align: right; padding-right: 10px; }
  .dline.add { background: var(--add-bg); color: var(--add-fg); }
  .dline.del { background: var(--del-bg); color: var(--del-fg); }
  .dline.hunk { background: var(--hunk-bg); color: var(--hunk-fg); }
  .dline.sel-sup { background: var(--sel-sup) !important; border-color: var(--sel-sup-b); font-weight: 600; }
  .dline.sel-con { background: var(--sel-con) !important; border-color: var(--sel-con-b); font-weight: 600; }
  .dline.search-hit { outline: 2px solid var(--accent); outline-offset: -1px; }
  .muted { color: var(--muted); font-size: 12.5px; }
  .error-banner { padding: 8px 10px; border-radius: 6px; border: 1px solid var(--sel-con-b); background: var(--sel-con); font-size: 13px; }
</style>
</head>
<body>
<header class="topbar">
  <div class="topbar-group">
    <strong>Diagnóstico Fase 3 (n=__TOTAL__)</strong>
    <span id="progressBadge" class="badge">0 / __TOTAL__ anotados</span>
    <span class="muted">Autoguardado local</span>
  </div>
  <div class="topbar-group">
    <button type="button" id="btnPrev" title="Alt+Flecha Izquierda">← Anterior</button>
    <button type="button" id="btnNext" title="Alt+Flecha Derecha">Siguiente →</button>
    <button type="button" id="btnValidate">Validar progreso</button>
    <button type="button" id="btnImportJsonl">Importar JSONL</button>
    <input type="file" id="fileImportInput" accept=".jsonl,.json" hidden />
    <button type="button" id="btnExportCsv">Exportar CSV</button>
    <button type="button" id="btnExportJsonl" class="primary">Exportar JSONL</button>
  </div>
</header>

<div class="layout">
  <aside class="sidebar">
    <div style="display:flex; flex-direction:column; gap:6px;">
      <input id="itemFilterInput" type="search" placeholder="Buscar item, PR o texto..." />
      <select id="itemFilterMode">
        <option value="all">Todos (__TOTAL__)</option>
        <option value="pending">Pendientes</option>
        <option value="done">Completos</option>
        <option value="invalid">Con avisos</option>
      </select>
    </div>
    <ul id="itemList" class="item-list"></ul>
  </aside>

  <main class="workspace">
    <div id="validationPanel" class="card" hidden></div>

    <section class="card" id="claimCard">
      <div style="display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px; align-items:center;">
        <div>
          <strong id="metaItemId" style="font-size:15px;"></strong>
          <span class="badge" id="metaRepo"></span>
          <span class="badge" id="metaFiles"></span>
        </div>
        <a id="metaUrl" href="#" target="_blank" rel="noopener noreferrer" class="muted">Abrir PR en GitHub ↗</a>
      </div>
      <div id="metaPrTitle" style="margin-top:6px; font-weight:600;"></div>
      <div class="claim-box" id="metaClaimText" style="margin-top:8px;"></div>
      <details style="margin-top:8px;">
        <summary class="muted" style="cursor:pointer;">Ver descripción completa del PR</summary>
        <pre id="metaPrBody" style="white-space:pre-wrap; font-family:inherit; margin:8px 0 0; font-size:13px;"></pre>
      </details>
    </section>

    <section class="card">
      <div class="form-grid">
        <label class="fld">Estado real en el diff (`annotation_status`)
          <select id="fldStatus"></select>
        </label>
        <label class="fld">Operación real (`gold_operation`)
          <select id="fldOperation"></select>
        </label>
        <label class="fld">Símbolo / fichero / clave objetivo (`gold_target`)
          <input id="fldTarget" type="text" placeholder="Ej: compute_retry_backoff o null si genérico" />
        </label>
        <label class="fld">Confianza (`confidence`)
          <select id="fldConfidence"></select>
        </label>
        <label class="fld inline">
          <input id="fldContradicts" type="checkbox" />
          <span>¿El diff contradice esta afirmación?</span>
        </label>
      </div>
      <label class="fld" style="margin-top:8px;">Notas de adjudicación (`notes`)
        <textarea id="fldNotes" rows="2" placeholder="Opcional: por qué es contradicted / no_premises / observación"></textarea>
      </label>

      <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between;">
        <div style="display:flex; gap:6px; align-items:center;">
          <strong style="font-size:13px;">Modo de clic en líneas del diff:</strong>
          <button type="button" id="modeSupportBtn" class="primary">Añadir a `gold_premises`</button>
          <button type="button" id="modeContraBtn">Añadir a `contradicting_premises`</button>
        </div>
        <button type="button" id="btnClearItemPrems">Vaciar líneas de este ítem</button>
      </div>

      <div class="form-grid" style="margin-top:8px;">
        <div>
          <strong style="font-size:12.5px;">`gold_premises` (respaldo, <span id="supCount">0</span>)</strong>
          <div id="supPremList" class="prem-list"></div>
        </div>
        <div>
          <strong style="font-size:12.5px;">`contradicting_premises` (contradicción, <span id="conCount">0</span>)</strong>
          <div id="conPremList" class="prem-list"></div>
        </div>
      </div>
    </section>

    <section class="card">
      <div style="display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">
          <strong>Diff completo numerado</strong>
          <input id="diffSearchInput" type="search" placeholder="Buscar en todo el diff del PR..." style="min-width:240px;" />
          <span id="diffSearchCount" class="muted">0 coincidencias</span>
          <button type="button" id="btnPrevHit">↑</button>
          <button type="button" id="btnNextHit">↓</button>
          <label style="font-size:12.5px; display:flex; align-items:center; gap:4px;">
            <input id="chkOnlyHitFiles" type="checkbox" /> Solo hunks con coincidencia
          </label>
        </div>
        <div style="display:flex; gap:6px;">
          <button type="button" id="btnExpandAll">Expandir todo</button>
          <button type="button" id="btnCollapseAll">Colapsar todo</button>
        </div>
      </div>
      <div class="diff-shell">
        <div id="fileNav" class="file-nav"></div>
        <div id="diffHunks" class="diff-hunks">Cargando diff...</div>
      </div>
    </section>
  </main>
</div>

<script src="viewer_index.js"></script>
<script>
(function () {
  "use strict";
  const STORAGE_KEY = "__STORAGE_KEY__";
  const STATUS_OPTS = __STATUS_OPTS__;
  const OP_OPTS = __OP_OPTS__;
  const CONF_OPTS = __CONF_OPTS__;
  const ITEMS = window.__DIAG_INDEX__ || [];
  const DIFF_CACHE = {};
  const PENDING_CB = {};
  window.__DIAG_RECEIVE_DIFF__ = function (prId, data) {
    DIFF_CACHE[String(prId)] = data;
    const cbs = PENDING_CB[String(prId)] || [];
    delete PENDING_CB[String(prId)];
    cbs.forEach((fn) => fn(data));
  };

  function loadDiff(prId, cb) {
    const key = String(prId);
    if (DIFF_CACHE[key]) { cb(DIFF_CACHE[key]); return; }
    if (!PENDING_CB[key]) {
      PENDING_CB[key] = [cb];
      const s = document.createElement("script");
      s.src = "pr_diffs/pr_" + encodeURIComponent(key) + ".js";
      s.onerror = function () {
        const host = document.getElementById("diffHunks");
        host.textContent = "Error al cargar pr_diffs/pr_" + key + ".js";
      };
      document.head.appendChild(s);
    } else {
      PENDING_CB[key].push(cb);
    }
  }

  let state = {};
  try {
    state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (e) { state = {}; }

  function defaultRec(it) {
    return {
      item_id: it.item_id,
      claim_id: it.claim_id,
      pr_id: it.pr_id,
      annotation_status: "",
      diff_contradicts_claim: false,
      gold_operation: "",
      gold_target: "",
      gold_premises: [],
      contradicting_premises: [],
      confidence: "",
      notes: ""
    };
  }
  function getRec(it) {
    if (!state[it.item_id]) state[it.item_id] = defaultRec(it);
    return state[it.item_id];
  }
  function saveState() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) {}
    updateProgress();
  }

  function validateRecord(it, rec) {
    const errs = [];
    if (!rec.annotation_status) errs.push("Falta annotation_status");
    if (!rec.gold_operation) errs.push("Falta gold_operation");
    if (!rec.confidence) errs.push("Falta confidence");
    if (rec.annotation_status === "supported" && (!rec.gold_premises || rec.gold_premises.length === 0)) {
      errs.push("supported requiere >= 1 línea en gold_premises");
    }
    if ((rec.annotation_status === "no_premises" || rec.annotation_status === "out_of_scope_meta") && rec.gold_premises && rec.gold_premises.length > 0) {
      errs.push(rec.annotation_status + " debe tener gold_premises vacío");
    }
    if (rec.annotation_status === "contradicted" && !rec.diff_contradicts_claim) {
      errs.push("contradicted requiere diff_contradicts_claim = true");
    }
    return errs;
  }

  let currentIdx = 0;
  let clickMode = "support";
  let currentHitIdx = -1;

  function initSelects() {
    const sEl = document.getElementById("fldStatus");
    const emptyS = document.createElement("option");
    emptyS.value = ""; emptyS.textContent = "-- Elegir estado --";
    sEl.appendChild(emptyS);
    STATUS_OPTS.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o.id; opt.textContent = o.label;
      sEl.appendChild(opt);
    });

    const oEl = document.getElementById("fldOperation");
    const emptyO = document.createElement("option");
    emptyO.value = ""; emptyO.textContent = "-- Elegir operación --";
    oEl.appendChild(emptyO);
    OP_OPTS.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o.id; opt.textContent = o.id + " — " + o.desc;
      oEl.appendChild(opt);
    });

    const cEl = document.getElementById("fldConfidence");
    const emptyC = document.createElement("option");
    emptyC.value = ""; emptyC.textContent = "-- Confianza --";
    cEl.appendChild(emptyC);
    CONF_OPTS.forEach((c) => {
      const opt = document.createElement("option");
      opt.value = c; opt.textContent = c;
      cEl.appendChild(opt);
    });
  }

  function updateProgress() {
    let complete = 0;
    ITEMS.forEach((it) => {
      if (validateRecord(it, getRec(it)).length === 0) complete += 1;
    });
    const badge = document.getElementById("progressBadge");
    badge.textContent = complete + " / " + ITEMS.length + " completos";
    badge.className = "badge " + (complete === ITEMS.length ? "ok" : "");
    renderSidebar();
  }

  function renderSidebar() {
    const ul = document.getElementById("itemList");
    ul.textContent = "";
    const q = (document.getElementById("itemFilterInput").value || "").trim().toLowerCase();
    const mode = document.getElementById("itemFilterMode").value;
    ITEMS.forEach((it, idx) => {
      const rec = getRec(it);
      const errs = validateRecord(it, rec);
      const isDone = errs.length === 0;
      const started = Boolean(rec.annotation_status || rec.gold_operation || (rec.gold_premises || []).length);
      if (mode === "pending" && isDone) return;
      if (mode === "done" && !isDone) return;
      if (mode === "invalid" && (!started || isDone)) return;
      if (q) {
        const hay = (it.item_id + " " + it.pr_id + " " + it.repo + " " + it.claim_text).toLowerCase();
        if (!hay.includes(q)) return;
      }
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "item-btn" + (idx === currentIdx ? " active" : "");
      btn.addEventListener("click", () => { currentIdx = idx; renderItem(); });

      const left = document.createElement("div");
      const title = document.createElement("div");
      title.textContent = it.item_id + " (" + it.pr_id + ")";
      const sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = it.claim_text;
      left.appendChild(title);
      left.appendChild(sub);

      const bdg = document.createElement("span");
      bdg.className = "badge " + (isDone ? "ok" : started ? "warn" : "");
      bdg.textContent = isDone ? (rec.annotation_status === "supported" ? "OK" : rec.annotation_status.slice(0, 5)) : started ? "inc" : "—";

      btn.appendChild(left);
      btn.appendChild(bdg);
      li.appendChild(btn);
      ul.appendChild(li);
    });
  }

  function renderPremiseLists(it) {
    const rec = getRec(it);
    document.getElementById("supCount").textContent = String((rec.gold_premises || []).length);
    document.getElementById("conCount").textContent = String((rec.contradicting_premises || []).length);

    function fillList(elId, arr, isContra) {
      const box = document.getElementById(elId);
      box.textContent = "";
      (arr || []).forEach((p, pidx) => {
        const chip = document.createElement("div");
        chip.className = "prem-chip" + (isContra ? " contra" : "");
        const txt = document.createElement("span");
        txt.textContent = p.file + ":" + p.diff_line_no + "  " + p.literal_quote;
        const rm = document.createElement("button");
        rm.type = "button";
        rm.textContent = "×";
        rm.title = "Quitar";
        rm.addEventListener("click", () => {
          arr.splice(pidx, 1);
          saveState();
          renderPremiseLists(it);
          refreshSelectedLineHighlights(it);
        });
        chip.appendChild(txt);
        chip.appendChild(rm);
        box.appendChild(chip);
      });
    }
    fillList("supPremList", rec.gold_premises, false);
    fillList("conPremList", rec.contradicting_premises, true);
  }

  function premKey(p) {
    return (p.chunk_id || "") + "::" + p.diff_line_no;
  }

  function refreshSelectedLineHighlights(it) {
    const rec = getRec(it);
    const supKeys = new Set((rec.gold_premises || []).map(premKey));
    const conKeys = new Set((rec.contradicting_premises || []).map(premKey));
    document.querySelectorAll(".dline").forEach((el) => {
      const k = el.dataset.premKey;
      el.classList.toggle("sel-sup", supKeys.has(k));
      el.classList.toggle("sel-con", conKeys.has(k));
    });
  }

  function togglePremiseLine(it, ch, ln) {
    const rec = getRec(it);
    const arr = clickMode === "contradict" ? rec.contradicting_premises : rec.gold_premises;
    const key = ch.chunk_id + "::" + ln.line_no;
    const pos = arr.findIndex((p) => premKey(p) === key);
    if (pos >= 0) {
      arr.splice(pos, 1);
    } else {
      arr.push({
        file: ch.file,
        chunk_id: ch.chunk_id,
        diff_line_no: ln.line_no,
        literal_quote: ln.text
      });
    }
    saveState();
    renderPremiseLists(it);
    refreshSelectedLineHighlights(it);
  }

  function renderDiff(it, diff) {
    const fileNav = document.getElementById("fileNav");
    const hunksHost = document.getElementById("diffHunks");
    fileNav.textContent = "";
    hunksHost.textContent = "";

    const searchQ = (document.getElementById("diffSearchInput").value || "").trim().toLowerCase();
    const onlyHits = document.getElementById("chkOnlyHitFiles").checked;
    const rec = getRec(it);
    const supKeys = new Set((rec.gold_premises || []).map(premKey));
    const conKeys = new Set((rec.contradicting_premises || []).map(premKey));
    let totalHits = 0;

    diff.chunks.forEach((ch, cidx) => {
      const chunkHits = searchQ
        ? ch.lines.filter((ln) => ln.text.toLowerCase().includes(searchQ) || ch.file.toLowerCase().includes(searchQ)).length
        : 0;
      totalHits += chunkHits;
      if (onlyHits && searchQ && chunkHits === 0) return;

      const fBtn = document.createElement("button");
      fBtn.type = "button";
      fBtn.className = "file-link";
      fBtn.textContent = ch.file + " (+" + ch.adds + "/-" + ch.dels + ")" + (chunkHits ? " [" + chunkHits + "]" : "");
      fBtn.addEventListener("click", () => {
        const targetDet = document.getElementById("chunk_" + cidx);
        if (targetDet) {
          targetDet.open = true;
          targetDet.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      fileNav.appendChild(fBtn);

      const det = document.createElement("details");
      det.className = "chunk";
      det.id = "chunk_" + cidx;
      const hasSel = ch.lines.some((ln) => supKeys.has(ch.chunk_id + "::" + ln.line_no) || conKeys.has(ch.chunk_id + "::" + ln.line_no));
      det.open = hasSel || chunkHits > 0 || (!ch.is_lock_or_generated && ch.line_count <= 450 && diff.chunks.length <= 20);

      const sum = document.createElement("summary");
      const leftSum = document.createElement("span");
      leftSum.textContent = ch.file + " [" + ch.chunk_id + "]";
      const rightSum = document.createElement("span");
      rightSum.textContent = "+" + ch.adds + " / -" + ch.dels + " (" + ch.line_count + " líneas)" + (ch.is_lock_or_generated ? " [lock/gen]" : "");
      sum.appendChild(leftSum);
      sum.appendChild(rightSum);
      det.appendChild(sum);

      const linesWrap = document.createElement("div");
      linesWrap.className = "diff-lines";
      ch.lines.forEach((ln) => {
        const row = document.createElement("div");
        const pk = ch.chunk_id + "::" + ln.line_no;
        const isHit = Boolean(searchQ && ln.text.toLowerCase().includes(searchQ));
        row.className =
          "dline " +
          ln.kind +
          (supKeys.has(pk) ? " sel-sup" : "") +
          (conKeys.has(pk) ? " sel-con" : "") +
          (isHit ? " search-hit" : "");
        row.dataset.premKey = pk;

        const lnSpan = document.createElement("span");
        lnSpan.className = "ln";
        lnSpan.textContent = "L" + ln.line_no;
        const txtSpan = document.createElement("span");
        txtSpan.textContent = ln.text;

        row.appendChild(lnSpan);
        row.appendChild(txtSpan);
        row.addEventListener("click", () => togglePremiseLine(it, ch, ln));
        linesWrap.appendChild(row);
      });
      det.appendChild(linesWrap);
      hunksHost.appendChild(det);
    });

    document.getElementById("diffSearchCount").textContent = totalHits + " coincidencias";
    currentHitIdx = -1;
  }

  function renderItem() {
    if (!ITEMS.length) return;
    const it = ITEMS[currentIdx];
    const rec = getRec(it);

    document.getElementById("metaItemId").textContent = it.item_id + " · PR " + it.pr_id;
    document.getElementById("metaRepo").textContent = it.repo;
    document.getElementById("metaFiles").textContent = it.diff_files_count + " ficheros";
    const urlEl = document.getElementById("metaUrl");
    urlEl.href = /^https?:\\/\\//i.test(it.pr_url || "") ? it.pr_url : "#";
    document.getElementById("metaPrTitle").textContent = it.pr_title;
    document.getElementById("metaClaimText").textContent = it.claim_text;
    document.getElementById("metaPrBody").textContent = "(cargando descripción...)";

    document.getElementById("fldStatus").value = rec.annotation_status || "";
    document.getElementById("fldOperation").value = rec.gold_operation || "";
    document.getElementById("fldTarget").value = rec.gold_target || "";
    document.getElementById("fldConfidence").value = rec.confidence || "";
    document.getElementById("fldContradicts").checked = Boolean(rec.diff_contradicts_claim);
    document.getElementById("fldNotes").value = rec.notes || "";

    renderPremiseLists(it);
    renderSidebar();
    document.getElementById("diffHunks").textContent = "Cargando diff del PR " + it.pr_id + "...";

    loadDiff(it.pr_id, (diff) => {
      if (ITEMS[currentIdx].item_id !== it.item_id) return;
      document.getElementById("metaPrBody").textContent = diff.pr_body || "(cuerpo vacío)";
      renderDiff(it, diff);
    });
  }

  function downloadFile(name, content, mime) {
    const blob = new Blob([content], { type: mime });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function csvCell(val) {
    const s = String(val == null ? "" : val);
    return '"' + s.replace(/"/g, '""') + '"';
  }

  function bindControls() {
    document.getElementById("fldStatus").addEventListener("change", (e) => {
      const it = ITEMS[currentIdx];
      const rec = getRec(it);
      rec.annotation_status = e.target.value;
      if (rec.annotation_status === "contradicted") {
        rec.diff_contradicts_claim = true;
        document.getElementById("fldContradicts").checked = true;
        clickMode = "contradict";
        document.getElementById("modeContraBtn").classList.add("primary");
        document.getElementById("modeSupportBtn").classList.remove("primary");
      } else if (rec.annotation_status === "supported") {
        clickMode = "support";
        document.getElementById("modeSupportBtn").classList.add("primary");
        document.getElementById("modeContraBtn").classList.remove("primary");
      }
      saveState();
    });
    document.getElementById("fldOperation").addEventListener("change", (e) => {
      getRec(ITEMS[currentIdx]).gold_operation = e.target.value;
      saveState();
    });
    document.getElementById("fldTarget").addEventListener("input", (e) => {
      getRec(ITEMS[currentIdx]).gold_target = e.target.value;
      saveState();
    });
    document.getElementById("fldConfidence").addEventListener("change", (e) => {
      getRec(ITEMS[currentIdx]).confidence = e.target.value;
      saveState();
    });
    document.getElementById("fldContradicts").addEventListener("change", (e) => {
      getRec(ITEMS[currentIdx]).diff_contradicts_claim = e.target.checked;
      saveState();
    });
    document.getElementById("fldNotes").addEventListener("input", (e) => {
      getRec(ITEMS[currentIdx]).notes = e.target.value;
      saveState();
    });

    document.getElementById("modeSupportBtn").addEventListener("click", () => {
      clickMode = "support";
      document.getElementById("modeSupportBtn").classList.add("primary");
      document.getElementById("modeContraBtn").classList.remove("primary");
    });
    document.getElementById("modeContraBtn").addEventListener("click", () => {
      clickMode = "contradict";
      document.getElementById("modeContraBtn").classList.add("primary");
      document.getElementById("modeSupportBtn").classList.remove("primary");
    });
    document.getElementById("btnClearItemPrems").addEventListener("click", () => {
      const it = ITEMS[currentIdx];
      const rec = getRec(it);
      rec.gold_premises = [];
      rec.contradicting_premises = [];
      saveState();
      renderPremiseLists(it);
      refreshSelectedLineHighlights(it);
    });

    document.getElementById("btnPrev").addEventListener("click", () => {
      if (currentIdx > 0) { currentIdx -= 1; renderItem(); }
    });
    document.getElementById("btnNext").addEventListener("click", () => {
      if (currentIdx < ITEMS.length - 1) { currentIdx += 1; renderItem(); }
    });
    document.addEventListener("keydown", (e) => {
      if (e.altKey && e.key === "ArrowLeft" && currentIdx > 0) { currentIdx -= 1; renderItem(); }
      if (e.altKey && e.key === "ArrowRight" && currentIdx < ITEMS.length - 1) { currentIdx += 1; renderItem(); }
    });

    document.getElementById("itemFilterInput").addEventListener("input", renderSidebar);
    document.getElementById("itemFilterMode").addEventListener("change", renderSidebar);
    document.getElementById("diffSearchInput").addEventListener("input", () => {
      const it = ITEMS[currentIdx];
      if (DIFF_CACHE[String(it.pr_id)]) renderDiff(it, DIFF_CACHE[String(it.pr_id)]);
    });
    document.getElementById("chkOnlyHitFiles").addEventListener("change", () => {
      const it = ITEMS[currentIdx];
      if (DIFF_CACHE[String(it.pr_id)]) renderDiff(it, DIFF_CACHE[String(it.pr_id)]);
    });
    function jumpHit(delta) {
      const hits = Array.from(document.querySelectorAll(".dline.search-hit"));
      if (!hits.length) return;
      currentHitIdx = (currentHitIdx + delta + hits.length) % hits.length;
      const el = hits[currentHitIdx];
      const det = el.closest("details.chunk");
      if (det) det.open = true;
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
    document.getElementById("btnPrevHit").addEventListener("click", () => jumpHit(-1));
    document.getElementById("btnNextHit").addEventListener("click", () => jumpHit(1));
    document.getElementById("btnExpandAll").addEventListener("click", () => {
      document.querySelectorAll("details.chunk").forEach((d) => { d.open = true; });
    });
    document.getElementById("btnCollapseAll").addEventListener("click", () => {
      document.querySelectorAll("details.chunk").forEach((d) => { d.open = false; });
    });

    document.getElementById("btnValidate").addEventListener("click", () => {
      const panel = document.getElementById("validationPanel");
      panel.hidden = false;
      panel.textContent = "";
      const bad = [];
      ITEMS.forEach((it, idx) => {
        const errs = validateRecord(it, getRec(it));
        if (errs.length) bad.push({ idx, item_id: it.item_id, errs });
      });
      const h = document.createElement("div");
      h.style.display = "flex";
      h.style.justifyContent = "space-between";
      const msg = document.createElement("strong");
      msg.textContent = bad.length === 0
        ? "¡Todos los 120 ítems pasan la validación local! Listo para Exportar JSONL."
        : "Ítems pendientes o con avisos: " + bad.length + " / " + ITEMS.length;
      const closeBtn = document.createElement("button");
      closeBtn.type = "button";
      closeBtn.textContent = "Cerrar";
      closeBtn.addEventListener("click", () => { panel.hidden = true; });
      h.appendChild(msg);
      h.appendChild(closeBtn);
      panel.appendChild(h);
      if (bad.length) {
        const ul = document.createElement("ul");
        bad.slice(0, 20).forEach((b) => {
          const li = document.createElement("li");
          const link = document.createElement("button");
          link.type = "button";
          link.textContent = b.item_id;
          link.addEventListener("click", () => { currentIdx = b.idx; renderItem(); });
          li.appendChild(link);
          li.appendChild(document.createTextNode(" — " + b.errs.join("; ")));
          ul.appendChild(li);
        });
        panel.appendChild(ul);
      }
    });

    document.getElementById("btnExportJsonl").addEventListener("click", () => {
      const lines = ITEMS.map((it) => {
        const r = getRec(it);
        return JSON.stringify({
          item_id: it.item_id,
          claim_id: it.claim_id,
          pr_id: it.pr_id,
          repo: it.repo,
          pr_url: it.pr_url,
          pr_title: it.pr_title,
          claim_text: it.claim_text,
          diff_files_count: it.diff_files_count,
          sheet_file: it.sheet_file,
          annotation_status: r.annotation_status || "",
          diff_contradicts_claim: Boolean(r.diff_contradicts_claim),
          gold_operation: r.gold_operation || "",
          gold_target: r.gold_target || "",
          gold_premises: r.gold_premises || [],
          contradicting_premises: r.contradicting_premises || [],
          confidence: r.confidence || "",
          notes: r.notes || ""
        });
      });
      downloadFile("annotations_completed.jsonl", lines.join("\\n") + "\\n", "application/x-ndjson;charset=utf-8");
    });

    document.getElementById("btnExportCsv").addEventListener("click", () => {
      const hdr = [
        "item_id","claim_id","pr_id","repo","pr_url","pr_title","claim_text",
        "diff_files_count","sheet_file","annotation_status","diff_contradicts_claim",
        "gold_operation","gold_target","gold_premises_json","contradicting_premises_json",
        "confidence","notes"
      ];
      const rows = [hdr.join(",")];
      ITEMS.forEach((it) => {
        const r = getRec(it);
        rows.push([
          it.item_id, it.claim_id, it.pr_id, it.repo, it.pr_url, it.pr_title, it.claim_text,
          it.diff_files_count, it.sheet_file, r.annotation_status || "",
          Boolean(r.diff_contradicts_claim), r.gold_operation || "", r.gold_target || "",
          JSON.stringify(r.gold_premises || []), JSON.stringify(r.contradicting_premises || []),
          r.confidence || "", r.notes || ""
        ].map(csvCell).join(","));
      });
      downloadFile("annotations_completed.csv", rows.join("\\n") + "\\n", "text/csv;charset=utf-8");
    });

    document.getElementById("btnImportJsonl").addEventListener("click", () => {
      document.getElementById("fileImportInput").click();
    });
    document.getElementById("fileImportInput").addEventListener("change", (e) => {
      const f = e.target.files && e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const text = String(reader.result || "");
          const lines = text.trim().startsWith("[") ? JSON.parse(text) : text.split(/\\r?\\n/).filter((x) => x.trim()).map((x) => JSON.parse(x));
          lines.forEach((obj) => {
            if (obj && obj.item_id) state[obj.item_id] = obj;
          });
          saveState();
          renderItem();
        } catch (err) {
          alert("Error al importar JSONL: " + err);
        }
      };
      reader.readAsText(f, "utf-8");
      e.target.value = "";
    });
  }

  initSelects();
  bindControls();
  updateProgress();
  renderItem();
})();
</script>
</body>
</html>
"""
    return (
        html.replace("__TOTAL__", str(total_items))
        .replace("__STORAGE_KEY__", storage_key)
        .replace("__STATUS_OPTS__", statuses_json)
        .replace("__OP_OPTS__", op_meta_json)
        .replace("__CONF_OPTS__", confs_json)
    )


def write_diagnostic_pack(
    ordered_items: Sequence[Mapping[str, Any]],
    pr_diffs_by_id: Mapping[str, Mapping[str, Any]],
    *,
    out_dir: str,
    manifest_out: str | None,
    pool_sizes: Mapping[str, int],
    strata_quotas: Mapping[str, int],
    excluded_ids: Sequence[str] | set[str] | frozenset[str],
    prereg_meta: Mapping[str, Any],
    seed: int = DIAGNOSTIC_SEED,
) -> dict[str, Any]:
    """Write all blinded external annotation assets and optional sanitized manifest."""
    sheets_dir = os.path.join(out_dir, "sheets")
    diffs_dir = os.path.join(out_dir, "pr_diffs")
    os.makedirs(sheets_dir, exist_ok=True)
    os.makedirs(diffs_dir, exist_ok=True)

    # 1. Per-PR JS diff modules (deduplicated across claims from the same PR)
    for pr_id, pr_diff in sorted(pr_diffs_by_id.items()):
        diff_js = (
            "window.__DIAG_RECEIVE_DIFF__("
            + safe_js_json(pr_id)
            + ", "
            + safe_js_json(pr_diff)
            + ");\n"
        )
        with open(os.path.join(diffs_dir, f"pr_{pr_id}.js"), "w", encoding="utf-8") as dfh:
            dfh.write(diff_js)

    # 2. Blinded JSONL, CSV, Sheets, and Viewer Index
    viewer_index: list[dict[str, Any]] = []
    jsonl_path = os.path.join(out_dir, "annotations_template.jsonl")
    csv_path = os.path.join(out_dir, "annotations_template.csv")
    with (
        open(jsonl_path, "w", encoding="utf-8") as jfh,
        open(csv_path, "w", newline="", encoding="utf-8") as cfh,
    ):
        cw = csv.DictWriter(cfh, fieldnames=list(_CSV_COLUMNS))
        cw.writeheader()
        for item in ordered_items:
            pr_diff = pr_diffs_by_id[item["pr_id"]]
            public_ref = {
                "item_id": item["item_id"],
                "claim_id": item["claim_id"],
                "pr_id": item["pr_id"],
                "repo": item["repo"],
                "pr_url": item["pr_url"],
                "pr_title": item["pr_title"],
                "claim_text": item["claim_text"],
                "diff_files_count": pr_diff["diff_files_count"],
                "sheet_file": item["sheet_file"],
            }
            viewer_index.append(public_ref)

            jfh.write(
                json.dumps(
                    {
                        **public_ref,
                        "annotation_status": "",
                        "diff_contradicts_claim": None,
                        "gold_operation": "",
                        "gold_target": "",
                        "gold_premises": [],
                        "contradicting_premises": [],
                        "confidence": "",
                        "notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            cw.writerow(
                {
                    **public_ref,
                    "annotation_status": "",
                    "diff_contradicts_claim": "",
                    "gold_operation": "",
                    "gold_target": "",
                    "gold_premises_json": "",
                    "contradicting_premises_json": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            sheet_path = os.path.join(out_dir, item["sheet_file"])
            with open(sheet_path, "w", encoding="utf-8") as sfh:
                sfh.write(_render_markdown_sheet(item, pr_diff))

    with open(os.path.join(out_dir, "viewer_index.js"), "w", encoding="utf-8") as vfh:
        vfh.write("window.__DIAG_INDEX__ = " + safe_js_json(viewer_index) + ";\n")

    # 3. Private blind key (strictly outside git in out_dir)
    blind_key = [
        {
            "item_id": item["item_id"],
            "claim_id": item["claim_id"],
            "pr_id": item["pr_id"],
            "agent": item["agent"],
            "stratum": item["stratum"],
            "baseline_d_full_status": item["baseline_d_full_status"],
            "baseline_operation": item["baseline_operation"],
            "baseline_target": item["baseline_target"],
            "baseline_rule_id": item["baseline_rule_id"],
            "baseline_reread_attempted": item["baseline_reread_attempted"],
            "baseline_p2_premises": item["baseline_p2_premises"],
            "baseline_merged_premises": item["baseline_merged_premises"],
            "baseline_verified_premises": item["baseline_verified_premises"],
            "baseline_rejected_premises": item["baseline_rejected_premises"],
        }
        for item in ordered_items
    ]
    blind_key_path = os.path.join(out_dir, "blind_key.json")
    with open(blind_key_path, "w", encoding="utf-8") as kfh:
        json.dump(blind_key, kfh, indent=2, sort_keys=True, ensure_ascii=False)
    blind_key_sha = hashlib.sha256(open(blind_key_path, "rb").read()).hexdigest()

    manifest_items = [
        {
            "item_id": item["item_id"],
            "claim_id": item["claim_id"],
            "pr_id": item["pr_id"],
        }
        for item in ordered_items
    ]
    manifest_digest = digest(canonical(manifest_items))
    viewer_html = _render_diff_viewer_html(
        manifest_digest=manifest_digest, total_items=len(ordered_items)
    )
    with open(os.path.join(out_dir, "diff_viewer.html"), "w", encoding="utf-8") as hfh:
        hfh.write(viewer_html)

    manifest = {
        "version": DIAGNOSTIC_VERSION,
        "seed": seed,
        "baseline_commit": prereg_meta.get("baseline_commit") or "",
        "baseline_preregistration_version": prereg_meta.get("preregistration_version") or "",
        "excluded_audit_count": len(excluded_ids),
        "excluded_claim_ids_sha256": digest(sorted(excluded_ids)),
        "strata_quotas": dict(strata_quotas),
        "strata_pool_sizes": dict(pool_sizes),
        "total_items": len(ordered_items),
        "unique_prs_count": len(pr_diffs_by_id),
        "blind_key_sha256": blind_key_sha,
        "manifest_items_sha256": manifest_digest,
        "items": manifest_items,
        "storage_policy": (
            "Zero third-party PR titles, descriptions, diffs, claim texts, or strata are stored inside git. "
            "Human annotation sheets and interactive diff viewer live strictly externally under AIDev_phase3/diagnostic_120/."
        ),
    }
    if manifest_out:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_out)), exist_ok=True)
        with open(manifest_out, "w", encoding="utf-8") as mfh:
            json.dump(manifest, mfh, indent=2, sort_keys=True)
    return manifest


def _load_pr_diff_js(js_path: str) -> dict[str, Any]:
    raw = open(js_path, encoding="utf-8").read().strip()
    m = re.match(r"^window\.__DIAG_RECEIVE_DIFF__\(\s*[^,]+,\s*(\{.*\})\s*\);?$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Unexpected diff JS format in {js_path}")
    return json.loads(m.group(1))


def _load_viewer_index_js(js_path: str) -> list[dict[str, Any]]:
    raw = open(js_path, encoding="utf-8").read().strip()
    m = re.match(r"^window\.__DIAG_INDEX__\s*=\s*(\[.*\]);?$", raw, re.DOTALL)
    if not m:
        raise ValueError(f"Unexpected viewer_index.js format in {js_path}")
    return json.loads(m.group(1))


def _parse_csv_bool(val: Any) -> bool | None:
    if isinstance(val, bool):
        return val
    s = str(val or "").strip().lower()
    if s in ("true", "1", "yes", "si", "sí"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _load_annotations_file(path: str) -> list[dict[str, Any]]:
    if path.lower().endswith(".csv"):
        rows: list[dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as fh:
            for raw in csv.DictReader(fh):
                gp_raw = (raw.get("gold_premises_json") or "").strip()
                cp_raw = (raw.get("contradicting_premises_json") or "").strip()
                rows.append(
                    {
                        **raw,
                        "diff_contradicts_claim": _parse_csv_bool(raw.get("diff_contradicts_claim")),
                        "gold_premises": json.loads(gp_raw) if gp_raw else [],
                        "contradicting_premises": json.loads(cp_raw) if cp_raw else [],
                    }
                )
        return rows
    raw_txt = open(path, encoding="utf-8").read().strip()
    if not raw_txt:
        return []
    if raw_txt.startswith("["):
        return json.loads(raw_txt)
    return [json.loads(ln) for ln in raw_txt.splitlines() if ln.strip()]


def validate_annotations(
    annotations_path: str,
    pack_dir: str,
) -> tuple[bool, dict[str, Any]]:
    """Strictly validate completed human annotations against *pack_dir* items and verbatim PR diff lines."""
    index_items = _load_viewer_index_js(os.path.join(pack_dir, "viewer_index.js"))
    expected_by_id = {it["item_id"]: it for it in index_items}
    records = _load_annotations_file(annotations_path)

    errors: list[str] = []
    seen_ids: set[str] = set()
    status_counts: collections.Counter[str] = collections.Counter()
    diff_cache: dict[str, dict[str, Any]] = {}

    for rec_idx, rec in enumerate(records):
        iid = str(rec.get("item_id") or "").strip()
        if not iid:
            errors.append(f"Row {rec_idx}: missing item_id")
            continue
        if iid in seen_ids:
            errors.append(f"{iid}: duplicate item_id")
            continue
        seen_ids.add(iid)
        exp = expected_by_id.get(iid)
        if exp is None:
            errors.append(f"{iid}: unknown item_id not in pack")
            continue
        if rec.get("claim_id") and str(rec["claim_id"]) != str(exp["claim_id"]):
            errors.append(f"{iid}: claim_id mismatch ({rec['claim_id']} != {exp['claim_id']})")
        if rec.get("pr_id") and str(rec["pr_id"]) != str(exp["pr_id"]):
            errors.append(f"{iid}: pr_id mismatch ({rec['pr_id']} != {exp['pr_id']})")

        st = str(rec.get("annotation_status") or "").strip()
        if st not in VALID_ANNOTATION_STATUSES:
            errors.append(f"{iid}: invalid annotation_status {st!r}")
        else:
            status_counts[st] += 1

        contra = rec.get("diff_contradicts_claim")
        if not isinstance(contra, bool):
            errors.append(f"{iid}: diff_contradicts_claim must be boolean true/false, got {contra!r}")
        elif st == "contradicted" and not contra:
            errors.append(f"{iid}: annotation_status='contradicted' requires diff_contradicts_claim=True")

        op = str(rec.get("gold_operation") or "").strip()
        if op not in VALID_OPERATIONS:
            errors.append(f"{iid}: invalid gold_operation {op!r}")

        conf = str(rec.get("confidence") or "").strip()
        if conf not in VALID_CONFIDENCES:
            errors.append(f"{iid}: invalid confidence {conf!r}")

        gold_prems = rec.get("gold_premises")
        contra_prems = rec.get("contradicting_premises")
        if not isinstance(gold_prems, list) or not isinstance(contra_prems, list):
            errors.append(f"{iid}: gold_premises and contradicting_premises must be lists")
            continue

        if st == "supported" and len(gold_prems) == 0:
            errors.append(f"{iid}: annotation_status='supported' requires >= 1 gold_premise")
        if st in ("no_premises", "out_of_scope_meta") and len(gold_prems) > 0:
            errors.append(f"{iid}: annotation_status={st!r} must have empty gold_premises")

        pr_id = str(exp["pr_id"])
        if pr_id not in diff_cache:
            diff_cache[pr_id] = _load_pr_diff_js(os.path.join(pack_dir, "pr_diffs", f"pr_{pr_id}.js"))
        pr_diff = diff_cache[pr_id]
        all_lines = {
            ln["text"]
            for ch in pr_diff.get("chunks", ())
            for ln in ch.get("lines", ())
        }
        lines_by_file: dict[str, set[str]] = collections.defaultdict(set)
        for ch in pr_diff.get("chunks", ()):
            cf = ch["file"]
            for ln in ch.get("lines", ()):
                lines_by_file[cf].add(ln["text"])

        for field_name, plist in (("gold_premises", gold_prems), ("contradicting_premises", contra_prems)):
            for p_i, p in enumerate(plist):
                if not isinstance(p, Mapping):
                    errors.append(f"{iid}: {field_name}[{p_i}] is not an object")
                    continue
                p_file = str(p.get("file") or "").strip()
                p_quote = str(p.get("literal_quote") or "")
                if not p_file or not p_quote:
                    errors.append(f"{iid}: {field_name}[{p_i}] requires non-empty 'file' and 'literal_quote'")
                    continue
                if p_file not in lines_by_file:
                    errors.append(f"{iid}: {field_name}[{p_i}] file {p_file!r} not in PR diff")
                elif p_quote not in lines_by_file[p_file] and p_quote not in all_lines:
                    errors.append(
                        f"{iid}: {field_name}[{p_i}] literal_quote not found verbatim in {p_file!r}"
                    )

    missing = sorted(set(expected_by_id) - seen_ids)
    if missing:
        errors.append(f"Missing {len(missing)} items (first 10: {missing[:10]})")

    summary = {
        "valid": len(errors) == 0,
        "total_expected": len(expected_by_id),
        "total_present": len(seen_ids),
        "missing_count": len(missing),
        "status_counts": dict(sorted(status_counts.items())),
        "error_count": len(errors),
        "errors": errors[:50],
    }
    return len(errors) == 0, summary


def render_cli_item_diff(
    pack_dir: str,
    item_id: str,
    *,
    search: str | None = None,
    file_filter: str | None = None,
    max_lines_per_chunk: int = 200,
) -> str:
    """Format an item's claim + numbered PR diff hunks cleanly for terminal inspection."""
    index_items = _load_viewer_index_js(os.path.join(pack_dir, "viewer_index.js"))
    item = next((it for it in index_items if it["item_id"] == item_id), None)
    if item is None:
        raise KeyError(f"Unknown item_id: {item_id}")
    pr_diff = _load_pr_diff_js(os.path.join(pack_dir, "pr_diffs", f"pr_{item['pr_id']}.js"))
    search_l = (search or "").strip().lower()
    file_l = (file_filter or "").strip().lower()

    out: list[str] = [
        f"=== {item['item_id']} | PR {item['pr_id']} | {item['repo']} ===",
        f"URL   : {item['pr_url']}",
        f"TITLE : {item['pr_title']}",
        f"CLAIM : {item['claim_text']}",
        f"FILES ({pr_diff['diff_files_count']}): {', '.join(pr_diff.get('files', ())[:15])}",
        "",
    ]
    for ch in pr_diff.get("chunks", ()):
        fname = ch["file"]
        if file_l and file_l not in fname.lower():
            continue
        lines = ch.get("lines", ())
        if search_l and not any(search_l in ln["text"].lower() or search_l in fname.lower() for ln in lines):
            continue
        out.append(f"--- {fname} [{ch['chunk_id']}] (+{ch['adds']}/-{ch['dels']}) ---")
        shown = 0
        for ln in lines:
            if search_l and search_l not in ln["text"].lower() and ln["kind"] != "hunk":
                continue
            out.append(f"  L{ln['line_no']:04d} | {ln['text']}")
            shown += 1
            if shown >= max_lines_per_chunk:
                out.append(f"  ... ({len(lines) - shown} more lines in hunk; pass --max-lines to expand)")
                break
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 n=120 diagnostic human annotation pack tool.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("build", help="Generate external n=120 stratified pack and sanitized manifest.")
    ap_b.add_argument("--aidev-dir", default=DEFAULT_AIDEV_DIR)
    ap_b.add_argument("--preregistration", default=DEFAULT_PREREG)
    ap_b.add_argument("--exclude-claim-ids", default="")
    ap_b.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap_b.add_argument("--manifest-out", default=DEFAULT_MANIFEST_OUT)
    ap_b.add_argument("--seed", type=int, default=DIAGNOSTIC_SEED)

    ap_v = sub.add_parser("validate", help="Validate completed human annotations (.jsonl or .csv).")
    ap_v.add_argument("--annotations", required=True)
    ap_v.add_argument("--pack-dir", default=DEFAULT_OUT_DIR)

    ap_view = sub.add_parser("view", help="Terminal diff viewer for a specific DIAG-XXX item.")
    ap_view.add_argument("item_id")
    ap_view.add_argument("--pack-dir", default=DEFAULT_OUT_DIR)
    ap_view.add_argument("--search", default="")
    ap_view.add_argument("--file", default="")
    ap_view.add_argument("--max-lines", type=int, default=200)

    args = ap.parse_args()
    if args.cmd == "build":
        orig_path = os.path.join(args.aidev_dir, "aidev_dev_originals.jsonl")
        prs = [json.loads(ln) for ln in open(orig_path, encoding="utf-8") if ln.strip()]
        m1 = p3.ResilientCachedModel(None, os.path.join(args.aidev_dir, "caches", "pass1_dev.jsonl"))
        m2 = p3.ResilientCachedModel(
            None,
            os.path.join(args.aidev_dir, "caches", "pass2_dev.jsonl"),
            allow_pass3_replay_miss_as_empty=True,
        )
        rel_cfg, th_cfg, prereg_meta = load_frozen_prereg_config(args.preregistration)
        excluded = load_excluded_claim_ids(args.exclude_claim_ids or None)
        ordered, pool_sizes, pr_diffs = sample_diagnostic_claims(
            prs,
            pass1_model=m1,
            pass2_model=m2,
            relevance_config=rel_cfg,
            thresholds=th_cfg,
            excluded_claim_ids=excluded,
            strata_quotas=DEFAULT_STRATA,
            seed=args.seed,
        )
        manifest = write_diagnostic_pack(
            ordered,
            pr_diffs,
            out_dir=args.out_dir,
            manifest_out=args.manifest_out,
            pool_sizes=pool_sizes,
            strata_quotas=DEFAULT_STRATA,
            excluded_ids=excluded,
            prereg_meta=prereg_meta,
            seed=args.seed,
        )
        print(
            json.dumps(
                {k: v for k, v in manifest.items() if k != "items"},
                indent=2,
                sort_keys=True,
            )
        )
    elif args.cmd == "validate":
        ok, summary = validate_annotations(args.annotations, args.pack_dir)
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
        raise SystemExit(0 if ok else 1)
    elif args.cmd == "view":
        print(
            render_cli_item_diff(
                args.pack_dir,
                args.item_id,
                search=args.search or None,
                file_filter=args.file or None,
                max_lines_per_chunk=args.max_lines,
            )
        )


if __name__ == "__main__":
    main()
