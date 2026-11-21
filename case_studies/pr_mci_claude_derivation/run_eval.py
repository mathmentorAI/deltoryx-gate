#!/usr/bin/env python3
"""Step 3: run the frozen 20 cases through the platform's public verification entrypoint.

New engine : POST /api/v1/v4/claims/verify on a locally started `uvicorn api.main:app`
             (platform repo), real Nemotron planner configured by the application.
Old evaluators, same claim + same evidence, each under its own contract:
  * gate_v4_vocab2  (case_studies/pr_mci_claude_code/eval_v2vocab.evaluate_pr): draft = claim,
    ledger built from the same net diff; live Nemotron reader with the study's original settings.
  * rules_v2_postreview_code_config (check_claims.check_pr_corrected): deterministic, offline.

Credential: read from NVIDIA_API_KEY or the platform .env; passed only through process
environments; never printed or written. Outputs: results/live_results.json and
results/old_eval_cache.jsonl (planner completions of the old evaluator, no secrets).

Usage (from the platform repository root, on a machine that can reach the provider):
    .venv/bin/python claim-layer/case_studies/pr_mci_claude_derivation/run_eval.py
    [--skip-old] [--only C01,C02]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CL_ROOT = HERE.parents[1]
PLATFORM = CL_ROOT.parent
OLD = HERE.parent / "pr_mci_claude_code"
OUT = HERE / "results" / "live_results.json"
EXPECTED_HASHES = {
    "inputs_manifest.json": "710b8b28c684a20a2fab924fbda6c419cf1c9fce45e7b540352d69db3973985a",
    "expected_manifest.json": "fd0382c9362f088ff0256d272f053473897292f880d22f3f75e03160d5b17cad",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _api_key() -> str:
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if key:
        return key
    for env in (PLATFORM / ".env", CL_ROOT / ".env"):
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("NVIDIA_API_KEY="):
                    return s.split("=", 1)[1].strip().strip("\"'")
    return ""


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, timeout: float = 600.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode() or "{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:500]}


def _old_pr(case: dict) -> dict:
    files = [s for s in case["sources"] if s["document_id"].startswith("file_")]
    patches, names = [], []
    for s in files:
        head, _, rest = s["text"].partition("\n")
        names.append(head[len("File: "):].split(" (status:")[0])
        patches.append(rest.partition("\n")[2])
    summ = next(s["text"] for s in case["sources"] if s["document_id"] == "diff_summary")
    adds = int(summ.split("Lines added: ")[1].split("\n")[0])
    dels = int(summ.split("Lines deleted: ")[1].split("\n")[0])
    return {"pr_id": case["pr_id"], "url": case["url"], "title": "", "body": case["claim"],
            "patch": "\n---FILE---\n".join(patches), "files": names, "adds": adds, "dels": dels}


def run_rules_v2(case: dict) -> dict:
    sys.path.insert(0, str(OLD))
    import check_claims  # noqa: E402
    rec = check_claims.check_pr_corrected(_old_pr(case), evidence_profile="code_config")
    return {"verdict": rec["verdict"], "signals": rec["signals"], "claims": rec["claims"],
            "n_claims": rec["n_claims"], "n_supported": rec["n_supported"]}


def make_old_reader(key: str):
    sys.path.insert(0, str(CL_ROOT / "src"))
    sys.path.insert(0, str(OLD))
    import eval_v2vocab  # noqa: E402
    from claimlayer.semantic.adapters import OpenAICompatibleModel
    from claimlayer.v4.reader import LLMProposalReader
    base = OpenAICompatibleModel(base_url="https://integrate.api.nvidia.com/v1", model="nvidia/nemotron-3-ultra-550b-a55b",
                                 api_key=key, timeout=180, retries=3,
                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    model = eval_v2vocab.CachedModelV2(base, str(HERE / "results" / "old_eval_cache.jsonl"))
    return eval_v2vocab, LLMProposalReader(model=model), model


def run_gate_v4_vocab2(case: dict, ev2, reader) -> dict:
    row_meta = {"consensus_label": "", "annotator1_label": "n/a", "agent": "Claude_Code",
                "binary_label": "", "predicted_heuristic": ""}
    t0 = time.monotonic()
    res = ev2.evaluate_pr(row_meta, _old_pr(case), reader)
    return {"gate": res["gate"], "reasons": res["reasons"], "claim_status": res["claim_status"],
            "pred_status": res["pred_status"], "draft_rejected": res["draft_rejected"],
            "error": res["error"], "receipt_sha256": res["receipt_sha256"],
            "wall_ms": round((time.monotonic() - t0) * 1000, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, default=OUT, help="output file (default: first-measurement path)")
    ap.add_argument("--frozen-dir", type=Path, default=None,
                    help="frozen sample directory with MANIFEST_HASHES.txt (e.g. ../pr_mci_transfer_sample/frozen); "
                         "implies --skip-old and skips the old rules evaluator")
    args = ap.parse_args()
    frozen = HERE / "frozen"
    hashes = dict(EXPECTED_HASHES)
    if args.frozen_dir is not None:
        frozen = args.frozen_dir.resolve()
        hashes = {}
        for ln in (frozen / "MANIFEST_HASHES.txt").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                h, name = ln.split()
                hashes[name] = h
        if set(hashes) != {"inputs_manifest.json", "expected_manifest.json"}:
            print("ERROR: MANIFEST_HASHES.txt incompleto", file=sys.stderr)
            return 2
        args.skip_old = True
    _early = Path(args.out) if Path(args.out).is_absolute() else (Path.cwd() / args.out).resolve()
    if _early.exists():
        print(f"ERROR: {_early.name} ya existe y conserva una medición anterior; usa --out con otro archivo.", file=sys.stderr)
        return 7
    for name, h in hashes.items():
        got = _sha(frozen / name)
        if got != h:
            print(f"ERROR: {name} hash {got} != frozen {h}", file=sys.stderr)
            return 2
    key = _api_key()
    if not key:
        print("ERROR: NVIDIA_API_KEY no disponible (entorno o .env). Ejecución en vivo pendiente.", file=sys.stderr)
        return 3
    cases = json.loads((frozen / "inputs_manifest.json").read_text(encoding="utf-8"))["cases"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["case_id"] in keep]

    port = _free_port()
    env = {**os.environ, "NVIDIA_API_KEY": key, "CLAIMLAYER_EMPTY_BOOT": "1",
           "CLAIMLAYER_AUDIT_DB": str(PLATFORM / ".build" / "prmci_claude_eval_audit.db"),
           "PYTHONPATH": str(CL_ROOT / "src") + os.pathsep + str(PLATFORM)}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1",
                               "--port", str(port), "--log-level", "warning"], cwd=PLATFORM, env=env)
    base = f"http://127.0.0.1:{port}"
    ev2 = reader = None
    if not args.skip_old:
        ev2, reader, _ = make_old_reader(key)
    rows = []
    t_run0 = time.monotonic()
    try:
        for _ in range(60):
            try:
                if _http("GET", base + "/health", timeout=2)[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            print("ERROR: el servidor no arrancó", file=sys.stderr)
            return 4
        for case in cases:
            body = {"claim": case["claim"], "sources": case["sources"]}
            t0 = time.monotonic()
            status, out = _http("POST", base + "/api/v1/v4/claims/verify", body)
            wall_ms = round((time.monotonic() - t0) * 1000, 1)
            ca = out.get("claim_assessment", {})
            od = out.get("operational_decision", {})
            row = {"case_id": case["case_id"], "pr_id": case["pr_id"], "http_status": status,
                   "client_wall_ms": wall_ms, "verdict": ca.get("verdict"), "outcome_category": ca.get("outcome_category"),
                   "routing": od.get("status"), "response": out}
            rows.append(row)
            grade = (out.get("reliability") or {}).get("grade")
            props = [p.get("name") + ":" + str(p.get("grade")) for p in ((out.get("reliability") or {}).get("proposed_operations") or [])]
            print(f"{case['case_id']} {case['pr_id']}: {row['verdict']} / {row['routing']} ({row['outcome_category']}) grade={grade} proposed={props} {wall_ms} ms", flush=True)
            if args.frozen_dir is None:
                row["rules_v2_postreview_code_config"] = run_rules_v2(case)
            if ev2 is not None:
                try:
                    row["gate_v4_vocab2"] = run_gate_v4_vocab2(case, ev2, reader)
                except Exception as exc:  # recorded, never hides the new-engine result
                    row["gate_v4_vocab2"] = {"gate": "ERROR", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
                print(f"      old gate_v4_vocab2: {row['gate_v4_vocab2'].get('gate')} {row['gate_v4_vocab2'].get('claim_status')}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    artifact = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "POST /api/v1/v4/claims/verify (uvicorn api.main:app, real HTTP)",
        "platform_commit": _git("rev-parse", "HEAD", cwd=PLATFORM),
        "platform_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=PLATFORM),
        "claim_layer_commit": _git("rev-parse", "HEAD", cwd=CL_ROOT),
        "claim_layer_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=CL_ROOT),
        "code_dirty": {"platform_api": bool(_git("status", "--porcelain", "--", "api", cwd=PLATFORM)),
                       "claim_layer_src": bool(_git("status", "--porcelain", "--", "src", cwd=CL_ROOT))},
        "frozen_dir": str(frozen.relative_to(CL_ROOT)) if frozen.is_relative_to(CL_ROOT) else str(frozen),
        "frozen_manifest_hashes": hashes,
        "prompt_version": (rows[0]["response"].get("provenance", {}).get("prompt_version") if rows else None),
        "old_evaluator_settings": None if args.skip_old else {
            "gate_v4_vocab2": "eval_v2vocab.evaluate_pr, CODE_CHANGE_VOCAB_V2 620ebd71..., Nemotron timeout=180 retries=3 thinking=off",
            "rules_v2": "check_claims.check_pr_corrected(evidence_profile='code_config')"},
        "run_wall_ms": round((time.monotonic() - t_run0) * 1000, 1),
        "cases": rows,
    }
    text = json.dumps(artifact, indent=1, ensure_ascii=False)
    if key in text or "nvapi-" in text:
        print("ERROR: la salida contendría una credencial; no se escribe.", file=sys.stderr)
        return 5
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    if out_path.exists():
        print(f"ERROR: {out_path.name} ya existe y conserva una medición anterior; usa --out con otro archivo.", file=sys.stderr)
        return 7
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    cache = HERE / "results" / "old_eval_cache.jsonl"  # shared replay cache of the old evaluator
    if cache.exists() and key in cache.read_text(encoding="utf-8"):
        cache.unlink()
        print("ERROR: la caché del evaluador anterior contenía la credencial; eliminada.", file=sys.stderr)
        return 6
    print(f"\n{len(rows)} casos -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
