#!/usr/bin/env python3
"""Executable Harness² `c7` gate script (`claimlayer_gate.py`) for OpenCode / Harvey LAB agents.

Commands:
  1. `python3 scripts/claimlayer_gate.py prepare [documents_dir]`
     Extracts plain text from task documents, registers them in `.claimlayer/ledger.sqlite`
     with deterministic authority tiers (Indenture/MSA = authoritative, OM/Markup = standard,
     Email/Unsigned = draft), seals the policy inside SQLite, and prints every divergent section
     that MUST be verified before completion.

  2. `python3 scripts/claimlayer_gate.py verify <proposal.json>`
     Verifies that `.claimlayer/policy.json` has not been tampered with against the sealed
     policy in `.claimlayer/ledger.sqlite`, then runs `claimlayer.verify()`.
     - Exit 0 (`PASS`): All divergent sections and quotes verified; writes `.claimlayer/receipt.json`.
     - Exit 2 (`RETRY`): Prints structured JSON remediation errors for immediate self-correction.
     - Exit 3 (`ESCALATE`): Genuine peer conflict or threshold exception documented for human sign-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# Locate claimlayer package from bundled scripts/claimlayer, repo checkout, or CLAIMLAYER_ROOT
_root_env = os.environ.get("CLAIMLAYER_ROOT", "").strip()
_candidate_src_paths = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[4] / "src",
    Path(__file__).resolve().parents[3] / "src",
]
if _root_env:
    _candidate_src_paths.append(Path(_root_env) / "src")
for candidate in _candidate_src_paths:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from claimlayer import Ledger, Policy, Source, build_comparison_policy


def _read_doc_text(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    lab_root = os.environ.get("LAB_BENCH_ROOT")
    if lab_root:
        parser_dir = Path(lab_root) / "sandbox" / "parsers"
        if (parser_dir / "parse_doc.py").is_file():
            if str(parser_dir) not in sys.path:
                sys.path.insert(0, str(parser_dir))
            import parse_doc  # type: ignore[import-not-found]

            if ext in parse_doc.PARSERS:
                return str(parse_doc.PARSERS[ext](str(path)))
    return path.read_text(encoding="utf-8", errors="replace")


def _classify_tier(filename: str, manifest_tiers: dict[str, str] | None = None) -> str:
    if manifest_tiers and filename in manifest_tiers:
        return manifest_tiers[filename]
    low = filename.lower()
    tokens = set(re.split(r"[^a-z0-9]+", Path(low).stem))
    if (
        "offering-memorandum" in low
        or "offering_memorandum" in low
        or "offering memorandum" in low
        or tokens & {"om", "prospectus", "markup", "redline", "proposed", "summary"}
    ):
        return "standard"
    if low.endswith(".eml") or tokens & {
        "email",
        "emails",
        "note",
        "notes",
        "comment",
        "comments",
        "wip",
        "negotiation",
        "memo",
        "working",
        "feedback",
        "discussion",
        "instruction",
        "instructions",
        "unsigned",
        "unexecuted",
        "v0",
        "draft",
        "preliminary",
        "chat",
        "slack",
    }:
        return "draft"
    if tokens & {
        "executed",
        "signed",
        "final",
        "governing",
        "statute",
        "charter",
    }:
        return "authoritative"
    return "standard"


def _compute_canonical_policy(sources: list[Source]) -> tuple[Policy, list[dict[str, Any]]]:
    auth_sources = [s for s in sources if s.tier == "authoritative"]
    std_sources = [s for s in sources if s.tier == "standard"]

    required_predicates: list[str] = []
    divergence_report: list[dict[str, Any]] = []
    required_values: dict[str, list[str]] = {}

    if auth_sources and std_sources:
        pairs = [(a, b) for a in auth_sources for b in std_sources]
    else:
        comparable = auth_sources or std_sources
        pairs = [
            (comparable[i], comparable[j])
            for i in range(len(comparable))
            for j in range(i + 1, len(comparable))
        ]

    for pri_src, sec_src in pairs:
        comparison, divergences = build_comparison_policy(pri_src, sec_src)
        for key, values in comparison.required_value_tokens.items():
            required_values[key] = sorted(set(required_values.get(key, [])) | set(values))
        for d in divergences:
            if d.section_key not in required_predicates:
                required_predicates.append(d.section_key)
            divergence_report.append(d.to_dict())

    policy = Policy(
        policy_id="harness2_computable_gate",
        min_tier_rank=100 if auth_sources else 50,
        required_predicates=required_predicates,
        required_source_ids=[s.id for s in auth_sources],
        required_value_tokens=required_values,
    )
    return policy, divergence_report


def _external_seal_path(workspace: Path) -> Path:
    ws_hash = hashlib.sha256(str(workspace.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"claimlayer_gate_seal_{ws_hash}.json"


def _load_sources_from_dir(docs_dir: Path) -> list[Source]:
    sources: list[Source] = []
    if not docs_dir.is_dir():
        return sources
    manifest_tiers: dict[str, str] = {}
    manifest_file = docs_dir / "manifest.json"
    if manifest_file.is_file():
        try:
            raw_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if isinstance(raw_manifest, dict):
                manifest_tiers = {str(k): str(v) for k, v in raw_manifest.items()}
        except Exception:
            manifest_tiers = {}
    for f in sorted(docs_dir.iterdir()):
        if not f.is_file() or f.name.startswith(".") or f.name == "manifest.json":
            continue
        sources.append(
            Source(
                id=f.name,
                title=f.name,
                text=_read_doc_text(f),
                tier=_classify_tier(f.name, manifest_tiers),
            )
        )
    return sources


def cmd_prepare(docs_dir: Path, workspace: Path) -> int:
    state_dir = workspace / ".claimlayer"
    db_path = state_dir / "ledger.sqlite"
    seal_file = _external_seal_path(workspace)
    if db_path.exists() or (state_dir / "policy.json").exists() or seal_file.exists():
        return _fail_tampered(
            "WORKSPACE_ALREADY_SEALED",
            "Workspace has already been prepared and sealed; re-running prepare is forbidden.",
            "Do not re-run prepare after initial workspace initialization.",
        )

    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(db_path=db_path)
    sources = _load_sources_from_dir(docs_dir)
    for src in sources:
        ledger.add_source(src)

    policy, divergence_report = _compute_canonical_policy(sources)
    ledger.save_sealed_policy(policy)

    (state_dir / "policy.json").write_text(
        json.dumps(policy.to_dict(), indent=2), encoding="utf-8"
    )
    seal_payload = {
        "docs_dir": str(docs_dir.resolve()),
        "sources": {
            s.id: hashlib.sha256(s.text.encode("utf-8")).hexdigest() for s in sources
        },
        "policy": policy.to_dict(),
    }
    seal_file.write_text(
        json.dumps(seal_payload, sort_keys=True), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "sources_registered": [
                    {"id": s.id, "tier": s.tier, "rank": s.rank} for s in sources
                ],
                "mandatory_divergent_sections": divergence_report,
                "next_step": (
                    "Analyze every mandatory section, write structured assertions to proposal.json, "
                    "and run `python3 scripts/claimlayer_gate.py verify proposal.json` before finalizing output/."
                ),
            },
            indent=2,
        )
    )
    return 0


def _fail_tampered(code: str, message: str, remediation: str) -> int:
    tampered_out = {
        "status": "RETRY",
        "allowed_to_proceed": False,
        "requires_human": False,
        "issues": [
            {
                "code": code,
                "disposition": "RETRY",
                "message": message,
                "remediation": remediation,
            }
        ],
    }
    print(json.dumps(tampered_out, indent=2))
    return 2


def cmd_verify(proposal_path: Path, workspace: Path) -> int:
    state_dir = workspace / ".claimlayer"
    ledger = Ledger(db_path=state_dir / "ledger.sqlite")
    sealed_policy_dict = ledger.load_sealed_policy()

    seal_file = _external_seal_path(workspace)
    if not seal_file.is_file():
        return _fail_tampered(
            "SEAL_MISSING",
            "External workspace seal is missing; cannot verify unsealed workspace.",
            "Run prepare once in a clean workspace before running verify.",
        )
    try:
        ext_seal = json.loads(seal_file.read_text(encoding="utf-8"))
    except Exception:
        return _fail_tampered(
            "SEAL_CORRUPTED",
            "External workspace seal is unreadable or corrupted.",
            "Do not modify the external workspace seal.",
        )

    docs_dir = Path(ext_seal.get("docs_dir") or (workspace / "documents"))
    disk_sources = _load_sources_from_dir(docs_dir)
    ledger_sources = ledger.list_sources()

    # 1. Require strict triple agreement across disk documents, external seal, and SQLite ledger
    seal_hashes = ext_seal.get("sources") or {}
    disk_hashes = {
        s.id: hashlib.sha256(s.text.encode("utf-8")).hexdigest() for s in disk_sources
    }
    actual_hashes = {
        s.id: hashlib.sha256(s.text.encode("utf-8")).hexdigest() for s in ledger_sources
    }
    if not disk_hashes or actual_hashes != seal_hashes or actual_hashes != disk_hashes:
        return _fail_tampered(
            "LEDGER_TAMPERED",
            "Sources in .claimlayer/ledger.sqlite or external seal do not match the original source documents.",
            "Do not modify .claimlayer/ledger.sqlite, the seal, or the source documents.",
        )

    recomputed_policy, _ = _compute_canonical_policy(disk_sources)
    canonical_policy_dict = recomputed_policy.to_dict()
    seal_policy_dict = ext_seal.get("policy")

    # 2. Require strict triple agreement across recomputed policy, external seal, SQLite sealed_policy, and policy.json
    if (
        seal_policy_dict != canonical_policy_dict
        or sealed_policy_dict is None
        or sealed_policy_dict != canonical_policy_dict
    ):
        return _fail_tampered(
            "POLICY_TAMPERED",
            "Sealed policy in .claimlayer/ledger.sqlite or external seal was modified and does not match the canonical document comparison policy.",
            "Do not modify .claimlayer/ledger.sqlite or the seal; satisfy the original sealed verification policy.",
        )

    policy_file = state_dir / "policy.json"
    if policy_file.exists():
        try:
            disk_policy_dict = json.loads(policy_file.read_text(encoding="utf-8"))
        except Exception:
            disk_policy_dict = {}
        if disk_policy_dict != canonical_policy_dict:
            return _fail_tampered(
                "POLICY_TAMPERED",
                "Workspace .claimlayer/policy.json was modified and does not match the sealed ledger policy.",
                "Do not modify .claimlayer/policy.json; satisfy the original sealed verification policy.",
            )

    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))

    verdict = ledger.verify(
        assertions=proposal.get("assertions") or [],
        policy=canonical_policy_dict,
        as_of=proposal.get("as_of"),
        agent_id="harness2_opencode_agent",
        record=True,
    )
    out = verdict.to_dict()
    (state_dir / "receipt.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    if verdict.status == "PASS":
        return 0
    if verdict.status == "RETRY":
        return 2
    return 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("docs_dir", nargs="?", default="documents")
    p_ver = sub.add_parser("verify")
    p_ver.add_argument("proposal", help="Path to agent structured assertions JSON")

    args = ap.parse_args()
    ws = Path.cwd()
    if args.cmd == "prepare":
        return cmd_prepare((ws / args.docs_dir).resolve(), ws)
    return cmd_verify((ws / args.proposal).resolve(), ws)


if __name__ == "__main__":
    raise SystemExit(main())
