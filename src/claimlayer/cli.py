"""Command-line interface for ClaimLayer computable verification."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from claimlayer.ledger import Ledger
from claimlayer.mcp_server import run_stdio_server
from claimlayer.models import Assertion, FactRecord, Policy, Source
from claimlayer.semantic.adapters import OpenAICompatibleModel
from claimlayer.v4 import SourceDocument, verify_claim, verify_plan
from claimlayer.verify import verify


def _cmd_verify(bundle_path: Path, db_path: str | None) -> int:
    raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    ledger = Ledger(
        sources=raw.get("sources") or [],
        facts=raw.get("facts") or [],
        db_path=db_path,
    )
    verdict = ledger.verify(
        assertions=raw.get("assertions") or [],
        policy=raw.get("policy"),
        as_of=raw.get("as_of"),
        agent_id=str(raw.get("agent_id") or "cli_agent"),
        record=bool(db_path),
    )
    print(json.dumps(verdict.to_dict(), indent=2, ensure_ascii=False))
    if verdict.status == "PASS":
        return 0
    if verdict.status == "RETRY":
        return 2
    return 3


def _cmd_demo() -> int:
    print("=== ClaimLayer v3: Computable Verification Demo (Unblocking Agents) ===\n")
    ledger = Ledger(
        sources=[
            Source(
                id="msa_2024",
                title="Master Services Agreement (2024)",
                text="Section 4.1: ACME shall pay all undisputed invoices within 45 days. Governing law is Delaware.",
                tier="standard",
            ),
            Source(
                id="amendment_2026",
                title="Signed First Amendment (2026)",
                text="Section 4.1 is hereby amended: ACME shall pay all undisputed invoices within 30 days.",
                tier="authoritative",
                supersedes=["msa_2024"],
            ),
        ],
        facts=[
            FactRecord(
                subject="ACME",
                predicate="payment_terms",
                value="30 days",
                source_id="amendment_2026",
                quote="ACME shall pay all undisputed invoices within 30 days.",
            )
        ],
    )
    policy = Policy(
        policy_id="contract_ops_v1",
        required_predicates=["payment_terms", "governing_law"],
    )

    # Attempt 1: Agent cites obsolete MSA 45 days and misses governing_law -> RETRY (0 human time!)
    v1 = verify(
        assertions=[
            Assertion(
                subject="ACME",
                predicate="payment_terms",
                value="45 days",
                source_id="msa_2024",
                quote="ACME shall pay all undisputed invoices within 45 days.",
            )
        ],
        ledger=ledger,
        policy=policy,
        agent_id="contract_agent_01",
    )
    print(f"[Turn 1] Agent submits flawed draft -> Verdict: {v1.status} (Requires Human: {v1.requires_human})")
    for issue in v1.issues:
        print(f"   - [{issue.code}] {issue.message}")
        print(f"     Fix -> {issue.remediation}")

    # Attempt 2: Agent self-corrects in milliseconds -> PASS (Auto-approved with Merkle proof!)
    v2 = verify(
        assertions=[
            Assertion(
                subject="ACME",
                predicate="payment_terms",
                value="30 days",
                source_id="amendment_2026",
                quote="ACME shall pay all undisputed invoices within 30 days.",
            ),
            Assertion(
                subject="ACME",
                predicate="governing_law",
                value="Delaware",
                source_id="msa_2024",
                quote="Governing law is Delaware.",
            ),
        ],
        ledger=ledger,
        policy=policy,
        agent_id="contract_agent_01",
    )
    print(f"\n[Turn 2] Agent self-corrects -> Verdict: {v2.status} (Allowed to Proceed: {v2.allowed_to_proceed})")
    print(f"   - Confidence: {v2.confidence:.2f} | Merkle Root: {v2.receipt.merkle_root[:24]}...")
    print(f"   - Offline Cryptographic Verification: {v2.verify_receipt()}")
    return 0


def _load_cli_sources(args: argparse.Namespace) -> dict[str, SourceDocument]:
    docs: dict[str, SourceDocument] = {}
    if getattr(args, "sources_json", None):
        raw = json.loads(Path(args.sources_json).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, str):
                    docs[str(k)] = SourceDocument(document_id=str(k), text=v, authority_rank=80, weight=1.0)
                elif isinstance(v, dict):
                    did = str(v.get("document_id") or v.get("id") or k)
                    docs[did] = SourceDocument(
                        document_id=did,
                        text=str(v["text"]),
                        authority_rank=int(v.get("authority_rank", 80)),
                        weight=float(v.get("weight", 1.0)),
                        dependence_group=v.get("dependence_group"),
                        attribution=str(v.get("attribution", "source_statement")),
                    )
        elif isinstance(raw, list):
            for idx, v in enumerate(raw):
                if isinstance(v, str):
                    did = f"doc_{idx + 1}"
                    docs[did] = SourceDocument(document_id=did, text=v, authority_rank=80, weight=1.0)
                elif isinstance(v, dict):
                    did = str(v.get("document_id") or v.get("id") or f"doc_{idx + 1}")
                    docs[did] = SourceDocument(
                        document_id=did,
                        text=str(v["text"]),
                        authority_rank=int(v.get("authority_rank", 80)),
                        weight=float(v.get("weight", 1.0)),
                        dependence_group=v.get("dependence_group"),
                        attribution=str(v.get("attribution", "source_statement")),
                    )

    auto_idx = len(docs) + 1
    for item in getattr(args, "source", None) or ():
        if "=" in item:
            cand_id, body = item.split("=", 1)
            if re.fullmatch(r"[A-Za-z0-9_.:-]+", cand_id):
                docs[cand_id] = SourceDocument(document_id=cand_id, text=body, authority_rank=80, weight=1.0)
                continue
        did = f"doc_{auto_idx}"
        auto_idx += 1
        docs[did] = SourceDocument(document_id=did, text=item, authority_rank=80, weight=1.0)

    for fspec in getattr(args, "source_file", None) or ():
        if "=" in fspec and re.fullmatch(r"[A-Za-z0-9_.:-]+", fspec.split("=", 1)[0]):
            did, fpath_s = fspec.split("=", 1)
            fpath = Path(fpath_s)
        else:
            fpath = Path(fspec)
            did = fpath.stem or f"doc_{auto_idx}"
            auto_idx += 1
        docs[did] = SourceDocument(
            document_id=did,
            text=fpath.read_text(encoding="utf-8"),
            authority_rank=80,
            weight=1.0,
        )
    return docs


def _cmd_verify_claim(args: argparse.Namespace) -> int:
    docs = _load_cli_sources(args)
    if not docs:
        print(
            json.dumps(
                {
                    "epistemic_verdict": "unresolved",
                    "outcome_category": "malformed_proposal",
                    "operational_routing": {"status": "ESCALATE", "allowed": False, "requires_human": True, "reasons": ["MISSING_SOURCES"]},
                    "explanation": "At least one source (--source, --source-file, or --sources-json) is required.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3

    if getattr(args, "plan_json", None):
        plan_payload: Any = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
        result = verify_plan(docs, args.claim, plan_payload, as_of=args.as_of, unit_profile=args.unit_profile)
    else:
        api_key = os.environ.get(args.api_key_env, "").strip()
        if not api_key:
            print(
                json.dumps(
                    {
                        "epistemic_verdict": "unresolved",
                        "outcome_category": "provider_error",
                        "operational_routing": {
                            "status": "ESCALATE",
                            "allowed": False,
                            "requires_human": True,
                            "reasons": ["MISSING_PROVIDER_API_KEY"],
                        },
                        "explanation": f"Missing environment variable '{args.api_key_env}' for live planner.",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
            return 3
        model = OpenAICompatibleModel(
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            timeout=float(args.timeout),
            max_tokens=2048,
            retries=2,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)}},
        )
        result = verify_claim(
            docs,
            args.claim,
            model=model,
            as_of=args.as_of,
            max_retries=int(args.max_retries),
            unit_profile=args.unit_profile,
        )

    rendered = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if getattr(args, "out", None):
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    status = result.operational_routing.status
    if status == "PASS":
        return 0
    if status == "RETRY":
        return 2
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claimlayer",
        description="ClaimLayer v3/v4 — Deterministic Computable Verification for AI Agents",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify", help="Verify an agent JSON proposal bundle")
    p_verify.add_argument("--bundle", "-b", required=True, type=Path, help="Path to JSON bundle")
    p_verify.add_argument("--db", default=None, help="Optional SQLite ledger path")

    p_vc = sub.add_parser("verify-claim", help="Verify a natural-language claim against corpus sources (v4)")
    p_vc.add_argument("--claim", "-c", required=True, help="Natural-language claim to contrast")
    p_vc.add_argument("--source", "-s", action="append", default=[], help="Source text as 'doc_id=text' or plain text")
    p_vc.add_argument("--source-file", "-f", action="append", default=[], help="Source file as 'doc_id=path' or path")
    p_vc.add_argument("--sources-json", type=Path, default=None, help="Path to JSON mapping/list of sources")
    p_vc.add_argument("--plan-json", type=Path, default=None, help="Optional pre-generated planner JSON (offline mode)")
    p_vc.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1", help="OpenAI-compatible endpoint base URL")
    p_vc.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b", help="Planner model ID")
    p_vc.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="Environment variable holding provider API key")
    p_vc.add_argument("--timeout", type=float, default=90.0, help="Provider request timeout in seconds")
    p_vc.add_argument("--max-retries", type=int, default=1, help="Bounded planner repair retries")
    p_vc.add_argument("--enable-thinking", action="store_true", help="Enable model thinking if supported")
    p_vc.add_argument("--as-of", default=None, help="Optional explicit reference date")
    p_vc.add_argument("--unit-profile", choices=("auto", "standard", "es"), default="auto",
                      help="Unit-label profile: 'auto' picks 'es' for Spanish text, else 'standard'")
    p_vc.add_argument("--out", type=Path, default=None, help="Optional path to write JSON verification artifact")

    p_mcp = sub.add_parser("mcp", help="Start ClaimLayer MCP server over stdio")
    p_mcp.add_argument("--db", default=None, help="Optional SQLite ledger path")

    sub.add_parser("demo", help="Run a quick self-contained verification loop demo")

    args = parser.parse_args(argv)
    if args.command == "verify":
        return _cmd_verify(args.bundle, args.db)
    if args.command == "verify-claim":
        return _cmd_verify_claim(args)
    if args.command == "mcp":
        return run_stdio_server(db_path=args.db)
    return _cmd_demo()


if __name__ == "__main__":
    raise SystemExit(main())
