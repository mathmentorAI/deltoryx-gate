"""Command-line entry point for the live six-layer Gate."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from claimlayer.semantic.models import SourceDocument
from claimlayer.v4 import verify_claim


def _load_cli_sources(args: argparse.Namespace) -> dict[str, Any]:
    docs: dict[str, SourceDocument] = {}
    if getattr(args, "sources_json", None):
        raw = json.loads(Path(args.sources_json).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            docs.update(raw)
        elif isinstance(raw, list):
            for idx, item in enumerate(raw, 1):
                did = str(item.get("document_id") or item.get("id") or f"doc_{idx}") if isinstance(item, dict) else f"doc_{idx}"
                if did in docs:
                    raise ValueError(f"Duplicate document_id: {did}")
                docs[did] = item
        else:
            raise ValueError("sources-json must contain an object or list")

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
        docs[did] = {"document_id": did, "text": fpath.read_text(encoding="utf-8"),
                     "path": str(fpath), "kind": "file", "completeness": "complete_for_declared_scope"}

    return docs


def _cmd_verify_claim(args: argparse.Namespace) -> int:
    docs = _load_cli_sources(args)
    if not docs:
        print(
            json.dumps(
                {
                    "contract_version": "gate-layers-1",
                    "stage": "context", "code": "MISSING_SOURCES",
                    "explanation": "At least one source (--source, --source-file, or --sources-json) is required.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 3

    from .v4 import build_derivation_planner_model
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        print(json.dumps({"contract_version": "gate-layers-1", "stage": "infrastructure",
                          "code": "MISSING_PROVIDER_API_KEY", "message": f"Set {args.api_key_env}"}), file=sys.stderr)
        return 3
    model = build_derivation_planner_model(api_key, base_url=args.base_url, model=args.model,
                                           timeout=float(args.timeout), max_tokens=int(args.max_tokens),
                                           enable_thinking=bool(args.enable_thinking))
    result = verify_claim(docs, args.claim, model=model, as_of=args.as_of,
                          unit_profile=args.unit_profile, max_retries=int(args.max_retries))
    rendered = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if getattr(args, "out", None):
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    status = result.decision.payload["disposition"]
    if status in ("PASS", "INFORM"):
        return 0
    if status == "RETRY":
        return 2
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deltoryx-gate",
        description="Deltoryx Gate — live six-layer verification",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_vc = sub.add_parser("verify-claim", help="Verify a natural-language claim against corpus sources (v4)")
    p_vc.add_argument("--claim", "-c", required=True, help="Natural-language claim to contrast")
    p_vc.add_argument("--source", "-s", action="append", default=[], help="Source text as 'doc_id=text' or plain text")
    p_vc.add_argument("--source-file", "-f", action="append", default=[], help="Source file as 'doc_id=path' or path")
    p_vc.add_argument("--sources-json", type=Path, default=None, help="Path to JSON mapping/list of sources")
    p_vc.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1", help="OpenAI-compatible endpoint base URL")
    p_vc.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b", help="Planner model ID")
    p_vc.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="Environment variable holding provider API key")
    p_vc.add_argument("--timeout", type=float, default=90.0, help="Provider request timeout in seconds")
    p_vc.add_argument("--max-retries", type=int, default=1, help="Bounded planner repair retries")
    p_vc.add_argument("--enable-thinking", action="store_true", help="Enable model thinking if supported")
    p_vc.add_argument("--as-of", default=None, help="Optional explicit reference date")
    p_vc.add_argument("--unit-profile", choices=("auto", "standard", "es"), default="auto",
                      help="Unit-label profile: 'auto' picks 'es' for Spanish text, else 'standard'")

    p_vc.add_argument("--max-tokens", type=int, default=8192, help="Planner output token budget")
    p_vc.add_argument("--out", type=Path, default=None, help="Optional path to write JSON verification artifact")

    args = parser.parse_args(argv)
    return _cmd_verify_claim(args)


if __name__ == "__main__":
    raise SystemExit(main())
