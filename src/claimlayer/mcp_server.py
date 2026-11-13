"""Zero-dependency Model Context Protocol (MCP) server for ClaimLayer agent verification."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claimlayer.ledger import Ledger
from claimlayer.models import FactRecord, Policy, Source
from claimlayer.verifiers.quote_quantity import normalize_span

JSONRPC_VERSION = "2.0"
LATEST_PROTOCOL_VERSION = "2025-11-25"
SUPPORTED_PROTOCOL_VERSIONS = (
    LATEST_PROTOCOL_VERSION,
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

_ALLOWED_MCP_TIERS = frozenset({"authoritative", "standard", "draft"})

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "register_source",
        "title": "Register Verification Source",
        "description": (
            "Register a new source document in the ClaimLayer verification ledger with its "
            "authority tier ('authoritative', 'standard', 'draft'), optional validity dates, "
            "supersedes list, and optional structured facts. Existing sources cannot be overwritten."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["id", "text"],
            "properties": {
                "id": {"type": "string", "description": "Unique source identifier"},
                "text": {"type": "string", "description": "Full verbatim source text"},
                "tier": {
                    "type": "string",
                    "enum": ["authoritative", "standard", "draft"],
                    "default": "standard",
                },
                "title": {"type": "string"},
                "valid_from": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "valid_until": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "supersedes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of earlier documents superseded by this source",
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["subject", "predicate", "value"],
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "value": {"type": "string"},
                            "quote": {"type": "string"},
                            "unit": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "verify_proposal",
        "title": "Verify Agent Proposal",
        "description": (
            "Deterministically verify structured agent assertions before execution or delivery. "
            "Returns status='PASS' (auto-approved with Merkle receipt; proceed immediately), "
            "'RETRY' (fixable grounding/precedence error; self-correct without waking a human), "
            "or 'ESCALATE' (genuine conflict or threshold gate queued for 15s human triage)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["assertions"],
            "properties": {
                "agent_id": {"type": "string", "default": "mcp_agent"},
                "as_of": {"type": "string", "description": "ISO evaluation date YYYY-MM-DD"},
                "assertions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["subject", "predicate", "value", "source_id", "quote"],
                        "properties": {
                            "id": {"type": "string"},
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "value": {"type": "string"},
                            "unit": {"type": "string"},
                            "source_id": {"type": "string"},
                            "quote": {"type": "string"},
                            "confidence": {"type": "number", "default": 1.0},
                        },
                    },
                },
                "policy": {
                    "type": "object",
                    "properties": {
                        "policy_id": {"type": "string"},
                        "min_tier_rank": {"type": "integer"},
                        "min_support": {"type": "number"},
                        "required_predicates": {"type": "array", "items": {"type": "string"}},
                        "required_source_ids": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    },
    {
        "name": "check_escalation",
        "title": "Check Human Escalation Status",
        "description": "Poll whether a human reviewer has resolved an escalated verdict.",
        "inputSchema": {
            "type": "object",
            "required": ["verdict_id"],
            "properties": {
                "verdict_id": {"type": "string"},
            },
        },
    },
]


class MCPError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


@dataclass
class MCPSession:
    session_id: str | None = None
    protocol_version: str = LATEST_PROTOCOL_VERSION
    initialize_sent: bool = False
    initialized: bool = False
    client_info: dict[str, Any] = field(default_factory=dict)


class MCPVerificationServer:
    """MCP JSON-RPC 2.0 handler backed by a ClaimLayer Ledger with trust-boundary enforcement."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        default_policy: Policy | None = None,
        allow_source_registration: bool = True,
    ) -> None:
        self.ledger = ledger
        self.default_policy = default_policy
        self.allow_source_registration = allow_source_registration

    def _validate_and_merge_policy(self, raw_policy: Any) -> Policy:
        base = self.default_policy or Policy()
        if raw_policy is None:
            return base
        if not isinstance(raw_policy, dict):
            raise MCPError(JSONRPC_INVALID_PARAMS, "policy must be an object")

        # Never allow an MCP caller to disable core verification invariants (C5a)
        for invariant_flag in (
            "require_exact_quote",
            "require_quantity_match",
            "block_superseded",
            "block_expired",
        ):
            if invariant_flag in raw_policy and raw_policy[invariant_flag] is False:
                raise MCPError(
                    JSONRPC_INVALID_PARAMS,
                    f"Caller policy cannot disable core verification invariant '{invariant_flag}'.",
                )

        min_rank = int(raw_policy.get("min_tier_rank", base.min_tier_rank))
        if min_rank < base.min_tier_rank:
            raise MCPError(
                JSONRPC_INVALID_PARAMS,
                f"Caller policy cannot lower min_tier_rank below {base.min_tier_rank}.",
            )

        min_sup = float(raw_policy.get("min_support", base.min_support))
        if min_sup < base.min_support:
            raise MCPError(
                JSONRPC_INVALID_PARAMS,
                f"Caller policy cannot lower min_support below {base.min_support}.",
            )

        req_preds = list(base.required_predicates)
        for p in raw_policy.get("required_predicates") or []:
            if p not in req_preds:
                req_preds.append(str(p))

        req_srcs = list(base.required_source_ids)
        for s in raw_policy.get("required_source_ids") or []:
            if s not in req_srcs:
                req_srcs.append(str(s))

        bounds = dict(base.numeric_bounds)
        if "numeric_bounds" in raw_policy and isinstance(raw_policy["numeric_bounds"], dict):
            for k, v in raw_policy["numeric_bounds"].items():
                bounds[str(k)] = (v[0], v[1])

        return Policy(
            policy_id=str(raw_policy.get("policy_id") or base.policy_id),
            require_exact_quote=True,
            require_quantity_match=True,
            min_tier_rank=min_rank,
            block_superseded=True,
            block_expired=True,
            min_support=min_sup,
            conflict_escalation_threshold=base.conflict_escalation_threshold,
            required_predicates=req_preds,
            required_source_ids=req_srcs,
            numeric_bounds=bounds,
            as_of=raw_policy.get("as_of") or base.as_of,
        )

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "register_source":
            if not self.allow_source_registration:
                raise MCPError(
                    JSONRPC_INVALID_PARAMS,
                    "Source registration is disabled on this verification server.",
                )
            if "id" not in arguments or "text" not in arguments:
                raise MCPError(JSONRPC_INVALID_PARAMS, "'id' and 'text' are required.")

            tier = str(arguments.get("tier") or "standard").strip().lower()
            if tier not in _ALLOWED_MCP_TIERS:
                raise MCPError(
                    JSONRPC_INVALID_PARAMS,
                    f"Invalid tier '{tier}'. Must be one of {sorted(_ALLOWED_MCP_TIERS)}.",
                )

            existing_sources = self.ledger.list_sources()
            has_existing_authoritative = any(s.rank >= 100 for s in existing_sources)
            supersedes_list = [str(x) for x in (arguments.get("supersedes") or [])]

            # Prevent an agent from injecting a new authoritative/superseding source over an existing authoritative corpus (C5b)
            if has_existing_authoritative and (tier == "authoritative" or supersedes_list):
                raise MCPError(
                    JSONRPC_INVALID_PARAMS,
                    "Cannot register new authoritative or superseding sources over an already-populated authoritative ledger.",
                )

            src = Source(
                id=str(arguments["id"]),
                text=str(arguments["text"]),
                tier=tier,
                title=arguments.get("title"),
                valid_from=arguments.get("valid_from"),
                valid_until=arguments.get("valid_until"),
                supersedes=supersedes_list,
            )
            try:
                # Forbid overwriting existing sources (C5c)
                self.ledger.add_source(src, allow_overwrite=False)
            except ValueError as exc:
                raise MCPError(JSONRPC_INVALID_PARAMS, str(exc)) from exc

            raw_facts = arguments.get("facts") or []
            norm_src_text = normalize_span(src.text).lower()
            for rf in raw_facts:
                quote_str = str(rf.get("quote") or "")
                if quote_str and normalize_span(quote_str).lower() not in norm_src_text:
                    raise MCPError(
                        JSONRPC_INVALID_PARAMS,
                        f"Fact quote '{quote_str[:60]}' does not appear in source '{src.id}'.",
                    )
                self.ledger.add_fact(
                    FactRecord(
                        subject=str(rf["subject"]),
                        predicate=str(rf["predicate"]),
                        value=str(rf["value"]),
                        source_id=src.id,
                        quote=quote_str,
                        unit=rf.get("unit"),
                        confidence=float(rf.get("confidence", 1.0)),
                    )
                )
            return {
                "registered": True,
                "source_id": src.id,
                "sha256": src.sha256,
                "authority_rank": src.rank,
                "facts_registered": len(raw_facts),
            }

        if name == "verify_proposal":
            pol = self._validate_and_merge_policy(arguments.get("policy"))
            verdict = self.ledger.verify(
                assertions=arguments.get("assertions") or [],
                policy=pol,
                as_of=arguments.get("as_of"),
                agent_id=str(arguments.get("agent_id") or "mcp_agent"),
                record=True,
            )
            return verdict.to_dict()

        if name == "check_escalation":
            vid = str(arguments.get("verdict_id") or "")
            entry = self.ledger.get_verdict_entry(vid)
            if entry is None:
                return {"found": False, "verdict_id": vid}
            return {
                "found": True,
                "verdict_id": vid,
                "status": entry["status"],
                "triage_state": entry["triage_state"],
                "resolution": entry["resolution"],
                "resolved_at": entry["resolved_at"],
            }

        raise MCPError(JSONRPC_INVALID_PARAMS, f"Unknown tool: {name}")

    def handle_message(
        self, message: Any, session: MCPSession, *, stateless_http: bool = False
    ) -> dict[str, Any] | None:
        if not isinstance(message, dict) or message.get("jsonrpc") != JSONRPC_VERSION:
            raise MCPError(JSONRPC_INVALID_REQUEST, "Invalid JSON-RPC 2.0 message")

        if "method" not in message and ("result" in message or "error" in message):
            return None

        method = message.get("method")
        if not isinstance(method, str) or not method:
            raise MCPError(JSONRPC_INVALID_REQUEST, "Method is required")

        # Notification
        if "id" not in message:
            if method == "notifications/initialized":
                session.initialized = True
            return None

        req_id = message["id"]
        params = message.get("params") or {}

        if method == "initialize":
            req_ver = str(params.get("protocolVersion") or LATEST_PROTOCOL_VERSION)
            negotiated = (
                req_ver if req_ver in SUPPORTED_PROTOCOL_VERSIONS else LATEST_PROTOCOL_VERSION
            )
            session.protocol_version = negotiated
            session.initialize_sent = True
            session.initialized = stateless_http
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": req_id,
                "result": {
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "claimlayer-mcp",
                        "version": "3.1.0",
                        "title": "ClaimLayer Computable Verification Server",
                    },
                    "instructions": (
                        "Call verify_proposal before completing tasks or executing actions. "
                        "On PASS, proceed immediately without human wait. On RETRY, fix the "
                        "reported issues and re-verify. On ESCALATE, pause only on the contested item."
                    ),
                },
            }

        if method == "ping":
            return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": {}}

        if not stateless_http and not session.initialize_sent:
            raise MCPError(JSONRPC_INVALID_REQUEST, "Server session is not initialized")

        if method == "tools/list":
            return {"jsonrpc": JSONRPC_VERSION, "id": req_id, "result": {"tools": MCP_TOOLS}}

        if method == "tools/call":
            if not isinstance(params, dict):
                raise MCPError(JSONRPC_INVALID_PARAMS, "tools/call params must be an object")
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                raise MCPError(JSONRPC_INVALID_PARAMS, "Tool name is required")
            try:
                output = self.execute_tool(tool_name, tool_args)
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(output, sort_keys=True)}
                        ],
                        "structuredContent": output,
                        "isError": False,
                    },
                }
            except MCPError:
                raise
            except Exception as exc:
                return {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution error: {exc}"}],
                        "isError": True,
                    },
                }

        raise MCPError(JSONRPC_METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_envelope(
        self, payload: Any, session: MCPSession, *, stateless_http: bool = False
    ) -> list[dict[str, Any]]:
        requests = payload if isinstance(payload, list) else [payload]
        if not requests:
            raise MCPError(JSONRPC_INVALID_REQUEST, "Empty JSON-RPC batch")

        responses: list[dict[str, Any]] = []
        for req in requests:
            try:
                resp = self.handle_message(req, session, stateless_http=stateless_http)
            except MCPError as exc:
                req_id = req.get("id") if isinstance(req, dict) else None
                resp = {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": req_id,
                    "error": {"code": exc.code, "message": exc.message},
                }
            except Exception as exc:
                req_id = req.get("id") if isinstance(req, dict) else None
                resp = {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": req_id,
                    "error": {"code": JSONRPC_INTERNAL_ERROR, "message": str(exc)},
                }
            if resp is not None:
                responses.append(resp)
        return responses


def run_stdio_server(db_path: str | Path | None = None) -> int:
    """Run the ClaimLayer MCP server over standard input/output."""
    ledger = Ledger(db_path=db_path)
    server = MCPVerificationServer(ledger)
    session = MCPSession()

    for raw_line in sys.stdin.buffer:
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line.decode("utf-8"))
            responses = server.handle_envelope(payload, session)
            if not responses:
                continue
            out = responses if isinstance(payload, list) else responses[0]
            encoded = json.dumps(out, separators=(",", ":")).encode("utf-8")
        except Exception as exc:
            encoded = json.dumps(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": None,
                    "error": {"code": JSONRPC_PARSE_ERROR, "message": str(exc)},
                },
                separators=(",", ":"),
            ).encode("utf-8")
        sys.stdout.buffer.write(encoded + b"\n")
        sys.stdout.buffer.flush()
    return 0
