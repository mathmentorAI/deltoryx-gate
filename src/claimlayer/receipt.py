"""RFC 9162 domain-separated SHA-256 Merkle receipts (with optional HMAC authentication)."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import TYPE_CHECKING, Any

from claimlayer.models import (
    Assertion,
    ConflictDiff,
    Issue,
    Policy,
    Source,
    Verdict,
    VerificationReceipt,
)
from claimlayer.verifiers.quote_quantity import extract_dates, extract_quantities

if TYPE_CHECKING:
    from claimlayer.ledger import Ledger

RECEIPT_SCHEME = "CLAIMLAYER-MERKLE-SHA256-v3"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sign_payload(payload: Any, signing_key: str | bytes | None = None) -> tuple[str, bool]:
    key = signing_key if signing_key is not None else os.environ.get("CLAIMLAYER_SIGNING_KEY")
    raw = _canonical_json_bytes(payload)
    if key:
        key_bytes = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        return hmac.new(key_bytes, raw, hashlib.sha256).hexdigest(), True
    return hashlib.sha256(raw).hexdigest(), False


def compute_assertion_leaf_hash(assertion: Assertion, *, verified: bool = True) -> str:
    """Compute RFC 9162 domain-separated leaf hash (0x00 prefix) committing to assertion verification state."""
    leaf_payload: dict[str, Any] = {
        "id": assertion.id,
        "subject": assertion.subject,
        "predicate": assertion.predicate,
        "value": assertion.value,
        "unit": assertion.unit,
        "source_id": assertion.source_id,
        "quote": assertion.quote,
        "confidence": round(float(assertion.confidence), 6),
        "verified": bool(verified),
    }
    if assertion.condition is not None:
        leaf_payload["condition"] = assertion.condition
    return hashlib.sha256(b"\x00" + _canonical_json_bytes(leaf_payload)).hexdigest()


def compute_merkle_root(leaf_hashes: list[str] | tuple[str, ...]) -> str:
    """Compute RFC 9162 domain-separated Merkle tree root (0x01 internal node prefix, no odd duplication)."""
    if not leaf_hashes:
        return hashlib.sha256(b"\x00CLAIMLAYER_EMPTY_PROPOSAL").hexdigest()
    level = list(leaf_hashes)
    while len(level) > 1:
        next_level: list[str] = []
        for idx in range(0, len(level), 2):
            if idx + 1 < len(level):
                left = level[idx].encode("ascii")
                right = level[idx + 1].encode("ascii")
                combined = hashlib.sha256(b"\x01" + left + b":" + right).hexdigest()
                next_level.append(combined)
            else:
                # RFC 9162: promote odd node without duplicating (prevents CVE-2012-2459 [a,b,c] == [a,b,c,c])
                next_level.append(level[idx])
        level = next_level
    return level[0]


def compute_sources_digest(sources: list[Source]) -> str:
    """Commit to full source state including tier name, rank, validity dates, supersedes, and metadata (E5)."""
    items = [
        {
            "id": s.id,
            "sha256": s.sha256,
            "tier": s.tier,
            "rank": s.rank,
            "valid_from": s.valid_from,
            "valid_until": s.valid_until,
            "supersedes": list(s.supersedes),
            "metadata": dict(s.metadata),
        }
        for s in sorted(sources, key=lambda item: item.id)
    ]
    return _sha256_json(items)


def compute_policy_digest(policy: Policy) -> str:
    return _sha256_json(policy.to_dict())


def compute_ledger_digest(ledger: Ledger | None) -> str:
    """Commit to full ledger state: sources, registered facts, and human resolutions (E5)."""
    if ledger is None:
        return ""
    facts_payload = [
        f.to_dict()
        for f in sorted(
            ledger.list_facts(),
            key=lambda x: (x.source_id, x.subject, x.predicate, x.value, x.quote),
        )
    ]
    resolutions_payload = [
        r.to_dict()
        for r in sorted(
            ledger.list_resolutions(),
            key=lambda x: (x.subject, x.predicate, x.selected_value),
        )
    ]
    return _sha256_json(
        {
            "sources_digest": compute_sources_digest(ledger.list_sources()),
            "facts": facts_payload,
            "resolutions": resolutions_payload,
        }
    )


def _build_signature_envelope(
    *,
    scheme: str,
    merkle_root: str,
    leaf_count: int,
    sources_digest: str,
    policy_digest: str,
    ledger_digest: str,
    status: str,
    allowed_to_proceed: bool,
    requires_human: bool,
    agent_id: str,
    policy_id: str,
    confidence: float,
    as_of: str | None,
    issues: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
    assertion_scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "scheme": scheme,
        "merkle_root": merkle_root,
        "leaf_count": leaf_count,
        "sources_digest": sources_digest,
        "policy_digest": policy_digest,
        "ledger_digest": ledger_digest,
        "status": status,
        "allowed_to_proceed": bool(allowed_to_proceed),
        "requires_human": bool(requires_human),
        "agent_id": str(agent_id),
        "policy_id": str(policy_id),
        "confidence": round(float(confidence), 6),
        "as_of": as_of,
        "issues": issues,
        "conflicts": conflicts,
        "assertion_scores": {
            str(k): round(float(v), 6) for k, v in sorted(assertion_scores.items())
        },
    }


def build_receipt(
    *,
    assertions: tuple[Assertion, ...],
    sources: list[Source],
    policy: Policy,
    status: str,
    confidence: float,
    issue_codes: list[str] | None = None,
    as_of: str | None = None,
    allowed_to_proceed: bool | None = None,
    requires_human: bool | None = None,
    agent_id: str = "agent",
    issues: tuple[Issue, ...] | list[Issue] = (),
    conflicts: tuple[ConflictDiff, ...] | list[ConflictDiff] = (),
    assertion_scores: dict[str, float] | None = None,
    ledger: Ledger | None = None,
    signing_key: str | bytes | None = None,
) -> VerificationReceipt:
    issues_dicts = [i.to_dict() if isinstance(i, Issue) else dict(i) for i in issues]
    failed_ids = {
        str(d.get("assertion_id"))
        for d in issues_dicts
        if d.get("disposition") == "RETRY" and d.get("assertion_id")
    }
    leaf_hashes = tuple(
        compute_assertion_leaf_hash(a, verified=(a.id not in failed_ids and status != "RETRY"))
        for a in assertions
    )
    merkle_root = compute_merkle_root(leaf_hashes)
    sources_digest = compute_sources_digest(sources)
    policy_digest = compute_policy_digest(policy)
    ledger_digest = compute_ledger_digest(ledger)

    eff_allowed = (status == "PASS") if allowed_to_proceed is None else bool(allowed_to_proceed)
    eff_human = (status == "ESCALATE") if requires_human is None else bool(requires_human)
    conflicts_dicts = [
        c.to_dict() if isinstance(c, ConflictDiff) else dict(c) for c in conflicts
    ]
    scores_map = dict(assertion_scores or {})

    envelope = _build_signature_envelope(
        scheme=RECEIPT_SCHEME,
        merkle_root=merkle_root,
        leaf_count=len(leaf_hashes),
        sources_digest=sources_digest,
        policy_digest=policy_digest,
        ledger_digest=ledger_digest,
        status=status,
        allowed_to_proceed=eff_allowed,
        requires_human=eff_human,
        agent_id=agent_id,
        policy_id=policy.policy_id,
        confidence=confidence,
        as_of=as_of,
        issues=issues_dicts,
        conflicts=conflicts_dicts,
        assertion_scores=scores_map,
    )
    eff_key = signing_key or (getattr(ledger, "signing_key", None) if ledger is not None else None)
    signature, is_hmac = _sign_payload(envelope, signing_key=eff_key)

    return VerificationReceipt(
        scheme=RECEIPT_SCHEME,
        merkle_root=merkle_root,
        policy_digest=policy_digest,
        sources_digest=sources_digest,
        ledger_digest=ledger_digest,
        signature=signature,
        leaf_hashes=leaf_hashes,
        leaf_count=len(leaf_hashes),
        hmac_authenticated=is_hmac,
    )


def _assertions_self_consistent_for_pass(assertions: tuple[Assertion, ...]) -> bool:
    """Verify intrinsic quote/quantity consistency for PASS verdicts (defeats E1 status-rewrite forgery)."""
    if not assertions:
        return False
    for a in assertions:
        if not a.quote or not a.quote.strip():
            return False
        for d in extract_dates(a.value):
            if d not in extract_dates(a.quote):
                return False
        val_q = extract_quantities(a.value, explicit_unit=a.unit)
        if val_q:
            quote_q = extract_quantities(a.quote, apply_table_scale=True)
            for vq in val_q:
                if not any(vq.matches(qq) and not vq.same_magnitude_wrong_unit(qq) for qq in quote_q):
                    return False
    return True


def verify_receipt(
    verdict: Verdict | dict[str, Any],
    *,
    ledger: Ledger | None = None,
    policy: Policy | dict[str, Any] | None = None,
    signing_key: str | bytes | None = None,
) -> bool:
    """Verify offline (or against a live Ledger/Policy/HMAC key) that a Verdict has not been forged or tampered with."""
    data = verdict.to_dict() if isinstance(verdict, Verdict) else dict(verdict)
    receipt = data.get("receipt") or {}
    if receipt.get("scheme") != RECEIPT_SCHEME:
        return False

    status = data.get("status", "")
    if status not in {"PASS", "RETRY", "ESCALATE"}:
        return False

    # 1. Verify status/flag invariant consistency (E1, E2)
    allowed = data.get("allowed_to_proceed")
    req_human = data.get("requires_human")
    if not isinstance(allowed, bool) or allowed != (status == "PASS"):
        return False
    if not isinstance(req_human, bool) or req_human != (status == "ESCALATE"):
        return False

    raw_issues = data.get("issues") or []
    raw_conflicts = data.get("conflicts") or []
    raw_scores = data.get("assertion_scores") or {}
    conf_val = round(float(data.get("confidence", 0.0)), 6)

    if status == "PASS":
        if raw_issues or raw_conflicts or conf_val <= 0.0:
            return False
        if not raw_scores or any(float(v) <= 0.0 for v in raw_scores.values()):
            return False
    elif status == "RETRY":
        if not raw_issues or conf_val != 0.0:
            return False
    elif status == "ESCALATE":
        if not raw_issues:
            return False

    # 2. Recompute leaf hashes and RFC 9162 Merkle root (E4)
    raw_assertions = data.get("assertions") or []
    try:
        typed_assertions = tuple(
            a if isinstance(a, Assertion) else Assertion(**a) for a in raw_assertions
        )
    except Exception:
        return False

    if status == "PASS" and not _assertions_self_consistent_for_pass(typed_assertions):
        return False

    failed_ids = {
        str(d.get("assertion_id"))
        for d in raw_issues
        if isinstance(d, dict) and d.get("disposition") == "RETRY" and d.get("assertion_id")
    }
    recomputed_leaves = tuple(
        compute_assertion_leaf_hash(a, verified=(a.id not in failed_ids and status != "RETRY"))
        for a in typed_assertions
    )
    if list(recomputed_leaves) != list(receipt.get("leaf_hashes") or []):
        return False
    if receipt.get("leaf_count", len(recomputed_leaves)) != len(recomputed_leaves):
        return False

    recomputed_root = compute_merkle_root(recomputed_leaves)
    if recomputed_root != receipt.get("merkle_root"):
        return False

    # 3. Verify external bindings against live Ledger and Policy if provided (E3, E5)
    sources_digest = str(receipt.get("sources_digest") or "")
    policy_digest = str(receipt.get("policy_digest") or "")
    ledger_digest = str(receipt.get("ledger_digest") or "")

    if ledger is not None:
        if sources_digest != compute_sources_digest(ledger.list_sources()):
            return False
        if ledger_digest and ledger_digest != compute_ledger_digest(ledger):
            return False

    if policy is not None:
        from claimlayer.verify import _coerce_policy

        pol_obj = _coerce_policy(policy)
        if policy_digest != compute_policy_digest(pol_obj):
            return False

    # 4. Recompute full envelope signature / HMAC (E1, E2)
    envelope = _build_signature_envelope(
        scheme=RECEIPT_SCHEME,
        merkle_root=recomputed_root,
        leaf_count=len(recomputed_leaves),
        sources_digest=sources_digest,
        policy_digest=policy_digest,
        ledger_digest=ledger_digest,
        status=status,
        allowed_to_proceed=allowed,
        requires_human=req_human,
        agent_id=str(data.get("agent_id") or "agent"),
        policy_id=str(data.get("policy_id") or "default"),
        confidence=conf_val,
        as_of=data.get("as_of"),
        issues=[dict(i) for i in raw_issues],
        conflicts=[dict(c) for c in raw_conflicts],
        assertion_scores=raw_scores,
    )

    eff_key = signing_key or (getattr(ledger, "signing_key", None) if ledger is not None else None)
    expected_sig, is_hmac = _sign_payload(envelope, signing_key=eff_key)
    if receipt.get("hmac_authenticated") and not is_hmac:
        return False

    actual_sig = str(receipt.get("signature") or "")
    if not hmac.compare_digest(expected_sig, actual_sig):
        return False

    verdict_id = str(data.get("verdict_id") or "")
    if verdict_id and verdict_id != f"vrd-{actual_sig[:12]}":
        return False

    return True
