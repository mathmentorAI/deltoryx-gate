"""Core data structures for ClaimLayer v3 computable verification."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

Disposition = Literal["RETRY", "ESCALATE"]
VerdictStatus = Literal["PASS", "RETRY", "ESCALATE"]
EvidenceProfile = Literal["legal_prose", "code_config"]
EVIDENCE_PROFILES: frozenset[str] = frozenset({"legal_prose", "code_config"})

TIER_RANKS: dict[str, int] = {
    "authoritative": 100,
    "primary": 100,
    "governing": 100,
    "amendment": 100,
    "standard": 50,
    "secondary": 50,
    "annex": 50,
    "markup": 50,
    "informational": 20,
    "external": 20,
    "draft": 10,
    "untrusted": 10,
}

_ENTITY_STOPWORDS = frozenset({"the", "a", "an", "el", "la", "los", "las"})
_ENTITY_CORP_SUFFIXES = frozenset(
    {
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "llc",
        "ltd",
        "limited",
        "plc",
        "co",
        "company",
        "sa",
        "sl",
        "gmbh",
        "ag",
    }
)
def resolve_tier_rank(tier: str, explicit_rank: int | None = None) -> int:
    """Resolve an authority rank integer from a named tier or explicit rank."""
    if explicit_rank is not None:
        return int(explicit_rank)
    key = (tier or "standard").strip().lower()
    if key not in TIER_RANKS:
        raise ValueError(
            f"Unknown authority tier '{tier}'. Valid tiers: {sorted(TIER_RANKS.keys())}"
        )
    return TIER_RANKS[key]


def normalize_key(raw: str) -> str:
    """Canonicalize subject or predicate identifiers deterministically (Unicode-preserving)."""
    norm = unicodedata.normalize("NFKC", (raw or "").strip().lower())
    cleaned = re.sub(r"[^\w]+", "_", norm, flags=re.UNICODE)
    return cleaned.strip("_")


def _singularize_token(tok: str) -> str:
    if len(tok) <= 3 or not tok.isalpha():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("ses") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is", "os", "as")):
        return tok[:-1]
    return tok


def normalize_entity_key(raw: str) -> str:
    """Canonicalize entity/subject names (stripping articles and corporate designators)."""
    base = normalize_key(raw)
    if not base:
        return ""
    tokens = [t for t in base.split("_") if t]
    while len(tokens) > 1 and tokens[0] in _ENTITY_STOPWORDS:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1] in _ENTITY_CORP_SUFFIXES:
        tokens.pop()
    singularized = [_singularize_token(t) for t in tokens]
    return "_".join(singularized) or base


def normalize_predicate_key(raw: str, aliases: Mapping[str, str] | None = None) -> str:
    """Canonicalize predicate identifiers (handling singular/plural variants and explicit aliases)."""
    base = normalize_key(raw)
    if not base:
        return ""
    if aliases:
        mapped = aliases.get(base) or aliases.get(raw)
        if mapped:
            base = normalize_key(mapped)
    tokens = [_singularize_token(t) for t in base.split("_") if t]
    return "_".join(tokens) or base


def _validate_confidence(val: Any) -> float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"confidence must be a finite float in [0.0, 1.0], got {val!r}")
    fval = float(val)
    if math.isnan(fval) or math.isinf(fval) or fval < 0.0 or fval > 1.0:
        raise ValueError(f"confidence must be a finite float in [0.0, 1.0], got {val!r}")
    return fval


@dataclass(frozen=True)
class Source:
    """An evidence document or state record registered in the verification ledger."""

    id: str
    text: str
    tier: str = "standard"
    authority_rank: int | None = None
    title: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: tuple[str, ...] | list[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    evidence_profile: EvidenceProfile = "legal_prose"
    _sha256: str = field(init=False, repr=False, compare=False)
    _content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate tier immediately
        _ = resolve_tier_rank(self.tier, self.authority_rank)
        if self.evidence_profile not in EVIDENCE_PROFILES:
            raise ValueError(
                f"Unknown evidence_profile '{self.evidence_profile}'. Valid profiles: {sorted(EVIDENCE_PROFILES)}"
            )
        if isinstance(self.supersedes, list):
            object.__setattr__(self, "supersedes", tuple(self.supersedes))
        elif not isinstance(self.supersedes, tuple):
            object.__setattr__(self, "supersedes", tuple(self.supersedes))
        frozen_meta = MappingProxyType(dict(self.metadata or {}))
        object.__setattr__(self, "metadata", frozen_meta)
        raw_text = self.text or ""
        digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "_sha256", digest)
        if self.evidence_profile == "code_config":
            body_lines = unicodedata.normalize("NFKC", raw_text).splitlines()
        else:
            body_lines = [
                ln
                for ln in unicodedata.normalize("NFKC", raw_text).splitlines()
                if not re.match(
                    r"^\s*(?:draft|borrador|confidential|confidencial|page\s+\d+|p[áa]gina\s+\d+)"
                    r"(?:\s*[-—:|]\s*(?:page\s+\d+|p[áa]gina\s+\d+|\w+))?\s*$",
                    ln,
                    flags=re.IGNORECASE,
                )
            ]
        norm_text = re.sub(r"\s+", " ", " ".join(body_lines).strip()).lower()
        content_digest = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "_content_sha256", content_digest)

    @property
    def rank(self) -> int:
        return resolve_tier_rank(self.tier, self.authority_rank)

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tier": self.tier,
            "authority_rank": self.rank,
            "title": self.title,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "supersedes": list(self.supersedes),
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
            "evidence_profile": self.evidence_profile,
        }


@dataclass(frozen=True)
class FactRecord:
    """A known structured fact tied to a source in the ledger."""

    subject: str
    predicate: str
    value: str
    source_id: str
    quote: str = ""
    unit: str | None = None
    confidence: float = 1.0
    condition: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("condition") is None:
            data.pop("condition", None)
        return data


@dataclass(frozen=True)
class Assertion:
    """A structured claim or proposed decision parameter emitted by an AI agent."""

    subject: str
    predicate: str
    value: str
    source_id: str
    quote: str
    id: str = ""
    unit: str | None = None
    confidence: float = 1.0
    condition: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))

    def with_default_id(self, index: int) -> Assertion:
        if self.id:
            return self
        return Assertion(
            id=f"a{index}",
            subject=self.subject,
            predicate=self.predicate,
            value=self.value,
            source_id=self.source_id,
            quote=self.quote,
            unit=self.unit,
            confidence=self.confidence,
            condition=self.condition,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data.get("condition") is None:
            data.pop("condition", None)
        return data


@dataclass(frozen=True)
class Policy:
    """Deterministic verification gates and invariants."""

    policy_id: str = "default"
    require_exact_quote: bool = True
    require_quantity_match: bool = True
    min_tier_rank: int = 20
    block_superseded: bool = True
    block_expired: bool = True
    min_support: float = 0.70
    confidence_threshold: float | None = None
    conflict_escalation_threshold: float = 0.20
    required_predicates: tuple[str, ...] | list[str] = field(default_factory=tuple)
    required_source_ids: tuple[str, ...] | list[str] = field(default_factory=tuple)
    numeric_bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    predicate_aliases: dict[str, str] = field(default_factory=dict)
    as_of: str | None = None
    required_value_tokens: dict[str, list[str]] = field(default_factory=dict)
    vocabulary: Any | None = None

    def __post_init__(self) -> None:
        if self.confidence_threshold is not None:
            object.__setattr__(
                self, "min_support", _validate_confidence(self.confidence_threshold)
            )
        else:
            object.__setattr__(self, "min_support", _validate_confidence(self.min_support))
        object.__setattr__(
            self,
            "conflict_escalation_threshold",
            _validate_confidence(self.conflict_escalation_threshold),
        )
        if not isinstance(self.numeric_bounds, Mapping):
            raise ValueError("numeric_bounds must be a mapping")
        validated_bounds: dict[str, tuple[float | None, float | None]] = {}
        for k, bounds in self.numeric_bounds.items():
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError(f"numeric_bounds[{k!r}] must be a 2-element (min, max) pair")
            raw_lo, raw_hi = bounds[0], bounds[1]
            lo: float | None = None
            hi: float | None = None
            if raw_lo is not None:
                if isinstance(raw_lo, bool) or not isinstance(raw_lo, (int, float)):
                    raise ValueError(f"numeric_bounds[{k!r}] lower bound must be finite number")
                lo = float(raw_lo)
                if not math.isfinite(lo):
                    raise ValueError(f"numeric_bounds[{k!r}] lower bound must be finite number")
            if raw_hi is not None:
                if isinstance(raw_hi, bool) or not isinstance(raw_hi, (int, float)):
                    raise ValueError(f"numeric_bounds[{k!r}] upper bound must be finite number")
                hi = float(raw_hi)
                if not math.isfinite(hi):
                    raise ValueError(f"numeric_bounds[{k!r}] upper bound must be finite number")
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"numeric_bounds[{k!r}] lower bound cannot exceed upper bound")
            validated_bounds[str(k)] = (lo, hi)
        object.__setattr__(self, "numeric_bounds", validated_bounds)
        if isinstance(self.required_predicates, list):
            object.__setattr__(self, "required_predicates", tuple(self.required_predicates))
        if isinstance(self.required_source_ids, list):
            object.__setattr__(self, "required_source_ids", tuple(self.required_source_ids))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy_id": self.policy_id,
            "require_exact_quote": self.require_exact_quote,
            "require_quantity_match": self.require_quantity_match,
            "min_tier_rank": self.min_tier_rank,
            "block_superseded": self.block_superseded,
            "block_expired": self.block_expired,
            "min_support": self.min_support,
            "conflict_escalation_threshold": self.conflict_escalation_threshold,
            "required_predicates": list(self.required_predicates),
            "required_source_ids": list(self.required_source_ids),
            "numeric_bounds": {
                k: [v[0], v[1]] for k, v in sorted(self.numeric_bounds.items())
            },
        }
        if self.required_value_tokens:
            payload["required_value_tokens"] = dict(sorted(self.required_value_tokens.items()))
        if self.predicate_aliases:
            payload["predicate_aliases"] = dict(sorted(self.predicate_aliases.items()))
        if self.as_of is not None:
            payload["as_of"] = self.as_of
        if self.vocabulary is not None:
            payload["vocabulary"] = (
                self.vocabulary.describe()
                if hasattr(self.vocabulary, "describe")
                else self.vocabulary
            )
        return payload


@dataclass(frozen=True)
class ConflictDiff:
    """Minimal side-by-side discrepancy payload for 15-second human triage."""

    assertion_id: str
    subject: str
    predicate: str
    proposed_value: str
    proposed_source_id: str
    proposed_quote: str
    proposed_tier: str
    competing_value: str
    competing_source_id: str
    competing_quote: str
    competing_tier: str
    proposed_support: float
    competing_support: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Issue:
    """A deterministic verification finding routed either to the agent (RETRY) or human (ESCALATE)."""

    code: str
    disposition: Disposition
    message: str
    remediation: str
    assertion_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    conflict_diff: ConflictDiff | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.conflict_diff is not None:
            payload["conflict_diff"] = self.conflict_diff.to_dict()
        return payload


@dataclass(frozen=True)
class VerificationReceipt:
    """RFC 9162 domain-separated SHA-256 Merkle receipt (with optional HMAC authentication)."""

    scheme: str
    merkle_root: str
    policy_digest: str
    sources_digest: str
    signature: str
    leaf_hashes: tuple[str, ...]
    ledger_digest: str = ""
    leaf_count: int = 0
    hmac_authenticated: bool = False
    corpus_obligations_total: int | None = None
    corpus_obligations_covered: int | None = None
    corpus_obligations_rejected: int | None = None
    corpus_obligation_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scheme": self.scheme,
            "merkle_root": self.merkle_root,
            "policy_digest": self.policy_digest,
            "sources_digest": self.sources_digest,
            "ledger_digest": self.ledger_digest,
            "signature": self.signature,
            "leaf_hashes": list(self.leaf_hashes),
            "leaf_count": self.leaf_count or len(self.leaf_hashes),
            "hmac_authenticated": self.hmac_authenticated,
        }
        if self.corpus_obligations_total is not None:
            out["corpus_obligations_total"] = int(self.corpus_obligations_total)
            out["corpus_obligations_covered"] = int(self.corpus_obligations_covered or 0)
            out["corpus_obligations_rejected"] = int(self.corpus_obligations_rejected or 0)
            out["corpus_obligation_ids"] = list(self.corpus_obligation_ids)
        return out


@dataclass(frozen=True)
class ResolutionRecord:
    """A human reviewer decision resolving an escalated conflict or exception."""

    subject: str
    predicate: str
    selected_value: str
    resolver: str
    note: str = ""
    verdict_id: str | None = None
    resolved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Verdict:
    """Outcome of ClaimLayer computable verification."""

    verdict_id: str
    status: VerdictStatus
    allowed_to_proceed: bool
    requires_human: bool
    confidence: float
    issues: tuple[Issue, ...]
    conflicts: tuple[ConflictDiff, ...]
    assertion_scores: dict[str, float]
    receipt: VerificationReceipt
    assertions: tuple[Assertion, ...] = field(default_factory=tuple)
    agent_id: str = "agent"
    policy_id: str = "default"
    as_of: str | None = None

    def verify_receipt(
        self,
        *,
        ledger: Any | None = None,
        policy: Policy | dict[str, Any] | None = None,
        signing_key: str | bytes | None = None,
        allow_unkeyed: bool = False,
    ) -> bool:
        from claimlayer.receipt import verify_receipt

        return verify_receipt(
            self,
            ledger=ledger,
            policy=policy,
            signing_key=signing_key,
            allow_unkeyed=allow_unkeyed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict_id": self.verdict_id,
            "status": self.status,
            "allowed_to_proceed": self.allowed_to_proceed,
            "requires_human": self.requires_human,
            "confidence": round(self.confidence, 6),
            "agent_id": self.agent_id,
            "policy_id": self.policy_id,
            "as_of": self.as_of,
            "issues": [issue.to_dict() for issue in self.issues],
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "assertion_scores": {
                k: round(v, 6) for k, v in sorted(self.assertion_scores.items())
            },
            "assertions": [a.to_dict() for a in self.assertions],
            "receipt": self.receipt.to_dict(),
        }
