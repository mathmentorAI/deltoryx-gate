#!/usr/bin/env python3
"""ClaimLayer v3 + v4 Harness2 Securitization & Financial Comparison Pipeline.

Integrates:
  - v3 Ledger, authority tiers/ranks, source-backed quotes, and SHA-256 provenance
  - v4 JustificationGraph, mandatory obligations, WaterfallStepSpec relational verification,
    quantitative rate/spread/principal assumptions, % / pp / bps and day-count unit checks,
    response-vs-graph correspondence, and consolidated non-evasion repair feedback.

Strictly enforces separation between:
  1. Runtime document inputs (via document_manifest.json + common_evidence.txt) and
     physically isolated evaluator_only/ ground-truth files.
  2. Analytical report quality verdict (`report_disposition`: ACCEPT / RETRY / ESCALATE / ABSTAIN)
     vs. operational execution authorization (`operational_authorization`: PASS / RETRY / ESCALATE).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_HARNESS_DIR = Path(__file__).resolve().parent
_CLAIMLAYER_ROOT = _HARNESS_DIR.parents[1]
_SRC_DIR = _CLAIMLAYER_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from claimlayer import (
    Assertion,
    EvidenceLayer,
    EvidenceUnit,
    FactRecord,
    Ledger,
    Policy,
    Source,
    UnitRegistry,
    build_comparison_policy,
    verify,
    verify_receipt,
)
from claimlayer.v4 import (
    ClaimNode,
    CorpusReferenceRegister,
    DomainProfile,
    InferenceStep,
    JustificationGraph,
    RecordedProposalReader,
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_VOCABULARY,
    VerificationBudget,
    VerificationCache,
    build_corpus_reference_register,
    collect_reference_divergences,
    review_response,
)
from claimlayer.v4.corpus_obligations import _value_in_text
from claimlayer.v4.correspondence import compute_response_digest
from claimlayer.v4.domain_profile import PolicyConfiguration
from claimlayer.v4.domains.banking_finance import (
    BANKING_EXCLUDED_COMPARISON_ROLES,
    BANKING_FINANCE_ACCEPTANCE_POLICY,
    BANKING_FINANCE_VOCABULARY,
    BANKING_SUBORDINATE_ROLES,
    resolve_banking_governing_role,
)
from claimlayer.v4.domains.securitization import (
    SECURITIZATION_EXCLUDED_COMPARISON_ROLES,
    WaterfallComparisonResult,
    WaterfallStepSpec,
    verify_waterfall_steps,
)
from claimlayer.semantic.models import SourceDocument

DEFAULT_MANIFEST_PATH = _HARNESS_DIR / "document_manifest.json"
DEFAULT_LOCAL_LLM_URL = os.environ.get("CLAIMLAYER_LOCAL_LLM_URL", "http://127.0.0.1:1234/v1")
DEFAULT_LOCAL_LLM_MODEL = os.environ.get("CLAIMLAYER_LOCAL_LLM_MODEL", "qwen/qwen3.5-9b")


def assert_no_evaluator_only_access(path: Path | str) -> None:
    """Guardrail guaranteeing agent/pipeline code never touches evaluator_only/ files."""
    normalized = str(path).replace("\\", "/")
    if "evaluator_only" in normalized.split("/") or "evaluator_only" in normalized:
        raise PermissionError(
            f"Access denied: runtime pipeline may not read evaluator-only path '{path}'."
        )


@dataclass(frozen=True)
class DocumentSection:
    heading: str
    start_line: int
    end_line: int
    text: str

    @property
    def char_len(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class LoadedDocument:
    doc_key: str
    filename: str
    role: str
    authority_tier: str
    authority_rank: int
    governs_on_conflict: bool
    text: str
    start_line: int
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True)
class LoadedCaseCorpus:
    case_id: str
    task_id: str
    dataset_role: str
    common_evidence_sha256: str
    common_characters: int
    total_lines: int
    documents: dict[str, LoadedDocument]
    v3_sources: tuple[Source, ...]
    ledger: Ledger
    v4_sources: dict[str, SourceDocument]
    evidence_layer: EvidenceLayer
    task_instructions: str = ""


def extract_document_sections(text: str, *, base_line: int = 1) -> tuple[DocumentSection, ...]:
    """Segment a full parsed document into heading-delimited sections covering 100% of text."""
    lines = text.splitlines()
    if not lines:
        return ()

    heading_re = re.compile(
        r"^(?:"
        r"\*\*\[[^\]]+\](?:\{\.underline\})?\*\*|"
        r"\*\*Section\s+\d+(?:\.\d+)*[^*]*\*\*|"
        r"(?:ARTICLE|Section)\s+[IVXLCM0-9]+[^\n]*|"
        r"===+\s*Sheet:[^\n]+|"
        r"---+\s*SECTION\s+\d+[^\n]+"
        r")"
    )
    sections: list[DocumentSection] = []
    cur_heading = "Header / Preamble"
    cur_start = base_line
    cur_lines: list[str] = []

    for offset, line in enumerate(lines):
        stripped = line.strip()
        if heading_re.match(stripped) and cur_lines:
            sections.append(
                DocumentSection(
                    heading=cur_heading,
                    start_line=cur_start,
                    end_line=base_line + offset - 1,
                    text="\n".join(cur_lines).strip(),
                )
            )
            cur_heading = stripped.strip("*[]").replace("{.underline}", "").strip() or f"Section@{base_line + offset}"
            cur_start = base_line + offset
            cur_lines = [line]
        else:
            if heading_re.match(stripped) and not cur_lines:
                cur_heading = stripped.strip("*[]").replace("{.underline}", "").strip()
                cur_start = base_line + offset
            cur_lines.append(line)

    if cur_lines:
        sections.append(
            DocumentSection(
                heading=cur_heading,
                start_line=cur_start,
                end_line=base_line + len(lines) - 1,
                text="\n".join(cur_lines).strip(),
            )
        )
    return tuple(sections)


def _map_origin_tier(authority_tier: str, authority_rank: int) -> str:
    if authority_tier == "authoritative" or authority_rank >= 90:
        return "primary_document"
    if authority_tier == "standard" or authority_rank >= 60:
        return "secondary_annex"
    return "external_injected"


def _build_corpus_objects(
    case_id: str,
    task_id: str,
    dataset_role: str,
    docs_cfg: list[tuple[str, str, str, str, int, bool, int, str]],
    *,
    common_sha256: str,
    common_chars: int,
    total_lines: int,
    task_instructions: str = "",
    reader: Any = None,
) -> LoadedCaseCorpus:
    """Assemble v3 Ledger + v4 SourceDocument mapping from normalized document entries."""
    loaded_docs: dict[str, LoadedDocument] = {}
    v3_sources: list[Source] = []
    v4_sources: dict[str, SourceDocument] = {}
    for dk, fname, role, tier, rank, gov, start_line, body_text in docs_cfg:
        sections = extract_document_sections(body_text, base_line=start_line)
        loaded_docs[fname] = LoadedDocument(
            doc_key=dk,
            filename=fname,
            role=role,
            authority_tier=tier,
            authority_rank=rank,
            governs_on_conflict=gov,
            text=body_text,
            start_line=start_line,
            sections=sections,
        )
        v3_sources.append(
            Source(
                id=fname,
                text=body_text,
                tier=tier,
                authority_rank=rank,
                title=f"{dk}: {fname}",
                metadata={"doc_key": dk, "role": role},
            )
        )
        v4_sources[fname] = SourceDocument(
            document_id=fname,
            text=body_text,
            authority_rank=rank,
            dependence_group=dk,
            attribution="authoritative_instrument" if gov else "source_statement",
        )
    ledger = Ledger(sources=v3_sources)
    layer = EvidenceLayer(
        vocabulary=SECURITIZATION_VOCABULARY,
        policy=SECURITIZATION_ACCEPTANCE_POLICY,
        reader=reader,
    )
    return LoadedCaseCorpus(
        case_id=case_id,
        task_id=task_id,
        dataset_role=dataset_role,
        common_evidence_sha256=common_sha256,
        common_characters=common_chars,
        total_lines=total_lines,
        documents=loaded_docs,
        v3_sources=tuple(v3_sources),
        ledger=ledger,
        v4_sources=v4_sources,
        evidence_layer=layer,
        task_instructions=task_instructions,
    )


def load_case_documents(
    case_id: str,
    manifest_path: Path | None = None,
) -> LoadedCaseCorpus:
    """Load full source documents and verify SHA-256 provenance against document_manifest.json."""
    mpath = manifest_path or DEFAULT_MANIFEST_PATH
    assert_no_evaluator_only_access(mpath)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    cases = manifest.get("cases") or {}
    if case_id not in cases:
        raise KeyError(f"Unknown case_id '{case_id}' in {mpath}")
    entry = cases[case_id]
    source_dir = Path(entry["source_dir"])
    assert_no_evaluator_only_access(source_dir)
    common_path = source_dir / entry["common_evidence_file"]
    raw_bytes = common_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    expected_sha256 = entry["common_evidence_sha256"]
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {common_path}: expected {expected_sha256}, got {actual_sha256}"
        )

    full_text = raw_bytes.decode("utf-8")
    all_lines = full_text.splitlines()

    doc_blocks: dict[str, tuple[str, int, str]] = {}
    cur_key: str | None = None
    cur_name: str = ""
    cur_start: int = 1
    buf: list[str] = []

    begin_re = re.compile(r"^BEGIN DOCUMENT (D\d+):\s*(.+?)\s*$")
    end_re = re.compile(r"^END DOCUMENT (D\d+)\s*$")
    line_prefix_re = re.compile(r"^L\d+:\s?(.*)$")

    for line_no, raw_line in enumerate(all_lines, 1):
        m_begin = begin_re.match(raw_line)
        if m_begin:
            cur_key = m_begin.group(1)
            cur_name = m_begin.group(2)
            cur_start = line_no + 1
            buf = []
            continue
        m_end = end_re.match(raw_line)
        if m_end and cur_key == m_end.group(1):
            doc_blocks[cur_key] = (cur_name, cur_start, "\n".join(buf))
            cur_key = None
            buf = []
            continue
        if cur_key is not None:
            m_lp = line_prefix_re.match(raw_line)
            buf.append(m_lp.group(1) if m_lp else raw_line)

    docs_cfg: list[tuple[str, str, str, str, int, bool, int, str]] = []
    for doc_meta in entry["documents"]:
        dk = doc_meta["doc_key"]
        fname = doc_meta["filename"]
        if dk not in doc_blocks:
            raise ValueError(f"Missing document block {dk} ({fname}) in {common_path}")
        parsed_fname, start_line, body_text = doc_blocks[dk]
        if parsed_fname != fname:
            raise ValueError(f"Document block {dk} filename mismatch: {parsed_fname!r} != {fname!r}")
        docs_cfg.append((
            dk,
            fname,
            doc_meta["role"],
            doc_meta["authority_tier"],
            int(doc_meta["authority_rank"]),
            bool(doc_meta["governs_on_conflict"]),
            start_line,
            body_text,
        ))

    return _build_corpus_objects(
        case_id=case_id,
        task_id=entry["task_id"],
        dataset_role=entry["dataset_role"],
        docs_cfg=docs_cfg,
        common_sha256=actual_sha256,
        common_chars=len(full_text),
        total_lines=len(all_lines),
        task_instructions=full_text.partition("\nDOCUMENT INVENTORY")[0].strip()
        if "\nDOCUMENT INVENTORY" in full_text else entry["task_id"],
    )


def build_synthetic_securitization_corpus(*, reader: Any = None) -> LoadedCaseCorpus:
    """Build an explicitly labeled synthetic_replay corpus aligned with Harness2 v3 excerpts.

    Reference readings are produced strictly through the configured ``reader`` on
    ``EvidenceLayer`` (defaulting to the recorded unedited real-model outputs).
    """
    indenture_text = (
        "Section 2.04: Note Interest Rates. The Class A Notes bear interest at 5.10% per annum, "
        "Class B Notes at 5.95% per annum, and Class C Notes ($50,000,000 principal) bear interest at 6.75% per annum.\n"
        "Section 5.01: Priority of Distributions (Waterfall). On each Payment Date, Available Funds distribute: "
        "Step 1 Servicing Fee; Step 2 Indenture Trustee Fee; Steps 3-7 Class A-1, A-2, A-3, B, and C Interest; "
        "Steps 8-12 Class A-1, A-2, A-3, B, and C Principal; "
        "Step 13 Reserve Account Replenishment (junior to Class B and Class C principal); "
        "Step 14 Certificateholder.\n"
        "Section 10.01: Events of Default. Failure to pay interest continuing for 5 business days constitutes an Event of Default.\n"
        "Section 12.01: Optional Redemption (Clean-Up Call). Servicer may redeem Notes when Pool Balance falls to 10% of Initial Pool Balance."
    )
    om_text = (
        "Summary & Description of Notes: Class A Notes 5.10% per annum, Class B Notes 5.95% per annum, "
        "Class C Notes ($50,000,000 principal) 6.50% per annum (25 bps / 0.25 percentage points lower than Indenture).\n"
        "Flow of Funds: Steps 1-9 Fees, Interest, and Class A Principal; Step 6/10 Reserve Account Replenishment "
        "(placed senior/ahead of Class B and Class C principal).\n"
        "Events of Default Draft Note: 5 calendar days (or 3 business days) cure period variant.\n"
        "Optional Redemption: Servicer may redeem Notes when Pool Balance falls to 15% of Initial Pool Balance."
    )
    email_text = (
        "From: Associate <associate@counsel.example>\n"
        "Please check Section 5.04 Reserve Account replenishment at Step 10 vs Step 13 and Class C 25 bps coupon gap ($125,000/yr). "
        "I only skimmed two sections."
    )
    docs_cfg = [
        ("D1", "associate-review-email.eml", "correspondence_notes", "draft", 10, False, 1, email_text),
        ("D2", "cart-2024-2-indenture.docx", "governing_operative_agreement", "authoritative", 100, True, 1, indenture_text),
        ("D3", "cart-2024-2-offering-memorandum.docx", "disclosure_document", "standard", 60, False, 1, om_text),
    ]
    combined = "\n".join(x[-1] for x in docs_cfg)
    eff_reader = (
        reader
        if reader is not None
        else RecordedProposalReader.from_file(_HARNESS_DIR / "recorded_reads" / "cart_synth.json")
    )
    corpus = _build_corpus_objects(
        case_id="cart_2024_2_synthetic_replay",
        task_id="structured-finance-securitization/synthetic-replay",
        dataset_role="synthetic_replay",
        docs_cfg=docs_cfg,
        common_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        common_chars=len(combined),
        total_lines=len(combined.splitlines()),
        reader=eff_reader,
    )
    corpus.evidence_layer.ingest(list(corpus.v4_sources.values()))
    return corpus


def build_section_comparison_packet(
    corpus: LoadedCaseCorpus,
    *,
    max_section_chars: int = 32000,
) -> dict[str, Any]:
    """Build a bounded, potentially lossy packet; section counts do not imply full-text coverage."""
    packet_docs: list[dict[str, Any]] = []
    # Keyword anchors prioritize financial/covenant/waterfall/default/definition clauses first if compacting,
    # while recording exact section coverage statistics across the full documents.
    priority_terms = (
        "interest", "coupon", "6.75", "6.50", "30/360", "actual/360", "waterfall", "priority",
        "reserve account", "servicing fee", "back-up", "backup", "clean-up", "optional redemption",
        "trigger", "cumulative net loss", "cnl", "5.01", "event of default", "supplemental",
        "sofr", "floor", "margin", "commitment fee", "leverage", "fixed charge", "fccr",
        "ebitda", "synerg", "acquisition", "restricted payment", "mfn", "excess cash flow",
        "change of control", "cross-default", "judgment", "reinvestment", "material adverse",
        "extension", "liquidity",
    )
    total_sections = 0
    included_sections = 0
    for fname, doc in sorted(corpus.documents.items(), key=lambda kv: (-kv[1].authority_rank, kv[0])):
        total_sections += len(doc.sections)
        selected: list[dict[str, Any]] = []
        used = 0
        # First pass: keep sections containing priority clauses or associate/partner notes
        ranked_secs = sorted(
            doc.sections,
            key=lambda s: (
                0 if (doc.authority_rank < 60 or doc.filename.endswith(".xlsx") or any(t in s.text.lower() or t in s.heading.lower() for t in priority_terms)) else 1,
                s.start_line,
            ),
        )
        for sec in ranked_secs:
            if used + len(sec.text) > max_section_chars and selected:
                continue
            chunk = sec.text[: max(400, max_section_chars - used)]
            selected.append({
                "heading": sec.heading,
                "lines": f"L{sec.start_line}-L{sec.end_line}",
                "text": chunk,
                "full_section_characters": len(sec.text),
                "included_characters": len(chunk),
                "truncated": len(chunk) < len(sec.text),
            })
            used += len(chunk)
            included_sections += 1
        selected.sort(key=lambda x: int(x["lines"].split("-")[0][1:]))
        packet_docs.append({
            "doc_key": doc.doc_key,
            "filename": doc.filename,
            "role": doc.role,
            "authority_tier": doc.authority_tier,
            "authority_rank": doc.authority_rank,
            "governs_on_conflict": doc.governs_on_conflict,
            "full_char_count": len(doc.text),
            "total_sections": len(doc.sections),
            "sections_in_packet": len(selected),
            "included_characters": sum(s["included_characters"] for s in selected),
            "full_text_included": sum(s["included_characters"] for s in selected) == sum(len(s.text) for s in doc.sections),
            "sections": selected,
        })
    return {
        "case_id": corpus.case_id,
        "dataset_role": corpus.dataset_role,
        "common_evidence_sha256": corpus.common_evidence_sha256,
        "full_corpus_characters": corpus.common_characters,
        "total_sections_across_corpus": total_sections,
        "sections_included_in_packet": included_sections,
        "documents": packet_docs,
    }


@dataclass
class LLMCallUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_clock_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    raw_latencies_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "wall_clock_ms": round(self.wall_clock_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
        }


class LocalQwenClient:
    """Zero-API-cost local OpenAI-compatible client targeting LM Studio (qwen/qwen3.5-9b)."""

    def __init__(
        self,
        base_url: str = DEFAULT_LOCAL_LLM_URL,
        model: str = DEFAULT_LOCAL_LLM_MODEL,
        timeout: float = 240.0,
        max_tokens: int = 6144,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        try:
            with urlopen(f"{self.base_url}/models", timeout=3.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        usage_tracker: LLMCallUsage | None = None,
        temperature: float = 0.0,
    ) -> str:
        sys_eff = system_prompt if "/no_think" in system_prompt else f"{system_prompt}\n/no_think"
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": sys_eff},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urlopen(req, timeout=self.timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        msg = raw["choices"][0]["message"]
        content = msg.get("content") or ""
        # Strip any residual thinking tags if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        if not content:
            content = str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        usage = raw.get("usage") or {}
        if usage_tracker is not None:
            usage_tracker.calls += 1
            usage_tracker.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            usage_tracker.completion_tokens += int(usage.get("completion_tokens") or 0)
            usage_tracker.total_tokens += int(usage.get("total_tokens") or 0)
            usage_tracker.wall_clock_ms += dt_ms
            usage_tracker.raw_latencies_ms.append(dt_ms)
        return content


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Parse strict or fenced JSON object from LLM text if present."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class ExtractedReportStructure:
    report_text: str
    assertions: list[Assertion]
    graph: JustificationGraph | None
    governing_waterfall: list[WaterfallStepSpec]
    claimed_waterfall: list[WaterfallStepSpec]
    claimed_waterfall_relations: list[tuple[str, str]]
    allow_cross_regime_waterfall: bool
    question: str
    required_section_keys: list[str]
    extraction_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SecuritizationReportExtractor:
    """Extracts verifiable v3 assertions + v4 justification/waterfall structures from a candidate report."""

    def extract(
        self,
        candidate: str | Mapping[str, Any],
        corpus: LoadedCaseCorpus,
        *,
        required_sections: Sequence[str] | None = None,
        vocabulary: Any | None = None,
    ) -> ExtractedReportStructure:
        default_q = (
            f"Compare documents in {corpus.case_id} and identify material discrepancies "
            "governed by highest-authority instruments."
        )
        if isinstance(candidate, Mapping):
            if candidate.get("force_extraction_failure"):
                return ExtractedReportStructure(
                    report_text=str(candidate.get("report_text") or ""),
                    assertions=[],
                    graph=None,
                    governing_waterfall=[],
                    claimed_waterfall=[],
                    claimed_waterfall_relations=[],
                    allow_cross_regime_waterfall=False,
                    question=str(candidate.get("question") or default_q),
                    required_section_keys=list(required_sections or candidate.get("required_sections") or []),
                    extraction_error=str(candidate.get("extraction_error") or "Structured extraction failed on malformed payload."),
                )
            text = str(candidate.get("report_text") or "").strip()
            raw_assertions = candidate.get("assertions") or []
            raw_graph = candidate.get("graph")
            raw_wf = candidate.get("waterfall") or {}
            req_secs = list(required_sections or candidate.get("required_sections") or [])
            question = str(candidate.get("question") or default_q)
            metadata = dict(candidate.get("metadata") or {})
        else:
            raw_str = str(candidate or "").strip()
            maybe_json = _extract_json_block(raw_str)
            if maybe_json and "report_text" in maybe_json:
                return self.extract(maybe_json, corpus, required_sections=required_sections)
            text = raw_str
            raw_assertions = []
            raw_graph = None
            raw_wf = {}
            req_secs = list(required_sections or [])
            question = default_q
            metadata = {}

        if not text:
            return ExtractedReportStructure(
                report_text="",
                assertions=[],
                graph=None,
                governing_waterfall=[],
                claimed_waterfall=[],
                claimed_waterfall_relations=[],
                allow_cross_regime_waterfall=False,
                question=question,
                required_section_keys=req_secs,
                extraction_error="Candidate report text is empty.",
            )

        assertions: list[Assertion] = []
        for idx, item in enumerate(raw_assertions):
            if isinstance(item, Assertion):
                assertions.append(item)
            elif isinstance(item, Mapping):
                try:
                    raw_pred = str(item["predicate"])
                    raw_subj = str(item["subject"])
                    role_val = str(item.get("role") or item.get("claim_role") or "").strip().lower()
                    if role_val == "subordinate_discrepancy":
                        pred_sub_low = f"{raw_pred} {raw_subj}".lower()
                        if not any(
                            tag in pred_sub_low
                            for tag in ("discrepancy", "om_", "subordinate", "email_", "erroneous", "counterpart")
                        ):
                            raw_pred = f"om_discrepancy_{raw_pred}"
                    raw_unit = item.get("unit")
                    if isinstance(raw_unit, str) and raw_unit.strip().lower() in {
                        "percent_per_annum",
                        "percent per annum",
                        "% per annum",
                    }:
                        raw_unit = "PERCENT"
                    assertions.append(
                        Assertion(
                            subject=raw_subj,
                            predicate=raw_pred,
                            value=str(item["value"]),
                            source_id=str(item["source_id"]),
                            quote=str(item["quote"]),
                            unit=raw_unit,
                            condition=item.get("condition"),
                        )
                    )
                except Exception as exc:
                    return ExtractedReportStructure(
                        report_text=text,
                        assertions=[],
                        graph=None,
                        governing_waterfall=[],
                        claimed_waterfall=[],
                        claimed_waterfall_relations=[],
                        allow_cross_regime_waterfall=False,
                        question=question,
                        required_section_keys=req_secs,
                        extraction_error=f"Malformed assertion at index {idx}: {exc}",
                    )
            else:
                return ExtractedReportStructure(
                    report_text=text,
                    assertions=[],
                    graph=None,
                    governing_waterfall=[],
                    claimed_waterfall=[],
                    claimed_waterfall_relations=[],
                    allow_cross_regime_waterfall=False,
                    question=question,
                    required_section_keys=req_secs,
                    extraction_error=f"Invalid assertion entry type at index {idx}.",
                )

        # If candidate is freeform Markdown without explicit assertions, infer grounded section references
        if not assertions and not isinstance(candidate, Mapping):
            assertions = self._infer_assertions_from_markdown(text, corpus)

        graph: JustificationGraph | None = None
        if isinstance(raw_graph, JustificationGraph):
            graph = raw_graph
        elif isinstance(raw_graph, Mapping):
            try:
                graph = JustificationGraph.from_response(
                    raw_graph,
                    vocabulary=vocabulary or SECURITIZATION_VOCABULARY,
                )
            except Exception as exc:
                return ExtractedReportStructure(
                    report_text=text,
                    assertions=assertions,
                    graph=None,
                    governing_waterfall=[],
                    claimed_waterfall=[],
                    claimed_waterfall_relations=[],
                    allow_cross_regime_waterfall=False,
                    question=question,
                    required_section_keys=req_secs,
                    extraction_error=f"Invalid JustificationGraph structure: {exc}",
                )

        def _parse_steps(items: Sequence[Any]) -> list[WaterfallStepSpec]:
            out: list[WaterfallStepSpec] = []
            for st in items:
                if isinstance(st, WaterfallStepSpec):
                    out.append(st)
                elif isinstance(st, Mapping):
                    out.append(
                        WaterfallStepSpec(
                            step_id=str(st["step_id"]),
                            beneficiary=str(st["beneficiary"]),
                            payment_type=str(st.get("payment_type") or "principal"),
                            position=st.get("position"),
                            before=tuple(st.get("before") or ()),
                            after=tuple(st.get("after") or ()),
                            conditions=tuple(st.get("conditions") or ()),
                            regime=str(st.get("regime") or "pre_acceleration"),
                            document_id=str(st.get("document_id") or ""),
                            clause_id=str(st.get("clause_id") or ""),
                            revision=int(st.get("revision") or 1),
                        )
                    )
            return out

        gov_wf = _parse_steps(raw_wf.get("governing_steps") or []) if isinstance(raw_wf, Mapping) else []
        sub_wf = _parse_steps(raw_wf.get("claimed_steps") or raw_wf.get("subordinate_steps") or []) if isinstance(raw_wf, Mapping) else []
        rel_claims: list[tuple[str, str]] = []
        if isinstance(raw_wf, Mapping):
            for pair in raw_wf.get("claimed_relations") or []:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    rel_claims.append((str(pair[0]), str(pair[1])))
        allow_cross = bool(raw_wf.get("allow_cross_regime_if_qualified")) if isinstance(raw_wf, Mapping) else False

        return ExtractedReportStructure(
            report_text=text,
            assertions=assertions,
            graph=graph,
            governing_waterfall=gov_wf,
            claimed_waterfall=sub_wf,
            claimed_waterfall_relations=rel_claims,
            allow_cross_regime_waterfall=allow_cross,
            question=question,
            required_section_keys=req_secs,
            metadata=metadata,
        )

    def _infer_assertions_from_markdown(
        self, text: str, corpus: LoadedCaseCorpus
    ) -> list[Assertion]:
        """Deterministically extract citeable value-section associations from freeform Markdown reports."""
        low = text.lower()
        docs_by_rank = sorted(corpus.documents.values(), key=lambda d: -d.authority_rank)
        gov_doc = docs_by_rank[0] if docs_by_rank else None
        if gov_doc is None:
            return []

        assertions: list[Assertion] = []
        # Extract section quotes matching numbers in text against the corpus sources to enable v3 quote + rank check
        section_probes = [
            ("coupon_class_c", r"(?:class\s+c[^\n]{0,80}?)(6\.75%|6\.50%)", "6.75%"),
            ("waterfall_reserve", r"(?:reserve\s+account[^\n]{0,120}?step\s*(?:thirteenth|13|tenth|10|sixth|6))", "Step 13"),
            ("day_count", r"(30/360|actual/360)", "30/360"),
            ("servicing_fee_base", r"(?:servicing\s+fee[^\n]{0,140}?(?:pool\s+balance|note\s+balance))", "Pool Balance"),
            ("clean_up_call", r"(?:\$71,230,000|\$68,750,000|10%\s+of\s+(?:initial\s+)?(?:pool|note)\s+balance|15%)", "10% Initial Pool Balance"),
            ("reserve_floor", r"(?:\$3,437,500|0\.50%\s+of\s+initial\s+note\s+balance)", "$3,437,500 floor"),
            ("cnl_trigger", r"(?:july\s+2027[^\n]{0,80}?(?:6\.00%|5\.75%))", "6.00%"),
            ("backup_servicer", r"(?:granite\s+loan\s+servicing|back-?up\s+servicer)", "Granite Loan Servicing LLC"),
        ]
        for pred, pat, _ in section_probes:
            m = re.search(pat, low)
            if not m:
                continue
            token = m.group(0).strip()
            # Find which source actually contains the claimed phrase
            matched_doc = None
            quote = token
            for doc in docs_by_rank:
                dm = re.search(re.escape(m.group(1)) if m.groups() else re.escape(token[:40]), doc.text, re.I)
                if dm:
                    matched_doc = doc
                    start = max(0, dm.start() - 40)
                    end = min(len(doc.text), dm.end() + 80)
                    quote = " ".join(doc.text[start:end].split())
                    break
            if matched_doc is None:
                matched_doc = docs_by_rank[-1]
            assertions.append(
                Assertion(
                    subject="Deal_Report",
                    predicate=pred,
                    value=m.group(1) if m.groups() else token,
                    source_id=matched_doc.filename,
                    quote=quote,
                )
            )
        return assertions


def _find_governing_doc(corpus: LoadedCaseCorpus) -> LoadedDocument:
    return max(corpus.documents.values(), key=lambda d: d.authority_rank)


def _check_class_coupon_binding(text: str, corpus: LoadedCaseCorpus) -> list[dict[str, Any]]:
    """Verify Class A-1 / A-2 / A-3 / B / C coupons are never swapped across classes or inverted to OM 6.50%."""
    gov = _find_governing_doc(corpus)
    expected: dict[str, str] = {}
    for cls_name in ("Class A-1", "Class A-2", "Class A-3", "Class B", "Class C"):
        m = re.search(
            rf"{re.escape(cls_name)}\s+Notes?[^.\n]{{0,120}}?(\d+\.\d+%)",
            gov.text,
            re.I,
        )
        if m:
            expected[cls_name.lower()] = m.group(1)
    if not expected:
        return []

    issues: list[dict[str, Any]] = []
    # Detect explicit claims attributing governing/correct/Indenture coupon to a class
    claim_patterns = [
        re.compile(
            r"(?P<cls>Class\s+(?:A-[123]|B|C))\s+Notes?[^.\n]{0,80}?(?:is|are|at|of|bears?(?:\s+interest\s+at)?|fixed\s+at|governs?\s+(?:at|as)|should\s+be)\s+(?P<rate>\d+\.\d+%)",
            re.I,
        ),
    ]
    for pat in claim_patterns:
        for m in pat.finditer(text):
            cls_key = " ".join(m.group("cls").lower().split())
            rate = m.group("rate")
            exp_rate = expected.get(cls_key)
            if not exp_rate or rate == exp_rate:
                continue
            span_ctx = text[max(0, m.start() - 50) : min(len(text), m.end() + 70)].lower()
            # Allow quoting OM's erroneous 6.50% ONLY when clearly contrasted as Offering Memorandum / erroneous / incorrect vs Indenture
            is_om_contrast = (
                rate == "6.50%"
                and cls_key == "class c"
                and any(tok in span_ctx for tok in ("offering memorandum", "om ", "om'", "discrepancy", "incorrect", "erroneous", "understate", "vs.", "versus", "instead of"))
                and exp_rate in text
            )
            if is_om_contrast:
                continue
            # Also allow stating OM summary table vs description contrast if 6.75% is stated as governing
            swapped_with = [k.title() for k, v in expected.items() if v == rate and k != cls_key]
            msg = (
                f"Coupon mismatch for {cls_key.title()}: text states {rate}, which belongs to {swapped_with[0]}"
                if swapped_with
                else f"Coupon mismatch for {cls_key.title()}: text asserts {rate}, whereas governing {gov.filename} specifies {exp_rate}"
            )
            issues.append({
                "code": "CLASS_COUPON_MISMATCH",
                "source_layer": "v4",
                "repairable": True,
                "message": msg,
                "remediation": f"Bind {cls_key.title()} strictly to governing rate {exp_rate} from {gov.filename} (do not swap across classes or adopt subordinate OM rate).",
                "forbidden_shortcuts": [f"Do not delete {cls_key.title()} coupon analysis to evade verification."],
            })
    return issues


def _check_authority_and_section_fidelity(
    extracted: ExtractedReportStructure, corpus: LoadedCaseCorpus
) -> list[dict[str, Any]]:
    """Detect authority inversions (Email/OM overriding Indenture) and false Section 5.04 citation trap."""
    issues: list[dict[str, Any]] = []
    text = extracted.report_text
    low = text.lower()
    gov = _find_governing_doc(corpus)

    # 1. Assertion-level authority rank check (cannot rely on draft/subordinate source to assert governing terms when governing exists)
    if corpus.case_id.startswith("cart_2024_2"):
        for a in extracted.assertions:
            doc = corpus.documents.get(a.source_id)
            if doc and doc.authority_rank < gov.authority_rank:
                # Only permit subordinate citation if predicate/subject explicitly marks an OM/email discrepancy rather than governing rule
                pred_sub_low = f"{a.predicate} {a.subject}".lower()
                if not any(tag in pred_sub_low for tag in ("discrepancy", "om_", "subordinate", "email_", "erroneous", "counterpart")):
                    issues.append({
                        "code": "AUTHORITY_INVERSION",
                        "source_layer": "v3",
                        "repairable": True,
                        "message": f"Assertion '{a.predicate}={a.value}' treats lower-rank source '{a.source_id}' (rank {doc.authority_rank}) as governing over '{gov.filename}' (rank {gov.authority_rank}).",
                        "remediation": f"Ground governing determination in '{gov.filename}' (rank {gov.authority_rank}).",
                        "forbidden_shortcuts": ["Do not drop the disputed section instead of citing the governing document."],
                    })

        # Prose-level check: claiming OM governs over Indenture, or Email governs, or repeating un-corrected Section 5.04(a) trap
        if re.search(r"(?:offering\s+memorandum|om|email)[^.\n]{0,60}(?:governs?|controls?|supersedes?|prevails?)[^.\n]{0,60}indenture", low):
            issues.append({
                "code": "AUTHORITY_INVERSION",
                "source_layer": "v4",
                "repairable": True,
                "message": "Report claims Offering Memorandum or Associate Email governs over the executed Indenture.",
                "remediation": "State explicitly that the executed Indenture (Rank 100) governs over the Offering Memorandum (Rank 60) and Associate Email (Rank 10).",
                "forbidden_shortcuts": ["Do not delete authority hierarchy discussion."],
            })
        if ("section 5.04" in low or "§5.04" in low) and not (
            "5.01" in low
            and any(
                k in low
                for k in (
                    "does not exist",
                    "non-existent",
                    "no section 5.04",
                    "mistakenly",
                    "mis-cite",
                    "miscite",
                    "incorrectly cite",
                    "error in the associate",
                    "actually section 5.01",
                    "correct cite",
                    "not §5.04",
                    "not section 5.04",
                    "uses §5.01",
                    "correct section is",
                )
            )
        ):
            issues.append({
                "code": "FALSE_SECTION_CITATION_FROM_EMAIL",
                "source_layer": "v4",
                "repairable": True,
                "message": "Report repeats associate email's false citation to 'Section 5.04' instead of Indenture Section 5.01(a).",
                "remediation": "Correct citation to Indenture Section 5.01(a) and flag that Associate Email D1 mis-cited Section 5.04(a).",
                "forbidden_shortcuts": ["Do not omit the waterfall section citation."],
            })
    return issues


def _check_calendar_vs_business_days_and_rate_units(text: str) -> list[dict[str, Any]]:
    """Detect day-count / calendar-vs-business day confusion and % vs percentage points / bps errors."""
    issues: list[dict[str, Any]] = []
    low = text.lower()
    # Explicit equality/equivalence claim between calendar days and business days
    if re.search(r"\bcalendar\s+days?\b[^.\n]{0,40}(?:same\s+as|equivalent\s+to|=|equal\s+to)\s+[^.\n]{0,30}\bbusiness\s+days?\b", low) or (
        "calendar days" in low and "business days" in low and re.search(r"5\s+calendar\s+days\s*(?:=|equals|means)\s*5\s+business\s+days", low)
    ):
        issues.append({
            "code": "UNIT_MISMATCH_CALENDAR_VS_BUSINESS_DAYS",
            "source_layer": "v4",
            "repairable": True,
            "message": "Text conflates calendar days with business days; these are incompatible time units.",
            "remediation": "Distinguish calendar days from business days explicitly and do not treat them as equal.",
            "forbidden_shortcuts": ["Do not delete the grace-period day count unit."],
        })
    # Check 25% vs 25 bps / 0.25 percentage points confusion on Class C / Margin grid
    if re.search(r"\b25\s*%\s+(?:discrepancy|difference|spread|gap|higher|lower|increase)\b", low) or re.search(
        r"\b25\s*(?:bps|basis\s+points?)\s*(?:=|equals|or)\s*25\s*%\b", low
    ):
        issues.append({
            "code": "PERCENT_VS_PERCENTAGE_POINT_CONFUSION",
            "source_layer": "v4",
            "repairable": True,
            "message": "Response confuses 25 bps (0.25 percentage points) with 25%.",
            "remediation": "Express 25 basis points accurately as 0.25 percentage points (0.0025 decimal), never 25%.",
            "forbidden_shortcuts": ["Do not omit the numeric spread differential."],
        })
    return issues


def _check_financial_assumptions_and_damages(text: str) -> list[dict[str, Any]]:
    """Enforce mandatory assumptions on annualized rate-spread dollar calculations and block automatic damages."""
    issues: list[dict[str, Any]] = []
    low = text.lower()
    has_annual_dollar = bool(
        re.search(r"\$(?:125,000|248,000|375,000|401,000)\b|\$125k|\$248k", low)
    )
    if has_annual_dollar:
        has_principal_qual = any(
            k in low
            for k in (
                "constant principal", "initial principal", "static principal",
                "assuming principal", "before amortization", "outstanding principal remains",
                "simple annual", "annualized", "simple interest", "illustrat",
            )
        )
        if not has_principal_qual:
            issues.append({
                "code": "OMITTED_DERIVATION_ASSUMPTIONS",
                "source_layer": "v4",
                "repairable": True,
                "message": "Annualized dollar impact (e.g., $125,000/yr) stated without explicit constant-principal / simple-annual calculation assumptions.",
                "remediation": "Explicitly state that the dollar impact is a simple annualized calculation assuming constant initial principal (before amortization or day-count effects).",
                "forbidden_shortcuts": ["Do not remove the quantitative impact estimate to evade qualification."],
            })
        damages_re = re.compile(
            r"\b(?:automatic(?:ally)?|guaranteed|certain)\s+(?:legal\s+)?damages\b"
            r"|\bproved?\s+(?:actual\s+)?(?:realized\s+)?(?:cashflow|cash[- ]flow)\s+losses\b"
        )
        unqualified_damages = False
        for m in damages_re.finditer(low):
            clause_prefix = low[max(0, m.start() - 80) : m.start()]
            # Split at sentence boundaries so earlier negation does not mask a later claim.
            clause_prefix = re.split(r"[.!?\n]", clause_prefix)[-1]
            if not re.search(
                r"\b(?:not|no|never|neither|nor|without|rather\s+than|instead\s+of|not\s+equal|cannot|does\s+not|do\s+not|should\s+not|is\s+not|are\s+not)\b",
                clause_prefix,
            ):
                unqualified_damages = True
                break
        if unqualified_damages:
            issues.append({
                "code": "UNQUALIFIED_DAMAGES_OR_CASHFLOW_CLAIM",
                "source_layer": "v4",
                "repairable": True,
                "message": "Annualized spread impact is presented as automatic legal damages or guaranteed realized cashflow losses.",
                "remediation": "Clarify that the figure is an illustrative annualized exposure sensitivity, not automatic legal damages or realized cashflow loss.",
                "forbidden_shortcuts": ["Do not replace with vague speculation."],
            })
    return issues


@dataclass
class PipelineVerificationResult:
    report_disposition: str  # ACCEPT | RETRY | ESCALATE | ABSTAIN
    accepted: bool
    operational_authorization: dict[str, Any]
    issues: list[dict[str, Any]]
    consolidated_feedback: dict[str, Any]
    v3_verdict_status: str | None
    v4_review_disposition: str | None
    response_digest: str | None
    assessment_digest: str | None
    ledger_state_digest: str
    verifier_ms: float
    cache_metrics: dict[str, int]
    corpus_obligations_total: int = 0
    corpus_obligations_covered: int = 0
    corpus_obligations_rejected: int = 0
    corpus_obligation_ids: tuple[str, ...] = ()
    corpus_covered_ids: tuple[str, ...] = ()
    corpus_uncovered_ids: tuple[str, ...] = ()
    corpus_rejected_ids: tuple[str, ...] = ()
    signed_receipt: dict[str, Any] | None = None


@dataclass(frozen=True)
class DomainAdapter:
    """Declarative domain contract decoupling verification rules from a single case."""

    domain_id: str
    vocabulary: Any
    acceptance_policy: Any
    subordinate_roles: frozenset[str]
    authority_check_fn: Callable[[ExtractedReportStructure, LoadedCaseCorpus], list[dict[str, Any]]]
    v4_prose_check_fn: Callable[[ExtractedReportStructure, LoadedCaseCorpus], list[dict[str, Any]]]
    resolve_governing_doc_fn: Callable[[LoadedCaseCorpus, str | None], LoadedDocument]
    excluded_comparison_roles: Mapping[str, str] = field(default_factory=dict)


def _check_securitization_v4_prose(
    parsed: ExtractedReportStructure, corpus: LoadedCaseCorpus
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    issues.extend(_check_class_coupon_binding(parsed.report_text, corpus))
    issues.extend(_check_calendar_vs_business_days_and_rate_units(parsed.report_text))
    issues.extend(_check_financial_assumptions_and_damages(parsed.report_text))
    return issues


def _resolve_banking_governing_doc(
    corpus: LoadedCaseCorpus, category: str | None = None
) -> LoadedDocument:
    """Resolve context-dependent governing document in banking/finance by role & category."""
    preferred_roles = resolve_banking_governing_role(category)
    for role in preferred_roles:
        for doc in corpus.documents.values():
            if doc.role == role:
                return doc
    return _find_governing_doc(corpus)


def _check_banking_authority_and_markup_fidelity(
    extracted: ExtractedReportStructure, corpus: LoadedCaseCorpus
) -> list[dict[str, Any]]:
    """Enforce context-dependent authority in banking-finance without CART heuristics."""
    issues: list[dict[str, Any]] = []
    low = extracted.report_text.lower()
    playbook_doc = _resolve_banking_governing_doc(corpus, "negotiation_policy")
    baseline_doc = _resolve_banking_governing_doc(corpus, "baseline_terms")

    for a in extracted.assertions:
        doc = corpus.documents.get(a.source_id)
        if doc and doc.role in BANKING_SUBORDINATE_ROLES:
            pred_sub_low = f"{a.predicate} {a.subject}".lower()
            if not any(
                tag in pred_sub_low
                for tag in ("discrepancy", "markup", "om_", "subordinate", "counterpart", "redline", "proposed")
            ):
                issues.append({
                    "code": "AUTHORITY_INVERSION",
                    "source_layer": "v3",
                    "repairable": True,
                    "message": (
                        f"Assertion '{a.predicate}={a.value}' treats counterparty markup '{a.source_id}' "
                        f"(role='{doc.role}') as governing over '{baseline_doc.filename}' / '{playbook_doc.filename}'."
                    ),
                    "remediation": (
                        f"Mark counterparty markup assertions with role='subordinate_discrepancy' and ground "
                        f"governing terms/policy in '{baseline_doc.filename}' or '{playbook_doc.filename}'."
                    ),
                    "forbidden_shortcuts": ["Do not drop counterparty markup comparisons to evade verification."],
                })

    if re.search(
        r"(?:lender|counterparty)\s+markup[^.\n]{0,60}(?:governs?|controls?|supersedes?|prevails?)[^.\n]{0,60}(?:playbook|original\s+term\s+sheet)",
        low,
    ):
        issues.append({
            "code": "AUTHORITY_INVERSION",
            "source_layer": "v4",
            "repairable": True,
            "message": "Report claims counterparty markup governs over client negotiation playbook or baseline term sheet.",
            "remediation": "Distinguish counterparty markup proposals from baseline term sheet and client playbook limits.",
            "forbidden_shortcuts": ["Do not delete authority hierarchy discussion."],
        })
    return issues


def _check_banking_v4_prose(
    parsed: ExtractedReportStructure, corpus: LoadedCaseCorpus
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    low = parsed.report_text.lower()
    if re.search(r"\b\d+\s*%\s+(?:leverage|x)\b", low):
        issues.append({
            "code": "UNIT_MISMATCH_RATIO_VS_MULTIPLE",
            "source_layer": "v4",
            "repairable": True,
            "message": "Leverage ratio is expressed with '%' instead of a leverage multiple ('x').",
            "remediation": "Express leverage ratios as multiples (e.g., '4.5x'), not percentages.",
            "forbidden_shortcuts": ["Do not remove leverage ratio figures."],
        })
    return issues


def get_domain_adapter(
    corpus: LoadedCaseCorpus, domain_profile: DomainProfile | None = None
) -> DomainAdapter:
    """Select the declarative DomainAdapter for the corpus or explicit DomainProfile."""
    vocab_name = getattr(getattr(domain_profile, "vocabulary", None), "name", "")
    is_banking = (
        corpus.case_id.startswith("banking")
        or "banking-finance" in (corpus.task_id or "").lower()
        or vocab_name == "banking-finance"
        or any(d.role in BANKING_SUBORDINATE_ROLES for d in corpus.documents.values())
    )
    if is_banking:
        return DomainAdapter(
            domain_id="banking_finance",
            vocabulary=BANKING_FINANCE_VOCABULARY,
            acceptance_policy=BANKING_FINANCE_ACCEPTANCE_POLICY,
            subordinate_roles=BANKING_SUBORDINATE_ROLES,
            authority_check_fn=_check_banking_authority_and_markup_fidelity,
            v4_prose_check_fn=_check_banking_v4_prose,
            resolve_governing_doc_fn=_resolve_banking_governing_doc,
            excluded_comparison_roles=dict(BANKING_EXCLUDED_COMPARISON_ROLES),
        )
    return DomainAdapter(
        domain_id="securitization",
        vocabulary=SECURITIZATION_VOCABULARY,
        acceptance_policy=SECURITIZATION_ACCEPTANCE_POLICY,
        subordinate_roles=frozenset({"disclosure_document", "correspondence_notes"}),
        authority_check_fn=_check_authority_and_section_fidelity,
        v4_prose_check_fn=_check_securitization_v4_prose,
        resolve_governing_doc_fn=lambda c, _cat=None: _find_governing_doc(c),
        excluded_comparison_roles=dict(SECURITIZATION_EXCLUDED_COMPARISON_ROLES),
    )


def _scope_report_text_for_graph(report_text: str, graph: JustificationGraph) -> str:
    """Scope long multi-section reports to sentences relevant to the graph's claims & equations."""
    subjects = {
        str(c.subject).strip().lower()
        for c in graph.claims.values()
        if getattr(c, "subject", None)
    }
    conclusion = graph.claims.get(graph.conclusion_id)
    conc_val = str(conclusion.value).strip().lower() if conclusion and conclusion.value is not None else ""
    lines = [ln.strip() for ln in report_text.splitlines() if ln.strip()]
    selected: list[str] = []
    for ln in lines:
        low = ln.lower()
        if (
            any(subj in low for subj in subjects)
            or (conc_val and conc_val in low)
            or "=" in ln
            or "governs" in low
            or "according to" in low
            or "per the" in low
        ):
            selected.append(ln)
    return "\n".join(selected) if selected else report_text


def _check_required_section_deliverable_coverage(
    *,
    required_keys: Sequence[str],
    assertions: Sequence[Assertion],
    v4_sources: Mapping[str, SourceDocument],
    graph: JustificationGraph | None,
    v4_review: Any | None,
    gov_filename: str,
) -> list[dict[str, Any]]:
    """Enforce unified v3+v4 coverage for required comparison sections.

    In combined mode, a bare v3 ``Assertion`` or surface keyword never satisfies
    ``MISSING_REQUIRED_SECTION`` on its own: each required section must be backed
    by a verified (non-refuted, non-opinion) graph leaf overlapping the cited
    source span and linked in a grounded deliverable correspondence clause.
    """
    issues: list[dict[str, Any]] = []
    unverified_ids: set[str] = set()
    clause_claims: Sequence[Mapping[str, Any]] = ()
    if v4_review is not None:
        for chk in getattr(v4_review.assessment, "checks", ()) or ():
            if chk.step_id.startswith("leaf::") and chk.status != "verified":
                unverified_ids.add(chk.step_id[len("leaf::") :])
        corr_rep = getattr(v4_review.assessment, "correspondence_report", None) or {}
        clause_claims = corr_rep.get("clause_claims") or ()

    verified_leaves = [
        node
        for cid, node in sorted((getattr(graph, "claims", {}) or {}).items())
        if cid not in unverified_ids
        and getattr(node, "kind", "") != "opinion"
        and getattr(node, "assertion_kind", "") != "opinion"
        and (getattr(node, "evidence", ()) or getattr(node, "evidence_unit_ids", ()))
    ]

    def _leaf_linked_to_clause(node: Any, sec_tokens: Sequence[str]) -> bool:
        for cc in clause_claims:
            if not isinstance(cc, Mapping):
                continue
            if cc.get("role") == "opinion" or not cc.get("is_grounded", False):
                continue
            rels = cc.get("relations") or ()
            if any(
                isinstance(r, Mapping)
                and r.get("matched_claim_id") == node.claim_id
                and r.get("status") == "verified"
                for r in rels
            ):
                return True
            ct = str(cc.get("clause_text", "") or "")
            if ct and _value_in_text(getattr(node, "value", None), ct):
                val_s = str(getattr(node, "value", "") or "").strip()
                is_alpha_enum = any(ch.isalpha() for ch in val_s) and len(val_s) >= 4
                anchor_tokens = [
                    tok
                    for tok in re.split(
                        r"[^a-z0-9]+",
                        f"{getattr(node, 'subject', '')} {getattr(node, 'predicate', '')} {' '.join(sec_tokens)}".lower(),
                    )
                    if len(tok) >= 3
                ]
                ct_low = ct.lower()
                if is_alpha_enum or any(tok in ct_low for tok in anchor_tokens):
                    return True
        return False

    for req_sec in required_keys:
        tokens = [t for t in re.split(r"[^a-z0-9]+", req_sec.lower()) if len(t) >= 3]
        matching_asserts = [a for a in assertions if a.predicate.lower() == req_sec.lower()]
        covered = False
        if graph is not None and v4_review is not None:
            if matching_asserts:
                for a in matching_asserts:
                    sdoc = v4_sources.get(a.source_id)
                    a_span: tuple[int, int] | None = None
                    if sdoc is not None and a.quote:
                        pos = sdoc.text.find(a.quote)
                        if pos >= 0:
                            a_span = (pos, pos + len(a.quote))
                    for node in verified_leaves:
                        span_match = False
                        for ref in getattr(node, "evidence", ()) or ():
                            if str(getattr(ref, "document_id", "")) != a.source_id:
                                continue
                            if a_span is None:
                                span_match = True
                                break
                            if int(ref.start) < a_span[1] and a_span[0] < int(ref.end):
                                span_match = True
                                break
                        if span_match and _leaf_linked_to_clause(node, tokens):
                            covered = True
                            break
                    if covered:
                        break
            else:
                for node in verified_leaves:
                    hay = (
                        f"{getattr(node, 'predicate', '')} {getattr(node, 'subject', '')} "
                        + " ".join(str(getattr(e, "quote", "")) for e in (getattr(node, "evidence", ()) or ()))
                    ).lower()
                    if (not tokens or any(tok in hay for tok in tokens[:2])) and _leaf_linked_to_clause(node, tokens):
                        covered = True
                        break

        if not covered:
            reason_code = (
                "ASSERTION_NOT_IN_DELIVERABLE"
                if matching_asserts
                else "SECTION_MISSING_FROM_DELIVERABLE"
            )
            issues.append({
                "code": "MISSING_REQUIRED_SECTION",
                "source_layer": "v3",
                "reason": reason_code,
                "section_key": req_sec,
                "repairable": True,
                "message": (
                    f"Mandatory section '{req_sec}' ({reason_code}) lacks a verified graph leaf and "
                    f"linked deliverable clause citing {gov_filename}."
                ),
                "remediation": (
                    f"Include '{req_sec}' as both a verified graph leaf overlapping the cited source span "
                    f"and a corresponding clause in the final deliverable text citing {gov_filename}."
                ),
                "forbidden_shortcuts": [
                    f"Do not leave '{req_sec}' only in structured assertions while omitting it from deliverable or graph."
                ],
            })
    return issues


def evaluate_securitization_candidate(
    candidate: str | Mapping[str, Any],
    corpus: LoadedCaseCorpus,
    *,
    extractor: SecuritizationReportExtractor | None = None,
    gate_mode: str = "v3_v4",  # "v3_only" | "v3_v4"
    required_sections: Sequence[str] | None = None,
    policy: Policy | None = None,
    cache: VerificationCache | None = None,
    budget: VerificationBudget | None = None,
    domain_profile: DomainProfile | None = None,
    operation_id: str = "release_investor_report",
    require_graph: bool = True,
    require_assertions: bool = False,
) -> PipelineVerificationResult:
    """Run v3-only or combined v3+v4 verification on a candidate report, strictly separating report quality from op auth."""
    t0 = time.perf_counter()
    adapter = get_domain_adapter(corpus, domain_profile=domain_profile)
    ext = extractor or SecuritizationReportExtractor()
    parsed = ext.extract(
        candidate,
        corpus,
        required_sections=required_sections,
        vocabulary=adapter.vocabulary,
    )

    ledger_state_digest = hashlib.sha256(
        corpus.common_evidence_sha256.encode("utf-8")
    ).hexdigest()
    resp_digest = compute_response_digest(parsed.report_text, parsed.question)

    if parsed.extraction_error:
        verifier_ms = (time.perf_counter() - t0) * 1000.0
        iss = [{
            "code": "EXTRACTION_FAILED",
            "source_layer": "extractor",
            "repairable": True,
            "message": parsed.extraction_error,
            "remediation": "Return a valid report containing structured assertions/sections matching schema.",
            "forbidden_shortcuts": [],
        }]
        return PipelineVerificationResult(
            report_disposition="RETRY",
            accepted=False,
            operational_authorization={
                "status": "ESCALATE",
                "authorized": False,
                "reason": "EXTRACTION_FAILED_BLOCKS_EXECUTION",
                "response_digest": resp_digest,
                "assessment_digest": None,
                "ledger_state_digest": ledger_state_digest,
            },
            issues=iss,
            consolidated_feedback={"disposition": "RETRY", "issues": iss},
            v3_verdict_status=None,
            v4_review_disposition=None,
            response_digest=resp_digest,
            assessment_digest=None,
            ledger_state_digest=ledger_state_digest,
            verifier_ms=verifier_ms,
            cache_metrics={},
        )

    issues: list[dict[str, Any]] = []
    if require_assertions and not parsed.assertions:
        issues.append({
            "code": "MISSING_STRUCTURED_ASSERTIONS",
            "source_layer": "extractor",
            "repairable": True,
            "message": "No source-backed assertions were extracted for verification.",
            "remediation": "Provide structured assertions with exact source quotations for material report claims.",
            "forbidden_shortcuts": ["Do not remove material claims to avoid verification."],
        })
    gov_doc = adapter.resolve_governing_doc_fn(corpus, None)

    # --- 1. v3 Controls (Ledger + Policy Coverage + Authority Rank + Typed Units) ---
    eff_policy = policy
    if eff_policy is None:
        eff_policy = Policy(
            policy_id=f"harness2_{corpus.case_id}_{gate_mode}",
            min_tier_rank=0 if adapter.domain_id == "banking_finance" else gov_doc.authority_rank,
            required_predicates=list(parsed.required_section_keys),
            required_source_ids=[gov_doc.filename] if parsed.required_section_keys else [],
        )
    v3_verdict = None
    if parsed.assertions:
        gov_assertions = [
            a for a in parsed.assertions
            if not any(
                tag in f"{a.predicate} {a.subject}".lower()
                for tag in ("discrepancy", "om_", "subordinate", "email_", "erroneous", "counterpart")
            )
        ]
        sub_assertions = [a for a in parsed.assertions if a not in gov_assertions]
        if gov_assertions:
            v3_verdict = verify(
                assertions=gov_assertions,
                ledger=corpus.ledger,
                policy=eff_policy,
                agent_id=f"harness2-{gate_mode}",
            )
        elif sub_assertions:
            v3_verdict = verify(
                assertions=sub_assertions,
                ledger=corpus.ledger,
                policy=eff_policy,
                agent_id=f"harness2-{gate_mode}",
            )
        if gov_assertions and sub_assertions:
            sub_policy = Policy(
                policy_id=f"{eff_policy.policy_id}_subordinate",
                min_tier_rank=0,
            )
            sub_sources = [
                Source(
                    id=d.filename,
                    text=d.text,
                    tier=d.authority_tier,
                    authority_rank=d.authority_rank,
                    title=f"{d.doc_key}: {d.filename}",
                    metadata={"doc_key": d.doc_key, "role": d.role},
                )
                for d in corpus.documents.values()
                if (
                    (d.role in adapter.subordinate_roles)
                    if adapter.domain_id == "banking_finance"
                    else (d.authority_rank < gov_doc.authority_rank)
                )
            ]
            sub_ledger = Ledger(sources=sub_sources) if sub_sources else corpus.ledger
            sub_verdict = verify(
                assertions=sub_assertions,
                ledger=sub_ledger,
                policy=sub_policy,
                agent_id=f"harness2-{gate_mode}-sub",
            )
            for v3_iss in sub_verdict.issues:
                issues.append({
                    "code": v3_iss.code,
                    "source_layer": "v3",
                    "repairable": v3_iss.disposition == "RETRY",
                    "message": v3_iss.message,
                    "remediation": v3_iss.remediation or "Ground subordinate discrepancy quote in exact source text.",
                    "forbidden_shortcuts": ["Do not delete discrepancy evidence to avoid verification."],
                })
        if v3_verdict is not None:
            for v3_iss in v3_verdict.issues:
                # Skip redundant v3 missing predicate errors if v3 policy intentionally empty; keep real failures
                issues.append({
                    "code": v3_iss.code,
                    "source_layer": "v3",
                    "repairable": v3_iss.disposition == "RETRY",
                    "message": v3_iss.message,
                    "remediation": v3_iss.remediation or "Ground every claim in governing document quote.",
                    "forbidden_shortcuts": ["Do not delete sections to avoid verification."],
                })
    elif eff_policy.required_predicates:
        for req_p in eff_policy.required_predicates:
            issues.append({
                "code": "MISSING_REQUIRED_SECTION",
                "source_layer": "v3",
                "repairable": True,
                "message": f"Required comparison section '{req_p}' was omitted from assertions/report.",
                "remediation": f"Include verified analysis of section '{req_p}' citing {gov_doc.filename}.",
                "forbidden_shortcuts": [f"Do not omit '{req_p}'."],
            })

    # Always run domain-specific authority rank & fidelity checks over assertions/text
    issues.extend(adapter.authority_check_fn(parsed, corpus))

    # Required section text completeness check (v3_only mode uses assertion/prose match;
    # v3_v4 mode enforces unified graph-leaf + correspondence deliverable coverage after v4_review)
    if gate_mode != "v3_v4":
        low_text = parsed.report_text.lower()
        for req_sec in parsed.required_section_keys:
            tokens = [t for t in re.split(r"[^a-z0-9]+", req_sec.lower()) if len(t) >= 3]
            # Require at least one assertion predicate match OR core section keyword coverage in report text
            has_assert = any(a.predicate.lower() == req_sec.lower() for a in parsed.assertions)
            has_prose = all(tok in low_text for tok in tokens[:2]) if tokens else False
            if not (has_assert or has_prose):
                issues.append({
                    "code": "MISSING_REQUIRED_SECTION",
                    "source_layer": "v3",
                    "repairable": True,
                    "message": f"Mandatory section '{req_sec}' is missing from candidate report.",
                    "remediation": f"Add complete discrepancy comparison for '{req_sec}' citing governing {gov_doc.filename}.",
                    "forbidden_shortcuts": [f"Do not drop previously verified section '{req_sec}'."],
                })

    v4_review = None
    assessment_digest = None
    cache_metrics: dict[str, int] = {}
    corpus_register: CorpusReferenceRegister | None = None
    prof: DomainProfile | None = None
    if gate_mode == "v3_v4":
        # Compute corpus reference register & origin="corpus" obligations independently of agent output.
        layer = corpus.evidence_layer
        if (
            getattr(layer, "reader", None) is not None
            and not layer.units
            and not getattr(layer, "document_read_statuses", {})
        ):
            # Populate EvidenceLayer once through the configured reader (errors stay in read_statuses; never swallowed).
            layer.ingest(list(corpus.v4_sources.values()))

        prof = domain_profile or DomainProfile(
            profile_id=f"{adapter.domain_id}_harness2",
            vocabulary=adapter.vocabulary,
            policy=PolicyConfiguration(
                tau=adapter.acceptance_policy.tau,
                delta=adapter.acceptance_policy.delta,
                authority_ranks={d.filename: d.authority_rank for d in corpus.documents.values()},
                excluded_comparison_roles=dict(adapter.excluded_comparison_roles),
            ),
        )
        div_collection = collect_reference_divergences(corpus.documents, prof)
        corpus_register = build_corpus_reference_register(
            layer,
            adapter.vocabulary,
            question=parsed.question,
            divergence_collection=div_collection,
            expected_documents=corpus.v4_sources,
        )
        if not corpus_register.available:
            issues.append({
                "code": "REFERENCE_REGISTER_MISSING",
                "source_layer": "v4",
                "repairable": False,
                "message": corpus_register.reason or "No corpus reference register is available.",
                "remediation": "Provide a corpus reader or ingest units via EvidenceLayer.ingest_units before evaluating.",
                "forbidden_shortcuts": [],
            })

        if require_graph and parsed.graph is None:
            issues.append({"code": "MISSING_JUSTIFICATION_GRAPH", "source_layer": "v4",
                           "repairable": True, "message": "No model-proposed graph is available for response verification.",
                           "remediation": "Supply a source-grounded justification graph; prose checks alone cannot certify the report.",
                           "forbidden_shortcuts": []})
        # --- 2. v4 Waterfall Relational Verification ---
        if parsed.governing_waterfall:
            wf_res: WaterfallComparisonResult = verify_waterfall_steps(
                parsed.governing_waterfall,
                parsed.claimed_waterfall or None,
                claimed_relations=parsed.claimed_waterfall_relations or None,
                allow_cross_regime_if_qualified=parsed.allow_cross_regime_waterfall,
            )
            if wf_res.status != "verified":
                detail_msgs: list[str] = []
                for bucket in (
                    wf_res.inversions,
                    wf_res.beneficiary_swaps,
                    wf_res.payment_type_confusions,
                    wf_res.cycles,
                    wf_res.regime_mismatches,
                ):
                    for item in bucket:
                        detail_msgs.append(json.dumps(item, default=str))
                for idx, rcode in enumerate(wf_res.reasons):
                    issues.append({
                        "code": rcode,
                        "source_layer": "v4",
                        "repairable": True,
                        "message": detail_msgs[idx] if idx < len(detail_msgs) else f"Waterfall verification failed: {rcode}",
                        "remediation": (
                            "Preserve exact relative priority order, beneficiaries, payment types (interest vs principal), "
                            "and execution regime (pre- vs post-acceleration) from governing Indenture."
                        ),
                        "forbidden_shortcuts": ["Do not delete waterfall ordering details to bypass verification."],
                    })

        # --- 3. v4 Quantitative, Unit, Domain-Binding, and Correspondence Checks ---
        issues.extend(adapter.v4_prose_check_fn(parsed, corpus))

        if parsed.graph is not None:
            assert prof is not None
            v4_review = review_response(
                parsed.question,
                parsed.report_text,
                corpus.v4_sources,
                parsed.graph,
                prof,
                cache=cache,
                budget=budget,
                operation_policy=prof,
                authorize_operation_id=operation_id,
                corpus_register=corpus_register,
                require_reference_register=True,
            )
            assessment_digest = v4_review.assessment_digest
            cache_metrics = dict(v4_review.assessment.cache_metrics)
            for iss in v4_review.issues:
                issues.append({
                    "code": iss.code,
                    "source_layer": "v4",
                    "repairable": bool(v4_review.required_repairs),
                    "message": iss.message,
                    "remediation": "; ".join(r.instruction for r in v4_review.required_repairs) or iss.message,
                    "forbidden_shortcuts": [
                        fs
                        for r in v4_review.required_repairs
                        for fs in r.forbidden_shortcuts
                    ],
                })
            corr_rep = v4_review.assessment.correspondence_report or {}
            if not v4_review.assessment.response_correspondence_verified:
                for cm in corr_rep.get("issues") or corr_rep.get("mismatches") or []:
                    code = str(cm.get("code") or "RESPONSE_GRAPH_MISMATCH") if isinstance(cm, Mapping) else str(cm)
                    if not any(existing["code"] == code for existing in issues):
                        issues.append({
                            "code": code,
                            "source_layer": "v4",
                            "repairable": True,
                            "message": str(cm.get("message") if isinstance(cm, Mapping) else cm),
                            "remediation": "Align candidate report prose strictly with verified subjects, values, units, and conditions.",
                            "forbidden_shortcuts": ["Do not leave contradictory figures or false equations in final text."],
                        })

        req_keys = list(dict.fromkeys([*parsed.required_section_keys, *(eff_policy.required_predicates or ())]))
        if req_keys:
            issues.extend(
                _check_required_section_deliverable_coverage(
                    required_keys=req_keys,
                    assertions=parsed.assertions,
                    v4_sources=corpus.v4_sources,
                    graph=parsed.graph,
                    v4_review=v4_review,
                    gov_filename=gov_doc.filename,
                )
            )

    # Deduplicate issues preserving order
    dedup_issues: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for iss in issues:
        key = (iss["code"], iss["message"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            dedup_issues.append(iss)

    if dedup_issues and all(i["code"] == "PENDING_BUDGET" for i in dedup_issues):
        report_disp = "ABSTAIN"
    elif v4_review is not None and v4_review.disposition == "ABSTAIN" and not any(
        i["code"] != "PENDING_BUDGET" for i in dedup_issues
    ):
        report_disp = "ABSTAIN"
    elif dedup_issues:
        non_budget_issues = [i for i in dedup_issues if i["code"] != "PENDING_BUDGET"]
        if non_budget_issues and all(i.get("repairable", True) for i in non_budget_issues):
            report_disp = "RETRY"
        elif not non_budget_issues:
            report_disp = "ABSTAIN"
        else:
            report_disp = "ESCALATE"
    elif v4_review is not None and v4_review.disposition != "ACCEPT":
        report_disp = v4_review.disposition
    else:
        report_disp = "ACCEPT"

    accepted = report_disp == "ACCEPT"

    cov_info = (
        dict(v4_review.assessment.corpus_coverage)
        if v4_review is not None and getattr(v4_review.assessment, "corpus_coverage", None)
        else {}
    )
    ob_ids = tuple(
        cov_info.get("obligation_ids")
        or (corpus_register.obligation_ids if corpus_register is not None else ())
    )
    cov_ids = tuple(cov_info.get("covered_ids") or ())
    uncov_ids = tuple(cov_info.get("uncovered_ids") or ())
    rej_ids = tuple(cov_info.get("rejected_ids") or ())
    total_obs = int(cov_info.get("total_obligations", len(ob_ids)))
    covered_obs = int(cov_info.get("covered_count", len(cov_ids)))
    rejected_obs = int(cov_info.get("rejected_count", len(rej_ids)))

    # Separate operational authorization: derived from corpus register (ignores agent metadata.has_material_discrepancies)
    has_material_discrepancies = bool(
        corpus_register is not None
        and (corpus_register.has_material_conflicts or corpus_register.has_authority_overrides)
    )
    excluded_docs = list(corpus_register.excluded_documents) if corpus_register is not None else []
    profile_fp = (
        corpus_register.domain_profile_fingerprint
        if corpus_register is not None and corpus_register.domain_profile_fingerprint
        else (prof.fingerprint() if prof is not None else None)
    )
    receipt_payload: dict[str, Any] = (
        v3_verdict.receipt.to_dict()
        if v3_verdict is not None and getattr(v3_verdict, "receipt", None) is not None
        else {
            "scheme": "claimlayer-v4-corpus-receipt-v1",
            "response_digest": resp_digest,
            "assessment_digest": assessment_digest,
            "ledger_digest": ledger_state_digest,
        }
    )
    receipt_payload["corpus_obligations_total"] = total_obs
    receipt_payload["corpus_obligations_covered"] = covered_obs
    receipt_payload["corpus_obligations_rejected"] = rejected_obs
    receipt_payload["corpus_obligation_ids"] = list(ob_ids)
    receipt_payload["excluded_documents"] = excluded_docs
    receipt_payload["domain_profile_fingerprint"] = profile_fp
    receipt_payload["corpus_coverage_digest"] = hashlib.sha256(
        json.dumps(
            {
                "report_disposition": report_disp,
                "response_digest": resp_digest,
                "assessment_digest": assessment_digest,
                "ledger_state_digest": ledger_state_digest,
                "corpus_obligations_total": total_obs,
                "corpus_obligations_covered": covered_obs,
                "corpus_obligations_rejected": rejected_obs,
                "corpus_obligation_ids": list(ob_ids),
                "excluded_documents": excluded_docs,
                "domain_profile_fingerprint": profile_fp,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if "signature" not in receipt_payload or not receipt_payload["signature"]:
        receipt_payload["signature"] = receipt_payload["corpus_coverage_digest"]

    if not accepted:
        op_auth = {
            "status": "RETRY" if report_disp == "RETRY" else "ESCALATE",
            "authorized": False,
            "reason": "REPORT_NOT_ACCEPTED",
            "response_digest": resp_digest,
            "assessment_digest": assessment_digest,
            "ledger_state_digest": ledger_state_digest,
            "corpus_obligations_total": total_obs,
            "corpus_obligations_covered": covered_obs,
            "corpus_obligations_rejected": rejected_obs,
            "corpus_obligation_ids": list(ob_ids),
            "excluded_documents": excluded_docs,
            "domain_profile_fingerprint": profile_fp,
            "receipt": receipt_payload,
        }
    elif has_material_discrepancies:
        op_auth = {
            "status": "ESCALATE",
            "authorized": False,
            "reason": "MATERIAL_DISCREPANCIES_REQUIRE_HUMAN_SIGNOFF",
            "response_digest": resp_digest,
            "assessment_digest": assessment_digest,
            "ledger_state_digest": ledger_state_digest,
            "corpus_obligations_total": total_obs,
            "corpus_obligations_covered": covered_obs,
            "corpus_obligations_rejected": rejected_obs,
            "corpus_obligation_ids": list(ob_ids),
            "excluded_documents": excluded_docs,
            "domain_profile_fingerprint": profile_fp,
            "receipt": receipt_payload,
        }
    else:
        op_auth = {
            "status": "PASS",
            "authorized": True,
            "reason": "VERIFIED_AND_CONSISTENT",
            "response_digest": resp_digest,
            "assessment_digest": assessment_digest,
            "ledger_state_digest": ledger_state_digest,
            "corpus_obligations_total": total_obs,
            "corpus_obligations_covered": covered_obs,
            "corpus_obligations_rejected": rejected_obs,
            "corpus_obligation_ids": list(ob_ids),
            "excluded_documents": excluded_docs,
            "domain_profile_fingerprint": profile_fp,
            "receipt": receipt_payload,
        }

    consolidated_feedback = {
        "disposition": report_disp,
        "issue_count": len(dedup_issues),
        "governing_source": gov_doc.filename,
        "issues": dedup_issues,
        "anti_evasion_rule": "Do NOT delete existing valid sections or quantitative figures to silence an error; fix every cited defect in place.",
        "corpus_obligations_total": total_obs,
        "corpus_obligations_covered": covered_obs,
        "corpus_obligations_rejected": rejected_obs,
        "corpus_obligation_ids": list(ob_ids),
    }
    verifier_ms = (time.perf_counter() - t0) * 1000.0

    return PipelineVerificationResult(
        report_disposition=report_disp,
        accepted=accepted,
        operational_authorization=op_auth,
        issues=dedup_issues,
        consolidated_feedback=consolidated_feedback,
        v3_verdict_status=v3_verdict.status if v3_verdict is not None else None,
        v4_review_disposition=v4_review.disposition if v4_review is not None else None,
        response_digest=resp_digest,
        assessment_digest=assessment_digest,
        ledger_state_digest=ledger_state_digest,
        verifier_ms=verifier_ms,
        cache_metrics=cache_metrics,
        corpus_obligations_total=total_obs,
        corpus_obligations_covered=covered_obs,
        corpus_obligations_rejected=rejected_obs,
        corpus_obligation_ids=ob_ids,
        corpus_covered_ids=cov_ids,
        corpus_uncovered_ids=uncov_ids,
        corpus_rejected_ids=rej_ids,
        signed_receipt=receipt_payload,
    )


def build_structured_repair_feedback(
    verdict: PipelineVerificationResult,
    candidate: Mapping[str, Any] | None = None,
    *,
    max_chars: int = 4500,
) -> str:
    """Build bounded, valid-JSON repair feedback with failing assertion/claim diffs."""
    compact_issues: list[dict[str, Any]] = []
    issue_blobs: list[str] = []
    for iss in verdict.issues:
        msg = str(iss.get("message") or "")[:320]
        rem = str(iss.get("remediation") or "")[:240]
        issue_blobs.append(msg.lower())
        compact_issues.append({
            "code": iss.get("code"),
            "source_layer": iss.get("source_layer"),
            "message": msg,
            "remediation": rem,
            "forbidden_shortcuts": list(iss.get("forbidden_shortcuts") or [])[:2],
        })

    combined_blob = " ".join(issue_blobs)
    failing_assertions: list[dict[str, Any]] = []
    failing_claims: list[dict[str, Any]] = []
    if isinstance(candidate, Mapping):
        for idx, a in enumerate(candidate.get("assertions") or []):
            if not isinstance(a, Mapping):
                continue
            pred = str(a.get("predicate") or "")
            val = str(a.get("value") or "")
            src = str(a.get("source_id") or "")
            if (
                (pred and pred.lower() in combined_blob)
                or (val and len(val) >= 2 and val.lower() in combined_blob)
                or (src and src.lower() in combined_blob)
            ):
                failing_assertions.append({
                    "index": idx,
                    "subject": a.get("subject"),
                    "predicate": pred,
                    "value": val,
                    "unit": a.get("unit"),
                    "role": a.get("role"),
                    "source_id": src,
                })
        graph = candidate.get("graph")
        if isinstance(graph, Mapping):
            claims = graph.get("claims") or {}
            claim_iter = claims.values() if isinstance(claims, Mapping) else claims
            for c in claim_iter:
                if not isinstance(c, Mapping):
                    continue
                cid = str(c.get("claim_id") or "")
                pred = str(c.get("predicate") or "")
                if (cid and cid.lower() in combined_blob) or (pred and pred.lower() in combined_blob):
                    failing_claims.append({
                        "claim_id": cid,
                        "kind": c.get("kind"),
                        "subject": c.get("subject"),
                        "predicate": pred,
                        "value": c.get("value"),
                        "unit": c.get("unit"),
                    })

    payload: dict[str, Any] = {
        "disposition": verdict.report_disposition,
        "issue_count": len(verdict.issues),
        "issues": compact_issues,
        "failing_assertions": failing_assertions[:8],
        "failing_claims": failing_claims[:8],
        "anti_evasion_rule": verdict.consolidated_feedback.get("anti_evasion_rule"),
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    while len(encoded) > max_chars and len(payload["issues"]) > 1:
        payload["issues"].pop()
        payload["truncated_issues"] = len(verdict.issues) - len(payload["issues"])
        encoded = json.dumps(payload, ensure_ascii=False)
    return encoded


def run_securitization_repair_loop(
    *,
    corpus: LoadedCaseCorpus,
    initial_candidate: str | Mapping[str, Any],
    repair_fn: Callable[[Any, PipelineVerificationResult, int], Any] | None,
    gate_mode: str = "v3_v4",
    required_sections: Sequence[str] | None = None,
    max_iterations: int = 3,
    max_wall_clock_ms: float = 180000.0,
    max_total_tokens: int | None = None,
    usage_tracker: LLMCallUsage | None = None,
    cache: VerificationCache | None = None,
) -> dict[str, Any]:
    """Bounded evaluate -> feedback -> repair -> reverify loop. Never accepts on budget exhaustion."""
    t_start = time.perf_counter()
    eff_cache = cache if cache is not None else VerificationCache()
    ver_budget = VerificationBudget(max_latency_ms=max_wall_clock_ms, max_semantic_calls=10)

    current_candidate: Any = initial_candidate
    attempts: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    seen_covered_corpus_ids: set[str] = set()
    prev_issue_sig: tuple[tuple[str, str], ...] | None = None
    stop_reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        res = evaluate_securitization_candidate(
            current_candidate,
            corpus,
            gate_mode=gate_mode,
            required_sections=required_sections,
            cache=eff_cache,
            budget=ver_budget,
        )
        curr_covered_corpus = set(res.corpus_covered_ids or ())
        dropped_corpus = sorted(seen_covered_corpus_ids - curr_covered_corpus)
        seen_covered_corpus_ids |= curr_covered_corpus

        effective_issues = list(res.issues)
        effective_disp = res.report_disposition
        effective_accepted = res.accepted
        if dropped_corpus:
            effective_issues.append({
                "code": "CORPUS_OBLIGATION_DROPPED",
                "source_layer": "v4",
                "repairable": iteration < max_iterations,
                "dropped_obligation_ids": dropped_corpus,
                "message": (
                    f"Corpus obligation(s) {dropped_corpus} covered in a prior attempt "
                    f"were dropped in attempt {iteration}."
                ),
                "remediation": "Restore all previously covered corpus obligations without pruning.",
                "forbidden_shortcuts": ["Do not drop or rename previously covered corpus obligations."],
            })
            effective_accepted = False
            effective_disp = "RETRY" if iteration < max_iterations else "ESCALATE"

        attempts.append({
            "iteration": iteration,
            "disposition": effective_disp,
            "accepted": effective_accepted,
            "issue_codes": [i["code"] for i in effective_issues],
            "issues": effective_issues,
            "operational_authorization": res.operational_authorization,
            "response_digest": res.response_digest,
            "verifier_ms": round(res.verifier_ms, 3),
            "cache_metrics": res.cache_metrics,
            "candidate": current_candidate,
            "corpus_covered_ids": list(res.corpus_covered_ids),
            "corpus_uncovered_ids": list(res.corpus_uncovered_ids),
        })
        if res.response_digest:
            seen_digests.add(res.response_digest)

        if effective_accepted:
            stop_reason = "accepted"
            break
        if effective_disp in ("ESCALATE", "ABSTAIN"):
            stop_reason = f"non_repairable_{effective_disp.lower()}"
            break

        curr_sig = tuple(sorted((i["code"], i["message"]) for i in effective_issues))
        if iteration >= 2 and prev_issue_sig is not None and curr_sig == prev_issue_sig:
            stop_reason = "stagnant_issues"
            break
        prev_issue_sig = curr_sig

        if iteration >= max_iterations:
            stop_reason = "max_iterations_reached"
            break
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        if elapsed_ms > max_wall_clock_ms:
            stop_reason = "latency_budget_exceeded"
            break
        if max_total_tokens is not None and usage_tracker is not None and usage_tracker.total_tokens >= max_total_tokens:
            stop_reason = "token_budget_exceeded"
            break
        if repair_fn is None:
            stop_reason = "no_repair_agent"
            break

        next_candidate = repair_fn(current_candidate, res, iteration)
        next_text = next_candidate.get("report_text", "") if isinstance(next_candidate, Mapping) else str(next_candidate)
        next_digest = compute_response_digest(next_text, None)
        if next_digest and next_digest in seen_digests:
            stop_reason = "duplicate_response"
            break
        current_candidate = next_candidate

    last = attempts[-1]
    cache_hits = 0
    cache_misses = 0
    if attempts and "cache_metrics" in attempts[-1]:
        cache_hits = sum(int(a["cache_metrics"].get("hits", 0)) for a in attempts)
        cache_misses = sum(int(a["cache_metrics"].get("misses", 0)) for a in attempts)
    if stop_reason == "accepted":
        final_disposition = "ACCEPT"
    elif last["disposition"] == "ABSTAIN" or "budget_exceeded" in stop_reason:
        final_disposition = "ABSTAIN"
    else:
        final_disposition = "ESCALATE"

    final_candidate = last["candidate"]
    final_text = final_candidate.get("report_text", "") if isinstance(final_candidate, Mapping) else str(final_candidate)
    init_text = initial_candidate.get("report_text", "") if isinstance(initial_candidate, Mapping) else str(initial_candidate)
    total_verifier_ms = sum(a["verifier_ms"] for a in attempts)
    return {
        "gate_mode": gate_mode,
        "final_disposition": final_disposition,
        "accepted": stop_reason == "accepted",
        "stop_reason": stop_reason,
        "iterations_used": len(attempts),
        "initial_report": init_text,
        "final_report": final_text,
        "initial_candidate": initial_candidate,
        "final_candidate": final_candidate,
        "operational_authorization": last["operational_authorization"],
        "attempts": attempts,
        "verifier_ms": round(total_verifier_ms, 3),
        "cache_metrics": {"hits": cache_hits, "misses": cache_misses},
        "usage": usage_tracker.to_dict() if usage_tracker is not None else {},
    }


evaluate_domain_candidate = evaluate_securitization_candidate
