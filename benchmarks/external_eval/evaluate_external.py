"""Evaluate v4 reader + cross-check on external public SEC EDGAR / corporate credit corpora.

Reads a downloaded corpus directory verified against ``manifest.json``, a frozen
``protocol.json``, and blind human annotations stored OUTSIDE the repository
(``--annotations <outside-repo-path>``). Computes:
- reader conflict recall (+ 95% Wilson CI)
- leave-one-out ablation detection rate (+ 95% Wilson CI)
- clean-control false-gap rate (+ 95% Wilson CI)
- clean-candidate escalation rate (+ 95% Wilson CI)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_EXT_DIR = Path(__file__).resolve().parent
_BENCH_DIR = _EXT_DIR.parent
_ROOT = _BENCH_DIR.parent
_SRC_DIR = _ROOT / "src"
for _p in (_SRC_DIR, _BENCH_DIR, _EXT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claimlayer.semantic.adapters import OpenAICompatibleModel  # noqa: E402
from claimlayer.semantic.models import SourceDocument  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    EvidenceLayer,
    LLMProposalReader,
    REREAD_PROMPT,
    REREAD_PROMPT_VERSION,
    RecordedProposalReader,
    SECURITIZATION_VOCABULARY,
    STRICT_PROMPT_VERSION,
    STRICT_READER_PROMPT,
    TargetedRereadConfig,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domains import (  # noqa: E402
    BANKING_FINANCE_ACCEPTANCE_POLICY,
    BANKING_FINANCE_DOMAIN_PROFILE,
    BANKING_FINANCE_VOCABULARY,
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_DOMAIN_PROFILE,
)
from edgar_loader import verify_manifest_integrity  # noqa: E402
from measure_reader_recall import (  # noqa: E402
    _RecordingRereadReader,
    _ablate_conflict_leave_one_out,
    _conflict_found_in_register,
    _count_sentence_checks,
    _snapshot_reader_documents,
    wilson_ci,
)


@dataclass(frozen=True)
class _ExtDoc:
    document_id: str
    filename: str
    role: str
    authority_tier: str
    authority_rank: int
    governs_on_conflict: bool
    text: str
    title: str = ""


class _ExtCorpusWrapper:
    """Minimal adapter exposing .v4_sources and .documents for shared measurement helpers."""

    def __init__(self, sources: dict[str, SourceDocument], loaded: dict[str, _ExtDoc]) -> None:
        self.v4_sources = sources
        self.documents = loaded


def assert_annotations_outside_repo(annotations_path: Path, repo_root: Path = _ROOT) -> Path:
    """Fail closed if an external annotation file sits inside the repository."""
    resolved_ann = Path(annotations_path).resolve()
    resolved_repo = Path(repo_root).resolve()
    try:
        resolved_ann.relative_to(resolved_repo)
    except ValueError:
        return resolved_ann
    raise ValueError(
        f"External annotations must live OUTSIDE the repository ({resolved_repo}), "
        f"got {resolved_ann}"
    )


def _domain_assets(domain: str) -> tuple[Any, Any, Any]:
    if domain == "securitization":
        return (
            SECURITIZATION_VOCABULARY,
            SECURITIZATION_ACCEPTANCE_POLICY,
            SECURITIZATION_DOMAIN_PROFILE,
        )
    if domain == "banking_finance":
        return (
            BANKING_FINANCE_VOCABULARY,
            BANKING_FINANCE_ACCEPTANCE_POLICY,
            BANKING_FINANCE_DOMAIN_PROFILE,
        )
    raise ValueError(f"Unsupported domain {domain!r}")


def evaluate_external_corpus(
    *,
    corpus_dir: Path,
    annotations_path: Path,
    protocol_path: Path = _EXT_DIR / "protocol.json",
    mode: str = "recorded",
    recordings_dir: Path | None = None,
    save_recordings_dir: Path | None = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Run external recall, leave-one-out ablation, and escalation evaluation."""
    ann_resolved = assert_annotations_outside_repo(annotations_path)
    manifest = verify_manifest_integrity(corpus_dir)
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    annotations = json.loads(ann_resolved.read_text(encoding="utf-8"))

    corpora_ann = annotations.get("corpora") or {}
    rr_cfg_raw = ((protocol.get("reader_configuration") or {}).get("reread_config") or {})
    reread_cfg = TargetedRereadConfig(
        enabled=bool(rr_cfg_raw.get("enabled", True)),
        max_attempts_per_document=int(rr_cfg_raw.get("max_attempts_per_document", 3)),
        max_total_attempts=int(rr_cfg_raw.get("max_total_attempts", 12)),
    )

    live_reader: LLMProposalReader | None = None
    if mode == "live":
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY") or ""
        if not resolved_key:
            raise RuntimeError("Missing API key for --mode live.")
        model = OpenAICompatibleModel(
            base_url=base_url,
            model=model_name,
            api_key=resolved_key,
            timeout=timeout,
            max_tokens=4096,
            retries=1,
            response_format={"type": "json_object"},
        )
        live_reader = LLMProposalReader(
            model=model,
            prompt=STRICT_READER_PROMPT,
            prompt_version=STRICT_PROMPT_VERSION,
            reread_prompt=REREAD_PROMPT,
            reread_prompt_version=REREAD_PROMPT_VERSION,
        )

    per_corpus: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    tot_docs = 0
    tot_read_ok = 0
    tot_annotated = 0
    tot_found = 0
    tot_sentence_checks = 0
    tot_gaps = 0
    clean_corpora = 0
    clean_escalated = 0
    clean_sentence_checks = 0
    clean_sents_with_gap = 0
    clean_false_gaps = 0
    reader_meta: dict[str, Any] = {}

    for corp_entry in manifest.get("corpora") or ():
        cid = str(corp_entry["corpus_id"])
        domain = str(corp_entry["domain"])
        vocab, pol, dprof = _domain_assets(domain)

        v4_sources: dict[str, SourceDocument] = {}
        loaded_docs: dict[str, _ExtDoc] = {}
        for d in corp_entry.get("documents") or ():
            did = str(d["document_id"])
            text = (Path(corpus_dir) / str(d["relative_path"])).read_text(encoding="utf-8")
            rank = int(d["authority_rank"])
            role = str(d["role"])
            tier = str(d["authority_tier"])
            v4_sources[did] = SourceDocument(
                document_id=did,
                text=text,
                attribution="controlling_instrument" if rank >= 90 else "source_statement",
                authority_rank=rank,
            )
            loaded_docs[did] = _ExtDoc(
                document_id=did,
                filename=did,
                role=role,
                authority_tier=tier,
                authority_rank=rank,
                governs_on_conflict=rank >= 90,
                text=text,
            )

        corpus = _ExtCorpusWrapper(v4_sources, loaded_docs)
        recorder_proxy: _RecordingRereadReader | None = None
        if mode == "live":
            assert live_reader is not None
            recorder_proxy = _RecordingRereadReader(live_reader)
            reader: Any = recorder_proxy
            reader_meta = dict(live_reader.identity)
        else:
            if recordings_dir is None:
                raise ValueError("--recordings-dir is required in --mode recorded")
            rec_path = Path(recordings_dir) / f"{cid}.json"
            reader = RecordedProposalReader.from_file(rec_path)
            reader_meta = dict(reader.identity)

        layer = EvidenceLayer(vocabulary=vocab, policy=pol, reader=reader)
        sdocs = list(v4_sources.values())
        layer.ingest(sdocs)
        div_coll = collect_reference_divergences(loaded_docs, dprof)
        register = build_corpus_reference_register(
            layer,
            vocab,
            divergence_collection=div_coll,
            expected_documents=v4_sources,
            reread_config=reread_cfg,
        )

        snapshot_docs = _snapshot_reader_documents(layer, reader, sdocs, model_name)
        if mode == "live" and save_recordings_dir is not None and recorder_proxy is not None:
            out_dir = Path(save_recordings_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "corpus_id": cid,
                "domain": domain,
                "model": model_name,
                "prompt_version": STRICT_PROMPT_VERSION,
                "prompt_digest": str(live_reader.identity.get("prompt_digest", "")),
                "reread_metadata": {
                    "reread_prompt_version": REREAD_PROMPT_VERSION,
                    "reread_prompt_digest": str(live_reader.identity.get("reread_prompt_digest", "")),
                },
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "documents": snapshot_docs,
                "rereads": recorder_proxy.recorded_rereads,
            }
            (out_dir / f"{cid}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        ann = corpora_ann.get(cid) or {}
        default_pri = str(ann.get("primary_source_id") or "")
        default_sec = str(ann.get("secondary_source_id") or "")
        conflicts = list(ann.get("conflicts") or ())
        found_ids: list[str] = []
        missed_ids: list[str] = []
        for conf in conflicts:
            if _conflict_found_in_register(conf, corpus, register, default_pri, default_sec):
                found_ids.append(str(conf["conflict_id"]))
                ab_row = _ablate_conflict_leave_one_out(
                    short_name=cid,
                    corpus=corpus,
                    vocab=vocab,
                    pol=pol,
                    div_coll=div_coll,
                    base_register=register,
                    base_docs=snapshot_docs,
                    base_reader_meta=reader_meta,
                    conflict=conf,
                    default_pri=default_pri,
                    default_sec=default_sec,
                )
                ablation_rows.append(ab_row)
            else:
                missed_ids.append(str(conf["conflict_id"]))

        divs = list(div_coll.divergences)
        sent_checks = _count_sentence_checks(corpus, divs)
        doc_statuses = layer.document_read_statuses
        read_ok_cnt = sum(1 for st in doc_statuses.values() if st.get("status") == "read_ok")
        gap_cnt = len(register.gap_divergences)
        sent_gap_keys = {
            (
                str(g.get("primary_source_id", "")),
                str(g.get("secondary_source_id", "")),
                str(g.get("section_key", "")),
                str(g.get("sentence") or ""),
            )
            for g in register.gap_divergences
        }

        is_clean = len(conflicts) > 0 and len(missed_ids) == 0
        if is_clean:
            clean_corpora += 1
            if gap_cnt > 0:
                clean_escalated += 1
            clean_sentence_checks += sent_checks
            clean_sents_with_gap += min(len(sent_gap_keys), max(sent_checks, 1))
            clean_false_gaps += gap_cnt

        tot_docs += len(sdocs)
        tot_read_ok += read_ok_cnt
        tot_annotated += len(conflicts)
        tot_found += len(found_ids)
        tot_sentence_checks += sent_checks
        tot_gaps += gap_cnt

        per_corpus.append({
            "corpus_id": cid,
            "domain": domain,
            "documents_total": len(sdocs),
            "documents_read_ok": read_ok_cnt,
            "annotated_conflicts": len(conflicts),
            "conflicts_found": len(found_ids),
            "missed_conflict_ids": missed_ids,
            "recall_wilson_95": wilson_ci(len(found_ids), len(conflicts)),
            "sentence_checks": sent_checks,
            "reference_register_gaps": gap_cnt,
            "clean_control": is_clean,
            "escalates_on_gap": gap_cnt > 0,
            "reread_provenance": dict(register.reread_provenance),
        })

    ab_detected = sum(1 for r in ablation_rows if r.get("gap_detected"))
    return {
        "protocol_id": protocol.get("protocol_id"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "mode": mode,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reader": reader_meta,
        "summary": {
            "corpora_total": len(per_corpus),
            "documents_total": tot_docs,
            "documents_read_ok": tot_read_ok,
            "annotated_conflicts": tot_annotated,
            "conflicts_found": tot_found,
            "reader_conflict_recall_wilson_95": wilson_ci(tot_found, tot_annotated),
            "sentence_checks_total": tot_sentence_checks,
            "reference_register_gaps_total": tot_gaps,
            "ablation": {
                "conflicts_removed": len(ablation_rows),
                "gaps_detected": ab_detected,
                "detection_rate_wilson_95": wilson_ci(ab_detected, len(ablation_rows)),
            },
            "clean_corpora_total": clean_corpora,
            "clean_candidate_escalation_rate_wilson_95": wilson_ci(clean_escalated, clean_corpora),
            "clean_corpus_false_gap_sentence_rate_wilson_95": wilson_ci(
                min(clean_sents_with_gap, clean_sentence_checks),
                clean_sentence_checks,
            ),
            "clean_false_gaps_total": clean_false_gaps,
        },
        "ablation": ablation_rows,
        "corpora": per_corpus,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate claimlayer v4 on external SEC EDGAR corpora.")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="External human annotations JSON (outside repo).")
    parser.add_argument("--protocol", type=Path, default=_EXT_DIR / "protocol.json")
    parser.add_argument("--mode", choices=("live", "recorded"), default="recorded")
    parser.add_argument("--recordings-dir", type=Path, default=None)
    parser.add_argument("--save-recordings-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = evaluate_external_corpus(
        corpus_dir=args.corpus_dir,
        annotations_path=args.annotations,
        protocol_path=args.protocol,
        mode=args.mode,
        recordings_dir=args.recordings_dir,
        save_recordings_dir=args.save_recordings_dir,
    )
    rendered = json.dumps(report, indent=2, sort_keys=False)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
