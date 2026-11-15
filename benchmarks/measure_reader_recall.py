#!/usr/bin/env python3
"""Measure reader conflict recall and REFERENCE_REGISTER_GAP rate against independent annotations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent
_SRC_DIR = _ROOT / "src"
_HARNESS_DIR = _ROOT / "examples" / "harness2_securitization"
_TESTS_DIR = _ROOT / "tests"
for _p in (_SRC_DIR, _HARNESS_DIR, _BENCH_DIR, _TESTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claimlayer.semantic.adapters import OpenAICompatibleModel  # noqa: E402
from claimlayer.semantic.models import digest  # noqa: E402
from claimlayer.v4 import (  # noqa: E402
    EvidenceLayer,
    LLMProposalReader,
    RecordedProposalReader,
    STRICT_READER_PROMPT,
    build_corpus_reference_register,
    collect_reference_divergences,
)
from claimlayer.v4.domains.banking_finance import (  # noqa: E402
    BANKING_FINANCE_ACCEPTANCE_POLICY,
    BANKING_FINANCE_DOMAIN_PROFILE,
    BANKING_FINANCE_VOCABULARY,
)
from claimlayer.v4.domains.securitization import (  # noqa: E402
    SECURITIZATION_ACCEPTANCE_POLICY,
    SECURITIZATION_DOMAIN_PROFILE,
    SECURITIZATION_VOCABULARY,
)
from claimlayer.v4.engine import CONTRACT  # noqa: E402
from harness2_v4_pipeline import build_synthetic_securitization_corpus, get_domain_adapter  # noqa: E402
from heldout_multidomain_eval import (  # noqa: E402
    _build_apex_banking_corpus,
    _build_meridian_securitization_corpus,
)
from test_banking_finance_domain import _build_synthetic_banking_corpus  # noqa: E402


_CORPUS_SPECS: dict[str, tuple[str, Callable[[], Any], Any, Any, Any]] = {
    "cart_2024_2_synthetic_replay": (
        "cart_synth",
        build_synthetic_securitization_corpus,
        SECURITIZATION_VOCABULARY,
        SECURITIZATION_ACCEPTANCE_POLICY,
        SECURITIZATION_DOMAIN_PROFILE,
    ),
    "heldout_sec_meridian_2026": (
        "meridian",
        _build_meridian_securitization_corpus,
        SECURITIZATION_VOCABULARY,
        SECURITIZATION_ACCEPTANCE_POLICY,
        SECURITIZATION_DOMAIN_PROFILE,
    ),
    "heldout_bank_apex_2026": (
        "apex",
        _build_apex_banking_corpus,
        BANKING_FINANCE_VOCABULARY,
        BANKING_FINANCE_ACCEPTANCE_POLICY,
        BANKING_FINANCE_DOMAIN_PROFILE,
    ),
    "banking_finance_synthetic": (
        "ridgeline",
        _build_synthetic_banking_corpus,
        BANKING_FINANCE_VOCABULARY,
        BANKING_FINANCE_ACCEPTANCE_POLICY,
        BANKING_FINANCE_DOMAIN_PROFILE,
    ),
}


def _collect_comparison_divergences(
    corpus: Any,
    domain_profile: Any = None,
    min_tier_rank: int | None = None,
) -> list[Any]:
    return list(
        _collect_reference_divergence_collection(
            corpus,
            domain_profile=domain_profile,
            min_tier_rank=min_tier_rank,
        ).divergences
    )


def _collect_reference_divergence_collection(
    corpus: Any,
    domain_profile: Any = None,
    min_tier_rank: int | None = None,
) -> Any:
    prof = domain_profile
    if prof is None:
        adapter = get_domain_adapter(corpus)
        prof = (
            BANKING_FINANCE_DOMAIN_PROFILE
            if adapter.domain_id == "banking_finance"
            else SECURITIZATION_DOMAIN_PROFILE
        )
    return collect_reference_divergences(corpus.documents, prof, min_tier_rank=min_tier_rank)


def _quote_span(doc_text: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    idx = doc_text.find(quote)
    if idx < 0:
        return None
    return (idx, idx + len(quote))


def _spans_overlap(a: tuple[int, int] | None, b: tuple[int, int] | None) -> bool:
    if a is None or b is None:
        return False
    return a[0] < b[1] and b[0] < a[1]


def _conflict_found_in_register(
    conflict: dict[str, Any],
    corpus: Any,
    register: Any,
    default_pri: str,
    default_sec: str,
) -> bool:
    pred = str(conflict.get("predicate") or "")
    pri_id = str(conflict.get("primary_source_id") or default_pri)
    sec_id = str(conflict.get("secondary_source_id") or default_sec)
    pri_doc = corpus.v4_sources.get(pri_id)
    sec_doc = corpus.v4_sources.get(sec_id)
    pri_span = _quote_span(pri_doc.text, str(conflict.get("primary_quote") or "")) if pri_doc else None
    sec_span = _quote_span(sec_doc.text, str(conflict.get("secondary_quote") or "")) if sec_doc else None

    for ob in register.obligations:
        params = ob.parameters or {}
        kind = str(params.get("corpus_obligation_kind") or "")
        if kind not in {"material_conflict", "authority_override"}:
            continue
        ob_pred = str(params.get("predicate") or "")
        if pred and ob_pred == pred:
            return True
        ob_spans = list(params.get("evidence_spans") or ())
        hit_pri = any(
            str(sp.get("document_id") or "") == pri_id
            and _spans_overlap((int(sp["start"]), int(sp["end"])), pri_span)
            for sp in ob_spans
            if sp.get("start") is not None and sp.get("end") is not None
        )
        hit_sec = any(
            str(sp.get("document_id") or "") == sec_id
            and _spans_overlap((int(sp["start"]), int(sp["end"])), sec_span)
            for sp in ob_spans
            if sp.get("start") is not None and sp.get("end") is not None
        )
        if hit_pri and hit_sec:
            return True
    return False


def measure_recall(
    *,
    annotations_path: Path,
    mode: str = "live",
    base_url: str = "https://integrate.api.nvidia.com/v1",
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b",
    api_key: str | None = None,
    timeout: float = 120.0,
    max_tokens: int = 4096,
    thinking_mode: str = "nvidia_off",
    recordings_dir: Path = _HARNESS_DIR / "recorded_reads",
    save_recordings_dir: Path | None = None,
) -> dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    corpora_ann = annotations["corpora"]

    live_reader: LLMProposalReader | None = None
    if mode == "live":
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("LLM_API_KEY") or ""
        if not resolved_key:
            raise RuntimeError("Missing API key: pass --api-key or set NVIDIA_API_KEY / LLM_API_KEY.")
        extra_body: dict[str, Any] = {}
        if thinking_mode == "nvidia_off":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif thinking_mode == "none":
            extra_body["reasoning_effort"] = "none"
        model = OpenAICompatibleModel(
            base_url=base_url,
            model=model_name,
            api_key=resolved_key,
            timeout=timeout,
            max_tokens=max_tokens,
            retries=1,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        live_reader = LLMProposalReader(model=model, prompt=STRICT_READER_PROMPT)

    per_corpus: list[dict[str, Any]] = []
    tot_docs = 0
    tot_read_ok = 0
    tot_annotated = 0
    tot_found = 0
    tot_divergences = 0
    tot_gaps = 0

    for case_id, spec in _CORPUS_SPECS.items():
        short_name, builder, vocab, pol, dprof = spec
        ann = corpora_ann[case_id]
        corpus = builder()

        if mode == "live":
            assert live_reader is not None
            reader: Any = live_reader
            reader_meta = dict(live_reader.identity)
        else:
            rec_path = recordings_dir / f"{short_name}.json"
            reader = RecordedProposalReader.from_file(rec_path)
            reader_meta = dict(reader.identity)

        layer = EvidenceLayer(vocabulary=vocab, policy=pol, reader=reader)
        sdocs = list(corpus.v4_sources.values())
        print(f"[{short_name}] ingesting {len(sdocs)} docs ({mode})...", file=sys.stderr, flush=True)
        ingest_res = layer.ingest(sdocs)
        print(
            f"[{short_name}] admitted={len(layer.units)} rejected={ingest_res.get('rejection_reasons')} errs={len(ingest_res.get('read_errors') or ())}",
            file=sys.stderr,
            flush=True,
        )

        if mode == "live" and save_recordings_dir is not None:
            save_recordings_dir.mkdir(parents=True, exist_ok=True)
            rec_payload: dict[str, Any] = {
                "corpus_id": short_name,
                "case_id": corpus.case_id,
                "model": model_name,
                "provider": "openai-compatible",
                "prompt_version": reader.prompt_version,
                "prompt_digest": digest(reader.prompt),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "fragment_chars": reader.fragment_chars,
                "documents": {},
            }
            for sdoc in sdocs:
                req = {
                    "task": "propose",
                    "contract": CONTRACT,
                    "reader": reader.identity,
                    "document": {
                        "document_id": sdoc.document_id,
                        "revision": sdoc.revision,
                        "text": sdoc.text,
                    },
                    "vocabulary": layer.vocabulary.describe(),
                    "profile": layer.profile.describe(),
                }
                _, stored = layer.store.head("read", digest(req))
                result = (stored or {}).get("result") or {}
                raw_props: list[dict[str, Any]] = []
                for c in result.get("calls") or ():
                    props = ((c.get("response") or {}).get("output") or {}).get("proposals") or ()
                    if isinstance(props, list):
                        raw_props.extend(props)
                rec_payload["documents"][sdoc.document_id] = {
                    "revision": sdoc.revision,
                    "raw_proposals": raw_props,
                    "calls": result.get("calls") or [],
                    "errors": result.get("errors") or [],
                }
            (save_recordings_dir / f"{short_name}.json").write_text(json.dumps(rec_payload, indent=2))

        div_coll = _collect_reference_divergence_collection(corpus, domain_profile=dprof)
        divs = list(div_coll.divergences)
        register = build_corpus_reference_register(
            layer,
            vocab,
            divergence_collection=div_coll,
            expected_documents=corpus.v4_sources,
        )
        conflicts = list(ann.get("conflicts") or ())
        default_pri = str(ann.get("primary_source_id") or "")
        default_sec = str(ann.get("secondary_source_id") or "")
        found_ids: list[str] = []
        missed_ids: list[str] = []
        for c_item in conflicts:
            cid = str(c_item["conflict_id"])
            if _conflict_found_in_register(c_item, corpus, register, default_pri, default_sec):
                found_ids.append(cid)
            else:
                missed_ids.append(cid)

        doc_statuses = layer.document_read_statuses
        read_ok_cnt = sum(1 for st in doc_statuses.values() if st.get("status") == "read_ok")
        gap_cnt = len(register.gap_divergences)
        div_cnt = len(divs)

        tot_docs += len(sdocs)
        tot_read_ok += read_ok_cnt
        tot_annotated += len(conflicts)
        tot_found += len(found_ids)
        tot_divergences += div_cnt
        tot_gaps += gap_cnt

        per_corpus.append(
            {
                "corpus_id": short_name,
                "case_id": case_id,
                "documents_total": len(sdocs),
                "documents_read_ok": read_ok_cnt,
                "units_admitted": len(layer.units),
                "rejection_reasons": dict(ingest_res.get("rejection_reasons") or {}),
                "annotated_conflicts": len(conflicts),
                "conflicts_found": len(found_ids),
                "found_conflict_ids": found_ids,
                "missed_conflict_ids": missed_ids,
                "recall": round(len(found_ids) / len(conflicts), 4) if conflicts else 1.0,
                "divergent_sections": div_cnt,
                "reference_register_gaps": gap_cnt,
                "gap_rate": round(gap_cnt / div_cnt, 4) if div_cnt else 0.0,
                "incomplete": register.incomplete,
                "incomplete_documents": list(register.incomplete_documents),
            }
        )

    return {
        "mode": mode,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reader": reader_meta,
        "annotations_path": str(annotations_path),
        "summary": {
            "documents_total": tot_docs,
            "documents_read_ok": tot_read_ok,
            "annotated_conflicts": tot_annotated,
            "conflicts_found": tot_found,
            "reader_conflict_recall": round(tot_found / tot_annotated, 4) if tot_annotated else 1.0,
            "divergent_sections_total": tot_divergences,
            "reference_register_gaps_total": tot_gaps,
            "reference_register_gap_rate": round(tot_gaps / tot_divergences, 4) if tot_divergences else 0.0,
        },
        "corpora": per_corpus,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure reader recall and REFERENCE_REGISTER_GAP rate.")
    parser.add_argument("--annotations", type=Path, default=_BENCH_DIR / "conflict_annotations.json")
    parser.add_argument("--mode", choices=("live", "recorded"), default="live")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-mode", choices=("nvidia_off", "none", "default"), default="nvidia_off")
    parser.add_argument("--recordings-dir", type=Path, default=_HARNESS_DIR / "recorded_reads")
    parser.add_argument("--save-recordings-dir", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = measure_recall(
        annotations_path=args.annotations,
        mode=args.mode,
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode,
        recordings_dir=args.recordings_dir,
        save_recordings_dir=args.save_recordings_dir,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
