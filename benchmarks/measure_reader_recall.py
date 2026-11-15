#!/usr/bin/env python3
"""Measure reader conflict recall and REFERENCE_REGISTER_GAP rate against independent annotations."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
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
from claimlayer.v4.corpus_obligations import _align_sentences  # noqa: E402
from claimlayer.v4.spans import locate_quote  # noqa: E402
from claimlayer.verifiers.quote_quantity import segment_sentences_with_spans  # noqa: E402
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


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    """Return binomial rate and two-sided Wilson score confidence interval (default 95%)."""
    if n <= 0:
        return {"k": k, "n": n, "rate": None, "ci_95_low": None, "ci_95_high": None}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)) / denom
    return {
        "k": int(k),
        "n": int(n),
        "rate": round(p, 4),
        "ci_95_low": max(0.0, round(center - half, 4)),
        "ci_95_high": min(1.0, round(center + half, 4)),
    }


def _count_sentence_checks(corpus: Any, divergences: list[Any]) -> int:
    total = 0
    for d in divergences:
        pri_id = str(getattr(d, "primary_source_id", "") or "")
        sec_id = str(getattr(d, "secondary_source_id", "") or "")
        pri_span = getattr(d, "primary_span", None)
        sec_span = getattr(d, "secondary_span", None)
        pfull = corpus.v4_sources[pri_id].text if pri_id in corpus.v4_sources else ""
        sfull = corpus.v4_sources[sec_id].text if sec_id in corpus.v4_sources else ""
        ptxt = pfull[pri_span[0]:pri_span[1]] if pri_span else ""
        stxt = sfull[sec_span[0]:sec_span[1]] if sec_span else ""
        ps = segment_sentences_with_spans(ptxt, base_offset=int(pri_span[0]) if pri_span else 0)
        ss = segment_sentences_with_spans(stxt, base_offset=int(sec_span[0]) if sec_span else 0)
        pairs, un_p, un_s = _align_sentences(ps, ss)
        total += len(pairs) + len(un_p) + len(un_s)
    return total


def _proposal_overlaps_conflict_quote(doc_text: str, prop_quote: str, target_quote: str) -> bool:
    if not prop_quote or not target_quote:
        return False
    sp1 = locate_quote(doc_text, prop_quote) if doc_text else None
    sp2 = locate_quote(doc_text, target_quote) if doc_text else None
    if sp1 is not None and sp2 is not None:
        return _spans_overlap(sp1, sp2)
    return (target_quote.lower() in prop_quote.lower()) or (prop_quote.lower() in target_quote.lower())


def _num_tokens(text: str) -> set[float]:
    return {float(tok) for tok in re.findall(r"\d+(?:\.\d+)?", text or "")}


def _snapshot_reader_documents(
    layer: EvidenceLayer,
    reader: Any,
    sdocs: list[Any],
    model_name: str,
) -> dict[str, Any]:
    if isinstance(reader, RecordedProposalReader):
        return copy.deepcopy(dict(reader.documents))
    docs: dict[str, Any] = {}
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
        docs[sdoc.document_id] = {
            "revision": sdoc.revision,
            "raw_proposals": raw_props,
            "calls": result.get("calls") or [],
            "errors": result.get("errors") or [],
        }
    return docs


def _ablate_conflict_leave_one_out(
    *,
    short_name: str,
    corpus: Any,
    vocab: Any,
    pol: Any,
    div_coll: Any,
    base_register: Any,
    base_docs: dict[str, Any],
    base_reader_meta: dict[str, Any],
    conflict: dict[str, Any],
    default_pri: str,
    default_sec: str,
) -> dict[str, Any]:
    cid = str(conflict["conflict_id"])
    pred = str(conflict.get("predicate") or "")
    pri_id = str(conflict.get("primary_source_id") or default_pri)
    sec_id = str(conflict.get("secondary_source_id") or default_sec)
    pri_q = str(conflict.get("primary_quote") or "")
    sec_q = str(conflict.get("secondary_quote") or "")
    target_nums = _num_tokens(str(conflict.get("primary_value") or "")) | _num_tokens(
        str(conflict.get("secondary_value") or "")
    )

    ablated_docs = copy.deepcopy(base_docs)
    removed = 0
    for did, rec in ablated_docs.items():
        src = corpus.v4_sources.get(did)
        dtext = src.text if src is not None else ""
        kept: list[Any] = []
        for prop in rec.get("raw_proposals") or ():
            same_span = (
                did == pri_id and _proposal_overlaps_conflict_quote(dtext, str(prop.get("quote") or ""), pri_q)
            ) or (
                did == sec_id and _proposal_overlaps_conflict_quote(dtext, str(prop.get("quote") or ""), sec_q)
            )
            if not same_span:
                kept.append(prop)
                continue
            pval = prop.get("value")
            scalar = pval.get("value") if isinstance(pval, dict) else pval
            val_match = isinstance(scalar, (int, float)) and float(scalar) in target_nums
            if str(prop.get("predicate") or "") == pred or val_match:
                removed += 1
                continue
            kept.append(prop)
        rec["raw_proposals"] = kept

    ab_reader = RecordedProposalReader(
        model_name=str(base_reader_meta.get("model") or "ablation-reader"),
        prompt_version=str(base_reader_meta.get("prompt_version") or "v4-proposals-3"),
        prompt_digest=str(base_reader_meta.get("prompt_digest") or "ablation"),
        recorded_at=str(base_reader_meta.get("recorded_at") or ""),
        documents=ablated_docs,
        fragment_chars=int(base_reader_meta.get("fragment_chars", 1800)),
        structural_reliability=base_reader_meta.get("structural_reliability"),
        recording_path=None,
    )
    ab_layer = EvidenceLayer(vocabulary=vocab, policy=pol, reader=ab_reader)
    ab_layer.ingest(list(corpus.v4_sources.values()))
    ab_register = build_corpus_reference_register(
        ab_layer,
        vocab,
        divergence_collection=div_coll,
        expected_documents=corpus.v4_sources,
    )
    base_keys = {
        (str(g.get("document_id")), int(g.get("offset", 0)), str(g.get("token")), str(g.get("kind")))
        for g in base_register.gap_divergences
    }
    ab_keys = {
        (str(g.get("document_id")), int(g.get("offset", 0)), str(g.get("token")), str(g.get("kind")))
        for g in ab_register.gap_divergences
    }
    new_keys = sorted(ab_keys - base_keys)
    new_gap_kinds: dict[str, int] = {}
    for _, _, _, gkind in new_keys:
        new_gap_kinds[gkind] = new_gap_kinds.get(gkind, 0) + 1
    return {
        "corpus_id": short_name,
        "conflict_id": cid,
        "predicate": pred,
        "proposals_removed": removed,
        "base_gaps": len(base_keys),
        "ablated_gaps": len(ab_keys),
        "new_gaps_detected": len(new_keys),
        "gap_detected": len(new_keys) > 0,
        "new_gap_kinds": new_gap_kinds,
        "sample_new_gaps": [
            {"document_id": did, "offset": off, "token": tok, "kind": kind}
            for did, off, tok, kind in new_keys[:3]
        ],
    }


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
    ablation_cases: list[dict[str, Any]] = []
    tot_docs = 0
    tot_read_ok = 0
    tot_annotated = 0
    tot_found = 0
    tot_divergences = 0
    tot_sentence_checks = 0
    tot_gaps = 0
    tot_gaps_by_kind: dict[str, int] = {"quantity": 0, "textual": 0, "section": 0}
    clean_corpora_ids: list[str] = []
    clean_divergences = 0
    clean_sentence_checks = 0
    clean_false_gaps = 0
    clean_false_gaps_by_kind: dict[str, int] = {"quantity": 0, "textual": 0, "section": 0}

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

        snapshot_docs = _snapshot_reader_documents(layer, reader, sdocs, model_name)
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
                "documents": snapshot_docs,
            }
            (save_recordings_dir / f"{short_name}.json").write_text(json.dumps(rec_payload, indent=2))

        div_coll = _collect_reference_divergence_collection(corpus, domain_profile=dprof)
        divs = list(div_coll.divergences)
        sent_checks = _count_sentence_checks(corpus, divs)
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
            ablation_cases.append(
                _ablate_conflict_leave_one_out(
                    short_name=short_name,
                    corpus=corpus,
                    vocab=vocab,
                    pol=pol,
                    div_coll=div_coll,
                    base_register=register,
                    base_docs=snapshot_docs,
                    base_reader_meta=reader_meta,
                    conflict=c_item,
                    default_pri=default_pri,
                    default_sec=default_sec,
                )
            )

        doc_statuses = layer.document_read_statuses
        read_ok_cnt = sum(1 for st in doc_statuses.values() if st.get("status") == "read_ok")
        gap_cnt = len(register.gap_divergences)
        div_cnt = len(divs)
        gaps_by_kind: dict[str, int] = {"quantity": 0, "textual": 0, "section": 0}
        for gap in register.gap_divergences:
            gkind = str(gap.get("kind") or "quantity")
            gaps_by_kind[gkind] = gaps_by_kind.get(gkind, 0) + 1
            tot_gaps_by_kind[gkind] = tot_gaps_by_kind.get(gkind, 0) + 1

        is_clean_corpus = len(missed_ids) == 0
        if is_clean_corpus:
            clean_corpora_ids.append(short_name)
            clean_divergences += div_cnt
            clean_sentence_checks += sent_checks
            clean_false_gaps += gap_cnt
            for gkind, gval in gaps_by_kind.items():
                clean_false_gaps_by_kind[gkind] = clean_false_gaps_by_kind.get(gkind, 0) + gval

        tot_docs += len(sdocs)
        tot_read_ok += read_ok_cnt
        tot_annotated += len(conflicts)
        tot_found += len(found_ids)
        tot_divergences += div_cnt
        tot_sentence_checks += sent_checks
        tot_gaps += gap_cnt

        per_corpus.append(
            {
                "corpus_id": short_name,
                "case_id": case_id,
                "clean_control_for_false_gaps": is_clean_corpus,
                "documents_total": len(sdocs),
                "documents_read_ok": read_ok_cnt,
                "units_admitted": len(layer.units),
                "rejection_reasons": dict(ingest_res.get("rejection_reasons") or {}),
                "annotated_conflicts": len(conflicts),
                "conflicts_found": len(found_ids),
                "found_conflict_ids": found_ids,
                "missed_conflict_ids": missed_ids,
                "recall": round(len(found_ids) / len(conflicts), 4) if conflicts else 1.0,
                "recall_wilson_95": wilson_ci(len(found_ids), len(conflicts)),
                "divergent_sections": div_cnt,
                "sentence_checks": sent_checks,
                "reference_register_gaps": gap_cnt,
                "gaps_by_kind": gaps_by_kind,
                "gap_rate": round(gap_cnt / div_cnt, 4) if div_cnt else 0.0,
                "incomplete": register.incomplete,
                "incomplete_documents": list(register.incomplete_documents),
            }
        )

    ab_removed = len(ablation_cases)
    ab_detected = sum(1 for row in ablation_cases if row["gap_detected"])
    try:
        rel_annotations_path = str(annotations_path.resolve().relative_to(_ROOT.resolve()))
    except ValueError:
        rel_annotations_path = annotations_path.name

    return {
        "mode": mode,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "reader": reader_meta,
        "annotations_path": rel_annotations_path,
        "annotations_provenance": str(annotations.get("description") or ""),
        "summary": {
            "documents_total": tot_docs,
            "documents_read_ok": tot_read_ok,
            "annotated_conflicts": tot_annotated,
            "conflicts_found": tot_found,
            "reader_conflict_recall": round(tot_found / tot_annotated, 4) if tot_annotated else 1.0,
            "reader_conflict_recall_wilson_95": wilson_ci(tot_found, tot_annotated),
            "divergent_sections_total": tot_divergences,
            "sentence_checks_total": tot_sentence_checks,
            "reference_register_gaps_total": tot_gaps,
            "gaps_by_kind": tot_gaps_by_kind,
            "reference_register_gap_rate": round(tot_gaps / tot_divergences, 4) if tot_divergences else 0.0,
            "textual_gaps_across_all_corpora_wilson_95": wilson_ci(
                tot_gaps_by_kind.get("textual", 0),
                tot_sentence_checks,
            ),
            "ablation": {
                "conflicts_removed": ab_removed,
                "gaps_detected": ab_detected,
                "detection_rate_wilson_95": wilson_ci(ab_detected, ab_removed),
            },
            "clean_corpus_false_gaps": {
                "clean_corpora_ids": clean_corpora_ids,
                "clean_divergences_total": clean_divergences,
                "clean_sentence_checks_total": clean_sentence_checks,
                "false_gaps_total": clean_false_gaps,
                "false_gaps_by_kind": clean_false_gaps_by_kind,
                "per_divergence_wilson_95": wilson_ci(clean_false_gaps, clean_divergences),
                "per_sentence_check_wilson_95": wilson_ci(clean_false_gaps, clean_sentence_checks),
                "quantity_per_sentence_check_wilson_95": wilson_ci(
                    clean_false_gaps_by_kind.get("quantity", 0),
                    clean_sentence_checks,
                ),
                "textual_per_sentence_check_wilson_95": wilson_ci(
                    clean_false_gaps_by_kind.get("textual", 0),
                    clean_sentence_checks,
                ),
            },
        },
        "ablation": ablation_cases,
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
