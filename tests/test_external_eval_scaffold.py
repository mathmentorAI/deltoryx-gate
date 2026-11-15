"""Tests for external public-document SEC EDGAR evaluation scaffold (zero repo annotations)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EXT_DIR = _ROOT / "benchmarks" / "external_eval"
_SRC_DIR = _ROOT / "src"
for _p in (_SRC_DIR, _EXT_DIR, _ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from edgar_loader import (  # noqa: E402
    EdgarFetchSpec,
    build_edgar_doc_url,
    fetch_edgar_corpus,
    select_filing_pairs_from_submissions,
    sha256_text,
    verify_manifest_integrity,
)
from evaluate_external import assert_annotations_outside_repo, evaluate_external_corpus  # noqa: E402


def test_external_protocol_and_template_zero_repo_annotations():
    """Protocol is frozen before eval; template is empty and contains zero model/human labels."""
    proto = json.loads((_EXT_DIR / "protocol.json").read_text(encoding="utf-8"))
    assert proto["status"] == "frozen_before_external_evaluation"
    assert proto["sources"]["provenance_requirements"]["no_repo_annotations"] is True
    assert "wilson_score" in proto["metrics_and_thresholds"]["confidence_interval_method"]
    assert proto["reader_configuration"]["reread_config"]["enabled"] is True

    tpl = json.loads((_EXT_DIR / "annotation_template.json").read_text(encoding="utf-8"))
    assert tpl["blind_to_model_outputs"] is True
    assert tpl["annotator_id"] == ""
    assert "example_corpus_id_replace_me" in tpl["corpora"]
    # Verify NO populated annotation files live inside benchmarks/external_eval
    json_files = {p.name for p in _EXT_DIR.glob("*.json")}
    assert json_files == {"protocol.json", "annotation_template.json"}

    with pytest.raises(ValueError, match="OUTSIDE the repository"):
        assert_annotations_outside_repo(_EXT_DIR / "annotations.json", repo_root=_ROOT)


def test_edgar_loader_manifest_provenance_and_external_eval(tmp_path: Path):
    """Offline EDGAR fetch builds immutable SHA-256 manifest and evaluates external recordings."""
    # Mocked public HTML filings served cleanly via dependency injection
    indenture_html = (
        "<html><body><h1>EX-4.1 INDENTURE</h1>"
        "<p>Section 2.04 Interest Rates: Class B Notes ($40,000,000 principal) bear interest at 5.80% per annum.</p>"
        "</body></html>"
    )
    supp_html = (
        "<html><body><p>PROSPECTUS SUPPLEMENT</p>"
        "<p>Class B Notes ($40,000,000 principal) bear interest at 5.55% per annum.</p>"
        "</body></html>"
    )
    u_pri = build_edgar_doc_url("0001234567", "0001234567-26-000101", "ex4_1.htm")
    u_sec = build_edgar_doc_url("0001234567", "0001234567-26-000101", "d424b2.htm")
    blobs = {u_pri: indenture_html.encode("utf-8"), u_sec: supp_html.encode("utf-8")}

    corpus_dir = tmp_path / "ext_corpus"
    spec = EdgarFetchSpec(
        corpus_id="edgar_abs_demo",
        domain="securitization",
        cik="0001234567",
        primary_accession="0001234567-26-000101",
        primary_filename="ex4_1.htm",
        secondary_accession="0001234567-26-000101",
        secondary_filename="d424b2.htm",
    )
    manifest = fetch_edgar_corpus([spec], output_dir=corpus_dir, fetcher=lambda url: blobs[url])
    assert manifest["manifest_sha256"]
    verified = verify_manifest_integrity(corpus_dir)
    doc_records = verified["corpora"][0]["documents"]
    assert len(doc_records) == 2
    assert all(d["source_url"].startswith("https://www.sec.gov/Archives/edgar/data/") for d in doc_records)
    assert all(d["downloaded_utc"] and d["normalized_sha256"] for d in doc_records)

    # Second fetch to same dir must refuse to overwrite immutable manifest
    with pytest.raises(FileExistsError):
        fetch_edgar_corpus([spec], output_dir=corpus_dir, fetcher=lambda url: blobs[url])

    # Submissions selector filters deterministic forms
    subs = {
        "filings": {
            "recent": {
                "form": ["8-K", "424B2", "10-K"],
                "accessionNumber": ["0001-26-0001", "0001-26-0003", "0001-26-0002"],
                "primaryDocument": ["f8k.htm", "f424b2.htm", "f10k.htm"],
                "filingDate": ["2026-01-10", "2026-03-01", "2026-02-01"],
            }
        }
    }
    assert select_filing_pairs_from_submissions(subs, domain="securitization")[0]["form"] == "424B2"
    assert len(select_filing_pairs_from_submissions(subs, domain="banking_finance")) == 2

    # Prepare outside-repo annotations & recorded proposals for end-to-end eval check
    pri_id = doc_records[0]["document_id"]
    sec_id = doc_records[1]["document_id"]
    pri_text = (corpus_dir / doc_records[0]["relative_path"]).read_text(encoding="utf-8")
    sec_text = (corpus_dir / doc_records[1]["relative_path"]).read_text(encoding="utf-8")
    from claimlayer.semantic.models import SourceDocument
    from claimlayer.v4.reader import _build_document_fragments

    pri_src = SourceDocument(document_id=pri_id, text=pri_text, attribution="controlling_instrument", authority_rank=100)
    sec_src = SourceDocument(document_id=sec_id, text=sec_text, attribution="source_statement", authority_rank=60)
    pri_ev = _build_document_fragments({"document_id": pri_id, "revision": pri_src.revision, "text": pri_text}, 1800)[0][0]
    sec_ev = _build_document_fragments({"document_id": sec_id, "revision": sec_src.revision, "text": sec_text}, 1800)[0][0]

    rec_dir = tmp_path / "ext_recordings"
    rec_dir.mkdir(parents=True)
    rec_payload = {
        "corpus_id": "edgar_abs_demo",
        "model": "test-recorder",
        "prompt_version": "v4-proposals-4",
        "prompt_digest": "test-digest",
        "recorded_at": "2026-11-15T18:15:00Z",
        "documents": {
            pri_id: {
                "revision": pri_src.revision,
                "raw_proposals": [
                    {
                        "entity": "Class B Notes",
                        "predicate": "note_principal",
                        "value": {"value": 40000000, "unit": "USD"},
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [pri_ev],
                        "quote": "Class B Notes ($40,000,000 principal)",
                        "field_evidence": {"entity": "Class B Notes", "value": "$40,000,000"},
                    },
                    {
                        "entity": "Class B Notes",
                        "predicate": "coupon_rate",
                        "value": {"value": 5.8, "unit": "percent_per_annum"},
                        "assertion_kind": "requirement",
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [pri_ev],
                        "quote": "bear interest at 5.80% per annum.",
                        "field_evidence": {"entity": "Class B Notes", "value": "5.80%"},
                    },
                ],
            },
            sec_id: {
                "revision": sec_src.revision,
                "raw_proposals": [
                    {
                        "entity": "Class B Notes",
                        "predicate": "note_principal",
                        "value": {"value": 40000000, "unit": "USD"},
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [sec_ev],
                        "quote": "Class B Notes ($40,000,000 principal)",
                        "field_evidence": {"entity": "Class B Notes", "value": "$40,000,000"},
                    },
                    {
                        "entity": "Class B Notes",
                        "predicate": "coupon_rate",
                        "value": {"value": 5.55, "unit": "percent_per_annum"},
                        "assertion_kind": "fact",
                        "condition_status": "unconditional",
                        "basis": "explicit",
                        "evidence": [sec_ev],
                        "quote": "bear interest at 5.55% per annum.",
                        "field_evidence": {"entity": "Class B Notes", "value": "5.55%"},
                    },
                ],
            },
        },
    }
    (rec_dir / "edgar_abs_demo.json").write_text(json.dumps(rec_payload), encoding="utf-8")

    ann_path = tmp_path / "outside_annotations.json"
    ann_path.write_text(
        json.dumps({
            "schema_version": "1.0.0",
            "corpora": {
                "edgar_abs_demo": {
                    "domain": "securitization",
                    "primary_source_id": pri_id,
                    "secondary_source_id": sec_id,
                    "conflicts": [{
                        "conflict_id": "edgar_b_coupon",
                        "predicate": "coupon_rate",
                        "entity": "Class B Notes",
                        "primary_quote": "bear interest at 5.80% per annum.",
                        "secondary_quote": "bear interest at 5.55% per annum.",
                    }],
                }
            },
        }),
        encoding="utf-8",
    )

    rep = evaluate_external_corpus(
        corpus_dir=corpus_dir,
        annotations_path=ann_path,
        mode="recorded",
        recordings_dir=rec_dir,
    )
    assert rep["summary"]["reader_conflict_recall_wilson_95"]["k"] == 1
    assert rep["summary"]["ablation"]["gaps_detected"] == 1
    assert rep["manifest_sha256"] == manifest["manifest_sha256"]
    assert sha256_text(pri_text) == doc_records[0]["normalized_sha256"]
