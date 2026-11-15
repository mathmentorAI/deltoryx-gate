# Delivery Report: Sentence- & Quantity-Level Cross-Check, Unified Comparison Policy, Provenance, and Ablation Benchmarks

**Baseline audited:** `mathmentorAI/deltoryx-gate` commit `ca2a5b0`  
**Commits delivered (`claim-layer` / `deltoryx-gate` `main`):**
- `3b5b99e` — Unify divergence collection in `DomainProfile` and propagate exclusion receipts (Point 2)
- `89c8dd1` — Sentence- and quantity-level divergence cross-check without section masking (Point 1)
- `ac3162c` — Record model-authored provenance metadata on `conflict_annotations.json` (Point 3)
- `48265ec` — Remove hardcoded local paths and repo directory name assumptions (Point 5 / AC8)
- `8d2de16` — Add 10-conflict leave-one-out ablation, `quantity`/`textual` gap breakdown, 95% Wilson CIs, and AC1–AC8 tests (Point 4)

---

## 1. Sentence- & Quantity-Level Cross-Check (`89c8dd1`, refined in `8d2de16`)

### Root Cause Addressed
Previously, `_divergence_covered_by_reader_obligations` judged coverage at the whole-divergence/section tier. In headerless or coarse documents (notably `heldout_sec_meridian_2026` where `meridian-2026-1-draft-oc.docx` has no headers), one extracted conflict (`meridian_class_b_coupon`: `5.80%` vs `5.55%`) masked an omitted conflict (`meridian_reserve_waterfall_step`: `Step 12` vs `Step 8`) inside the same `SectionDivergence`.

### Implementation
- **`src/claimlayer/verifiers/quote_quantity.py`**:
  - Added `segment_sentences_with_spans(text, base_offset=0)` reusing v3's `_is_sentence_boundary_dot` so decimals (`5.80%`), dollar amounts, and legal abbreviations (`Sec.`, `No.`, `Inc.`) do not split sentences falsely while preserving exact character offsets.
  - Tightened `_SECTION_HEADER_ONLY_RE` (fixing a backtracking match where `"Section 9.01 <body...>"` could match `"9"` + `"."` + `"01 <body...>"` as header-only).
- **`src/claimlayer/v4/corpus_obligations.py`**:
  - Added `_extract_sentence_quantity_tokens(sentence, sentence_start)` extracting normalized quantities (`extract_quantities`), explicit step/priority integers (`STEP:<n>`), and dates (`extract_dates_with_spans`), while masking structural identifiers (`Section 4.02`, `Class A-1`, deal-year tokens like `Series 2025-1`, multi-tranche phrases) with length-preserving `"~"` characters so `.strip()` inside `normalize_span` cannot shift offsets.
  - Added `_align_sentences(...)` and `_divergence_gaps_for_item(...)` in `build_corpus_reference_register(...)`:
    - Differing aligned sentences and unmatched differing sentences check every divergent `quantity` token against admitted unit `EvidenceRef` spans on that same `document_id`. Any uncovered token emits a `REFERENCE_REGISTER_GAP` (`kind="quantity"`, `document_id`, `offset`, `token`, `sentence`, unique `obligation_id`).
    - Aligned differing sentences with no distinct quantities on either side and lacking overlapping admitted unit spans on both sides emit `kind="textual"` (`token="TEXTUAL_DIFF"`), rather than suppressing textual divergences with ad-hoc exceptions.

### Empirical Impact on Unmodified Recorded Reads (`recorded_reads/*.json`)
- **`meridian`**: Detects **`1` gap** — `quantity` token `STEP:8` at offset `127` in `meridian-2026-1-draft-oc.docx` (`"Reserve Account replenishment occurs at Step 8 ahead of Class B principal."`).
  - Consequence: `sec_meridian_clean` in `benchmarks/heldout_multidomain_eval.py` (which omitted the Reserve Account waterfall discrepancy) flips from false `ACCEPT` to **`ESCALATE`** with `REFERENCE_REGISTER_GAP` and is reclassified as `kind="seeded_defect"`.
- **`cart_synth`**: Detects **`7` gaps** (all `kind="quantity"` on step numbers `STEP:1, 2, 8, 9, 10, 12` across Indenture, OM, and Email where `cart_synth.json` recorded no per-step waterfall unit spans). Accordingly, `test_07` in `tests/test_harness2_v4_securitization.py` asserts fail-closed `ESCALATE` (`REFERENCE_REGISTER_GAP`) on incomplete `cart_synth.json` and exercises full `ACCEPT` (report) + `ESCALATE` (operational authorization) on a complete 2-document synthetic reader fixture.
- **`apex`** and **`ridgeline`**: **`0` gaps**.

---

## 2. Single Divergence Collection & Declarative Exclusion Provenance (`3b5b99e`)

- Added `PolicyConfiguration.excluded_comparison_roles` (`Mapping[str, str]`, included in `describe()` and therefore `DomainProfile.fingerprint()`).
- Configured `BANKING_EXCLUDED_COMPARISON_ROLES` and `SECURITIZATION_EXCLUDED_COMPARISON_ROLES` (excluding `quantitative_financial_model` and `client_instructions_scope`) on `BANKING_FINANCE_DOMAIN_PROFILE` and `SECURITIZATION_DOMAIN_PROFILE`.
- Unified both production (`examples/harness2_securitization/harness2_v4_pipeline.py`) and benchmark measurement (`benchmarks/measure_reader_recall.py`) on `collect_reference_divergences(corpus_documents, domain_profile)` (exported from `claimlayer.v4`).
- Propagated `excluded_documents` and `domain_profile_fingerprint` through `CorpusReferenceRegister`, `ResponseAssessment.corpus_coverage`, `decision_snapshot`, and `signed_receipt`.
- Under the unified production policy on `apex`, `apex-financial-summary.xlsx` and `apex-partner-scope.eml` are excluded with recorded rationale, yielding `2` production comparisons and `0` measurement-only Apex gaps.

---

## 3. Honest Annotation Provenance (`ac3162c`)

- Updated `benchmarks/conflict_annotations.json` without modifying any conflict definition, predicate, quote, source ID, or value:
  - Updated top-level `"description"` to `"model-authored, pending human review (not independent human ground truth)"`.
  - Added `"annotator": "llm_draft_pending_human_review"` and `"annotated_at": "2026-11-15T00:00:00Z"` to all `10` conflict records.

---

## 4. Benchmark Measurement: Leave-One-Out Ablation, False-Gap Breakdown & 95% Wilson CIs (`8d2de16`)

Measurements recorded in `benchmarks/reader_recall_baseline.json` by `benchmarks/measure_reader_recall.py --mode recorded`:

| Metric | Result (`k / n`) | Rate | 95% Wilson CI |
| :--- | :---: | :---: | :---: |
| **Reader conflict recall** | `9 / 10` | `0.9000` | `[0.5958, 0.9821]` |
| **Leave-one-out conflict ablation** (production `collect_reference_divergences` path, all 10 annotated conflicts) | `10 / 10` | `1.0000` | `[0.7225, 1.0000]` |
| **Textual gaps (`kind="textual"`) across all corpus sentence checks** | `0 / 32` | `0.0000` | `[0.0000, 0.1072]` |
| **Clean-annotated corpora (`cart_synth`, `apex`, `ridgeline`) false gaps per divergence** | `7 / 12` (`7` `quantity`, `0` `textual`) | `0.5833` | `[0.3195, 0.8067]` |
| **Clean-annotated corpora (`cart_synth`, `apex`, `ridgeline`) false gaps per sentence check** | `7 / 27` (`7` `quantity`, `0` `textual`) | `0.2593` | `[0.1317, 0.4468]` |
| **Clean-annotated corpora `kind="textual"` per sentence check** | `0 / 27` | `0.0000` | `[0.0000, 0.1246]` |

Note on the `7` `quantity` gaps in `cart_synth`: `cart_synth.json` extracted the `waterfall_priority` enum conflict (`cart_waterfall_reserve_priority`) using spans covering Steps 3–7, Step 13, Step 14, and Step 6/10, leaving individual step numbers `1, 2, 8, 9, 10, 12` unextracted on divergent sentences. On `apex` (`n=2` divergences, `n=4` sentence checks) and `ridgeline` (`n=2` divergences, `n=5` sentence checks), observed gaps were `0` (`0` `quantity`, `0` `textual`).

---

## 5. Portability & Test Hygiene (`48265ec`)

- Removed all `/Users/...` hardcoded local paths and `"claim-layer"` checkout directory name assumptions from `benchmarks/heldout_multidomain_eval.py`, `examples/harness2_securitization/document_manifest.json`, `examples/harness2_securitization/harness2_v4_pipeline.py`, `examples/harness2_securitization/harness_surface/scripts/claimlayer_gate.py`, and `benchmarks/reader_recall_baseline.json`.
- Committed `common_evidence.txt` fixtures under `examples/harness2_securitization/fixtures/<case_id>/common_evidence.txt` with verified SHA-256 checksums (`document_manifest.json`).

---

## Verification

- **Workspace test suite (`claim-layer/tests`)**: `421 passed, 1 skipped` in `10.04s` (including new AC1–AC8 tests in `tests/test_v4_corpus_closure_and_anti_evasion.py`).
- **Root workspace tests (`tests/`)**: `3 passed` in `0.26s`.
- **Renamed-directory clone check (AC8)** at `/tmp/deltoryx-gate-ac8-check`: `421 passed, 1 skipped` in `10.40s`.
