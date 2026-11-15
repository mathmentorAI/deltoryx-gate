# Delivery Summary: ClaimLayer v4 List Extraction, Bounded Re-Read, Token-Level Textual Diff, and External Evaluation Harness

**Model tested live:** `nvidia/nemotron-3-ultra-550b-a55b` (`https://integrate.api.nvidia.com/v1`, `temperature=0`)  
**Prompt versions:** Initial reader `v4-proposals-7` (`STRICT_PROMPT_VERSION`), targeted re-reader `v4-reread-2` (`REREAD_PROMPT_VERSION`)

---

## 0. Leave-One-Out Ablation Integrity (`10/10` Preserved)

The cross-document gap checker was NOT relaxed. Leave-one-out ablation across all 10 annotated conflicts passes `10/10` (`100%`, 95% Wilson CI `[0.7225, 1.0]`) on both recordings:
- `benchmarks/reader_recall_baseline.json` (`recorded_reads/`, `v4-proposals-3`): `conflicts_removed=10`, `gaps_detected=10`, CI `[0.7225, 1.0]`.
- `benchmarks/reader_recall_v2.json` (`recorded_reads_v2/`, `v4-proposals-7` + `v4-reread-2`): `conflicts_removed=10`, `gaps_detected=10`, CI `[0.7225, 1.0]`.

---

## 1. Lowering Escalations via Better Extraction (Requirements 1a, 1b, 1c)

### Implemented Changes
1. **Generic `list` value type & per-element `FieldEvidence` (`src/claimlayer/v4/`):**
   - `Vocabulary` (`FieldSpec.list`) supports ordered lists with indexed `FieldEvidence` (`value[0]`, `value[1]`, ...) resolved in `reader.py` (`_bind_raw_proposals`) and checked for step/item spans in `corpus_obligations.py`.
   - `SECURITIZATION_VOCABULARY` adds `waterfall_schedule` (`src/claimlayer/v4/domains/securitization.py:648-665`).
2. **Blind, bounded gap-targeted re-reading (`TargetedRereadConfig`):**
   - Defined `TargetedRereadConfig` (`src/claimlayer/v4/corpus_obligations.py:170`) with strict per-document and total caps (`max_attempts_per_document=3`, `max_total_attempts=12`), `REREAD_PROMPT` (`src/claimlayer/v4/reader.py`), `EvidenceLayer.admit_reread_items` (`src/claimlayer/v4/engine.py:778`), and `CorpusReferenceRegister.reread_provenance` (`src/claimlayer/v4/corpus_obligations.py:199, 1817`), surfaced through `harness2_v4_pipeline.py:1762-1868`.
   - Rereads receive only the local fragments covering the uncovered `target_sentence` and its `target_tokens` — zero downstream agent answers and zero sibling-document values.
3. **Production admission & immutability fixes discovered under live testing:**
   - Added `"authoritative_instrument"` to `allowed_source_attributions` on `SECURITIZATION_ACCEPTANCE_POLICY` (`src/claimlayer/v4/domains/securitization.py:718-727`) and `BANKING_FINANCE_ACCEPTANCE_POLICY` (`src/claimlayer/v4/domains/banking_finance.py:139-148`) so targeted re-read admissions on governing instruments are not rejected with `source_attribution_not_admitted` (without altering `SourceDocument.revision` digests).
   - Added `EvidenceLayer._put_unit_if_new` (`src/claimlayer/v4/engine.py:868-878`) so repeated semantic emissions with differing raw `proposal` metadata do not raise `ValueError: Cannot overwrite an immutable semantic record`.

### Live Measurement Comparison: `before` (`v1`) vs `after` (`v2`)

| Metric | `reader_recall_baseline.json` (`recorded_reads/`, `v4-proposals-3`) | `reader_recall_v2.json` (`recorded_reads_v2/`, `v4-proposals-7` + `v4-reread-2`) |
| --- | --- | --- |
| Documents read cleanly | `15/15` | `15/15` |
| Leave-one-out ablation (`95%` Wilson CI) | `10/10` (`1.0`, `[0.7225, 1.0]`) | `10/10` (`1.0`, `[0.7225, 1.0]`) |
| Annotated conflict recall (`95%` Wilson CI) | `9/10` (`0.9`, `[0.5958, 0.9821]`) | `8/10` (`0.8`, `[0.4902, 0.9433]`) |
| Total quantity gaps across corpora | `8` (`cart_synth=7`, `meridian=1`, `apex=0`, `ridgeline=0`) | `4` (`cart_synth=3`, `meridian=1`, `apex=0`, `ridgeline=0`) |
| Gaps closed by targeted re-read (`95%` Wilson CI) | `0/21` (`0.0`, `[0.0, 0.1546]`, disabled in v1) | `2/21` (`0.0952`, `[0.0265, 0.2891]`) |
| Clean-candidate escalation rate (`95%` Wilson CI) | `1/3` (`0.3333`, `[0.0615, 0.7923]`) | `0/2` (`0.0`, `[0.0, 0.6576]`; on clean full-recall `apex` & `ridgeline`) |
| Clean-corpus false gaps / sentence check | `9/54` (`0.1667`, `[0.0902, 0.2874]`, `17` raw gaps across `cart_synth`/`apex`/`ridgeline`) | `0/9` (`0.0`, `[0.0, 0.2991]`, on `apex` & `ridgeline`) |

**Honest status on securitization escalation:**  
- In `meridian`, pass-2 targeted re-reading closed both textual gaps on `meridian_class_b_coupon` (`attempts_total=3`, `closed_gap_count=2`), leaving `2` genuine open gaps on `Reserve Account replenishment occurs at Step 8 ahead of Class B principal.` (`STEP:8` + `TEXTUAL_DIFF`) where the model's `waterfall_schedule` re-read failed strict `fidelity:PREDICATE_NOT_IN_EVIDENCE`, keeping `clean_candidate_escalates_on_gap=true` (`meridian_reserve_waterfall_step` still missed cleanly -> fails closed).
- In `cart_synth`, quantity gaps dropped from `7` to `3`, while `17` total open gaps (`3` quantity, `14` textual — including unextracted `Steps 1-9` and `Step 10` cross-references) remain un-admitted under strict fidelity, so `cart_synth` also continues to escalate cleanly rather than weakening verification.

---

## 2. Token-Level Diff Textual Cross-Check (Requirement 2)

- Updated `src/claimlayer/v4/corpus_obligations.py` (`_differing_spans_from_obligations` at `:521`, `_divergence_gaps_for_item` at `:664`):
  - Runs token-level `difflib.SequenceMatcher` across **all** differing aligned sentence pairs (both with and without differing numeric values), so clauses like `bear interest at 5.80% and rank senior to Class C` vs `subordinated to Class C` emit `kind="textual"` gaps unless an admitted differing-value unit covers the differing spans.
  - Ignores only formatting (whitespace, punctuation, capitalization) and structural document-identifier / role aliases.
- Covered by deterministic unit tests in `tests/test_v4_list_reread_and_textual_diff.py`:
  - `test_textual_diff_same_coupon_plus_subordination_and_joint_numeric_shift`
  - `test_textual_diff_catches_non_numeric_variants_and_ignores_formatting_and_doc_ids`
- False-gap rate on clean banking corpora (`apex`, `ridgeline`): `0/9` sentence checks (`0.0%`, 95% Wilson CI `[0.0, 0.2991]`) across both v1 and v2; across all 59 sentence checks (`all corpora`), distinct textual-gap sentence checks = `9/59` (`0.1525`, 95% CI `[0.0824, 0.2652]`).

---

## 3. External Public-Document Evaluation Scaffold (Requirement 3)

Created `benchmarks/external_eval/` with zero annotations inside the repository:
- `benchmarks/external_eval/edgar_loader.py` — SEC EDGAR downloader producing an immutable SHA-256 provenance manifest (`--user-agent` required, `424B*` vs `EX-4.*` / `EX-10.*` / `EX-99.*` securitization pairs and Form `8-K`/`10-K`/`10-Q` credit agreement/amendment pairs).
- `benchmarks/external_eval/ANNOTATION_INSTRUCTIONS.md` & `benchmarks/external_eval/annotation_template.json` — independent human annotation guide & blank template (`--annotations` stays strictly outside git).
- `benchmarks/external_eval/protocol.json` — frozen protocol (`status: "frozen_before_external_evaluation"`, updated to `strict_prompt_version: "v4-proposals-7"`, `reread_prompt_version: "v4-reread-2"`).
- `benchmarks/external_eval/evaluate_external.py` — manifest-verified runner enforcing external `--annotations` path, computing Wilson 95% CIs, and refusing non-external paths.
- Tested offline in `tests/test_external_eval_scaffold.py`.

---

## 4. Verification Commands

- `.venv/bin/pytest claim-layer/tests -q`
- `.venv/bin/python claim-layer/benchmarks/measure_reader_recall.py --mode recorded --recordings-dir claim-layer/examples/harness2_securitization/recorded_reads --json-out /tmp/v1_check.json`
- `.venv/bin/python claim-layer/benchmarks/measure_reader_recall.py --mode recorded --recordings-dir claim-layer/examples/harness2_securitization/recorded_reads_v2 --json-out /tmp/v2_check.json`
