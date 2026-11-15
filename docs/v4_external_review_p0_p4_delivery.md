# ClaimLayer v3 + v4 — External Review Remediation Report (P0–P4, Points #1–#15)

**Baseline audited:** `mathmentorAI/deltoryx-gate@4c818bd`  
**Branch:** `main`  
**Remediation commits (local history on top of `4c818bd`):**
- **P4 (#13–#15):** `56a13ff` — `test(v4): verify authoritative_instrument admissibility diff with zero unintended admissions`
- **P1 (#4):** `be5604d` — `perf(v4): eliminate superquadratic scaling in corpus reference register`
- **P2 (#5–#9):** `f35af19` — `v4: complete Priority 2 (#5-#9) tabular inheritance, KV lists, omitted condition_status, normalizers, and OCR diff noise` (+ KV nested-heading & scalar typed matching utility fix in P0)
- **P3 (#10–#12):** `32cb7b0` — `v4+v3(P3): deleted clauses, semantic coverage replacing span masking, and local quantifier/scope checks`
- **P0 (#1–#3):** Current HEAD commit (`benchmarks/matched_p0_eval.py`, `benchmarks/matched_p0_summary.json`, `benchmarks/external_eval/evaluate_external.py`, `benchmarks/edgar_heldout_v2_summary.json`, `tests/test_v4_p0_matched_benchmark.py`, and this report)

---

## 1. Summary of Changes by Priority

### Priority 4 (#13–#15) — Hygiene & Prompt Version Lock (`56a13ff`)
- **#13 Admissibility & `authoritative_instrument`:** Regression tests in `tests/test_v4_list_reread_and_textual_diff.py` verify that `authoritative_instrument` admission semantics produce zero unintended admissions outside the explicit `governs_on_conflict` / role contract.
- **#14 README accuracy:** Verified README documentation matches observed runtime behavior and scope boundaries.
- **#15 Prompt version lock:** `STRICT_PROMPT_VERSION == "v4-proposals-7"` and `REREAD_PROMPT_VERSION == "v4-reread-2"` remain completely frozen across all benchmarks and live external evaluations.

### Priority 1 (#4) — Scaling `build_corpus_reference_register` (`be5604d`)
- Eliminated superquadratic scaling (`O(N^2)` scans across corpus units and divergence spans) in `src/claimlayer/v4/corpus_obligations.py`.
- Added `benchmarks/register_scaling_bench.py` and bit-for-bit regression coverage in `tests/test_v4_scaling_and_equivalence.py`.
- On `register_scaling_bench.py` (`build_dense_identical_layer`, `include_divergence=False`), wall-clock time across `50 → 100 → 200 → 400 → 1000 → 2000` admitted units on `HEAD` measures `0.0178s → 0.0295s → 0.0569s → 0.2047s → 0.2717s → 0.5491s` (`ols_log_log_exponent = 0.9585`, down from the reported `492s` at `394` units on `4c818bd`).
- On matched identical-workloads across `50 / 100 / 198` units in `matched_p0_eval.py`: `4c818bd` ran `44.84 ms → 100.78 ms → 258.58 ms` (`exponent = 1.273`), vs `HEAD` `13.15 ms → 24.01 ms → 43.97 ms` (`exponent = 0.877`, `5.88x` faster at 198 units).

### Priority 2 (#5–#9) — Extraction Fidelity & OCR Diff Hygiene (`f35af19` + P0 KV/normalizer utility fix)
- **#5 Audited tabular header inheritance:** Explicit `table_context` + `FieldEvidence("table_header", ...)` provenance checks (`MISSING_HEADER_CITATION`, `TABULAR_SCALE_MISMATCH`, wrong-column rejection) in `src/claimlayer/v4/fidelity.py`.
- **#6 KV / JSON comma & semicolon lists:** `_KV_PAIR_RE`, `_parse_kv_list_items`, and `_verify_kv_json_relational_association` handle nested heading prefixes, comma-containing dates/numbers (e.g. `March 15, 2026`, `$500,000 USD`), and typed boolean/date equivalence (`yes` ≡ `True`, `March 15, 2026` ≡ `2026-03-15`).
- **#7 Omitted `condition_status` context rule:** `EvidenceLayer.check_claim` rejects assertions that drop `condition_status` against explicit `unconditional` / `conditional` ledger entries (`status="not_supported"`, `relation="equal"`, `/condition_status` difference) rather than auto-supporting.
- **#8 Typed versioned normalizers:** `src/claimlayer/v4/normalizers.py` (`NORMALIZER_VERSION = "v4-norm-1"`) provides deterministic `NormalizationTrace` provenance for dates, separator numbers, percentages, and time intervals.
- **#9 PDF/OCR diff noise filtering:** `_normalize_pdf_ocr_with_map` in `src/claimlayer/v4/corpus_obligations.py` normalizes line-wrap hyphenation, soft line wraps, page footers/headers, email-header boundaries, and dash/whitespace noise without masking genuine semantic changes (`must remain` vs `must not remain`).

### Priority 3 (#10–#12) — Coverage Completeness & Local Scope Verification (`32cb7b0`)
- **#10 Deleted clauses:** `_divergence_gaps_for_item` emits explicit `DELETED_CLAUSE` textual gaps for uncovered substantive clauses dropped between primary and secondary documents, while keeping matching clean controls free of false gaps.
- **#11 Eliminating blind span-overlap masking:** `_build_unit_coverage_index`, `_unit_covers_unmatched_clause`, and `_span_quantity_covered` verify semantic/quantity/polarity coverage instead of treating any overlapping unit span as full coverage. Unextracted differences sharing a sentence span with an extracted unit (e.g., `v1` `meridian` Step 11/12 or `v2` `cart_synth` `STEP:6`, plus dense 1-unit-per-sentence tests) are surfaced deterministically.
- **#12 v3 `verify()` local quantifier, restrictor & negation scope:** `src/claimlayer/verifiers/quote_quantity.py` checks conditional `only if` / scope swaps, comparative bounds (`at least` / `not less than` vs `at most` / `not exceed`), universal vs partial/empty quantifiers, exclusive restrictors (`only` / `solely`), and clause-local negation even on numeric assertions, while preserving equivalence between synonymous bounds (`not less than 5 business days` ↔ `at least 5 business days`). `README.md` (`Scope & Verification Boundary`) explicitly documents that v3 `verify()` enforces deterministic structural, lexical, and local scope grounding and not full semantic entailment.

---

## 2. Priority 0 (#1 & #3): Matched-Denominator Before/After & Utility Evaluation

All comparisons below are generated by `benchmarks/matched_p0_eval.py` running identically over both `4c818bd` (pre-remediation baseline) and `HEAD`.

### 2.1 Matched `v1` vs `v2` Corpus Cross-Check (`n = 4` corpora, `59` sentence checks, `13` divergent sections, `10` annotated conflicts)

Why previous reporting had mismatched denominators (`9/54 → 0/9` and `1/3 → 0/2`): `cart_synth` (45 sentence checks) had `4/4` annotated conflict recall in `v1` (`recorded_reads`) so it entered `clean_corpora_ids` (`n = 54` sentences across `{cart_synth, apex, ridgeline}`, `n = 3` corpora), whereas in `v2` (`recorded_reads_v2`) `cart_synth` had `3/4` recall and was excluded from `clean_corpora_ids`, shrinking the clean denominator to `{apex, ridgeline}` (`n = 9` sentences, `n = 2` corpora).

Evaluating **`v1` and `v2` over the exact same subsets**:

| Subset / Metric | `4c818bd` (`v1`) | `4c818bd` (`v2`) | `HEAD` (`v1`) | `HEAD` (`v2`) | `n` | `HEAD` `v2` Wilson 95% CI |
|---|---|---|---|---|---:|---|
| Common clean subset (`apex` + `ridgeline`) false-gap sentences | `0/9` (`0.00%`) | `0/9` (`0.00%`) | `0/9` (`0.00%`) | `0/9` (`0.00%`) | 9 | `[0.00%, 29.91%]` |
| Common clean subset (`apex` + `ridgeline`) escalations | `0/2` (`0.00%`) | `0/2` (`0.00%`) | `0/2` (`0.00%`) | `0/2` (`0.00%`) | 2 | `[0.00%, 65.76%]` |
| All 4 corpora matched: sentence checks with any gap | `12/59` (`20.34%`) | `9/59` (`15.25%`) | `19/59` (`32.20%`) | `10/59` (`16.95%`) | 59 | `[9.48%, 28.46%]` |
| All 4 corpora matched: sentence checks with quantity gap | `6/59` (`10.17%`) | `3/59` (`5.08%`) | `12/59` (`20.34%`) | `4/59` (`6.78%`) | 59 | `[2.67%, 16.18%]` |
| All 4 corpora matched: sentence checks with textual gap | `9/59` (`15.25%`) | `9/59` (`15.25%`) | `16/59` (`27.12%`) | `10/59` (`16.95%`) | 59 | `[9.48%, 28.46%]` |
| All 4 corpora matched: reader conflict recall | `9/10` (`90.00%`) | `8/10` (`80.00%`) | `9/10` (`90.00%`) | `8/10` (`80.00%`) | 10 | `[49.02%, 94.33%]` |
| All 4 corpora matched: corpus escalation rate | `2/4` (`50.00%`) | `2/4` (`50.00%`) | `2/4` (`50.00%`) | `2/4` (`50.00%`) | 4 | `[15.00%, 85.00%]` |
| Leave-one-out ablation detection (`measure_reader_recall.py`) | — | — | `9/10` (`90.00%`) | `10/10` (`100.00%`) | 10 | `[72.25%, 100.00%]` |

*Note on `HEAD` `v1` (`12/59 → 19/59`) and `v2` (`9/59 → 10/59` and `5` quantity gaps vs `4`): eliminating blind span-overlap masking (P3 #11) legitimately exposed previously masked unextracted steps (`meridian` Step 11/12 in `v1`; `cart_synth` `STEP:6` in `v2`). On `v1` ablation, `meridian` Step 11/12 was already unextracted and flagged upfront, so `9/10` leave-one-out removals isolate single conflicts while `10/10` conflicts are surfaced.*

### 2.2 Paired Utility (`PASS` / `supported` / `ACCEPT`) vs Rejection Rates (`4c818bd` vs `HEAD`)

| Battery / Metric | `4c818bd` Before | `HEAD` After | `n` | `HEAD` Wilson 95% CI |
|---|---|---|---:|---|
| P2 positive utility (`admissible` & `check_claim == "supported"`) | `3/5` (`60.00%`) | `5/5` (`100.00%`) | 5 | `[56.55%, 100.00%]` |
| P2 adversarial rejections (fidelity & condition omission) | `4/4` (`100.00%`) | `4/4` (`100.00%`) | 4 | `[51.01%, 100.00%]` |
| P2 OCR clean pairs with zero false gaps | `3/3` (`100.00%`) | `3/3` (`100.00%`) | 3 | `[43.85%, 100.00%]` |
| P2 OCR genuine subtle change detected | `0/1` (`0.00%`) | `1/1` (`100.00%`) | 1 | `[20.65%, 100.00%]` |
| P3 v3 `verify()` scope/quantifier/negation true reject (`RETRY`) | `5/8` (`62.50%`) | `8/8` (`100.00%`) | 8 | `[67.56%, 100.00%]` |
| P3 v3 `verify()` positive utility (`PASS` on valid claims) | `5/6` (`83.33%`) | `6/6` (`100.00%`) | 6 | `[60.97%, 100.00%]` |
| P3 deleted clauses detected | `1/2` (`50.00%`) | `2/2` (`100.00%`) | 2 | `[34.24%, 100.00%]` |
| P3 overlap masking eliminated (`1` unit/sentence masking unextracted conflicts) | `0/2` (`0.00%`) | `2/2` (`100.00%`) | 2 | `[34.24%, 100.00%]` |
| P0 held-out multidomain (`v3_v4` true reject on seeded defects, `0` false accept) | `6/6` (`100.00%`) | `6/6` (`100.00%`) | 6 | `[60.97%, 100.00%]` |
| P0 held-out multidomain (`v3_v4` clean control + repair `ACCEPT` utility) | `2/2` (`100.00%`) | `2/2` (`100.00%`) | 2 | `[34.24%, 100.00%]` |

---

## 3. Priority 0 (#2): Live Held-Out SEC EDGAR Evaluation (`12` New Public Documents, Frozen Prompts)

Executed with `benchmarks/external_eval/evaluate_external.py --mode live --thinking-mode nvidia_off` against `nvidia/nemotron-3-ultra-550b-a55b` (`https://integrate.api.nvidia.com/v1`), freezing `STRICT_PROMPT_VERSION = "v4-proposals-7"` (`prompt_digest = 24c37d7b1d39222609c9dc754aa43d432c39c31f13edd0bab18f6e4f1ee49491`) and `REREAD_PROMPT_VERSION = "v4-reread-2"`.

- **Corpus directory & manifest provenance:** 6 new SEC EDGAR pairs (`12` documents, `max_chars_per_doc = 8000`) stored outside git at `/tmp/claimlayer_edgar_heldout_v2` (`manifest_sha256: 1b96b7b1f5752754d3a429741feb7ca13f9595878d297dd3132186624eb13db7`), covering `424B2`/`424B5` prospectus supplements + `EX-4.1` indentures/officer's certificates (`edgar_apple_2026_notes`, `edgar_apple_2025_notes`, `edgar_oracle_2026_notes`, `edgar_cat_2025_424b5`) and `8-K` `EX-10.1` corporate credit agreements/renewals (`edgar_ibm_3yr_credit`, `edgar_gm_2026_credit`).
- **External human annotations protocol:** Blank annotation sheet lives strictly outside the repository at `/tmp/claimlayer_edgar_heldout_v2_annotations.json` (`annotated_conflicts = 0`, `recall_status = "awaiting_external_human_annotation"`). No self-annotation was performed and prompts were not modified.
- **Observed live summary (`benchmarks/edgar_heldout_v2_summary.json`):**
  - `documents_read_ok`: **`11/12`** (`91.67%`, Wilson 95% CI `[64.61%, 98.51%]`; `edgar_ibm_3yr_credit` had 1 short cover confirmation doc read without units).
  - `units_admitted_total`: **`60`** (`23` Apple 2026, `17` Apple 2025, `11` Oracle 2026, `5` CAT 2025, `2` IBM Credit, `2` GM Credit).
  - **Targeted re-reading (`v4-reread-2`):** `9/32` (`28.12%`, CI `[15.56%, 45.37%]`) reread proposals admitted (`6` Apple 2026, `3` Apple 2025), closing **`8/675`** initial open gaps (`1.19%`, CI `[0.60%, 2.32%]`).
  - **Cross-check & fail-closed escalation across `3,082` sentence checks:** **`667`** open gaps remain (`406` quantity, `261` textual, `0` section) due to multi-tranche/covenant differences exceeding the narrow `v4-proposals-7` schema & `max_proposals=40` cap; fail-closed escalation triggers on **`6/6` corpora (`100.00%`, Wilson 95% CI `[60.97%, 100.00%]`)**, preventing silent false accepts on unseen public filings.
  - **Conflict recall & false-gap specificity:** **`awaiting_external_human_annotation`** (must be scored against independent human labels placed in `/tmp/claimlayer_edgar_heldout_v2_annotations.json`).
