# Delivery Report: Corpus Closure & Reference-Register Hardening (v4 / `v3_v4`)

- **Repository:** `mathmentorAI/deltoryx-gate` (submodule `claim-layer/`)
- **Base commit audited:** `f3bf7e2`
- **Hardening commits (`main`):**
  1. `b5ddd9b` — `v4: reader-built reference register with status, divergence cross-check, and recorded NVIDIA recall baseline`
  2. `d5e1723` — `v4: unify required-section coverage with deliverable graph-leaf and correspondence verification`
  3. `9439284` — `v4: verify governing source direction and value-to-document attribution in conflict disclosures`
  4. `56279a0` — `v4: bound opinion claims with register_status, fact leakage, and premise relevance`
  5. `5797761` — `test(v4): run RepairOrchestrator & repair loop in AC5, fix ReviewResult bug, expand AC9-AC11`

---

## 1. Summary of Changes by Audit Item

### 1.1 Reader-Built Reference Register, Document Status & Divergence Cross-Check
- Removed all manual reference-register ingestion (`attestor="corpus_reference_builder"` now has 0 occurrences across `claim-layer/`). The Reference Register is built through the `EvidenceLayer` configured reader (`build_corpus_reference_register` in `src/claimlayer/v4/corpus_obligations.py:175`).
- Added per-document ingestion status tracking (`read_ok`, `read_failed`, `not_read`, plus captured error records) with explicit propagation instead of `except Exception: pass`. Any authoritative/corpus document not in `read_ok` marks `CorpusReferenceRegister.incomplete = True` and returns `ESCALATE` with `REFERENCE_REGISTER_INCOMPLETE` (`src/claimlayer/v4/corpus_obligations.py`, `src/claimlayer/v4/evaluation.py`).
- Added deterministic cross-check against v3 `build_comparison_policy` section divergences: any divergent section between sources lacking an overlapping reader-derived `origin="corpus"` obligation triggers `REFERENCE_REGISTER_GAP` (`ESCALATE`), preventing silent misses by the reader.
- Added unedited real-model reader recordings (`examples/harness2_securitization/recorded_reads/{cart_synth,meridian,apex,ridgeline}.json`) via `RecordedProposalReader` (`src/claimlayer/v4/reader.py:144`) using NVIDIA NIM (`nvidia/nemotron-3-ultra-550b-a55b`), independent conflict annotations (`benchmarks/conflict_annotations.json`), and measurement script (`benchmarks/measure_reader_recall.py`).
- Improved span matching in `_claim_matches_obligation` (`src/claimlayer/v4/corpus_obligations.py:935`) so matching `(document_id, start, end)` overlap satisfies corpus obligation leaf matching without brittle entity-alias mismatches, while preserving condition-status directionality (`_leaf_matches_admitted_corpus_unit`).

### 1.2 Unified Coverage Between v3 Assertions and v4 Deliverable
- Added `_check_required_section_deliverable_coverage` (`examples/harness2_securitization/harness2_v4_pipeline.py:1241`) wired into `evaluate_securitization_candidate` (`examples/harness2_securitization/harness2_v4_pipeline.py:1373`) in `gate_mode == "v3_v4"` (preserving `v3_only` mode behavior when `gate_mode != "v3_v4"`).
- Retaining a v3 `Assertion` for a required/divergent section (e.g., `waterfall`) while removing the corresponding deliverable text or graph leaves now fails with `RETRY`, `code="MISSING_REQUIRED_SECTION"`, and `reason="ASSERTION_NOT_IN_DELIVERABLE"`.

### 1.3 Conflict Disclosure Content & Value Attribution Checks
- Added `_check_conflict_disclosure_content` and unit-aware token matcher `_value_in_segment_with_unit` (`src/claimlayer/v4/corpus_obligations.py:720`, `:767`), alongside structural document alias derivation from corpus metadata (zero hardcoded fixture/domain word lists) and decimal-safe clause splitting (`(?<!\d)[.,](?!\d)`).
- Naming the lower-ranked source as governing (`GOVERNING_SOURCE_MISMATCH`), falsely declaring an unequal-rank conflict `"unresolved"` (`GOVERNING_SOURCE_MISMATCH`), or swapping conflicting values across source documents (`VALUE_ATTRIBUTION_MISMATCH`) triggers `RETRY` even when `in_graph` is `False` or `True`. Equal-rank `"unresolved"` disclosures remain `ACCEPT`.
- Registered repairable specs and per-obligation deduplication in `src/claimlayer/v4/repair.py:773`, `:789`.

### 1.4 Bounded Opinion Claims (`register_status`, Fact Leakage, Premise Relevance)
- Added `Vocabulary.register_status_terms` (`src/claimlayer/v4/vocabulary.py:386`) included in `Vocabulary.describe()` (`src/claimlayer/v4/vocabulary.py:424`) and therefore in `Vocabulary.fingerprint`. Populated `register_status_terms` and bumped version to `"1.2.0"` on both `SECURITIZATION_VOCABULARY` (`src/claimlayer/v4/domains/securitization.py:559`) and `BANKING_FINANCE_VOCABULARY` (`src/claimlayer/v4/domains/banking_finance.py:22`).
- Added `"register_status"` to `CLAIM_KINDS` and `EPISTEMIC_KIND_RANK` (`"register_status": 5`) (`src/claimlayer/v4/justification.py:26`, `:34`) and synthesized engine-derived `kind="register_status"` claim telemetry in `corpus_coverage` (`src/claimlayer/v4/evaluation.py:1596`).
- Added `_check_register_status_contradiction` (`src/claimlayer/v4/correspondence.py:2473`) and `_audit_opinion_clause_factual_leakage` (`src/claimlayer/v4/correspondence.py:2569`), checking even `_OPINION_MARKER_RE`-marked clauses for:
  - False discrepancy denials matching `Vocabulary.register_status_terms` (`REGISTER_STATUS_CONTRADICTED`, `RETRY`).
  - Subordinate conflict values asserted without governing resolution (`GOVERNING_VALUE_MISMATCH`, `RETRY`).
  - Concrete `Vocabulary` predicate values or non-opinion graph/corpus scalar values carried inside opinion clauses (`OPINION_CARRIES_FACT`, `RETRY`).
- Replaced the permissive `opinion_premise_guard` bypass with `_opinion_premise_is_related` (`src/claimlayer/v4/evaluation.py:68`) requiring every opinion premise (or its non-opinion ancestor lineage) to share subject or content relevance (`OPINION_PREMISE_UNRELATED`, `RETRY`), while still accepting legitimate interpretive opinions (`test_opinion_claim_requires_premises_and_explicit_marker`, `test_ac11_bounded_opinion_and_vocabulary_register_status`).
- Added repair specs for `REGISTER_STATUS_CONTRADICTED`, `OPINION_CARRIES_FACT`, `GOVERNING_VALUE_MISMATCH`, and `OPINION_PREMISE_UNRELATED` (`src/claimlayer/v4/repair.py:327`).

### 1.5 Real Library AC5 Execution & Genuine `ReviewResult` Bugfix
- Rewrote `test_ac5_anti_pruning_across_attempts_with_renamed_predicates_or_agent_id` (`tests/test_v4_corpus_closure_and_anti_evasion.py:358`) to invoke `RepairOrchestrator.run` (`src/claimlayer/v4/repair.py:1424`) and `run_securitization_repair_loop` (`examples/harness2_securitization/harness2_v4_pipeline.py:1980`) directly, removing test-side `seen_covered` / `dropped` tracking.
- Executing `RepairOrchestrator.run` directly exposed and resolved a live `TypeError` in `ReviewResult(...)` construction inside `CORPUS_OBLIGATION_DROPPED` handling (`src/claimlayer/v4/repair.py:1497`, missing `question`, `candidate_response`, and `unresolved_obligations`).

---

## 2. Reader Recall & `REFERENCE_REGISTER_GAP` Measurement (NVIDIA NIM)

Measured with `benchmarks/measure_reader_recall.py --mode recorded` (replaying verbatim unedited outputs from `nvidia/nemotron-3-ultra-550b-a55b`, `prompt_version="v4-proposals-3"`, `prompt_digest="5736166d93ad3ec377deea4111f9bd67e578cb6a78f445517cce5f4c4dc0a1b3"`, `fragment_chars=1800`, recorded `2026-11-15T14:40:13.194859+00:00`):

| Corpus | Case ID | Docs `read_ok` | Units Admitted | Conflict Recall | Divergent Sections | `REFERENCE_REGISTER_GAP` | Gap Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| `cart_synth` | `cart_2024_2_synthetic_replay` | 3 / 3 | 24 | 4 / 4 (`1.00`) | 8 | 0 | `0.00` |
| `meridian` | `heldout_sec_meridian_2026` | 2 / 2 | 5 | 1 / 2 (`0.50`) | 1 | 0 | `0.00` |
| `apex` | `heldout_bank_apex_2026` | 5 / 5 | 8 | 2 / 2 (`1.00`) | 4 | 1 | `0.25` |
| `ridgeline` | `banking_finance_synthetic` | 5 / 5 | 9 | 2 / 2 (`1.00`) | 4 | 0 | `0.00` |
| **Total** | — | **15 / 15** | **46** | **9 / 10 (`0.90`)** | **17** | **1** | **`0.0588`** |

- Notice that in `meridian`, the model missed `meridian_reserve_waterfall_step`, and in `apex`, the deterministic v3 comparison cross-check surfaced `1` `REFERENCE_REGISTER_GAP` (`0.25`) out of `4` divergent sections, preventing silent acceptance when a recorded reader misses a divergent section.

---

## 3. Acceptance Criteria Test Matrix (`tests/test_v4_corpus_closure_and_anti_evasion.py`)

| Test Function | Coverage |
|---|---|
| `test_ac1_omitted_conflict_returns_retry_with_corpus_obligation_uncovered` (`:270`) | Omitted corpus conflict -> `RETRY` + `CORPUS_OBLIGATION_UNCOVERED` across `metadata` settings |
| `test_ac2_disclosed_conflict_with_both_values_and_governing_source_accepts` (`:308`) | Disclosed conflict with both values & governing source -> `ACCEPT` |
| `test_ac3_long_report_unsupported_line_without_keywords_returns_retry` (`:319`) | >1,200-char report unsupported keyword-free line -> `RETRY` + `UNSUPPORTED_FACTUAL_ADDITION` |
| `test_ac4_missing_graph_never_accepts` (`:338`) | Missing `JustificationGraph` never returns `ACCEPT` (`require_graph=True`) |
| `test_ac5_anti_pruning_across_attempts_with_renamed_predicates_or_agent_id` (`:358`) | Library `RepairOrchestrator.run` + `run_securitization_repair_loop` anti-pruning (`CORPUS_OBLIGATION_DROPPED` -> `RETRY` -> `ESCALATE`) |
| `test_ac6_prompt_injection_in_corpus_cannot_pass` (`:504`) | Corpus prompt injection cannot suppress conflict obligation |
| `test_ac7_multi_quantity_distinct_dimensions_source_sentence_accepts` (`:518`) | Multi-quantity source sentence (principal + coupon) accepts without false rejection |
| `test_ac8_missing_reference_register_escalates_with_reference_register_missing` (`:536`) | Empty/missing reference register -> `ESCALATE` (`REFERENCE_REGISTER_MISSING`) |
| `test_explicit_rejected_obligation_and_signed_receipt_telemetry` (`:562`) | Explicit obligation rejection -> `ESCALATE`; signed receipt telemetry |
| `test_opinion_claim_requires_premises_and_explicit_marker` (`:599`) | Valid interpretive `opinion` claim with aligned premises & marker -> `ACCEPT` |
| `test_ac9_reader_incomplete_document_escalates_without_silencing_errors` (`:633`) | `read_failed` corpus document -> `incomplete=True` -> `ESCALATE` (`REFERENCE_REGISTER_INCOMPLETE`) |
| `test_ac10_recorded_reader_gap_detected_on_meridian_and_apex` (`:704`) | Unedited NIM `apex` recording + omitted-conflict `meridian` slice -> `ESCALATE` (`REFERENCE_REGISTER_GAP`) |
| `test_ac11_v3_assertion_without_deliverable_section_or_graph_returns_retry` (`:785`) | v3 `Assertion` kept while deliverable/graph omits section -> `RETRY` (`MISSING_REQUIRED_SECTION`, `ASSERTION_NOT_IN_DELIVERABLE`) |
| `test_ac11_conflict_disclosure_content_and_equal_rank` (`:870`) | Inverted governing source (`GOVERNING_SOURCE_MISMATCH`), swapped values (`VALUE_ATTRIBUTION_MISMATCH`), false unequal-rank unresolved (`GOVERNING_SOURCE_MISMATCH`), equal-rank unresolved (`ACCEPT`) |
| `test_ac11_bounded_opinion_and_vocabulary_register_status` (`:914`) | `REGISTER_STATUS_CONTRADICTED`, `GOVERNING_VALUE_MISMATCH`, `OPINION_CARRIES_FACT`, `OPINION_PREMISE_UNRELATED`, `register_status` claim, and `Vocabulary.fingerprint` sensitivity |

---

## 4. Verification Results & Known Limitations

- **Test runs:** Verified post-edit on `.venv/bin/pytest claim-layer/tests/test_v4_corpus_closure_and_anti_evasion.py -v` (`21 passed in 0.87s`), `.venv/bin/pytest claim-layer/tests/ -q` (`413 passed, 1 skipped in 9.34s`), and `.venv/bin/pytest tests/ -q` (`3 passed in 0.28s`). (Re-run once more after committing this document.)
- **Known limitations:**
  1. `register_status_terms` defaults to `()` on custom `Vocabulary` instances unless explicitly declared (populated on `SECURITIZATION_VOCABULARY` and `BANKING_FINANCE_VOCABULARY`); custom vocabularies that do not declare `register_status_terms` still reject ungrounded denials via `UNSUPPORTED_FACTUAL_ADDITION`, not `REGISTER_STATUS_CONTRADICTED`.
  2. Equal-rank `"unresolved"` conflict disclosure tests rely on phrasing supported by the correspondence grounding lexicon (e.g., `"the conflict is unresolved"` rather than ungrounded verbs outside evidence vocabulary).
  3. Live NIM recall measurement was recorded against the 4 benchmark/heldout corpora (`10` annotated conflicts, `17` divergent sections); larger production corpora will depend on domain `Vocabulary` completeness and `build_comparison_policy` section divergence coverage.
