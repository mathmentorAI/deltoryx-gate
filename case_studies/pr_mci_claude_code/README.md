# PR-MCI Case Study: Auditable Verification of Coding-Agent PR Descriptions (`deltoryx-gate`)

This case study evaluates deterministic (`verify()`) and LLM-assisted (`Gate v4` + `verify_draft()`) verification of coding-agent pull request descriptions against their actual git diffs using the [PR-MCI](https://github.com/gjz78910/PR-MCI) replication benchmark (Gong, Pinna, Bian, Zhang — MSR '26, arXiv:2601.04886; dataset pinned to commit `ac6bbf4`).

> **Headline finding (blind held-out Phase 2, `n = 528`):**
> Gate v4 with `code_change` v2.0 + Nemotron does **not** improve inconsistency detection over simpler deterministic rules or the PR-MCI baseline: on `partial | misaligned`, Gate v4 scores F1 `0.4039` (vs `rules_v1` `0.4370`, `rules_v2_legal_prose` `0.4188`, `rules_v2_postreview_code_config` `0.4127`), and on strict `misaligned` the paper's lightweight heuristic leads (`F1 = 0.6377` vs Gate v4 `0.4463` and `rules_v2_postreview_code_config` `0.4486`). What the system demonstrates is **auditable, fail-closed triage**: across `3,237` extracted agent claims in Phase 2, **`851` (`26.29%`)** are literally checkable against the diff (`453` supported, `389` insufficient evidence, `9` contradicted, each with deterministic SHA-256 receipts), while **`2,386` (`73.71%`)** high-level behavioral descriptions are explicitly isolated as `unverifiable` (`describes_behavior`) rather than silently firing false alarms—and any truncated diff registry fails closed as `ESCALATE` (never `CONSISTENT`).

---

## 1. Study Protocol and Contamination Discipline

The 600 manually annotated PRs in `validation/validation_results.csv` are strictly split once and never re-tuned across phases:

1. **Development set (Phase 1, `Claude_Code`, `n = 72`):** `59 aligned`, `10 partial`, `3 misaligned` (`13` broad positives, `3` strict positives). Used exclusively to inspect failure modes, design `rules_v2`, diagnose library bugs (`00adfab`, `c526fb2`), and freeze the `code_change` v2.0 schema before touching Phase 2.
2. **Blind held-out test set (Phase 2, `n = 528`):** Remaining validation agents (`OpenAI_Codex` `239`, `Copilot` `118`, `Devin` `96`, `Cursor` `75`) totaling `407 aligned`, `82 partial`, `39 misaligned` (`121` broad positives, `39` strict positives). Evaluated without tuning prompts, rules, thresholds, or filters.
3. **Final packaging & bug-fix policy:** In this release, no rule, prompt, vocabulary, threshold, or claim filter was modified. Only engineering bugs identified during audit were fixed—`O(N^2)` multi-cardinality resolution and fail-closed `error_escalated` handling in `engine.py` (`08c598c`), property/alias forwarding in `autovocab.with_entities()` (`22e3590`), November 16 `v3` verifier provenance/table fixes committed cleanly with tests (`f42c7b3`), and Task 1 script bug fixes published separately under *"corrected after review"* while keeping pre-registered `check_claims_v1.py` and Nemotron `v1` results intact.

All metrics report both label definitions used by PR-MCI:
- **Broad (`partial | misaligned`)**: both `partial` and `misaligned` count as positive (`aligned` is negative).
- **Strict (`misaligned`)**: only `misaligned` counts as positive (`aligned` and `partial` are negative).
- Confidence intervals are **95% Wilson score intervals** (`z = 1.96`).

---

## 2. Main Result: Blind Held-Out Phase 2 (`n = 528`)

**Library commit:** `22e3590` (`deltoryx-gate` `main`; rules use `verify()` with `c526fb2`+`f42c7b3`; Gate v4 uses `00adfab`+`08c598c`+`22e3590`).
**Dataset commit:** `PR-MCI` `ac6bbf4` (`n = 528`: `407 aligned`, `82 partial`, `39 misaligned`; `528/528` evaluated cleanly, `0` errors).
**Frozen vocabulary fingerprint (`code_change` v2.0):** `620ebd71c7dda399cb8b90e5ac038a2a2cf510782a86202b79b75c8d84900d2c`.

### 2.1 Broad Definition: `partial | misaligned` (`pos = 121`, `neg = 407`, Gate Commit `22e3590`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_heuristic` | PR-MCI baseline | 26 | 4 | 95 | 403 | 0.8667 [0.7032, 0.9469] | 0.2149 [0.1511, 0.2962] | 0.3444 | 0.0098 [0.0038, 0.0250] |
| `paper_embedding` | PR-MCI baseline | 121 | 405 | 0 | 2 | 0.2300 [0.1961, 0.2679] | 1.0000 [0.9692, 1.0000] | 0.3740 | 0.9951 [0.9823, 0.9987] |
| `paper_both` | PR-MCI baseline | 20 | 4 | 101 | 403 | 0.8333 [0.6415, 0.9332] | 0.1653 [0.1096, 0.2416] | 0.2759 | 0.0098 [0.0038, 0.0250] |
| `rules_v1_preregistered` | Pre-registered | 52 | 65 | 69 | 342 | 0.4444 [0.3576, 0.5348] | 0.4298 [0.3450, 0.5188] | **0.4370** | 0.1597 [0.1273, 0.1985] |
| `rules_v2_legal_prose` | Frozen from P1 | 40 | 30 | 81 | 377 | 0.5714 [0.4548, 0.6806] | 0.3306 [0.2531, 0.4185] | 0.4188 | 0.0737 [0.0521, 0.1033] |
| `rules_v2_code_config` | Explicit profile | 39 | 31 | 82 | 376 | 0.5571 [0.4408, 0.6675] | 0.3223 [0.2456, 0.4099] | 0.4084 | 0.0762 [0.0542, 0.1061] |
| `rules_v2_postreview_legal_prose` | Corrected after review | 40 | 30 | 81 | 377 | 0.5714 [0.4548, 0.6806] | 0.3306 [0.2531, 0.4185] | 0.4188 | 0.0737 [0.0521, 0.1033] |
| `rules_v2_postreview_code_config` | Corrected after review | 39 | 29 | 82 | 378 | 0.5735 [0.4552, 0.6840] | 0.3223 [0.2456, 0.4099] | 0.4127 | 0.0713 [0.0501, 0.1005] |
| `gate_v4_vocab2` | Frozen from P1 | 41 | 41 | 80 | 366 | 0.5000 [0.3942, 0.6058] | 0.3388 [0.2606, 0.4270] | 0.4039 | 0.1007 [0.0751, 0.1338] |

Note: `rules_v1_postreview_parse` is identical in all aggregate counts (`TP=52, FP=65, FN=69, TN=342, F1=0.4370`).

### 2.2 Strict Definition: `misaligned` only (`pos = 39`, `neg = 489`, Gate Commit `22e3590`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_heuristic` | PR-MCI baseline | 22 | 8 | 17 | 481 | 0.7333 [0.5555, 0.8582] | 0.5641 [0.4098, 0.7070] | **0.6377** | 0.0164 [0.0083, 0.0319] |
| `paper_embedding` | PR-MCI baseline | 39 | 487 | 0 | 2 | 0.0741 [0.0547, 0.0998] | 1.0000 [0.9103, 1.0000] | 0.1381 | 0.9959 [0.9852, 0.9989] |
| `paper_both` | PR-MCI baseline | 18 | 6 | 21 | 483 | 0.7500 [0.5510, 0.8800] | 0.4615 [0.3157, 0.6143] | 0.5714 | 0.0123 [0.0056, 0.0265] |
| `rules_v1_preregistered` | Pre-registered | 26 | 91 | 13 | 398 | 0.2222 [0.1564, 0.3057] | 0.6667 [0.5098, 0.7937] | 0.3333 | 0.1861 [0.1541, 0.2230] |
| `rules_v2_legal_prose` | Frozen from P1 | 24 | 46 | 15 | 443 | 0.3429 [0.2425, 0.4596] | 0.6154 [0.4590, 0.7511] | 0.4404 | 0.0941 [0.0713, 0.1232] |
| `rules_v2_code_config` | Explicit profile | 24 | 46 | 15 | 443 | 0.3429 [0.2425, 0.4596] | 0.6154 [0.4590, 0.7511] | 0.4404 | 0.0941 [0.0713, 0.1232] |
| `rules_v2_postreview_legal_prose` | Corrected after review | 24 | 46 | 15 | 443 | 0.3429 [0.2425, 0.4596] | 0.6154 [0.4590, 0.7511] | 0.4404 | 0.0941 [0.0713, 0.1232] |
| `rules_v2_postreview_code_config` | Corrected after review | 24 | 44 | 15 | 445 | 0.3529 [0.2500, 0.4716] | 0.6154 [0.4590, 0.7511] | 0.4486 | 0.0900 [0.0677, 0.1186] |
| `gate_v4_vocab2` | Frozen from P1 | 27 | 55 | 12 | 434 | 0.3293 [0.2372, 0.4366] | 0.6923 [0.5358, 0.8143] | 0.4463 | 0.1125 [0.0874, 0.1436] |

### 2.3 Phase 2 Per-Agent Breakdown (`n = 528`, Gate Commit `22e3590`)

Distribution: `OpenAI_Codex` (`239`: `220/14/5`), `Copilot` (`118`: `69/22/27`), `Devin` (`96`: `64/25/7`), `Cursor` (`75`: `54/21/0`).

#### `partial | misaligned` by agent

| Agent (pos/neg) | Method | TP | FP | FN | TN | P [95% CI] | R [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OpenAI_Codex` (19/220) | `paper_heuristic` | 3 | 2 | 16 | 218 | 0.6000 [0.2307, 0.8824] | 0.1579 [0.0552, 0.3757] | 0.2500 | 0.0091 [0.0025, 0.0325] |
| `OpenAI_Codex` (19/220) | `rules_v1_preregistered` | 7 | 17 | 12 | 203 | 0.2917 [0.1491, 0.4917] | 0.3684 [0.1915, 0.5896] | **0.3256** | 0.0773 [0.0488, 0.1203] |
| `OpenAI_Codex` (19/220) | `rules_v2_legal_prose` | 4 | 3 | 15 | 217 | 0.5714 [0.2505, 0.8418] | 0.2105 [0.0851, 0.4333] | 0.3077 | 0.0136 [0.0046, 0.0393] |
| `OpenAI_Codex` (19/220) | `rules_v2_postreview_code_config` | 3 | 3 | 16 | 217 | 0.5000 [0.1876, 0.8124] | 0.1579 [0.0552, 0.3757] | 0.2400 | 0.0136 [0.0046, 0.0393] |
| `OpenAI_Codex` (19/220) | `gate_v4_vocab2` | 4 | 2 | 15 | 218 | 0.6667 [0.3000, 0.9032] | 0.2105 [0.0851, 0.4333] | 0.3200 | 0.0091 [0.0025, 0.0325] |
| `Copilot` (49/69) | `paper_heuristic` | 19 | 1 | 30 | 68 | 0.9500 [0.7639, 0.9911] | 0.3878 [0.2643, 0.5275] | 0.5507 | 0.0145 [0.0026, 0.0776] |
| `Copilot` (49/69) | `rules_v1_preregistered` | 30 | 17 | 19 | 52 | 0.6383 [0.4954, 0.7603] | 0.6122 [0.4725, 0.7357] | 0.6250 | 0.2464 [0.1599, 0.3596] |
| `Copilot` (49/69) | `rules_v2_legal_prose` | 29 | 8 | 20 | 61 | 0.7838 [0.6280, 0.8861] | 0.5918 [0.4525, 0.7178] | 0.6744 | 0.1159 [0.0599, 0.2125] |
| `Copilot` (49/69) | `rules_v2_postreview_code_config` | 29 | 7 | 20 | 62 | 0.8056 [0.6497, 0.9025] | 0.5918 [0.4525, 0.7178] | **0.6824** | 0.1014 [0.0500, 0.1949] |
| `Copilot` (49/69) | `gate_v4_vocab2` | 28 | 25 | 21 | 44 | 0.5283 [0.3966, 0.6562] | 0.5714 [0.4327, 0.6998] | 0.5490 | 0.3623 [0.2590, 0.4802] |
| `Devin` (32/64) | `paper_heuristic` | 0 | 0 | 32 | 64 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.1072] | 0.0000 | 0.0000 [0.0000, 0.0566] |
| `Devin` (32/64) | `rules_v1_preregistered` | 5 | 16 | 27 | 48 | 0.2381 [0.1063, 0.4509] | 0.1562 [0.0686, 0.3175] | 0.1887 | 0.2500 [0.1601, 0.3682] |
| `Devin` (32/64) | `rules_v2_legal_prose` | 4 | 12 | 28 | 52 | 0.2500 [0.1018, 0.4950] | 0.1250 [0.0497, 0.2807] | 0.1667 | 0.1875 [0.1106, 0.2997] |
| `Devin` (32/64) | `rules_v2_postreview_code_config` | 4 | 12 | 28 | 52 | 0.2500 [0.1018, 0.4950] | 0.1250 [0.0497, 0.2807] | 0.1667 | 0.1875 [0.1106, 0.2997] |
| `Devin` (32/64) | `gate_v4_vocab2` | 8 | 11 | 24 | 53 | 0.4211 [0.2314, 0.6372] | 0.2500 [0.1325, 0.4211] | **0.3137** | 0.1719 [0.0988, 0.2821] |
| `Cursor` (21/54) | `paper_heuristic` | 4 | 1 | 17 | 53 | 0.8000 [0.3755, 0.9638] | 0.1905 [0.0767, 0.4000] | 0.3077 | 0.0185 [0.0033, 0.0977] |
| `Cursor` (21/54) | `rules_v1_preregistered` | 10 | 15 | 11 | 39 | 0.4000 [0.2340, 0.5926] | 0.4762 [0.2834, 0.6763] | **0.4348** | 0.2778 [0.1762, 0.4089] |
| `Cursor` (21/54) | `rules_v2_legal_prose` | 3 | 7 | 18 | 47 | 0.3000 [0.1078, 0.6032] | 0.1429 [0.0498, 0.3464] | 0.1935 | 0.1296 [0.0642, 0.2442] |
| `Cursor` (21/54) | `rules_v2_postreview_code_config` | 3 | 7 | 18 | 47 | 0.3000 [0.1078, 0.6032] | 0.1429 [0.0498, 0.3464] | 0.1935 | 0.1296 [0.0642, 0.2442] |
| `Cursor` (21/54) | `gate_v4_vocab2` | 1 | 3 | 20 | 51 | 0.2500 [0.0456, 0.6994] | 0.0476 [0.0085, 0.2267] | 0.0800 | 0.0556 [0.0191, 0.1511] |

#### Strict `misaligned` by agent

| Agent (pos/neg) | Method | TP | FP | FN | TN | P [95% CI] | R [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `OpenAI_Codex` (5/234) | `paper_heuristic` | 3 | 2 | 2 | 232 | 0.6000 [0.2307, 0.8824] | 0.6000 [0.2307, 0.8824] | **0.6000** | 0.0085 [0.0023, 0.0306] |
| `OpenAI_Codex` (5/234) | `rules_v1_preregistered` | 4 | 20 | 1 | 214 | 0.1667 [0.0668, 0.3585] | 0.8000 [0.3755, 0.9638] | 0.2759 | 0.0855 [0.0560, 0.1283] |
| `OpenAI_Codex` (5/234) | `rules_v2_legal_prose` | 3 | 4 | 2 | 230 | 0.4286 [0.1582, 0.7495] | 0.6000 [0.2307, 0.8824] | 0.5000 | 0.0171 [0.0067, 0.0431] |
| `OpenAI_Codex` (5/234) | `rules_v2_postreview_code_config` | 3 | 3 | 2 | 231 | 0.5000 [0.1876, 0.8124] | 0.6000 [0.2307, 0.8824] | 0.5455 | 0.0128 [0.0044, 0.0370] |
| `OpenAI_Codex` (5/234) | `gate_v4_vocab2` | 3 | 3 | 2 | 231 | 0.5000 [0.1876, 0.8124] | 0.6000 [0.2307, 0.8824] | 0.5455 | 0.0128 [0.0044, 0.0370] |
| `Copilot` (27/91) | `paper_heuristic` | 19 | 1 | 8 | 90 | 0.9500 [0.7639, 0.9911] | 0.7037 [0.5152, 0.8415] | **0.8085** | 0.0110 [0.0019, 0.0597] |
| `Copilot` (27/91) | `rules_v1_preregistered` | 21 | 26 | 6 | 65 | 0.4468 [0.3141, 0.5875] | 0.7778 [0.5924, 0.8939] | 0.5676 | 0.2857 [0.2031, 0.3857] |
| `Copilot` (27/91) | `rules_v2_legal_prose` | 20 | 17 | 7 | 74 | 0.5405 [0.3838, 0.6896] | 0.7407 [0.5532, 0.8683] | 0.6250 | 0.1868 [0.1200, 0.2790] |
| `Copilot` (27/91) | `rules_v2_postreview_code_config` | 20 | 16 | 7 | 75 | 0.5556 [0.3958, 0.7046] | 0.7407 [0.5532, 0.8683] | 0.6349 | 0.1758 [0.1112, 0.2667] |
| `Copilot` (27/91) | `gate_v4_vocab2` | 21 | 32 | 6 | 59 | 0.3962 [0.2759, 0.5306] | 0.7778 [0.5924, 0.8939] | 0.5250 | 0.3516 [0.2614, 0.4539] |
| `Devin` (7/89) | `paper_heuristic` | 0 | 0 | 7 | 89 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.3543] | 0.0000 | 0.0000 [0.0000, 0.0414] |
| `Devin` (7/89) | `rules_v1_preregistered` | 1 | 20 | 6 | 69 | 0.0476 [0.0085, 0.2267] | 0.1429 [0.0257, 0.5131] | 0.0714 | 0.2247 [0.1504, 0.3218] |
| `Devin` (7/89) | `rules_v2_legal_prose` | 1 | 15 | 6 | 74 | 0.0625 [0.0111, 0.2833] | 0.1429 [0.0257, 0.5131] | 0.0870 | 0.1685 [0.1049, 0.2596] |
| `Devin` (7/89) | `rules_v2_postreview_code_config` | 1 | 15 | 6 | 74 | 0.0625 [0.0111, 0.2833] | 0.1429 [0.0257, 0.5131] | 0.0870 | 0.1685 [0.1049, 0.2596] |
| `Devin` (7/89) | `gate_v4_vocab2` | 3 | 16 | 4 | 73 | 0.1579 [0.0552, 0.3757] | 0.4286 [0.1582, 0.7495] | **0.2308** | 0.1798 [0.1138, 0.2723] |
| `Cursor` (0/75) | `paper_heuristic` | 0 | 5 | 0 | 70 | 0.0000 [0.0000, 0.4345] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0667 [0.0288, 0.1468] |
| `Cursor` (0/75) | `rules_v1_preregistered` | 0 | 25 | 0 | 50 | 0.0000 [0.0000, 0.1332] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.3333 [0.2371, 0.4458] |
| `Cursor` (0/75) | `rules_v2_legal_prose` | 0 | 10 | 0 | 65 | 0.0000 [0.0000, 0.2775] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.1333 [0.0741, 0.2283] |
| `Cursor` (0/75) | `rules_v2_postreview_code_config` | 0 | 10 | 0 | 65 | 0.0000 [0.0000, 0.2775] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.1333 [0.0741, 0.2283] |
| `Cursor` (0/75) | `gate_v4_vocab2` | 0 | 4 | 0 | 71 | 0.0000 [0.0000, 0.4899] | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0533 [0.0209, 0.1293] |

### 2.4 Phase 2 Triage Breakdown & Literal Coverage (`gate_v4_vocab2`, `n = 528`, Gate Commit `22e3590`)

- **Claim disposition totals (`3,237` admitted proposals):** `supported: 453`, `insufficient_evidence: 389`, `contradicted: 9`, `unverifiable: 2,386` (`describes_behavior` only); `escalated: 0`, `truncated: 0`.
- **Literal-verifiability coverage:** **`851 / 3,237 (26.29%)`** of agent claims fall under literally checkable code/diff predicates (`453` supported = `13.99%`, `389` insufficient_evidence = `12.02%`, `9` contradicted = `0.28%`), while **`2,386 / 3,237 (73.71%)`** are high-level behavioral descriptions explicitly routed to `describes_behavior` (`unverifiable`, `0` unsupported alerts triggered).
- **By predicate (`supported / insufficient_evidence / contradicted`):**
  - `changes_file`: `273 / 82 / 0`
  - `adds_identifier`: `56 / 116 / 0`
  - `adds_test`: `47 / 97 / 0`
  - `modifies_identifier`: `36 / 51 / 0`
  - `changes_dependency`: `19 / 17 / 0`
  - `removes_identifier`: `15 / 21 / 0`
  - `removes_test`: `0 / 5 / 0`
  - `files_changed_count`: `2 / 0 / 3`
  - `lines_added_count`: `2 / 0 / 4`
  - `lines_deleted_count`: `3 / 0 / 2`
  - `describes_behavior`: `2,386` marked `unverifiable` (`0` unsupported alerts triggered)
- **Draft reader rejections (excluded before engine scoring):** `reader_schema: 45`, `unknown_or_foreign_evidence_id: 22`, `assertion_kind_not_declared_for_predicate: 11`.

### 2.5 Recovery of 3 Formerly Crashing Large Phase 2 PRs (`08c598c`, Uncapped)

Before commit `08c598c`, `engine.ask()` and `_decide()` materialized quadratic `competitors` cross-products across `cardinality="multi"` predicates, crashing with `DataError`/`OverflowError` on 3 large Phase 2 PRs (leaving `525/528` evaluated). Fixing `_decide()` to bucket `multi` claims by exact equal key with opposite polarity (`O(N)` time/memory) and removing silent `500`/`300` diff-fact caps resolved all 3 PRs cleanly from the existing replay cache (`528/528`, `0` errors, `truncated: False`):

| `pr_id` | Agent | Gold Label | Uncapped Diff Facts | Gate v4 Verdict (`22e3590`) | Claim Status Counts | Receipt Prefix |
|---|---|---|---|---|---|---|
| `3111511868` | `Copilot` | `aligned` | 346 files, 103 mod IDs, 17 add IDs, 12 rem IDs, 6 deps, 3 add tests (`truncated: False`) | `CONSISTENT` | `supported: 8`, `insufficient_evidence: 3`, `unverifiable: 7` | `aa56e703df43` |
| `3152257828` | `Copilot` | `misaligned` | 324 files, 5,079 mod IDs, 27 add IDs, 5 rem IDs, 6 deps, 84 add tests (`truncated: False`) | `CONSISTENT` | `supported: 3`, `unverifiable: 2` | `e246a7a9b69f` |
| `3158683531` | `Devin` | `partial` | 542 files, 1,333 add IDs, 1,191 mod IDs, 474 rem IDs, 68 deps, 335 add tests (`truncated: False`) | `CONSISTENT` | `supported: 1`, `insufficient_evidence: 1`, `unverifiable: 10` | `f7266c8467d0` |

Note also on `22e3590`: replacing the temporary `verify_draft_closed()` workaround in `eval_v2vocab.py` with official `claimlayer.v4.triage.verify_draft(guard=False)` (after fixing `autovocab.with_entities()` to preserve `closed_entities`, `require_explicit_condition_status`, and entity aliases) produced **zero verdict or metric changes** across all `72` Phase 1 PRs and all `528` Phase 2 PRs.

---

## 3. Development Set Results: Phase 1 (`Claude_Code`, `n = 72`)

Distribution: `59 aligned`, `10 partial`, `3 misaligned` (`pos = 13` broad, `3` strict). Pre-registered baselines are reported strictly separately from adjusted / post-review runs.

> **Note on Gate commit provenance:** The initial live Nemotron `v1` cache (`prmci_llm_cache.jsonl`, 82 calls) and early development runs were recorded against baseline commit `f72098a`. Non-determinism across `70/72` vs `2/72` replay runs was fixed in `00adfab`, explicit `code_config` profile support was added in `c526fb2`, and all unified tables below were regenerated at commit `22e3590` using offline replay against those recorded completions.

### 3.1 Phase 1 Rules and Baselines (Pre-Registered vs Post-Design vs Corrected After Review, Gate Commit `22e3590`)

#### Broad: `partial | misaligned` (`pos = 13`, `neg = 59`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_heuristic` | PR-MCI baseline | 1 | 0 | 12 | 59 | 1.0000 [0.2065, 1.0000] | 0.0769 [0.0137, 0.3331] | 0.1429 | 0.0000 [0.0000, 0.0611] |
| `paper_embedding` | PR-MCI baseline | 13 | 59 | 0 | 0 | 0.1806 [0.1087, 0.2848] | 1.0000 [0.7719, 1.0000] | 0.3059 | 1.0000 [0.9389, 1.0000] |
| `paper_both` | PR-MCI baseline | 1 | 0 | 12 | 59 | 1.0000 [0.2065, 1.0000] | 0.0769 [0.0137, 0.3331] | 0.1429 | 0.0000 [0.0000, 0.0611] |
| `rules_v1_preregistered` | Pre-registered | 6 | 13 | 7 | 46 | 0.3158 [0.1536, 0.5399] | 0.4615 [0.2321, 0.7086] | 0.3750 | 0.2203 [0.1335, 0.3413] |
| `rules_v1_postreview_parse` | Corrected after review | 6 | 13 | 7 | 46 | 0.3158 [0.1536, 0.5399] | 0.4615 [0.2321, 0.7086] | 0.3750 | 0.2203 [0.1335, 0.3413] |
| `rules_v2_legal_prose` | Adjusted on P1 | 7 | 6 | 6 | 53 | 0.5385 [0.2914, 0.7679] | 0.5385 [0.2914, 0.7679] | **0.5385** | 0.1017 [0.0474, 0.2046] |
| `rules_v2_code_config` | Adjusted + `code_config` | 7 | 7 | 6 | 52 | 0.5000 [0.2680, 0.7320] | 0.5385 [0.2914, 0.7679] | 0.5185 | 0.1186 [0.0587, 0.2252] |
| `rules_v2_postreview_legal_prose` | Corrected after review | 7 | 6 | 6 | 53 | 0.5385 [0.2914, 0.7679] | 0.5385 [0.2914, 0.7679] | **0.5385** | 0.1017 [0.0474, 0.2046] |
| `rules_v2_postreview_code_config` | Corrected after review | 7 | 6 | 6 | 53 | 0.5385 [0.2914, 0.7679] | 0.5385 [0.2914, 0.7679] | **0.5385** | 0.1017 [0.0474, 0.2046] |

#### Strict: `misaligned` (`pos = 3`, `neg = 69`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_heuristic` | PR-MCI baseline | 1 | 0 | 2 | 69 | 1.0000 [0.2065, 1.0000] | 0.3333 [0.0615, 0.7923] | **0.5000** | 0.0000 [0.0000, 0.0527] |
| `paper_embedding` | PR-MCI baseline | 3 | 69 | 0 | 0 | 0.0417 [0.0143, 0.1155] | 1.0000 [0.4385, 1.0000] | 0.0800 | 1.0000 [0.9473, 1.0000] |
| `paper_both` | PR-MCI baseline | 1 | 0 | 2 | 69 | 1.0000 [0.2065, 1.0000] | 0.3333 [0.0615, 0.7923] | **0.5000** | 0.0000 [0.0000, 0.0527] |
| `rules_v1_preregistered` | Pre-registered | 2 | 17 | 1 | 52 | 0.1053 [0.0294, 0.3139] | 0.6667 [0.2077, 0.9385] | 0.1818 | 0.2464 [0.1599, 0.3596] |
| `rules_v2_legal_prose` | Adjusted on P1 | 2 | 11 | 1 | 58 | 0.1538 [0.0433, 0.4224] | 0.6667 [0.2077, 0.9385] | 0.2500 | 0.1594 [0.0914, 0.2633] |
| `rules_v2_code_config` | Adjusted + `code_config` | 2 | 12 | 1 | 57 | 0.1429 [0.0401, 0.3994] | 0.6667 [0.2077, 0.9385] | 0.2353 | 0.1739 [0.1024, 0.2798] |
| `rules_v2_postreview_legal_prose` | Corrected after review | 2 | 11 | 1 | 58 | 0.1538 [0.0433, 0.4224] | 0.6667 [0.2077, 0.9385] | 0.2500 | 0.1594 [0.0914, 0.2633] |
| `rules_v2_postreview_code_config` | Corrected after review | 2 | 11 | 1 | 58 | 0.1538 [0.0433, 0.4224] | 0.6667 [0.2077, 0.9385] | 0.2500 | 0.1594 [0.0914, 0.2633] |

### 3.2 Phase 1 Gate v4 + Nemotron Results (`n = 72`, `code_change` v1.0 vs v2.0, Gate Commit `22e3590`)

> **Caveat:** Phase 1 LLM caches (`prmci_llm_cache.jsonl` and `prmci_llm_v2vocab_phase1_cache.jsonl`) were recorded against original `parse.section()` (`legacy=True`); `reproduce.py` preserves `legacy_section=True` during Phase 1 replay so cached prompt hashes match identically.

#### Broad: `partial | misaligned` (`pos = 13`, `neg = 59`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] | Claim Totals (`sup` / `insuf` / `contra` / `unverif`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `llm_v1_preregistered_none` | Pre-registered (`f72098a` cache -> `22e3590` replay) | 11 | 36 | 2 | 23 | 0.2340 [0.1360, 0.3722] | 0.8462 [0.5776, 0.9567] | 0.3667 | 0.6102 [0.4827, 0.7242] | `145 / 458 / 0 / 0` |
| `llm_v1_preregistered_codefilter` | Post-hoc filter | 5 | 7 | 8 | 52 | 0.4167 [0.1933, 0.6805] | 0.3846 [0.1771, 0.6448] | 0.4000 | 0.1186 [0.0587, 0.2252] | `135 / 200 / 0 / 0` |
| `llm_v1_postreview_none` | Corrected after review | 10 | 36 | 3 | 23 | 0.2174 [0.1226, 0.3557] | 0.7692 [0.4974, 0.9182] | 0.3390 | 0.6102 [0.4827, 0.7242] | `158 / 485 / 0 / 0` |
| `llm_v1_postreview_codefilter` | Corrected after review | 5 | 6 | 8 | 53 | 0.4545 [0.2127, 0.7199] | 0.3846 [0.1771, 0.6448] | 0.4167 | 0.1017 [0.0474, 0.2046] | `148 / 199 / 0 / 0` |
| `gate_v4_vocab2` | Vocabulary v2.0 design (`P1`) | 3 | 9 | 10 | 50 | 0.2500 [0.0889, 0.5323] | 0.2308 [0.0818, 0.5026] | 0.2400 | 0.1525 [0.0824, 0.2652] | `145 / 221 / 15 / 833` |

Note on Phase 1 `gate_v4_vocab2` literal-verifiability coverage: `381 / 1,214 (31.38%)` admitted propositions are literally checkable (`145` supported = `11.94%`, `221` insufficient_evidence = `18.20%`, `15` contradicted = `1.24%`), vs `833 / 1,214 (68.62%)` `describes_behavior` (`unverifiable`); `escalated = 0`, `truncated = 0`.

#### Strict: `misaligned` (`pos = 3`, `neg = 69`)

| Method | Status | TP | FP | FN | TN | Precision [95% CI] | Recall [95% CI] | F1 | FPR [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm_v1_preregistered_none` | Pre-registered | 2 | 45 | 1 | 24 | 0.0426 [0.0117, 0.1425] | 0.6667 [0.2077, 0.9385] | 0.0800 | 0.6522 [0.5345, 0.7538] |
| `llm_v1_preregistered_codefilter` | Post-hoc filter | 2 | 10 | 1 | 59 | 0.1667 [0.0470, 0.4480] | 0.6667 [0.2077, 0.9385] | 0.2667 | 0.1449 [0.0807, 0.2466] |
| `llm_v1_postreview_none` | Corrected after review | 2 | 44 | 1 | 25 | 0.0435 [0.0120, 0.1453] | 0.6667 [0.2077, 0.9385] | 0.0816 | 0.6377 [0.5198, 0.7410] |
| `llm_v1_postreview_codefilter` | Corrected after review | 2 | 9 | 1 | 60 | 0.1818 [0.0514, 0.4770] | 0.6667 [0.2077, 0.9385] | **0.2857** | 0.1304 [0.0702, 0.2297] |
| `gate_v4_vocab2` | Vocabulary v2.0 design (`P1`) | 2 | 10 | 1 | 59 | 0.1667 [0.0470, 0.4480] | 0.6667 [0.2077, 0.9385] | 0.2667 | 0.1449 [0.0807, 0.2466] |

---

## 4. Audit Fixes & Case-by-Case Flip Explanations (`results_summary.json`)

Every script bug found in review is isolated into explicit post-review functions (`check_pr_corrected` in `check_claims.py`, `legacy_section=False` in `parse.py`, and `--mode postreview` in `llm_gate.py`) while leaving pre-registered baselines untouched:

1. **`parse.section()` early header truncation (`_LEGACY_STOP` -> `_TEMPLATE_STOP`):**
   - Original `_LEGACY_STOP` matched any `## <emoji>` line, truncating PR bodies containing emoji section headers inside the description (`3` Phase 1 PRs; `0` diffs across all `528` Phase 2 PRs).
   - **Effect on `rules_v1` (`v1_parse_fix`):** `0` verdict flips in either Phase 1 or Phase 2 (two Phase 1 PRs, `3119417980` and `3206510254`, gained `0 -> 7/7` and `0 -> 10/10` extracted/supported claims without flipping `CONSISTENT`).
2. **Evidence-profile and quote-centering interaction on long diff lines (`rules_v2` `[:300]` prefix truncation vs `_centered_quote` + word-boundary matching + non-collapsing directory prefix + doc-checkbox scope):**
   - Under original `check_pr()` (`ev.strip()[:300]`), switching `legal_prose -> code_config` flipped `3214782537` (Phase 1, `aligned`) and `3255269165` (Phase 2, `aligned`) from `CONSISTENT` to `INCONSISTENT` because truncating at column 300 severed identifiers mid-word on long diff lines; `check_pr_corrected()` centers quotes around the matched token (`_centered_quote`), restoring both to `CONSISTENT` (`rules_v2_postreview_code_config`: `I_ev_rejected_by_verify` drops from `5 -> 0` in Phase 1 and `41 -> 15` in Phase 2).
   - **All `rules_v2` verdict flips:**
     - **Phase 1 `3214782537` (`Claude_Code`, `aligned`):** `v2_code_config` `INCONSISTENT` (`0/2`, `[:300]` truncation) -> `postreview_code_config` `CONSISTENT` (`1/2`, matches `legal_prose`). `v2_legal_orig_to_postreview`: `0` flips.
     - **Phase 2 `3119422525` (`OpenAI_Codex`, `partial`):** `v2_legal_prose` `INCONSISTENT` (`2/4`, two `I` claims rejected under `legal_prose`) -> `v2_code_config` `CONSISTENT` (`4/4`, becomes FN under strict `code_config`).
     - **Phase 2 `3255269165` (`Copilot`, `aligned`):** `v2_code_config` `INCONSISTENT` (`2/4`, `[:300]` truncation) -> `postreview_code_config` `CONSISTENT` (`4/4`, matches `legal_prose`).
     - **Phase 2 `3143303411` (`Copilot`, `aligned`):** `v2_legal_prose` & `v2_code_config` `INCONSISTENT` (`1/3`, two `F` claims unverified) -> `postreview_legal_prose` & `postreview_code_config` `CONSISTENT` (`2/3`) because replacing `m.group(1).strip("./")` with `re.sub(r"^\./+", "", m.group(1))` stopped stripping the leading dot from a dotfile name (`val_len 10 -> 11`), allowing the file claim to match its changed file.
     - **Phase 2 `3097621471` (`Devin`, `aligned`):** `v2_legal_prose` (`1/2`, `CONSISTENT`) -> `postreview_legal_prose` (`0/2`, `INCONSISTENT`) because word-boundary `_has_identifier()` stopped matching an accidental substring hit on one `I` claim, while the other `I` claim is accepted under `postreview_code_config` (`1/2`, `CONSISTENT`).
3. **LLM `v1` post-review fixes (`llm_gate.py --mode postreview`, Phase 1 only):**
   - Fixes unique-basename file matching, removes `vals[:300]` ledger cap, includes `unlocated_claims` in verdict tallying, and deduplicates `unit_id` across overlapping segments:
     - `3146561196` (`partial`): `insufficient_evidence 4 -> 2` (`supported 3`, `4/7 -> 2/5 < 50%`), flips `INCONSISTENT -> CONSISTENT` under `none` (segment deduplication); unchanged (`CONSISTENT`) under `codefilter`.
     - `3196885081` (`aligned`): `insufficient_evidence 1 -> 7` (`supported 5`, `1/6 -> 7/12`), flips `CONSISTENT -> INCONSISTENT` under `none` (surfacing previously dropped `unlocated_claims` behavioral assertions); unchanged (`CONSISTENT`) under `codefilter`.
     - `3246950577` (`aligned`): `insufficient_evidence 2 -> 1`, flips `INCONSISTENT -> CONSISTENT` under both `none` and `codefilter` (dedup drops unsupported count below `>= 2`).
4. **Uncommitted November 16 `v3` verifier changes (`f42c7b3`):**
   - Audited `src/claimlayer/verifiers/diff.py` (+46 lines: `normalization_provenance`, `internal_quantity_conflict`) and `quote_quantity.py` (`canonicalize_quantity_notations_with_provenance`, `detect_internal_quantity_conflicts`, heading-case folding, table-header years). Measured **`0` diffs** vs prior `HEAD` across all `600` PR-MCI PRs (`rules_v1` and `rules_v2`) and committed cleanly with tests as `f42c7b3`.
5. **Removal of silent diff-fact caps (`500` symbols / `300` tests) and fail-closed `truncated -> ESCALATE` (`eval_v2vocab.py`):**
   - `diff_facts_v2()` previously sliced `add_syms[:500]`, `rem_syms[:500]`, `mod_syms[:500]`, `add_tests[:300]`, and `rem_tests[:300]` silently, which risked marking an incompletely indexed PR as `CONSISTENT`. By default (`max_syms=None`, `max_tests=None`), all extracted facts are now kept; when explicit bounds are configured and exceeded, `diff_facts_v2()` records `"truncated": True` plus `truncated_fields` counts and `verdict_v2()` returns `"ESCALATE"` (never `"CONSISTENT"`).
   - **Affected PRs (41 total across 600; 559 unaffected PRs had `0` diffs across `gate`, `claim_status`, `ledger_units`, and `receipt_sha256`):**
     - **Phase 1 (`11 / 72`):** `3078006902`, `3079303421`, `3138264959`, `3214782537`, `3219102497`, `3226475067`, `3226500911`, `3229710961`, `3234102722`, `3235103212`, `3271045504`.
     - **Phase 2 (`30 / 528`):** `OpenAI_Codex` (`3130011634`, `3131785378`, `3206215915`, `3235674228`, `3235857607`, `3238767869`, `3264925676`, `3276240606`); `Copilot` (`3074548447`, `3094488685`, `3141023608`, `3141477809`, `3152257828`, `3152602811`, `3200393827`, `3230611789`, `3242497585`, `3262902843`); `Devin` (`2832575560`, `2879844166`, `2982204377`, `3007216622`, `3060701727`, `3158683531`, `3215670862`, `3216501010`); `Cursor` (`3189263416`, `3216373575`, `3232357145`, `3250808281`).
   - **Claim and verdict changes from uncapping (`5` PRs changed `claim_status`, `2` flipped verdict, `0` truncated):**
     - **Phase 1 `3214782537` (`Claude_Code`, `aligned`):** `supported 12 -> 13`, `insufficient_evidence 143 -> 142` (`1` modified identifier past index 500 recovered); verdict stays `INCONSISTENT`.
     - **Phase 2 `3215670862` (`Devin`, `partial`):** `supported 5 -> 6`, `insufficient_evidence 10 -> 9`; verdict stays `INCONSISTENT`.
     - **Phase 2 `3262902843` (`Copilot`, `aligned`):** `supported 2 -> 3`, `insufficient_evidence 2 -> 1`; verdict stays `INCONSISTENT` (`contradicted: 1`).
     - **Phase 2 `3230611789` (`Copilot`, `misaligned`):** `supported 1 -> 3`, `insufficient_evidence 2 -> 0` (the two remaining identifiers were modified symbols past the old `[:500]` cap), flipping `INCONSISTENT -> CONSISTENT`.
     - **Phase 2 `3250808281` (`Cursor`, `partial`):** `supported 1 -> 4`, `insufficient_evidence 5 -> 2` (three added identifiers recovered past `[:500]`), flipping `INCONSISTENT -> CONSISTENT`.

---

## 5. Honest Conclusions

1. **No detection superiority claim:** On broad `partial | misaligned` in blind held-out Phase 2 (`n = 528`), `gate_v4_vocab2` (`F1 = 0.4039`) trails `rules_v1` (`F1 = 0.4370`), `rules_v2_legal_prose` (`F1 = 0.4188`), and `rules_v2_postreview_code_config` (`F1 = 0.4127`). On strict `misaligned`, the PR-MCI `paper_heuristic` clearly leads (`F1 = 0.6377`, `P = 0.7333`, `R = 0.5641`, `FPR = 0.0164`) over both `gate_v4_vocab2` (`F1 = 0.4463`) and `rules_v2_postreview_code_config` (`F1 = 0.4486`).
2. **Where the value lies (auditable, fail-closed triage):**
   - Across Phase 2, **`26.29%` (`851 / 3,237`)** of extracted agent claims target literally verifiable code/diff artifacts (`453` supported, `389` insufficient evidence, `9` contradicted), while **`73.71%` (`2,386 / 3,237`)** are high-level behavioral descriptions routed explicitly to `describes_behavior` (`unverifiable`)—eliminating silent false alarms (`458` unsupported claims and `61.0%` FPR in pre-registered `llm_v1` -> `10.1%` FPR in Phase 2 `gate_v4_vocab2`).
   - Every checkable claim carries a deterministic verification status backed by literal diff evidence or explicit conflict/escalation receipts (`receipt_sha256`), internal resolution exceptions fail closed as `error_escalated` (`ESCALATE`) instead of crashing multi-hundred-file PR runs (`528/528`, `0` runtime errors), and capped diff ledgers fail closed as `ESCALATE` (never `CONSISTENT`).
3. **Why deterministic + heuristic signals outperform on strict misalignment:** Strict `misaligned` PRs often fail due to unmentioned files, fabricated features described purely in prose, or PR-level metadata contradictions (`Copilot` is where `paper_heuristic` achieves `F1 = 0.8085`), whereas checking individual quoted symbols via an LLM extractor introduces reader noise (`45` schema rejections, `22` unknown evidence IDs) without improving aggregate PR-level classification over deterministic structural checks.

---

## 6. Limitations

- **Sample size & wide uncertainty bounds:** Phase 1 has `n = 72` (`13` broad / `3` strict positives), giving wide 95% Wilson intervals (e.g., Phase 1 `paper_heuristic` precision `[0.2065, 1.0000]`). Phase 2 has `n = 528` with `39` strict `misaligned` cases overall and `0` in `Cursor`.
- **Strong cross-phase & cross-agent instability:** `rules_v2_legal_prose` dropped from `F1 = 0.5385` (Phase 1) to `0.4188` (Phase 2), while `gate_v4_vocab2` shifted from `0.2400` (Phase 1) to `0.4039` (Phase 2). Across agents in Phase 2, `gate_v4_vocab2` broad F1 spans `0.5490` (`Copilot`), `0.3200` (`OpenAI_Codex`), `0.3137` (`Devin`), and `0.0800` (`Cursor`), while `paper_heuristic` spans `0.5507` (`Copilot`) to `0.0000` (`Devin`, `0/32` recall).
- **Low literal-claim coverage on behavioral PR prose:** Only `26.29%` (Phase 2) / `31.38%` (Phase 1) of extracted propositions can be checked directly against literal diff artifacts; the remaining `73.71%` / `68.62%` describe behavior (`describes_behavior`) without multi-step inference verification.
- **Dataset licensing constraint:** PR-MCI declares no open-source license, so neither the raw markdown annotations/diffs nor recorded LLM response caches (`*.jsonl`) are committed to this repository; reproduction requires fetching `ac6bbf4` externally via `fetch_data.sh`.
- **LLM reader & single-model dependency:** Results reflect one extractor model (`nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA NIM). Reader extraction boundary ambiguity (literal symbol vs `describes_behavior`) and malformed/foreign-evidence outputs (`reader_schema: 45`, `unknown_or_foreign_evidence_id: 22`, `assertion_kind_not_declared_for_predicate: 11` in Phase 2) remain upstream bottlenecks.
- **Inference cost & latency:** Live reproduction across all `600` PRs issues multi-segment LLM API calls over up to hundreds of diff files per PR, making offline replay caches essential for deterministic regression testing and CI verification.

---

## 7. Reproducing All Results

### External Data & Cache Layout (Not Committed)

Because PR-MCI declares no license, raw dataset files and LLM replay caches live outside git (ignored via `case_studies/pr_mci_claude_code/.gitignore`):

1. **Dataset (`PRMCI_DIR`, pinned to `ac6bbf4`):**
   ```bash
   cd claim-layer/case_studies/pr_mci_claude_code
   PRMCI_DIR=/path/to/PR-MCI ./fetch_data.sh
   ```
   (Defaults to `./data/PR-MCI` if `PRMCI_DIR` is omitted.)
2. **Expected LLM replay cache filenames in `--cache-dir`** (default `claim-layer/case_studies/pr_mci_claude_code`, or pass e.g. `--cache-dir /path/to/caches`):
   - `prmci_llm_cache.jsonl` — Phase 1 Nemotron `v1` completions (82 calls)
   - `prmci_llm_v2vocab_phase1_cache.jsonl` — Phase 1 `code_change` v2.0 completions (`n = 72`)
   - `prmci_llm_v2vocab_phase2_cache.jsonl` — Phase 2 `code_change` v2.0 completions (`n = 528`)

### Offline Bit-Identical Replay & Double-Run Check (No Network)

Regenerate all 6 sanitized result files (`results_phase1_rules.json`, `results_phase1_llm_v1.json`, `results_phase1_llm_v2vocab.json`, `results_phase2_rules.json`, `results_phase2_llm_v2vocab.json`, `results_summary.json`):

```bash
python claim-layer/case_studies/pr_mci_claude_code/reproduce.py \
  --reader replay \
  --prmci-dir /path/to/PR-MCI \
  --cache-dir /path/to/caches \
  --out-dir claim-layer/case_studies/pr_mci_claude_code
```

Run two full consecutive offline replays in temporary directories and assert bit-for-bit identical bytes and `receipt_sha256` digests across all 6 files:

```bash
python claim-layer/case_studies/pr_mci_claude_code/reproduce.py \
  --reader replay \
  --verify-replay \
  --prmci-dir /path/to/PR-MCI \
  --cache-dir /path/to/caches
```

### Live Regeneration (`NVIDIA_API_KEY`)

To regenerate caches and outputs from live NVIDIA NIM inference (never commit `*.jsonl` caches or keys):

```bash
export NVIDIA_API_KEY="<your-nvidia-nim-api-key>"
python claim-layer/case_studies/pr_mci_claude_code/reproduce.py \
  --reader llm \
  --workers 4 \
  --prmci-dir /path/to/PR-MCI \
  --cache-dir /path/to/caches \
  --out-dir claim-layer/case_studies/pr_mci_claude_code
```

### Standalone Script Reference

| File | Purpose |
|---|---|
| `reproduce.py` | Single-command orchestrator for all Phase 1 + Phase 2 rule and LLM tables (`--reader replay\|llm`, `--verify-replay`) |
| `eval_v2vocab.py` | Frozen `code_change` v2.0 vocabulary evaluator (`fingerprint == 620ebd71...`) calling official `claimlayer.v4.triage.verify_draft()` |
| `check_claims_v1.py` | Unmodified pre-registered rule verifier (`v1`) |
| `check_claims.py` | `check_pr()` (`v2` rules) and `check_pr_corrected()` (Task 1 bug fixes isolated for comparison) |
| `llm_gate.py` | Gate v4 + `code_change` v1.0 evaluator with `--mode preregistered\|postreview` |
| `parse.py` | PR-MCI markdown annotation loader with `_TEMPLATE_STOP` (`legacy=False` default) and optional `legacy=True` |
| `run.py` | Standalone deterministic rule-runner (`--agent`, `--out`) |
| `fetch_data.sh` | Fetches PR-MCI at commit `ac6bbf4` into `PRMCI_DIR` without storing raw data in git |
