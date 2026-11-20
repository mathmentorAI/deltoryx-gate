# Phase 3 Preregistration: v4 Evidence Intelligence for Coding-Agent PR Verification

- **Preregistration Version:** `phase3-v2.0` (supersedes `phase3-v1.0` commit `8c29f2b` / `46b3e18ecad62d8abff8f83f5d1a15683c4d038083a0d1779325e6867551250c`, prior library commit `a343156`)
- **Status:** `FROZEN_BEFORE_EVAL_A_AND_EVAL_B`
- **`PREREGISTRATION.json` SHA-256:** `c2fd43d9d6d9946e9f137b75e5adb7e0d240410ed1154aa4840217a52299f6c5`
- **Model & Decoding:** `nvidia/nemotron-3-ultra-550b-a55b`, `enable_thinking=false`, `temperature=0`, `retries=0`, `duplicate_json_recovery=v1-discard-malformed-proposal`
- **Non-Modification Pledge:** No rule, prompt, stopword list, vocabulary, relevance threshold, verdict threshold, or evaluation filter may be modified between committing this preregistration and completing Evaluation A (PR-MCI Phase 2, `n=528`) and Evaluation B (AIDev blind dual-annotator, `n=125`).

---

## 1. Changes Since `phase3-v1.0` (Zero Prompts Changed, 100% Offline Cached Dev)

Prior to running Evaluation A or Evaluation B, seven implementation bugs identified in the v1 manual audit were fixed in `code_change_inference.py`, Pass-3 blind reread was hardened, and the broad `REVIEW` operating point was decoupled from `>=1` claim triggers and recalibrated strictly on AIDev-Dev:
1. **Seven implementation fixes (`reconcile_phase3_literal_claim`, `build_typed_diff_ledger`, `evaluate_i5_structural_signals`, `_detect_hard_factual_contradiction`, `verify_behavior_graph`):** exact package boundary matching (`eslint` vs `eslint-config-next`); PEP 508 quoted bounds and lockfile `[[package]]` stanza tracking (`rich`, `spec-kitty-runtime`, `h11`); excluding manifest `version`/`lock-version` metadata keys and non-manifest/fixture JSON files; excluding PR checklist templates (`- [x] Documentation updated (or not required)`) from `D_doc_only_but_code`; stripping trailing `()` and reconciling YAML workflow tokens while treating shell/parenthetical expressions as `unverifiable`; matching pre-rename file paths; reclassifying local test execution/linter reports in `adds_test_for` to `out_of_scope_meta` (`unverifiable`).
2. **Pass-3 target-grounding guard (`pass3_require_target_grounded_in_claim = True`):** prevents blind reread from bypassing `claim_text` specificity via a spoofed `target` symbol absent from the claim sentence (reducing `N2_hard` Pass-3 any-premise leakage from `3/60` to `1/61`, `0` derived leaks).
3. **Decoupled broad `REVIEW` calibration (`FPR(originals) <= 0.25`):** added `review_k_phantom`, `review_theta_phantom`, `review_k_lit`, `review_theta_lit`, cutting raw `originals` broad flag rate from `73.00%` (`146/200`) to **`25.00%` (`50/200`)** at `F1 = 0.4805`.

---

## 2. Preregistered Operating Points & Baselines

| Operating Point | Gate Verdict Mapping | PR-MCI Gold Target | Primary Comparison Baselines (PR-MCI Phase 2, `n = 528`) |
|---|---|---|---|
| **Strict** (`INCONSISTENT`) | `verdict == "INCONSISTENT"` | `misaligned` (39 pos / 489 neg) | `paper_heuristic`: **F1 = 0.6377** (`P = 0.7333`, `R = 0.5641`, `FPR = 0.0164`); `gate_v4_vocab2`: `F1 = 0.4463` |
| **Broad** (`INCONSISTENT + REVIEW`) | `verdict in {"INCONSISTENT", "REVIEW"}` | `partial \| misaligned` (121 pos / 407 neg) | `paper_heuristic`: `F1 = 0.3444`; `gate_v4_vocab2`: `F1 = 0.4039`; `rules_v1`: `F1 = 0.4370` |

**Fail-Closed `ESCALATE` Policy:** `ESCALATE` fires whenever `ledger.truncated == True`, `error_escalated >= 1`, or unrecoverable `read_errors` occur; `ESCALATE` never counts as `CONSISTENT` and is reported explicitly alongside both evaluated-subset and full-denominator metrics. In offline replay, Pass-3 subset reassembly raises `KeyError` on uncached claims by default; `calibrate_dev.py` explicitly enabled `--allow-pass3-replay-miss` for the `24` claims (`20` PRs across Dev splits) that stopped short-circuiting to `premise_contradicted` after bug fixes so zero live API calls were required (`escalate_count` equals v1 baseline: `originals=2`, `n1_easy_50=0`, `n2_hard=1`, `n3_mutation=2`).

---

## 3. Frozen Prompt Digests, Profiles & Rule Catalog

| Artifact | Identifier / Version | Exact SHA-256 / Fingerprint |
|---|---|---|
| Pass-1 Vocabulary | `code_change` `v2.0` (frozen untouched) | `620ebd71c7dda399cb8b90e5ac038a2a2cf510782a86202b79b75c8d84900d2c` |
| Pass-2 Behavior Graph Prompt | `v3-behavior-graph-2` (unchanged) | `792908b5f1ed3d53d555d5e416a6637d106ffbfae5cf05b5f7fba32862429545` |
| Pass-3 Blind Reread Prompt | `v3-blind-reread-2` (unchanged) | `3fd039356e6cc1a2af882f4c22288f9e4858c8b5146dbea6eb4fcc6dfc2a39fe` |
| v4 DomainProfile | `code_change_inference@1.0.0` (unchanged) | `de7814a16885f87dad24df71d986bacabcecf3a70414e57c4d1f0540b85fa9c4` |
| EpistemicProfile | `code-diff-strict@1.0` (`explicit`, `syntactic`, `deductive`) | Built into DomainProfile |
| Code Stopwords | `543` frozen tokens | `b1bb3b69a783048c94b8ebd00e5a9b2b18129610584b4c5b370b3becad39ff42` |
| `aidev_dev_manifest.json` | 200 orig, 50 N1, 184 N2, 200 N3 | `29b1d7579b3443881e388e28858ae3c52a970cbce60f955d8713da745f0f7f29` |
| Eval A PR-MCI Phase 2 IDs (`n=528`) | JSON / newline sorted IDs | `dd3460f172904419e26335cf627789bfa6a4f7cc5eeba49453ed4eeac3fcf757` / `13ead90ab9d80d63e68361d4926240289d4211045dbe454c4c378bb01452abf6` |
| `eval_b_manifest.json` (`n=125`) | 25/agent, 125 distinct repos | `642c0a524709a4d2e52b7bf45194ced2fe52e56cdb54b12941455581cd868444` |
| `ANNOTATION_GUIDELINES.md` | Dual-annotator blind rubric | `4f89d73f56906771ad29e1d190a45b7bbb29608116c443a57970a186080f6da1` |
| `results_aidev_dev.json` | Final Dev calibration + 20-PR & 50-claim audits | `e2381329dbfed82d30c946efb167260e20e06ac5060a17fe850c937f00c20d36` |

---

## 4. Frozen Calibrated Parameters (Two-Stage on AIDev-Dev `N2_hard` vs `originals`)

```json
{
  "calibration_criterion": "Two-stage lexicographic optimization on AIDev-Dev (N2_hard positives vs originals negatives) only; N1_50 and N3_mutation reported only. Stage A maximizes strict INCONSISTENT F1 subject to raw originals strict FPR <= 10.0% and N2 strict leakage <= 10.0%. Stage B freezes Stage A's RelevanceConfig and strict INCONSISTENT thresholds and maximizes broad (INCONSISTENT+REVIEW) F1 over decoupled REVIEW thresholds subject to raw originals broad FPR <= 25.0%.",
  "relevance_config": {
    "min_specific_anchors": 2,
    "pass3_require_target_grounded_in_claim": true,
    "require_category_compat": true,
    "require_target_match_when_present": true
  },
  "thresholds": {
    "i5_placeholder_max_title_words": 3,
    "i5_placeholder_max_words": 8,
    "i5_placeholder_min_churn": 100,
    "i5_understated_min_enumerated": 2,
    "i5_understated_min_share": 0.95,
    "include_i5": true,
    "k_lit": 2,
    "k_phantom": 4,
    "review_k_lit": 3,
    "review_k_phantom": 3,
    "review_on_all_unruled_hypothesis": false,
    "review_on_single_unsupported_or_phantom": true,
    "review_theta_lit": 0.25,
    "review_theta_phantom": 0.0,
    "singleton_phantom_inconsistent": true,
    "theta_lit": 0.67,
    "theta_phantom": 0.85
  }
}
```

### AIDev-Dev Calibration & Ablation Summary (`phase3-v2.0`)

- **Strict (`INCONSISTENT`) `N2_hard` vs `originals`:** **F1 = 0.2564** (`TP=30/184`, `FP=20/200`), `P = 0.6000`, `R = 0.1630`, raw `originals` FPR = **0.1000 (`20/200`)**, Pass-1-clean `originals` FPR = **`0.0174`** (down from `0.0432` in v1).
- **Broad (`INCONSISTENT + REVIEW`) `N2_hard` vs `originals`:** **F1 = 0.4805** (`TP=74/184`, `FP=50/200`), `P = 0.5968`, `R = 0.4022`, raw `originals` broad FPR = **0.2500 (`50/200`)** (meeting `<= 0.2500`; default `>=1` review rate would be `0.6600`).
- **Claim-Level Anti-Laxity Leakage (`d_full`):** `N1_easy_50` strict=`0.0000` (`0/26`), broad=`0.0385` (`1/26`), catch=`0.9231`; `N2_hard` strict=`0.0367` (`4/109`), broad=`0.1193` (`13/109`, down from `0.1376`), catch=`0.8165`; `N3_mutation` (`n=200`) `INCONSISTENT = 0.5550`, `INCONSISTENT+REVIEW = 0.6600` (note `17/53` `dependency_version_mismatch` N3 synthetics in v1 referenced eliminated bogus keys `version`/`openapi`/`MaxDisplayCount`/`cpu`).

| Ablation (`N2` vs `originals`) | Strict F1 | Strict FPR | Broad F1 | Broad FPR | `originals` `derived` | `originals` `hypothesis` | `originals` `no_premises` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `a_literal` | 0.1909 | 0.0750 | 0.3150 | 0.1500 | 0 | 0 | 0 |
| `b_graph_no_rules` | 0.2435 | 0.0900 | 0.4887 | 0.2550 | 0 | 861 | 224 |
| `c_graph_rules_no_reread` | 0.2424 | 0.0950 | 0.5238 | 0.3200 | 200 | 576 | 309 |
| `e_full_no_i5` | 0.2435 | 0.0900 | 0.4787 | 0.2400 | 211 | 650 | 224 |
| `d_full` | 0.2564 | 0.1000 | 0.4805 | 0.2500 | 211 | 650 | 224 |

### Broad `INCONSISTENT + REVIEW` Pareto Curve (Stage B on `N2_hard` vs `originals`)

| FPR Cap | Achieved FPR | Broad F1 | Precision | Recall | Calibrated `REVIEW` Params |
|---|---:|---:|---:|---:|---|
| <= 0.10 | 0.1000 (`20/200`) | **0.2785** | 0.6226 | 0.1793 (`33/184`) | `k_lit=2, theta_lit=0.67, k_ph=99, theta_ph=0.0, all_hyp=False` |
| <= 0.15 | 0.1500 (`30/200`) | **0.3602** | 0.6104 | 0.2554 (`47/184`) | `k_lit=3, theta_lit=0.34, k_ph=4, theta_ph=0.75, all_hyp=False` |
| <= 0.20 | 0.1950 (`39/200`) | **0.4296** | 0.6100 | 0.3315 (`61/184`) | `k_lit=2, theta_lit=0.34, k_ph=3, theta_ph=0.75, all_hyp=False` |
| <= 0.25 | 0.2500 (`50/200`) | **0.4805** | 0.5968 | 0.4022 (`74/184`) | `k_lit=3, theta_lit=0.25, k_ph=3, theta_ph=0.0, all_hyp=False` |
| <= 0.30 | 0.3000 (`60/200`) | **0.5212** | 0.5890 | 0.4674 (`86/184`) | `k_lit=2, theta_lit=0.25, k_ph=3, theta_ph=0.0, all_hyp=False` |
| <= 0.40 | 0.4000 (`80/200`) | **0.5768** | 0.5722 | 0.5815 (`107/184`) | `k_lit=2, theta_lit=0.25, k_ph=2, theta_ph=0.0, all_hyp=False` |
| <= 0.50 | 0.4850 (`97/200`) | **0.6293** | 0.5708 | 0.7011 (`129/184`) | `k_lit=2, theta_lit=0.0, k_ph=1, theta_ph=0.25, all_hyp=False` |
| <= 0.75 | 0.6600 (`132/200`) | **0.6723** | 0.5479 | 0.8696 (`160/184`) | `k_lit=1, theta_lit=0.0, k_ph=1, theta_ph=0.0, all_hyp=False` |

### Pass-3 Blind Reread Effect (`Before` vs `After` Target-Grounding Guard)

- `N2_hard` reread (`61` injected claims): leaked any premise **3 -> 1**, leaked derived `0 -> 0` (`274` reread claims attempted).
- `N1_easy_50` reread (`18` injected claims): leaked `0 -> 0`; `originals` (`235` reread claims): `derived_supported = 11`, `hypothesis = 74`.

### `N3_mutation` Breakdown (`d_full`, reported only)

| Subtype | `n` | Strict `INCONSISTENT` Recall | Broad `INCONSISTENT+REVIEW` Recall |
|---|---:|---:|---:|
| `claimed_doc_file_but_zero_docs` | 41 | 0.6829 | 0.7561 |
| `claimed_tests_but_zero_tests` | 90 | 0.5444 | 0.6333 |
| `dependency_version_mismatch` | 53 | 0.4151 | 0.5849 |
| `polarity_flip_remove_added_symbol` | 16 | 0.7500 | 0.8125 |

---

## 5. Read-Only Manual Audits (Zero Post-Audit Threshold Edits)

1. **PR-level Audit (`100%` of `20` `originals` flagged `INCONSISTENT` under `d_full`):** **7/20 genuine inconsistencies** (`3315462760`, `3492184706`, `3998484014`, `3301299783`, `3978765167`, `3373697596`, `3924461875`) and **13/20 false alarms** -> raw FPR = `0.1000`, audit-corrected FPR = **`0.0650`** (`13/200` full denom) / **`0.0674`** (`13/193` excluding real).
2. **Claim-level Audit (fixed `Random(20261120)` sample of `50` of `224` `originals` `no_premises_in_diff` claims):** **11/50 genuinely unbacked in diff** (**claim-level precision = `0.2200`**, 95% Wilson CI **`[0.1275, 0.3524]`**) vs `39/50` reader/gate failures (`17` Pass-2/3 reader extraction/quoting misses, `22` gate strictness limits such as enclosing function context outside single diff line, single-token config/grammar lines, `_wbtest.mbt`/`__test__/` categorization, and `code_config` octal/CJK checks). This empirically validates requiring quorum (`k_phantom = 4`, `review_k_phantom = 3`) rather than flagging single phantom claims.

---

## 6. Statistical Analysis Plan & Locked Pre-Eval Protocol

- **Primary Comparisons on Eval A (`n=528`, untouched):** `d_full` strict `INCONSISTENT` vs `paper_heuristic` (`F1=0.6377`), `d_full` broad `INCONSISTENT+REVIEW` vs `paper_heuristic` (`F1=0.3444`) and literal `gate_v4_vocab2` (`F1=0.4039`), and `e_full_no_i5` / `c_graph_rules_no_reread` vs `b_graph_no_rules` and `a_literal`.
- **Uncertainty & Significance Tests:** 95% Wilson score intervals, two-sided exact paired McNemar tests, and 10,000-resample paired bootstrap CIs for `delta-F1`.
- **Stratified Reporting & Dual-Annotator Eval B:** Separate per-agent breakdown on Eval A and Eval B; dual-annotator blind labeling (`n=125`, target weighted Cohen's `kappa >= 0.65`) adjudicated strictly prior to running model predictions.
