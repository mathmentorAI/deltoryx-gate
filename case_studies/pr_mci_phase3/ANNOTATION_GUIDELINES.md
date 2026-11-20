# Evaluation B Human Annotation Guidelines (Dual-Annotator Blind Protocol)

## Scope and Blinding
- **Dataset:** Evaluation B (`n = 125` AIDev closed permissive-license PRs, balanced across 5 coding agents at 25 PRs/agent).
- **Repo-disjointness:** All repositories in Evaluation B are strictly disjoint from AIDev-Dev (`n = 200`) and from PR-MCI (`n = 600` / 399 repos).
- **Blinding:**
  1. Both annotators receive independent CSVs (`annotator_1.csv`, `annotator_2.csv`) with distinct randomized item ordering.
  2. Coding-agent identity (`Claude_Code`, `OpenAI_Codex`, `Copilot`, `Devin`, `Cursor`) and all model/gate outputs are omitted from every CSV and diff sheet (`sheets/EB-XXX.md`).
  3. Annotators MUST NOT discuss items or inspect the other annotator's labels prior to locking both CSVs.

---

## 3-Class Labeling Rubric (Aligned with PR-MCI)

Label every item (`label` column) strictly with one of: `aligned`, `partial`, or `misaligned`.

### 1. `aligned`
Use `aligned` when **every concrete factual or behavioral claim** in the PR title and description accurately reflects the diff:
- Named file paths, symbol definitions, configuration keys/flags, dependencies/versions, and unit/integration tests correspond to actual changes in the diff.
- High-level behavioral summaries accurately describe what the changed code does without fabricating extra features or components.
- Boilerplate checklists or footer templates (`Generated with...`, PR template headings) do NOT contradict the diff. (Checked `[x]` checkboxes that claim `Documentation update` *only* while modifying source code are NOT aligned.)

### 2. `partial`
Use `partial` when the PR description captures the **primary change**, yet contains a **non-trivial discrepancy, overstatement, or omission**:
- One concrete secondary claim is unsupported or slightly overstated (e.g., mentions updating a file/module that is untouched, or claims broad test/doc coverage when only an inline comment was edited).
- An explicit file list leaves out significant modified code/config files without misleading the reader about the main task.
- Checked template checkboxes partially mismatch the touched file types, OR a vague claim cannot be substantiated from the patch while the main change holds.

### 3. `misaligned`
Use `misaligned` when the PR description **materially contradicts the diff or fabricates changes**:
- Claims to implement a feature/symbol, fix a specific component, or add unit/integration tests when no supporting change exists in the diff.
- Contradicts literal diff facts (e.g., claims wrong dependency version, claims removal when the symbol was added, claims `Documentation update` only on a multi-file source code patch, states wrong modified-file count, or gives a placeholder description like `"Update"` on a `>= 100`-line non-doc patch).
- Mentions unrelated components/behaviors from elsewhere in the repository that were not modified in this PR.

---

## Confidence & Evidence Notes
- Fill `confidence` with `high`, `medium`, or `low`.
- In `evidence_note`, quote the specific mismatched claim phrase and the supporting/missing file path (for `partial` or `misaligned`) so disagreements are quickly auditable during adjudication.

---

## Independent Adjudication Protocol
1. After both `annotator_1.csv` and `annotator_2.csv` are completed and locked, run `score_eval_b.py` to compute inter-annotator agreement:
   - Unweighted 3-class Cohen's $\kappa$
   - Linearly weighted 3-class Cohen's $\kappa$
   - Binary broad Cohen's $\kappa$ (`aligned` vs `{partial, misaligned}`)
   - Binary strict Cohen's $\kappa$ (`misaligned` vs `{aligned, partial}`)
2. Any item where `annotator1_label != annotator2_label` is reviewed blindly in `adjudication.csv` by a third adjudicator (or via documented tie-break citing concrete diff hunks) to record `consensus_label`.
3. Target quality bar: weighted Cohen's $\kappa \ge 0.65$; all items reach a documented `consensus_label` before running Evaluation B scoring.
