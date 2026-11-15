# External Public Document Annotation Instructions (Blind Protocol)

These instructions govern independent human annotation of real public document pairs fetched from SEC EDGAR (`424B*` prospectus supplements vs. `EX-4.*` operative indentures/pooling-and-servicing agreements, and `8-K`/`10-K`/`10-Q` corporate credit agreements vs. amendments).

## Core Integrity Rules

1. **Zero repository contamination**: Completed annotation files MUST be stored **outside** the `claim-layer` / `deltoryx-gate` repository (e.g. `~/external_eval_private/annotations.json`) and NEVER committed to git.
2. **Blind to reader outputs**: Annotators MUST read ONLY the raw normalized source texts listed in the immutable fetch manifest (`manifest.json`), WITHOUT inspecting any `EvidenceLayer`, `RecordedProposalReader`, LLM output, or `REFERENCE_REGISTER_GAP` output.
3. **Zero model-generated labels**: Do NOT use LLMs to draft, pre-fill, or verify ground-truth labels.

## What Counts as a Conflict

Record each material inconsistency between the higher-authority/operative document (`primary_source_id`) and the companion/disclosure/amendment document (`secondary_source_id`) under `"conflicts"`:

- **Numeric / Economic Conflict (`numeric`)**: Distinct interest rates, spreads, fees, commitment caps, thresholds, cure periods, day-count conventions (`business days` vs `calendar days`), dates, or ratios for the same tranche/facility/covenant.
- **Waterfall / Priority Order Conflict (`priority_order`)**: Different step numbers, ordering of payments/replenishments, or relative seniority (`senior to` vs `subordinated to`) in distributions of funds or collateral proceeds.
- **Textual Scope / Condition / Obligated Party Conflict (`textual_scope`, `condition`)**: Non-numeric changes that alter obligations, such as adding or removing exceptions (`unless waived`), shifting the obligated party (`Servicer` vs `Master Servicer`), or changing trigger/consent timing (`upon closing` vs `after written notice`).
- **Genuine Material Ambiguity (`ambiguity`)**: Vague or internally conflicting phrasing across documents that prevents deterministic reconciliation without escalation. Set `is_genuine_conflict: true` and explain in `annotator_notes`.

## What NOT to Record as Conflicts

Do NOT record mere differences in:
- Formatting (whitespace, capitalization, punctuation, hyphenation).
- Document self-references (e.g., `"per this Indenture"` vs `"per this Prospectus Supplement"`, or `"Exhibit 4.1"` vs `"Supplement"`).

## Clean Control Clauses

Record at least 2 `"clean_controls"` per corpus pair where two aligned clauses express the same operative terms (even with formatting or document-identifier differences) so clean-corpus false-gap and false-escalation rates can be measured reliably.

## Running Evaluation

Once annotated externally, run:

```bash
python benchmarks/external_eval/evaluate_external.py \
  --corpus-dir /path/to/downloaded_edgar_corpus \
  --annotations /path/outside/repo/annotations.json \
  --mode recorded \
  --recordings-dir /path/to/recordings \
  --json-out /path/outside/repo/eval_report.json
```
