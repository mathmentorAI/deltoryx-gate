# ClaimLayer future work

## Current use in an assisted pilot

ClaimLayer can already check structured proposals against a registered corpus and an operator-defined policy. A bounded pilot can use it to check amounts, units, citations and explicit limits, return concrete repair instructions, and surface candidate discrepancies and authority conflicts for a reviewer.

Start with a narrow review task such as comparing coupons by note class across an indenture and an offering memorandum. The operator supplies the corpus, document roles, evaluation date, and required fields. The reviewer confirms the final report and approves any downstream action. Interpret `PASS` as the outcome of the configured checks on the submitted assertions. Keep the source material and reviewer decision with the pilot record.

The [securitization replay](../examples/harness2_securitization/) illustrates the repair loop with prepared proposals. The [Harvey LAB studies](harvey_lab_benchmark.md) report historical results for structured retrieval and a bundled harness intervention. Measure the gate's contribution separately in the pilot. Record material defects detected, defects reaching the reviewer, incorrect blocks, repair attempts, and reviewer minutes per completed matter.

If the pilot depends on receipts as audit evidence, first verify round-trip consistency using the same source revision, policy, parser context, and evaluation date. Include a concurrent update case and a table whose scale is inherited from a header outside the quoted row.

## Priority 1. Bind review obligations to the deliverable

Extend the existing v4 vocabulary, field evidence, clause relations, corpus obligations, and authorization components. Let an operator-defined profile create stable obligations for a known domain. A comparison obligation binds the entity, field, scope, source revisions, and the kind of change to examine. A coupon comparison, for example, binds each rate to its note class and its source. A waterfall comparison binds the step, beneficiary, and relative priority.

Require supporting spans for the material fields and declared rules for derivations. Record whether each obligation is satisfied, disputed, or unresolved. Use `RETRY` for correctable defects and `ESCALATE` for unresolved material gaps and exhausted repair budgets. Use evidence scores to order re-reads and reviewer work. Require every mandatory check independently of the score.

For autonomous publication or execution, enforce the decision outside the agent. Bind the exact deliverable or action payload, operation, policy version, and evidence snapshot. Prefer rendering critical fields from verified records. Evaluate correspondence separately wherever the application allows free-form prose.

## Priority 2. Share a versioned, layout-aware evidence representation

Parse each source revision once. Preserve the original artifact digest, extraction version, page and span coordinates, normalization provenance, and table cell context. Represent units, scale, sign, period, basis, and comparison operator explicitly. Carry unresolved references and ambiguous table scopes into the review workflow.

Align clauses and cells by structural identity and revision relations. Emit typed differences for value, basis, priority, modality, additions, and deletions. Use authority scoped to the proposition, clause, and effective date. Index candidates by entity, field, and clause. Evaluate and issue receipts against one immutable snapshot, using the same interpretation when reverifying. Reject ambiguous policy keys before calculating a policy digest.

Regression cases should include swapped coupons with identical numeric token sets, a new clause in an unsegmented document that already has other reported differences, and a quoted table row whose scale comes from a separate header. For larger corpora, measure ingestion, verification latency in cold and warm runs, memory, and reviewer load.

## Priority 3. Establish risk and utility on a blind evaluation

Freeze the code, profiles, corpus-selection rules, and budgets before opening a reserved set. Use independently annotated matters with clean controls, material changes, missing evidence, and proposals designed to satisfy superficial checks. Keep CART and Ridgeline as development regressions. Obtain blinded adjudication of material correctness.

Run a fixed-proposal comparison to measure the verifier in isolation and a second end-to-end comparison of solver-only, self-review, semantic monitoring, deterministic gate, and combined approaches. Match the evidence available to each arm and record inference cost, attempts, and human review time. Report materially incorrect authorizations among authorized operations alongside authorization coverage, legitimate task completion, escalation, and reviewer effort. Compute uncertainty by matter and document family.

The [implementation and evaluation report](v3_2_1_evaluation.md) and [three-arm protocol](../examples/harness2_securitization/THREE_ARM_EVALUATION.md) document the existing baselines.
