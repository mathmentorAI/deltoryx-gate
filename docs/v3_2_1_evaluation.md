# ClaimLayer v3.2.1 + experimental v4: implementation and evaluation

Date: 2026-11-13. Baselines: monorepo `569ca80`, public package `7d5aeef`.

## Results and scope

| Evaluation | Result |
|---|---:|
| Existing regression suite | All original 238 tests retained and passing |
| Complete suite, including 18 new generated/structural tests | 256 passed, 1 skipped |
| External 110-case evaluation, before this continuation | 74/110 |
| External evaluation after this continuation | 92/110 |
| External white-box cases | 19/19 |
| External generalization cases | 73/91 |
| Remaining false PASS / false rejection | 16 / 2 |
| Standalone v4 typed contracts, block A / B | 20/20 and 11/11 |

This is not a claim of 110/110, end-to-end semantic accuracy, or production safety.
The 18 remaining external cases were **not** rerun through an LLM extraction
pipeline. v4 provides the appropriate typed representation and abstention
architecture; passing its supplied-record contracts does not prove it extracts
those cases correctly.

The external file remains outside both repositories. No external fixture text was
added to source code or internal tests. New tests generate numeric values, list
permutations and table lengths from independent templates. Because external
results informed the failure-family diagnosis, this is an evaluation-guided repair,
not a fresh blind generalization estimate. A new reserved suite is needed for that.

## All 36 original failures, classified

| Family | External cases | Outcome / mechanism |
|---|---|---|
| Dimensionless value used for a monetary field | g2-A1b | Fixed: an inferred currency field requires an explicit unit, as duration fields already did |
| Identifier versus measurement with an undeclared field type | g2-A1a | Remains: arbitrary natural-language field semantics require an operator-declared type; no new identifier-word list |
| Date role and calendar syntax | g2-A3a, g2-B3c | Fixed: date-local clause binding and dotted day/month/year parsing, with date spans masked from quantity parsing |
| Rate time denominator | g2-A7b | Fixed: explicitly different rate periods cannot match on the numerator alone; no automatic annualization |
| Parallel ordered lists | g2-A8c, g2-A8c-ok | Fixed for explicit equal-cardinality designator/value lists; unsupported shapes reject instead of nearest-neighbour guessing |
| Accounting percentage sign | g2-A12a | Fixed for a parenthesized numeric percentage; descriptive parenthesized currency references remain distinct |
| Contextual table scales | g2-A13a, g2-A13b, g2-A13c, g2-A13d, g2-A13e-ok | Fixed: scale/currency grammar, symbolic powers of ten, preserved line/paragraph boundaries, no 120-character cut-off, row-local scope |
| Fractions and exact conversions | g2-B8, g2-B9 | Fixed: cardinal/ordinal fraction grammar and exact years-to-months conversion; no months-to-days assumption |
| Compound duration syntax | g2-B14a, g2-B14b | Fixed: hyphens and spaces normalize in compound units; business/calendar distinction preserved |
| Distinct qualified subjects | g2-C2d-ok | Fixed for strict modifier inclusion in copular noun phrases; not a general semantic subject parser |
| Comparison coverage without changed values | g2-C4a | Fixed: comparison policy records required governing value tokens; naming a section alone cannot discharge them |
| Rights, negation, exceptions, conditions | g2-A4a through g2-A4g | Remain: open propositional semantics, requiring typed polarity and conditions in v4 |
| Direction of change versus absolute value | g2-A12b | Remains: interpreting a natural-language change as signed growth needs an explicit semantic proposition |
| Semantic predicate aliases and partial amendment scope | g2-C1a, g2-C1b, g2-C1d-ok | Remain: two missed amendments and one false supersession; operator-declared predicate/scope identity is needed |
| Different phrases for the same fee | g2-C2c | Remains: semantic predicate mapping, not numeric parsing |
| Priority relation and reserve identity | g2-C7a, g2-C7b | Remain: closed relation enum/entity-predicate mapping needed |
| Different reporting schedules | g2-C7c-ok | Remains: false override; reporting frequency must remain part of the typed scope |
| Actor/object reversal | g2-C4b | Remains: typed role relation needed; bag-of-words equality cannot verify this |
| Undeclared synonym for a bounded predicate | g2-C11a | Remains in the unchanged external policy; explicit operator aliases bind numeric limits in independent tests |

Remaining false rejections: `g2-C1d-ok`, `g2-C7c-ok`. The other 16 remaining cases
are false PASS. In particular, legacy v3 still must not be treated as a fail-closed
semantic verifier for arbitrary prose.

## Implementation details

The pre-existing trust/state fixes and v4 import were preserved. This continuation
adds parser changes above, generated regression checks, `Policy.required_value_tokens`,
comparison-value coverage, and propagation of operator vocabulary/coverage through
MCP. Caller-injected predicate aliases are rejected. The vocabulary bridge now
calls `Vocabulary.proposition`, validating values, units and conditions rather
than only checking entity/predicate names. Unrepresentable v3 inputs return
`UNREPRESENTABLE_IN_VOCABULARY` with RETRY; v3 has no new fourth verdict state.

The bridge is validation, not a substitute for v4 evidence ingestion or a semantic
translation layer. A v3 free-text condition cannot silently become a typed v4
condition. Serialized vocabulary descriptions must not be mistaken for executable
objects: provide the operator-owned `Vocabulary` object to v3. Unsupported shapes
fail closed. The standalone `EvidenceLayer` remains the typed v4 execution path.

## Standalone benchmark

The copied runner imported root `CandidateClaim` / `ClaimLayer`,
`claimlayer.legacy.LegacyClaimLayer`, and a missing `functional_parity/run.py`.
Those belonged to a different historical engine in the old repository. Exporting
aliases would incorrectly imply API/semantic equivalence.

The runner now uses local `claimlayer.v4`, the existing block-A JSON fixtures and
unchanged block-B contracts. It runs with Python isolated mode, returns a nonzero
exit code for a failed contract, preserves existing output runs, and hashes the
runner, fixtures and local sources. Historical multi-arm artifacts beside the
runner are explicitly labelled as historical, not newly reproduced results.

## Reproduction

From the monorepo root:

```sh
PYTHONPATH=claim-layer/src:claim-layer:src .venv/bin/pytest claim-layer/tests/ tests/test_api_and_triage.py -v
.venv/bin/python -I claim-layer/benchmarks/library_contracts_v4/run.py --out /tmp/claimlayer-contracts-new
TMPDIR=/tmp PYTHONPATH=claim-layer/src:claim-layer .venv/bin/python /Users/franciscohdb/Downloads/test_reaudit_v3_2_heldout.py
```

Use a new benchmark output directory for each run. The external evaluator is not
part of this repository. On macOS, setting `TMPDIR=/tmp` aligns the parent with
the evaluator's environment-cleared subprocesses. Without this, two seal attacks
were reported as “ok” despite a missing precondition/exception. With aligned temp
directories all three attacks actually run and return exit 2: ledger tampering
for the first two, policy tampering for the third. White-box count is then 19/19
without those precondition exceptions.

## Remaining trust boundary

Three-way agreement between documents, SQLite and the temporary seal is an
integrity consistency check, not a protected root of trust against a process able
to rewrite every copy or replace the gate. Production needs operator-owned source
and policy metadata plus a signing key or verification service outside the agent's
write/read authority. The same-user CLI alone does not provide that isolation.

Unkeyed receipts reject by default unless reverified against trusted ledger state;
`allow_unkeyed=True` establishes internal consistency only. Authenticated offline
verification requires a protected signing key.

## v4 classification of the 18 remaining external v3 failures

The 92/110 external score reflects v3 surface string/policy evaluation without an
operator-declared domain `Vocabulary` or v4 typed propositions. Classifying the 18
remaining v3 failures (`16` false PASS, `2` false rejections) against v4
(`Vocabulary` + `verify_extraction_fidelity` + `authorize_operation`):

| Failure group | Cases | v4 boundary behavior | Classification |
|---|---|---|---|
| Rights, negation, exceptions, conditions | `g2-A4a`–`g2-A4g` (7) | `EvidenceUnit.proposition` types `assertion_kind`, `polarity`, `condition_status`, `conditions`, and `exceptions`; `verify_extraction_fidelity` checks `UNCAPTURED_CONDITION`, `POLARITY_MISMATCH`, `BOUND_OPERATOR_MISMATCH`, `TRUNCATED_CLAUSE_CONTEXT`, and `CONDITION_NOT_IN_FIELD_EVIDENCE`; `authorize_operation` fails closed on `UNREPRESENTABLE_CONDITION`, `UNKNOWN_CONDITION`, `PENDING_CONDITION`, or `UNRESOLVED_CONFLICT` | Handled deterministically once extracted into typed propositions; uncaptured informal conditions without lexical cues remain an upstream reader boundary |
| Direction of change vs absolute value | `g2-A12b` (1) | `verify_extraction_fidelity` emits `DIRECTION_OF_CHANGE_MISMATCH`, and `_numeric_marker_present` requires `UnitRegistry` unit compatibility on typed quantities rather than bare decimal matching | Handled at the v4 fidelity boundary when modeled with typed units/change direction |
| Identifier vs measurement with undeclared field type | `g2-A1a` (1) | Operator `Vocabulary` declares explicit `FieldSpec` types and units; `AMBIGUOUS_OR_UNMAPPED_PROPOSAL` / `VALUE_NOT_IN_EVIDENCE` / `POLICY_BOUND_UNIT_MISMATCH` prevent untyped identifier numbers from passing as currency/duration quantities | Requires operator-declared `Vocabulary` + fail closed otherwise |
| Partial amendment scope & reporting schedule separation | `g2-C1a`, `g2-C1b`, `g2-C1d-ok`, `g2-C7c-ok` (4; includes both false rejections) | `resolve_scope_and_precedence` uses exact canonical `query_scope` matching and explicit `ClauseRelation` overrides instead of loose section matching, avoiding false overrides across distinct reporting schedules (`g2-C7c-ok`) or scopes (`g2-C1d-ok`) | Handled deterministically at the v4 scope/relation layer |
| Predicate aliases, fee wording, priority & reserve identity | `g2-C2c`, `g2-C7a`, `g2-C7b` (3) | Operator `Vocabulary` (`PredicateSpec`, canonical `entity` / `predicate` / `scope`) and `ClauseRelation` force explicit canonical mapping; unmapped phrasings fail closed (`AMBIGUOUS_OR_UNMAPPED_PROPOSAL` / `MISSING_REQUIRED_PREDICATE`) | Handled when mapped to operator `Vocabulary`; open paraphrase normalization stays upstream |
| Actor / object reversal | `g2-C4b` (1) | Triple `(entity, predicate, value)` proposals do not parse open-domain grammatical subject/object roles unless the operator `Vocabulary` models directed role predicates and binds them via `FieldEvidence` | Upstream NL-to-proposal formalization boundary unless directed role predicates are declared |
| Undeclared synonym for a bounded predicate | `g2-C11a` (1) | `numeric_bounds` bind strictly to operator-declared predicate names; undeclared synonyms fail closed (`MISSING_REQUIRED_PREDICATE`) rather than guessing aliases | Intentional fail-closed boundary without explicit operator alias |
