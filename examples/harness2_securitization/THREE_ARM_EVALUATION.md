# Three-arm local evaluation

`run_three_arm_eval.py` continues the in-progress Harness2 integration. It runs
one exploratory, paired comparison using a local OpenAI-compatible model:

1. A shared initial answer (baseline, explicitly NOT_VERIFIED).
2. The same answer followed by v3 feedback and bounded model repair.
3. The same answer followed by v3/v4 feedback and bounded model repair.

This is an incremental-repair experiment, not three independent solver trials
and not an equal-total-compute experiment. Initial source-scan and generation
costs are shared and recorded separately from each arm's additional usage.

## Run

From the monorepo root:

```bash
PYTHONPATH=claim-layer/src:claim-layer .venv/bin/python \
  claim-layer/examples/harness2_securitization/run_three_arm_eval.py \
  --case cart_2024_2 --out .build/three_arm_eval/cart_fresh
```

Use `--dry-run` with a separate fresh output directory to inspect partition size
and provenance without calling the model. Defaults target the existing local
`qwen/qwen3.5-9b` endpoint and its 8192-token loaded context. No cloud endpoint or
paid API is required. API cost is zero; electricity/hardware cost is unmeasured.

Every parsed source character is scanned in non-overlapping, offset-addressed
chunks. The model generates notes, which are hierarchically reduced to fit the
context. All arms receive the same reduced notes. This is **lossy compression**:
100% source-character submission does not establish 100% fact retention or
semantic coverage. Document extraction itself may also lose original layout.

For verification, the model proposes assertions and a graph. A limited,
deterministic numeric-window retriever supplies original source quotations.
Its selected and candidate window counts are logged. It is not exhaustive and
may miss evidence; the bounded windows are allocated round-robin across sources.
Missing/ambiguous quotes fail extraction rather than being
silently fabricated. The report is never replaced by the graph extractor's
output. Both gated arms explicitly require nonempty structured assertions;
the v4 arm also requires a graph. v3 does not parse or validate a v4 graph.
Operational authorization, report disposition, issue details and verifier time
are logged separately for each attempt.

## Protocol and outputs

Before inference, `protocol.json` freezes source hash, code hashes, prompts,
model configuration, budget, and methodological limitations. Calls include exact
requests, responses, usage and finish reasons. Empty/truncated completions fail
closed. An incomplete structured extraction becomes a failed attempt eligible
for bounded repair; incomplete source readings or report generation abort the
run. A failure leaves `failure.json` and completed scan/call artifacts.
Output directories must be empty, preventing accidental overwrite of runs.

After an interrupted scan, use a fresh output directory and
`--reuse-scan .build/three_arm_eval/previous_run`. Reuse requires the same corpus
hash, model, temperature, reasoning setting and exact message contents. Only
complete successful scan responses are eligible. Complete intermediate reductions
can also be reused when their entire input messages match exactly. The new run records the paths
and hashes of reused call artifacts. Their usage belongs to the earlier run and
must be added when reporting total experiment cost; it is not free inference.
`reused_preprocessing_usage.json` accounts for those historical calls separately
from newly executed calls. This total includes `reused_scan_usage.json`; do not
add both files a second time.
An increased output cap is allowed when reusing a response that already finished
normally. This is a resumed exploratory experiment, not an independent replicate.

The independent evaluator is imported only after generation of all three arms.
It does not supply model prompts. Its pattern-based scores are a diagnostic
screen and require manual adjudication: they are not legal correctness labels.
In particular, neither matching expected words nor passing the gate proves that
all statements in a report are correct.

The inherited development pipeline includes CART-specific checks. This limits
generalization even though ground-truth files are kept out of the runtime
prompts. The corpus loaders also require the external paths in the manifest;
original documents are not copied into the repository.

## Banking and generalization

Both CART and banking-finance were used in earlier project benchmarks. Banking
must not be labelled unseen. Its files can be loaded/provenance-checked, but the
current runner deliberately does not route banking through a securitization
policy. A banking domain adapter and independently reviewed authority policy
are required before a comparable three-arm banking evaluation. In particular,
a financial spreadsheet does not globally override negotiation instructions.

A genuinely new corpus, independent rubric, multiple repetitions and matched
compute/LLM-self-review controls remain necessary before generalization claims.
