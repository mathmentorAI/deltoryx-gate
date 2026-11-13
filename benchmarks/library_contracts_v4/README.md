# Standalone v4 contracts

Run from the monorepo root:

```sh
.venv/bin/python -I claim-layer/benchmarks/library_contracts_v4/run.py --out /tmp/claimlayer-contracts-new
```

The runner inserts only this package's `src` directory. It requires neither the
old `/Users/franciscohdb/Documents/claim-layer` checkout nor its `core`,
`ClaimLayer`, `CandidateClaim` root exports or `LegacyClaimLayer` adapter.
It loads the existing block-A JSON fixtures and executes only v4. A failed
contract returns a nonzero exit code; an existing output run is never overwritten.

`protocol.json` and `summary.json` beside the runner are historical multi-arm
artifacts from the original repository, not the output of this standalone run.
The old benchmark called a different semantic engine “v3”; that API is not the
monorepo's deterministic `verify` API. Adding misleading root exports would not
make the engines equivalent. Fresh output records `execution_mode` and hashes.

These are contracts over supplied, typed, attested evidence. They do not measure
whether a language model reads, extracts or interprets a new document correctly.
