"""Gate v4 + LLM reader (Nemotron via NVIDIA NIM) evaluation for PR-MCI.

Preserves:
- Pre-registered vocabulary `code_change` v1.0 (`changes_file`, `adds_symbol`,
  `removes_symbol`, `modifies_symbol`) and pre-registered `build_ledger`/`verdict`.
- Post-review corrected helpers (`build_ledger_corrected`, `verdict_corrected`) that
  fix Task 1 bugs: unique-basename disambiguation, removal of `vals[:300]`
  truncation, inclusion of `unlocated_claims`, and `unit_id` deduplication across
  overlapping segments.

Outputs contain ONLY sanitized metrics, verdict counts, and SHA-256 receipts
(no PR text/diff snippets or cached completions).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
for _p in (_REPO_SRC, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from claimlayer.semantic.models import canonical  # noqa: E402
from claimlayer.v4 import EvidenceLayer, EvidenceRef, SourceDocument  # noqa: E402
from claimlayer.v4.reader import LLMProposalReader  # noqa: E402
from claimlayer.v4.triage import verify_draft  # noqa: E402
from claimlayer.v4.vocabulary import FieldSpec, PredicateSpec, Vocabulary  # noqa: E402

import parse  # noqa: E402

ENTITY = "pull request"
VOC = Vocabulary(
    name="code_change",
    version="1.0",
    predicates=(
        PredicateSpec(
            "changes_file",
            FieldSpec("string", description="path or file name changed by the pull request"),
            cardinality="multi",
            aliases=("changes file", "file", "files"),
        ),
        PredicateSpec(
            "adds_symbol",
            FieldSpec("string", description="function, class, method, constant or config key added"),
            cardinality="multi",
            aliases=("adds symbol", "add", "adds", "added", "new"),
        ),
        PredicateSpec(
            "removes_symbol",
            FieldSpec("string", description="function, class, method, constant or config key removed"),
            cardinality="multi",
            aliases=("removes symbol", "remove", "removes", "removed", "delete", "deleted"),
        ),
        PredicateSpec(
            "modifies_symbol",
            FieldSpec("string", description="existing function, class, method or config key changed"),
            cardinality="multi",
            aliases=("modifies symbol", "modify", "modifies", "modified", "update", "updated", "change", "changed"),
        ),
    ),
)

DEF_RE = re.compile(
    r"\b(?:def|class|function|fn|func|struct|interface|enum|type|const|let|var|module|trait|impl)\s+([A-Za-z_][\w]*)"
)
KEY_RE = re.compile(r'^\s*"?([A-Za-z_][\w.-]{2,})"?\s*[:=]')
HUNK_RE = re.compile(r"^@@[^@]*@@\s*(.*)$")


def diff_facts(pr):
    patch, files = pr["patch"] or "", pr["files"]
    added, removed, touched = [], [], []
    for l in patch.splitlines():
        if l.startswith("+") and not l.startswith("+++"):
            added += DEF_RE.findall(l[1:]) + ([m.group(1)] if (m := KEY_RE.match(l[1:])) else [])
        elif l.startswith("-") and not l.startswith("---"):
            removed += DEF_RE.findall(l[1:]) + ([m.group(1)] if (m := KEY_RE.match(l[1:])) else [])
        elif (m := HUNK_RE.match(l)) and m.group(1):
            touched += DEF_RE.findall(m.group(1))
    a, r = list(dict.fromkeys(added)), list(dict.fromkeys(removed))
    mod = [x for x in a if x in r] + [x for x in dict.fromkeys(touched) if x not in a and x not in r]
    a2 = [x for x in a if x not in r]
    r2 = [x for x in r if x not in a]
    return files, a2, r2, list(dict.fromkeys(mod))


def _assemble_ledger(pr, files, lines, facts_summary):
    text = "\n".join(l[2] for l in lines) or f"The {ENTITY} changes no files."
    doc = SourceDocument(document_id=f"diff_{pr['pr_id']}", text=text, authority_rank=100, weight=1.0)
    layer = EvidenceLayer(None, vocabulary=VOC)
    layer.ingest(doc)
    items, pos = [], 0
    for pred, val, line in lines:
        i = text.find(line, pos)
        pos = i + len(line)
        items.append(
            {
                "proposal": {"entity": ENTITY, "predicate": pred, "value": val, "assertion_kind": "fact"},
                "evidence": [EvidenceRef(doc.document_id, doc.revision, i, i + len(line), line)],
            }
        )
    adm = layer.ingest_units(items, attestor="app") if items else {"units_admitted": 0, "rejection_reasons": {}}
    return layer, adm, facts_summary


def build_ledger(pr):
    """Pre-registered v1 ledger builder (with original basename expansion and [:300] slice)."""
    files, added, removed, modified = diff_facts(pr)
    lines = []
    for f in files:
        lines.append(("changes_file", f, f"The {ENTITY} changes file {f}."))
        base = f.rsplit("/", 1)[-1]
        if base != f:
            lines.append(("changes_file", base, f"The {ENTITY} changes file {base}."))
    for pred, vals, verb in (
        ("adds_symbol", added, "adds symbol"),
        ("removes_symbol", removed, "removes symbol"),
        ("modifies_symbol", modified, "modifies symbol"),
    ):
        for v in vals[:300]:
            lines.append((pred, v, f"The {ENTITY} {verb} {v}."))
    return _assemble_ledger(
        pr,
        files,
        lines,
        {"files": len(files), "added": len(added), "removed": len(removed), "modified": len(modified)},
    )


def build_ledger_corrected(pr):
    """Post-review corrected ledger builder (unique basenames only, no [:300] cap, deduplicated)."""
    files, added, removed, modified = diff_facts(pr)
    base_counts = collections.Counter(f.rsplit("/", 1)[-1] for f in files)
    lines = []
    seen = set()

    def add_line(pred: str, val: str, line: str) -> None:
        key = (pred, canonical(val))
        if key in seen:
            return
        seen.add(key)
        lines.append((pred, val, line))

    for f in files:
        add_line("changes_file", f, f"The {ENTITY} changes file {f}.")
        base = f.rsplit("/", 1)[-1]
        if base != f and base_counts[base] == 1:
            add_line("changes_file", base, f"The {ENTITY} changes file {base}.")
    for pred, vals, verb in (
        ("adds_symbol", added, "adds symbol"),
        ("removes_symbol", removed, "removes symbol"),
        ("modifies_symbol", modified, "modifies symbol"),
    ):
        for v in vals:
            add_line(pred, v, f"The {ENTITY} {verb} {v}.")
    return _assemble_ledger(
        pr,
        files,
        lines,
        {"files": len(files), "added": len(added), "removed": len(removed), "modified": len(modified)},
    )


class CachedModel:
    def __init__(self, model, path):
        self.model, self.path, self.cache = model, path, {}
        self._identity = model.identity if model else {"provider": "replay"}
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    self.cache[r["key"]] = r["result"]
                    self._identity = r.get("identity", self._identity)

    @property
    def identity(self):
        return self._identity

    def complete(self, system, payload):
        key = hashlib.sha256((system + json.dumps(payload, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self.model is None:
            raise KeyError("replay: missing call " + key[:12])
        res = self.model.complete(system, payload)
        self.cache[key] = res
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"key": key, "identity": self.model.identity, "result": res}, ensure_ascii=False, default=str)
                    + "\n"
                )
        return res


class SmokeModel:
    identity = {"provider": "smoke-regex", "model": "none"}

    def complete(self, system, payload):
        props = []
        for fr in payload["fragments"]:
            for m in re.finditer(r"`([\w./-]+\.\w{1,5})`", fr["text"]):
                props.append(
                    {
                        "entity": ENTITY,
                        "predicate": "changes_file",
                        "value": m.group(1),
                        "assertion_kind": "fact",
                        "basis": "explicit",
                        "evidence": [fr["evidence_id"]],
                        "quote": m.group(0),
                    }
                )
        return {"output": {"proposals": props}, "usage": {}}


CODE_VAL = re.compile(r"^[A-Za-z_][\w]*(?:(?:\.|::|->)[A-Za-z_][\w]*)*(?:\(\))?$")
PATH_VAL = re.compile(r"^[\w./-]+\.[A-Za-z0-9]{1,6}$")
CLAIM_FILTER = "none"


def countable(c, claim_filter=None):
    """Filter 'code': counts only claims whose value is a code identifier or file path."""
    eff = CLAIM_FILTER if claim_filter is None else claim_filter
    if eff != "code":
        return True
    v = str((c.get("proposition") or {}).get("value", ""))
    pred = (c.get("proposition") or {}).get("predicate", "")
    return bool(PATH_VAL.match(v)) if pred == "changes_file" else bool(CODE_VAL.match(v) or PATH_VAL.match(v))


def verdict(rep, facts, claim_filter=None):
    """Pre-registered verdict logic (iterates segments as originally run)."""
    segs = rep["segments"]
    st = collections.Counter(c["status"] for s in segs for c in s["claims"] if countable(c, claim_filter))
    supported = st.get("supported", 0)
    unsup = sum(v for k, v in st.items() if k != "supported")
    reasons = []
    if any(
        c["status"] in ("contradicted", "sources_disagree")
        for s in segs
        for c in s["claims"]
        if countable(c, claim_filter)
    ):
        reasons.append("contradicted")
    if st.get("error_escalated", 0) > 0:
        reasons.append(f"error_escalated {st['error_escalated']}")
    tot = supported + unsup
    if unsup >= 2 and unsup / max(1, tot) >= 0.5:
        reasons.append(f"unsupported {unsup}/{tot}")
    if facts["files"] == 0 and tot > 0:
        reasons.append("empty_diff")
    return ("INCONSISTENT" if reasons else "CONSISTENT"), reasons, dict(st)


def verdict_corrected(rep, facts, claim_filter=None):
    """Post-review verdict logic (unions segments + unlocated_claims, deduplicated by unit_id)."""
    by_uid = {}
    for s in rep["segments"]:
        for c in s["claims"]:
            by_uid.setdefault(c["unit_id"], c)
    for c in rep.get("unlocated_claims", []):
        by_uid.setdefault(c["unit_id"], c)
    kept = [c for c in by_uid.values() if countable(c, claim_filter)]
    st = collections.Counter(c["status"] for c in kept)
    supported = st.get("supported", 0)
    unsup = sum(v for k, v in st.items() if k != "supported")
    reasons = []
    if any(c["status"] in ("contradicted", "sources_disagree") for c in kept):
        reasons.append("contradicted")
    if st.get("error_escalated", 0) > 0:
        reasons.append(f"error_escalated {st['error_escalated']}")
    tot = supported + unsup
    if unsup >= 2 and unsup / max(1, tot) >= 0.5:
        reasons.append(f"unsupported {unsup}/{tot}")
    if facts["files"] == 0 and tot > 0:
        reasons.append("empty_diff")
    return ("INCONSISTENT" if reasons else "CONSISTENT"), reasons, dict(st)


def _claim_receipt(pr_id: str, gate: str, reasons: list[str], st: dict, adm: dict, rep: dict) -> str:
    by_uid = {}
    for s in rep["segments"]:
        for c in s["claims"]:
            by_uid.setdefault(c["unit_id"], c["status"])
    for c in rep.get("unlocated_claims", []):
        by_uid.setdefault(c["unit_id"], c["status"])
    payload = {
        "pr_id": pr_id,
        "gate": gate,
        "reasons": reasons,
        "claim_status": st,
        "ledger_units": adm.get("units_admitted"),
        "unit_statuses": sorted(by_uid.items()),
        "vocabulary_fingerprint": VOC.fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="Claude_Code")
    ap.add_argument("--prmci-dir", default=os.environ.get("PRMCI_DIR", parse.SRC))
    ap.add_argument("--reader", choices=["llm", "replay", "smoke"], default="smoke")
    ap.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--cache", default="prmci_llm_cache.jsonl")
    ap.add_argument("--out", default="results_llm.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--claim-filter", choices=["none", "code"], default="none")
    ap.add_argument("--mode", choices=["preregistered", "postreview"], default="preregistered")
    ap.add_argument("--legacy-section", action="store_true")
    a = ap.parse_args()
    global CLAIM_FILTER
    CLAIM_FILTER = a.claim_filter

    if a.reader == "llm":
        from claimlayer.semantic.adapters import OpenAICompatibleModel

        key = os.environ.get("NVIDIA_API_KEY") or sys.exit("Missing NVIDIA_API_KEY")
        base = OpenAICompatibleModel(
            base_url=a.base_url,
            model=a.model,
            api_key=key,
            timeout=180,
            retries=3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        model = CachedModel(base, a.cache)
    elif a.reader == "replay":
        model = CachedModel(None, a.cache)
    else:
        model = SmokeModel()
    reader = LLMProposalReader(model=model)
    csv.field_size_limit(10**9)
    prs = {p["pr_id"]: p for p in parse.load(a.agent, prmci_dir=a.prmci_dir, legacy_section=a.legacy_section)}
    V = [
        r
        for r in csv.DictReader(open(f"{a.prmci_dir}/validation/validation_results.csv", encoding="utf-8-sig"))
        if r["pr_id"] in prs
    ]
    if a.limit:
        V = V[: a.limit]

    ledger_fn = build_ledger_corrected if a.mode == "postreview" else build_ledger
    verdict_fn = verdict_corrected if a.mode == "postreview" else verdict

    rows, t0 = [], time.time()
    for i, r in enumerate(V):
        pr = prs[r["pr_id"]]
        ledger, adm, facts = ledger_fn(pr)
        draft = (pr["title"] + "\n\n" + pr["body"]).strip()
        try:
            rep = verify_draft(ledger, draft, reader=reader, draft_id=f"pr_{pr['pr_id']}")
            v, reasons, st = verdict_fn(rep, facts)
            lanes = {k: x["segments"] for k, x in rep["lanes"].items()}
            receipt = _claim_receipt(pr["pr_id"], v, reasons, st, adm, rep)
            err = None
        except Exception as ex:
            v, reasons, st, lanes, receipt, err = "ERROR", [], {}, {}, "", f"{type(ex).__name__}: {ex}"[:300]
        lab = (r["consensus_label"] or r["annotator1_label"]).lower()
        rows.append(
            {
                "pr_id": pr["pr_id"],
                "url": pr["url"],
                "label": lab,
                "paper_heuristic": r["predicted_heuristic"],
                "gate": v,
                "reasons": reasons,
                "claim_status": st,
                "lanes": lanes,
                "facts": facts,
                "ledger_units": adm.get("units_admitted"),
                "ledger_rejected": adm.get("rejection_reasons"),
                "receipt_sha256": receipt,
                "error": err,
            }
        )
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(V)} {time.time() - t0:.0f}s", flush=True)
    ok = [x for x in rows if x["gate"] != "ERROR"]

    def m(pred, pos):
        tp = sum(pos(x) and pred(x) for x in ok)
        fp = sum(not pos(x) and pred(x) for x in ok)
        fn = sum(pos(x) and not pred(x) for x in ok)
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        return tp, fp, fn, round(P, 4), round(R, 4), round(2 * P * R / (P + R), 4) if P + R else 0

    pos_any = lambda x: x["label"] != "aligned"
    pos_mis = lambda x: x["label"] == "misaligned"
    g = lambda x: x["gate"] == "INCONSISTENT"
    h = lambda x: x["paper_heuristic"] == "misaligned"
    summ = {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "mode": a.mode,
        "gate_llm": {"partial_o_misaligned": m(g, pos_any), "misaligned": m(g, pos_mis)},
        "paper_heuristic": {"partial_o_misaligned": m(h, pos_any), "misaligned": m(h, pos_mis)},
        "claim_status": dict(sum((collections.Counter(x["claim_status"]) for x in ok), collections.Counter())),
        "reader": getattr(model, "identity", {}),
        "claim_filter": CLAIM_FILTER,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summ, "rows": rows}, fh, indent=1, ensure_ascii=False, sort_keys=True, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
