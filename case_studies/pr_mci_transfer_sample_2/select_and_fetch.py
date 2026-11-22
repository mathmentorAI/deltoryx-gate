"""Second transfer sample (M5), reserved and frozen BEFORE any stage-3 prompt change: PR-MCI Phase 2
held-out partition (OpenAI_Codex, Copilot, Devin, Cursor), excluding the 10 PRs of the first transfer
sample (inspected in M4). Same declared rule as the first sample; new seed.

Rule (declared before reading any case): 1 <= files <= 15, churn <= 600 (corpus metadata);
net GitHub base->head diff must match corpus stats (files, +adds, -dels) for EVERY stratum,
every file must carry a textual patch, total net patch <= 40,000 chars.
Strata (difficulty-weighted, declared): 2 partial + 2 misaligned + 6 aligned, each drawn in
random.Random(20261122) order over eligible PRs sorted by pr_id; failing PRs are skipped.
"""
from __future__ import annotations

import csv, json, os, random, sys, time, urllib.error, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pr_mci_claude_code"))
import parse  # noqa: E402

csv.field_size_limit(10**9)
PRMCI = Path(os.environ.get("PRMCI_DIR", HERE.parent / "pr_mci_claude_code" / "data" / "PR-MCI"))
CACHE = HERE / "github_cache"
SEED, QUOTA = 20261123, {"partial": 2, "misaligned": 2, "aligned": 6}
AGENTS = ["OpenAI_Codex", "Copilot", "Devin", "Cursor"]


def gh(url, name):
    p = CACHE / name
    if p.exists():
        return json.loads(p.read_text())
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "deltoryx-prmci-eval"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:  # PR or repository no longer available: the PR fails the rule
        if exc.code in (404, 410, 451):
            return None
        raise
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    time.sleep(0.5)
    return data


def main():
    val = {r["pr_id"]: r for r in csv.DictReader(open(PRMCI / "validation/validation_results.csv", encoding="utf-8-sig"))}
    prs = []
    for ag in AGENTS:
        for p in parse.load(ag, prmci_dir=str(PRMCI)):
            if p["pr_id"] in val:
                p["agent"] = ag
                p["gold"] = (val[p["pr_id"]]["consensus_label"] or val[p["pr_id"]]["annotator1_label"]).lower()
                prs.append(p)
    used = {e["pr_id"] for e in json.loads((HERE.parent / "pr_mci_transfer_sample" / "selection_log.json").read_text())["selected"]}
    prs = [p for p in prs if p["pr_id"] not in used]
    log = {"seed": SEED, "quota": QUOTA, "excluded_first_transfer_sample": sorted(used), "selected": [], "skipped": []}
    for stratum, k in QUOTA.items():
        pool = sorted([p for p in prs if p["gold"] == stratum and 1 <= len(p["files"]) <= 15
                       and (p["adds"] or 0) + (p["dels"] or 0) <= 600], key=lambda p: p["pr_id"])
        random.Random(f"{SEED}:{stratum}").shuffle(pool)
        got = 0
        for p in pool:
            if got >= k:
                break
            o, r, n = p["url"].rstrip("/").split("/")[-4], p["url"].rstrip("/").split("/")[-3], int(p["url"].rstrip("/").split("/")[-1])
            meta = gh(f"https://api.github.com/repos/{o}/{r}/pulls/{n}", f"{p['pr_id']}_pull.json")
            if meta is None:
                log["skipped"].append({"pr_id": p["pr_id"], "agent": p["agent"], "gold": p["gold"], "url": p["url"], "issues": ["github_unavailable"]})
                print("SKIP", p["pr_id"], stratum, ["github_unavailable"]); continue
            files = gh(f"https://api.github.com/repos/{o}/{r}/pulls/{n}/files?per_page=100", f"{p['pr_id']}_files.json")
            if files is None:
                log["skipped"].append({"pr_id": p["pr_id"], "agent": p["agent"], "gold": p["gold"], "url": p["url"], "issues": ["github_unavailable"]})
                print("SKIP", p["pr_id"], stratum, ["github_unavailable"]); continue
            issues = []
            if meta.get("changed_files") != len(p["files"]):
                issues.append("files")
            if meta.get("additions") != p["adds"] or meta.get("deletions") != p["dels"]:
                issues.append("lines")
            if any("patch" not in f for f in files):
                issues.append("binary_or_missing_patch")
            if sum(len(f.get("patch", "")) for f in files) > 40000:
                issues.append("size")
            e = {"pr_id": p["pr_id"], "agent": p["agent"], "gold": p["gold"], "url": p["url"],
                 "base_sha": meta["base"]["sha"], "head_sha": meta["head"]["sha"]}
            if issues:
                log["skipped"].append({**e, "issues": issues}); print("SKIP", p["pr_id"], stratum, issues)
                continue
            got += 1
            log["selected"].append(e); print("SELECT", p["pr_id"], p["agent"], stratum)
    (HERE / "selection_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print(len(log["selected"]), "selected")


if __name__ == "__main__":
    main()
