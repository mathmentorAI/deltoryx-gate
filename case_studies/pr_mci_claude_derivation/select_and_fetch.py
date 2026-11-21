"""Step 1: deterministic case selection + public GitHub metadata/diff fetch (cached).

Selection rule (declared before reading any case in detail):
  * Population: PR-MCI (ac6bbf4) Claude_Code PRs present in validation/validation_results.csv (n=72).
  * Size filter (planner context budget): 1 <= files_changed <= 25 and churn (adds+dels) <= 1500,
    from corpus metadata; after fetching, the net GitHub diff must match the corpus stats
    (files, +adds, -dels), every file must carry a textual patch, and total patch <= 60,000 chars.
  * All non-aligned (partial | misaligned) PRs passing the filter are included (difficulty oversampling).
  * Aligned PRs fill the remaining slots up to 20, in random.Random(20261121) order over the
    eligible aligned PRs sorted by pr_id; a PR failing the post-fetch checks is skipped.
  * PRs whose diff is empty (0 files) are kept if non-aligned: the empty diff is the evidence.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pr_mci_claude_code"))
import parse  # noqa: E402

csv.field_size_limit(10**9)
PRMCI = Path(os.environ.get("PRMCI_DIR", HERE.parent / "pr_mci_claude_code" / "data" / "PR-MCI"))
CACHE = HERE / "github_cache"
SEED = 20261121
TARGET = 20


def gh(url: str, cache_name: str):
    path = CACHE / cache_name
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "deltoryx-prmci-eval"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode("utf-8"))
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.5)
    return data


def owner_repo_num(url: str):
    parts = url.rstrip("/").split("/")
    return parts[-4], parts[-3], int(parts[-1])


def fetch(pr: dict):
    o, r, n = owner_repo_num(pr["url"])
    meta = gh(f"https://api.github.com/repos/{o}/{r}/pulls/{n}", f"{pr['pr_id']}_pull.json")
    files = gh(f"https://api.github.com/repos/{o}/{r}/pulls/{n}/files?per_page=100", f"{pr['pr_id']}_files.json") if (pr["adds"] or pr["dels"] or pr["files"]) else []
    return meta, files


def post_check(pr, meta, files):
    reasons = []
    if meta.get("changed_files") != len(pr["files"]):
        reasons.append(f"files {meta.get('changed_files')}!={len(pr['files'])}")
    if meta.get("additions") != pr["adds"] or meta.get("deletions") != pr["dels"]:
        reasons.append(f"lines +{meta.get('additions')}/-{meta.get('deletions')} != +{pr['adds']}/-{pr['dels']}")
    if any("patch" not in f for f in files):
        reasons.append("file without textual patch: " + ",".join(f["filename"] for f in files if "patch" not in f)[:200])
    total = sum(len(f.get("patch", "")) for f in files)
    if total > 60000:
        reasons.append(f"net patch {total} chars > 60000")
    return reasons


def main():
    val = {r["pr_id"]: r for r in csv.DictReader(open(PRMCI / "validation/validation_results.csv", encoding="utf-8-sig"))}
    prs = [p for p in parse.load("Claude_Code", prmci_dir=str(PRMCI)) if p["pr_id"] in val]
    for p in prs:
        p["gold"] = (val[p["pr_id"]]["consensus_label"] or val[p["pr_id"]]["annotator1_label"]).lower()

    def size_ok(p):
        return len(p["files"]) <= 25 and (p["adds"] or 0) + (p["dels"] or 0) <= 1500

    non = sorted([p for p in prs if p["gold"] != "aligned" and size_ok(p)], key=lambda p: p["pr_id"])
    aligned = sorted([p for p in prs if p["gold"] == "aligned" and size_ok(p) and p["files"]], key=lambda p: p["pr_id"])
    random.Random(SEED).shuffle(aligned)

    log = {"population": len(prs), "seed": SEED, "selected": [], "skipped": []}
    for p in non + aligned:
        if len(log["selected"]) >= TARGET:
            break
        meta, files = fetch(p)
        reasons = post_check(p, meta, files)
        entry = {"pr_id": p["pr_id"], "gold": p["gold"], "url": p["url"],
                 "base_sha": meta["base"]["sha"], "head_sha": meta["head"]["sha"],
                 "merged": meta.get("merged"), "merge_commit_sha": meta.get("merge_commit_sha")}
        if reasons and p["gold"] == "aligned":
            log["skipped"].append({**entry, "reasons": reasons})
            print("SKIP", p["pr_id"], reasons)
            continue
        entry["post_check_issues"] = reasons
        log["selected"].append(entry)
        print("SELECT", p["pr_id"], p["gold"], reasons)
    (HERE / "selection_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(log["selected"]), "selected")


if __name__ == "__main__":
    main()
