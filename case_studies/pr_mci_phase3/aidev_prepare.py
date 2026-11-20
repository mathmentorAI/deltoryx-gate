"""Prepare clean, permissive-license, repo-disjoint AIDev-Dev (n=200 + N1/N2/N3) and Eval B (n=125) datasets.

All third-party PR descriptions and commit diffs remain strictly outside git (default:
`~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/`). Only sanitized manifests
containing PR IDs, agent/repository counts, and license metadata are written into the repository.

Pinned upstream source:
- Dataset: `hao-li/AIDev` (revision `b8dc67df93e7c11f6f20c84129db7232eb1ec253`, v5)
- Dataset card license: `CC-BY-4.0` (upstream repository licenses retained; filtered strictly
  to permissive licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense, 0BSD).
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import json
import os
import random
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_SRC = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "src"))
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from claimlayer.v4.domains.code_change_inference import (  # noqa: E402
    CODE_STOPWORDS,
    build_typed_diff_ledger,
    split_code_tokens,
)

AIDEV_REPO = "hao-li/AIDev"
AIDEV_REVISION = "b8dc67df93e7c11f6f20c84129db7232eb1ec253"
AIDEV_LICENSE = "CC-BY-4.0"
PERMISSIVE_LICENSES = frozenset({
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "0BSD"
})
AGENTS = ("Claude_Code", "OpenAI_Codex", "Copilot", "Devin", "Cursor")
TARGET_RGS = (4, 7, 8, 9, 10, 15, 42, 52)
SEED = 20261119

_CHANGE_VERB_RE = re.compile(
    r"\b(?:add|added|adds|implement|implemented|implements|update|updated|updates|"
    r"remove|removed|removes|delete|deleted|deletes|fix|fixed|fixes|refactor|refactored|"
    r"refactors|introduce|introduced|introduces|configure|configured|configures|"
    r"enable|enabled|enables|disable|disabled|disables|support|supports|supported|"
    r"integrate|integrated|integrates|migrate|migrated|migrates|replace|replaced|replaces|"
    r"validate|validates|validated|handle|handles|handled|prevent|prevents|prevented)\b",
    re.I,
)
_URL_OR_ISSUE_RE = re.compile(r"https?://|#[0-9]+\b|\B@[A-Za-z0-9_-]+", re.I)


def _load_prmci_exclusions(prmci_dir: str) -> tuple[set[str], set[str]]:
    prmci_ids: set[str] = set()
    prmci_repos: set[str] = set()
    if not prmci_dir or not os.path.isdir(prmci_dir):
        raise FileNotFoundError(f"Missing PR-MCI directory required for exclusion check: {prmci_dir}")
    csv.field_size_limit(10**9)
    for rel in ("validation/validation_results.csv", "validation/annotations_collected.csv"):
        fp = os.path.join(prmci_dir, rel)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    if r.get("pr_id"):
                        prmci_ids.add(str(r["pr_id"]))
                    if r.get("repo_url"):
                        m = re.search(r"github\.com/([^/]+/[^/]+)", r["repo_url"])
                        if m:
                            prmci_repos.add(m.group(1).lower())
    for md in glob.glob(os.path.join(prmci_dir, "data", "manual_annotations", "pr_*.md")):
        txt = open(md, encoding="utf-8", errors="ignore").read()
        if m := re.search(r"\*\*PR ID:\*\*\s*(\d+)", txt):
            prmci_ids.add(m.group(1))
        if m := re.search(r"\*\*GitHub URL:\*\*\s*https://github\.com/([^/]+/[^/]+)/pull/", txt):
            prmci_repos.add(m.group(1).lower())
    return prmci_ids, prmci_repos


def _is_english_ascii(title: str, body: str) -> bool:
    comb = f"{title}\n{body}"
    ratio = sum(1 for ch in comb if ord(ch) < 128) / max(1, len(comb))
    return ratio >= 0.92 and len(body.strip()) >= 80 and bool(title.strip())


def _extract_candidate_behavior_sentences(body: str) -> list[str]:
    clean = re.sub(r"(?m)<!--.*?-->", " ", body, flags=re.S)
    clean = re.sub(r"🤖 Generated with.*", " ", clean, flags=re.S)
    cands: list[str] = []
    for raw_ln in clean.splitlines():
        s = raw_ln.strip()
        if not s or s.startswith("#") or s.startswith("```") or s.startswith("|") or re.match(r"^\s*-\s*\[[ xX]\]", s):
            continue
        s = re.sub(r"^(?:[-*•]+|\d+[.)])\s+", "", s).strip()
        for sent in re.split(r"(?<=[.!?])\s+", s):
            st = sent.strip()
            words = re.findall(r"[A-Za-z0-9_./:-]+", st)
            if not (8 <= len(words) <= 28):
                continue
            if _URL_OR_ISSUE_RE.search(st):
                continue
            if not _CHANGE_VERB_RE.search(st):
                continue
            cands.append(st)
    return cands


def _inject_sentence(body: str, sentence: str) -> str:
    clean_s = sentence.rstrip(". ") + "."
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*[-*•]\s+)(?!\[[ xX]\])", ln)
        if m:
            prefix = m.group(1)
            lines.insert(i + 1, f"{prefix}{clean_s}")
            return "\n".join(lines)
    # No plain bullet list -> append after first non-empty paragraph
    paras = [p for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
    if paras:
        paras[0] = paras[0].rstrip() + " " + clean_s
        return "\n\n".join(paras)
    return (body.strip() + "\n\n" + clean_s).strip()


def _target_diff_symbol_set(ledger) -> set[str]:
    out = set(s.lower() for s in ledger.add_syms + ledger.rem_syms + ledger.mod_syms + ledger.add_tests + ledger.rem_tests)
    for f in ledger.files:
        out.add(f.lower())
        out.add(f.rsplit("/", 1)[-1].lower())
        out.add(f.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower())
    for d in ledger.dependencies:
        out.add(d["package"].lower())
    return {x for x in out if len(x) >= 3 and x not in CODE_STOPWORDS}


def _select_phantom_phrase(donor_body: str, target_diff_syms: set[str], donor_repo: str) -> str | None:
    donor_repo_toks = set(split_code_tokens(donor_repo)) - CODE_STOPWORDS
    for sent in _extract_candidate_behavior_sentences(donor_body):
        # Clean inline backticks for behavioral check while verifying zero symbol/file overlap
        toks = {t for t in split_code_tokens(sent) if len(t) >= 3 and t not in CODE_STOPWORDS}
        if not toks:
            continue
        # Ensure phrase doesn't mention explicit repo owner/name from N1 donor
        if toks & target_diff_syms:
            continue
        # Require at least 2 meaningful non-stopword tokens so sentence isn't empty boilerplate
        if len(toks - donor_repo_toks) >= 2:
            # Strip backticks so phrase enters Pasada 1 cleanly as behavioral description rather than literal `sym`
            return sent.replace("`", "")
    return None


def _build_n3_mutation(pr: dict, ledger) -> tuple[str, str]:
    """Deterministically mutate or prepend one verifiable factual contradiction against the diff."""
    body = pr["body"]
    # Priority 1: dependency version mismatch
    for dep in ledger.dependencies:
        if dep.get("version"):
            fake_ver = "99.9.9" if not str(dep["version"]).startswith("99.9") else "88.8.8"
            inj = f"Updates dependency {dep['package']} to version {fake_ver}."
            return _inject_sentence(body, inj), "dependency_version_mismatch"

    # Priority 2: explicit unit test claim when diff changes 0 test files and 0 test functions
    if ledger.category_counts["test"] == 0 and not ledger.add_tests:
        sym = next((s for s in ledger.add_syms if len(s) >= 4 and s.lower() not in CODE_STOPWORDS), None)
        tgt_txt = f" `{sym}`" if sym else " the updated module"
        inj = f"Adds comprehensive unit tests covering{tgt_txt} behavior."
        return _inject_sentence(body, inj), "claimed_tests_but_zero_tests"

    # Priority 3: explicit README documentation claim when diff changes 0 doc files
    if ledger.category_counts["docs"] == 0:
        sym = next((s for s in ledger.add_syms if len(s) >= 4 and s.lower() not in CODE_STOPWORDS), None)
        tgt_txt = f" `{sym}`" if sym else " the new changes"
        inj = f"Updates README documentation file describing{tgt_txt}."
        return _inject_sentence(body, inj), "claimed_doc_file_but_zero_docs"

    # Priority 4: add <-> remove polarity flip on a concrete added symbol
    net_add = [s for s in ledger.add_syms if s not in ledger.rem_syms and len(s) >= 4 and s.lower() not in CODE_STOPWORDS]
    if net_add:
        sym = net_add[0]
        inj = f"Removes `{sym}` completely from the codebase."
        return _inject_sentence(body, inj), "polarity_flip_remove_added_symbol"

    # Priority 5: contradicted file count
    wrong_n = ledger.files_count + 17
    inj = f"This pull request modifies {wrong_n} files across the repository."
    return _inject_sentence(body, inj), "files_count_mismatch"


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare AIDev-Dev and Eval B splits externally.")
    ap.add_argument("--aidev-dir", default=os.path.expanduser("~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3"))
    ap.add_argument("--prmci-dir", default=os.path.expanduser("~/Documents/Evidence Intelligence/caso_prmci_llm/PR-MCI"))
    ap.add_argument("--dev-per-agent", type=int, default=40)
    ap.add_argument("--eval-b-per-agent", type=int, default=25)
    ap.add_argument("--max-donors-per-repo", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.aidev_dir
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(SEED)

    prmci_ids, prmci_repos = _load_prmci_exclusions(args.prmci_dir)

    repo_path = os.path.join(out_dir, "repository.parquet")
    pr_path = os.path.join(out_dir, "pull_request.parquet")
    if not os.path.exists(repo_path):
        hf_hub_download(repo_id=AIDEV_REPO, filename="repository.parquet", repo_type="dataset", revision=AIDEV_REVISION, local_dir=out_dir)
    if not os.path.exists(pr_path):
        hf_hub_download(repo_id=AIDEV_REPO, filename="pull_request.parquet", repo_type="dataset", revision=AIDEV_REVISION, local_dir=out_dir)

    idx_path = os.path.join(out_dir, "pr_rg_index.json")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Expected {idx_path} index; build it first or download commit details.")
    pr_to_rgs = json.load(open(idx_path, encoding="utf-8"))["pr_to_rgs"]

    t_repo = pq.read_table(repo_path, columns=["id", "full_name", "license", "language", "stars"])
    repo_info = {}
    for rid, fn, lic, lang, stars in zip(
        t_repo["id"].to_pylist(), t_repo["full_name"].to_pylist(),
        t_repo["license"].to_pylist(), t_repo["language"].to_pylist(), t_repo["stars"].to_pylist(),
    ):
        if lic in PERMISSIVE_LICENSES and fn:
            fn_l = fn.lower()
            if fn_l not in prmci_repos:
                repo_info[rid] = {"repo": fn_l, "license": lic, "language": lang or "Unknown", "stars": int(stars or 0)}

    t_pr = pq.read_table(pr_path, columns=["id", "number", "title", "body", "agent", "state", "repo_id", "html_url"])
    allowed_rg_set = set(TARGET_RGS)
    pool_by_agent_repo: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))

    for pid, num, title, body, ag, st, rid, url in zip(
        t_pr["id"].to_pylist(), t_pr["number"].to_pylist(), t_pr["title"].to_pylist(),
        t_pr["body"].to_pylist(), t_pr["agent"].to_pylist(), t_pr["state"].to_pylist(),
        t_pr["repo_id"].to_pylist(), t_pr["html_url"].to_pylist(),
    ):
        if ag not in AGENTS or st != "closed":
            continue
        spid = str(pid)
        if spid in prmci_ids:
            continue
        rgs = pr_to_rgs.get(spid)
        if not rgs or not set(rgs) <= allowed_rg_set:
            continue
        ri = repo_info.get(rid)
        if not ri:
            continue
        t_str = (title or "").strip()
        b_str = (body or "").strip()
        if not _is_english_ascii(t_str, b_str):
            continue
        if not _extract_candidate_behavior_sentences(b_str):
            continue
        pool_by_agent_repo[ag][ri["repo"]].append({
            "pr_id": spid,
            "number": int(num),
            "agent": ag,
            "title": t_str,
            "body": b_str,
            "url": url or f"https://github.com/{ri['repo']}/pull/{num}",
            "repo": ri["repo"],
            "repo_id": int(rid),
            "license": ri["license"],
            "language": ri["language"],
            "rgs": rgs,
        })

    # Step A: Select strictly repo-disjoint oversampled candidate pools per agent:
    # - AIDev-Dev prioritizes multi-PR repos (>=2 eligible PRs in repo, max 2 Dev targets/repo + remaining as N2 donor candidates)
    # - Eval B selects strictly disjoint repos not used anywhere in Dev
    dev_pool_target = args.dev_per_agent + 15
    eval_b_pool_target = args.eval_b_per_agent + 10
    dev_cands_by_agent: dict[str, list[dict]] = collections.defaultdict(list)
    eval_b_cands_by_agent: dict[str, list[dict]] = collections.defaultdict(list)
    dev_n2_donors_by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    dev_repos: set[str] = set()
    eval_b_repos: set[str] = set()

    for ag in AGENTS:
        repos_map = pool_by_agent_repo[ag]
        multi_names = sorted(r for r, lst in repos_map.items() if len(lst) >= 2 and r not in dev_repos and r not in eval_b_repos)
        rng.shuffle(multi_names)

        single_names = sorted(r for r, lst in repos_map.items() if len(lst) == 1 and r not in dev_repos and r not in eval_b_repos)
        rng.shuffle(single_names)

        ag_dev: list[dict] = []
        ag_dev_repos: list[str] = []
        # Reserve tail multi-PR repos for Eval B first so Eval B always stays strictly repo-disjoint
        max_dev_repos = max(dev_pool_target // 2, len(multi_names) - eval_b_pool_target)
        dev_repo_cands = multi_names[:max_dev_repos]
        for rname in dev_repo_cands:
            if len(ag_dev) >= dev_pool_target:
                break
            prs_r = sorted(repos_map[rname], key=lambda x: x["pr_id"])
            rng.shuffle(prs_r)
            ag_dev.append(prs_r[0])
            dev_n2_donors_by_repo[rname].extend(prs_r[1 : 1 + args.max_donors_per_repo])
            ag_dev_repos.append(rname)
            dev_repos.add(rname)
        if len(ag_dev) < dev_pool_target:
            for rname in ag_dev_repos:
                if len(ag_dev) >= dev_pool_target:
                    break
                donors = dev_n2_donors_by_repo[rname]
                if len(donors) >= 2:
                    ag_dev.append(donors.pop(0))
        if len(ag_dev) < dev_pool_target:
            extra = [r for r in single_names if r not in dev_repos and r not in eval_b_repos]
            for rname in extra:
                if len(ag_dev) >= dev_pool_target:
                    break
                ag_dev.append(repos_map[rname][0])
                dev_repos.add(rname)

        assert len(ag_dev) >= args.dev_per_agent, f"{ag}: only {len(ag_dev)} dev candidates"
        dev_cands_by_agent[ag] = ag_dev

        avail_eval_b = [r for r in (multi_names[max_dev_repos:] + single_names) if r not in dev_repos and r not in eval_b_repos]
        rng.shuffle(avail_eval_b)
        ag_evb: list[dict] = []
        for rname in avail_eval_b:
            if len(ag_evb) >= eval_b_pool_target:
                break
            prs_r = sorted(repos_map[rname], key=lambda x: x["pr_id"])
            rng.shuffle(prs_r)
            ag_evb.append(prs_r[0])
            eval_b_repos.add(rname)
        if len(ag_evb) < eval_b_pool_target:
            for rname in avail_eval_b:
                if len(ag_evb) >= eval_b_pool_target:
                    break
                prs_r = repos_map[rname]
                if len(prs_r) > 1:
                    ag_evb.append(prs_r[1])
        assert len(ag_evb) >= args.eval_b_per_agent, f"{ag}: only {len(ag_evb)} eval_b candidates"
        eval_b_cands_by_agent[ag] = ag_evb

    assert not (dev_repos & eval_b_repos), "Dev and Eval B candidate repos must be disjoint!"

    # Needed PR IDs to fetch commits for: oversampled dev/eval_b candidates + dev_n2_donors
    dev_cand_all = [x for ag in AGENTS for x in dev_cands_by_agent[ag]]
    evb_cand_all = [x for ag in AGENTS for x in eval_b_cands_by_agent[ag]]
    all_donor_recs = [d for lst in dev_n2_donors_by_repo.values() for d in lst]
    all_needed = dev_cand_all + evb_cand_all + all_donor_recs
    needed_ids_int = {int(x["pr_id"]) for x in all_needed}
    needed_rgs = sorted({rg for x in all_needed for rg in x["rgs"]})

    commits_path = os.path.join(out_dir, "aidev_selected_commits.parquet")
    commits_by_pr: dict[str, list[dict]] = collections.defaultdict(list)
    cols = ["sha", "pr_id", "filename", "status", "additions", "deletions", "patch"]
    existing_tables: list[pa.Table] = []
    existing_ids: set[int] = set()
    if os.path.exists(commits_path):
        existing_tab = pq.read_table(commits_path, columns=cols)
        if existing_tab.num_rows > 0:
            existing_tables.append(existing_tab)
            existing_ids = set(existing_tab["pr_id"].to_pylist())

    missing_ids = needed_ids_int - existing_ids
    if missing_ids:
        missing_rgs = sorted({rg for x in all_needed if int(x["pr_id"]) in missing_ids for rg in x["rgs"]})
        print(f"Fetching commits for {len(missing_ids)} missing PRs (of {len(needed_ids_int)}) across row groups {missing_rgs}...")
        fs = HfFileSystem()
        rows_tables = list(existing_tables)
        import pyarrow.compute as pc

        with fs.open(f"datasets/{AIDEV_REPO}@{AIDEV_REVISION}/pr_commit_details.parquet", "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg_i in missing_rgs:
                tab = pf.read_row_group(rg_i, columns=cols)
                mask = pc.is_in(tab["pr_id"], value_set=pa.array(sorted(missing_ids), type=pa.int64()))
                filt = tab.filter(mask)
                if filt.num_rows > 0:
                    rows_tables.append(filt)
                print(f"  RG {rg_i:2d}: filtered {filt.num_rows} new commit file rows")
        combined = pa.concat_tables(rows_tables) if rows_tables else pa.table({c: [] for c in cols})
        pq.write_table(combined, commits_path)
    else:
        combined = existing_tables[0]

    for sha, pid, fn, st, adds, dels, patch in zip(
        combined["sha"].to_pylist(), combined["pr_id"].to_pylist(), combined["filename"].to_pylist(),
        combined["status"].to_pylist(), combined["additions"].to_pylist(),
        combined["deletions"].to_pylist(), combined["patch"].to_pylist(),
    ):
        commits_by_pr[str(pid)].append({
            "sha": sha or "",
            "filename": fn or "",
            "status": st or "",
            "additions": int(adds or 0),
            "deletions": int(dels or 0),
            "patch": patch or "",
        })

    def _hydrate_pr(rec: dict) -> dict | None:
        fps = [fp for fp in commits_by_pr.get(rec["pr_id"], ()) if fp["filename"] or fp["patch"]]
        if not fps or not any((fp["patch"] or "").strip() for fp in fps):
            return None
        files = list(dict.fromkeys(fp["filename"] for fp in fps if fp["filename"]))
        adds = sum(fp["additions"] for fp in fps)
        dels = sum(fp["deletions"] for fp in fps)
        patch = "\n---FILE---\n".join(fp["patch"] for fp in fps if (fp["patch"] or "").strip())
        return {
            **{k: v for k, v in rec.items() if k != "rgs"},
            "files": files,
            "file_patches": fps,
            "patch": patch,
            "adds": adds,
            "dels": dels,
            "label": "unknown",
        }

    # Hydrate exact balanced counts per agent (40 Dev, 25 Eval B)
    dev_hydrated: list[dict] = []
    donor_hydrated_by_repo: dict[str, list[dict]] = collections.defaultdict(list)
    for ag in AGENTS:
        kept_ag: list[dict] = []
        for rec in dev_cands_by_agent[ag]:
            h = _hydrate_pr(rec)
            if h is None:
                continue
            if len(kept_ag) < args.dev_per_agent:
                kept_ag.append(h)
            else:
                donor_hydrated_by_repo[h["repo"]].append(h)
        assert len(kept_ag) == args.dev_per_agent, f"{ag}: hydrated {len(kept_ag)} < {args.dev_per_agent}"
        dev_hydrated.extend(kept_ag)

    eval_b_hydrated: list[dict] = []
    for ag in AGENTS:
        kept_ag = []
        for rec in eval_b_cands_by_agent[ag]:
            h = _hydrate_pr(rec)
            if h is not None:
                kept_ag.append(h)
                if len(kept_ag) == args.eval_b_per_agent:
                    break
        assert len(kept_ag) == args.eval_b_per_agent, f"{ag}: hydrated {len(kept_ag)} < {args.eval_b_per_agent}"
        eval_b_hydrated.extend(kept_ag)

    for rname, lst in dev_n2_donors_by_repo.items():
        for rec in lst:
            h = _hydrate_pr(rec)
            if h is not None:
                donor_hydrated_by_repo[rname].append(h)

    final_dev_repos = {r["repo"] for r in dev_hydrated}
    final_eval_b_repos = {r["repo"] for r in eval_b_hydrated}
    assert len(dev_hydrated) == 5 * args.dev_per_agent
    assert len(eval_b_hydrated) == 5 * args.eval_b_per_agent
    assert not (final_dev_repos & final_eval_b_repos)
    assert not ({r["pr_id"] for r in dev_hydrated + eval_b_hydrated} & prmci_ids)
    assert not ((final_dev_repos | final_eval_b_repos) & prmci_repos)

    print(f"Hydrated exact Dev={len(dev_hydrated)} ({args.dev_per_agent}/agent), EvalB={len(eval_b_hydrated)} ({args.eval_b_per_agent}/agent)")

    # Build N1 (cross-repo easy), N2 (same-repo hard), N3 (factual contradiction mutation)
    ledgers_by_id = {p["pr_id"]: build_typed_diff_ledger(p) for p in dev_hydrated}
    diff_syms_by_id = {pid: _target_diff_symbol_set(ld) for pid, ld in ledgers_by_id.items()}

    n1_rows, n2_rows, n3_rows = [], [], []
    # Pool of all hydrated Dev + donor candidates for N1 cross-repo owner lookup
    n1_donor_pool = list(dev_hydrated)
    n3_type_counts = collections.Counter()

    for idx, pr in enumerate(dev_hydrated):
        pid = pr["pr_id"]
        t_owner = pr["repo"].split("/")[0]
        t_syms = diff_syms_by_id[pid]
        ledger = ledgers_by_id[pid]

        # N1 easy: donor from strictly different owner & repo with 0 symbol/path overlap
        n1_phrase, n1_donor_id, n1_donor_repo = None, None, None
        for shift in range(1, len(n1_donor_pool)):
            cand = n1_donor_pool[(idx + shift * 7) % len(n1_donor_pool)]
            c_owner = cand["repo"].split("/")[0]
            if cand["repo"] == pr["repo"] or c_owner == t_owner:
                continue
            ph = _select_phantom_phrase(cand["body"], t_syms, cand["repo"])
            if ph:
                n1_phrase, n1_donor_id, n1_donor_repo = ph, cand["pr_id"], cand["repo"]
                break
        if n1_phrase:
            n1_rows.append({
                **pr,
                "pr_id": f"{pid}__N1",
                "base_pr_id": pid,
                "injection_level": "N1_easy",
                "injected_phrase": n1_phrase,
                "donor_pr_id": n1_donor_id,
                "donor_repo": n1_donor_repo,
                "body": _inject_sentence(pr["body"], n1_phrase),
            })

        # N2 hard: donor from SAME repo whose changed files/symbols do NOT appear in target PR's diff
        same_repo_cands = [
            d for d in (donor_hydrated_by_repo.get(pr["repo"], []) + [x for x in dev_hydrated if x["repo"] == pr["repo"] and x["pr_id"] != pid])
            if d["pr_id"] != pid
        ]
        n2_phrase, n2_donor_id = None, None
        for d in same_repo_cands:
            d_ledger = build_typed_diff_ledger(d)
            d_diff_syms = _target_diff_symbol_set(d_ledger)
            # Also exclude donor PR's phrases whose tokens actually ARE in target PR's diff!
            ph = _select_phantom_phrase(d["body"], t_syms, "")
            if ph and not (d_diff_syms & t_syms):
                n2_phrase, n2_donor_id = ph, d["pr_id"]
                break
        if n2_phrase is None:
            # Relax whole-diff overlap check between donor & target while keeping strict 0 overlap on injected sentence tokens
            for d in same_repo_cands:
                ph = _select_phantom_phrase(d["body"], t_syms, "")
                if ph:
                    n2_phrase, n2_donor_id = ph, d["pr_id"]
                    break
        if n2_phrase:
            n2_rows.append({
                **pr,
                "pr_id": f"{pid}__N2",
                "base_pr_id": pid,
                "injection_level": "N2_hard",
                "injected_phrase": n2_phrase,
                "donor_pr_id": n2_donor_id,
                "donor_repo": pr["repo"],
                "body": _inject_sentence(pr["body"], n2_phrase),
            })

        # N3 mutation: factual flip on target PR itself
        mut_body, mut_kind = _build_n3_mutation(pr, ledger)
        n3_type_counts[mut_kind] += 1
        n3_rows.append({
            **pr,
            "pr_id": f"{pid}__N3",
            "base_pr_id": pid,
            "injection_level": "N3_mutation",
            "mutation_kind": mut_kind,
            "body": mut_body,
        })

    def _write_jsonl(path: str, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    _write_jsonl(os.path.join(out_dir, "aidev_dev_originals.jsonl"), dev_hydrated)
    _write_jsonl(os.path.join(out_dir, "aidev_dev_n1_easy.jsonl"), n1_rows)
    _write_jsonl(os.path.join(out_dir, "aidev_dev_n2_hard.jsonl"), n2_rows)
    _write_jsonl(os.path.join(out_dir, "aidev_dev_n3_mutation.jsonl"), n3_rows)
    _write_jsonl(os.path.join(out_dir, "aidev_eval_b.jsonl"), eval_b_hydrated)

    # Save ONLY sanitized manifests (no PR bodies/diffs/titles) inside the git repo
    def _manifest_entry(rec: dict, **extra) -> dict:
        return {
            "pr_id": rec["pr_id"],
            "agent": rec["agent"],
            "repo": rec["repo"],
            "license": rec["license"],
            "language": rec["language"],
            "url": rec["url"],
            "files_changed": len(rec["files"]),
            "adds": rec["adds"],
            "dels": rec["dels"],
            **extra,
        }

    dev_manifest = {
        "source": f"{AIDEV_REPO}@{AIDEV_REVISION}",
        "dataset_card_license": AIDEV_LICENSE,
        "repo_licenses_filter": sorted(PERMISSIVE_LICENSES),
        "seed": SEED,
        "counts": {
            "originals": len(dev_hydrated),
            "n1_easy": len(n1_rows),
            "n2_hard": len(n2_rows),
            "n3_mutation": len(n3_rows),
            "n3_mutation_kinds": dict(sorted(n3_type_counts.items())),
            "distinct_repos": len({r["repo"] for r in dev_hydrated}),
            "by_agent_originals": dict(sorted(collections.Counter(r["agent"] for r in dev_hydrated).items())),
            "by_agent_n2": dict(sorted(collections.Counter(r["agent"] for r in n2_rows).items())),
        },
        "originals": [_manifest_entry(r) for r in dev_hydrated],
        "n1_easy": [
            {"pr_id": r["pr_id"], "base_pr_id": r["base_pr_id"], "agent": r["agent"], "donor_pr_id": r["donor_pr_id"], "donor_repo": r["donor_repo"]}
            for r in n1_rows
        ],
        "n2_hard": [
            {"pr_id": r["pr_id"], "base_pr_id": r["base_pr_id"], "agent": r["agent"], "donor_pr_id": r["donor_pr_id"], "donor_repo": r["donor_repo"]}
            for r in n2_rows
        ],
        "n3_mutation": [
            {"pr_id": r["pr_id"], "base_pr_id": r["base_pr_id"], "agent": r["agent"], "mutation_kind": r["mutation_kind"]}
            for r in n3_rows
        ],
    }
    eval_b_manifest = {
        "source": f"{AIDEV_REPO}@{AIDEV_REVISION}",
        "dataset_card_license": AIDEV_LICENSE,
        "repo_licenses_filter": sorted(PERMISSIVE_LICENSES),
        "seed": SEED,
        "counts": {
            "total": len(eval_b_hydrated),
            "distinct_repos": len({r["repo"] for r in eval_b_hydrated}),
            "by_agent": dict(sorted(collections.Counter(r["agent"] for r in eval_b_hydrated).items())),
        },
        "rows": [_manifest_entry(r) for r in eval_b_hydrated],
    }
    with open(os.path.join(_THIS_DIR, "aidev_dev_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(dev_manifest, fh, indent=2, sort_keys=True)
    with open(os.path.join(_THIS_DIR, "eval_b_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(eval_b_manifest, fh, indent=2, sort_keys=True)

    print("Prepared manifests cleanly:")
    print("  AIDev-Dev:", json.dumps(dev_manifest["counts"], indent=2))
    print("  Eval B:", json.dumps(eval_b_manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
