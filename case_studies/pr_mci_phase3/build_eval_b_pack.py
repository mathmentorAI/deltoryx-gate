"""Build external blank 2-annotator human labeling pack for Eval B (n=125).

Writes strictly OUTSIDE git (default:
`~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/eval_b_pack_local/`):
- `annotator_1.csv` (blank `label`, `evidence_note`, `confidence`)
- `annotator_2.csv` (independent blank copy with different sheet presentation order)
- `adjudication.csv` (blank consensus sheet for third adjudicator / disagreement resolution)
- `sheets/item_XXX.md` (blinded PR description + changed files + diff hunks, omitting agent name)
- `blind_key.json` (private mapping item_id <-> pr_id/agent kept strictly outside git)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Any

DEFAULT_IN_JSONL = os.path.expanduser(
    "~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/aidev_eval_b.jsonl"
)
DEFAULT_OUT_DIR = os.path.expanduser(
    "~/Documents/Evidence Intelligence/caso_prmci_llm/AIDev_phase3/eval_b_pack_local"
)
PACK_SEED = 20261120


def _render_sheet(item_id: str, pr: dict[str, Any], *, max_diff_chars: int = 36000) -> str:
    lines: list[str] = [
        f"# Evaluation Item {item_id}",
        "",
        f"- **Repository:** `{pr.get('repo', '')}` (License: `{pr.get('license', '')}`)",
        f"- **PR Number:** `{pr.get('number', '')}`",
        f"- **URL:** {pr.get('url', '')}",
        f"- **Changed files count:** {len(pr.get('files', ()))} (+{pr.get('adds', 0)} / -{pr.get('dels', 0)})",
        "",
        "## Title",
        "",
        str(pr.get("title") or "").strip(),
        "",
        "## Description",
        "",
        str(pr.get("body") or "").strip(),
        "",
        "## Changed Files",
        "",
    ]
    for fn in pr.get("files", ()):
        lines.append(f"- `{fn}`")
    lines.extend(["", "## Diff", ""])

    rendered_chars = 0
    fps = pr.get("file_patches") or ()
    if fps:
        for fp in fps:
            fname = fp.get("filename") or "<unknown>"
            status = fp.get("status") or ""
            patch = (fp.get("patch") or "").strip()
            block = f"### `{fname}` ({status})\n```diff\n{patch}\n```\n"
            if rendered_chars + len(block) > max_diff_chars and rendered_chars > 0:
                lines.append("_[Remaining file diffs omitted from sheet preview; inspect GitHub URL if needed.]_")
                break
            lines.append(block)
            rendered_chars += len(block)
    else:
        patch = str(pr.get("patch") or "")[:max_diff_chars]
        lines.append(f"```diff\n{patch}\n```")

    lines.extend(
        [
            "",
            "## Label Rubric Reminder",
            "- `aligned`: All concrete claims in the title/body match the diff; any non-code summary statements are faithful.",
            "- `partial`: Main change is real, but minor claims, file lists, tests, docs, or unmentioned secondary edits overstate/omit material scope.",
            "- `misaligned`: Description fabricates functionality, contradicts the diff, cites unedited modules/tests/docs as modified, or fundamentally misrepresents the patch.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_pack(in_jsonl: str, out_dir: str, *, seed: int = PACK_SEED) -> dict[str, Any]:
    with open(in_jsonl, encoding="utf-8") as fh:
        prs = [json.loads(ln) for ln in fh if ln.strip()]

    # Deterministic blind shuffling so annotators do not see agent clusters or model predictions
    rng1 = random.Random(seed)
    order1 = list(range(len(prs)))
    rng1.shuffle(order1)

    rng2 = random.Random(seed + 1)
    order2 = list(range(len(prs)))
    rng2.shuffle(order2)

    sheets_dir = os.path.join(out_dir, "sheets")
    os.makedirs(sheets_dir, exist_ok=True)

    item_by_idx: dict[int, str] = {}
    blind_key: list[dict[str, Any]] = []
    for seq, orig_idx in enumerate(order1, start=1):
        item_id = f"EB-{seq:03d}"
        item_by_idx[orig_idx] = item_id
        pr = prs[orig_idx]
        sheet_rel = f"sheets/{item_id}.md"
        with open(os.path.join(out_dir, sheet_rel), "w", encoding="utf-8") as sfh:
            sfh.write(_render_sheet(item_id, pr))
        blind_key.append(
            {
                "item_id": item_id,
                "pr_id": str(pr["pr_id"]),
                "agent": pr["agent"],
                "repo": pr["repo"],
                "sheet_path": sheet_rel,
            }
        )

    cols = [
        "item_id",
        "pr_id",
        "repo",
        "url",
        "files_changed",
        "sheet_file",
        "label",
        "evidence_note",
        "confidence",
    ]

    def _write_csv(filename: str, order: list[int], *, include_adj_cols: bool = False) -> None:
        fp = os.path.join(out_dir, filename)
        with open(fp, "w", newline="", encoding="utf-8") as cfh:
            hdr = cols + (["annotator1_label", "annotator2_label", "consensus_label", "adjudication_note"] if include_adj_cols else [])
            w = csv.DictWriter(cfh, fieldnames=hdr)
            w.writeheader()
            for idx in order:
                pr = prs[idx]
                item_id = item_by_idx[idx]
                row = {
                    "item_id": item_id,
                    "pr_id": str(pr["pr_id"]),
                    "repo": pr["repo"],
                    "url": pr["url"],
                    "files_changed": len(pr.get("files", ())),
                    "sheet_file": f"sheets/{item_id}.md",
                    "label": "",
                    "evidence_note": "",
                    "confidence": "",
                }
                if include_adj_cols:
                    row.update(
                        {
                            "annotator1_label": "",
                            "annotator2_label": "",
                            "consensus_label": "",
                            "adjudication_note": "",
                        }
                    )
                w.writerow(row)

    _write_csv("annotator_1.csv", order1)
    _write_csv("annotator_2.csv", order2)
    _write_csv("adjudication.csv", sorted(order1, key=lambda idx: item_by_idx[idx]), include_adj_cols=True)

    with open(os.path.join(out_dir, "blind_key.json"), "w", encoding="utf-8") as kfh:
        json.dump(blind_key, kfh, indent=2, sort_keys=True)

    return {
        "total_items": len(prs),
        "out_dir": out_dir,
        "files": ["annotator_1.csv", "annotator_2.csv", "adjudication.csv", "blind_key.json", "sheets/*.md"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate external blank 2-annotator Eval B pack.")
    ap.add_argument("--in-jsonl", default=DEFAULT_IN_JSONL)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    summary = build_pack(args.in_jsonl, args.out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
