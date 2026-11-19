"""Parser for PR-MCI markdown annotation files."""
from __future__ import annotations

import collections
import glob
import os
import re

_DEFAULT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "PR-MCI")
SRC = os.environ.get("PRMCI_DIR", _DEFAULT_SRC)

_LEGACY_STOP = r"(?=^## (?:📁|🔧|🔖|✅|📊|📈|📝|📄)|\Z)"
_TEMPLATE_STOP = (
    r"(?=^## (?:📊 PR Metadata|📈 Code Change Statistics|📝 PR Title|"
    r"📄 PR Description \(Body\)|📁 Files Changed|🔧 Code Patches|"
    r"🔖 ANNOTATION SECTION|✅ Annotation Complete\?)|\Z)"
)


def section(txt: str, head: str, *, legacy: bool = False) -> str:
    stop = _LEGACY_STOP if legacy else _TEMPLATE_STOP
    m = re.search(r"^## " + re.escape(head) + r"[^\n]*\n(.*?)" + stop, txt, re.S | re.M)
    return m.group(1).strip() if m else ""


def field(txt: str, name: str) -> str | None:
    m = re.search(r"\*\*" + re.escape(name) + r":\*\*\s*(.+)", txt)
    return m.group(1).strip() if m else None


def load(agent: str = "Claude_Code", *, prmci_dir: str | None = None, legacy_section: bool = False) -> list[dict]:
    root = prmci_dir or os.environ.get("PRMCI_DIR") or SRC
    out = []
    for f in sorted(glob.glob(f"{root}/data/manual_annotations/pr_*_{agent}_*.md")):
        with open(f, encoding="utf-8") as fh:
            t = fh.read()
        body = section(t, "📄 PR Description (Body)", legacy=legacy_section)
        body = re.sub(r"^```\n?|\n?```$", "", body.strip())
        ann = t.split("ANNOTATION SECTION", 1)[1] if "ANNOTATION SECTION" in t else ""
        a1 = re.search(r"### Annotator 1.*?\*\*Label[^:]*:\*\*\s*(\w+)", ann, re.S)
        a2 = re.search(r"### Annotator 2.*?\*\*Label[^:]*:\*\*\s*(\w+)", ann, re.S)
        cons = field(ann, "Consensus Label")
        lab = cons if cons and cons.lower() not in ("", "n/a", "-") else (a1.group(1) if a1 else None)
        m = re.search(r"\*\*Lines:\*\*\s*\+(\d+)\s*/\s*-(\d+)", t)
        out.append(
            {
                "pr_id": field(t, "PR ID"),
                "url": field(t, "GitHub URL"),
                "task": field(t, "Task Type"),
                "title": re.sub(r"^```\n?|\n?```$", "", section(t, "📝 PR Title", legacy=legacy_section)).strip(),
                "body": body,
                "files": [
                    x.strip()
                    for x in section(t, "📁 Files Changed", legacy=legacy_section).replace("\n", "|").split("|")
                    if x.strip() and not x.strip().startswith("---")
                ],
                "patch": section(t, "🔧 Code Patches", legacy=legacy_section),
                "adds": int(m.group(1)) if m else None,
                "dels": int(m.group(2)) if m else None,
                "label": (lab or "").lower(),
                "a1": a1.group(1).lower() if a1 else None,
                "a2": a2.group(1).lower() if a2 else None,
                "file": os.path.basename(f),
            }
        )
    return out


if __name__ == "__main__":
    import csv

    prs = load()
    print(len(prs), collections.Counter(p["label"] for p in prs))
    csv.field_size_limit(10**9)
    tax = {r["pr_id"]: r for r in csv.DictReader(open(f"{SRC}/data/manual_annotations_rq2_taxonomy.csv", encoding="utf-8-sig"))}
    val = {r["pr_id"]: r for r in csv.DictReader(open(f"{SRC}/validation/validation_results.csv", encoding="utf-8-sig"))}
    print("in validation:", sum(p["pr_id"] in val for p in prs), "in taxonomy:", sum(p["pr_id"] in tax for p in prs))
    print(collections.Counter(tax[p["pr_id"]]["failure_type"] for p in prs if p["pr_id"] in tax))
