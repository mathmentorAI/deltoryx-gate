"""SEC EDGAR public-document fetcher and immutable provenance manifest builder.

Downloads public securitization (424B* prospectus supplement vs EX-4.* indenture /
pooling-and-servicing agreement) and corporate credit (credit agreement + amendment)
pairs from SEC EDGAR, normalizes text, and records source URLs, UTC timestamps,
and SHA-256 checksums into an immutable ``manifest.json``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_SECURITIZATION_FORMS = ("424B2", "424B5")
DEFAULT_SECURITIZATION_EXHIBIT_PREFIXES = ("EX-4.", "EX-10.", "EX-99.")
DEFAULT_CREDIT_FORMS = ("8-K", "10-K", "10-Q")
DEFAULT_CREDIT_EXHIBIT_PREFIXES = ("EX-10.", "EX-4.")


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML/XBRL to readable text normalizer."""

    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
    })
    _SKIP_TAGS = frozenset({"script", "style", "head"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and t in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif self._skip_depth == 0 and t in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self._chunks.append(data)

    def get_text(self) -> str:
        raw = "".join(self._chunks)
        lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in raw.splitlines()]
        out_lines = [ln for ln in lines if ln]
        return "\n".join(out_lines)


def html_or_text_to_clean_text(raw: str) -> str:
    """Convert raw HTML/SGML/plain SEC filing payload into normalized UTF-8 text."""
    stripped = (raw or "").lstrip()
    if stripped.startswith("<") or "<html" in stripped[:500].lower() or "<p" in stripped[:500].lower():
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        parser.close()
        return parser.get_text()
    lines = [re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in (raw or "").splitlines()]
    return "\n".join(ln for ln in lines if ln)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EdgarFetchSpec:
    """Target SEC CIK + filing pair specification for fetching."""

    corpus_id: str
    domain: str  # "securitization" | "banking_finance"
    cik: str
    primary_accession: str = ""
    primary_filename: str = ""
    secondary_accession: str = ""
    secondary_filename: str = ""
    max_chars_per_doc: int = 24000


class EdgarHttpClient:
    """Rate-limited SEC EDGAR HTTP client requiring an explicit User-Agent string."""

    def __init__(self, user_agent: str, *, min_interval_seconds: float = 0.12, timeout: float = 30.0) -> None:
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC EDGAR requires a descriptive User-Agent header including contact info "
                "(e.g. 'ClaimLayerResearch/1.0 research@example.org')."
            )
        self.user_agent = user_agent.strip()
        self.min_interval_seconds = max(0.10, float(min_interval_seconds))
        self.timeout = float(timeout)
        self._last_request_ts = 0.0

    def get_bytes(self, url: str) -> bytes:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        req = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        finally:
            self._last_request_ts = time.monotonic()

    def get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self.get_bytes(url).decode("utf-8"))


def normalize_cik(cik: str) -> tuple[str, str]:
    digits = re.sub(r"\D", "", str(cik or ""))
    if not digits:
        raise ValueError(f"Invalid CIK: {cik!r}")
    return str(int(digits)), digits.zfill(10)


def build_edgar_doc_url(cik: str, accession_number: str, filename: str) -> str:
    cik_int, _ = normalize_cik(cik)
    acc_nodash = re.sub(r"[^0-9]", "", str(accession_number or ""))
    if not acc_nodash or not filename:
        raise ValueError("accession_number and filename must be non-empty")
    return f"{SEC_ARCHIVES_BASE}/{cik_int}/{acc_nodash}/{filename.lstrip('/')}"


def select_filing_pairs_from_submissions(
    submissions: Mapping[str, Any],
    *,
    domain: str,
) -> list[dict[str, str]]:
    """Deterministically discover candidate accession/document pairs from SEC submissions JSON."""
    recent = ((submissions.get("filings") or {}).get("recent") or {})
    forms = list(recent.get("form") or ())
    accessions = list(recent.get("accessionNumber") or ())
    primary_docs = list(recent.get("primaryDocument") or ())
    dates = list(recent.get("filingDate") or ())

    rows = [
        {
            "form": str(forms[i] if i < len(forms) else ""),
            "accession": str(accessions[i] if i < len(accessions) else ""),
            "primary_document": str(primary_docs[i] if i < len(primary_docs) else ""),
            "filing_date": str(dates[i] if i < len(dates) else ""),
        }
        for i in range(min(len(forms), len(accessions), len(primary_docs)))
    ]
    rows.sort(key=lambda r: (r["filing_date"], r["accession"]), reverse=True)

    target_forms = DEFAULT_SECURITIZATION_FORMS if domain == "securitization" else DEFAULT_CREDIT_FORMS
    out: list[dict[str, str]] = []
    for row in rows:
        if any(row["form"].upper().startswith(prefix) for prefix in target_forms):
            out.append(row)
    return out


def fetch_edgar_corpus(
    specs: Sequence[EdgarFetchSpec],
    *,
    output_dir: Path,
    client: EdgarHttpClient | None = None,
    user_agent: str | None = None,
    fetcher: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Download external SEC EDGAR document pairs and write an immutable manifest.json.

    Either *fetcher* (dependency injection for offline/test replay) or *client* / *user_agent*
    must be supplied. Writes normalized UTF-8 ``.txt`` documents and ``manifest.json``
    inside *output_dir*.
    """
    if fetcher is None:
        if client is None:
            if not user_agent:
                raise ValueError("Must provide client, user_agent, or custom fetcher.")
            client = EdgarHttpClient(user_agent=user_agent)
        fetcher = client.get_bytes

    output_dir = Path(output_dir)
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing immutable manifest at {manifest_path}. "
            "Use a new output directory."
        )

    corpus_entries: list[dict[str, Any]] = []
    for spec in specs:
        if spec.domain not in ("securitization", "banking_finance"):
            raise ValueError(f"Unsupported domain {spec.domain!r}")
        if not (spec.primary_accession and spec.primary_filename and spec.secondary_accession and spec.secondary_filename):
            raise ValueError(
                f"Spec {spec.corpus_id!r} requires explicit primary and secondary accession + filename."
            )

        corpus_docs_dir = docs_dir / spec.corpus_id
        corpus_docs_dir.mkdir(parents=True, exist_ok=True)

        pair_records: list[dict[str, Any]] = []
        role_cfgs = (
            ("primary", "governing_operative_agreement", "authoritative", 100, spec.primary_accession, spec.primary_filename),
            (
                "secondary",
                "disclosure_document" if spec.domain == "securitization" else "counterparty_markup",
                "standard",
                60,
                spec.secondary_accession,
                spec.secondary_filename,
            ),
        )
        for slot, role, tier, rank, acc, fname in role_cfgs:
            url = build_edgar_doc_url(spec.cik, acc, fname)
            fetched_at = datetime.now(timezone.utc).isoformat()
            raw_bytes = fetcher(url)
            raw_sha = sha256_bytes(raw_bytes)
            clean_txt = html_or_text_to_clean_text(raw_bytes.decode("utf-8", errors="replace"))
            if spec.max_chars_per_doc > 0 and len(clean_txt) > spec.max_chars_per_doc:
                clean_txt = clean_txt[: spec.max_chars_per_doc]
            norm_sha = sha256_text(clean_txt)
            local_name = f"{slot}__{Path(fname).stem}.txt"
            rel_path = f"documents/{spec.corpus_id}/{local_name}"
            (corpus_docs_dir / local_name).write_text(clean_txt, encoding="utf-8")
            pair_records.append({
                "slot": slot,
                "document_id": local_name,
                "relative_path": rel_path,
                "source_url": url,
                "accession_number": acc,
                "sec_filename": fname,
                "downloaded_utc": fetched_at,
                "raw_sha256": raw_sha,
                "normalized_sha256": norm_sha,
                "normalized_chars": len(clean_txt),
                "role": role,
                "authority_tier": tier,
                "authority_rank": rank,
            })

        corpus_entries.append({
            "corpus_id": spec.corpus_id,
            "domain": spec.domain,
            "cik": normalize_cik(spec.cik)[1],
            "documents": pair_records,
        })

    manifest = {
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_provider": "SEC EDGAR",
        "corpora": corpus_entries,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = sha256_bytes(manifest_bytes)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_manifest_integrity(corpus_dir: Path) -> dict[str, Any]:
    """Verify every file listed in *corpus_dir/manifest.json* matches its recorded SHA-256."""
    corpus_dir = Path(corpus_dir)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for corp in manifest.get("corpora") or ():
        for doc in corp.get("documents") or ():
            fpath = corpus_dir / str(doc["relative_path"])
            if not fpath.exists():
                raise FileNotFoundError(f"Missing manifest document: {fpath}")
            actual = sha256_text(fpath.read_text(encoding="utf-8"))
            if actual != doc["normalized_sha256"]:
                raise ValueError(
                    f"SHA-256 mismatch for {fpath}: expected {doc['normalized_sha256']}, got {actual}"
                )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch SEC EDGAR external corpus pairs with immutable manifest.")
    parser.add_argument("--specs", type=Path, required=True, help="JSON file containing list of EdgarFetchSpec objects.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory (outside repository recommended).")
    parser.add_argument("--user-agent", required=True, help="Descriptive User-Agent with contact email per SEC EDGAR policy.")
    args = parser.parse_args()

    raw_specs = json.loads(args.specs.read_text(encoding="utf-8"))
    specs = [EdgarFetchSpec(**item) for item in raw_specs]
    manifest = fetch_edgar_corpus(specs, output_dir=args.output_dir, user_agent=args.user_agent)
    print(json.dumps({"created_utc": manifest["created_utc"], "manifest_sha256": manifest["manifest_sha256"], "corpora": len(manifest["corpora"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
