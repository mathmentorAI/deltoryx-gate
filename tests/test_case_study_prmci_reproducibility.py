"""Reproducibility and hygiene tests for the PR-MCI case study."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "case_studies" / "pr_mci_claude_code"
EXPECTED_GATE_COMMIT = "22e3590"
EXPECTED_FINGERPRINT = "620ebd71c7dda399cb8b90e5ac038a2a2cf510782a86202b79b75c8d84900d2c"


def _load_case_module(name: str):
    if str(CASE_DIR) not in sys.path:
        sys.path.insert(0, str(CASE_DIR))
    path = CASE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_prmci_test_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _DeterministicPRStubModel:
    """Offline stub model that emits schema-valid code_change v2.0 proposals."""

    @property
    def identity(self) -> dict[str, str]:
        return {"provider": "stub", "model": "stub-code-change-v2"}

    def complete(self, system: str, payload: dict) -> dict:
        fragments = payload.get("fragments") or []
        if not fragments:
            return {"output": {"proposals": []}}
        frag = fragments[0]
        eid = frag["evidence_id"]
        text = frag["text"]
        proposals = []

        if "src/auth.py" in text:
            proposals.append(
                {
                    "entity": "pull request",
                    "predicate": "changes_file",
                    "value": "src/auth.py",
                    "assertion_kind": "fact",
                    "basis": "explicit",
                    "evidence": [eid],
                    "quote": "src/auth.py",
                }
            )
        if "refresh_token" in text:
            proposals.append(
                {
                    "entity": "pull request",
                    "predicate": "adds_identifier",
                    "value": "refresh_token",
                    "assertion_kind": "fact",
                    "basis": "explicit",
                    "evidence": [eid],
                    "quote": "refresh_token",
                }
            )
        behavior_phrase = "improves token refresh flow"
        if behavior_phrase in text:
            proposals.append(
                {
                    "entity": "pull request",
                    "predicate": "describes_behavior",
                    "value": behavior_phrase,
                    "assertion_kind": "proposal",
                    "basis": "explicit",
                    "evidence": [eid],
                    "quote": behavior_phrase,
                }
            )
        return {"output": {"proposals": proposals}}


def test_vocab_v2_fingerprint_and_no_closed_workaround() -> None:
    eval_v2vocab = _load_case_module("eval_v2vocab")
    assert eval_v2vocab.EXPECTED_VOCAB_FINGERPRINT == EXPECTED_FINGERPRINT
    assert eval_v2vocab.CODE_CHANGE_VOCAB_V2.fingerprint == EXPECTED_FINGERPRINT
    for fname in ("eval_v2vocab.py", "reproduce.py"):
        src = (CASE_DIR / fname).read_text(encoding="utf-8")
        assert "verify_draft_closed" not in src


def test_eval_v2vocab_double_replay_bit_identical_with_receipts(tmp_path: Path) -> None:
    eval_v2vocab = _load_case_module("eval_v2vocab")
    from claimlayer.v4.reader import LLMProposalReader

    cache_path = tmp_path / "synthetic_v2_cache.jsonl"
    pr = {
        "pr_id": "9990001",
        "url": "https://example.invalid/org/repo/pull/1",
        "title": "Add refresh_token helper in src/auth.py",
        "body": "Updates src/auth.py with `refresh_token` and improves token refresh flow.",
        "files": ["src/auth.py"],
        "patch": (
            "@@ -1,2 +1,4 @@\n"
            "+def refresh_token(tok):\n"
            "+    return tok\n"
        ),
        "adds": 2,
        "dels": 0,
        "label": "aligned",
    }
    row_meta = {
        "pr_id": "9990001",
        "agent": "Synthetic_Agent",
        "consensus_label": "aligned",
        "annotator1_label": "aligned",
        "binary_label": "aligned",
        "predicted_heuristic": "aligned",
    }

    # Populate cache with stub model, then execute two independent offline replays.
    warm_model = eval_v2vocab.CachedModelV2(_DeterministicPRStubModel(), str(cache_path))
    warm_res = eval_v2vocab.evaluate_pr(row_meta, pr, LLMProposalReader(model=warm_model))
    assert cache_path.exists()

    replay_a = eval_v2vocab.CachedModelV2(None, str(cache_path))
    res_a = eval_v2vocab.evaluate_pr(row_meta, pr, LLMProposalReader(model=replay_a))

    replay_b = eval_v2vocab.CachedModelV2(None, str(cache_path))
    res_b = eval_v2vocab.evaluate_pr(row_meta, pr, LLMProposalReader(model=replay_b))

    bytes_a = json.dumps(res_a, sort_keys=True, ensure_ascii=False).encode("utf-8")
    bytes_b = json.dumps(res_b, sort_keys=True, ensure_ascii=False).encode("utf-8")
    bytes_w = json.dumps(warm_res, sort_keys=True, ensure_ascii=False).encode("utf-8")
    assert bytes_a == bytes_b == bytes_w
    assert hashlib.sha256(bytes_a).hexdigest() == hashlib.sha256(bytes_b).hexdigest()
    assert res_a["receipt_sha256"] == res_b["receipt_sha256"]
    assert res_a["gate"] == "CONSISTENT"
    assert res_a["claim_status"].get("supported", 0) >= 2
    assert res_a["claim_status"].get("unverifiable", 0) == 1


def test_parse_section_preserves_emoji_headings_by_default() -> None:
    parse = _load_case_module("parse")
    doc = (
        "## 📄 PR Description (Body)\n\n"
        "Intro summary.\n\n"
        "## 🚀 Features\n\n"
        "Adds `refresh_token` in src/auth.py.\n\n"
        "## ✅ Checklist\n\n"
        "- [x] Tests added\n\n"
        "## 📁 Files Changed\n\n"
        "src/auth.py\n"
    )
    default_body = parse.section(doc, "📄 PR Description (Body)")
    legacy_body = parse.section(doc, "📄 PR Description (Body)", legacy=True)
    assert "Adds `refresh_token` in src/auth.py." in default_body
    assert "Tests added" in default_body
    assert "Files Changed" not in default_body
    assert "Tests added" not in legacy_body


def _walk_forbidden_keys(obj: object, forbidden: set[str], trail: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in forbidden:
                hits.append(f"{trail}.{k}")
            hits.extend(_walk_forbidden_keys(v, forbidden, f"{trail}.{k}"))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            hits.extend(_walk_forbidden_keys(item, forbidden, f"{trail}[{idx}]"))
    return hits


def test_case_study_hygiene_no_secrets_caches_or_raw_pr_text() -> None:
    assert CASE_DIR.is_dir()
    assert not list(CASE_DIR.rglob("*.jsonl")), "LLM caches must stay outside the repository"

    for p in sorted(CASE_DIR.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        key_prefix = "nvapi" + "-"
        assert key_prefix not in text, f"API key prefix found in {p}"

    forbidden_keys = {"body", "patch", "quote", "context", "diff"}
    result_files = sorted(CASE_DIR.glob("results_*.json"))
    assert len(result_files) == 6
    for jf in result_files:
        payload = json.loads(jf.read_text(encoding="utf-8"))
        if "gate_commit" in payload:
            assert payload["gate_commit"] == EXPECTED_GATE_COMMIT
        hits = _walk_forbidden_keys(payload, forbidden_keys)
        assert not hits, f"Raw PR text keys {hits} found in {jf.name}"
