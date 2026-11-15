#!/usr/bin/env python3
"""Tests for Phase 3 blind held-out multi-domain evaluation & SLA suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BENCH_DIR = _ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

from heldout_multidomain_eval import run_heldout_evaluation  # noqa: E402


def test_heldout_multidomain_eval_passes_v3_v4_sla_and_writes_blind_sheet(tmp_path):
    res = run_heldout_evaluation(repetitions=3, out_dir=tmp_path)
    assert res["dataset_role"] == "unseen_heldout_case"
    assert res["domains"] == ["banking_finance", "securitization"]
    assert res["repetitions"] == 3
    assert res["overall_sla_passed"] is True
    assert res["v3_v4_strictly_better_than_v3_only"] is True

    v3_v4 = res["summary_by_arm"]["v3_v4"]
    assert v3_v4["false_accepts"] == 0
    assert v3_v4["false_rejects"] == 0
    assert v3_v4["repair_convergence_rate"] == 1.0
    assert v3_v4["sla_passed"] is True

    v3_only = res["summary_by_arm"]["v3_only"]
    assert v3_only["false_accepts"] > 0

    sheet = json.loads((tmp_path / "adjudication_sheet.json").read_text())
    key = json.loads((tmp_path / "adjudication_key.json").read_text())
    assert len(sheet) == len(key) > 0
    for row in sheet:
        assert "arm" not in row
        assert "system_disposition" not in row
        assert row["blind_id"].startswith("BLIND-")
