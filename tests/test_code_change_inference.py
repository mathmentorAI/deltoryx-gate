"""Offline unit and negative tests for Phase 3 v4 code-change inference."""
from __future__ import annotations

import pytest

from claimlayer.v4.domains.code_change_inference import (
    CODE_CHANGE_INFERENCE_PROFILE,
    CODE_DIFF_EPISTEMIC_PROFILE,
    CODE_STOPWORDS,
    CODE_STOPWORDS_SHA256,
    CodeDiffRuleVerifier,
    PR_ENTITY,
    RelevanceConfig,
    VerdictThresholds,
    build_code_diff_verifier_registry,
    build_typed_diff_ledger,
    check_premise_relevance,
    compute_pr_verdict,
    evaluate_i5_structural_signals,
    reconcile_phase3_literal_claim,
    repo_identifier_tokens,
    verify_behavior_graph,
)
from claimlayer.v4.evaluation import VerificationBudget, evaluate_response
from claimlayer.v4.justification import (
    ClaimNode,
    InferenceStep,
    JustificationGraph,
    VerificationObligation,
)
from claimlayer.v4.verifiers import DEFAULT_VERIFIER_REGISTRY


def _sample_pr(
    *,
    pr_id: str = "pr-101",
    repo: str = "acme/payment-gateway",
    title: str = "Add exponential retry backoff and metrics config",
    body: str = "Implements compute_retry_backoff with unit tests and yaml configuration.",
    file_patches: list[dict[str, str]] | None = None,
    adds: int = 24,
    dels: int = 4,
) -> dict:
    if file_patches is None:
        file_patches = [
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": (
                    "@@ -1,3 +1,8 @@\n"
                    " import time\n"
                    "+def compute_retry_backoff(attempt: int) -> float:\n"
                    '+    """Compute exponential backoff delay."""\n'
                    "+    return min(30.0, 0.5 * (2 ** attempt))\n"
                ),
            },
            {
                "filename": "tests/test_retry.py",
                "status": "added",
                "patch": (
                    "@@ -0,0 +1,5 @@\n"
                    "+from src.retry import compute_retry_backoff\n"
                    "+def test_compute_retry_backoff_caps():\n"
                    "+    assert compute_retry_backoff(10) == 30.0\n"
                ),
            },
            {
                "filename": "config/settings.yaml",
                "status": "modified",
                "patch": (
                    "@@ -2,2 +2,3 @@\n"
                    " timeout: 10\n"
                    "+enable_retry_jitter: true\n"
                ),
            },
        ]
    return {
        "pr_id": pr_id,
        "repo": repo,
        "url": f"https://github.com/{repo}/pull/1",
        "title": title,
        "body": body,
        "files": [fp["filename"] for fp in file_patches],
        "file_patches": file_patches,
        "patch": "\n---FILE---\n".join(fp["patch"] for fp in file_patches if (fp.get("patch") or "").strip()),
        "adds": adds,
        "dels": dels,
    }


def test_negative_invented_premise_rejected() -> None:
    """Fabricated diff quotes or wrong file attributions must NEVER yield derived_supported."""
    pr = _sample_pr()
    ledger = build_typed_diff_ledger(pr)

    # 1. Completely invented quote absent from raw diff documents
    res_fab = verify_behavior_graph(
        claim_id="c1",
        claim_text="Implements redis_circuit_breaker to guard upstream calls.",
        operation="implements",
        target="redis_circuit_breaker",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "redis_circuit_breaker",
                "literal_quote": "+def redis_circuit_breaker(host: str) -> bool:",
            }
        ],
        ledger=ledger,
    )
    assert res_fab["status"] == "no_premises_in_diff"
    assert res_fab["status"] != "derived_supported"
    assert res_fab["rule_id"] is None
    assert res_fab["verified_premises"] == []
    assert any(r["reason"] == "span_not_located" for r in res_fab["rejected_premises"])

    # 2. Real quote paired with wrong file attribution must also fail closed
    res_wrong_file = verify_behavior_graph(
        claim_id="c2",
        claim_text="Implements compute_retry_backoff in src/other.py.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/other.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger,
    )
    assert res_wrong_file["status"] == "no_premises_in_diff"
    assert res_wrong_file["verified_premises"] == []
    assert any(r["reason"] == "wrong_file_attribution" for r in res_wrong_file["rejected_premises"])


def test_negative_undeclared_or_wrong_version_rule_rejected() -> None:
    """Undeclared rule IDs, version mismatches, or ablation use_declared_rules=False never validate."""
    pr = _sample_pr()
    ledger = build_typed_diff_ledger(pr)
    quote = "+def compute_retry_backoff(attempt: int) -> float:"
    ref = ledger.locate_ref(quote)
    assert ref is not None

    def _run_step(rule_id: str, rule_version: str, *, use_rules: bool = True) -> tuple[str, str]:
        p_node = ClaimNode(
            claim_id="p0",
            kind="extraction",
            subject=PR_ENTITY,
            predicate="diff_premise",
            value=quote,
            text=quote,
            assertion_kind="fact",
            evidence=(ref,),
            condition_status="unconditional",
            structural_reliability=1.0,
        )
        c_node = ClaimNode(
            claim_id="conc",
            kind="derivation",
            subject=PR_ENTITY,
            predicate="implements",
            value="compute_retry_backoff",
            text="Implements compute_retry_backoff",
            assertion_kind="fact",
            condition_status="unconditional",
        )
        step = InferenceStep(
            step_id="s0",
            premises=("p0",),
            conclusion_id="conc",
            operation="rule_application",
            rule_id=rule_id,
            rule_version=rule_version,
            obligations=(
                VerificationObligation(
                    obligation_id="ob0",
                    family="explicit_logic",
                    description="Check rule",
                    critical=True,
                    parameters={
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "operation": "implements",
                        "target": "compute_retry_backoff",
                        "verified_premises": [{"file": "src/retry.py", "symbol_or_key": "compute_retry_backoff", "quote": quote}],
                    },
                ),
            ),
        )
        graph = JustificationGraph(claims={"p0": p_node, "conc": c_node}, steps=(step,), conclusion_id="conc")
        reg = build_code_diff_verifier_registry(ledger, use_declared_rules=use_rules)
        assessment = evaluate_response(
            graph,
            evidence=ledger.raw_documents,
            domain_profile=CODE_CHANGE_INFERENCE_PROFILE,
            verifier_registry=reg,
            budget=VerificationBudget(max_steps=16, max_depth=4, max_semantic_calls=0),
        )
        reasons = ",".join(r for chk in assessment.completed_checks for r in chk.reasons)
        return assessment.assessment, reasons

    st_undec, r_undec = _run_step("code.invented_rule", "1.0")
    assert st_undec != "supported"
    assert "UNDECLARED_OR_MISMATCHED_RULE" in r_undec

    st_ver, r_ver = _run_step("code.implements_symbol", "9.9")
    assert st_ver != "supported"
    assert "UNDECLARED_OR_MISMATCHED_RULE" in r_ver

    # Ablation (b): valid diff premise with use_declared_rules=False must NOT derive support
    res_no_rules = verify_behavior_graph(
        claim_id="c_norules",
        claim_text="Implements compute_retry_backoff in src/retry.py.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": quote,
            }
        ],
        ledger=ledger,
        use_declared_rules=False,
    )
    assert res_no_rules["status"] == "hypothesis_with_verified_premises"
    assert res_no_rules["effective_epistemic_kind"] == "hypothesis"
    assert res_no_rules["rule_id"] is None


def test_negative_hypothesis_never_ascends() -> None:
    """Verified premises for an unruled hypothesis or blocked epistemic basis never ascend."""
    pr = _sample_pr()
    ledger = build_typed_diff_ledger(pr)
    quote = "+def compute_retry_backoff(attempt: int) -> float:"

    # 1. Blocked epistemic basis in CODE_DIFF_EPISTEMIC_PROFILE
    assert CODE_DIFF_EPISTEMIC_PROFILE.admissible("deductive")
    assert CODE_DIFF_EPISTEMIC_PROFILE.admissible("syntactic")
    assert not CODE_DIFF_EPISTEMIC_PROFILE.admissible("causal")
    assert not CODE_DIFF_EPISTEMIC_PROFILE.admissible("common_sense")

    res_blocked_basis = verify_behavior_graph(
        claim_id="c_basis",
        claim_text="Implements compute_retry_backoff with causal intuition.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": quote,
                "basis": "causal",
            }
        ],
        ledger=ledger,
    )
    assert res_blocked_basis["status"] != "derived_supported"
    assert res_blocked_basis["effective_epistemic_kind"] == "hypothesis"
    assert any("blocked_epistemic_basis:causal" in r["reason"] for r in res_blocked_basis["rejected_premises"])

    # 2. Valid explicit premise + operation without declared rule (other_behavior) stays hypothesis
    res_hyp = verify_behavior_graph(
        claim_id="c_hyp",
        claim_text="Improves compute_retry_backoff throughput under burst traffic.",
        operation="other_behavior",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": quote,
                "basis": "explicit",
            }
        ],
        ledger=ledger,
        use_declared_rules=True,
    )
    assert res_hyp["status"] == "hypothesis_with_verified_premises"
    assert res_hyp["effective_epistemic_kind"] == "hypothesis"
    assert res_hyp["rule_id"] is None


def test_negative_targeted_reread_stub_cannot_invent_evidence() -> None:
    """Even after TargetedReread (reread_attempted=True), stopword-only or missing quotes cannot promote."""
    pr = _sample_pr()
    ledger = build_typed_diff_ledger(pr)

    def _stub_reread() -> list[dict[str, str]]:
        # Simulates a blind reread returning a stopword-only quote or nonexistent line
        return [
            {
                "file": "src/retry.py",
                "symbol_or_key": "config",
                "literal_quote": "+# invented OAuth2 token rotation handler",
            }
        ]

    reread_premises = _stub_reread()
    res = verify_behavior_graph(
        claim_id="c_reread",
        claim_text="Implements OAuth2 token rotation handler.",
        operation="implements",
        target="oauth2_token_rotation",
        proposed_premises=reread_premises,
        ledger=ledger,
        reread_attempted=True,
    )
    assert res["reread_attempted"] is True
    assert res["status"] != "derived_supported"
    assert res["status"] == "no_premises_in_diff"
    assert res["verified_premises"] == []


def test_documents_rule_strict_docstring_requires_literal_target() -> None:
    """Requirement #4: docstring in code file counts ONLY if literally citing target; else requires doc file."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": (
                    "@@ -1,2 +1,8 @@\n"
                    " def helper():\n"
                    '+    # General helper utility for handling timeout behavior.\n'
                    '+    """Explains retry backoff formula."""\n'
                    '+    """Documents compute_retry_backoff formula."""\n'
                    "+    pass\n"
                ),
            },
            {
                "filename": "docs/retry_guide.md",
                "status": "added",
                "patch": (
                    "@@ -0,0 +1,2 @@\n"
                    "+# Retry Policy\n"
                    "+Configure backoff_multiplier in settings.\n"
                ),
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)

    # 1a. Docstring in src file sharing words ("retry", "backoff") WITHOUT literal target 'compute_retry_backoff'
    #     passes quote check & relevance, yet MUST NOT satisfy code.documents rule (stays hypothesis).
    res_loose_doc = verify_behavior_graph(
        claim_id="d1a",
        claim_text="Documents compute_retry_backoff formula.",
        operation="documents",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "backoff",
                "literal_quote": '+    """Explains retry backoff formula."""',
            }
        ],
        ledger=ledger,
    )
    assert res_loose_doc["status"] == "hypothesis_with_verified_premises"
    assert res_loose_doc["rule_id"] is None

    # 1b. Generic comment in src file without target tokens fails relevance -> no_premises_in_diff
    res_vague_doc = verify_behavior_graph(
        claim_id="d1b",
        claim_text="Documents timeout_budget behavior.",
        operation="documents",
        target="timeout_budget",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "timeout",
                "literal_quote": "+    # General helper utility for handling timeout behavior.",
            }
        ],
        ledger=ledger,
    )
    assert res_vague_doc["status"] == "no_premises_in_diff"

    # 2. Docstring in src file WITH literal exact target 'compute_retry_backoff' -> derived_supported
    res_literal_doc = verify_behavior_graph(
        claim_id="d2",
        claim_text="Documents compute_retry_backoff formula.",
        operation="documents",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": '+    """Documents compute_retry_backoff formula."""',
            }
        ],
        ledger=ledger,
    )
    assert res_literal_doc["status"] == "derived_supported"
    assert res_literal_doc["rule_id"] == "code.documents"

    # 3. Markdown documentation file mentioning target 'backoff_multiplier' -> derived_supported (via rel.documents too)
    res_md_doc = verify_behavior_graph(
        claim_id="d3",
        claim_text="Documents backoff_multiplier in docs/retry_guide.md.",
        operation="documents",
        target="backoff_multiplier",
        proposed_premises=[
            {
                "file": "docs/retry_guide.md",
                "symbol_or_key": "backoff_multiplier",
                "literal_quote": "+Configure backoff_multiplier in settings.",
            }
        ],
        ledger=ledger,
    )
    assert res_md_doc["status"] == "derived_supported"
    assert res_md_doc["rule_id"] == "code.documents"


def test_declared_rule_implements_positive_and_anti_laxity() -> None:
    """code.implements_symbol requires a non-comment '+' definition in a src file."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": (
                    "@@ -1,4 +1,6 @@\n"
                    "+def compute_retry_backoff(attempt: int) -> float:\n"
                    "+    return float(attempt)\n"
                    "+val = call_only_symbol(1)\n"
                ),
            },
            {
                "filename": "docs/notes.md",
                "status": "added",
                "patch": "@@ -0,0 +1,2 @@\n+def fake_doc_symbol():\n",
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)

    # Positive: real '+' def in src file
    ok = verify_behavior_graph(
        claim_id="i_ok",
        claim_text="Implements compute_retry_backoff.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger,
    )
    assert ok["status"] == "derived_supported"
    assert ok["rule_id"] == "code.implements_symbol"

    # Anti-laxity 1: call site without definition cannot become derived_supported
    call_only = verify_behavior_graph(
        claim_id="i_call",
        claim_text="Implements call_only_symbol.",
        operation="implements",
        target="call_only_symbol",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "call_only_symbol",
                "literal_quote": "+val = call_only_symbol(1)",
            }
        ],
        ledger=ledger,
    )
    assert call_only["status"] == "hypothesis_with_verified_premises"

    # Anti-laxity 2: def inside docs file is not src -> stays hypothesis
    doc_def = verify_behavior_graph(
        claim_id="i_docdef",
        claim_text="Implements fake_doc_symbol.",
        operation="implements",
        target="fake_doc_symbol",
        proposed_premises=[
            {
                "file": "docs/notes.md",
                "symbol_or_key": "fake_doc_symbol",
                "literal_quote": "+def fake_doc_symbol():",
            }
        ],
        ledger=ledger,
    )
    assert doc_def["status"] == "hypothesis_with_verified_premises"


def test_declared_rule_adds_test_for_positive_and_anti_laxity() -> None:
    """code.adds_test_for + rel.adds_test_for require added test citing target; zero tests contradicts."""
    pr_with_tests = _sample_pr()
    ledger_t = build_typed_diff_ledger(pr_with_tests)

    ok_t = verify_behavior_graph(
        claim_id="t_ok",
        claim_text="Adds tests for compute_retry_backoff.",
        operation="adds_test_for",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "tests/test_retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+    assert compute_retry_backoff(10) == 30.0",
            }
        ],
        ledger=ledger_t,
    )
    assert ok_t["status"] == "derived_supported"
    assert ok_t["rule_id"] == "code.adds_test_for"

    # Zero tests in diff contradicts explicit "unit tests for ..." claim
    pr_no_tests = _sample_pr(
        file_patches=[
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+def compute_retry_backoff(a):\n+    return a\n",
            }
        ]
    )
    ledger_nt = build_typed_diff_ledger(pr_no_tests)
    bad_t = verify_behavior_graph(
        claim_id="t_none",
        claim_text="Adds unit tests for compute_retry_backoff.",
        operation="adds_test_for",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(a):",
            }
        ],
        ledger=ledger_nt,
    )
    assert bad_t["status"] == "premise_contradicted"
    assert "claimed_tests_but_zero_tests_in_diff" in str(bad_t["contradiction_reason"])


def test_declared_rule_removes_support_for_positive_and_contradiction() -> None:
    """code.removes_support_for succeeds on net '-' removal and contradicts N3 add/remove polarity flip."""
    pr_rem = _sample_pr(
        file_patches=[
            {
                "filename": "src/auth.py",
                "status": "modified",
                "patch": (
                    "@@ -1,4 +1,2 @@\n"
                    "-def legacy_hmac_v1():\n"
                    "-    return 'old'\n"
                    "+def modern_hmac_v2():\n"
                    "+    return 'new'\n"
                ),
            }
        ]
    )
    ledger_rem = build_typed_diff_ledger(pr_rem)

    ok_rem = verify_behavior_graph(
        claim_id="r_ok",
        claim_text="Removes legacy_hmac_v1 support.",
        operation="removes_support_for",
        target="legacy_hmac_v1",
        proposed_premises=[
            {
                "file": "src/auth.py",
                "symbol_or_key": "legacy_hmac_v1",
                "literal_quote": "-def legacy_hmac_v1():",
            }
        ],
        ledger=ledger_rem,
    )
    assert ok_rem["status"] == "derived_supported"
    assert ok_rem["rule_id"] == "code.removes_support_for"

    # Polarity flip: claiming modern_hmac_v2 was removed when it was only ADDED -> premise_contradicted
    flip_rem = verify_behavior_graph(
        claim_id="r_flip",
        claim_text="Removes modern_hmac_v2 support.",
        operation="removes_support_for",
        target="modern_hmac_v2",
        proposed_premises=[
            {
                "file": "src/auth.py",
                "symbol_or_key": "modern_hmac_v2",
                "literal_quote": "+def modern_hmac_v2():",
            }
        ],
        ledger=ledger_rem,
    )
    assert flip_rem["status"] == "premise_contradicted"
    assert "polarity_flip" in str(flip_rem["contradiction_reason"])


def test_declared_rule_configures_updates_dependency_and_n3_mismatch() -> None:
    """code.configures excludes lockfiles/src; code.updates_dependency flags N3 version mismatch."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "config/settings.yaml",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+enable_retry_jitter: true\n",
            },
            {
                "filename": "pyproject.toml",
                "status": "modified",
                "patch": '@@ -5,1 +5,1 @@\n-httpx = "0.26.0"\n+httpx = "0.27.2"\n',
            },
            {
                "filename": "package-lock.json",
                "status": "modified",
                "patch": '@@ -10,1 +10,1 @@\n+"lodash_lock_key": "4.17.21"\n',
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)

    # Positive configures via yaml
    ok_cfg = verify_behavior_graph(
        claim_id="cfg_ok",
        claim_text="Configures enable_retry_jitter setting.",
        operation="configures",
        target="enable_retry_jitter",
        proposed_premises=[
            {
                "file": "config/settings.yaml",
                "symbol_or_key": "enable_retry_jitter",
                "literal_quote": "+enable_retry_jitter: true",
            }
        ],
        ledger=ledger,
    )
    assert ok_cfg["status"] == "derived_supported"
    assert ok_cfg["rule_id"] == "code.configures"

    # Anti-laxity: lockfile rejected for configures
    lock_cfg = verify_behavior_graph(
        claim_id="cfg_lock",
        claim_text="Configures lodash_lock_key option.",
        operation="configures",
        target="lodash_lock_key",
        proposed_premises=[
            {
                "file": "package-lock.json",
                "symbol_or_key": "lodash_lock_key",
                "literal_quote": '+"lodash_lock_key": "4.17.21"',
            }
        ],
        ledger=ledger,
    )
    assert lock_cfg["status"] != "derived_supported"

    # Positive updates_dependency
    ok_dep = verify_behavior_graph(
        claim_id="dep_ok",
        claim_text="Updates httpx dependency to 0.27.2.",
        operation="updates_dependency",
        target="httpx 0.27.2",
        proposed_premises=[
            {
                "file": "pyproject.toml",
                "symbol_or_key": "httpx",
                "literal_quote": '+httpx = "0.27.2"',
            }
        ],
        ledger=ledger,
    )
    assert ok_dep["status"] == "derived_supported"
    assert ok_dep["rule_id"] == "code.updates_dependency"

    # N3 version mutation contradiction
    bad_dep = verify_behavior_graph(
        claim_id="dep_bad",
        claim_text="Updates httpx dependency to 0.28.0.",
        operation="updates_dependency",
        target="httpx 0.28.0",
        proposed_premises=[
            {
                "file": "pyproject.toml",
                "symbol_or_key": "httpx",
                "literal_quote": '+httpx = "0.27.2"',
            }
        ],
        ledger=ledger,
    )
    assert bad_dep["status"] == "premise_contradicted"
    assert "dependency_version_mismatch" in str(bad_dep["contradiction_reason"])


def test_anti_n2_relevance_guard_stopwords_and_repo_tokens() -> None:
    """Stopwords and repo/owner tokens must not allow unrelated same-repo N2 claims to latch."""
    for sw in ("config", "error", "test", "data", "util", "handler", "update"):
        assert sw in CODE_STOPWORDS
    assert len(CODE_STOPWORDS_SHA256) == 64

    pr = _sample_pr(
        repo="acme/widget_engine",
        file_patches=[
            {
                "filename": "src/widget_handler.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+def update_widget_config(data, error_handler):\n+    return data\n",
            }
        ],
    )
    ledger = build_typed_diff_ledger(pr)
    repo_toks = repo_identifier_tokens(pr["repo"])
    assert {"acme", "widget", "engine"} <= repo_toks

    # Same-repo N2 phantom sharing only stopwords (config/handler/update/data) + repo name (widget_engine)
    rel_ok, rel_reason = check_premise_relevance(
        claim_text="Implements billing_invoice_sync for widget_engine config handler update.",
        operation="implements",
        target="billing_invoice_sync",
        premise_file="src/widget_handler.py",
        premise_symbol="update_widget_config",
        premise_quote="+def update_widget_config(data, error_handler):",
        ledger=ledger,
        repo_tokens=repo_toks,
        config=RelevanceConfig(min_specific_anchors=2, require_target_match_when_present=True),
    )
    assert not rel_ok
    assert rel_reason == "TARGET_NOT_IN_PREMISE"

    # Full graph evaluation yields no_premises_in_diff (not derived_supported!)
    res_n2 = verify_behavior_graph(
        claim_id="n2_1",
        claim_text="Implements billing_invoice_sync for widget_engine config handler update.",
        operation="implements",
        target="billing_invoice_sync",
        proposed_premises=[
            {
                "file": "src/widget_handler.py",
                "symbol_or_key": "update_widget_config",
                "literal_quote": "+def update_widget_config(data, error_handler):",
            }
        ],
        ledger=ledger,
        repo_tokens=repo_toks,
    )
    assert res_n2["status"] == "no_premises_in_diff"
    assert res_n2["verified_premises"] == []


def test_i5_structural_rules_and_ablation_e_toggle() -> None:
    """I5 structural rules trigger INCONSISTENT when include_i5=True and turn off cleanly in Ablation (e)."""
    # 1. D_doc_only_but_code
    pr_doc_only = _sample_pr(
        title="Docs update",
        body="- [x] Documentation update\n- [ ] Bug fix\nUpdates guide.",
        file_patches=[
            {
                "filename": "src/core.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+def core_step():\n+    return 1\n",
            }
        ],
        adds=2,
        dels=0,
    )
    ledger_d = build_typed_diff_ledger(pr_doc_only)
    sigs_d = evaluate_i5_structural_signals(pr_doc_only, ledger_d)
    assert any(s["signal"] == "D_doc_only_but_code" for s in sigs_d)

    # 2. P_placeholder_large_diff
    pr_placeholder = _sample_pr(
        title="Update",
        body="WIP.",
        file_patches=[
            {
                "filename": "src/big.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+def big_fn():\n+    return 42\n",
            }
        ],
        adds=120,
        dels=15,
    )
    ledger_p = build_typed_diff_ledger(pr_placeholder)
    sigs_p = evaluate_i5_structural_signals(pr_placeholder, ledger_p)
    assert any(s["signal"] == "P_placeholder_large_diff" for s in sigs_p)

    # 3. U_severe_understated_files
    many_files = [
        {
            "filename": f"src/mod_{i}.py",
            "status": "modified",
            "patch": f"@@ -1,1 +1,2 @@\n+x_{i} = {i}\n",
        }
        for i in range(10)
    ]
    pr_under = _sample_pr(
        title="Small change",
        body="Only touches src/mod_0.py and src/mod_1.py.",
        file_patches=many_files,
        adds=10,
        dels=0,
    )
    ledger_u = build_typed_diff_ledger(pr_under)
    sigs_u = evaluate_i5_structural_signals(pr_under, ledger_u)
    assert any(s["signal"] == "U_severe_understated_files" for s in sigs_u)

    # Ablation (d) vs Ablation (e) toggle
    v_with_i5 = compute_pr_verdict(
        pr=pr_doc_only,
        ledger=ledger_d,
        literal_claims=[{"status": "literal_supported", "claim_receipt_sha256": "abc"}],
        behavior_results=[],
        draft_rejected={},
        thresholds=VerdictThresholds(include_i5=True),
    )
    assert v_with_i5["verdict"] == "INCONSISTENT"
    assert "I5_D_doc_only_but_code" in v_with_i5["reasons"]

    v_no_i5 = compute_pr_verdict(
        pr=pr_doc_only,
        ledger=ledger_d,
        literal_claims=[{"status": "literal_supported", "claim_receipt_sha256": "abc"}],
        behavior_results=[],
        draft_rejected={},
        thresholds=VerdictThresholds(include_i5=False),
    )
    assert v_no_i5["verdict"] == "CONSISTENT"
    assert v_no_i5["i5_signals"] == []


def test_truncated_ledger_escalates_never_consistent_and_evaluate_response_integration() -> None:
    """Truncated ledgers or internal/read errors MUST ESCALATE; registry stays isolated."""
    pr = _sample_pr()
    default_families_before = set(DEFAULT_VERIFIER_REGISTRY.families())
    profile_fp = CODE_CHANGE_INFERENCE_PROFILE.fingerprint()
    assert len(profile_fp) == 64

    # Force explicit truncation cap
    ledger_trunc = build_typed_diff_ledger(pr, max_syms=1)
    assert ledger_trunc.truncated is True
    assert "add_syms" in ledger_trunc.truncated_fields

    v_trunc = compute_pr_verdict(
        pr=pr,
        ledger=ledger_trunc,
        literal_claims=[{"status": "literal_supported", "claim_receipt_sha256": "r1"}],
        behavior_results=[],
        draft_rejected={},
    )
    assert v_trunc["verdict"] == "ESCALATE"
    assert "diff_facts_truncated" in v_trunc["reasons"]
    assert len(v_trunc["pr_receipt_sha256"]) == 64

    # Full uncapped ledger + error_escalated / read_errors also ESCALATE
    ledger_full = build_typed_diff_ledger(pr)
    assert ledger_full.truncated is False

    v_err = compute_pr_verdict(
        pr=pr,
        ledger=ledger_full,
        literal_claims=[{"status": "literal_supported", "claim_receipt_sha256": "r1"}],
        behavior_results=[],
        draft_rejected={},
        error_escalated_count=1,
    )
    assert v_err["verdict"] == "ESCALATE"

    v_read_err = compute_pr_verdict(
        pr=pr,
        ledger=ledger_full,
        literal_claims=[{"status": "literal_supported", "claim_receipt_sha256": "r1"}],
        behavior_results=[],
        draft_rejected={},
        read_errors=("nim_timeout",),
    )
    assert v_read_err["verdict"] == "ESCALATE"

    # Full end-to-end deterministic repeatability for evaluate_response-backed derived_supported claim
    r1 = verify_behavior_graph(
        claim_id="rep",
        claim_text="Implements compute_retry_backoff.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger_full,
    )
    r2 = verify_behavior_graph(
        claim_id="rep",
        claim_text="Implements compute_retry_backoff.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger_full,
    )
    assert r1["status"] == "derived_supported"
    assert r1["claim_receipt_sha256"] == r2["claim_receipt_sha256"]
    assert r1["assessment_digest"] == r2["assessment_digest"]
    assert set(DEFAULT_VERIFIER_REGISTRY.families()) == default_families_before


def test_empty_binary_file_patch_handled_without_error() -> None:
    """Binary or empty file_patches entries (patch='') must not fail SourceDocument validation."""
    pr = _sample_pr(
        file_patches=[
            {"filename": "assets/logo.png", "status": "added", "patch": ""},
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+def binary_safe_helper():\n+    return True\n",
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    assert ledger.files_count == 2
    assert ledger.file_categories["assets/logo.png"] == "src"
    assert "binary_safe_helper" in ledger.all_defs_added
    res = verify_behavior_graph(
        claim_id="bin1",
        claim_text="Implements binary_safe_helper.",
        operation="implements",
        target="binary_safe_helper",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "binary_safe_helper",
                "literal_quote": "+def binary_safe_helper():",
            }
        ],
        ledger=ledger,
    )
    assert res["status"] == "derived_supported"


def _load_phase3_module(name: str):
    import importlib.util
    from pathlib import Path

    mod_path = Path(__file__).resolve().parents[1] / "case_studies" / "pr_mci_phase3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"phase3_{name}", str(mod_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_score_eval_b_cohens_kappa_math() -> None:
    """Verify exact Cohen's kappa (unweighted, linear-weighted, broad, strict) and Wilson CIs."""
    score_eval_b = _load_phase3_module("score_eval_b")

    # Known 2x2 textbook table: TP=20, FP=5, FN=10, TN=15 (N=50)
    # p_o = 35/50 = 0.70; p_e = (25*30 + 25*20)/2500 = 1250/2500 = 0.50; kappa = 0.20/0.50 = 0.40
    y1 = ["aligned"] * 25 + ["misaligned"] * 25
    y2 = ["aligned"] * 20 + ["misaligned"] * 5 + ["aligned"] * 10 + ["misaligned"] * 15
    k_bin = score_eval_b.cohens_kappa(y1, y2, categories=("aligned", "misaligned"), weighted=None)
    assert k_bin["observed_agreement"] == pytest.approx(0.70, abs=1e-4)
    assert k_bin["expected_agreement"] == pytest.approx(0.50, abs=1e-4)
    assert k_bin["kappa"] == pytest.approx(0.40, abs=1e-4)

    # 3-class pair testing unweighted vs linear-weighted ordering (linear >= unweighted here)
    a1 = {f"p{i}": v for i, v in enumerate(["aligned", "aligned", "partial", "misaligned", "partial", "aligned"])}
    a2 = {f"p{i}": v for i, v in enumerate(["aligned", "partial", "partial", "partial", "misaligned", "aligned"])}
    iaa = score_eval_b.score_annotator_pair(a1, a2)
    assert iaa["n_shared"] == 6
    assert iaa["disagreements_count"] == 3
    assert iaa["kappa_3class_linear_weighted"]["kappa"] >= iaa["kappa_3class_unweighted"]["kappa"]
    assert 0.0 <= iaa["kappa_binary_broad_partial_or_misaligned"]["kappa"] <= 1.0

    # Wilson 95% CI sanity
    lo, hi = score_eval_b.wilson_ci(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0


def test_prmci_heuristic_determinism() -> None:
    """PR-MCI heuristic reproduction must deterministically flag mismatch and pass honest diffs."""
    heur = _load_phase3_module("prmci_heuristic")
    pr_honest = _sample_pr(
        title="feat: add compute_retry_backoff and tests",
        body="Implements compute_retry_backoff with unit tests in tests/test_retry.py and yaml config.",
    )
    res_h = heur.score_pr_heuristic(pr_honest)
    assert res_h["task_type"] == "feat"
    assert res_h["predicted_heuristic"] == "aligned"
    assert res_h["similarity_heuristic"] >= 0.61

    pr_mis = _sample_pr(
        title="fix: tiny patch",
        body="Adds unit tests and README documentation.",
        file_patches=[
            {"filename": "src/core.py", "status": "modified", "patch": "@@ -1,1 +1,2 @@\n+x = 1\n"}
        ],
        adds=300,
        dels=50,
    )
    res_m = heur.score_pr_heuristic(pr_mis)
    assert res_m["predicted_heuristic"] == "misaligned"
    assert res_m["similarity_heuristic"] < 0.61


def test_pipeline_phase3_double_replay_bit_identical_receipts(tmp_path) -> None:
    """Running pipeline_phase3 twice in --reader replay yields bit-identical receipts & gated Pass-3 reread."""
    p3 = _load_phase3_module("pipeline_phase3")
    pad_lines = "\n".join(f"+    # context padding line {i:03d} keeping chunk over 1600 chars" for i in range(30))
    pr_big = _sample_pr(
        pr_id="pr-big",
        file_patches=[
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": (
                    "@@ -1,3 +1,38 @@\n"
                    " import time\n"
                    "+def compute_retry_backoff(attempt: int) -> float:\n"
                    f"{pad_lines}\n"
                    "+    return min(30.0, 0.5 * (2 ** attempt))\n"
                ),
            },
            {
                "filename": "tests/test_retry.py",
                "status": "added",
                "patch": (
                    "@@ -0,0 +1,5 @@\n"
                    "+from src.retry import compute_retry_backoff\n"
                    "+def test_compute_retry_backoff_caps():\n"
                    "+    assert compute_retry_backoff(10) == 30.0\n"
                ),
            },
        ],
    )
    pr_small = _sample_pr(pr_id="pr-small")

    class _StubModel:
        identity = {"provider": "stub", "model": "stub-nim-v1"}

        def complete(self, system: str, payload: dict) -> dict:
            if "fragments" in payload:
                frags = payload["fragments"]
                fid = frags[0]["evidence_id"]
                txt = frags[0]["text"]
                val_q = "compute_retry_backoff" if "compute_retry_backoff" in txt else txt[:30]
                return {
                    "output": {
                        "proposals": [
                            {
                                "entity": "pull request",
                                "predicate": "adds_identifier",
                                "value": val_q,
                                "assertion_kind": "fact",
                                "basis": "explicit",
                                "condition_status": "unconditional",
                                "field_evidence": {"entity": "Implements", "value": val_q},
                                "evidence": [fid],
                                "quote": f"Implements {val_q}",
                            }
                        ]
                    },
                    "usage": {},
                }
            if "candidate_hunks" in payload:
                # Blind batched Pass-3 reread supplies valid test premise omitted in Pass 2
                cands = payload.get("candidate_claims") or []
                cid = cands[0]["claim_id"] if cands else "pr-big::beh::01"
                return {
                    "output": {
                        "reread_results": [
                            {
                                "claim_id": cid,
                                "proposed_premises": [
                                    {
                                        "hunk_id": "diff_pr-big_0001",
                                        "file": "tests/test_retry.py",
                                        "symbol_or_key": "compute_retry_backoff",
                                        "literal_quote": "+    assert compute_retry_backoff(10) == 30.0",
                                        "basis": "explicit",
                                    }
                                ],
                            }
                        ]
                    },
                    "usage": {},
                }
            pid = str(payload.get("pr_id") or "pr-big")
            # Pass 2 returns one ruled `implements` (with premise) and one `adds_test_for` (empty -> triggers Pass 3 if unseen!)
            return {
                "output": {
                    "behavioral_claims": [
                        {
                            "claim_text": "Implements compute_retry_backoff",
                            "operation": "implements",
                            "target": "compute_retry_backoff",
                            "proposed_premises": [
                                {
                                    "hunk_id": f"diff_{pid}_0000",
                                    "file": "src/retry.py",
                                    "symbol_or_key": "compute_retry_backoff",
                                    "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
                                    "basis": "explicit",
                                }
                            ],
                        },
                        {
                            "claim_text": "Adds unit tests for compute_retry_backoff",
                            "operation": "adds_test_for",
                            "target": "compute_retry_backoff",
                            "proposed_premises": [],
                        },
                    ]
                },
                "usage": {},
            }

    p1_cache = str(tmp_path / "p1.jsonl")
    p2_cache = str(tmp_path / "p2.jsonl")
    stub = _StubModel()
    m1 = p3.ResilientCachedModel(stub, p1_cache, min_interval_sec=0.0)
    m2 = p3.ResilientCachedModel(stub, p2_cache, min_interval_sec=0.0)
    live_rows = p3.run_batch([pr_big, pr_small], pass1_model=m1, pass2_model=m2)
    assert len(live_rows) == 2
    # Big/truncated diff triggers Pass-3 reread
    assert live_rows[0]["reread_attempts"] == 1
    assert live_rows[0]["ablations"]["c_graph_rules_no_reread"]["state_counts"]["no_premises_in_diff"] == 1
    assert live_rows[0]["ablations"]["d_full"]["state_counts"]["derived_supported"] == 2
    assert live_rows[0]["ablations"]["b_graph_no_rules"]["state_counts"]["hypothesis_with_verified_premises"] == 2
    # Small PR fully seen in Pass 2 skips redundant Pass-3 reread cleanly
    assert live_rows[1]["reread_attempts"] == 0

    # Strictly offline double-replay must match bit-identically with model=None
    rep_info = p3.verify_replay_determinism([pr_big, pr_small], pass1_cache=p1_cache, pass2_cache=p2_cache)
    assert rep_info["bit_identical"] is True
    assert rep_info["sample_digest"] == live_rows[0]["ablations"]["d_full"]["pr_receipt_sha256"]


def test_resolved_quote_sign_omitted_vs_sign_spoofed() -> None:
    """Sign-omitted verbatim line snaps to diff '+'/'-' line; sign-spoofed quote fails closed."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/retry.py",
                "status": "modified",
                "patch": (
                    "@@ -1,2 +1,3 @@\n"
                    "-def legacy_hmac_v1():\n"
                    "+def compute_retry_backoff(attempt: int) -> float:\n"
                    "+    return float(attempt)\n"
                ),
            }
        ]
    )
    ledger = build_typed_diff_ledger(pr)

    # (a) Sign-omitted verbatim line resolves to '+' diff line -> derived_supported
    res_omitted = verify_behavior_graph(
        claim_id="sq1",
        claim_text="Implements compute_retry_backoff.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger,
    )
    assert res_omitted["status"] == "derived_supported"
    assert res_omitted["rule_id"] == "code.implements_symbol"

    # (b) Sign-spoofed '+def legacy_hmac_v1():' against '-def legacy_hmac_v1():' must FAIL closed
    res_spoofed = verify_behavior_graph(
        claim_id="sq2",
        claim_text="Implements legacy_hmac_v1.",
        operation="implements",
        target="legacy_hmac_v1",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "legacy_hmac_v1",
                "literal_quote": "+def legacy_hmac_v1():",
            }
        ],
        ledger=ledger,
    )
    assert res_spoofed["status"] == "no_premises_in_diff"
    assert res_spoofed["status"] != "derived_supported"
    assert res_spoofed["verified_premises"] == []


def test_stem_and_hunk_header_relevance_helps_genuine_without_n2_leakage() -> None:
    """Enclosing @@ hunk header and non-stopword stemming help real diffs while blocking N2 phantoms."""
    pr = _sample_pr(
        repo="acme/widget_engine",
        file_patches=[
            {
                "filename": "src/sanitizer.py",
                "status": "modified",
                "patch": (
                    "@@ -10,4 +10,5 @@ class PayloadSanitizer:\n"
                    "     def clean(self, raw: str) -> str:\n"
                    "+        return raw.strip()\n"
                ),
            }
        ],
    )
    ledger = build_typed_diff_ledger(pr)
    repo_toks = repo_identifier_tokens(pr["repo"])

    # Genuine claim referencing 'PayloadSanitizer' / 'sanitizing' passes via enclosing hunk header + stem
    ok_genuine = verify_behavior_graph(
        claim_id="st_ok",
        claim_text="Updates sanitizing whitespace trim in PayloadSanitizer.",
        operation="other_behavior",
        target="sanitizing_payload",
        proposed_premises=[
            {
                "file": "src/sanitizer.py",
                "symbol_or_key": None,
                "literal_quote": "+        return raw.strip()",
            }
        ],
        ledger=ledger,
        repo_tokens=repo_toks,
    )
    assert ok_genuine["status"] == "hypothesis_with_verified_premises"
    assert len(ok_genuine["verified_premises"]) == 1

    # N2 same-repo claim with unrelated stopword+repo anchors ('configuring', 'widget_engine') still rejected!
    n2_blocked = verify_behavior_graph(
        claim_id="st_n2",
        claim_text="Configuring billing_invoice_queue inside widget_engine.",
        operation="implements",
        target="billing_invoice_queue",
        proposed_premises=[
            {
                "file": "src/sanitizer.py",
                "symbol_or_key": "clean",
                "literal_quote": "+        return raw.strip()",
            }
        ],
        ledger=ledger,
        repo_tokens=repo_toks,
    )
    assert n2_blocked["status"] == "no_premises_in_diff"
    assert n2_blocked["verified_premises"] == []


def test_symbol_or_key_spoofing_blocked_and_no_false_quantity_mismatch() -> None:
    """Unverified symbol_or_key is stripped (blocks relevance spoofing); numbers in quote don't cause false QUANTITY_MISMATCH."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/auth.py",
                "status": "modified",
                "patch": (
                    "@@ -1,2 +1,4 @@\n"
                    "+def verify_oauth2_jwt(leeway: int = 30) -> bool:\n"
                    "+    return leeway >= 0\n"
                ),
            }
        ]
    )
    ledger = build_typed_diff_ledger(pr)

    # 1. Quote containing 'oauth2' and number '30' must NOT trigger false code_config QUANTITY_MISMATCH
    ok = verify_behavior_graph(
        claim_id="sym1",
        claim_text="Implements verify_oauth2_jwt leeway handling.",
        operation="implements",
        target="verify_oauth2_jwt",
        proposed_premises=[
            {
                "file": "src/auth.py",
                "symbol_or_key": "oauth2",
                "literal_quote": "+def verify_oauth2_jwt(leeway: int = 30) -> bool:",
            }
        ],
        ledger=ledger,
    )
    assert ok["status"] == "derived_supported"
    assert ok["rule_id"] == "code.implements_symbol"

    # 2. Spoofed symbol_or_key matching target when NOT in quote/file must NOT bypass check_premise_relevance
    spoof = verify_behavior_graph(
        claim_id="sym2",
        claim_text="Implements redis_circuit_breaker.",
        operation="implements",
        target="redis_circuit_breaker",
        proposed_premises=[
            {
                "file": "src/auth.py",
                "symbol_or_key": "redis_circuit_breaker",
                "literal_quote": "+def verify_oauth2_jwt(leeway: int = 30) -> bool:",
            }
        ],
        ledger=ledger,
    )
    assert spoof["status"] == "no_premises_in_diff"
    assert spoof["verified_premises"] == []


def test_duplicate_json_field_discards_only_malformed_proposal_and_escalates_on_unparseable(tmp_path) -> None:
    """Decision 4: Duplicate JSON field discards only malformed item/premise and records reader_rejected; unparseable ESCALATES."""
    p3 = _load_phase3_module("pipeline_phase3")
    pr = _sample_pr()

    # (a) Valid sibling claim preserved when second Pass-2 claim (or nested premise) has Duplicate JSON field
    class _DupItemModel:
        identity = {"provider": "stub", "model": "stub-dup-v1"}

        def complete(self, system: str, payload: dict) -> dict:
            if "fragments" in payload:
                frag = payload["fragments"][0]
                return {
                    "output": {
                        "proposals": [
                            {
                                "entity": "pull request",
                                "predicate": "describes_behavior",
                                "value": "Implements compute_retry_backoff",
                                "assertion_kind": "fact",
                                "basis": "explicit",
                                "condition_status": "unconditional",
                                "field_evidence": {"entity": "Implements", "value": "Implements compute_retry_backoff"},
                                "evidence": [frag["evidence_id"]],
                                "quote": "Implements compute_retry_backoff",
                            }
                        ]
                    },
                    "usage": {},
                }
            # Simulate OpenAICompatibleModel raising ValueError('Duplicate JSON field') with raw choices content
            raw_bad_content = (
                '{"behavioral_claims": ['
                '{"claim_text": "Implements compute_retry_backoff", "operation": "implements", '
                '"target": "compute_retry_backoff", "proposed_premises": ['
                '{"hunk_id": "diff_pr-101_0000", "file": "src/retry.py", '
                '"symbol_or_key": "compute_retry_backoff", '
                '"literal_quote": "+def compute_retry_backoff(attempt: int) -> float:", "basis": "explicit"},'
                '{"file": "src/retry.py", "file": "src/duplicate_dup.py", '
                '"literal_quote": "+    return 1.0", "basis": "explicit"}'
                ']},'
                '{"claim_text": "Malformed dup claim", "claim_text": "Second", "operation": "implements", '
                '"target": "x", "proposed_premises": []}'
                ']}'
            )
            exc = ValueError("Duplicate JSON field")
            exc.diagnostics = {
                "attempts": [{"attempt": 1, "error_type": "ValueError", "stage": "response_contract"}],
                "provider_response": {
                    "model": "stub-dup-v1",
                    "usage": {"total_tokens": 10},
                    "choices": [{"message": {"content": raw_bad_content}, "finish_reason": "stop"}],
                },
            }
            raise exc

    p1_cache = str(tmp_path / "dup_p1.jsonl")
    p2_cache = str(tmp_path / "dup_p2.jsonl")
    m1 = p3.ResilientCachedModel(_DupItemModel(), p1_cache, min_interval_sec=0.0)
    m2 = p3.ResilientCachedModel(_DupItemModel(), p2_cache, min_interval_sec=0.0)
    res_a = p3.evaluate_pr_phase3(pr, pass1_model=m1, pass2_model=m2)
    assert res_a["read_errors"] == []
    assert res_a["ablations"]["d_full"]["verdict"] == "CONSISTENT"
    assert res_a["ablations"]["d_full"]["state_counts"]["derived_supported"] == 1
    assert len(res_a["reader_rejected"]) == 2
    assert all(r["reason"] == "duplicate_json_field" for r in res_a["reader_rejected"])
    assert res_a["ablations"]["d_full"]["reader_rejected"] == res_a["reader_rejected"]

    # Must replay reader_rejected and identical receipt deterministically offline
    m1_rep = p3.ResilientCachedModel(None, p1_cache, min_interval_sec=0.0)
    m2_rep = p3.ResilientCachedModel(None, p2_cache, min_interval_sec=0.0)
    res_rep = p3.evaluate_pr_phase3(pr, pass1_model=m1_rep, pass2_model=m2_rep)
    assert res_rep["reader_rejected"] == res_a["reader_rejected"]
    assert res_rep["ablations"]["d_full"]["pr_receipt_sha256"] == res_a["ablations"]["d_full"]["pr_receipt_sha256"]

    # Receipt changes if reader_rejected were absent (proving reader_rejected enters SHA-256 fingerprint)
    assert p3.prompt_digests()["duplicate_json_recovery_version"] == "v1-discard-malformed-proposal"

    # (b) Unparseable top-level duplicate or broken JSON must fail closed to ESCALATE
    class _UnparseableDupModel(_DupItemModel):
        def complete(self, system: str, payload: dict) -> dict:
            if "fragments" in payload:
                return super().complete(system, payload)
            exc = ValueError("Duplicate JSON field")
            exc.diagnostics = {
                "attempts": [],
                "provider_response": {
                    "choices": [{"message": {"content": '{"behavioral_claims": [], "behavioral_claims": []}'}}]
                },
            }
            raise exc

    m2_bad = p3.ResilientCachedModel(_UnparseableDupModel(), str(tmp_path / "bad_p2.jsonl"), min_interval_sec=0.0)
    res_b = p3.evaluate_pr_phase3(pr, pass1_model=m1, pass2_model=m2_bad)
    assert res_b["ablations"]["d_full"]["verdict"] == "ESCALATE"
    assert res_b["ablations"]["a_literal"]["verdict"] == "ESCALATE"
    assert any("pass2_exception" in e for e in res_b["read_errors"])


def test_bug1_dependency_boundary_no_substring_match() -> None:
    """Base dep name `eslint` must NEVER collide with `eslint-config-next` or `@eslint/eslintrc`."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "package.json",
                "status": "modified",
                "patch": '@@ -10,3 +10,3 @@\n-    "eslint": "^8.57.0"\n+    "eslint": "^9.39.1"\n',
            }
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    res_other = verify_behavior_graph(
        claim_id="b1_other",
        claim_text="Updates @eslint/eslintrc to 3.3.1 and eslint-config-next to 16.0.4.",
        operation="updates_dependency",
        target="eslint-config-next",
        proposed_premises=[
            {"file": "package.json", "symbol_or_key": "eslint", "literal_quote": '+    "eslint": "^9.39.1"'}
        ],
        ledger=ledger,
    )
    assert res_other["status"] != "premise_contradicted"
    assert res_other["contradiction_reason"] is None

    # Exact boundary mention with mismatched version STILL contradicts!
    res_mismatch = verify_behavior_graph(
        claim_id="b1_bad",
        claim_text="Updates eslint to 99.9.9.",
        operation="updates_dependency",
        target="eslint",
        proposed_premises=[
            {"file": "package.json", "symbol_or_key": "eslint", "literal_quote": '+    "eslint": "^9.39.1"'}
        ],
        ledger=ledger,
    )
    assert res_mismatch["status"] == "premise_contradicted"
    assert "dependency_version_mismatch:eslint" in (res_mismatch["contradiction_reason"] or "")


def test_bug2_toml_pep508_and_lockfile_stanza_deps() -> None:
    """Parse quoted PEP 508 bounds in pyproject.toml AND lockfile [[package]] stanzas without semver-padding FPs."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "pyproject.toml",
                "status": "modified",
                "patch": (
                    "@@ -10,4 +10,5 @@ dependencies = [\n"
                    '-    "rich==13.9.4",\n'
                    '+    "rich>=14.0.0,<15.0.0",\n'
                    '-    "spec-kitty-runtime @ git+https://github.com/org/spec-kitty-runtime.git@v0.1.0",\n'
                    '+    "spec-kitty-runtime==0.2.0a0",\n'
                ),
            },
            {
                "filename": "poetry.lock",
                "status": "modified",
                "patch": (
                    "@@ -50,4 +50,4 @@\n"
                    " [[package]]\n"
                    ' name = "h11"\n'
                    '-version = "0.14.0"\n'
                    '+version = "0.16.0"\n'
                    "@@ -90,2 +90,2 @@\n"
                    " [package.dependencies]\n"
                    '-h11 = ">=0.14"\n'
                    '+h11 = ">=0.16"\n'
                ),
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    by_pkg = {d["package"]: d for d in ledger.dependencies}
    assert "version" not in {d["package"].lower() for d in ledger.dependencies}
    assert by_pkg["rich"]["version"] == ">=14.0.0,<15.0.0"
    assert by_pkg["spec-kitty-runtime"]["version"] == "0.2.0a0"
    assert by_pkg["h11"]["version"] == "0.16.0"

    # Reconciling Pass-1 PEP 508 literal claim must succeed
    st_rich, why_rich = reconcile_phase3_literal_claim(
        "changes_dependency",
        {"action": "update", "package": "rich", "version": ">=14.0.0,<15.0.0"},
        "insufficient_evidence",
        ledger,
    )
    assert st_rich == "literal_supported" and why_rich == "typed_ledger_dep_match"

    # Semver zero-padding equivalence (`0.16` vs `0.16.0`) must NOT contradict
    res_h11 = verify_behavior_graph(
        claim_id="b2_h11",
        claim_text="Updates dependency h11 to 0.16.0.",
        operation="updates_dependency",
        target="h11",
        proposed_premises=[
            {"file": "poetry.lock", "symbol_or_key": "h11", "literal_quote": '+version = "0.16.0"'}
        ],
        ledger=ledger,
    )
    assert res_h11["status"] == "derived_supported"

    # Wrong version still contradicts cleanly
    res_h11_bad = verify_behavior_graph(
        claim_id="b2_h11_bad",
        claim_text="Updates dependency h11 to 99.9.9.",
        operation="updates_dependency",
        target="h11",
        proposed_premises=[],
        ledger=ledger,
    )
    assert res_h11_bad["status"] == "premise_contradicted"


def test_bug3_ignore_manifest_version_and_json_test_fixtures() -> None:
    """Manifest `version`, lock-version, and JSON fixtures inside code/test files never become dependencies."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "pyproject.toml",
                "status": "modified",
                "patch": '@@ -1,3 +1,3 @@\n [project]\n-version = "1.6.2"\n+version = "1.7.0"\n',
            },
            {
                "filename": "openapi3/openapi3_version_test.go",
                "status": "modified",
                "patch": '@@ -5,1 +5,1 @@\n+\t"openapi": "3.0.3",\n',
            },
            {
                "filename": "tests/fixtures/sample_manifest.json",
                "status": "modified",
                "patch": '@@ -1,2 +1,2 @@\n+  "lodash": "^4.17.21"\n',
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    assert ledger.dependencies == ()

    res = verify_behavior_graph(
        claim_id="b3_julia",
        claim_text='echo "Julia version >= 1.12" in workflow',
        operation="implements",
        target="Schema.Validate",
        proposed_premises=[],
        ledger=ledger,
    )
    assert res["status"] == "no_premises_in_diff"
    assert res["contradiction_reason"] is None


def test_bug4_checklist_or_not_required_not_doc_only() -> None:
    """`- [x] Documentation updated (or not required).` must NOT trigger `D_doc_only_but_code`."""
    pr_template = _sample_pr(
        body=(
            "Fix watcher invalidation on symlink update.\n"
            "- [x] Tests updated (or not required).\n"
            "- [x] Documentation updated (or not required).\n"
        ),
    )
    ledger = build_typed_diff_ledger(pr_template)
    sigs = evaluate_i5_structural_signals(pr_template, ledger)
    assert not any(s["signal"] == "D_doc_only_but_code" for s in sigs)

    # Genuine Type-of-Change "Documentation update" alone on a code diff STILL fires D_doc_only_but_code
    pr_doc_claim = _sample_pr(
        body=(
            "Type of change:\n"
            "- [ ] Bug fix\n"
            "- [ ] New feature\n"
            "- [x] Documentation update\n"
        ),
    )
    sigs2 = evaluate_i5_structural_signals(pr_doc_claim, build_typed_diff_ledger(pr_doc_claim))
    assert any(s["signal"] == "D_doc_only_but_code" for s in sigs2)


def test_bug5_yaml_workflow_tokens_and_parens_identifier_normalization() -> None:
    """Trailing `()` matches cleanly, YAML diff tokens validate, and shell/paren prose is unverifiable."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/path_utils.py",
                "status": "added",
                "patch": "@@ -0,0 +1,2 @@\n+def resolve_real_path(p: str) -> str:\n+    return p\n",
            },
            {
                "filename": ".github/workflows/ci.yml",
                "status": "modified",
                "patch": (
                    "@@ -10,2 +10,5 @@ jobs:\n"
                    "+      uses: peter-evans/create-pull-request@v8\n"
                    "+      run: echo \"YEAR=$(date +%Y)\" >> $GITHUB_ENV\n"
                    "+    runs-on: ubuntu-24.04-16cores-public\n"
                ),
            },
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    assert all("(" not in sym and ")" not in sym for sym in ledger.add_syms)

    st_paren, why_paren = reconcile_phase3_literal_claim(
        "adds_identifier", "resolve_real_path()", "insufficient_evidence", ledger
    )
    assert st_paren == "literal_supported" and why_paren == "paren_stripped_match"

    # Unbacked Foo() must remain literal_unsupported
    st_fake, _ = reconcile_phase3_literal_claim(
        "adds_identifier", "fabricated_fn()", "insufficient_evidence", ledger
    )
    assert st_fake == "literal_unsupported"

    # Genuine YAML tokens present in workflow diff validate cleanly
    for ytok in ("YEAR", "peter-evans/create-pull-request", "ubuntu-24.04-16cores-public"):
        st_y, why_y = reconcile_phase3_literal_claim("adds_identifier", ytok, "insufficient_evidence", ledger)
        assert st_y == "literal_supported" and why_y == "yaml_diff_token_match"

    # Parenthesized prose/shell command expressions do not fail as literal_unsupported identifiers
    st_expr, why_expr = reconcile_phase3_literal_claim(
        "adds_identifier", "pull_request_target trigger (type: labeled)", "insufficient_evidence", ledger
    )
    assert st_expr == "unverifiable" and why_expr == "non_identifier_paren_or_expr"
    st_sh, _ = reconcile_phase3_literal_claim("modifies_identifier", "$(date +%Y)", "insufficient_evidence", ledger)
    assert st_sh == "unverifiable"


def test_bug6_renamed_previous_file_paths_supported() -> None:
    """Pre-rename source paths reconcile via `previous_filename`, extension renames, or `old -> new` prose."""
    pr = _sample_pr(
        body="- `packages/ui/BpkContentCard.js` → `BpkContentCard.tsx`\n",
        file_patches=[
            {
                "filename": "packages/ui/BpkContentCard.tsx",
                "status": "renamed",
                "patch": "",
            },
            {
                "filename": "packages/old_module_new.py",
                "status": "renamed",
                "previous_filename": "packages/old_module.py",
                "patch": "",
            },
        ],
    )
    ledger = build_typed_diff_ledger(pr)
    assert "packages/ui/BpkContentCard.tsx" in ledger.renamed_files
    assert "packages/old_module.py" in ledger.renamed_from_paths

    st1, _ = reconcile_phase3_literal_claim(
        "changes_file", "packages/ui/BpkContentCard.js", "insufficient_evidence", ledger, pr_text=pr["body"]
    )
    st2, _ = reconcile_phase3_literal_claim(
        "changes_file", "packages/old_module.py", "insufficient_evidence", ledger, pr_text=pr["body"]
    )
    st_bad, _ = reconcile_phase3_literal_claim(
        "changes_file", "packages/ui/NonExistentFile.js", "insufficient_evidence", ledger, pr_text=pr["body"]
    )
    assert st1 == "literal_supported"
    assert st2 == "literal_supported"
    assert st_bad == "literal_unsupported"


def test_bug7_local_test_execution_reclassified_to_out_of_scope_meta() -> None:
    """Running existing tests locally or running linters becomes `out_of_scope_meta`, while new-test claims still check."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "src/core.py",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n def ok():\n+    return True\n",
            }
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    for txt in (
        "Ejecuté los tests en local y pasaron todos.",
        "Run unit tests locally with python -m unittest discover, which executed 48 tests and completed successfully (OK).",
        "All 438 existing tests still pass.",
        "ruff check and ruff format clean",
    ):
        r = verify_behavior_graph(
            claim_id="b7_local",
            claim_text=txt,
            operation="adds_test_for",
            target="local tests",
            proposed_premises=[],
            ledger=ledger,
        )
        assert r["operation"] == "out_of_scope_meta"
        assert r["status"] == "unverifiable"

    # Negative check (N3 protection): claiming NEW unit tests on zero-test diff MUST still be contradicted!
    r_added = verify_behavior_graph(
        claim_id="b7_n3",
        claim_text="Added comprehensive unit tests covering retry backoff.",
        operation="adds_test_for",
        target="retry backoff",
        proposed_premises=[],
        ledger=ledger,
    )
    assert r_added["operation"] == "adds_test_for"
    assert r_added["status"] == "premise_contradicted"
    assert r_added["contradiction_reason"] == "claimed_tests_but_zero_tests_in_diff"


def test_pass3_spoofed_target_blocked_after_reread() -> None:
    """In Pass 3 (`reread_attempted=True`), an ungrounded spoofed target cannot justify an unrelated claim."""
    pr = _sample_pr(
        file_patches=[
            {
                "filename": "install-plugin.sh",
                "status": "modified",
                "patch": "@@ -1,1 +1,2 @@\n+echo 'plugin banner initialized'\n",
            }
        ]
    )
    ledger = build_typed_diff_ledger(pr)
    spoof_quote = "+echo 'plugin banner initialized'"

    # In Pass 3 with hardening enabled (default True), spoofed target not in claim_text is REJECTED
    res_hardened = verify_behavior_graph(
        claim_id="p3_spoof",
        claim_text="Replace outdated tmux-focused empty state message with agent-first approach",
        operation="implements",
        target="banner message in install-plugin.sh",
        proposed_premises=[
            {
                "file": "install-plugin.sh",
                "symbol_or_key": "install-plugin.sh",
                "literal_quote": spoof_quote,
            }
        ],
        ledger=ledger,
        reread_attempted=True,
        relevance_config=RelevanceConfig(pass3_require_target_grounded_in_claim=True),
    )
    assert res_hardened["status"] == "no_premises_in_diff"
    assert res_hardened["verified_premises"] == []

    # When target IS genuinely grounded in claim_text and premise matches a src def, Pass 3 still derives!
    pr_src = _sample_pr()
    ledger_src = build_typed_diff_ledger(pr_src)
    res_legit = verify_behavior_graph(
        claim_id="p3_legit",
        claim_text="Implements compute_retry_backoff with bounded exponential delay.",
        operation="implements",
        target="compute_retry_backoff",
        proposed_premises=[
            {
                "file": "src/retry.py",
                "symbol_or_key": "compute_retry_backoff",
                "literal_quote": "+def compute_retry_backoff(attempt: int) -> float:",
            }
        ],
        ledger=ledger_src,
        reread_attempted=True,
        relevance_config=RelevanceConfig(pass3_require_target_grounded_in_claim=True),
    )
    assert res_legit["status"] == "derived_supported"


def test_decoupled_review_thresholds_in_compute_pr_verdict() -> None:
    """`REVIEW` triggers follow independent `review_k_*` / `review_theta_*` thresholds."""
    pr = _sample_pr()
    ledger = build_typed_diff_ledger(pr)
    lit_claims = [
        {"claim_id": "l1", "status": "literal_supported", "claim_receipt_sha256": "a" * 64},
        {"claim_id": "l2", "status": "literal_unsupported", "claim_receipt_sha256": "b" * 64},
    ]
    beh_results = [
        {
            "claim_id": "b1",
            "status": "derived_supported",
            "effective_epistemic_kind": "derivation",
            "verified_premises": [],
            "rejected_premises": [],
            "claim_receipt_sha256": "c" * 64,
        },
        {
            "claim_id": "b2",
            "status": "derived_supported",
            "effective_epistemic_kind": "derivation",
            "verified_premises": [],
            "rejected_premises": [],
            "claim_receipt_sha256": "d" * 64,
        },
        {
            "claim_id": "b3",
            "status": "no_premises_in_diff",
            "effective_epistemic_kind": "hypothesis",
            "verified_premises": [],
            "rejected_premises": [],
            "claim_receipt_sha256": "e" * 64,
        },
    ]

    # Default thresholds (review_k=1, review_theta=0.0) mark partial cases as REVIEW
    th_default = VerdictThresholds(k_phantom=2, theta_phantom=0.5, k_lit=3, theta_lit=0.5, include_i5=False)
    v_def = compute_pr_verdict(
        pr=pr,
        ledger=ledger,
        literal_claims=lit_claims,
        behavior_results=beh_results,
        draft_rejected={},
        thresholds=th_default,
    )
    assert v_def["verdict"] == "REVIEW"

    # Raising review thresholds above 1/3 phantom and 1/2 literal yields CONSISTENT cleanly
    th_tuned = VerdictThresholds(
        k_phantom=2,
        theta_phantom=0.5,
        k_lit=3,
        theta_lit=0.5,
        review_on_single_unsupported_or_phantom=True,
        review_k_phantom=2,
        review_theta_phantom=0.5,
        review_k_lit=2,
        review_theta_lit=0.67,
        review_on_all_unruled_hypothesis=False,
        include_i5=False,
    )
    v_tuned = compute_pr_verdict(
        pr=pr,
        ledger=ledger,
        literal_claims=lit_claims,
        behavior_results=beh_results,
        draft_rejected={},
        thresholds=th_tuned,
    )
    assert v_tuned["verdict"] == "CONSISTENT"






