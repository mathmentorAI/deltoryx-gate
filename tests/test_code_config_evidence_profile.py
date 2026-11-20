"""Tests for explicit per-source `evidence_profile` ('legal_prose' vs 'code_config')."""
from __future__ import annotations

import pytest

from claimlayer import Assertion, Ledger, Policy, Source, verify
from claimlayer.models import EVIDENCE_PROFILES


def _verify_single(
    *,
    source_text: str,
    quote: str,
    value: str,
    unit: str | None = None,
    evidence_profile: str = "code_config",
    source_id: str = "diff_src",
    policy: Policy | None = None,
):
    ledger = Ledger(
        sources=[
            Source(
                id=source_id,
                text=source_text,
                tier="authoritative",
                evidence_profile=evidence_profile,  # type: ignore[arg-type]
            )
        ]
    )
    assertion = Assertion(
        subject="pull_request",
        predicate="contains_change",
        value=value,
        unit=unit,
        source_id=source_id,
        quote=quote,
    )
    return ledger, verify([assertion], ledger, policy)


def test_real_prmci_prose_false_rejections_pass_under_code_config_and_stay_rejected_in_legal_prose():
    cases = [
        (
            '"dependencies": {\n  "zod": "^3.0.0"\n},\n"optionalDependencies": {\n  "fsevents": "^2.3.3"\n}',
            '"optionalDependencies": {',
            "optionalDependencies",
        ),
        (
            "export const schema = [\n  tableSchema,\n  columnUniqueSchema,\n];",
            "columnUniqueSchema,",
            "columnUniqueSchema",
        ),
        (
            "CODEOWNERS\npackage.json\nsrc/index.ts",
            "CODEOWNERS",
            "CODEOWNERS",
        ),
    ]
    for source_text, quote, value in cases:
        _, v_legal = _verify_single(
            source_text=source_text,
            quote=quote,
            value=value,
            evidence_profile="legal_prose",
        )
        assert v_legal.status == "RETRY"
        assert v_legal.issues[0].code == "QUOTE_NOT_IN_SOURCE"

        ledger_code, v_code = _verify_single(
            source_text=source_text,
            quote=quote,
            value=value,
            evidence_profile="code_config",
        )
        assert v_code.status == "PASS"
        assert v_code.verify_receipt(ledger=ledger_code, allow_unkeyed=True) is True


def test_code_config_positive_literals_versions_hex_units_and_adjacent_negation():
    cases = [
        ("try:\n    run()\nexcept ValueError:\n    fallback()", "    fallback()", "fallback"),
        ("if not valid:\n    raise ValueError('bad')\nreturn ok_flag", "return ok_flag", "ok_flag"),
        ('"section 1": true\n"other": false', '"section 1": true', '"section 1": true'),
        ('{\n  "version": "2.1.0"\n}', '"version": "2.1.0"', "2.1.0"),
        ("/// Base gas cost for SHA256 precompile (address 0x02)\nconst x = 0x02;", "const x = 0x02;", "0x02"),
        ("retries = -1\nmax_wait = 10", "retries = -1", "-1"),
        ("def add(x, y):\n    return x + y", "def add(x, y):", "add()"),
        ("src/app-1.py\nsrc/app-2.py", "src/app-1.py", "app-1.py"),
        ("fn TxPool.add(x) {\n  return x;\n}", "fn TxPool.add(x) {", "TxPool.add()"),
        ("type DebugStream = riscv::hio::HostStream;", "type DebugStream = riscv::hio::HostStream;", "riscv::hio::HostStream"),
        ("if x != 0:\n    return True", "if x != 0:", "x != 0"),
        ("release_date = \"2025-03-01\"\n", 'release_date = "2025-03-01"', "2025-03-01"),
    ]
    for source_text, quote, value in cases:
        ledger, v = _verify_single(source_text=source_text, quote=quote, value=value, evidence_profile="code_config")
        assert v.status == "PASS", f"Expected PASS for {value!r} against {quote!r}, got {v.issues}"
        assert v.verify_receipt(ledger=ledger, allow_unkeyed=True) is True

    # Dimensioned code/config values
    for quote, value, unit in [
        ('"timeout": "30 seconds"', "30", "seconds"),
        ('"timeout": "30s"', "30", "seconds"),
        ('"timeout": "30 seconds"', "30s", None),
        ('"ttl": "30 hours"', "30", "hours"),
    ]:
        ledger, v = _verify_single(
            source_text=quote, quote=quote, value=value, unit=unit, evidence_profile="code_config"
        )
        assert v.status == "PASS", f"Expected PASS for {value!r} ({unit}) vs {quote!r}, got {v.issues}"
        assert v.verify_receipt(ledger=ledger, allow_unkeyed=True) is True


@pytest.mark.parametrize(
    ("source_text", "quote", "value", "unit", "expected_code"),
    [
        (
            "/// Base gas cost for SHA256 precompile (address 0x02)",
            "/// Base gas cost for SHA256 precompile (address 0x02)",
            "add()",
            None,
            "QUANTITY_MISMATCH",
        ),
        (
            "long result = nextDeliveryTimeUnsafe();",
            "long result = nextDeliveryTimeUnsafe();",
            "nextDeliveryTime()",
            None,
            "QUANTITY_MISMATCH",
        ),
        ("def createOrder(id):\n    pass", "def createOrder(id):", "createUser", None, "QUANTITY_MISMATCH"),
        (
            "const columns = processColumns(tblsTable.columns);",
            "const columns = processColumns(tblsTable.columns);",
            "Column",
            None,
            "QUANTITY_MISMATCH",
        ),
        (
            "def _test_webhook(webhook_id: str):",
            "def _test_webhook(webhook_id: str):",
            "test_webhook()",
            None,
            "QUANTITY_MISMATCH",
        ),
        ("def foo_v3():\n    pass", "def foo_v3():", "foo_v2", None, "QUANTITY_MISMATCH"),
        ("b/a/src.py", "b/a/src.py", "src/a/b.py", None, "QUANTITY_MISMATCH"),
        ("bar.Foo()", "bar.Foo()", "Foo.bar()", None, "QUANTITY_MISMATCH"),
        ('"version": "2.1.0"', '"version": "2.1.0"', "2.1", None, "QUANTITY_MISMATCH"),
        ('"version": "2.1.0"', '"version": "2.1.0"', "1.0", None, "QUANTITY_MISMATCH"),
        ('"port": 8080', '"port": 8080', "80", None, "QUANTITY_MISMATCH"),
        ('"ratio": 80.5', '"ratio": 80.5', "80", None, "QUANTITY_MISMATCH"),
        ("retries = -1", "retries = -1", "1", None, "QUANTITY_MISMATCH"),
        ("retries = 1", "retries = 1", "-1", None, "QUANTITY_MISMATCH"),
        ('"enabled": false', '"enabled": false', "enabled: true", None, "QUANTITY_MISMATCH"),
        ('"timeout": 30', '"timeout": 30', "timeout: 60", None, "QUANTITY_MISMATCH"),
        ("if x == 0:", "if x == 0:", "x != 0", None, "QUANTITY_MISMATCH"),
        ('"timeout_ms": 30', '"timeout_ms": 30', "30", "seconds", "UNIT_MISMATCH"),
        ('"timeout": "30ms"', '"timeout": "30ms"', "30", "seconds", "UNIT_MISMATCH"),
        ('"timeout": "30s"', '"timeout": "30s"', "30", None, "QUANTITY_MISMATCH"),
        ('release_date = "2025-03-01"', 'release_date = "2025-03-01"', "2025-03-02", None, "DATE_MISMATCH"),
        ("address = 1", "add", "add", None, "QUOTE_NOT_IN_SOURCE"),
        ("{\n  key: 1\n}", "{", "key", None, "QUOTE_NOT_IN_SOURCE"),
        ("--- a/foo.py\n+++ b/foo.py", "---", "foo.py", None, "QUOTE_NOT_IN_SOURCE"),
        ("def fn(): pass", "()", "fn", None, "QUOTE_NOT_IN_SOURCE"),
    ],
)
def test_code_config_negative_rejections(source_text, quote, value, unit, expected_code):
    _, v = _verify_single(
        source_text=source_text,
        quote=quote,
        value=value,
        unit=unit,
        evidence_profile="code_config",
    )
    assert v.status == "RETRY", f"Expected RETRY for value={value!r} quote={quote!r}"
    assert v.issues[0].code == expected_code


def test_source_profile_validation_sqlite_roundtrip_and_receipt_tampering(tmp_path):
    assert EVIDENCE_PROFILES == frozenset({"legal_prose", "code_config"})
    with pytest.raises(ValueError, match="Unknown evidence_profile"):
        Source(id="bad", text="x = 1", evidence_profile="unknown")  # type: ignore[arg-type]

    # Code content_sha256 does not strip lines literally named 'draft'
    s_with_draft = Source(id="c1", text="draft\nkey = 1", evidence_profile="code_config")
    s_no_draft = Source(id="c2", text="key = 1", evidence_profile="code_config")
    assert s_with_draft.content_sha256 != s_no_draft.content_sha256

    db_path = tmp_path / "ledger.sqlite"
    src_text = '"optionalDependencies": {\n  "pkg": "1.0.0"\n}'
    led1 = Ledger(
        sources=[Source(id="pkg", text=src_text, tier="authoritative", evidence_profile="code_config")],
        db_path=db_path,
    )
    led2 = Ledger(db_path=db_path)
    loaded = led2.get_source("pkg")
    assert loaded is not None
    assert loaded.evidence_profile == "code_config"
    assert loaded.to_dict()["evidence_profile"] == "code_config"

    assertion = Assertion(
        subject="pull_request",
        predicate="contains_change",
        value="optionalDependencies",
        source_id="pkg",
        quote='"optionalDependencies": {',
    )
    v = verify([assertion], led2)
    assert v.status == "PASS"
    assert v.verify_receipt(ledger=led2, allow_unkeyed=True) is True

    # Changing evidence_profile on the ledger source invalidates the receipt
    tampered_ledger = Ledger(
        sources=[Source(id="pkg", text=src_text, tier="authoritative", evidence_profile="legal_prose")]
    )
    assert v.verify_receipt(ledger=tampered_ledger, allow_unkeyed=True) is False


def test_default_legal_prose_profile_is_not_relaxed():
    # Default evidence_profile remains 'legal_prose'
    src_default = Source(id="msa", text="Section 4.2: Payment Terms. Invoices are due in 30 days.")
    assert src_default.evidence_profile == "legal_prose"

    led_default = Ledger(sources=[src_default])
    # Section header only still rejected
    v_hdr = verify(
        [
            Assertion(
                subject="MSA",
                predicate="payment_terms",
                value="30 days",
                source_id="msa",
                quote="Section 4.2: Payment Terms.",
            )
        ],
        led_default,
    )
    assert v_hdr.status == "RETRY"
    assert v_hdr.issues[0].code == "QUOTE_NOT_IN_SOURCE"

    # Truncated negation still rejected
    led_neg = Ledger(
        sources=[Source(id="msa2", text="In no event shall the cap exceed $100,000.")]
    )
    v_neg = verify(
        [
            Assertion(
                subject="MSA",
                predicate="cap",
                value="$100,000",
                source_id="msa2",
                quote="shall the cap exceed $100,000.",
            )
        ],
        led_neg,
    )
    assert v_neg.status == "RETRY"
    assert v_neg.issues[0].code == "QUOTE_NOT_IN_SOURCE"

    # Bare single word still rejected
    led_word = Ledger(sources=[Source(id="msa3", text="This Agreement is governed by Delaware law.")])
    v_word = verify(
        [
            Assertion(
                subject="MSA",
                predicate="governing_law",
                value="Delaware",
                source_id="msa3",
                quote="Delaware",
            )
        ],
        led_word,
    )
    assert v_word.status == "RETRY"
    assert v_word.issues[0].code == "QUOTE_NOT_IN_SOURCE"


def test_code_config_radix_literals_and_negative_substring_rejections():
    from claimlayer.verifiers.code_evidence import is_code_config_assertion_self_consistent

    # Positive: octal, binary, underscore hex, and digit-prefixed alphanumeric tokens
    for src, q, val in [
        ("+\terr = store.Chmod(privateKeyPath, 0o600)", "+\terr = store.Chmod(privateKeyPath, 0o600)", "0o600"),
        ("+\terr = store.Chmod(privateKeyPath, 0O755)", "+\terr = store.Chmod(privateKeyPath, 0O755)", "store.Chmod(privateKeyPath, 0O755)"),
        ("+const mask = 0b1010_0101;", "+const mask = 0b1010_0101;", "0b1010_0101"),
        ("+const magic = 0xdead_beef;", "+const magic = 0xdead_beef;", "0xdead_beef"),
        ('+  "dimension": "2D",', '+  "dimension": "2D",', '+  "dimension": "2D",'),
        ("+    runs-on: ubuntu-24.04-16cores-public", "+    runs-on: ubuntu-24.04-16cores-public", "ubuntu-24.04-16cores-public"),
        ('+  "2002:7f00:0001::",', '+  "2002:7f00:0001::",', "2002:7f00:0001::"),
        ("+  - uses: WalletConnect/actions/github/cta-assistant@7a3ca8b03e62", "+  - uses: WalletConnect/actions/github/cta-assistant@7a3ca8b03e62", "WalletConnect/actions/github/cta-assistant@7a3ca8b03e62"),
        ('+echo "Customize: ai-maestro.23blocks.com/plugin-builder.html"', '+echo "Customize: ai-maestro.23blocks.com/plugin-builder.html"', '+echo "Customize: ai-maestro.23blocks.com/plugin-builder.html"'),
    ]:
        led, res = _verify_single(source_text=src, quote=q, value=val)
        assert res.status == "PASS", f"Expected PASS for {val!r}, got {res.issues}"
        assert res.verify_receipt(ledger=led, allow_unkeyed=True) is True
        assert is_code_config_assertion_self_consistent(
            Assertion(subject="pull_request", predicate="diff_evidence", value=val, source_id="diff_src", quote=q)
        ) is True

    # Negative: prefix/substring truncation of radix & digit-prefixed tokens must fail strictly
    for src, q, val in [
        ("+err = store.Chmod(p, 0o600)", "+err = store.Chmod(p, 0o600)", "0o60"),
        ("+err = store.Chmod(p, 0o600)", "+err = store.Chmod(p, 0o600)", "600"),
        ("+const mask = 0b1010_0101;", "+const mask = 0b1010_0101;", "0b1010"),
        ("+const magic = 0xdead_beef;", "+const magic = 0xdead_beef;", "0xdead"),
        ('+  "dimension": "2D",', '+  "dimension": "2D",', "2"),
        ('+  "dimension": "2D",', '+  "dimension": "2D",', "D"),
        ("+    runs-on: ubuntu-24.04-16cores-public", "+    runs-on: ubuntu-24.04-16cores-public", "16"),
        ("+  - uses: org/act@7a3ca8b03e62", "+  - uses: org/act@7a3ca8b03e62", "7a3ca8"),
        ('+echo "ai-maestro.23blocks.com"', '+echo "ai-maestro.23blocks.com"', "blocks"),
    ]:
        _, res = _verify_single(source_text=src, quote=q, value=val)
        assert res.status == "RETRY", f"Expected RETRY for substring {val!r} in {q!r}"
        assert res.issues[0].code == "QUANTITY_MISMATCH"


def test_code_config_generics_chained_calls_and_negative_mismatch():
    from claimlayer.verifiers.code_evidence import is_code_config_assertion_self_consistent

    # Positive: Rust/TS/C# generic type parameters and chained method calls
    for src, q, val in [
        (
            '+Config::global().get_param::<bool>("GOOSE_SHOW_FULL_OUTPUT").unwrap_or(false)',
            '+Config::global().get_param::<bool>("GOOSE_SHOW_FULL_OUTPUT").unwrap_or(false)',
            '+Config::global().get_param::<bool>("GOOSE_SHOW_FULL_OUTPUT").unwrap_or(false)',
        ),
        (
            '+Config::global().get_param::<bool>("GOOSE_SHOW_FULL_OUTPUT").unwrap_or(false)',
            '+Config::global().get_param::<bool>("GOOSE_SHOW_FULL_OUTPUT").unwrap_or(false)',
            "<bool>",
        ),
        (
            "+log = directory.GetFiles().OrderByDescending(f => f.LastWriteTime).FirstOrDefault();",
            "+log = directory.GetFiles().OrderByDescending(f => f.LastWriteTime).FirstOrDefault();",
            "+log = directory.GetFiles().OrderByDescending(f => f.LastWriteTime).FirstOrDefault();",
        ),
        (
            "+return getRefContext().measureText(text);",
            "+return getRefContext().measureText(text);",
            "+return getRefContext().measureText(text);",
        ),
        (
            "+static SHOW_FULL_TOOL_OUTPUT: RefCell<bool> = RefCell::new(false);",
            "+static SHOW_FULL_TOOL_OUTPUT: RefCell<bool> = RefCell::new(false);",
            "RefCell<bool>",
        ),
        (
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "Vec<String>",
        ),
        (
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "Option<bool>",
        ),
        (
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "fn convert(items: Vec<String>) -> Vec<Option<bool>>",
            "Vec<Option<bool>>",
        ),
    ]:
        led, res = _verify_single(source_text=src, quote=q, value=val)
        assert res.status == "PASS", f"Expected PASS for {val!r}, got {res.issues}"
        assert res.verify_receipt(ledger=led, allow_unkeyed=True) is True
        assert is_code_config_assertion_self_consistent(
            Assertion(subject="pull_request", predicate="diff_evidence", value=val, source_id="diff_src", quote=q)
        ) is True

    # Negative: wrong generic parameter, fake generic against comparison ops, flipped comparison
    for src, q, val in [
        ("fn convert(items: Vec<String>)", "fn convert(items: Vec<String>)", "Vec<i32>"),
        ("let flag: bool = a < b && c > d;", "let flag: bool = a < b && c > d;", "a<bool>"),
        ("let flag: bool = a < b && c > d;", "let flag: bool = a < b && c > d;", "<bool>"),
        ("if (a < b) { return; }", "if (a < b) { return; }", "a > b"),
    ]:
        _, res = _verify_single(source_text=src, quote=q, value=val)
        assert res.status == "RETRY", f"Expected RETRY for mismatched generic/op {val!r}"
        assert res.issues[0].code == "QUANTITY_MISMATCH"


def test_code_config_unicode_cjk_and_strict_substring_stem_fabricated_rejections():
    from claimlayer.verifiers.code_evidence import is_code_config_assertion_self_consistent

    # Positive: non-ASCII / CJK & Greek Unicode headings/identifiers are substantive verbatim citations
    for src, q, val in [
        ("+## 全局规则\n+- 代码注释使用中文", "+## 全局规则", "+## 全局规则"),
        ("+## 全局规则\n+- 代码注释使用中文", "+- 代码注释使用中文", "代码注释使用中文"),
        ("+Δspan = abs(p_src_span - c_src_span)", "+Δspan = abs(p_src_span - c_src_span)", "Δspan"),
    ]:
        led, res = _verify_single(source_text=src, quote=q, value=val)
        assert res.status == "PASS", f"Expected PASS for Unicode {val!r}, got {res.issues}"
        assert res.verify_receipt(ledger=led, allow_unkeyed=True) is True
        assert is_code_config_assertion_self_consistent(
            Assertion(subject="pull_request", predicate="diff_evidence", value=val, source_id="diff_src", quote=q)
        ) is True

    # Negative 1: CJK & Greek substrings rejected on word boundaries
    for src, q, val in [
        ("+## 全局规则", "+## 全局规则", "全局"),
        ("+Δspan = abs(x)", "+Δspan = abs(x)", "span"),
    ]:
        _, res_sub = _verify_single(source_text=src, quote=q, value=val)
        assert res_sub.status == "RETRY"
        assert res_sub.issues[0].code == "QUANTITY_MISMATCH"

    _, res_sub_q = _verify_single(source_text="+## 全局规则", quote="全局", value="全局")
    assert res_sub_q.status == "RETRY"
    assert res_sub_q.issues[0].code == "QUOTE_NOT_IN_SOURCE"

    # Negative 2: ASCII substring and stemming remain rejected
    _, res_ascii_sub = _verify_single(
        source_text="+const address = getAddress();",
        quote="+const address = getAddress();",
        value="add",
    )
    assert res_ascii_sub.status == "RETRY"
    assert res_ascii_sub.issues[0].code == "QUANTITY_MISMATCH"

    _, res_stem = _verify_single(
        source_text="+const running = true;",
        quote="+const running = true;",
        value="run",
    )
    assert res_stem.status == "RETRY"
    assert res_stem.issues[0].code == "QUANTITY_MISMATCH"

    # Negative 3: fabricated CJK quote and punctuation-only quote remain rejected
    _, res_fab = _verify_single(
        source_text="+## 全局规则",
        quote="+## 全局配置",
        value="+## 全局配置",
    )
    assert res_fab.status == "RETRY"
    assert res_fab.issues[0].code == "QUOTE_NOT_IN_SOURCE"

    for punct_q in ("---", "{", "()"):
        _, res_punct = _verify_single(
            source_text=f"diff\n{punct_q}\nend",
            quote=punct_q,
            value=punct_q,
        )
        assert res_punct.status == "RETRY"
        assert res_punct.issues[0].code == "QUOTE_NOT_IN_SOURCE"

