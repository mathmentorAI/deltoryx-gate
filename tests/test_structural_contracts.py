"""Generated structural invariants; no external evaluation fixtures."""
from decimal import Decimal
from itertools import permutations

import pytest

from claimlayer import Assertion, Ledger, Policy, Source, verify
from claimlayer.verifiers.diff import build_comparison_policy
from claimlayer.v4 import Vocabulary, PredicateSpec, FieldSpec


def check(text, value, predicate="metric", subject="Asset", policy=None, quote=None, unit=None):
    return verify([Assertion(subject=subject, predicate=predicate, value=value, unit=unit,
                             quote=quote or text, source_id="record")],
                  Ledger(sources=[Source(id="record", text=text, tier="authoritative")]), policy)


@pytest.mark.parametrize("n", [3, 8, 17, 26, 49])
def test_generated_compound_units_and_exact_calendar_conversion(n):
    for qualifier in ("Business", "Calendar"):
        text = f"The review lasts {n}-{qualifier}-Day."
        assert check(text, f"{n} {qualifier} Days").status == "PASS"
        assert check(text, f"{n+1} {qualifier} Days").status == "RETRY"
        opposite = "Calendar" if qualifier == "Business" else "Business"
        assert check(text, f"{n} {opposite} Days").status == "RETRY"
    assert check(f"The lease lasts {n*12} months.", f"{n} years").status == "PASS"
    assert check(f"The lease lasts {n*12} days.", f"{n} years").status == "RETRY"


@pytest.mark.parametrize("n", [2, 7, 13, 22])
def test_generated_sign_period_and_date_binding(n):
    assert check(f"The ratio is ({n}.0)%.", f"{n}%").status == "RETRY"
    assert check(f"The ratio is ({n}.0)%.", f"-{n}%").status == "PASS"
    assert check(f"The rate is {n}% per week.", f"{n}% per month").status == "RETRY"
    assert check(f"The date is {n:02}.08.2034.", f"2034-08-{n:02}").status == "PASS"
    text = f"The opening date is {n:02} August 2034 closing date is {n:02} September 2035."
    assert check(text, f"2035-09-{n:02}", "closing_date").status == "PASS"
    assert check(text, f"2034-08-{n:02}", "closing_date").status == "RETRY"


def test_generated_parallel_list_binding_all_permutations():
    for values in permutations([3, 7, 11]):
        text = ("Tranche D, Tranche E and Tranche F receive "
                + ", ".join(f"{v}%" for v in values) + ", respectively.")
        for index, name in enumerate(["Tranche D", "Tranche E", "Tranche F"]):
            for proposed in values:
                verdict = check(text, f"{proposed}%", subject=name)
                assert (verdict.status == "PASS") == (proposed == values[index])


@pytest.mark.parametrize("scale,factor", [("thousands", 1000), ("millions", 1000000), ("billions", 1000000000)])
def test_generated_table_scope_and_padding(scale, factor):
    for row_count in (0, 3, 60):
        row = "Receivable balance: 321"
        text = f"Figures in {scale}\n" + "\n".join(f"Line {i}: 123" for i in range(row_count)) + "\n" + row
        # Avoid an empty paragraph when there are no filler rows.
        text = text.replace("\n\n", "\n")
        assert check(text, str(321*factor), quote=row).status == "PASS"
        assert check(text, "321", quote=row).status == "RETRY"
    quote = "The surcharge is $73."
    for separator in ("\n", "\n\n"):
        text = f"Inventory (in {scale}): $21" + separator + quote
        assert check(text, "$73", quote=quote).status == "PASS"
    text = f"Figures in {scale}\n\n" + quote
    assert check(text, "$73", quote=quote).status == "PASS"


def test_fraction_grammar_rounding():
    for numerator, word in [(1, "one"), (2, "two"), (3, "three")]:
        for denominator, ordinal in [(4, "fourths"), (5, "fifths"), (8, "eighths")]:
            text = f"The share is {word}-{ordinal} of the pool."
            expected = Decimal(numerator) / Decimal(denominator) * 100
            assert check(text, f"{expected}%").status == "PASS"
            assert check(text, f"{expected+1}%").status == "RETRY"


def test_comparison_requires_value_not_only_topic_and_roundtrips_policy():
    a = Source(id="record", tier="authoritative", text="Section 8: Allowance. The allowance is EUR 640.")
    b = Source(id="draft", tier="standard", text="Section 8: Allowance. The allowance is EUR 810.")
    policy, _ = build_comparison_policy(a, b)
    for pol in (policy, Policy(**policy.to_dict())):
        assert check(a.text, "allowance", "section_8", policy=pol).status == "RETRY"
        assert check(a.text, "EUR 640", "section_8", policy=pol).status == "PASS"


def test_vocabulary_bridge_validates_value_and_unit_not_only_predicate():
    vocab = Vocabulary("contracts", "1", (
        PredicateSpec("state", FieldSpec("enum", enum=("OPEN", "CLOSED"))),
        PredicateSpec("measurement", FieldSpec("quantity", dimensions=("duration_months",))),
    ))
    policy = Policy(vocabulary=vocab)
    assert check("The state is OPEN.", "OPEN", "state", policy=policy).status == "PASS"
    rejected = check("The state is UNKNOWN.", "UNKNOWN", "state", policy=policy)
    assert "UNREPRESENTABLE_IN_VOCABULARY" in {i.code for i in rejected.issues}
    rejected = check("The measurement is 9 USD.", "9", "measurement", unit="USD", policy=policy)
    assert "UNREPRESENTABLE_IN_VOCABULARY" in {i.code for i in rejected.issues}


def test_operator_aliases_bind_numeric_limits_without_quote_synonyms():
    for alias in ("measure_alpha", "measure_beta", "measure_gamma"):
        policy = Policy(numeric_bounds={"ceiling": (None, 120)}, predicate_aliases={alias: "ceiling"})
        assert check("The measured amount is 130 EUR.", "130 EUR", alias, policy=policy).status == "ESCALATE"
        assert check("The measured amount is 110 EUR.", "110 EUR", alias, policy=policy).status == "PASS"


def test_mcp_preserves_operator_schema_and_rejects_alias_injection():
    from claimlayer import MCPVerificationServer
    from claimlayer.mcp_server import MCPError
    vocabulary = Vocabulary("server", "1", (PredicateSpec("state", FieldSpec("string")),))
    policy = Policy(vocabulary=vocabulary, predicate_aliases={"alternate": "state"},
                    required_value_tokens={"section_8": ["640:EUR"]})
    server = MCPVerificationServer(Ledger(), default_policy=policy)
    for raw in (None, {}):
        merged = server._validate_and_merge_policy(raw)
        assert merged.vocabulary is vocabulary
        assert merged.required_value_tokens == policy.required_value_tokens
        assert merged.predicate_aliases == policy.predicate_aliases
    with pytest.raises(MCPError):
        server._validate_and_merge_policy({"predicate_aliases": {"other": "state"}})
