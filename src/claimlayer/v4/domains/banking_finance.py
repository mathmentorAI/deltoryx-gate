"""Bounded domain contract: Banking & Leveraged Finance (Term Sheets, Playbooks, Financial Models)."""

from __future__ import annotations

from ..models import AcceptancePolicy
from ..vocabulary import FieldSpec, PredicateSpec, Vocabulary

BANKING_CONTEXT_AUTHORITY_MAP: dict[str, tuple[str, ...]] = {
    "financial_metrics": ("quantitative_financial_model",),
    "negotiation_policy": ("client_playbook_policy",),
    "baseline_terms": ("baseline_term_sheet",),
    "review_scope": ("client_instructions_scope",),
}

BANKING_SUBORDINATE_ROLES: frozenset[str] = frozenset({"counterparty_markup"})

BANKING_FINANCE_VOCABULARY = Vocabulary(
    name="banking-finance",
    version="1.1.0",
    require_explicit_condition_status=False,
    scope_keys=("facility_id", "clause_id", "tranche", "covenant_type", "period"),
    predicates=(
        PredicateSpec(
            name="facility_amount",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR"),
                description="Committed principal amount of a credit facility.",
            ),
            assertion_kinds=("definition", "fact", "requirement", "proposal"),
            aliases=("facility", "commitment", "term loan", "revolving credit", "principal"),
            description="Principal commitment of a credit facility.",
        ),
        PredicateSpec(
            name="interest_margin",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio", "annual_rate"),
                description="Applicable interest rate margin or spread over benchmark.",
            ),
            conditions=(
                ("pricing_grid_step_up", FieldSpec("boolean")),
                ("default_rate_applied", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "definition", "fact", "proposal"),
            aliases=("applicable margin", "interest rate", "spread", "bps", "per annum"),
            description="Applicable interest margin on a credit facility.",
        ),
        PredicateSpec(
            name="leverage_ratio",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio", "multiplier"),
                description="Total or senior secured net leverage covenant ratio.",
            ),
            conditions=(
                ("covenant_step_down", FieldSpec("boolean")),
                ("testing_period", FieldSpec("enum", enum=("ltm", "quarterly", "pro_forma"))),
            ),
            assertion_kinds=("requirement", "definition", "fact", "proposal"),
            aliases=("leverage", "net leverage", "total leverage", "consolidated leverage"),
            description="Financial covenant leverage ratio threshold or actual multiple.",
        ),
        PredicateSpec(
            name="ebitda",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR"),
                description="Consolidated or Adjusted EBITDA from financial summary.",
            ),
            assertion_kinds=("fact", "definition"),
            aliases=("adjusted ebitda", "consolidated ebitda", "ebitda"),
            description="Baseline EBITDA figure from financial model.",
        ),
        PredicateSpec(
            name="covenant_threshold",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio", "multiplier", "currency:USD", "currency:EUR"),
                description="Financial covenant maintenance or incurrence threshold.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            aliases=("covenant", "minimum liquidity", "interest coverage", "fixed charge"),
            description="Financial covenant threshold.",
        ),
        PredicateSpec(
            name="cure_period",
            value=FieldSpec(
                "quantity",
                dimensions=("duration_calendar_days", "duration_business_days"),
                description="Grace or equity cure period before Event of Default.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            aliases=("grace period", "equity cure", "cure period", "business days", "calendar days"),
            description="Contractual cure window.",
        ),
        PredicateSpec(
            name="baskets_and_carveouts",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR", "ratio"),
                description="Permitted debt, lien, or restricted payment basket size.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            aliases=("basket", "carveout", "carve-out", "permitted"),
            description="Permitted basket or carve-out limit.",
        ),
        PredicateSpec(
            name="markup_deviation",
            value=FieldSpec(
                "enum",
                enum=("material_deviation", "minor_deviation", "aligned"),
                description="Documented counterparty markup deviation against baseline term sheet or playbook.",
            ),
            assertion_kinds=("fact", "requirement"),
            aliases=("markup", "redline", "counterparty", "deviation", "proposed"),
            description="Counterparty markup deviation.",
        ),
    ),
)

BANKING_FINANCE_ACCEPTANCE_POLICY = AcceptancePolicy(
    tau=0.80,
    delta=0.20,
    allowed_assertion_kinds=("fact", "definition", "requirement", "proposal"),
    version="banking-finance-v1.1",
)


def resolve_banking_governing_role(category: str | None) -> tuple[str, ...]:
    """Return authoritative document roles for a banking-finance fact category."""
    if not category:
        return (
            "quantitative_financial_model",
            "client_playbook_policy",
            "baseline_term_sheet",
            "client_instructions_scope",
        )
    return BANKING_CONTEXT_AUTHORITY_MAP.get(
        category.strip().lower(),
        ("client_playbook_policy", "baseline_term_sheet"),
    )
