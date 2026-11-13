"""Bounded domain contract for Structured Finance & Securitization (v4).

Defines the typed Vocabulary, scope keys, condition semantics, and
AcceptancePolicy for prospectuses, pooling-and-servicing agreements (PSAs),
indentures, supplemental indentures, and servicer/trustee reports.
"""
from __future__ import annotations

from ..models import AcceptancePolicy
from ..vocabulary import FieldSpec, PredicateSpec, Vocabulary

SECURITIZATION_VOCABULARY = Vocabulary(
    name="securitization",
    version="1.0.0",
    require_explicit_condition_status=True,
    scope_keys=("tranche", "clause_id", "reporting_period", "event_id"),
    predicates=(
        PredicateSpec(
            name="coupon_rate",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio",),
                description="Interest rate or spread payable on a tranche of notes.",
            ),
            conditions=(
                ("step_up_triggered", FieldSpec("boolean")),
                ("rate_cap_applied", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Interest rate payable on a note tranche.",
        ),
        PredicateSpec(
            name="clean_up_call_threshold",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio",),
                description="Pool factor or principal percentage at which the clean-up call option may be exercised.",
            ),
            conditions=(
                ("pool_factor_below_threshold", FieldSpec("boolean")),
                ("notes_outstanding_pct", FieldSpec("quantity", dimensions=("ratio",))),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Threshold for exercising the clean-up call option.",
        ),
        PredicateSpec(
            name="cure_period",
            value=FieldSpec(
                "quantity",
                dimensions=("duration_calendar_days", "duration_business_days"),
                description="Grace or cure period allowed to remedy a servicer or issuer breach before default.",
            ),
            conditions=(
                ("breach_type", FieldSpec("enum", enum=("payment", "servicer", "covenant", "reporting"))),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Contractual cure period before an Event of Default.",
        ),
        PredicateSpec(
            name="waterfall_priority",
            value=FieldSpec(
                "enum",
                enum=("senior", "mezzanine", "junior", "equity"),
                description="Payment priority tier in the cashflow waterfall.",
            ),
            conditions=(
                ("sequential_paydown_active", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Waterfall payment priority tier.",
        ),
        PredicateSpec(
            name="reporting_frequency",
            value=FieldSpec(
                "enum",
                enum=("monthly", "quarterly", "semi_annual", "annual"),
                description="Cadence at which servicer/investor distribution reports must be delivered.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Investor/servicer reporting cadence.",
        ),
        PredicateSpec(
            name="note_principal",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR"),
                description="Initial or current principal balance of a note tranche or reserve account.",
            ),
            assertion_kinds=("definition", "fact", "requirement"),
            description="Principal balance of a note tranche or reserve account.",
        ),
        PredicateSpec(
            name="delinquency_trigger",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio",),
                description="Delinquency ratio threshold that triggers sequential paydown or credit enhancement.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Delinquency trigger threshold.",
        ),
        PredicateSpec(
            name="event_of_default",
            value=FieldSpec(
                "boolean",
                description="Whether an Event of Default has occurred or been triggered.",
            ),
            conditions=(
                ("cure_period_expired", FieldSpec("boolean")),
            ),
            assertion_kinds=("fact", "definition", "requirement"),
            description="Event of Default status.",
        ),
    ),
)

SECURITIZATION_ACCEPTANCE_POLICY = AcceptancePolicy(
    tau=0.70,
    delta=0.20,
    allowed_assertion_kinds=("fact", "definition", "requirement"),
    version="securitization-v1",
)

__all__ = ["SECURITIZATION_VOCABULARY", "SECURITIZATION_ACCEPTANCE_POLICY"]
