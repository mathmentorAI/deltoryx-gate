"""Bounded domain contract: B2B Commercial Contracts (MSA + Amendments + SLA).

Every proposition admitted under `B2B_CONTRACTS_VOCABULARY` MUST declare an
explicit `condition_status` ('unconditional' | 'conditional' |
'unknown_condition' | 'unrepresentable_condition'), preventing missing fields
from silently collapsing into unconditional claims.
"""
from __future__ import annotations

from ..models import AcceptancePolicy
from ..vocabulary import FieldSpec, PredicateSpec, Vocabulary

B2B_CONTRACTS_VOCABULARY = Vocabulary(
    name="b2b-commercial-contracts",
    version="1.0.0",
    require_explicit_condition_status=True,
    scope_keys=("service_tier", "region", "clause_id", "event_id"),
    predicates=(
        PredicateSpec(
            name="payment_terms",
            value=FieldSpec(
                "quantity",
                dimensions=("duration_calendar_days", "duration_business_days"),
                description="Invoice payment deadline.",
            ),
            conditions=(
                ("invoice_disputed", FieldSpec("boolean")),
                ("invoice_amount", FieldSpec("quantity", dimensions=("currency:USD", "currency:EUR"))),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Contractual payment period for invoices.",
        ),
        PredicateSpec(
            name="liability_cap",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR"),
                description="Aggregate monetary liability cap.",
            ),
            conditions=(
                ("breach_type", FieldSpec("enum", enum=("general", "confidentiality", "ip_indemnity", "gross_negligence"))),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Maximum aggregate liability under the agreement.",
        ),
        PredicateSpec(
            name="sla_uptime_target",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio",),
                description="Committed monthly service availability percentage.",
            ),
            conditions=(
                ("force_majeure", FieldSpec("boolean")),
                ("scheduled_maintenance", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "fact"),
            description="Minimum service uptime target.",
        ),
        PredicateSpec(
            name="sla_credit_payout",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR"),
                description="Monetary credit payable upon SLA breach.",
            ),
            conditions=(
                ("downtime_hours", FieldSpec("quantity", dimensions=("duration_hours",))),
                ("force_majeure", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "fact"),
            description="Service-level credit owed for qualifying downtime.",
        ),
        PredicateSpec(
            name="cure_period",
            value=FieldSpec(
                "quantity",
                dimensions=("duration_calendar_days", "duration_business_days"),
                description="Period allowed to cure a material breach.",
            ),
            conditions=(
                ("breach_type", FieldSpec("enum", enum=("general", "payment", "confidentiality", "ip_indemnity"))),
            ),
            assertion_kinds=("requirement", "fact"),
            description="Contractual cure window after written notice.",
        ),
        PredicateSpec(
            name="auto_renewal",
            value=FieldSpec("boolean", description="Whether the agreement auto-renews."),
            conditions=(
                ("notice_given_days", FieldSpec("quantity", dimensions=("duration_calendar_days",))),
            ),
            assertion_kinds=("requirement", "fact"),
            description="Automatic renewal rule.",
        ),
        PredicateSpec(
            name="governing_law",
            value=FieldSpec("entity", description="Governing jurisdiction."),
            assertion_kinds=("fact", "definition", "requirement"),
            description="Governing law jurisdiction.",
        ),
    ),
)

B2B_DEFAULT_ACCEPTANCE_POLICY = AcceptancePolicy(
    tau=0.70,
    delta=0.20,
    allowed_assertion_kinds=("fact", "definition", "requirement"),
    version="b2b-contracts-v1",
)
