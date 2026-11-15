"""Bounded domain contract for Structured Finance & Securitization (v4).

Defines the typed Vocabulary, scope keys, condition semantics, and
AcceptancePolicy for prospectuses, pooling-and-servicing agreements (PSAs),
indentures, supplemental indentures, and servicer/trustee reports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Mapping, Sequence

from ..domain_profile import DomainProfile, PolicyConfiguration
from ..models import AcceptancePolicy
from ..vocabulary import FieldSpec, PredicateSpec, Vocabulary, label_key

VALID_WATERFALL_PAYMENT_TYPES: frozenset[str] = frozenset({
    "fees",
    "interest",
    "principal",
    "reserve",
    "residual",
    "other",
})


def _norm_beneficiary(raw: str | None) -> str:
    txt = label_key(raw or "")
    # Normalize common phrasing like "class c notes" -> "class_c", "reserve account replenishment" -> "reserve_account"
    txt = re.sub(r"\b(notes?|tranche|account|replenishment|payments?|distribution)\b", " ", txt)
    txt = re.sub(r"\s+", "_", txt.strip())
    return txt or "unknown"


def _norm_payment_type(raw: str | None) -> str:
    pt = label_key(raw or "other").replace(" ", "_")
    if pt in VALID_WATERFALL_PAYMENT_TYPES:
        return pt
    if "interest" in pt or "coupon" in pt:
        return "interest"
    if "principal" in pt or "amortization" in pt:
        return "principal"
    if "reserve" in pt or "replenish" in pt:
        return "reserve"
    if "fee" in pt or "expense" in pt or "servicer" in pt or "trustee" in pt:
        return "fees"
    if "residual" in pt or "equity" in pt or "certificate" in pt:
        return "residual"
    return "other"


def _norm_position_range(pos: int | tuple[int, int] | Sequence[int] | None) -> tuple[int, int] | None:
    if pos is None:
        return None
    if isinstance(pos, int) and not isinstance(pos, bool):
        return (pos, pos)
    if isinstance(pos, Sequence) and not isinstance(pos, (str, bytes)) and len(pos) == 2:
        lo, hi = int(pos[0]), int(pos[1])
        return (min(lo, hi), max(lo, hi))
    raise ValueError(f"Invalid waterfall position: {pos!r}")


def _slot_key(beneficiary: str, payment_type: str) -> str:
    return f"{_norm_beneficiary(beneficiary)}:{_norm_payment_type(payment_type)}"


def _cond_signature(conditions: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, Any], ...]:
    items: list[tuple[str, str]] = []
    for cond in conditions:
        if isinstance(cond, Mapping):
            items.extend((label_key(str(k)), str(v)) for k, v in sorted(cond.items()))
    return tuple(sorted(set(items)))


@dataclass(frozen=True)
class WaterfallStepSpec:
    """Structured representation of a single cashflow waterfall step with provenance."""

    step_id: str
    beneficiary: str
    payment_type: str
    position: int | tuple[int, int] | None = None
    before: tuple[str, ...] = ()
    after: tuple[str, ...] = ()
    conditions: tuple[dict[str, Any], ...] = ()
    regime: str = "pre_acceleration"
    document_id: str = ""
    clause_id: str = ""
    revision: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "payment_type", _norm_payment_type(self.payment_type))
        if isinstance(self.position, list):
            if len(self.position) == 2:
                object.__setattr__(self, "position", (int(self.position[0]), int(self.position[1])))
            elif len(self.position) == 1:
                object.__setattr__(self, "position", int(self.position[0]))
        object.__setattr__(self, "before", tuple(self.before))
        object.__setattr__(self, "after", tuple(self.after))
        object.__setattr__(
            self,
            "conditions",
            tuple(dict(c) for c in self.conditions if isinstance(c, Mapping)),
        )
        object.__setattr__(self, "regime", label_key(self.regime or "pre_acceleration").replace(" ", "_"))

    @property
    def slot_key(self) -> str:
        return _slot_key(self.beneficiary, self.payment_type)

    @property
    def position_range(self) -> tuple[int, int] | None:
        return _norm_position_range(self.position)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WaterfallStepSpec":
        pos = data.get("position")
        if isinstance(pos, Sequence) and not isinstance(pos, (str, bytes)) and len(pos) == 2:
            pos = (int(pos[0]), int(pos[1]))
        elif pos is not None and not isinstance(pos, bool):
            pos = int(pos)
        return cls(
            step_id=str(data.get("step_id") or _slot_key(str(data.get("beneficiary", "")), str(data.get("payment_type", "")))),
            beneficiary=str(data.get("beneficiary") or ""),
            payment_type=str(data.get("payment_type") or "other"),
            position=pos,
            before=tuple(str(x) for x in data.get("before", ()) or ()),
            after=tuple(str(x) for x in data.get("after", ()) or ()),
            conditions=tuple(dict(c) for c in data.get("conditions", ()) or () if isinstance(c, Mapping)),
            regime=str(data.get("regime") or "pre_acceleration"),
            document_id=str(data.get("document_id") or ""),
            clause_id=str(data.get("clause_id") or ""),
            revision=int(data.get("revision", 1)),
        )


@dataclass(frozen=True)
class WaterfallComparisonResult:
    """Deterministic result from verifying single or cross-document waterfall priorities."""

    status: str  # "verified" | "refuted" | "unknown"
    reasons: tuple[str, ...]
    inversions: tuple[dict[str, Any], ...] = ()
    beneficiary_swaps: tuple[dict[str, Any], ...] = ()
    payment_type_confusions: tuple[dict[str, Any], ...] = ()
    cycles: tuple[tuple[str, ...], ...] = ()
    regime_mismatches: tuple[dict[str, Any], ...] = ()
    step_shift_only_non_conflicts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "inversions": list(self.inversions),
            "beneficiary_swaps": list(self.beneficiary_swaps),
            "payment_type_confusions": list(self.payment_type_confusions),
            "cycles": [list(c) for c in self.cycles],
            "regime_mismatches": list(self.regime_mismatches),
            "step_shift_only_non_conflicts": list(self.step_shift_only_non_conflicts),
        }


def _normalize_target_ref(raw_ref: str, step_id_to_slot: Mapping[str, str]) -> str:
    if raw_ref in step_id_to_slot:
        return step_id_to_slot[raw_ref]
    if ":" in raw_ref:
        b_raw, pt_raw = raw_ref.split(":", 1)
        return _slot_key(b_raw, pt_raw)
    return _norm_beneficiary(raw_ref)


def _build_precedence_graph(
    steps: Sequence[WaterfallStepSpec],
) -> tuple[dict[str, set[str]], dict[str, WaterfallStepSpec], list[tuple[str, ...]]]:
    """Build directed slot precedence graph and detect cycles deterministically."""
    id_to_slot: dict[str, str] = {s.step_id: s.slot_key for s in steps}
    slot_to_step: dict[str, WaterfallStepSpec] = {}
    edges: dict[str, set[str]] = {}
    nodes: set[str] = set()

    for s in steps:
        sk = s.slot_key
        nodes.add(sk)
        slot_to_step.setdefault(sk, s)
        edges.setdefault(sk, set())

    for s in steps:
        sk = s.slot_key
        for b_ref in s.before:
            tgt = _normalize_target_ref(b_ref, id_to_slot)
            if tgt and tgt != sk:
                nodes.add(tgt)
                edges.setdefault(sk, set()).add(tgt)
                edges.setdefault(tgt, set())
        for a_ref in s.after:
            src = _normalize_target_ref(a_ref, id_to_slot)
            if src and src != sk:
                nodes.add(src)
                edges.setdefault(src, set()).add(sk)
                edges.setdefault(sk, set())

    # Positional edges within same condition signature
    for i, s1 in enumerate(steps):
        p1 = s1.position_range
        if p1 is None:
            continue
        c1 = _cond_signature(s1.conditions)
        for j, s2 in enumerate(steps):
            if i == j:
                continue
            p2 = s2.position_range
            if p2 is None:
                continue
            c2 = _cond_signature(s2.conditions)
            if c1 != c2:
                continue
            if p1[1] < p2[0] and s1.slot_key != s2.slot_key:
                edges.setdefault(s1.slot_key, set()).add(s2.slot_key)

    # Detect cycles via 3-color DFS
    state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=done
    stack: list[str] = []
    cycles: list[tuple[str, ...]] = []
    seen_cycle_sets: set[frozenset[str]] = set()

    def _dfs(u: str) -> None:
        state[u] = 1
        stack.append(u)
        for v in sorted(edges.get(u, ())):
            st = state.get(v, 0)
            if st == 1:
                idx = stack.index(v)
                cyc = tuple(stack[idx:] + [v])
                key = frozenset(cyc[:-1])
                if key not in seen_cycle_sets:
                    seen_cycle_sets.add(key)
                    cycles.append(cyc)
            elif st == 0:
                _dfs(v)
        stack.pop()
        state[u] = 2

    for n in sorted(nodes):
        if state.get(n, 0) == 0:
            _dfs(n)

    return edges, slot_to_step, cycles


def _transitive_closure(edges: Mapping[str, set[str]]) -> dict[str, set[str]]:
    reach: dict[str, set[str]] = {u: set() for u in edges}
    for start in edges:
        frontier = list(edges.get(start, ()))
        visited: set[str] = set()
        while frontier:
            cur = frontier.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reach[start].add(cur)
            for nxt in edges.get(cur, ()):
                if nxt not in visited:
                    frontier.append(nxt)
    return reach


def _parse_slot_str(raw: str, id_map: Mapping[str, str] | None = None) -> tuple[str, str]:
    if id_map and raw in id_map:
        raw = id_map[raw]
    if ":" in raw:
        b_raw, pt_raw = raw.split(":", 1)
        return _norm_beneficiary(b_raw), _norm_payment_type(pt_raw)
    return _norm_beneficiary(raw), "other"


def verify_waterfall_steps(
    governing_steps: Sequence[WaterfallStepSpec | Mapping[str, Any]],
    subordinate_steps: Sequence[WaterfallStepSpec | Mapping[str, Any]] | None = None,
    *,
    claimed_relations: Sequence[Mapping[str, Any]] | None = None,
    allow_cross_regime_if_qualified: bool = False,
) -> WaterfallComparisonResult:
    """Verify waterfall priority ordering, regime comparability, and candidate assertions."""
    gov = [
        s if isinstance(s, WaterfallStepSpec) else WaterfallStepSpec.from_dict(s)
        for s in governing_steps
    ]
    sub = (
        [s if isinstance(s, WaterfallStepSpec) else WaterfallStepSpec.from_dict(s) for s in subordinate_steps]
        if subordinate_steps is not None
        else []
    )

    if not gov:
        return WaterfallComparisonResult(
            status="unknown",
            reasons=("MISSING_GOVERNING_WATERFALL_STEPS",),
        )

    cycles_all: list[tuple[str, ...]] = []
    gov_edges_by_regime: dict[str, dict[str, set[str]]] = {}
    gov_reach_by_regime: dict[str, dict[str, set[str]]] = {}
    gov_slots_by_regime: dict[str, dict[str, WaterfallStepSpec]] = {}

    for regime in sorted({s.regime for s in gov}):
        r_steps = [s for s in gov if s.regime == regime]
        edges, slot_map, cycs = _build_precedence_graph(r_steps)
        cycles_all.extend(cycs)
        gov_edges_by_regime[regime] = edges
        gov_reach_by_regime[regime] = _transitive_closure(edges)
        gov_slots_by_regime[regime] = slot_map

    sub_edges_by_regime: dict[str, dict[str, set[str]]] = {}
    sub_reach_by_regime: dict[str, dict[str, set[str]]] = {}
    sub_slots_by_regime: dict[str, dict[str, WaterfallStepSpec]] = {}
    for regime in sorted({s.regime for s in sub}):
        r_steps = [s for s in sub if s.regime == regime]
        edges, slot_map, cycs = _build_precedence_graph(r_steps)
        cycles_all.extend(cycs)
        sub_edges_by_regime[regime] = edges
        sub_reach_by_regime[regime] = _transitive_closure(edges)
        sub_slots_by_regime[regime] = slot_map

    inversions: list[dict[str, Any]] = []
    beneficiary_swaps: list[dict[str, Any]] = []
    payment_type_confusions: list[dict[str, Any]] = []
    regime_mismatches: list[dict[str, Any]] = []
    non_conflicts: list[dict[str, Any]] = []

    if cycles_all:
        return WaterfallComparisonResult(
            status="refuted",
            reasons=("WATERFALL_ORDER_CYCLE",),
            cycles=tuple(cycles_all),
        )

    # Check regime comparability when comparing two waterfalls
    if sub:
        gov_regimes = set(gov_slots_by_regime)
        sub_regimes = set(sub_slots_by_regime)
        disjoint = not (gov_regimes & sub_regimes)
        if disjoint and not allow_cross_regime_if_qualified:
            regime_mismatches.append({
                "governing_regimes": sorted(gov_regimes),
                "subordinate_regimes": sorted(sub_regimes),
                "reason": "Waterfalls govern incompatible acceleration/default regimes and cannot be directly equated.",
            })
        for regime in sorted(gov_regimes & sub_regimes):
            gov_by_id = {s.step_id: s for s in gov if s.regime == regime and s.step_id}
            sub_by_id = {s.step_id: s for s in sub if s.regime == regime and s.step_id}
            for shared_id in sorted(set(gov_by_id) & set(sub_by_id)):
                g_step = gov_by_id[shared_id]
                s_step = sub_by_id[shared_id]
                if _norm_beneficiary(g_step.beneficiary) != _norm_beneficiary(s_step.beneficiary):
                    beneficiary_swaps.append({
                        "regime": regime,
                        "step_id": shared_id,
                        "position": s_step.position,
                        "expected_beneficiary": g_step.beneficiary,
                        "claimed_beneficiary": s_step.beneficiary,
                        "governing_doc": g_step.document_id,
                        "subordinate_doc": s_step.document_id,
                    })
                elif g_step.payment_type != s_step.payment_type:
                    payment_type_confusions.append({
                        "regime": regime,
                        "step_id": shared_id,
                        "position": s_step.position,
                        "beneficiary": g_step.beneficiary,
                        "expected_payment_type": g_step.payment_type,
                        "claimed_payment_type": s_step.payment_type,
                        "governing_doc": g_step.document_id,
                        "subordinate_doc": s_step.document_id,
                    })
            g_reach = gov_reach_by_regime[regime]
            s_reach = sub_reach_by_regime[regime]
            g_slots = gov_slots_by_regime[regime]
            s_slots = sub_slots_by_regime[regime]
            common = sorted(set(g_slots) & set(s_slots))
            for i, u in enumerate(common):
                for v in common[i + 1 :]:
                    g_uv = v in g_reach.get(u, ())
                    g_vu = u in g_reach.get(v, ())
                    s_uv = v in s_reach.get(u, ())
                    s_vu = u in s_reach.get(v, ())
                    # Check condition comparability between pairs before declaring inversion
                    c_g_u = _cond_signature(g_slots[u].conditions)
                    c_g_v = _cond_signature(g_slots[v].conditions)
                    c_s_u = _cond_signature(s_slots[u].conditions)
                    c_s_v = _cond_signature(s_slots[v].conditions)
                    if (g_uv and s_vu) or (g_vu and s_uv):
                        if c_g_u != c_s_u or c_g_v != c_s_v:
                            regime_mismatches.append({
                                "slot_a": u,
                                "slot_b": v,
                                "governing_conditions": [list(c_g_u), list(c_g_v)],
                                "subordinate_conditions": [list(c_s_u), list(c_s_v)],
                                "reason": "Slots carry different triggering conditions between documents.",
                            })
                        else:
                            first_in_gov, second_in_gov = (u, v) if g_uv else (v, u)
                            inversions.append({
                                "regime": regime,
                                "governing_first": first_in_gov,
                                "governing_second": second_in_gov,
                                "governing_positions": [
                                    g_slots[first_in_gov].position,
                                    g_slots[second_in_gov].position,
                                ],
                                "subordinate_positions": [
                                    s_slots[first_in_gov].position,
                                    s_slots[second_in_gov].position,
                                ],
                                "governing_doc": g_slots[first_in_gov].document_id,
                                "subordinate_doc": s_slots[first_in_gov].document_id,
                            })
                    elif (g_uv and s_uv) or (g_vu and s_vu):
                        first_s, second_s = (u, v) if g_uv else (v, u)
                        if (
                            g_slots[first_s].position != s_slots[first_s].position
                            or g_slots[second_s].position != s_slots[second_s].position
                        ):
                            non_conflicts.append({
                                "regime": regime,
                                "earlier_slot": first_s,
                                "later_slot": second_s,
                                "governing_positions": [
                                    g_slots[first_s].position,
                                    g_slots[second_s].position,
                                ],
                                "subordinate_positions": [
                                    s_slots[first_s].position,
                                    s_slots[second_s].position,
                                ],
                                "note": "Step numbering shifted without changing relative payment priority.",
                            })

    # Validate candidate-claimed relations against governing/subordinate waterfalls
    gov_id_map = {s.step_id: s.slot_key for s in gov}
    sub_id_map = {s.step_id: s.slot_key for s in sub}
    for claim in claimed_relations or ():
        if not isinstance(claim, Mapping):
            continue
        source_target = label_key(str(claim.get("source") or "governing")).replace(" ", "_")
        regime = label_key(str(claim.get("regime") or gov[0].regime)).replace(" ", "_")
        if source_target == "subordinate" and sub:
            reach_map = sub_reach_by_regime.get(regime, {})
            slots_map = sub_slots_by_regime.get(regime, {})
            id_map = sub_id_map
        else:
            reach_map = gov_reach_by_regime.get(regime, {})
            slots_map = gov_slots_by_regime.get(regime, {})
            id_map = gov_id_map

        if not slots_map and gov_slots_by_regime:
            regime_mismatches.append({
                "claimed_regime": regime,
                "available_regimes": sorted(gov_slots_by_regime),
                "claim": dict(claim),
            })
            continue

        # Pairwise precedence claim: before -> after
        if "before" in claim and "after" in claim:
            b_ben, b_pt = _parse_slot_str(str(claim["before"]), id_map)
            a_ben, a_pt = _parse_slot_str(str(claim["after"]), id_map)
            b_slot = _slot_key(b_ben, b_pt) if "other" not in (b_pt,) or ":" in str(claim["before"]) else None
            a_slot = _slot_key(a_ben, a_pt) if "other" not in (a_pt,) or ":" in str(claim["after"]) else None
            if b_slot is None:
                b_slot = next((k for k in slots_map if k.startswith(f"{b_ben}:")), _slot_key(b_ben, b_pt))
            if a_slot is None:
                a_slot = next((k for k in slots_map if k.startswith(f"{a_ben}:")), _slot_key(a_ben, a_pt))

            if b_slot in slots_map and a_slot in slots_map:
                if a_slot in reach_map and b_slot in reach_map[a_slot]:
                    # Claim inverted relative priority! Determine whether beneficiary swap, payment type confusion, or priority inversion
                    if b_ben == a_ben and b_pt != a_pt:
                        payment_type_confusions.append({
                            "beneficiary": b_ben,
                            "claimed_earlier": b_slot,
                            "claimed_later": a_slot,
                            "reason": f"Payment types '{b_pt}' and '{a_pt}' for '{b_ben}' are inverted.",
                        })
                    else:
                        beneficiary_swaps.append({
                            "claimed_earlier": b_slot,
                            "claimed_later": a_slot,
                            "source": source_target,
                            "regime": regime,
                        })
                        inversions.append({
                            "claimed_earlier": b_slot,
                            "claimed_later": a_slot,
                            "source": source_target,
                            "regime": regime,
                        })

        # Slot-at-position claim: {"position": 6, "beneficiary": "...", "payment_type": "..."}
        if "position" in claim and ("beneficiary" in claim or "payment_type" in claim):
            c_pos = _norm_position_range(claim.get("position"))
            c_ben = _norm_beneficiary(str(claim.get("beneficiary") or ""))
            c_pt = _norm_payment_type(str(claim.get("payment_type") or "other"))
            for sk, sp in slots_map.items():
                if sp.position_range == c_pos:
                    sp_ben = _norm_beneficiary(sp.beneficiary)
                    if c_ben and c_ben != "unknown" and sp_ben != c_ben:
                        beneficiary_swaps.append({
                            "position": c_pos,
                            "expected_beneficiary": sp.beneficiary,
                            "claimed_beneficiary": claim.get("beneficiary"),
                            "source": source_target,
                        })
                    elif c_pt != "other" and sp.payment_type != c_pt:
                        payment_type_confusions.append({
                            "position": c_pos,
                            "beneficiary": sp.beneficiary,
                            "expected_payment_type": sp.payment_type,
                            "claimed_payment_type": c_pt,
                            "source": source_target,
                        })

    reasons: list[str] = []
    if inversions:
        reasons.append("WATERFALL_PRIORITY_INVERSION")
    if beneficiary_swaps:
        reasons.append("WATERFALL_BENEFICIARY_SWAP")
    if payment_type_confusions:
        reasons.append("WATERFALL_PAYMENT_TYPE_CONFUSION")
    if regime_mismatches:
        reasons.append("WATERFALL_INCOMPARABLE_REGIME")

    if inversions or beneficiary_swaps or payment_type_confusions:
        status = "refuted"
    elif regime_mismatches:
        status = "unknown"
    else:
        status = "verified"
        reasons.append("WATERFALL_PRIORITY_VERIFIED")

    return WaterfallComparisonResult(
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        inversions=tuple(inversions),
        beneficiary_swaps=tuple(beneficiary_swaps),
        payment_type_confusions=tuple(payment_type_confusions),
        cycles=tuple(cycles_all),
        regime_mismatches=tuple(regime_mismatches),
        step_shift_only_non_conflicts=tuple(non_conflicts),
    )


SECURITIZATION_VOCABULARY = Vocabulary(
    name="securitization",
    version="1.3.0",
    require_explicit_condition_status=True,
    scope_keys=("tranche", "clause_id", "reporting_period", "event_id", "regime"),
    register_status_terms=(
        "no material discrepancy",
        "no material discrepancies",
        "no discrepancy",
        "no discrepancies",
        "no conflict",
        "no conflicts",
        "fully consistent",
        "consistent across documents",
        "no contradiction",
    ),
    predicates=(
        PredicateSpec(
            name="coupon_rate",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio", "annual_rate"),
                description="Interest rate or coupon payable on a tranche of notes.",
            ),
            conditions=(
                ("step_up_triggered", FieldSpec("boolean")),
                ("rate_cap_applied", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            aliases=("interest rate", "per annum", "bears interest", "coupon"),
            description="Interest rate payable on a note tranche.",
        ),
        PredicateSpec(
            name="coupon_spread",
            value=FieldSpec(
                "quantity",
                dimensions=("ratio_difference", "annual_rate_difference"),
                description="Interest rate spread between two documents expressed in percentage points or basis points.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Rate spread across governing and subordinate documents.",
        ),
        PredicateSpec(
            name="annual_interest_impact",
            value=FieldSpec(
                "quantity",
                dimensions=("currency:USD", "currency:EUR", "currency_rate:USD/year", "currency_rate:EUR/year"),
                description="Illustrative annual interest impact derived from notional principal and rate spread under explicit assumptions.",
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            description="Illustrative annual interest dollar impact under explicit assumptions.",
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
            name="waterfall_schedule",
            value=FieldSpec.list(
                FieldSpec.record(
                    description="Ordered distribution step in a payment waterfall schedule.",
                    step=FieldSpec("string", description="Step identifier or range (e.g., '1', '3-7', '13')."),
                    beneficiary=FieldSpec("string", description="Fee, note class, account, or payee at this step."),
                ),
                min_items=1,
                description="Ordered sequence of payment priority steps in the waterfall.",
            ),
            conditions=(
                ("sequential_paydown_active", FieldSpec("boolean")),
            ),
            assertion_kinds=("requirement", "definition", "fact"),
            aliases=(
                "priority of distributions",
                "waterfall",
                "flow of funds",
                "available funds distribute",
                "order of priority",
                "distribution sequence",
            ),
            description="Ordered waterfall distribution schedule across payment steps.",
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
            aliases=("note balance", "initial note balance", "principal balance", "principal amount", "initial principal", "notes"),
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

SECURITIZATION_EXCLUDED_COMPARISON_ROLES: dict[str, str] = {}

SECURITIZATION_DOMAIN_PROFILE = DomainProfile(
    profile_id="securitization_default",
    version="1.1.0",
    vocabulary=SECURITIZATION_VOCABULARY,
    policy=PolicyConfiguration(
        tau=SECURITIZATION_ACCEPTANCE_POLICY.tau,
        delta=SECURITIZATION_ACCEPTANCE_POLICY.delta,
        excluded_comparison_roles=dict(SECURITIZATION_EXCLUDED_COMPARISON_ROLES),
    ),
)

__all__ = [
    "SECURITIZATION_VOCABULARY",
    "SECURITIZATION_ACCEPTANCE_POLICY",
    "SECURITIZATION_EXCLUDED_COMPARISON_ROLES",
    "SECURITIZATION_DOMAIN_PROFILE",
    "VALID_WATERFALL_PAYMENT_TYPES",
    "WaterfallStepSpec",
    "WaterfallComparisonResult",
    "verify_waterfall_steps",
]
