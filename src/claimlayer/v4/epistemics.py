"""Deterministic basis weights for the current verification core; these are policy values, not probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

INFERENCE_WEIGHTS = {
        "arithmetic": 1.00,
        "logical": 1.00,
        "deductive": 1.00,
        "syntactic": 1.00,
        "reductio": 1.00,
        "semantic": 0.90,
        "fixed_point": 0.90,
        "spatial": 0.90,
        "bayesian": 0.85,
        "evolutionary": 0.80,
        "common_sense": 0.65,
        "temporal": 0.60,
        "inductive": 0.55,
        "causal": 0.55,
        "elimination": 0.55,
        "occam": 0.55,
        "abductive_inv": 0.55,
        "dialectical": 0.55,
        "multi_hop": 0.50,
        "default": 0.50,
        "authority": 0.50,
        "transductive": 0.50,
        "probabilistic": 0.40,
        "analogical": 0.35,
        "pragmatic": 0.35,
        "intentionality": 0.25,
        "attribution": 0.20,
        "heuristic": 0.10,
        "abductive": 0.10,
        "counterfactual": 0.05,
        "narrative": 0.00,
    }
BLOCKED_INFERENCE_TYPES = frozenset({
        "heuristic",
        "abductive",
        "counterfactual",
        "narrative",
    })

from ..semantic.models import digest

EXPLICIT = 'explicit'


@dataclass(frozen=True)
class EpistemicProfile:
    """Basis weights are policy values, not calibrated probabilities."""
    name: str = 'gate-inference-registry'
    version: str = '1'
    weights: MappingProxyType = field(default_factory=lambda: MappingProxyType({EXPLICIT: 1.0, **INFERENCE_WEIGHTS}))
    blocked: frozenset = BLOCKED_INFERENCE_TYPES

    def __post_init__(self):
        if EXPLICIT in self.blocked:
            raise ValueError('Explicit source statements cannot be blocked')
        for basis, weight in self.weights.items():
            if not isinstance(basis, str) or not 0 <= float(weight) <= 1:
                raise ValueError('Basis weights must be labelled values in [0,1]')
        object.__setattr__(self, 'weights', MappingProxyType(dict(self.weights)))
        object.__setattr__(self, 'blocked', frozenset(self.blocked))

    def admissible(self, basis):
        return isinstance(basis, str) and basis in self.weights and basis not in self.blocked

    def weight(self, basis):
        if not self.admissible(basis):
            raise ValueError(f'Basis {basis!r} is not admissible under {self.name}')
        return float(self.weights[basis])

    def describe(self):
        return {'name': self.name, 'version': self.version,
                'admissible_bases': {k: float(v) for k, v in sorted(self.weights.items()) if k not in self.blocked},
                'blocked_bases': sorted(self.blocked)}

    @property
    def fingerprint(self):
        return digest(self.describe())


DEFAULT_PROFILE = EpistemicProfile()
