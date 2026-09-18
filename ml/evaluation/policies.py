"""Ranking policies for offline evaluation.

A policy maps a frame of candidates (decision-time columns only) to one score per row. Within a booking, higher scores
are offered first and ties are broken by driver_id. Policies never see the simulator's hidden truth; the oracle upper
bound lives in ml/evaluation/offline.py because it is the one exception.

Phase 6 adds the two ML policies of PRD Q2:

* **Option A** - rank by the model's P(accept) (`model_scores`).
* **Option B** - rank by P(accept) but only among candidates whose pickup ETA is within a margin of the fastest
  candidate of that booking. The margin is chosen on the validation split, and the rule itself lives in
  ml/inference/ranking.py because serving applies it too.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ml.features import build_features
from ml.inference.ranking import apply_eta_constraint

Policy = Callable[[pd.DataFrame], np.ndarray]

__all__ = [
    "IDLE_CAP_MIN", "Policy", "RATING_PRIOR", "WEIGHTED_RULE_WEIGHTS", "apply_eta_constraint", "model_policy",
    "model_scores", "nearest_driver", "random_order", "weighted_rule",
]

# Weights of the hand-written rule, fixed BEFORE any evaluation was run and never tuned on results.
# Unit: points. One minute of pickup ETA costs 0.2 points, so +10 percentage points of historical acceptance
# rate, -10 pp of cancellation rate, +0.4 rating stars or +10 idle minutes are each worth one minute of ETA.
WEIGHTED_RULE_WEIGHTS = {
    "estimated_eta_min": -0.2,  # per minute
    "acceptance_rate": 2.0,  # per unit (0..1)
    "cancellation_rate": -2.0,  # per unit (0..1)
    "rating": 0.5,  # per star above RATING_PRIOR; a missing rating counts as RATING_PRIOR
    "idle_time_min": 0.02,  # per minute, capped at IDLE_CAP_MIN
}
RATING_PRIOR = 4.7
IDLE_CAP_MIN = 30.0


def nearest_driver(candidates: pd.DataFrame) -> np.ndarray:
    """Baseline: the closest candidate by straight-line distance to the pickup is offered first."""
    return -candidates["distance_km"].to_numpy(dtype=float)


def weighted_rule(candidates: pd.DataFrame) -> np.ndarray:
    """Secondary baseline: an operations heuristic using the same kinds of signals as the ML model."""
    weights = WEIGHTED_RULE_WEIGHTS
    rating = candidates["rating"].astype(float).fillna(RATING_PRIOR).to_numpy()
    return (
        weights["estimated_eta_min"] * candidates["estimated_eta_min"].to_numpy(dtype=float)
        + weights["acceptance_rate"] * candidates["acceptance_rate"].to_numpy(dtype=float)
        + weights["cancellation_rate"] * candidates["cancellation_rate"].to_numpy(dtype=float)
        + weights["rating"] * (rating - RATING_PRIOR)
        + weights["idle_time_min"] * np.minimum(candidates["idle_time_min"].to_numpy(dtype=float), IDLE_CAP_MIN)
    )


def random_order(seed: int) -> Policy:
    """Sanity floor: a random offer order, reproducible for a given seed and candidate frame."""

    def policy(candidates: pd.DataFrame) -> np.ndarray:
        return np.random.default_rng(seed).random(len(candidates))

    return policy


def model_scores(model, candidates: pd.DataFrame) -> np.ndarray:
    """Option A: P(accept) of a trained model, computed through the feature code that training used."""
    return model.predict_proba(build_features(candidates))[:, 1]


def model_policy(model, max_extra_eta_min: float = float("inf")) -> Policy:
    """A ready-to-use policy for the replay and, later, for serving."""

    def policy(candidates: pd.DataFrame) -> np.ndarray:
        return apply_eta_constraint(model_scores(model, candidates), candidates, max_extra_eta_min)

    return policy
