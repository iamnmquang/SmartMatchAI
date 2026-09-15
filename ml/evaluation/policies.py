"""Ranking policies for offline evaluation.

A policy maps a frame of candidates (decision-time columns only) to one score per row. Within a booking, higher scores
are offered first and ties are broken by driver_id. Policies never see the simulator's hidden truth; the oracle upper
bound lives in ml/evaluation/offline.py because it is the one exception.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Policy = Callable[[pd.DataFrame], np.ndarray]

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
