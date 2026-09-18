"""The ranking rule that serving applies, and offline evaluation measures (ADR-015).

It lives here rather than in ml/evaluation so that the online path owns the policy and the offline path
imports it - the same reason ml/features is shared: two implementations of one rule would drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_eta_constraint(scores: np.ndarray, candidates: pd.DataFrame, max_extra_eta_min: float) -> np.ndarray:
    """Keep the order given by `scores`, but offer first only the candidates within `max_extra_eta_min`
    minutes of the fastest candidate of the same booking.

    Ineligible candidates are not dropped - dropping them would lower the matching success rate of bookings
    where every eligible driver refuses. They are pushed behind every eligible one by a constant larger than
    the whole score range, so the relative order inside each of the two groups is untouched.
    A margin of infinity therefore reproduces plain P(accept) ranking (PRD Q2, option A).
    """
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(max_extra_eta_min):
        return scores
    eta = candidates["estimated_eta_min"].astype(float)
    fastest = eta.groupby(candidates["booking_id"]).transform("min")
    eligible = (eta <= fastest + max_extra_eta_min).to_numpy(dtype=float)
    return scores + eligible * (float(scores.max() - scores.min()) + 1.0)


def offer_order(scores: np.ndarray, driver_id: np.ndarray) -> np.ndarray:
    """Indices of one booking's candidates in offer order: best score first, ties by driver_id.

    The same rule the offline replay uses (ml/evaluation/offline.simulate_offers), so a recommendation list
    and a replayed dispatch put the same driver first.
    """
    return np.lexsort((np.asarray(driver_id), -np.asarray(scores, dtype=float)))
