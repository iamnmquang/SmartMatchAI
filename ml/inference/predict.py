"""Score and rank the candidate drivers of one booking (Phase 7).

    from ml.inference.predict import rank_drivers

    rank_drivers({"booking": {...}, "candidate_drivers": [{...}, ...]}, top_k=5)

The request is plain JSON-shaped data, not a DataFrame: this is what an API handler has after validating a
request body (Phase 9), and what a script or a notebook can type by hand. Everything after that is the same
code the offline pipeline used - ml/features builds the matrix, ml/inference/ranking applies the ETA
constraint - so a recommendation here and a replayed dispatch in ml/evaluation put the same driver first.

Out of scope on purpose:

* **Candidate generation.** Which drivers are available, close enough and of the right vehicle type is the
  matching module's job (Phase 9). This function ranks exactly the candidates it is given.
* **ETA estimation.** `estimated_eta_min` is an input; a real system gets it from a routing service.
  `distance_km` and `time_of_day` are derived when missing, because both are pure functions of fields that
  are already required, and recomputing them elsewhere would be a second implementation to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.generate import TIME_OF_DAY_BY_HOUR
from ml.data.geo import haversine_km
from ml.features import VIEW_COLUMNS, build_features
from ml.inference.artifact import LinearAcceptanceModel, read_artifact
from ml.inference.ranking import apply_eta_constraint, offer_order

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "ml" / "models"
DEFAULT_TOP_K = 5

BOOKING_FIELDS = [
    "booking_id", "request_time", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon",
    "passenger_type", "traffic_level", "weather",
]
DRIVER_FIELDS = [
    "driver_id", "estimated_eta_min", "idle_time_min", "vehicle_type",
    "acceptance_rate", "cancellation_rate", "completed_trips",
]
OPTIONAL_DRIVER_FIELDS = ["rating", "distance_km", "driver_lat", "driver_lon"]

# Only contributions above this many log-odds are worth showing; below it the wording would suggest more
# certainty than the number carries.
REASON_MIN_CONTRIBUTION = 0.05
MAX_REASONS = 3
REASON_TEMPLATES = {
    "estimated_eta_min": "pickup ETA {:.1f} min",
    "distance_km": "{:.1f} km from the pickup",
    "distance_minus_min_km": "{:.1f} km further than the closest candidate",
    "pickup_to_trip_ratio": "pickup is {:.0%} of the trip length",
    "trip_km": "trip length {:.1f} km",
    "idle_time_min": "idle for {:.0f} min",
    "acceptance_rate": "accepts {:.0%} of offers historically",
    "cancellation_rate": "cancels {:.0%} of trips historically",
    "completed_trips": "{:.0f} completed trips",
    "rating": "rating {:.2f}",
    "hour": "hour of day {:.0f}",
}
ONE_HOT_LABELS = {
    "traffic_level": "traffic is {}",
    "weather": "weather is {}",
    "time_of_day": "time of day is {}",
    "passenger_type": "{} booking",
    "vehicle_type": "vehicle type {}",
}
FLAG_LABELS = {"has_rating": "driver has a rating", "is_weekend": "weekend booking"}


@dataclass(frozen=True)
class AcceptanceRanker:
    """The served model plus the policy it is served with."""

    model: LinearAcceptanceModel
    metadata: dict

    @classmethod
    def load(cls, directory: Path = MODEL_DIR) -> AcceptanceRanker:
        model, metadata = read_artifact(directory)
        return cls(model=model, metadata=metadata)

    @property
    def model_version(self) -> str:
        return self.metadata["model_version"]

    @property
    def max_extra_eta_min(self) -> float:
        margin = self.metadata["policy"]["max_extra_eta_min"]
        return float("inf") if margin is None else float(margin)

    def rank(self, booking: dict, candidate_drivers: list[dict], top_k: int = DEFAULT_TOP_K) -> dict:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        view = build_view(booking, candidate_drivers)
        features = build_features(view)

        probabilities = self.model.predict_proba(features)
        ranking_scores = apply_eta_constraint(probabilities, view, self.max_extra_eta_min)
        order = offer_order(ranking_scores, view["driver_id"].to_numpy())[:top_k]

        contributions = self.model.contributions(features.iloc[order])
        eta = view["estimated_eta_min"].to_numpy(dtype=float)
        fastest = float(eta.min())
        return {
            "booking_id": _native(booking["booking_id"]),
            "model_version": self.model_version,
            "policy": {"max_extra_eta_min": self.metadata["policy"]["max_extra_eta_min"]},
            "n_candidates": len(view),
            "drivers": [
                {
                    "driver_id": _native(view["driver_id"].iloc[position]),
                    "rank": rank,
                    "score": round(float(probabilities[position]), 6),
                    "eta": round(float(eta[position]), 2),
                    "distance_km": round(float(view["distance_km"].iloc[position]), 3),
                    "within_eta_margin": bool(eta[position] <= fastest + self.max_extra_eta_min),
                    "reasons": explain(features.iloc[position], contributions.iloc[index]),
                }
                for rank, (index, position) in enumerate(enumerate(order), start=1)
            ],
        }


@lru_cache(maxsize=4)
def _cached_ranker(directory: str) -> AcceptanceRanker:
    return AcceptanceRanker.load(Path(directory))


def get_ranker(directory: Path = MODEL_DIR) -> AcceptanceRanker:
    """The shared ranker. Loading is cached: the artifact is read once per process, not once per request."""
    return _cached_ranker(str(directory))


def rank_drivers(request: dict, top_k: int = DEFAULT_TOP_K, directory: Path = MODEL_DIR) -> dict:
    """Entry point on the request shape of the project brief: {"booking": {...}, "candidate_drivers": [...]}."""
    missing = [key for key in ("booking", "candidate_drivers") if key not in request]
    if missing:
        raise ValueError(f"request is missing keys: {missing}")
    return get_ranker(directory).rank(request["booking"], request["candidate_drivers"], top_k=top_k)


def build_view(booking: dict, candidate_drivers: list[dict]) -> pd.DataFrame:
    """Plain dicts -> the decision-time view that ml/features expects, with the derivable fields filled in."""
    if not candidate_drivers:
        raise ValueError("candidate_drivers is empty; candidate generation happens before ranking")
    _require(booking, BOOKING_FIELDS, "booking")
    for index, driver in enumerate(candidate_drivers):
        _require(driver, DRIVER_FIELDS, f"candidate_drivers[{index}]")
        if "distance_km" not in driver and not {"driver_lat", "driver_lon"} <= driver.keys():
            raise ValueError(f"candidate_drivers[{index}] needs distance_km, or driver_lat and driver_lon")

    request_time = pd.Timestamp(booking["request_time"])
    view = pd.DataFrame(
        [{key: driver.get(key) for key in DRIVER_FIELDS + OPTIONAL_DRIVER_FIELDS} for driver in candidate_drivers]
    )
    for key in BOOKING_FIELDS:
        view[key] = booking[key]
    view["request_time"] = request_time
    view["time_of_day"] = booking.get("time_of_day") or TIME_OF_DAY_BY_HOUR[request_time.hour]
    view["rating"] = view["rating"].astype(float)  # missing stays NaN: the model reads it as "no rating yet"

    derived = haversine_km(view["driver_lat"], view["driver_lon"], view["pickup_lat"], view["pickup_lon"])
    view["distance_km"] = view["distance_km"].astype(float).fillna(pd.Series(derived, index=view.index))
    if view["distance_km"].isna().any():
        raise ValueError("distance_km could not be derived for every candidate")
    return view[VIEW_COLUMNS]


def explain(feature_values: pd.Series, contributions: pd.Series) -> list[dict]:
    """The features that moved this candidate's score the most, as text (ADR-007).

    A contribution is `coefficient x standardised value`, in log-odds: how far this candidate sits from an
    average one because of that feature. Deterministic, cheap, and it cannot invent a fact - unlike a reason
    written by an LLM. Both arguments are indexed by feature name, so the two orders cannot drift apart.
    """
    reasons = []
    for column in contributions.abs().sort_values(ascending=False, kind="stable").index[:MAX_REASONS]:
        contribution = float(contributions[column])
        if abs(contribution) < REASON_MIN_CONTRIBUTION:
            break
        reasons.append({
            "feature": column,
            "text": _phrase(column, float(feature_values[column])),
            "effect": "increases" if contribution > 0 else "decreases",
            "contribution": round(contribution, 4),
        })
    return reasons


def _phrase(column: str, value: float) -> str:
    if column in REASON_TEMPLATES:
        return REASON_TEMPLATES[column].format(value)
    if column in FLAG_LABELS:
        return FLAG_LABELS[column] if value else f"not: {FLAG_LABELS[column]}"
    if "=" in column:
        name, level = column.split("=", 1)
        level = level.replace("_", " ")
        return ONE_HOT_LABELS.get(name, name + " is {}").format(level) if value else f"not: {level}"
    return column


def _require(payload: dict, fields: list[str], where: str) -> None:
    missing = [field for field in fields if payload.get(field) is None]
    if missing:
        raise ValueError(f"{where} is missing fields: {missing}")


def _native(value):
    return value.item() if hasattr(value, "item") else value
