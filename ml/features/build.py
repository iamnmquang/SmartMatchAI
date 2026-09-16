"""Feature engineering shared by training and serving (docs/architecture.md §4.1).

One function turns a decision-time view (ml/features/view.py) into the model input matrix. Training and online
inference must call this same function: if the backend recomputed features with different code, the model would
silently receive inputs it was never trained on.

Design rules that keep it usable online:

* Pure and stateless. Nothing is fitted here, so there is no encoder state to ship or to leak from validation
  into training. Model-specific preprocessing that *must* be fitted (impute, scale for the linear baseline)
  lives in the scikit-learn pipeline of ml/training and is fitted on the train split only.
* Categorical levels are fixed constants, not inferred from the data, so column order and meaning are identical
  for 780,000 training rows and for the 12 candidates of a single live booking. An unknown level is an error,
  not a silently different encoding.
* Only columns of the decision-time view are read. The label, the log artifacts and data/oracle/ are unreachable
  from here.

Feature choice follows docs/research.md §9.8: every input named in the project brief, plus the derived features
that EDA found informative. Of the two relative features, distance_minus_min_km is kept and
distance_rank_in_booking dropped: they are nearly the same signal (Spearman 0.97, research §9.7) and the
kilometre gap keeps its meaning when a booking has an unusual number of candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.geo import haversine_km

PICKUP_RATIO_OFFSET_KM = 0.5  # same definition as the EDA notebook; avoids dividing by a near-zero trip distance

# Value at request time (candidates.idle_time_min), not the current snapshot in drivers.
NUMERIC_FEATURES = [
    "estimated_eta_min",
    "distance_km",
    "idle_time_min",
    "trip_km",
    "pickup_to_trip_ratio",
    "distance_minus_min_km",
    "rating",  # NaN is kept: XGBoost handles it natively, the linear pipeline imputes it
    "acceptance_rate",
    "cancellation_rate",
    "completed_trips",
    "hour",
]
BINARY_FEATURES = ["has_rating", "is_weekend"]
CATEGORICAL_LEVELS = {
    "traffic_level": ["low", "medium", "high"],
    "weather": ["clear", "rain", "heavy_rain"],
    "time_of_day": ["night", "morning_peak", "midday", "evening_peak", "evening"],
    "passenger_type": ["individual", "group"],
    "vehicle_type": ["car_4", "car_7"],
}
# All levels are kept (no drop-first) so that the encoding of a level never depends on which level is missing.
ONE_HOT_FEATURES = [f"{column}={level}" for column, levels in CATEGORICAL_LEVELS.items() for level in levels]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + ONE_HOT_FEATURES

# Features where a zero is a real value and log1p is safe; used by the linear pipeline only.
LOG1P_FEATURES = ["completed_trips", "idle_time_min"]


def build_features(view: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix for a decision-time view, one row per candidate, columns in FEATURE_COLUMNS order."""
    _check_input(view)
    distance_km = view["distance_km"].astype(float)
    trip_km = pd.Series(
        haversine_km(view["pickup_lat"], view["pickup_lon"], view["destination_lat"], view["destination_lon"]),
        index=view.index,
    )
    request_time = view["request_time"]

    features = pd.DataFrame(index=view.index)
    features["estimated_eta_min"] = view["estimated_eta_min"].astype(float)
    features["distance_km"] = distance_km
    features["idle_time_min"] = view["idle_time_min"].astype(float)
    features["trip_km"] = trip_km
    features["pickup_to_trip_ratio"] = distance_km / (trip_km + PICKUP_RATIO_OFFSET_KM)
    # Relative to the other candidates of the same booking: how much further than the closest one.
    features["distance_minus_min_km"] = distance_km - distance_km.groupby(view["booking_id"]).transform("min")
    features["rating"] = view["rating"].astype(float)
    features["acceptance_rate"] = view["acceptance_rate"].astype(float)
    features["cancellation_rate"] = view["cancellation_rate"].astype(float)
    features["completed_trips"] = view["completed_trips"].astype(float)
    features["hour"] = request_time.dt.hour.astype(float)

    features["has_rating"] = view["rating"].notna().astype(float)
    features["is_weekend"] = (request_time.dt.dayofweek >= 5).astype(float)

    for column, levels in CATEGORICAL_LEVELS.items():
        values = view[column].astype("string")
        unknown = set(values.dropna().unique()) - set(levels)
        if unknown:
            raise ValueError(f"unknown levels {sorted(unknown)} in column {column!r}; expected {levels}")
        for level in levels:
            features[f"{column}={level}"] = (values == level).astype(float)

    return features[FEATURE_COLUMNS]


def _check_input(view: pd.DataFrame) -> None:
    required = [
        "booking_id", "distance_km", "estimated_eta_min", "idle_time_min", "pickup_lat", "pickup_lon",
        "destination_lat", "destination_lon", "request_time", "rating", "acceptance_rate", "cancellation_rate",
        "completed_trips", *CATEGORICAL_LEVELS,
    ]
    missing = [column for column in required if column not in view.columns]
    if missing:
        raise ValueError(f"decision-time view is missing columns: {missing}")
    if not np.issubdtype(view["request_time"].dtype, np.datetime64):
        raise ValueError("request_time must be datetime64")
