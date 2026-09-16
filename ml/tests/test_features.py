"""Tests for the feature code shared by training and serving (ml/features).

The point of most of these is not that a number is right, but that the same code produces the same feature
for 100,000 training rows and for the handful of candidates of one live booking.
"""

import numpy as np
import pandas as pd
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate
from ml.features import (
    CATEGORICAL_LEVELS,
    FEATURE_COLUMNS,
    FORBIDDEN_COLUMNS,
    VIEW_COLUMNS,
    build_decision_time_view,
    build_features,
)
from ml.features.build import PICKUP_RATIO_OFFSET_KM


def toy_view() -> pd.DataFrame:
    """Two bookings; booking 1 has two candidates, booking 2 has one. Coordinates are around (0, 0)."""
    return pd.DataFrame({
        "booking_id": [1, 1, 2],
        "driver_id": [10, 11, 20],
        "driver_lat": [0.01, 0.02, 0.0],
        "driver_lon": [0.0, 0.0, 0.03],
        "distance_km": [1.0, 3.5, 2.0],
        "estimated_eta_min": [4.0, 11.0, 7.0],
        "idle_time_min": [5.0, 0.0, 40.0],
        "request_time": pd.to_datetime(["2026-06-01 08:00", "2026-06-01 08:00", "2026-06-06 23:30"]),
        "pickup_lat": [0.0, 0.0, 0.0],
        "pickup_lon": [0.0, 0.0, 0.0],
        "destination_lat": [0.05, 0.05, 0.0],
        "destination_lon": [0.0, 0.0, 0.1],
        "passenger_type": ["individual", "individual", "group"],
        "traffic_level": ["low", "low", "high"],
        "weather": ["clear", "clear", "rain"],
        "time_of_day": ["morning_peak", "morning_peak", "evening"],
        "vehicle_type": ["car_4", "car_4", "car_7"],
        "rating": [4.8, np.nan, 4.5],
        "acceptance_rate": [0.7, 0.4, 0.55],
        "cancellation_rate": [0.05, 0.2, 0.1],
        "completed_trips": [1200, 3, 400],
    })


@pytest.fixture(scope="module")
def generated():
    ds = generate(GeneratorConfig(seed=17, n_drivers=1_000, n_bookings=4_000))
    view = build_decision_time_view(ds.bookings, ds.candidates, ds.drivers)
    return ds, view


def test_feature_columns_are_unique_and_never_a_forbidden_column():
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS).isdisjoint(FORBIDDEN_COLUMNS)


def test_view_carries_only_decision_time_columns(generated):
    _, view = generated
    assert list(view.columns) == VIEW_COLUMNS
    assert set(view.columns).isdisjoint(FORBIDDEN_COLUMNS)


def test_view_rejects_candidates_without_booking_context(generated):
    ds, _ = generated
    with pytest.raises(ValueError, match="no booking context"):
        build_decision_time_view(ds.bookings.iloc[:10], ds.candidates, ds.drivers)


def test_derived_features_match_hand_computed_values():
    features = build_features(toy_view())
    # 0.05 degrees of latitude on the sphere used by ml/data/geo.py
    assert features["trip_km"].iloc[0] == pytest.approx(5.5597, abs=1e-3)
    assert features["pickup_to_trip_ratio"].iloc[0] == pytest.approx(
        1.0 / (features["trip_km"].iloc[0] + PICKUP_RATIO_OFFSET_KM), rel=1e-9
    )
    assert features["hour"].tolist() == [8.0, 8.0, 23.0]
    assert features["is_weekend"].tolist() == [0.0, 0.0, 1.0]  # 2026-06-06 is a Saturday
    assert features["has_rating"].tolist() == [1.0, 0.0, 1.0]
    assert features["distance_minus_min_km"].tolist() == [0.0, 2.5, 0.0]


def test_missing_rating_stays_nan_for_the_tree_model():
    features = build_features(toy_view())
    assert np.isnan(features["rating"].iloc[1])
    assert features["rating"].iloc[0] == 4.8


def test_one_hot_columns_do_not_depend_on_the_levels_present_in_the_data():
    """A single booking sees one weather and one traffic level; the matrix must still have every column."""
    one_booking = toy_view().iloc[:2]
    features = build_features(one_booking)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features["weather=clear"].tolist() == [1.0, 1.0]
    assert features["weather=heavy_rain"].tolist() == [0.0, 0.0]


def test_unknown_category_level_is_rejected():
    view = toy_view()
    view.loc[0, "weather"] = "snow"
    with pytest.raises(ValueError, match="unknown levels"):
        build_features(view)


def test_missing_input_column_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        build_features(toy_view().drop(columns="idle_time_min"))


def test_labels_and_log_artifacts_are_ignored_even_when_present():
    """Serving passes a frame without labels; training data has them. Both must give the same features."""
    view = toy_view()
    polluted = view.assign(accepted=[True, False, True], offer_rank=[1, 2, 1], logging_policy="nearest",
                           cancelled=[False, None, True], p_accept=[0.9, 0.1, 0.5])
    pd.testing.assert_frame_equal(build_features(view), build_features(polluted))


def test_scoring_one_booking_gives_the_same_features_as_the_full_batch(generated):
    """The training-serving skew check (docs/architecture.md §4.1): online, features are built per booking."""
    _, view = generated
    batch = build_features(view)
    sample = view["booking_id"].drop_duplicates().head(50)
    for booking_id in sample:
        rows = view[view["booking_id"] == booking_id]
        pd.testing.assert_frame_equal(build_features(rows), batch.loc[rows.index])


def test_features_are_deterministic(generated):
    _, view = generated
    pd.testing.assert_frame_equal(build_features(view), build_features(view))


def test_every_categorical_level_of_the_generator_is_known(generated):
    _, view = generated
    for column, levels in CATEGORICAL_LEVELS.items():
        assert set(view[column].astype("string").dropna().unique()) <= set(levels)
