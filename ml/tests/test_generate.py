import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import (
    CANDIDATE_COLUMNS,
    HIDDEN_TABLES,
    OBSERVABLE_TABLES,
    generate,
    roc_auc,
    write_dataset,
)
from ml.data.geo import haversine_km

SMALL = GeneratorConfig(seed=7, n_drivers=1_000, n_bookings=5_000)
TINY = replace(SMALL, n_drivers=300, n_bookings=800)


@pytest.fixture(scope="module")
def ds():
    return generate(SMALL)


def test_observable_tables_have_expected_columns(ds):
    assert list(ds.drivers.columns) == [
        "driver_id", "vehicle_type", "rating", "acceptance_rate", "cancellation_rate",
        "completed_trips", "idle_time_min", "current_lat", "current_lon",
    ]
    assert list(ds.bookings.columns) == [
        "booking_id", "request_time", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon",
        "passenger_type", "traffic_level", "weather", "time_of_day",
    ]
    assert list(ds.candidates.columns) == CANDIDATE_COLUMNS


def test_hidden_truth_never_leaks_into_observable_tables(ds):
    hidden = set(ds.candidate_truth.columns) | set(ds.driver_latents.columns)
    hidden -= {"booking_id", "driver_id"}
    for name in OBSERVABLE_TABLES:
        assert hidden.isdisjoint(getattr(ds, name).columns), name


def test_value_ranges(ds):
    drivers, bookings, candidates = ds.drivers, ds.bookings, ds.candidates
    assert drivers["rating"].dropna().between(1.0, 5.0).all()
    assert drivers["acceptance_rate"].between(0.0, 1.0).all()
    assert drivers["cancellation_rate"].between(0.0, 1.0).all()
    assert (drivers["completed_trips"] >= 0).all()
    # Missing rating is expected only for drivers without enough trips.
    assert (drivers.loc[drivers["rating"].isna(), "completed_trips"] < 5).all()

    start = pd.Timestamp(SMALL.start_date)
    assert bookings["request_time"].between(start, start + pd.Timedelta(days=SMALL.n_days), inclusive="left").all()
    assert bookings["request_time"].is_monotonic_increasing

    assert candidates["distance_km"].between(0.0, SMALL.search_radius_km).all()
    assert (candidates["estimated_eta_min"] > 0).all()
    assert (candidates["idle_time_min"] >= 0).all()


def test_referential_integrity(ds):
    candidates = ds.candidates
    assert ds.drivers["driver_id"].is_unique
    assert ds.bookings["booking_id"].is_unique
    assert candidates["booking_id"].isin(ds.bookings["booking_id"]).all()
    assert candidates["driver_id"].isin(ds.drivers["driver_id"]).all()
    assert not candidates.duplicated(["booking_id", "driver_id"]).any()
    assert ds.candidate_truth[["booking_id", "driver_id"]].equals(candidates[["booking_id", "driver_id"]])


def test_group_bookings_only_get_7_seat_cars(ds):
    merged = ds.candidates.merge(ds.bookings[["booking_id", "passenger_type"]], on="booking_id").merge(
        ds.drivers[["driver_id", "vehicle_type"]], on="driver_id"
    )
    assert (merged.loc[merged["passenger_type"] == "group", "vehicle_type"] == "car_7").all()


def test_distance_matches_coordinates(ds):
    merged = ds.candidates.merge(ds.bookings[["booking_id", "pickup_lat", "pickup_lon"]], on="booking_id")
    recomputed = haversine_km(merged["pickup_lat"], merged["pickup_lon"], merged["driver_lat"], merged["driver_lon"])
    assert np.allclose(recomputed, merged["distance_km"], atol=1e-3)


def test_logged_offers_follow_sequential_protocol(ds):
    candidates = ds.candidates
    offers = candidates[candidates["offer_rank"].notna()]
    ranks = offers.groupby("booking_id")["offer_rank"].agg(["min", "max", "count"])
    assert (ranks["min"] == 1).all()
    assert (ranks["count"] == ranks["max"]).all()
    assert (ranks["max"] <= SMALL.max_offers).all()

    accepted = offers[offers["accepted"].to_numpy(dtype=bool)]
    assert accepted["booking_id"].is_unique, "at most one acceptance per booking"
    assert (accepted["offer_rank"].to_numpy() == ranks.loc[accepted["booking_id"], "max"].to_numpy()).all()

    # Without an acceptance, offers continue until candidates or max_offers run out.
    no_accept = ranks.drop(index=accepted["booking_id"])
    n_candidates = candidates.groupby("booking_id").size().loc[no_accept.index]
    assert (no_accept["max"] == np.minimum(n_candidates, SMALL.max_offers)).all()


def test_nearest_logging_policy_offers_closest_driver_first(ds):
    nearest = ds.candidates[ds.candidates["logging_policy"] == "nearest"]
    first = nearest[nearest["offer_rank"] == 1].set_index("booking_id")["distance_km"]
    closest = nearest.groupby("booking_id")["distance_km"].min().loc[first.index]
    assert np.allclose(first, closest)


def test_labels_exist_only_where_they_would_in_a_real_log(ds):
    candidates = ds.candidates
    offered = candidates["offer_rank"].notna()
    assert candidates.loc[~offered, "accepted"].isna().all()
    assert candidates.loc[offered, "accepted"].notna().all()
    accepted = candidates["accepted"].fillna(False).to_numpy(dtype=bool)
    assert (candidates["cancelled"].notna().to_numpy() == accepted).all()


def test_labels_are_consistent_with_oracle(ds):
    merged = pd.concat([ds.candidates, ds.candidate_truth[["p_accept", "u_accept", "p_cancel", "u_cancel"]]], axis=1)
    offers = merged[merged["offer_rank"].notna()]
    assert (offers["accepted"].to_numpy(dtype=bool) == (offers["u_accept"] < offers["p_accept"])).all()
    accepted = offers[offers["accepted"].to_numpy(dtype=bool)]
    assert (accepted["cancelled"].to_numpy(dtype=bool) == (accepted["u_cancel"] < accepted["p_cancel"])).all()


def test_data_is_not_trivially_separable(ds):
    auc = ds.oracle_summary["oracle_auc_accept_logged_offers"]
    assert 0.65 < auc < 0.93, auc


def test_same_seed_reproduces_dataset_and_other_seed_changes_it():
    first, second = generate(TINY), generate(TINY)
    for name in OBSERVABLE_TABLES + HIDDEN_TABLES:
        pd.testing.assert_frame_equal(getattr(first, name), getattr(second, name))
    assert not first.candidates.equals(generate(replace(TINY, seed=TINY.seed + 1)).candidates)


def test_write_dataset_roundtrip(tmp_path):
    ds = generate(TINY)
    write_dataset(ds, tmp_path / "raw", tmp_path / "oracle")
    for name in OBSERVABLE_TABLES:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "raw" / f"{name}.parquet"), getattr(ds, name))
    for name in HIDDEN_TABLES:
        assert (tmp_path / "oracle" / f"{name}.parquet").exists()
        assert not (tmp_path / "raw" / f"{name}.parquet").exists()
    metadata = json.loads((tmp_path / "raw" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"]["seed"] == TINY.seed
    assert "behavior_params" not in metadata


def test_roc_auc_known_values():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([0, 1], [0.2, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        GeneratorConfig(n_bookings=0)
    with pytest.raises(ValueError):
        GeneratorConfig(exploration_rate=1.5)
