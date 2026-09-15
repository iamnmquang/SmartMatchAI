import numpy as np
import pandas as pd
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate
from ml.evaluation.offline import (
    EvaluationData,
    build_evaluation_data,
    business_metrics,
    oracle_scores,
    paired_bootstrap,
    simulate_offers,
)
from ml.evaluation.policies import nearest_driver, random_order, weighted_rule


def toy_data() -> EvaluationData:
    """Booking 1: the 3rd-nearest driver is the first to accept (then cancels); the 4th would accept too.
    Booking 2: nobody accepts. Booking 3: no candidates at all."""
    candidates = pd.DataFrame({
        "booking_id": [1, 1, 1, 1, 2, 2],
        "driver_id": [10, 11, 12, 13, 20, 21],
        "distance_km": [1.0, 2.0, 3.0, 4.0, 0.5, 1.5],
        "estimated_eta_min": [4.0, 6.0, 8.0, 10.0, 3.0, 5.0],
    })
    truth = pd.DataFrame({
        "booking_id": candidates["booking_id"],
        "driver_id": candidates["driver_id"],
        "p_accept": [0.2, 0.3, 0.9, 0.95, 0.1, 0.1],
        "u_accept": [0.5] * 6,
        "p_cancel": [0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
        "u_cancel": [0.5] * 6,
    })
    return EvaluationData(bookings=pd.DataFrame({"booking_id": [1, 2, 3]}), candidates=candidates, truth=truth)


def test_nearest_driver_replay_matches_hand_computed_outcomes():
    data = toy_data()
    outcomes = simulate_offers(data, nearest_driver(data.candidates), max_offers=3).set_index("booking_id")

    assert outcomes.loc[1, "matched"] and outcomes.loc[1, "n_offers"] == 3
    assert outcomes.loc[1, "matched_driver_id"] == 12 and outcomes.loc[1, "cancelled"]
    assert not outcomes.loc[2, "matched"] and outcomes.loc[2, "n_offers"] == 2
    assert outcomes.loc[3, "n_candidates"] == 0 and outcomes.loc[3, "n_offers"] == 0 and not outcomes.loc[3, "matched"]

    metrics = business_metrics(outcomes.reset_index())
    assert metrics["matching_success_rate"] == pytest.approx(1 / 3)  # booking without candidates counts as a failure
    assert metrics["acceptance_rate"] == pytest.approx(1 / 5)
    assert metrics["avg_eta_matched_min"] == pytest.approx(8.0)
    assert metrics["cancellation_rate"] == pytest.approx(1.0)
    assert metrics["completed_rate"] == pytest.approx(0.0)
    assert metrics["first_offer_acceptance_rate"] == pytest.approx(0.0)
    assert metrics["avg_offers_per_booking_with_candidates"] == pytest.approx(2.5)


def test_oracle_orders_by_true_probability_and_breaks_ties_by_driver_id():
    data = toy_data()
    outcomes = simulate_offers(data, oracle_scores(data.truth), max_offers=3).set_index("booking_id")
    assert outcomes.loc[1, "matched_driver_id"] == 13 and outcomes.loc[1, "n_offers"] == 1
    assert not outcomes.loc[1, "cancelled"]
    metrics = business_metrics(outcomes.reset_index())
    assert metrics["first_offer_acceptance_rate"] == pytest.approx(0.5)
    assert metrics["avg_eta_matched_min"] == pytest.approx(10.0)


def test_offers_stop_at_max_offers():
    data = toy_data()
    outcomes = simulate_offers(data, nearest_driver(data.candidates), max_offers=2).set_index("booking_id")
    assert not outcomes.loc[1, "matched"] and outcomes.loc[1, "n_offers"] == 2


def test_invalid_scores_are_rejected():
    data = toy_data()
    with pytest.raises(ValueError):
        simulate_offers(data, np.zeros(len(data.candidates) - 1))
    with pytest.raises(ValueError):
        simulate_offers(data, np.full(len(data.candidates), np.nan))


def test_bootstrap_of_identical_policies_has_zero_difference_and_covers_point_estimate():
    data = toy_data()
    same = simulate_offers(data, nearest_driver(data.candidates))
    intervals = paired_bootstrap({"a": same, "b": same.copy()}, reference="a", n_resamples=200, seed=1)
    assert intervals["b"]["matching_success_rate"]["diff_vs_reference_ci95"] == [0.0, 0.0]
    low, high = intervals["a"]["matching_success_rate"]["ci95"]
    assert low <= business_metrics(same)["matching_success_rate"] <= high


def test_weighted_rule_prefers_closer_and_more_reliable_drivers_and_handles_missing_rating():
    candidates = pd.DataFrame({
        "estimated_eta_min": [5.0, 5.0, 10.0],
        "acceptance_rate": [0.8, 0.4, 0.8],
        "cancellation_rate": [0.05, 0.05, 0.05],
        "rating": [np.nan, 4.7, 4.7],
        "idle_time_min": [10.0, 10.0, 10.0],
    })
    scores = weighted_rule(candidates)
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1] and scores[0] > scores[2]


def test_random_order_is_reproducible():
    data = toy_data()
    assert np.array_equal(random_order(3)(data.candidates), random_order(3)(data.candidates))


@pytest.fixture(scope="module")
def generated_test_split():
    ds = generate(GeneratorConfig(seed=11, n_drivers=1_000, n_bookings=5_000))
    data = build_evaluation_data(ds.bookings, ds.candidates, ds.drivers, ds.candidate_truth, split="test")
    return ds, data


def test_decision_time_view_excludes_labels_log_artifacts_and_truth(generated_test_split):
    _, data = generated_test_split
    forbidden = {"offer_rank", "logging_policy", "accepted", "cancelled", "p_accept", "u_accept", "p_cancel", "u_cancel"}
    assert forbidden.isdisjoint(data.candidates.columns)


def test_replaying_nearest_policy_reproduces_the_logged_offers(generated_test_split):
    """The logging policy 'nearest' and the replayed baseline must agree exactly: same world, same order."""
    ds, data = generated_test_split
    outcomes = simulate_offers(data, nearest_driver(data.candidates), max_offers=ds.config.max_offers)
    outcomes = outcomes.set_index("booking_id")

    logged = ds.candidates[
        ds.candidates["booking_id"].isin(data.bookings["booking_id"]) & (ds.candidates["logging_policy"] == "nearest")
    ]
    offers = logged[logged["offer_rank"].notna()]
    logged_offers = offers.groupby("booking_id").size()
    logged_matches = offers[offers["accepted"].to_numpy(dtype=bool)].set_index("booking_id")

    assert len(logged_offers) > 100
    assert (outcomes.loc[logged_offers.index, "n_offers"] == logged_offers).all()
    assert set(outcomes.index[outcomes["matched"]]) & set(logged_offers.index) == set(logged_matches.index)
    assert (outcomes.loc[logged_matches.index, "matched_driver_id"].astype(int) == logged_matches["driver_id"]).all()
    assert (outcomes.loc[logged_matches.index, "cancelled"].astype(bool) == logged_matches["cancelled"].astype(bool)).all()


def test_misaligned_truth_is_rejected(generated_test_split):
    ds, _ = generated_test_split
    shuffled = ds.candidate_truth.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError):
        build_evaluation_data(ds.bookings, ds.candidates, ds.drivers, shuffled, split="test")
