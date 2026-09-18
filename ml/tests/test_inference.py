"""Tests for Phase 7 serving: the JSON artifact, the ranking entry point and the reasons it produces."""

import json
from pathlib import Path

import numpy as np
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate
from ml.evaluation.offline import build_evaluation_data
from ml.features import FEATURE_COLUMNS, build_features
from ml.inference.artifact import LinearAcceptanceModel, read_artifact, write_artifact
from ml.inference.benchmark import to_request
from ml.inference.predict import AcceptanceRanker, build_view, get_ranker, rank_drivers
from ml.inference.ranking import apply_eta_constraint, offer_order
from ml.training.dataset import RawData, labeled_offers
from ml.training.models import linear_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "ml" / "models"
EVALUATION_RESULTS = REPO_ROOT / "ml" / "evaluation" / "results" / "evaluation.json"
MARGIN_MIN = 1.0


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A model trained, exported and loaded exactly like the committed one, but small and self-contained."""
    ds = generate(GeneratorConfig(seed=37, n_drivers=1_000, n_bookings=6_000))
    raw = RawData(bookings=ds.bookings, candidates=ds.candidates, drivers=ds.drivers)
    pipeline = linear_pipeline(1.0, seed=3).fit(*_train_xy(raw))
    model = LinearAcceptanceModel.from_pipeline(pipeline)

    directory = tmp_path_factory.mktemp("models")
    write_artifact(model, {
        "model_version": "test-1", "model_type": "logistic_regression",
        "policy": {"max_extra_eta_min": MARGIN_MIN}, "features": FEATURE_COLUMNS,
    }, directory)
    data = build_evaluation_data(ds.bookings, ds.candidates, ds.drivers, ds.candidate_truth, "test")
    return {"pipeline": pipeline, "model": model, "directory": directory, "data": data,
            "ranker": AcceptanceRanker.load(directory)}


def _train_xy(raw: RawData):
    split = labeled_offers(raw, "train")
    return split.features, split.y


def request_for(data, booking_id: int, count: int | None = None) -> dict:
    rows = data.candidates[data.candidates["booking_id"] == booking_id]
    return to_request(rows, count or len(rows))


def test_json_artifact_reproduces_the_fitted_pipeline(served):
    """The whole point of exporting coefficients instead of pickling: same numbers, no scikit-learn needed."""
    features = build_features(served["data"].candidates)
    from_json = served["model"].predict_proba(features)
    from_pipeline = served["pipeline"].predict_proba(features)[:, 1]
    assert np.allclose(from_json, from_pipeline, rtol=1e-10, atol=1e-12)


def test_artifact_survives_a_json_roundtrip(served, tmp_path):
    write_artifact(served["model"], {"model_version": "x", "policy": {"max_extra_eta_min": None}}, tmp_path)
    reloaded, metadata = read_artifact(tmp_path)
    features = build_features(served["data"].candidates.head(200))
    assert np.array_equal(reloaded.predict_proba(features), served["model"].predict_proba(features))
    assert metadata["policy"]["max_extra_eta_min"] is None


def test_artifact_with_an_unknown_feature_is_rejected(served, tmp_path):
    payload = json.loads((served["directory"] / "acceptance_model.json").read_text(encoding="utf-8"))
    payload["feature_order"] = ["a_feature_the_code_does_not_build"] + payload["feature_order"][1:]
    (tmp_path / "acceptance_model.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="features the current code does not build"):
        read_artifact(tmp_path)


def test_contributions_add_up_to_the_score(served):
    features = build_features(served["data"].candidates.head(500))
    model = served["model"]
    logit = model.contributions(features).to_numpy().sum(axis=1) + model.intercept
    assert np.allclose(1.0 / (1.0 + np.exp(-logit)), model.predict_proba(features), rtol=1e-12)


def test_serving_one_booking_matches_the_offline_ranking(served):
    """The claim behind ADR-013 and ADR-015: one request through the API path and a batch replay agree."""
    data, ranker = served["data"], served["ranker"]
    view = data.candidates
    probabilities = served["model"].predict_proba(build_features(view))
    constrained = apply_eta_constraint(probabilities, view, MARGIN_MIN)

    sizes = view.groupby("booking_id").size()
    for booking_id in sizes[sizes >= 3].index[:25]:
        rows = view[view["booking_id"] == booking_id]
        positions = rows.index.to_numpy()
        offline = rows["driver_id"].to_numpy()[offer_order(constrained[positions], rows["driver_id"].to_numpy())]

        served_order = ranker.rank(**request_for(data, booking_id), top_k=len(rows))
        assert [driver["driver_id"] for driver in served_order["drivers"]] == offline.tolist()
        # The response rounds the score to 6 decimals, so that is the precision the comparison can claim.
        assert served_order["drivers"][0]["score"] == pytest.approx(
            float(probabilities[positions][rows["driver_id"].to_numpy() == offline[0]][0]), abs=5e-7
        )


def test_top_k_returns_the_best_k_in_order(served):
    data, ranker = served["data"], served["ranker"]
    sizes = data.candidates.groupby("booking_id").size()
    booking_id = int(sizes[sizes >= 6].index[0])
    full = ranker.rank(**request_for(data, booking_id), top_k=100)
    top3 = ranker.rank(**request_for(data, booking_id), top_k=3)
    assert len(top3["drivers"]) == 3
    assert [d["driver_id"] for d in top3["drivers"]] == [d["driver_id"] for d in full["drivers"]][:3]
    assert [d["rank"] for d in top3["drivers"]] == [1, 2, 3]
    assert full["n_candidates"] == len(data.candidates[data.candidates["booking_id"] == booking_id])


def test_the_eta_margin_outranks_a_likelier_but_slower_driver(served):
    booking = _booking()
    fast = _driver(1, eta=4.0, distance=1.0, acceptance=0.30)
    slow_but_willing = _driver(2, eta=9.0, distance=3.0, acceptance=0.95)
    ranker = served["ranker"]

    ranked = ranker.rank(booking, [fast, slow_but_willing], top_k=2)["drivers"]
    assert [driver["driver_id"] for driver in ranked] == [1, 2]
    assert ranked[0]["within_eta_margin"] and not ranked[1]["within_eta_margin"]
    # Without the margin the ranking flips: the slower driver really is the likelier one.
    assert ranked[1]["score"] > ranked[0]["score"]


def test_ineligible_candidates_are_ranked_last_not_dropped(served):
    booking = _booking()
    drivers = [_driver(1, eta=4.0, distance=1.0, acceptance=0.3), _driver(2, eta=20.0, distance=8.0, acceptance=0.9)]
    ranked = served["ranker"].rank(booking, drivers, top_k=10)["drivers"]
    assert len(ranked) == 2


def test_distance_and_time_of_day_are_derived_when_missing(served):
    booking = _booking()
    driver = _driver(1, eta=5.0, distance=None, acceptance=0.5)
    driver["driver_lat"], driver["driver_lon"] = 0.01, 0.0
    view = build_view(booking, [driver])
    assert view["distance_km"].iloc[0] == pytest.approx(1.112, abs=1e-3)
    assert view["time_of_day"].iloc[0] == "morning_peak"  # 08:00


@pytest.mark.parametrize("mutate, message", [
    (lambda booking, drivers: booking.pop("weather"), "booking is missing fields"),
    (lambda booking, drivers: drivers[0].pop("acceptance_rate"), "missing fields"),
    (lambda booking, drivers: drivers[0].pop("distance_km"), "needs distance_km"),
    (lambda booking, drivers: drivers.clear(), "candidate_drivers is empty"),
])
def test_incomplete_requests_are_rejected_with_a_clear_message(served, mutate, message):
    booking, drivers = _booking(), [_driver(1, eta=5.0, distance=1.0, acceptance=0.5)]
    mutate(booking, drivers)
    with pytest.raises(ValueError, match=message):
        served["ranker"].rank(booking, drivers)


def test_top_k_must_be_positive(served):
    with pytest.raises(ValueError, match="top_k"):
        served["ranker"].rank(_booking(), [_driver(1, eta=5.0, distance=1.0, acceptance=0.5)], top_k=0)


def test_reasons_name_real_features_with_their_own_values(served):
    booking = _booking()
    driver = _driver(1, eta=3.0, distance=0.8, acceptance=0.95)
    ranked = served["ranker"].rank(booking, [driver], top_k=1)["drivers"][0]
    reasons = ranked["reasons"]
    assert reasons, "a candidate this extreme must have something to explain"
    assert all(reason["feature"] in FEATURE_COLUMNS for reason in reasons)
    assert [abs(reason["contribution"]) for reason in reasons] == sorted(
        (abs(reason["contribution"]) for reason in reasons), reverse=True
    )
    top = reasons[0]
    assert top["feature"] == "estimated_eta_min" and top["effect"] == "increases"
    assert "3.0 min" in top["text"]


def test_reasons_are_deterministic(served):
    booking, drivers = _booking(), [_driver(1, eta=6.0, distance=2.0, acceptance=0.6)]
    first = served["ranker"].rank(booking, drivers, top_k=1)
    second = served["ranker"].rank(booking, drivers, top_k=1)
    assert first == second


def test_the_committed_artifact_matches_the_code_and_the_evaluation():
    model, metadata = read_artifact(MODEL_DIR)
    evaluation = json.loads(EVALUATION_RESULTS.read_text(encoding="utf-8"))
    selected = evaluation["validation_selection"]["selected"]
    assert set(model.feature_order) == set(FEATURE_COLUMNS)
    assert len(model.coefficients) == len(FEATURE_COLUMNS)
    assert metadata["model_type"] == selected["model"]
    assert metadata["policy"]["max_extra_eta_min"] == selected["max_extra_eta_min"]
    assert metadata["features"] == FEATURE_COLUMNS
    assert metadata["metrics"]["test_business"]["completed_rate"] == (
        evaluation["test"]["policies"]["ml_selected"]["metrics"]["completed_rate"]
    )


def test_the_committed_artifact_serves_a_handmade_request():
    result = rank_drivers({"booking": _booking(), "candidate_drivers": [
        _driver(1, eta=4.0, distance=1.2, acceptance=0.8),
        _driver(2, eta=12.0, distance=4.0, acceptance=0.4),
    ]}, top_k=2)
    assert result["model_version"] == read_artifact(MODEL_DIR)[1]["model_version"]
    assert [driver["driver_id"] for driver in result["drivers"]] == [1, 2]
    assert json.dumps(result)  # the response is plain JSON, no numpy scalars


def test_the_ranker_is_loaded_once_per_directory():
    assert get_ranker() is get_ranker()


def _booking(**overrides) -> dict:
    booking = {
        "booking_id": 1, "request_time": "2026-07-20 08:00:00",
        "pickup_lat": 0.0, "pickup_lon": 0.0, "destination_lat": 0.05, "destination_lon": 0.0,
        "passenger_type": "individual", "traffic_level": "medium", "weather": "clear",
    }
    return {**booking, **overrides}


def _driver(driver_id: int, eta: float, distance: float | None, acceptance: float) -> dict:
    return {
        "driver_id": driver_id, "estimated_eta_min": eta, "distance_km": distance, "idle_time_min": 10.0,
        "vehicle_type": "car_4", "rating": 4.7, "acceptance_rate": acceptance,
        "cancellation_rate": 0.08, "completed_trips": 500,
    }
