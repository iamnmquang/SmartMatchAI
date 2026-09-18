"""Tests for the Phase 6 comparison: the ML policies, the selection rule and the report it produces."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate, write_dataset
from ml.evaluation.policies import apply_eta_constraint, model_policy, model_scores
from ml.evaluation.run_ablations import ABLATIONS, FEATURE_GROUPS, feature_subset
from ml.evaluation.run_evaluation import gap_closed, hypotheses, main as evaluation_main, select_policy
from ml.features import FEATURE_COLUMNS
from ml.tests.test_features import toy_view
from ml.training.train import main as train_main

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_RESULTS = REPO_ROOT / "ml" / "evaluation" / "results" / "evaluation.json"


class RecordingModel:
    """A stand-in for a trained model: returns fixed probabilities and remembers what it was asked to score."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.seen_columns = None

    def predict_proba(self, features):
        self.seen_columns = list(features.columns)
        return np.column_stack([1 - self.probabilities, self.probabilities])


def search_row(model: str, delta, completed: float, eta: float, cancellation: float) -> dict:
    return {
        "model": model,
        "max_extra_eta_min": delta,
        "option": "A" if delta is None else "B",
        "label": f"{model} {delta}",
        "metrics": {"completed_rate": completed, "avg_eta_matched_min": eta, "cancellation_rate": cancellation},
    }


def test_model_scores_go_through_the_shared_feature_code():
    view = toy_view()
    model = RecordingModel([0.9, 0.2, 0.5])
    scores = model_scores(model, view)
    assert model.seen_columns == FEATURE_COLUMNS
    assert scores.tolist() == [0.9, 0.2, 0.5]


def test_an_infinite_eta_margin_reproduces_option_a():
    view = toy_view()
    scores = np.array([0.9, 0.2, 0.5])
    assert np.array_equal(apply_eta_constraint(scores, view, float("inf")), scores)


def test_the_eta_margin_moves_slow_candidates_behind_without_dropping_them():
    """Booking 1: driver 10 is 4 minutes away, driver 11 is 11 minutes away but likelier to accept."""
    view = toy_view()
    scores = np.array([0.30, 0.95, 0.50])
    constrained = apply_eta_constraint(scores, view, max_extra_eta_min=1.0)
    assert constrained[0] > constrained[1]  # the fast candidate is offered first
    assert len(constrained) == len(scores) and np.isfinite(constrained).all()  # nobody is dropped


def test_the_eta_margin_keeps_the_order_inside_each_group():
    view = pd.concat([toy_view(), toy_view().iloc[[1]].assign(driver_id=12, estimated_eta_min=12.0)], ignore_index=True)
    scores = np.array([0.30, 0.95, 0.50, 0.80])
    constrained = apply_eta_constraint(scores, view, max_extra_eta_min=1.0)
    # Booking 1: driver 10 eligible; drivers 11 and 12 are not, and keep their relative order (0.95 > 0.80).
    assert constrained[0] > constrained[1] > constrained[3]


def test_a_wide_margin_makes_every_candidate_eligible():
    view = toy_view()
    scores = np.array([0.30, 0.95, 0.50])
    constrained = apply_eta_constraint(scores, view, max_extra_eta_min=100.0)
    assert np.argsort(-constrained).tolist() == np.argsort(-scores).tolist()


def test_model_policy_returns_one_score_per_candidate():
    view = toy_view()
    policy = model_policy(RecordingModel([0.9, 0.2, 0.5]), max_extra_eta_min=2.0)
    scores = policy(view)
    assert scores.shape == (len(view),) and np.isfinite(scores).all()


def test_selection_prefers_the_best_completed_rate_within_both_guardrails():
    baseline = {"avg_eta_matched_min": 10.0, "cancellation_rate": 0.12}
    rows = [
        search_row("xgboost", None, completed=0.80, eta=12.0, cancellation=0.11),  # fails H2 (+20%)
        search_row("xgboost", 5.0, completed=0.78, eta=11.0, cancellation=0.13),  # fails H3
        search_row("xgboost", 1.0, completed=0.76, eta=10.5, cancellation=0.11),  # keeps both
        search_row("xgboost", 0.0, completed=0.74, eta=10.1, cancellation=0.10),  # keeps both, worse
    ]
    selected = select_policy(rows, baseline)
    assert selected["max_extra_eta_min"] == 1.0
    assert selected["n_eligible"] == 2
    assert "H2 and H3" in selected["rule"]


def test_selection_falls_back_to_the_eta_guardrail_when_no_policy_keeps_cancellation():
    baseline = {"avg_eta_matched_min": 10.0, "cancellation_rate": 0.12}
    rows = [
        search_row("xgboost", None, completed=0.80, eta=12.0, cancellation=0.13),  # fails H2
        search_row("xgboost", 1.0, completed=0.76, eta=10.5, cancellation=0.13),  # fails H3 only
    ]
    selected = select_policy(rows, baseline)
    assert selected["max_extra_eta_min"] == 1.0
    assert "none satisfied H3" in selected["rule"]


def test_gap_closed_is_reported_only_where_the_oracle_is_an_upper_bound():
    policies = {
        "nearest_driver": {"metrics": {"matching_success_rate": 0.80, "completed_rate": 0.75}},
        "oracle": {"metrics": {"matching_success_rate": 0.90, "completed_rate": 0.74}},
        "ml_selected": {"metrics": {"matching_success_rate": 0.85, "completed_rate": 0.76}},
    }
    assert gap_closed(policies, "matching_success_rate", "ml_selected") == 0.5
    assert gap_closed(policies, "completed_rate", "ml_selected") is None  # oracle below the baseline


def test_hypotheses_follow_the_bootstrap_intervals():
    def policy(msr, completed, eta, cancellation, intervals):
        return {
            "metrics": {"matching_success_rate": msr, "completed_rate": completed,
                        "avg_eta_matched_min": eta, "cancellation_rate": cancellation},
            "bootstrap": intervals,
        }

    intervals = {
        "matching_success_rate": {"diff_vs_reference_ci95": [0.005, 0.015]},
        "completed_rate": {"diff_vs_reference_ci95": [-0.002, 0.006]},
        "cancellation_rate": {"diff_vs_reference_ci95": [0.002, 0.008]},
    }
    policies = {
        "nearest_driver": policy(0.85, 0.75, 7.0, 0.12, {}),
        "ml_selected": policy(0.86, 0.755, 8.5, 0.125, intervals),
    }
    verdicts = {row["id"]: row["verdict"] for row in hypotheses(policies, {"xgboost": {"roc_auc": 0.88}}, "xgboost")}
    assert verdicts["H1"] == "supported"  # interval strictly above 0
    assert verdicts["H1b"] == "not supported"  # interval contains 0
    assert verdicts["H2"] == "not supported"  # +21% ETA
    assert verdicts["H3"] == "not supported"  # cancellation significantly higher
    assert verdicts["H4"] == "supported"


def test_each_ablation_removes_exactly_the_intended_feature_group():
    for name, _, explicit in ABLATIONS:
        columns = feature_subset(name, explicit)
        assert set(columns) <= set(FEATURE_COLUMNS)
        if name == "full":
            assert columns == FEATURE_COLUMNS
        elif name == "AB2_no_driver_history":
            assert set(FEATURE_GROUPS["driver_history"]).isdisjoint(columns)
            assert set(FEATURE_GROUPS["context"]) <= set(columns)
        elif name == "AB5_no_relative":
            assert len(columns) == len(FEATURE_COLUMNS) - 1


def test_evaluation_cli_runs_end_to_end(tmp_path):
    ds = generate(GeneratorConfig(seed=31, n_drivers=1_000, n_bookings=6_000))
    write_dataset(ds, tmp_path / "raw", tmp_path / "oracle")
    train_main(["--trials", "1", "--seed", "3", "--raw-dir", str(tmp_path / "raw"),
                "--artifact-dir", str(tmp_path / "models"), "--output", str(tmp_path / "training.json")])
    output = tmp_path / "evaluation.json"
    evaluation_main(["--resamples", "20", "--raw-dir", str(tmp_path / "raw"), "--oracle-dir", str(tmp_path / "oracle"),
                     "--artifact-dir", str(tmp_path / "models"), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    selected = payload["validation_selection"]["selected"]
    assert selected["model"] in {"logistic_regression", "xgboost"}
    assert {"random", "nearest_driver", "weighted_rule", "ml_selected", "oracle"} <= set(payload["test"]["policies"])
    assert {row["id"] for row in payload["test"]["hypotheses"]} == {"H1", "H1b", "H2", "H3", "H4"}
    for policy in payload["test"]["policies"].values():
        assert 0 <= policy["metrics"]["matching_success_rate"] <= 1
    # The oracle is the ceiling for acceptance; random is the floor for matching.
    metrics = {name: policy["metrics"] for name, policy in payload["test"]["policies"].items()}
    assert metrics["oracle"]["acceptance_rate"] >= metrics["ml_selected"]["acceptance_rate"]
    assert metrics["random"]["matching_success_rate"] < metrics["nearest_driver"]["matching_success_rate"]


def test_committed_evaluation_is_consistent_with_itself():
    payload = json.loads(EVALUATION_RESULTS.read_text(encoding="utf-8"))
    selected = payload["validation_selection"]["selected"]
    matching = [row for row in payload["validation_selection"]["candidates"]
                if row["model"] == selected["model"] and row["max_extra_eta_min"] == selected["max_extra_eta_min"]]
    assert len(matching) == 1
    assert matching[0]["metrics"] == selected["validation_metrics"]
    best = max(row["metrics"]["completed_rate"] for row in payload["validation_selection"]["candidates"]
               if row["metrics"]["avg_eta_matched_min"] <= selected["eta_cap_min"])
    assert selected["validation_metrics"]["completed_rate"] == pytest.approx(best)
    assert payload["settings"]["model_features"] == FEATURE_COLUMNS
