"""Tests for Phase 5 training: what the model is allowed to learn from, and whether the metrics are right."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate, write_dataset
from ml.features import FEATURE_COLUMNS, build_features
from ml.training.dataset import RawData, candidate_features, labeled_offers
from ml.training.metrics import calibration_bins, classification_metrics, ranking_metrics
from ml.training.models import linear_pipeline
from ml.training.train import TRAIN_SPLITS, main as train_main, search_linear

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def raw():
    ds = generate(GeneratorConfig(seed=23, n_drivers=1_000, n_bookings=5_000))
    return RawData(bookings=ds.bookings, candidates=ds.candidates, drivers=ds.drivers), ds


def test_only_logged_offers_become_training_rows(raw):
    data, ds = raw
    split = labeled_offers(data, "train")
    view_all, _, labels = candidate_features(data, "train")
    assert len(split) == int(labels.notna().sum()) < len(view_all)  # most candidates were never offered
    assert split.features.shape == (len(split), len(FEATURE_COLUMNS))
    assert np.array_equal(split.y, labels[labels.notna()].to_numpy(dtype=bool))


def test_relative_features_are_computed_over_the_whole_candidate_set(raw):
    """Online, every candidate of the booking is present. Building features from the offered rows alone would
    give a different 'how much further than the closest candidate' - a silent training-serving skew."""
    data, _ = raw
    split = labeled_offers(data, "train")
    view_all, features_all, labels = candidate_features(data, "train")
    labeled = labels.notna().to_numpy()

    from_full_set = features_all.loc[labeled, "distance_minus_min_km"].to_numpy()
    from_offered_rows = build_features(view_all.loc[labeled])["distance_minus_min_km"].to_numpy()
    assert np.array_equal(split.features["distance_minus_min_km"].to_numpy(), from_full_set)
    assert not np.allclose(from_full_set, from_offered_rows)


def test_training_never_reads_the_test_split(raw):
    data, _ = raw
    assert "test" not in TRAIN_SPLITS
    train, validation = labeled_offers(data, "train"), labeled_offers(data, "validation")
    assert train.view["request_time"].max() < validation.view["request_time"].min()
    assert set(train.booking_id).isdisjoint(validation.booking_id)


def test_linear_preprocessing_is_fitted_on_the_train_split_only(raw):
    data, _ = raw
    train, validation = labeled_offers(data, "train"), labeled_offers(data, "validation")
    model, trials = search_linear(train, validation, seed=3)
    names = list(model.named_steps["log1p"].get_feature_names_out())
    statistic = model.named_steps["impute"].statistics_[names.index("rating")]
    assert statistic == pytest.approx(float(np.nanmedian(train.features["rating"])))
    assert len(trials) == len(set(trial["params"]["C"] for trial in trials))


def code_references(path: Path) -> list[str]:
    """Identifiers, imports and string literals of a module.

    Comments and docstrings are exempt: they may legitimately talk about the hidden truth. So is a deny-list
    constant such as FORBIDDEN_COLUMNS, whose whole job is to name the columns that must never be used.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exempt = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                exempt.add(id(first.value))
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and "FORBIDDEN" in target.id for target in node.targets
        ):
            exempt.update(id(child) for child in ast.walk(node.value) if isinstance(child, ast.Constant))
    references = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in exempt:
            references.append(node.value)
        elif isinstance(node, ast.Name):
            references.append(node.id)
        elif isinstance(node, ast.Attribute):
            references.append(node.attr)
        elif isinstance(node, ast.alias):
            references.append(node.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            references.append(node.module)
    return references


def test_feature_and_training_code_never_touch_the_hidden_truth():
    """Oracle leakage would not fail any numeric assertion - only the whole experiment. So it is checked here."""
    forbidden = ["oracle", "candidate_truth", "driver_latents", "behavior", "u_accept", "p_cancel", "p_accept"]
    for directory in ("ml/features", "ml/training"):
        for file in (REPO_ROOT / directory).rglob("*.py"):
            for reference in code_references(file):
                hit = next((token for token in forbidden if token in reference), None)
                assert hit is None, f"{file.relative_to(REPO_ROOT)} refers to {hit!r} in code: {reference!r}"


def test_classification_metrics_on_a_hand_checked_example():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.4])
    metrics = classification_metrics(y, p)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision_at_0.5"] == 1.0
    assert metrics["recall_at_0.5"] == 1.0
    assert metrics["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert metrics["positive_rate"] == 0.5


def test_ranking_metrics_on_a_hand_checked_example():
    booking_id = np.array([1, 1, 1, 2, 2, 3, 3, 4])
    y = np.array([1, 0, 0, 0, 1, 0, 0, 1])
    p = np.array([0.9, 0.5, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6])
    metrics = ranking_metrics(booking_id, y, p)
    # Booking 3 has no acceptance and booking 4 a single labelled offer: neither can inform an ordering.
    assert metrics["n_bookings"] == 2
    assert metrics["hit_at_1"] == 0.5
    assert metrics["ndcg_at_5"] == pytest.approx((1.0 + 1.0 / np.log2(3)) / 2)


def test_ranking_metrics_reward_the_better_order():
    booking_id = np.array([1, 1, 2, 2])
    y = np.array([1, 0, 1, 0])
    good = ranking_metrics(booking_id, y, np.array([0.9, 0.1, 0.8, 0.2]))
    bad = ranking_metrics(booking_id, y, np.array([0.1, 0.9, 0.2, 0.8]))
    assert good["hit_at_1"] == 1.0 and good["ndcg_at_5"] == 1.0
    assert bad["hit_at_1"] == 0.0 and bad["ndcg_at_5"] < good["ndcg_at_5"]


def test_calibration_of_a_perfectly_calibrated_score_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=20_000)
    y = rng.random(size=20_000) < p
    assert calibration_bins(y, p)["expected_calibration_error"] < 0.02


def test_linear_pipeline_handles_missing_ratings(raw):
    data, _ = raw
    train = labeled_offers(data, "train")
    assert train.features["rating"].isna().any()
    model = linear_pipeline(1.0, seed=3).fit(train.features, train.y)
    predictions = model.predict_proba(train.features)[:, 1]
    assert np.isfinite(predictions).all() and ((0 < predictions) & (predictions < 1)).all()


def test_train_cli_writes_a_complete_report(tmp_path):
    ds = generate(GeneratorConfig(seed=29, n_drivers=1_000, n_bookings=5_000))
    write_dataset(ds, tmp_path / "raw", tmp_path / "oracle")
    output = tmp_path / "training.json"
    train_main(["--trials", "2", "--seed", "3", "--no-save", "--raw-dir", str(tmp_path / "raw"), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["settings"]["features"] == FEATURE_COLUMNS
    assert payload["settings"]["tuning_split"] == "validation"
    assert set(payload["models"]) == {"logistic_regression", "xgboost"}
    assert len(payload["search"]["xgboost"]) == 2
    for report in payload["models"].values():
        validation = report["metrics"]["validation"]
        assert 0.7 < validation["roc_auc"] < 0.95  # PRD H4: clearly better than chance, not suspiciously perfect
        assert report["calibration_validation"]["expected_calibration_error"] < 0.05
        assert len(report["calibration_by_eta_validation"]) >= 3
    assert "p_accept" not in output.read_text(encoding="utf-8")


def test_committed_training_results_match_the_current_feature_set():
    """The committed run must stay in sync with the code it documents."""
    payload = json.loads((REPO_ROOT / "ml" / "training" / "results" / "training.json").read_text(encoding="utf-8"))
    assert payload["settings"]["features"] == FEATURE_COLUMNS
    assert payload["settings"]["test_split"].startswith("not used")
    validation = payload["models"]["xgboost"]["metrics"]["validation"]
    assert 0.5 < validation["roc_auc"] < 0.95  # PRD H4
    search = payload["search"]["xgboost"]
    assert len(search) == payload["settings"]["xgboost_trials"]
    assert validation["log_loss"] == min(trial["validation_log_loss"] for trial in search)
