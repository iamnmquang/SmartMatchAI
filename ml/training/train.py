"""Phase 5: train the acceptance model P(driver accepts | booking, driver, context).

Run from the repository root, after `python -m ml.data.generate`:

    python -m ml.training.train                 # full run: linear grid + random search for XGBoost
    python -m ml.training.train --trials 5      # quick run

What it does, and why in this order:

1. Loads the logged offers of the train and validation splits (ml/training/dataset.py). The test split is not
   loaded at all - it is spent once, in Phase 6.
2. Fits the linear reference over a small C grid and XGBoost over a random search with early stopping.
3. Selects each model family by **validation log loss**: a proper scoring rule, so it rewards a probability
   that is both discriminative and calibrated, which is what a later utility score would need (PRD Q2).
   ROC-AUC, PR-AUC and the in-booking ranking metrics are reported next to it, so a disagreement is visible.
4. Writes ml/training/results/training.json (deterministic, no timestamps) and saves the fitted models.

The selected model stays trained on the train split only. Validation is deliberately left unused by the final
fit so that Phase 6 can still tune a policy on it (for instance the ETA margin of option B) against a model
that has never seen it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.features import FEATURE_COLUMNS
from ml.training.dataset import DEFAULT_RAW_DIR, LabeledSplit, load_labeled_splits
from ml.training.metrics import calibration_bins, calibration_by_eta, classification_metrics, ranking_metrics
from ml.training.models import (
    ARTIFACT_DIR,
    LINEAR_C_GRID,
    XGB_FIXED_PARAMS,
    fit_xgb,
    linear_pipeline,
    sample_xgb_params,
    save_models,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "ml" / "training" / "results" / "training.json"
TRAIN_SPLITS = ("train", "validation")  # the test split is out of bounds until Phase 6
SELECTION_METRIC = "log_loss"
TOP_IMPORTANCE = 15


def predict_accept(model, features) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def evaluate(model, split: LabeledSplit) -> dict:
    p = predict_accept(model, split.features)
    return {**classification_metrics(split.y, p), "ranking": ranking_metrics(split.booking_id, split.y, p)}


def search_linear(train: LabeledSplit, validation: LabeledSplit, seed: int):
    trials = []
    for C in LINEAR_C_GRID:
        model = linear_pipeline(C, seed).fit(train.features, train.y)
        metrics = classification_metrics(validation.y, predict_accept(model, validation.features))
        trials.append({
            "params": {"C": C},
            "validation_log_loss": round(float(metrics["log_loss"]), 6),
            "validation_roc_auc": round(float(metrics["roc_auc"]), 6),
        })
    best = min(trials, key=lambda trial: trial["validation_log_loss"])
    return linear_pipeline(best["params"]["C"], seed).fit(train.features, train.y), trials


def search_xgboost(train: LabeledSplit, validation: LabeledSplit, n_trials: int, seed: int):
    rng = np.random.default_rng(seed)
    best_model, best_score, trials = None, np.inf, []
    for index in range(n_trials):
        params = sample_xgb_params(rng)
        model = fit_xgb(params, seed, train.features, train.y, validation.features, validation.y)
        metrics = classification_metrics(validation.y, predict_accept(model, validation.features))
        trials.append({
            "trial": index,
            "params": params,
            "best_iteration": int(model.best_iteration),
            "validation_log_loss": round(float(metrics["log_loss"]), 6),
            "validation_roc_auc": round(float(metrics["roc_auc"]), 6),
        })
        if metrics[SELECTION_METRIC] < best_score:
            best_model, best_score = model, metrics[SELECTION_METRIC]
    return best_model, trials


def gain_importance(model, top: int = TOP_IMPORTANCE) -> list[dict]:
    scores = model.get_booster().get_score(importance_type="gain")
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top]
    return [{"feature": name, "gain_share": round(value / total, 6)} for name, value in ranked]


def linear_coefficients(model) -> list[dict]:
    names = model[:-1].get_feature_names_out()
    coefficients = model[-1].coef_[0]
    order = np.argsort(-np.abs(coefficients))
    return [{"feature": str(names[i]), "coefficient": round(float(coefficients[i]), 4)} for i in order]


def model_report(model, splits: dict[str, LabeledSplit], params: dict) -> dict:
    validation = splits["validation"]
    p = predict_accept(model, validation.features)
    return {
        "params": params,
        "metrics": {name: round_floats(evaluate(model, split)) for name, split in splits.items()},
        "calibration_validation": round_floats(calibration_bins(validation.y, p)),
        "calibration_by_eta_validation": calibration_by_eta(
            validation.view["estimated_eta_min"].to_numpy(), validation.y, p
        ),
    }


def round_floats(value, digits: int = 6):
    if isinstance(value, dict):
        return {key: round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, digits) for item in value]
    return round(value, digits) if isinstance(value, float) else value


TABLE_ROWS = [
    ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("log_loss", "Log loss"), ("brier", "Brier"),
    ("precision_at_0.5", "Precision @0.5"), ("recall_at_0.5", "Recall @0.5"), ("f1_at_0.5", "F1 @0.5"),
]


def markdown_table(payload: dict) -> str:
    names = list(payload["models"])
    lines = ["| Validation metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key, label in TABLE_ROWS:
        cells = [format(payload["models"][name]["metrics"]["validation"][key], ".4f") for name in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in [("hit_at_1", "Hit@1 (in booking)"), ("ndcg_at_5", "NDCG@5 (in booking)")]:
        cells = [format(payload["models"][name]["metrics"]["validation"]["ranking"][key], ".4f") for name in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = [
        format(payload["models"][name]["calibration_validation"]["expected_calibration_error"], ".4f") for name in names
    ]
    lines.append("| Expected calibration error | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_payload(linear, linear_trials, xgboost, xgb_trials, splits, args, data_metadata) -> dict:
    best_trial = min(xgb_trials, key=lambda trial: trial["validation_log_loss"])
    return {
        "experiment": "phase5-acceptance-model",
        "data": {"generator_version": data_metadata["generator_version"], "seed": data_metadata["config"]["seed"]},
        "settings": {
            "target": "accepted",
            "train_split": "train",
            "tuning_split": "validation",
            "test_split": "not used in Phase 5 (spent once in Phase 6)",
            "selection_metric": f"validation {SELECTION_METRIC}",
            "seed": args.seed,
            "xgboost_trials": args.trials,
            "xgboost_fixed_params": XGB_FIXED_PARAMS,
            "n_features": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS,
        },
        "models": {
            "logistic_regression": {
                **model_report(linear, splits, {"C": linear[-1].C}),
                "coefficients": linear_coefficients(linear),
            },
            "xgboost": {
                **model_report(
                    xgboost, splits, {**best_trial["params"], "n_estimators": int(xgboost.best_iteration) + 1}
                ),
                "gain_importance": gain_importance(xgboost),
            },
        },
        "search": {"logistic_regression": linear_trials, "xgboost": xgb_trials},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the acceptance model (Phase 5).")
    parser.add_argument("--trials", type=int, default=30, help="random search trials for XGBoost")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--no-save", action="store_true", help="skip writing the fitted models")
    args = parser.parse_args(argv)

    splits = load_labeled_splits(TRAIN_SPLITS, args.raw_dir)
    train, validation = splits["train"], splits["validation"]
    print(f"train: {len(train):,} logged offers from {train.n_bookings:,} bookings (accept {train.positive_rate:.3f})")
    print(f"validation: {len(validation):,} offers from {validation.n_bookings:,} bookings "
          f"(accept {validation.positive_rate:.3f})")

    linear, linear_trials = search_linear(train, validation, args.seed)
    print(f"logistic regression: best C = {linear[-1].C}")
    xgboost, xgb_trials = search_xgboost(train, validation, args.trials, args.seed)
    best_trial = min(xgb_trials, key=lambda trial: trial["validation_log_loss"])
    print("xgboost: best trial {} of {}, {} trees, validation log loss {:.4f}".format(
        best_trial["trial"], args.trials, best_trial["best_iteration"] + 1, best_trial["validation_log_loss"]))

    data_metadata = json.loads((args.raw_dir / "metadata.json").read_text(encoding="utf-8"))
    payload = build_payload(linear, linear_trials, xgboost, xgb_trials, splits, args, data_metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    if not args.no_save:
        save_models(linear, xgboost, args.artifact_dir)
        print(f"Saved fitted models to {args.artifact_dir} (serving format is decided in Phase 7)")

    print()
    print(markdown_table(payload))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
