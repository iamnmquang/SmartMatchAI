"""Phase 6 ablations: how much does each group of features actually contribute? (docs/research.md §7.5)

Run from the repository root, after `python -m ml.data.generate`:

    python -m ml.evaluation.run_ablations

Each run changes exactly **one** thing - which features the model may use - and keeps everything else fixed:
the same train split, the same hyperparameters (the ones the Phase 5 search selected), the same early stopping
on validation, the same replay protocol. Everything is measured on **validation**; the test split belongs to
run_evaluation.py alone.

Business metrics use option A ordering (rank by P(accept), no ETA margin) for every variant, so the comparison
isolates the features rather than mixing in the policy shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.evaluation.offline import REPO_ROOT, build_evaluation_data, business_metrics, simulate_offers
from ml.evaluation.policies import nearest_driver
from ml.features import FEATURE_COLUMNS, build_features
from ml.training.dataset import DEFAULT_RAW_DIR, labeled_offers, load_raw
from ml.training.metrics import classification_metrics, ranking_metrics
from ml.training.models import fit_xgb

RESULTS_PATH = REPO_ROOT / "ml" / "evaluation" / "results" / "ablations.json"
TRAINING_RESULTS = REPO_ROOT / "ml" / "training" / "results" / "training.json"
DEFAULT_ORACLE_DIR = REPO_ROOT / "data" / "oracle"
SPLIT = "validation"

FEATURE_GROUPS = {
    "pair": ["estimated_eta_min", "distance_km", "pickup_to_trip_ratio", "trip_km"],
    "relative": ["distance_minus_min_km"],
    "driver_history": ["rating", "has_rating", "acceptance_rate", "cancellation_rate", "completed_trips", "idle_time_min"],
    "context": [column for column in FEATURE_COLUMNS
                if column.startswith(("traffic_level=", "weather=", "time_of_day=")) or column in ("hour", "is_weekend")],
    "booking": [column for column in FEATURE_COLUMNS if column.startswith(("passenger_type=", "vehicle_type="))],
}

ABLATIONS = [
    ("full", "Every feature (reference for this table)", FEATURE_COLUMNS),
    ("AB2_no_driver_history", "Drop driver history and idle time", None),
    ("AB3_no_context", "Drop traffic, weather and time of day", None),
    ("AB4_distance_eta_only", "Only pickup distance and ETA", ["estimated_eta_min", "distance_km"]),
    ("AB5_no_relative", "Drop the in-booking relative feature", None),
]
DROPPED_GROUPS = {
    "AB2_no_driver_history": ["driver_history"],
    "AB3_no_context": ["context"],
    "AB5_no_relative": ["relative"],
}


def feature_subset(name: str, explicit: list[str] | None) -> list[str]:
    if explicit is not None:
        return list(explicit)
    dropped = {column for group in DROPPED_GROUPS[name] for column in FEATURE_GROUPS[group]}
    return [column for column in FEATURE_COLUMNS if column not in dropped]


def selected_hyperparameters() -> dict:
    """The winner of the Phase 5 random search, minus the tree count that early stopping decides per run."""
    params = dict(json.loads(TRAINING_RESULTS.read_text(encoding="utf-8"))["models"]["xgboost"]["params"])
    params.pop("n_estimators", None)
    return params


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feature ablations on the validation split (Phase 6).")
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    raw = load_raw(args.raw_dir)
    truth = pd.read_parquet(args.oracle_dir / "candidate_truth.parquet")
    data = build_evaluation_data(raw.bookings, raw.candidates, raw.drivers, truth, SPLIT)
    train, validation = labeled_offers(raw, "train"), labeled_offers(raw, SPLIT)
    candidate_features = build_features(data.candidates)
    params = selected_hyperparameters()

    baseline = business_metrics(simulate_offers(data, nearest_driver(data.candidates), args.max_offers))
    runs = []
    for name, description, explicit in ABLATIONS:
        columns = feature_subset(name, explicit)
        model = fit_xgb(params, args.seed, train.features[columns], train.y, validation.features[columns], validation.y)
        p_offers = model.predict_proba(validation.features[columns])[:, 1]
        offer_metrics = classification_metrics(validation.y, p_offers)
        ranking = ranking_metrics(validation.booking_id, validation.y, p_offers)
        metrics = business_metrics(
            simulate_offers(data, model.predict_proba(candidate_features[columns])[:, 1], args.max_offers)
        )
        runs.append({
            "name": name,
            "description": description,
            "n_features": len(columns),
            "features_removed": [column for column in FEATURE_COLUMNS if column not in columns],
            "n_trees": int(model.best_iteration) + 1,
            "roc_auc": round(float(offer_metrics["roc_auc"]), 6),
            "log_loss": round(float(offer_metrics["log_loss"]), 6),
            "hit_at_1": round(float(ranking["hit_at_1"]), 6),
            "business": {key: round(value, 6) for key, value in metrics.items()},
        })
        print(f"{name:24s} {len(columns):3d} features  AUC {runs[-1]['roc_auc']:.4f}  "
              f"MSR {100 * metrics['matching_success_rate']:.1f}%  completed {100 * metrics['completed_rate']:.1f}%")

    payload = {
        "experiment": "phase6-feature-ablations",
        "settings": {
            "split": SPLIT,
            "policy": "option A (rank by P(accept))",
            "model": "XGBoost with the hyperparameters selected in Phase 5",
            "hyperparameters": params,
            "seed": args.seed,
            "max_offers": args.max_offers,
            "feature_groups": FEATURE_GROUPS,
        },
        "baseline_nearest_driver": {key: round(value, 6) for key, value in baseline.items()},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    print()
    print(markdown_table(runs))
    print(f"\nWrote {args.output}")


def markdown_table(runs: list[dict]) -> str:
    full = runs[0]
    lines = ["| Ablation | Features | ROC-AUC | AUC change | Hit@1 | MSR | Completed rate |", "|---|---|---|---|---|---|---|"]
    for run in runs:
        business = run["business"]
        lines.append(
            f"| {run['description']} | {run['n_features']} | {run['roc_auc']:.4f} | "
            f"{run['roc_auc'] - full['roc_auc']:+.4f} | {run['hit_at_1']:.4f} | "
            f"{100 * business['matching_success_rate']:.1f}% | {100 * business['completed_rate']:.1f}% |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
