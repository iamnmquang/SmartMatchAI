"""Export the selected model to ml/models/ as the serving artifact (Phase 7).

Run from the repository root, after training and evaluation:

    python -m ml.inference.export

Reads three things, all of them results of earlier phases rather than new decisions:

* the fitted pipeline from ml/training/artifacts (Phase 5),
* which model and which ETA margin were selected, from ml/evaluation/results/evaluation.json (Phase 6),
* the metrics both runs reported, so the artifact carries the numbers it is accountable for.

The output in ml/models/ is committed (ADR-018): it is small, textual, and it is what the backend, the
Docker image and the model_versions table of Phase 8 all point at.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from ml.features import FEATURE_COLUMNS
from ml.inference.artifact import LinearAcceptanceModel, write_artifact
from ml.training.models import ARTIFACT_DIR, LINEAR_FILE

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "ml" / "models"
TRAINING_RESULTS = REPO_ROOT / "ml" / "training" / "results" / "training.json"
EVALUATION_RESULTS = REPO_ROOT / "ml" / "evaluation" / "results" / "evaluation.json"

# Bumped by hand when the served model changes. Phase 8 stores it in model_versions.
MODEL_VERSION = "1.0.0"
SUPPORTED_MODEL = "logistic_regression"

METRIC_KEYS = ["roc_auc", "pr_auc", "log_loss", "brier", "precision_at_0.5", "recall_at_0.5", "f1_at_0.5"]
BUSINESS_KEYS = ["matching_success_rate", "acceptance_rate", "avg_eta_matched_min", "cancellation_rate", "completed_rate"]


def build_metadata(training: dict, evaluation: dict) -> dict:
    selected = evaluation["validation_selection"]["selected"]
    if selected["model"] != SUPPORTED_MODEL:
        raise ValueError(
            f"Phase 6 selected {selected['model']!r}, but the JSON artifact only supports {SUPPORTED_MODEL!r}. "
            "Export the selected model in its own format, or re-run the evaluation."
        )
    test = evaluation["test"]
    return {
        "model_version": MODEL_VERSION,
        "model_type": SUPPORTED_MODEL,
        "target": "accepted",
        "n_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "policy": {
            "type": "rank by P(accept) within an ETA margin",
            "max_extra_eta_min": selected["max_extra_eta_min"],
            "decision": "ADR-015",
            "note": "candidates outside the margin are ranked last, never dropped",
        },
        "training": {
            "generator_version": training["data"]["generator_version"],
            "data_seed": training["data"]["seed"],
            "train_split": training["settings"]["train_split"],
            "tuning_split": training["settings"]["tuning_split"],
            "seed": training["settings"]["seed"],
            "hyperparameters": training["models"][SUPPORTED_MODEL]["params"],
        },
        "metrics": {
            "validation_offers": {key: training["models"][SUPPORTED_MODEL]["metrics"]["validation"][key] for key in METRIC_KEYS},
            "test_offers": {key: test["ml_metrics"][SUPPORTED_MODEL][key] for key in METRIC_KEYS},
            "test_business": {key: test["policies"]["ml_selected"]["metrics"][key] for key in BUSINESS_KEYS},
            "test_business_baseline_nearest_driver": {
                key: test["policies"]["nearest_driver"]["metrics"][key] for key in BUSINESS_KEYS
            },
        },
        "reproduce": [
            "python -m ml.data.generate",
            "python -m ml.training.train",
            "python -m ml.evaluation.run_evaluation",
            "python -m ml.inference.export",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the selected model as the serving artifact (Phase 7).")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR, help="fitted models from training")
    parser.add_argument("--training-results", type=Path, default=TRAINING_RESULTS)
    parser.add_argument("--evaluation-results", type=Path, default=EVALUATION_RESULTS)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)

    pipeline = joblib.load(args.artifact_dir / LINEAR_FILE)
    model = LinearAcceptanceModel.from_pipeline(pipeline)
    metadata = build_metadata(
        json.loads(args.training_results.read_text(encoding="utf-8")),
        json.loads(args.evaluation_results.read_text(encoding="utf-8")),
    )
    write_artifact(model, metadata, args.output)

    print(f"model {metadata['model_version']} ({metadata['model_type']}), {len(model.coefficients)} coefficients")
    print(f"policy: P(accept) within +{metadata['policy']['max_extra_eta_min']} min of the fastest candidate")
    print(f"Wrote {args.output / 'acceptance_model.json'} and {args.output / 'metadata.json'}")


if __name__ == "__main__":
    main()
