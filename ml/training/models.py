"""The two models compared in Phase 5 (ADR-006): a linear reference and the gradient boosting candidate.

Both consume the shared feature matrix of ml/features. The difference is what each needs on top of it:

* Logistic regression cannot read a NaN and is sensitive to scale and to long tails, so it gets a fitted
  pipeline: log1p on the two heavy-tailed counts, median imputation of the missing ratings (the "is it
  missing" information is already carried by the has_rating feature), then standardisation. The pipeline is
  fitted on the train split only - fitting it on train + validation would leak the development set into
  preprocessing (docs/research.md §5.3).
* XGBoost needs none of that: it splits on raw values and routes NaN to its own branch.

Ablation AB1 of docs/research.md §7.5 is exactly the comparison between the two.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from ml.features import LOG1P_FEATURES

LINEAR_C_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]

XGB_FIXED_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 3000,  # an upper bound; early stopping on the validation split picks the real number
    "early_stopping_rounds": 50,
    "n_jobs": 4,  # fixed: hist is deterministic for a given thread count
}


def linear_pipeline(C: float, seed: int) -> Pipeline:
    log1p = ColumnTransformer(
        [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), LOG1P_FEATURES)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("log1p", log1p),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
    ])


def sample_xgb_params(rng: np.random.Generator) -> dict[str, float | int]:
    """One draw of the random search. Ranges are the usual safe ones for ~100k rows and 28 features."""
    return {
        "max_depth": int(rng.integers(3, 10)),
        "learning_rate": round(float(_loguniform(rng, 0.02, 0.25)), 6),
        "min_child_weight": round(float(_loguniform(rng, 1.0, 60.0)), 6),
        "subsample": round(float(rng.uniform(0.6, 1.0)), 6),
        "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 6),
        "reg_lambda": round(float(_loguniform(rng, 0.5, 25.0)), 6),
        "reg_alpha": round(float(_loguniform(rng, 1e-3, 1.0)) if rng.random() < 0.5 else 0.0, 6),
        "gamma": round(float(rng.uniform(0.0, 2.0)) if rng.random() < 0.5 else 0.0, 6),
    }


def fit_xgb(params: dict, seed: int, X_train, y_train, X_validation, y_validation) -> XGBClassifier:
    model = XGBClassifier(**XGB_FIXED_PARAMS, **params, random_state=seed)
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
    return model


def _loguniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# Working files for the offline pipeline, not the serving contract: how the backend loads a model, and whether
# the artifact is committed or built, is decided in Phase 7 (AQ1, AQ2). XGBoost gets its own JSON format rather
# than pickle because it stays readable across library versions.
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "ml" / "training" / "artifacts"
XGB_FILE = "xgboost_accept.json"
LINEAR_FILE = "logistic_regression_accept.joblib"


def save_models(linear, xgb, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    xgb.save_model(directory / XGB_FILE)
    joblib.dump(linear, directory / LINEAR_FILE)


def load_models(directory: Path) -> dict[str, object]:
    xgb = XGBClassifier()
    xgb.load_model(directory / XGB_FILE)
    return {"logistic_regression": joblib.load(directory / LINEAR_FILE), "xgboost": xgb}
