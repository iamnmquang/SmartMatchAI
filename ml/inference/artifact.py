"""The serving artifact: the selected acceptance model as plain JSON (Phase 7).

The model chosen in Phase 6 is a logistic regression (ADR-016), so its entire state is a few dozen numbers:
the median used for each missing value, the mean and scale of the standardiser, one coefficient per feature
and an intercept. Writing those to JSON instead of pickling the scikit-learn pipeline buys a lot:

* **No version coupling.** A pickle only loads back into a compatible scikit-learn; a JSON file loads
  anywhere. The backend never has to match the training environment.
* **Reviewable in git.** The artifact is a diffable text file, so a model change shows up in code review
  instead of arriving as an opaque binary blob (ADR-018).
* **No scikit-learn at serving time.** Scoring is one matrix product; the backend needs numpy and pandas only.

The price is that the export has to reproduce the pipeline exactly - which is what
`ml/tests/test_inference.py` checks against the fitted pipeline on real data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS

ARTIFACT_VERSION = 1  # format of this file, not the model


@dataclass(frozen=True)
class LinearAcceptanceModel:
    """P(accept) = sigmoid(intercept + sum_j coefficient_j * z_j), where z is the standardised feature vector.

    The steps mirror ml/training/models.linear_pipeline, in the same order: log1p on the heavy-tailed counts,
    median imputation of what is still missing, standardisation, then the linear score.
    """

    feature_order: list[str]  # column order the coefficients belong to
    log1p_features: list[str]
    imputer_medians: list[float]
    scaler_mean: list[float]
    scaler_scale: list[float]
    coefficients: list[float]
    intercept: float

    @classmethod
    def from_pipeline(cls, pipeline) -> LinearAcceptanceModel:
        transformer = pipeline.named_steps["log1p"]
        return cls(
            feature_order=[str(name) for name in transformer.get_feature_names_out()],
            log1p_features=list(transformer.transformers_[0][2]),
            imputer_medians=pipeline.named_steps["impute"].statistics_.tolist(),
            scaler_mean=pipeline.named_steps["scale"].mean_.tolist(),
            scaler_scale=pipeline.named_steps["scale"].scale_.tolist(),
            coefficients=pipeline.named_steps["model"].coef_[0].tolist(),
            intercept=float(pipeline.named_steps["model"].intercept_[0]),
        )

    def standardise(self, features: pd.DataFrame) -> np.ndarray:
        """Feature matrix -> standardised values, in this model's column order."""
        missing = [column for column in self.feature_order if column not in features.columns]
        if missing:
            raise ValueError(f"feature matrix is missing columns: {missing}")
        values = features[self.feature_order].to_numpy(dtype=float, copy=True)
        log1p_index = [self.feature_order.index(column) for column in self.log1p_features]
        values[:, log1p_index] = np.log1p(values[:, log1p_index])
        medians = np.asarray(self.imputer_medians, dtype=float)
        values = np.where(np.isnan(values), medians, values)
        return (values - np.asarray(self.scaler_mean)) / np.asarray(self.scaler_scale)

    def contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Each feature's share of the log-odds: coefficient * standardised value, labelled by feature.

        A standardised value of 0 is the training average, so a contribution reads as "how much this candidate
        differs from an average candidate because of this feature" - which is exactly what a reason should say.
        Returned as a frame, not an array: the column order here is the artifact's, not the caller's.
        """
        values = self.standardise(features) * np.asarray(self.coefficients)
        return pd.DataFrame(values, index=features.index, columns=self.feature_order)

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        return self.standardise(features) @ np.asarray(self.coefficients) + self.intercept

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(features)))


def write_artifact(model: LinearAcceptanceModel, metadata: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"artifact_version": ARTIFACT_VERSION, "model_type": "logistic_regression", **asdict(model)}
    _write_json(directory / "acceptance_model.json", payload)
    _write_json(directory / "metadata.json", metadata)


def read_artifact(directory: Path) -> tuple[LinearAcceptanceModel, dict]:
    payload = json.loads((directory / "acceptance_model.json").read_text(encoding="utf-8"))
    if payload["artifact_version"] != ARTIFACT_VERSION:
        raise ValueError(f"unsupported artifact version {payload['artifact_version']}, expected {ARTIFACT_VERSION}")
    fields = {key: payload[key] for key in LinearAcceptanceModel.__dataclass_fields__}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    unknown = set(fields["feature_order"]) - set(FEATURE_COLUMNS)
    if unknown:
        raise ValueError(f"artifact expects features the current code does not build: {sorted(unknown)}")
    return LinearAcceptanceModel(**fields), metadata


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
