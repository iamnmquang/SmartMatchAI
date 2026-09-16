"""Metrics for the acceptance model (PRD §8.4, docs/research.md §7.1).

Three questions, three groups of metrics:

* Classification - does the score separate accepted from refused offers, and is it a usable probability?
  ROC-AUC and PR-AUC judge the ranking of scores, log loss and Brier judge the probability itself. Precision /
  recall / F1 at a 0.5 threshold are diagnostics only: ranking never applies a threshold.
* Ranking - inside one booking, is the driver who accepted put first? This is what dispatch actually needs, but
  offline it can only be measured on the 1-3 candidates per booking that were offered and therefore have a
  label, and those were chosen by the logging policy. Treat it as a directional check; the honest ranking
  measurement is the dispatch replay of Phase 6.
* Calibration - if the score is used inside a utility later (PRD Q2, option B/C), 0.7 must mean 70%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ETA_BUCKET_EDGES = [0.0, 6.0, 10.0, 15.0, 20.0, np.inf]
DECISION_THRESHOLD = 0.5


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    predicted = p >= DECISION_THRESHOLD
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "precision_at_0.5": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_0.5": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_0.5": float(f1_score(y, predicted, zero_division=0)),
    }


def ranking_metrics(booking_id: np.ndarray, y: np.ndarray, p: np.ndarray, k: int = 5) -> dict[str, float]:
    """Hit@1 and NDCG@k within a booking, over bookings that have at least two labelled offers and one acceptance.

    Bookings without an accepted offer are excluded: no order can put an acceptance first, so they would only
    dilute the metric with a constant zero.
    """
    frame = pd.DataFrame({"booking_id": booking_id, "y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame = frame.sort_values(["booking_id", "p"], ascending=[True, False], kind="stable")
    frame["position"] = frame.groupby("booking_id").cumcount()

    by_booking = frame.groupby("booking_id")
    eligible = (by_booking["y"].transform("size") >= 2) & (by_booking["y"].transform("sum") >= 1)
    frame = frame[eligible]
    if frame.empty:
        return {"n_bookings": 0, "hit_at_1": float("nan"), f"ndcg_at_{k}": float("nan")}

    top = frame[frame["position"] == 0]
    discount = 1.0 / np.log2(frame["position"].to_numpy() + 2.0)
    frame = frame.assign(gain=np.where(frame["position"] < k, frame["y"].to_numpy() * discount, 0.0))
    dcg = frame.groupby("booking_id")["gain"].sum()
    ideal_discounts = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(k) + 2.0))])
    n_positive = frame.groupby("booking_id")["y"].sum().to_numpy().astype(int)
    idcg = ideal_discounts[np.minimum(n_positive, k)]
    return {
        "n_bookings": int(len(dcg)),
        "hit_at_1": float(top["y"].mean()),
        f"ndcg_at_{k}": float(np.mean(dcg.to_numpy() / idcg)),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability table over quantile bins of the predicted probability, plus the expected calibration error."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bin"] = pd.qcut(frame["p"], q=n_bins, duplicates="drop", labels=False)
    grouped = frame.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    gap = (grouped["observed"] - grouped["predicted"]).abs()
    return {
        "expected_calibration_error": float((gap * grouped["n"]).sum() / grouped["n"].sum()),
        "bins": [
            {"n": int(row.n), "mean_predicted": round(float(row.predicted), 6), "observed": round(float(row.observed), 6)}
            for row in grouped.itertuples()
        ],
    }


def calibration_by_eta(eta_min: np.ndarray, y: np.ndarray, p: np.ndarray) -> list[dict]:
    """Calibration per pickup-ETA band: far candidates are labelled mostly by the 20% exploration policy
    (docs/research.md §9.8.9), so that is where the model is expected to be least sure of itself."""
    frame = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    frame["bucket"] = pd.cut(np.asarray(eta_min, dtype=float), bins=ETA_BUCKET_EDGES, right=False)
    grouped = frame.groupby("bucket", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return [
        {
            "eta_min": f"[{row.Index.left:g}, {row.Index.right:g})",
            "n": int(row.n),
            "mean_predicted": round(float(row.predicted), 6),
            "observed": round(float(row.observed), 6),
        }
        for row in grouped.itertuples()
    ]
