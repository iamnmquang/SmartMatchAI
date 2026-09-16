"""Training data for the acceptance model: the logged offers of one split (Phase 5).

A training row is one offer that actually went out and therefore has a label. Rows are assembled through the
same decision-time view and the same feature code that online serving will use (ml/features), so the model
cannot learn from anything a live request would not have.

Two details that are easy to get wrong:

* Relative features are computed over the *whole* candidate set of a booking and only then restricted to the
  labelled rows. Computing "how much further than the closest candidate" over the two drivers that happened to
  be offered would give a different value than serving, where every candidate is present.
* Labels are structurally missing (80.6% of candidate rows were never offered, docs/research.md §9.1). Rows
  without a label are dropped, not filled: they are unobserved, not negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.splits import assign_split
from ml.features import build_decision_time_view, build_features

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class RawData:
    bookings: pd.DataFrame
    candidates: pd.DataFrame
    drivers: pd.DataFrame


@dataclass(frozen=True)
class LabeledSplit:
    """Logged offers of one split, ready for scikit-learn / XGBoost."""

    name: str
    view: pd.DataFrame  # decision-time columns, for diagnostics (ETA buckets, per-booking grouping)
    features: pd.DataFrame  # model input matrix, columns in ml.features.FEATURE_COLUMNS order
    y: np.ndarray  # bool: the driver accepted the offer
    booking_id: np.ndarray

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_bookings(self) -> int:
        return int(pd.unique(self.booking_id).size)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())


def load_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> RawData:
    return RawData(
        bookings=pd.read_parquet(raw_dir / "bookings.parquet"),
        candidates=pd.read_parquet(raw_dir / "candidates.parquet"),
        drivers=pd.read_parquet(raw_dir / "drivers.parquet"),
    )


def candidate_features(raw: RawData, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Every candidate row of a split: the decision-time view, its features and the (mostly missing) labels."""
    split_bookings = raw.bookings[assign_split(raw.bookings["request_time"]) == split].reset_index(drop=True)
    in_split = raw.candidates["booking_id"].isin(split_bookings["booking_id"]).to_numpy()
    candidates = raw.candidates.loc[in_split].reset_index(drop=True)
    view = build_decision_time_view(split_bookings, candidates, raw.drivers)
    return view, build_features(view), candidates["accepted"]


def labeled_offers(raw: RawData, split: str) -> LabeledSplit:
    view, features, labels = candidate_features(raw, split)
    labeled = labels.notna().to_numpy()
    return LabeledSplit(
        name=split,
        view=view.loc[labeled].reset_index(drop=True),
        features=features.loc[labeled].reset_index(drop=True),
        y=labels[labeled].to_numpy(dtype=bool),
        booking_id=view.loc[labeled, "booking_id"].to_numpy(),
    )


def load_labeled_splits(splits: tuple[str, ...], raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, LabeledSplit]:
    raw = load_raw(raw_dir)
    return {split: labeled_offers(raw, split) for split in splits}
