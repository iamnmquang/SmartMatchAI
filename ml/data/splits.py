"""Time-based train / validation / test split at booking level.

Candidates inherit the split of their booking, so all candidates of one booking stay together (no group
leakage) and models are always evaluated on bookings that happen after the ones they learned from (no
temporal leakage). Shared by EDA (Phase 3 looks at the train split only) and training (Phase 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.config import GeneratorConfig

SPLIT_NAMES = ["train", "validation", "test"]
DEFAULT_SPLIT_DAYS = (40, 8, 8)  # sums to the default GeneratorConfig.n_days


def split_edges(
    start_date: str = GeneratorConfig.start_date, days: tuple[int, ...] = DEFAULT_SPLIT_DAYS
) -> list[pd.Timestamp]:
    """Boundaries [train_start, validation_start, test_start, test_end); every interval is left-closed."""
    start = pd.Timestamp(start_date)
    return [start + pd.Timedelta(days=int(offset)) for offset in np.cumsum((0, *days))]


def assign_split(
    request_time: pd.Series, start_date: str = GeneratorConfig.start_date, days: tuple[int, ...] = DEFAULT_SPLIT_DAYS
) -> pd.Series:
    labels = pd.cut(request_time, bins=split_edges(start_date, days), labels=SPLIT_NAMES, right=False)
    if labels.isna().any():
        raise ValueError("some request_time values fall outside the split window")
    return labels
