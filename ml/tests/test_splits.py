import pandas as pd
import pytest

from ml.data.config import GeneratorConfig
from ml.data.generate import generate
from ml.data.splits import assign_split, split_edges


def test_edges_cover_the_default_generation_window():
    edges = split_edges()
    config = GeneratorConfig()
    assert edges[0] == pd.Timestamp(config.start_date)
    assert edges[-1] == pd.Timestamp(config.start_date) + pd.Timedelta(days=config.n_days)


def test_boundaries_are_left_closed():
    times = pd.Series(pd.to_datetime([
        "2026-06-01 00:00:00", "2026-07-10 23:59:59",
        "2026-07-11 00:00:00", "2026-07-18 23:59:59",
        "2026-07-19 00:00:00", "2026-07-26 23:59:59",
    ]))
    assert list(assign_split(times)) == ["train", "train", "validation", "validation", "test", "test"]


def test_out_of_window_timestamps_are_rejected():
    with pytest.raises(ValueError):
        assign_split(pd.Series(pd.to_datetime(["2026-07-27 00:00:00"])))


def test_generated_bookings_split_in_time_order():
    bookings = generate(GeneratorConfig(seed=3, n_drivers=300, n_bookings=2_000)).bookings
    split = assign_split(bookings["request_time"])
    times = bookings["request_time"]
    assert times[split == "train"].max() < times[split == "validation"].min()
    assert times[split == "validation"].max() < times[split == "test"].min()
    assert 0.6 < (split == "train").mean() < 0.8
