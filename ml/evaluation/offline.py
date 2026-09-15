"""Offline evaluation of ranking policies with the simulator as oracle (ADR-005, docs/evaluation.md).

Every policy is replayed on the same bookings, the same candidate sets and the same common random numbers
(a driver accepts iff u_accept < p_accept), so metric differences come only from the order of the offers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.splits import assign_split

REPO_ROOT = Path(__file__).resolve().parents[2]

# What a policy may see at decision time. Log artifacts (offer_rank, logging_policy) and labels are excluded.
DECISION_TIME_CANDIDATE_COLUMNS = [
    "booking_id", "driver_id", "driver_lat", "driver_lon", "distance_km", "estimated_eta_min", "idle_time_min",
]
BOOKING_CONTEXT_COLUMNS = [
    "booking_id", "request_time", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon",
    "passenger_type", "traffic_level", "weather", "time_of_day",
]
DRIVER_PROFILE_COLUMNS = ["driver_id", "vehicle_type", "rating", "acceptance_rate", "cancellation_rate", "completed_trips"]
TRUTH_COLUMNS = ["booking_id", "driver_id", "p_accept", "u_accept", "p_cancel", "u_cancel"]

# Every metric is a ratio of per-booking sums, which makes point estimates and bootstrap resamples consistent.
METRIC_RATIOS = {
    "matching_success_rate": ("matched", "bookings"),
    "acceptance_rate": ("matched", "offers"),  # each matched booking has exactly one accepted offer
    "avg_eta_matched_min": ("matched_eta", "matched"),
    "cancellation_rate": ("cancelled", "matched"),
    "completed_rate": ("completed", "bookings"),
    "first_offer_acceptance_rate": ("first_offer_accepted", "with_candidates"),
    "avg_offers_per_booking_with_candidates": ("offers", "with_candidates"),
}


@dataclass
class EvaluationData:
    bookings: pd.DataFrame  # every booking of the split, including bookings without candidates
    candidates: pd.DataFrame  # decision-time view: candidate + booking context + driver profile
    truth: pd.DataFrame  # hidden truth, aligned row by row with candidates


def build_evaluation_data(
    bookings: pd.DataFrame, candidates: pd.DataFrame, drivers: pd.DataFrame, truth: pd.DataFrame, split: str
) -> EvaluationData:
    if not np.array_equal(candidates[["booking_id", "driver_id"]].to_numpy(), truth[["booking_id", "driver_id"]].to_numpy()):
        raise ValueError("candidate_truth is not aligned with candidates")
    split_bookings = bookings[assign_split(bookings["request_time"]) == split].reset_index(drop=True)
    in_split = candidates["booking_id"].isin(split_bookings["booking_id"]).to_numpy()
    view = (
        candidates.loc[in_split, DECISION_TIME_CANDIDATE_COLUMNS]
        .merge(split_bookings[BOOKING_CONTEXT_COLUMNS], on="booking_id", how="left")
        .merge(drivers[DRIVER_PROFILE_COLUMNS], on="driver_id", how="left")
        .reset_index(drop=True)
    )
    split_truth = truth.loc[in_split, TRUTH_COLUMNS].reset_index(drop=True)
    if not np.array_equal(view[["booking_id", "driver_id"]].to_numpy(), split_truth[["booking_id", "driver_id"]].to_numpy()):
        raise ValueError("decision-time view lost alignment with the truth table")
    return EvaluationData(bookings=split_bookings, candidates=view, truth=split_truth)


def load_split(
    split: str, raw_dir: Path = REPO_ROOT / "data" / "raw", oracle_dir: Path = REPO_ROOT / "data" / "oracle"
) -> EvaluationData:
    return build_evaluation_data(
        bookings=pd.read_parquet(raw_dir / "bookings.parquet"),
        candidates=pd.read_parquet(raw_dir / "candidates.parquet"),
        drivers=pd.read_parquet(raw_dir / "drivers.parquet"),
        truth=pd.read_parquet(oracle_dir / "candidate_truth.parquet"),
        split=split,
    )


def oracle_scores(truth: pd.DataFrame) -> np.ndarray:
    """Upper bound, not attainable in reality: order by the simulator's true acceptance probability."""
    return truth["p_accept"].to_numpy(dtype=float)


def simulate_offers(data: EvaluationData, scores, max_offers: int = 3) -> pd.DataFrame:
    """Replay sequential dispatch for one policy; returns one row per booking of the split.

    Offers go out in descending score order (ties by driver_id) until a driver accepts or max_offers is reached.
    """
    view, truth = data.candidates, data.truth
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(view),) or not np.isfinite(scores).all():
        raise ValueError("a policy must return one finite score per candidate row")

    booking = view["booking_id"].to_numpy()
    order = np.lexsort((view["driver_id"].to_numpy(), -scores, booking))
    sorted_booking = booking[order]
    ranked = pd.DataFrame({
        "booking_id": sorted_booking,
        "rank": np.arange(len(order)) - np.searchsorted(sorted_booking, sorted_booking, side="left") + 1,
        "driver_id": view["driver_id"].to_numpy()[order],
        "eta": view["estimated_eta_min"].to_numpy(dtype=float)[order],
        "accept": (truth["u_accept"].to_numpy() < truth["p_accept"].to_numpy())[order],
        "cancel": (truth["u_cancel"].to_numpy() < truth["p_cancel"].to_numpy())[order],
    })
    offered = ranked[ranked["rank"] <= max_offers]
    hits = offered[offered["accept"]].drop_duplicates("booking_id").set_index("booking_id")  # first acceptance

    outcomes = pd.DataFrame(index=pd.Index(data.bookings["booking_id"].to_numpy(), name="booking_id"))
    outcomes["n_candidates"] = ranked.groupby("booking_id").size().reindex(outcomes.index, fill_value=0).astype(int)
    outcomes["matched"] = outcomes.index.isin(hits.index)
    outcomes["n_offers"] = np.minimum(outcomes["n_candidates"], max_offers)
    outcomes.loc[hits.index, "n_offers"] = hits["rank"].to_numpy()
    outcomes["matched_driver_id"] = pd.Series(pd.NA, index=outcomes.index, dtype="Int32")
    outcomes.loc[hits.index, "matched_driver_id"] = hits["driver_id"].to_numpy()
    outcomes["matched_eta_min"] = np.nan
    outcomes.loc[hits.index, "matched_eta_min"] = hits["eta"].to_numpy()
    outcomes["cancelled"] = pd.Series(pd.NA, index=outcomes.index, dtype="boolean")
    outcomes.loc[hits.index, "cancelled"] = hits["cancel"].to_numpy()
    outcomes["first_offer_accepted"] = outcomes["matched"] & (outcomes["n_offers"] == 1)
    return outcomes.reset_index()


def _components(outcomes: pd.DataFrame) -> dict[str, np.ndarray]:
    matched = outcomes["matched"].to_numpy(dtype=bool)
    cancelled = outcomes["cancelled"].fillna(False).to_numpy(dtype=bool)
    return {
        "bookings": np.ones(len(outcomes)),
        "matched": matched.astype(float),
        "offers": outcomes["n_offers"].to_numpy(dtype=float),
        "matched_eta": np.where(matched, outcomes["matched_eta_min"].fillna(0.0).to_numpy(dtype=float), 0.0),
        "cancelled": (matched & cancelled).astype(float),
        "completed": (matched & ~cancelled).astype(float),
        "first_offer_accepted": outcomes["first_offer_accepted"].to_numpy(dtype=float),
        "with_candidates": (outcomes["n_candidates"].to_numpy() > 0).astype(float),
    }


def business_metrics(outcomes: pd.DataFrame) -> dict[str, float]:
    components = _components(outcomes)
    metrics = {
        name: _ratio(components[numerator].sum(), components[denominator].sum())
        for name, (numerator, denominator) in METRIC_RATIOS.items()
    }
    matched_eta = outcomes.loc[outcomes["matched"], "matched_eta_min"]
    metrics["p50_eta_matched_min"] = float(matched_eta.quantile(0.50)) if len(matched_eta) else float("nan")
    metrics["p90_eta_matched_min"] = float(matched_eta.quantile(0.90)) if len(matched_eta) else float("nan")
    return metrics


def paired_bootstrap(
    outcomes: dict[str, pd.DataFrame], reference: str, n_resamples: int = 1000, seed: int = 2026, chunk: int = 100
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """95% percentile intervals for every policy's metrics and for its difference to the reference policy.

    Bookings are resampled with replacement and every policy is measured on the same resample (paired bootstrap),
    so the interval of a difference reflects the uncertainty of the comparison, not of each policy separately.
    """
    booking_ids = outcomes[reference]["booking_id"].to_numpy()
    for name, frame in outcomes.items():
        if not np.array_equal(frame["booking_id"].to_numpy(), booking_ids):
            raise ValueError(f"outcomes of {name!r} are not aligned with the reference")
    components = {name: _components(frame) for name, frame in outcomes.items()}
    samples = {name: {metric: [] for metric in METRIC_RATIOS} for name in outcomes}
    rng = np.random.default_rng(seed)
    n = len(booking_ids)
    for start in range(0, n_resamples, chunk):
        index = rng.integers(0, n, size=(min(chunk, n_resamples - start), n))
        for name, parts in components.items():
            sums = {key: values[index].sum(axis=1) for key, values in parts.items()}
            for metric, (numerator, denominator) in METRIC_RATIOS.items():
                with np.errstate(divide="ignore", invalid="ignore"):
                    samples[name][metric].append(sums[numerator] / sums[denominator])

    intervals = {}
    for name in outcomes:
        intervals[name] = {}
        for metric in METRIC_RATIOS:
            values = np.concatenate(samples[name][metric])
            difference = values - np.concatenate(samples[reference][metric])
            intervals[name][metric] = {"ci95": _interval(values), "diff_vs_reference_ci95": _interval(difference)}
    return intervals


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _interval(values: np.ndarray) -> list[float]:
    return [round(float(np.nanpercentile(values, 2.5)), 6), round(float(np.nanpercentile(values, 97.5)), 6)]
