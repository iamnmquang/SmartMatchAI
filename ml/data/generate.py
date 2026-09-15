"""Generate the SmartMatch AI synthetic dataset.

Run from the repository root:

    python -m ml.data.generate                                   # default size (ml/data/config.py)
    python -m ml.data.generate --n-bookings 20000 --n-drivers 5000 --seed 7

Outputs
    data/raw/     observable world: drivers, bookings, candidates (with logged offers), metadata.json
    data/oracle/  hidden truth for offline evaluation only: candidate_truth, driver_latents, metadata.json

The generation model and its assumptions are documented in data/README.md.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.behavior import BehaviorParams, acceptance_probability, cancellation_probability, sigmoid
from ml.data.config import GeneratorConfig
from ml.data.geo import haversine_km, offset_to_latlon

GENERATOR_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]

VEHICLE_TYPES = ["car_4", "car_7"]
PASSENGER_TYPES = ["individual", "group"]
TIME_OF_DAY = ["night", "morning_peak", "midday", "evening_peak", "evening"]
WEATHER = ["clear", "rain", "heavy_rain"]
TRAFFIC_LEVELS = ["low", "medium", "high"]
LOGGING_POLICIES = ["nearest", "explore"]

OBSERVABLE_TABLES = ("drivers", "bookings", "candidates")
HIDDEN_TABLES = ("candidate_truth", "driver_latents")
CANDIDATE_COLUMNS = [
    "booking_id", "driver_id", "driver_lat", "driver_lon", "distance_km", "estimated_eta_min",
    "idle_time_min", "logging_policy", "offer_rank", "accepted", "cancelled",
]
TRUTH_COLUMNS = ["booking_id", "driver_id", "p_accept", "u_accept", "p_cancel", "u_cancel"]

# ---- Context -----------------------------------------------------------------------------------
TIME_OF_DAY_BY_HOUR = np.array(
    ["night"] * 6 + ["morning_peak"] * 4 + ["midday"] * 6 + ["evening_peak"] * 4 + ["evening"] * 4
)
# Hour-to-hour weather transitions; rows = current state, columns = next state (order of WEATHER).
WEATHER_TRANSITION = np.array([
    [0.93, 0.06, 0.01],
    [0.25, 0.65, 0.10],
    [0.15, 0.45, 0.40],
])
TRAFFIC_SCORE_BY_TIME_OF_DAY = {"night": -1.5, "morning_peak": 1.0, "midday": 0.0, "evening_peak": 1.2, "evening": -0.3}
TRAFFIC_SCORE_BY_WEATHER = {"clear": 0.0, "rain": 0.5, "heavy_rain": 1.0}
TRAFFIC_THRESHOLDS = (-0.3, 1.3)  # score < first: low; < second: medium; otherwise high

# ---- Supply and movement ------------------------------------------------------------------------
SUPPLY_BY_TIME_OF_DAY = {"night": 0.6, "morning_peak": 0.7, "midday": 1.1, "evening_peak": 0.65, "evening": 0.9}
SUPPLY_BY_WEATHER = {"clear": 1.0, "rain": 0.8, "heavy_rain": 0.55}
SPEED_KMH_BY_TRAFFIC = {"low": 32.0, "medium": 22.0, "high": 14.0}
DETOUR_FACTOR_MEDIAN = 1.3  # road distance / straight-line distance
DETOUR_FACTOR_LOG_SD = 0.1
PICKUP_OVERHEAD_MIN = 1.0
IDLE_MEAN_MIN_BY_TIME_OF_DAY = {"night": 20.0, "morning_peak": 6.0, "midday": 12.0, "evening_peak": 6.0, "evening": 10.0}
CURRENT_IDLE_MEAN_MIN = 10.0
MAX_IDLE_MIN = 180.0

# ---- Driver population ---------------------------------------------------------------------------
NEW_DRIVER_SHARE = 0.15
TRIPS_LOG_MEAN, TRIPS_LOG_SD, MAX_TRIPS = 6.0, 1.1, 20_000
RATING_PRIOR, RATING_PRIOR_WEIGHT, MIN_TRIPS_FOR_RATING = 4.7, 20, 5


@dataclass
class SyntheticDataset:
    """Observable tables: drivers, bookings, candidates. Hidden tables: candidate_truth, driver_latents."""

    config: GeneratorConfig
    params: BehaviorParams
    drivers: pd.DataFrame
    bookings: pd.DataFrame
    candidates: pd.DataFrame
    candidate_truth: pd.DataFrame
    driver_latents: pd.DataFrame
    summary: dict
    oracle_summary: dict


def generate(cfg: GeneratorConfig, params: BehaviorParams = BehaviorParams()) -> SyntheticDataset:
    rngs = _stage_rngs(cfg.seed)
    drivers, latents = generate_drivers(cfg, rngs["drivers"])
    weather_by_hour = generate_weather(cfg, rngs["weather"])
    bookings, pickup_x, pickup_y = generate_bookings(cfg, rngs["bookings"], weather_by_hour)
    pairs = generate_candidates(
        cfg, rngs["candidates"], rngs["outcomes"], bookings, pickup_x, pickup_y, drivers, latents, params
    )
    pairs = simulate_logged_offers(cfg, rngs["logging"], pairs, len(bookings))
    candidates = pairs[CANDIDATE_COLUMNS].reset_index(drop=True)
    truth = pairs[TRUTH_COLUMNS].reset_index(drop=True)
    return SyntheticDataset(
        config=cfg,
        params=params,
        drivers=drivers,
        bookings=bookings,
        candidates=candidates,
        candidate_truth=truth,
        driver_latents=latents,
        summary=summarize(bookings, drivers, candidates),
        oracle_summary=summarize_oracle(candidates, truth),
    )


def _stage_rngs(seed: int) -> dict[str, np.random.Generator]:
    """One independent random stream per stage, so changing one stage does not reshuffle the others."""
    names = ("drivers", "weather", "bookings", "candidates", "outcomes", "logging")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children)}


def sample_city_points(cfg: GeneratorConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample points (km offsets) from a mixture of hotspots and a uniform background."""
    weights = np.array([h.weight for h in cfg.hotspots] + [cfg.background_weight])
    component = rng.choice(len(weights), size=n, p=weights / weights.sum())
    x = rng.uniform(-cfg.half_extent_km, cfg.half_extent_km, n)
    y = rng.uniform(-cfg.half_extent_km, cfg.half_extent_km, n)
    for i, hotspot in enumerate(cfg.hotspots):
        mask = component == i
        k = int(mask.sum())
        x[mask] = rng.normal(hotspot.x_km, hotspot.sigma_km, k)
        y[mask] = rng.normal(hotspot.y_km, hotspot.sigma_km, k)
    extent = cfg.half_extent_km
    return np.clip(x, -extent, extent), np.clip(y, -extent, extent)


def generate_drivers(cfg: GeneratorConfig, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = cfg.n_drivers
    vehicle_type = np.where(rng.random(n) < cfg.car7_share, "car_7", "car_4")
    is_new = rng.random(n) < NEW_DRIVER_SHARE
    experienced_trips = np.round(rng.lognormal(TRIPS_LOG_MEAN, TRIPS_LOG_SD, n))
    completed_trips = np.clip(np.where(is_new, rng.integers(0, 50, n), experienced_trips), 0, MAX_TRIPS).astype(np.int32)

    # Latent traits: never exported with the observable tables.
    quality = rng.normal(size=n)
    unreliability = -0.4 * quality + np.sqrt(1 - 0.4**2) * rng.normal(size=n)
    selectivity = rng.normal(size=n)

    # Historical stats are noisy estimates of the latent traits, computed before the simulated window.
    # Drivers with little history get noisier values, like real small-sample statistics.
    history_offers = np.maximum(10, np.round(completed_trips * 1.6)).astype(np.int64)
    acceptance_rate = rng.binomial(history_offers, sigmoid(0.6 - selectivity)) / history_offers
    history_accepted = np.maximum(5, completed_trips).astype(np.int64)
    cancellation_rate = rng.binomial(history_accepted, sigmoid(-2.6 + 0.8 * unreliability)) / history_accepted

    true_rating = np.clip(4.65 + 0.2 * quality, 3.5, 5.0)
    rating = (completed_trips * true_rating + RATING_PRIOR_WEIGHT * RATING_PRIOR) / (completed_trips + RATING_PRIOR_WEIGHT)
    rating = np.clip(rating + rng.normal(0.0, 0.03, n), 1.0, 5.0).round(2)
    rating = np.where(completed_trips < MIN_TRIPS_FOR_RATING, np.nan, rating)  # not rated yet

    x, y = sample_city_points(cfg, rng, n)
    current_lat, current_lon = _to_latlon(cfg, x, y)
    driver_id = np.arange(1, n + 1, dtype=np.int32)
    drivers = pd.DataFrame({
        "driver_id": driver_id,
        "vehicle_type": pd.Categorical(vehicle_type, categories=VEHICLE_TYPES),
        "rating": rating,
        "acceptance_rate": acceptance_rate.round(3),
        "cancellation_rate": cancellation_rate.round(3),
        "completed_trips": completed_trips,
        "idle_time_min": np.minimum(rng.exponential(CURRENT_IDLE_MEAN_MIN, n), MAX_IDLE_MIN).round(1),
        "current_lat": current_lat,
        "current_lon": current_lon,
    })
    latents = pd.DataFrame({
        "driver_id": driver_id,
        "selectivity": selectivity,
        "quality": quality,
        "unreliability": unreliability,
    })
    return drivers, latents


def generate_weather(cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """City-wide weather for every hour of the window, as a 3-state Markov chain."""
    n_hours = cfg.n_days * 24
    cumulative = WEATHER_TRANSITION.cumsum(axis=1)
    draws = rng.random(n_hours)
    states = np.zeros(n_hours, dtype=np.int64)
    for t in range(1, n_hours):
        states[t] = min(int(np.searchsorted(cumulative[states[t - 1]], draws[t], side="right")), len(WEATHER) - 1)
    return np.array(WEATHER)[states]


def generate_bookings(
    cfg: GeneratorConfig, rng: np.random.Generator, weather_by_hour: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n = cfg.n_bookings
    start = pd.Timestamp(cfg.start_date)
    is_weekend = (start.dayofweek + np.arange(cfg.n_days)) % 7 >= 5
    day_weights = np.where(is_weekend, cfg.weekend_demand_factor, 1.0)
    day = rng.choice(cfg.n_days, size=n, p=day_weights / day_weights.sum())
    hour_weights = np.asarray(cfg.hourly_demand)
    hour = rng.choice(24, size=n, p=hour_weights / hour_weights.sum())
    offset_s = day * 86_400 + hour * 3_600 + rng.integers(0, 3_600, size=n)

    pickup_x, pickup_y = sample_city_points(cfg, rng, n)
    dest_x, dest_y = sample_city_points(cfg, rng, n)
    for _ in range(100):
        too_short = np.hypot(dest_x - pickup_x, dest_y - pickup_y) < cfg.min_trip_km
        if not too_short.any():
            break
        dest_x[too_short], dest_y[too_short] = sample_city_points(cfg, rng, int(too_short.sum()))

    time_of_day = TIME_OF_DAY_BY_HOUR[hour]
    weather = weather_by_hour[day * 24 + hour]
    centrality = np.exp(-((np.hypot(pickup_x, pickup_y) / 6.0) ** 2))
    traffic_score = (
        _lookup(time_of_day, TRAFFIC_SCORE_BY_TIME_OF_DAY)
        + _lookup(weather, TRAFFIC_SCORE_BY_WEATHER)
        + 0.6 * centrality
        + rng.normal(0.0, 0.7, n)
    )
    low_max, medium_max = TRAFFIC_THRESHOLDS
    traffic = np.select([traffic_score < low_max, traffic_score < medium_max], ["low", "medium"], default="high")
    passenger_type = np.where(rng.random(n) < cfg.group_booking_share, "group", "individual")

    order = np.argsort(offset_s, kind="stable")  # booking_id increases with request time
    pickup_x, pickup_y, dest_x, dest_y = pickup_x[order], pickup_y[order], dest_x[order], dest_y[order]
    pickup_lat, pickup_lon = _to_latlon(cfg, pickup_x, pickup_y)
    destination_lat, destination_lon = _to_latlon(cfg, dest_x, dest_y)
    bookings = pd.DataFrame({
        "booking_id": np.arange(1, n + 1, dtype=np.int32),
        "request_time": start + pd.to_timedelta(offset_s[order], unit="s"),
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "destination_lat": destination_lat,
        "destination_lon": destination_lon,
        "passenger_type": pd.Categorical(passenger_type[order], categories=PASSENGER_TYPES),
        "traffic_level": pd.Categorical(traffic[order], categories=TRAFFIC_LEVELS),
        "weather": pd.Categorical(weather[order], categories=WEATHER),
        "time_of_day": pd.Categorical(time_of_day[order], categories=TIME_OF_DAY),
    })
    return bookings, pickup_x, pickup_y


def generate_candidates(
    cfg: GeneratorConfig,
    rng: np.random.Generator,
    outcome_rng: np.random.Generator,
    bookings: pd.DataFrame,
    pickup_x: np.ndarray,
    pickup_y: np.ndarray,
    drivers: pd.DataFrame,
    latents: pd.DataFrame,
    params: BehaviorParams,
) -> pd.DataFrame:
    """Snapshot of available, compatible drivers inside the search radius of each booking, plus hidden truth."""
    n_bookings = len(bookings)
    time_of_day = _as_str(bookings["time_of_day"])
    weather = _as_str(bookings["weather"])
    traffic = _as_str(bookings["traffic_level"])
    is_group = (bookings["passenger_type"] == "group").to_numpy()
    pickup_lat = bookings["pickup_lat"].to_numpy()
    pickup_lon = bookings["pickup_lon"].to_numpy()
    trip_km = haversine_km(pickup_lat, pickup_lon, bookings["destination_lat"], bookings["destination_lon"])

    # How many available drivers are nearby depends on time, weather and how central the pickup is.
    centrality = 0.6 + 0.6 * np.exp(-((np.hypot(pickup_x, pickup_y) / 7.0) ** 2))
    expected_nearby = (
        cfg.mean_nearby_drivers
        * _lookup(time_of_day, SUPPLY_BY_TIME_OF_DAY)
        * _lookup(weather, SUPPLY_BY_WEATHER)
        * centrality
    )
    nearby = rng.poisson(expected_nearby)
    compatible = np.where(is_group, rng.binomial(nearby, cfg.car7_share), nearby)
    b = np.repeat(np.arange(n_bookings), np.minimum(compatible, cfg.max_candidates))

    car7 = np.flatnonzero((drivers["vehicle_type"] == "car_7").to_numpy())
    if len(car7) == 0:
        raise ValueError("no car_7 drivers: increase n_drivers or car7_share")
    d = rng.integers(0, len(drivers), size=len(b))
    group_rows = is_group[b]
    d[group_rows] = car7[rng.integers(0, len(car7), size=int(group_rows.sum()))]
    unique = ~pd.DataFrame({"b": b, "d": d}).duplicated().to_numpy()
    b, d = b[unique], d[unique]

    # Uniform position inside the search disc around the pickup.
    radius = cfg.search_radius_km * np.sqrt(rng.random(len(b)))
    angle = rng.uniform(0.0, 2.0 * np.pi, len(b))
    driver_lat, driver_lon = _to_latlon(cfg, pickup_x[b] + radius * np.cos(angle), pickup_y[b] + radius * np.sin(angle))
    distance_km = haversine_km(pickup_lat[b], pickup_lon[b], driver_lat, driver_lon).round(3)
    inside = distance_km <= cfg.search_radius_km
    b, d, driver_lat, driver_lon, distance_km = (a[inside] for a in (b, d, driver_lat, driver_lon, distance_km))
    m = len(b)

    tod_c, weather_c, traffic_c = time_of_day[b], weather[b], traffic[b]
    detour = DETOUR_FACTOR_MEDIAN * np.exp(rng.normal(0.0, DETOUR_FACTOR_LOG_SD, m))
    speed_kmh = _lookup(traffic_c, SPEED_KMH_BY_TRAFFIC)
    eta_min = (PICKUP_OVERHEAD_MIN + distance_km * detour / speed_kmh * 60.0).round(2)
    idle_min = np.minimum(rng.exponential(_lookup(tod_c, IDLE_MEAN_MIN_BY_TIME_OF_DAY)), MAX_IDLE_MIN).round(1)

    p_accept = acceptance_probability(
        params,
        eta_min=eta_min,
        pickup_km=distance_km,
        trip_km=trip_km[b],
        idle_min=idle_min,
        weather=weather_c,
        traffic=traffic_c,
        time_of_day=tod_c,
        is_group=is_group[b],
        completed_trips=drivers["completed_trips"].to_numpy()[d],
        selectivity=latents["selectivity"].to_numpy()[d],
        quality=latents["quality"].to_numpy()[d],
        noise=outcome_rng.normal(0.0, params.accept_noise_sd, m),
    )
    p_cancel = cancellation_probability(
        params,
        eta_min=eta_min,
        weather=weather_c,
        time_of_day=tod_c,
        unreliability=latents["unreliability"].to_numpy()[d],
        noise=outcome_rng.normal(0.0, params.cancel_noise_sd, m),
    )
    return pd.DataFrame({
        "booking_idx": b,
        "booking_id": bookings["booking_id"].to_numpy()[b],
        "driver_id": drivers["driver_id"].to_numpy()[d],
        "driver_lat": driver_lat,
        "driver_lon": driver_lon,
        "distance_km": distance_km,
        "estimated_eta_min": eta_min,
        "idle_time_min": idle_min,
        # Common random numbers: the driver accepts iff u_accept < p_accept, for every policy.
        "p_accept": p_accept,
        "u_accept": outcome_rng.random(m),
        "p_cancel": p_cancel,
        "u_cancel": outcome_rng.random(m),
    })


def simulate_logged_offers(
    cfg: GeneratorConfig, rng: np.random.Generator, pairs: pd.DataFrame, n_bookings: int
) -> pd.DataFrame:
    """Replay the historical dispatch that produced the training log.

    Logging policy: per booking, with probability 1 - exploration_rate offer the nearest candidate first,
    otherwise offer in random order. Offers go out one at a time until a driver accepts or max_offers is
    reached. Only offered rows get labels; the rest stay <NA> - exactly like a real log.
    """
    b = pairs["booking_idx"].to_numpy()
    explore = (rng.random(n_bookings) < cfg.exploration_rate)[b]
    sort_key = np.where(explore, rng.random(len(b)), pairs["distance_km"].to_numpy())
    order = np.lexsort((pairs["driver_id"].to_numpy(), sort_key, b))
    sorted_b = b[order]
    rank = np.empty(len(b), dtype=np.int64)
    rank[order] = np.arange(len(b)) - np.searchsorted(sorted_b, sorted_b, side="left") + 1

    would_accept = pairs["u_accept"].to_numpy() < pairs["p_accept"].to_numpy()
    would_cancel = pairs["u_cancel"].to_numpy() < pairs["p_cancel"].to_numpy()
    no_accept = cfg.max_offers + 1
    first_accept = np.full(n_bookings, no_accept)
    np.minimum.at(first_accept, b, np.where(would_accept & (rank <= cfg.max_offers), rank, no_accept))
    offered = rank <= np.minimum(first_accept, cfg.max_offers)[b]

    out = pairs.copy()
    out["logging_policy"] = pd.Categorical(np.where(explore, "explore", "nearest"), categories=LOGGING_POLICIES)
    out["offer_rank"] = pd.Series(rank, index=out.index).astype("Int8").where(offered)
    out["accepted"] = pd.Series(would_accept, index=out.index, dtype="boolean").where(offered)
    out["cancelled"] = pd.Series(would_cancel, index=out.index, dtype="boolean").where(offered & would_accept)
    return out.sort_values(["booking_id", "driver_id"], ignore_index=True)


def summarize(bookings: pd.DataFrame, drivers: pd.DataFrame, candidates: pd.DataFrame) -> dict:
    """Descriptive stats of the observable data and of the logging policy (not model results)."""
    n_bookings = len(bookings)
    per_booking = candidates.groupby("booking_id").size().reindex(bookings["booking_id"], fill_value=0)
    offers = candidates[candidates["offer_rank"].notna()]
    accepted = offers[offers["accepted"].to_numpy(dtype=bool)]
    logged_policy = offers.drop_duplicates("booking_id")["logging_policy"]
    return {
        "rows": {
            "drivers": len(drivers),
            "bookings": n_bookings,
            "candidates": len(candidates),
            "logged_offers": len(offers),
        },
        "candidates_per_booking": {
            "mean": _r(per_booking.mean()),
            "p50": _r(per_booking.quantile(0.50)),
            "p95": _r(per_booking.quantile(0.95)),
            "max": int(per_booking.max()),
        },
        "bookings_without_candidates_pct": _r(100 * (per_booking == 0).mean()),
        "driver_rating_missing_pct": _r(100 * drivers["rating"].isna().mean()),
        "explore_share_of_logged_bookings_pct": _r(100 * (logged_policy == "explore").mean()),
        "logging_policy_metrics": {
            "offer_acceptance_rate": _r(len(accepted) / len(offers)),
            "matching_success_rate": _r(accepted["booking_id"].nunique() / n_bookings),
            "avg_eta_matched_min": _r(accepted["estimated_eta_min"].mean()),
            "cancellation_rate": _r(accepted["cancelled"].to_numpy(dtype=bool).mean()),
            "avg_offers_per_booking_with_candidates": _r(len(offers) / int((per_booking > 0).sum())),
        },
        "context_mix_pct": {
            column: {str(k): _r(100 * v) for k, v in bookings[column].value_counts(normalize=True, sort=False).items()}
            for column in ("time_of_day", "weather", "traffic_level", "passenger_type")
        },
    }


def summarize_oracle(candidates: pd.DataFrame, truth: pd.DataFrame) -> dict:
    offered = candidates["offer_rank"].notna().to_numpy()
    accepted = candidates["accepted"].fillna(False).to_numpy(dtype=bool)
    cancelled = candidates["cancelled"].fillna(False).to_numpy(dtype=bool)
    p_accept = truth["p_accept"].to_numpy()
    p_cancel = truth["p_cancel"].to_numpy()
    return {
        "mean_p_accept_all_candidates": _r(p_accept.mean()),
        "mean_p_accept_logged_offers": _r(p_accept[offered].mean()),
        # Upper bound for any model: the oracle also knows latent traits and per-offer noise.
        "oracle_auc_accept_logged_offers": _r(roc_auc(accepted[offered], p_accept[offered])),
        "mean_p_cancel_accepted": _r(p_cancel[accepted].mean()),
        "oracle_auc_cancel_accepted": _r(roc_auc(cancelled[accepted], p_cancel[accepted])),
    }


def roc_auc(y_true, score) -> float:
    """ROC-AUC via the Mann-Whitney rank statistic (ties get average ranks)."""
    y = np.asarray(y_true, dtype=bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(score, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def write_dataset(ds: SyntheticDataset, raw_dir: Path, oracle_dir: Path) -> None:
    raw_dir, oracle_dir = Path(raw_dir), Path(oracle_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    for name in OBSERVABLE_TABLES:
        getattr(ds, name).to_parquet(raw_dir / f"{name}.parquet", index=False)
    for name in HIDDEN_TABLES:
        getattr(ds, name).to_parquet(oracle_dir / f"{name}.parquet", index=False)
    _write_json(raw_dir / "metadata.json", {
        "generator_version": GENERATOR_VERSION,
        "config": asdict(ds.config),
        "tables": _table_info(ds, OBSERVABLE_TABLES),
        "summary": ds.summary,
    })
    _write_json(oracle_dir / "metadata.json", {
        "warning": "Hidden truth of the simulator. Only offline evaluation may read this directory; "
                   "never use it for training, feature engineering or serving.",
        "generator_version": GENERATOR_VERSION,
        "behavior_params": asdict(ds.params),
        "tables": _table_info(ds, HIDDEN_TABLES),
        "summary": ds.oracle_summary,
    })


def _table_info(ds: SyntheticDataset, names: tuple[str, ...]) -> dict:
    return {
        name: {"rows": len(df), "columns": {col: str(dtype) for col, dtype in df.dtypes.items()}}
        for name in names
        for df in [getattr(ds, name)]
    }


def _write_json(path: Path, payload: dict) -> None:
    # newline="\n": identical bytes on every OS (Windows would otherwise write CRLF).
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _to_latlon(cfg: GeneratorConfig, x_km, y_km) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = offset_to_latlon(cfg.center_lat, cfg.center_lon, x_km, y_km)
    return np.round(lat, 6), np.round(lon, 6)


def _lookup(values: np.ndarray, table: dict[str, float]) -> np.ndarray:
    return pd.Series(values).map(table).to_numpy(dtype=float)


def _as_str(series: pd.Series) -> np.ndarray:
    return np.asarray(series.astype(object), dtype=str)


def _r(value, digits: int = 4) -> float:
    return round(float(value), digits)


def main(argv: list[str] | None = None) -> None:
    defaults = GeneratorConfig()
    parser = argparse.ArgumentParser(description="Generate the SmartMatch AI synthetic dataset.")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--n-drivers", type=int, default=defaults.n_drivers)
    parser.add_argument("--n-bookings", type=int, default=defaults.n_bookings)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--oracle-dir", type=Path, default=REPO_ROOT / "data" / "oracle")
    args = parser.parse_args(argv)

    cfg = replace(defaults, seed=args.seed, n_drivers=args.n_drivers, n_bookings=args.n_bookings)
    started = time.perf_counter()
    ds = generate(cfg)
    write_dataset(ds, args.raw_dir, args.oracle_dir)
    print(json.dumps(ds.summary, indent=2))
    print("oracle (evaluation only):", json.dumps(ds.oracle_summary))
    print(f"Wrote {args.raw_dir} and {args.oracle_dir} in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
