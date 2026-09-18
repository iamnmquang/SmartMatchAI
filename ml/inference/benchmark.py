"""Measure how long it takes to rank one booking (Phase 7, preview of Phase 17/20).

    python -m ml.inference.benchmark                # 10, 20 and 50 candidates
    python -m ml.inference.benchmark --repeats 500

Requests are taken from the generated test split, so candidate counts, coordinates and driver profiles are
realistic rather than synthetic round numbers. Each stage is timed separately, because they scale differently
and Phase 9 will need to know which one to worry about:

* **build_view** - dicts to a frame: cost per candidate, and the price of accepting JSON rather than a frame.
* **build_features** - the shared feature code; pandas overhead dominates at these sizes.
* **predict** - one matrix product against 28 coefficients.
* **explain** - contributions for the Top-K only.

Sizes are timed **interleaved**, one request of each size per round, not one size after another: measured
first, a size came out about twice as fast, because the CPU is still boosting early in the process. Round-robin
gives every size the same machine conditions. Absolute numbers stay machine-specific, so the committed file
records the machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import pandas as pd

from ml.evaluation.offline import REPO_ROOT, load_split
from ml.features import build_features
from ml.inference.predict import MODEL_DIR, build_view, explain, get_ranker
from ml.inference.ranking import apply_eta_constraint, offer_order

RESULTS_PATH = REPO_ROOT / "ml" / "inference" / "results" / "latency.json"
CANDIDATE_COUNTS = [10, 20, 50]
TOP_K = 5


def sample_requests(candidate_counts: list[int], n_requests: int) -> tuple[dict[int, list[dict]], list[int]]:
    """One request per booking, sized to exactly `count` candidates so the sizes are comparable.

    The busiest booking of the test split has 28 candidates, so a 50-candidate request cannot be real. Those
    sizes are padded by repeating candidates of the same booking under new driver ids: latency depends on the
    number of rows, not on their values, and the padded sizes are flagged in the results so the number is not
    read as a measurement of real traffic.
    """
    view = load_split("test").candidates
    sizes = view.groupby("booking_id").size()
    largest = int(sizes.max())
    requests: dict[int, list[dict]] = {}
    padded = []
    for count in candidate_counts:
        available = sizes[sizes >= min(count, largest)].index[:n_requests]
        requests[count] = [
            to_request(view[view["booking_id"] == booking_id].head(count), count) for booking_id in available
        ]
        if count > largest:
            padded.append(count)
    return requests, padded


def to_request(rows: pd.DataFrame, count: int) -> dict:
    first = rows.iloc[0]
    booking = {
        "booking_id": int(first["booking_id"]),
        "request_time": str(first["request_time"]),
        "pickup_lat": float(first["pickup_lat"]), "pickup_lon": float(first["pickup_lon"]),
        "destination_lat": float(first["destination_lat"]), "destination_lon": float(first["destination_lon"]),
        "passenger_type": str(first["passenger_type"]), "traffic_level": str(first["traffic_level"]),
        "weather": str(first["weather"]),
    }
    drivers = [
        {
            "driver_id": int(row.driver_id),
            "driver_lat": float(row.driver_lat), "driver_lon": float(row.driver_lon),
            "distance_km": float(row.distance_km),  # the matching module computes it once; serving reuses it
            "estimated_eta_min": float(row.estimated_eta_min), "idle_time_min": float(row.idle_time_min),
            "vehicle_type": str(row.vehicle_type),
            "rating": None if pd.isna(row.rating) else float(row.rating),
            "acceptance_rate": float(row.acceptance_rate), "cancellation_rate": float(row.cancellation_rate),
            "completed_trips": int(row.completed_trips),
        }
        for row in rows.itertuples()
    ]
    while len(drivers) < count:  # padding for sizes no booking of the split reaches
        clone = dict(drivers[len(drivers) % len(rows)])
        clone["driver_id"] = max(driver["driver_id"] for driver in drivers) + 1
        drivers.append(clone)
    return {"booking": booking, "candidate_drivers": drivers}


def time_stages(ranker, request: dict) -> dict[str, float]:
    """Milliseconds per stage for one request, using the same calls ranker.rank makes."""
    timings = {}
    start = time.perf_counter()
    view = build_view(request["booking"], request["candidate_drivers"])
    timings["build_view"] = time.perf_counter() - start

    start = time.perf_counter()
    features = build_features(view)
    timings["build_features"] = time.perf_counter() - start

    start = time.perf_counter()
    probabilities = ranker.model.predict_proba(features)
    scores = apply_eta_constraint(probabilities, view, ranker.max_extra_eta_min)
    order = offer_order(scores, view["driver_id"].to_numpy())[:TOP_K]
    timings["predict"] = time.perf_counter() - start

    start = time.perf_counter()
    contributions = ranker.model.contributions(features.iloc[order])
    for index, position in enumerate(order):
        explain(features.iloc[position], contributions.iloc[index])
    timings["explain"] = time.perf_counter() - start

    timings["total"] = sum(timings.values())
    return {stage: 1000 * seconds for stage, seconds in timings.items()}


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "p99": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 3),
        "mean": round(statistics.fmean(ordered), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark one-booking inference latency (Phase 7).")
    parser.add_argument("--repeats", type=int, default=200, help="timed requests per candidate count")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    ranker = get_ranker(args.model_dir)
    requests, padded = sample_requests(CANDIDATE_COUNTS, args.repeats)

    samples: dict[int, dict[str, list[float]]] = {count: {} for count in requests}
    for index in range(args.warmup + args.repeats):
        for count, pool in requests.items():  # round-robin, so no size is measured under different conditions
            timings = time_stages(ranker, pool[index % len(pool)])
            if index >= args.warmup:
                for stage, milliseconds in timings.items():
                    samples[count].setdefault(stage, []).append(milliseconds)
    measurements = {
        str(count): {stage: percentiles(values) for stage, values in stages.items()}
        for count, stages in samples.items()
    }

    payload = {
        "experiment": "phase7-inference-latency",
        "settings": {
            "repeats": args.repeats, "warmup": args.warmup, "top_k": TOP_K,
            "candidate_counts": CANDIDATE_COUNTS, "unit": "milliseconds", "process": "single, warm artifact",
            "padded_candidate_counts": padded,
            "padding_note": "sizes above the busiest real booking are reached by repeating candidates",
        },
        "machine": {
            "platform": platform.platform(), "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "model_version": ranker.model_version,
        "latency_ms": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(markdown_table(measurements))
    print(f"\nWrote {args.output}")


def markdown_table(measurements: dict) -> str:
    stages = ["build_view", "build_features", "predict", "explain", "total"]
    lines = ["| Candidates | " + " | ".join(f"{stage} P50" for stage in stages) + " | total P95 | total P99 |",
             "|---" * (len(stages) + 3) + "|"]
    for count, stage_metrics in measurements.items():
        cells = [f"{stage_metrics[stage]['p50']:.2f}" for stage in stages]
        lines.append(f"| {count} | " + " | ".join(cells) +
                     f" | {stage_metrics['total']['p95']:.2f} | {stage_metrics['total']['p99']:.2f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
