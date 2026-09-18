"""Phase 4: evaluate the baseline policies offline and save the results.

Run from the repository root, after `python -m ml.data.generate`:

    python -m ml.evaluation.run_baseline

Writes ml/evaluation/results/baseline.json (deterministic: no timestamps) and prints a markdown summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.evaluation.offline import (
    REPO_ROOT,
    business_metrics,
    load_split,
    oracle_scores,
    paired_bootstrap,
    simulate_offers,
)
from ml.evaluation.policies import (
    IDLE_CAP_MIN,
    RATING_PRIOR,
    WEIGHTED_RULE_WEIGHTS,
    nearest_driver,
    random_order,
    weighted_rule,
)
from ml.evaluation.report import policy_table

REFERENCE = "nearest_driver"
RANDOM_POLICY_SEED = 7
POLICY_DESCRIPTIONS = {
    "random": "Sanity floor: random offer order within the candidate set.",
    "nearest_driver": "Baseline and reference: closest candidate by straight-line distance first, ties by driver_id.",
    "weighted_rule": "Secondary baseline: hand-set operations heuristic, weights fixed before evaluation.",
    "oracle": "Upper bound, not attainable: order by the simulator's true acceptance probability.",
}


def evaluate_split(split: str, max_offers: int, n_resamples: int, seed: int) -> dict:
    data = load_split(split)
    scores = {
        "random": random_order(RANDOM_POLICY_SEED)(data.candidates),
        "nearest_driver": nearest_driver(data.candidates),
        "weighted_rule": weighted_rule(data.candidates),
        "oracle": oracle_scores(data.truth),
    }
    outcomes = {name: simulate_offers(data, policy_scores, max_offers) for name, policy_scores in scores.items()}
    intervals = paired_bootstrap(outcomes, reference=REFERENCE, n_resamples=n_resamples, seed=seed)
    return {
        "n_bookings": len(data.bookings),
        "n_bookings_with_candidates": int((outcomes[REFERENCE]["n_candidates"] > 0).sum()),
        "n_candidate_rows": len(data.candidates),
        "policies": {
            name: {
                "metrics": {key: round(value, 6) for key, value in business_metrics(frame).items()},
                "bootstrap": intervals[name],
            }
            for name, frame in outcomes.items()
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline policies offline (Phase 4).")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "ml" / "evaluation" / "results" / "baseline.json")
    args = parser.parse_args(argv)

    data_metadata = json.loads((REPO_ROOT / "data" / "raw" / "metadata.json").read_text(encoding="utf-8"))
    payload = {
        "experiment": "phase4-baselines",
        "data": {"generator_version": data_metadata["generator_version"], "seed": data_metadata["config"]["seed"]},
        "settings": {
            "max_offers": args.max_offers,
            "reference_policy": REFERENCE,
            "bootstrap": {"method": "paired percentile bootstrap over bookings", "resamples": args.resamples, "seed": args.seed},
            "random_policy_seed": RANDOM_POLICY_SEED,
        },
        "policies": POLICY_DESCRIPTIONS,
        "weighted_rule": {"weights": WEIGHTED_RULE_WEIGHTS, "rating_prior": RATING_PRIOR, "idle_cap_min": IDLE_CAP_MIN},
        "splits": {split: evaluate_split(split, args.max_offers, args.resamples, args.seed) for split in args.splits},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    for split, result in payload["splits"].items():
        print(f"\n### {split}: {result['n_bookings']:,} bookings "
              f"({result['n_bookings_with_candidates']:,} with candidates), {result['n_candidate_rows']:,} candidate rows\n")
        print(policy_table(result["policies"], REFERENCE))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
