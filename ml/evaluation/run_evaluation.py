"""Phase 6: ML ranking vs the baselines, measured in the same simulated world.

Run from the repository root, after `python -m ml.data.generate` and `python -m ml.training.train`:

    python -m ml.evaluation.run_evaluation

Two stages, in this order on purpose:

1. **Validation - choose.** Both trained models are replayed with several ETA margins (PRD Q2: option A is the
   margin "infinity", option B a finite one). The winner is picked by a rule fixed before the run: highest
   completed rate among the policies that keep both guardrails - average ETA no more than +15% over the
   baseline (H2) and cancellation rate not above it (H3).
2. **Test - measure once.** The chosen policy, the two option-A policies, the baselines and the oracle are
   replayed on the test split, with paired bootstrap intervals against Nearest Driver, plus the classification
   metrics of both models on the test offers. Nothing is tuned here; this run is the report.

Metric definitions, the replay protocol and the bootstrap are unchanged from Phase 4 (docs/evaluation.md §2),
so baseline numbers stay comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.offline import (
    REPO_ROOT,
    EvaluationData,
    build_evaluation_data,
    business_metrics,
    oracle_scores,
    paired_bootstrap,
    simulate_offers,
)
from ml.evaluation.policies import (
    apply_eta_constraint,
    model_scores,
    nearest_driver,
    random_order,
    weighted_rule,
)
from ml.evaluation.report import format_interval, format_value, policy_table
from ml.evaluation.run_baseline import RANDOM_POLICY_SEED
from ml.training.dataset import DEFAULT_RAW_DIR, labeled_offers, load_raw
from ml.training.metrics import classification_metrics, ranking_metrics
from ml.training.models import ARTIFACT_DIR, load_models

RESULTS_PATH = REPO_ROOT / "ml" / "evaluation" / "results" / "evaluation.json"
DEFAULT_ORACLE_DIR = REPO_ROOT / "data" / "oracle"
REFERENCE = "nearest_driver"

# Extra pickup minutes over the fastest candidate of the booking that a policy may still offer first.
# "inf" is option A (rank by P(accept) alone); the finite values are option B.
DELTA_GRID_MIN = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, float("inf")]
MAX_ETA_INCREASE = 0.15  # PRD H2
SELECTION_METRIC = "completed_rate"  # second North Star metric since Phase 5 (PRD §8.2)

HEADLINE_ROWS = [
    ("matching_success_rate", "Matching success rate", "pct"),
    ("acceptance_rate", "Acceptance rate (offer-level)", "pct"),
    ("avg_eta_matched_min", "Average ETA of matched driver (min)", "min"),
    ("cancellation_rate", "Cancellation rate", "pct"),
    ("completed_rate", "Completed rate", "pct"),
]


def policy_label(model_name: str, delta: float) -> str:
    return f"{model_name} + ETA margin {delta:g} min" if np.isfinite(delta) else f"{model_name} (P(accept) only)"


def validation_search(data: EvaluationData, scores: dict[str, np.ndarray], max_offers: int) -> list[dict]:
    """Replay every (model, ETA margin) pair on the validation split. No bootstrap: this only ranks options."""
    rows = []
    for model_name, base in scores.items():
        for delta in DELTA_GRID_MIN:
            constrained = apply_eta_constraint(base, data.candidates, delta)
            metrics = business_metrics(simulate_offers(data, constrained, max_offers))
            rows.append({
                "model": model_name,
                "max_extra_eta_min": delta if np.isfinite(delta) else None,
                "option": "A" if not np.isfinite(delta) else "B",
                "label": policy_label(model_name, delta),
                "metrics": {key: round(value, 6) for key, value in metrics.items()},
            })
    return rows


def select_policy(rows: list[dict], baseline: dict[str, float]) -> dict:
    """The selection rule, fixed before the run: highest completed rate among policies that keep the guardrails.

    Guardrails are read on validation, against the validation baseline: average ETA at most +15% (H2) and
    cancellation rate not above the baseline (H3).
    """
    eta_cap = baseline["avg_eta_matched_min"] * (1 + MAX_ETA_INCREASE)
    within_eta = [row for row in rows if row["metrics"]["avg_eta_matched_min"] <= eta_cap]
    within_both = [row for row in within_eta if row["metrics"]["cancellation_rate"] <= baseline["cancellation_rate"]]

    if within_both:
        eligible, rule = within_both, "highest completed rate among policies satisfying H2 and H3"
    elif within_eta:
        eligible, rule = within_eta, "highest completed rate among policies satisfying H2 only (none satisfied H3)"
    else:
        eligible, rule = rows, "highest completed rate; no policy satisfied H2"
    best = max(eligible, key=lambda row: row["metrics"][SELECTION_METRIC])
    return {
        "rule": rule,
        "eta_cap_min": round(eta_cap, 6),
        "baseline_cancellation_rate": round(baseline["cancellation_rate"], 6),
        "n_eligible": len(eligible),
        "model": best["model"],
        "max_extra_eta_min": best["max_extra_eta_min"],
        "option": best["option"],
        "label": best["label"],
        "validation_metrics": best["metrics"],
    }


def test_policy_scores(data: EvaluationData, scores: dict[str, np.ndarray], selected: dict) -> dict[str, np.ndarray]:
    delta = selected["max_extra_eta_min"]
    delta = float("inf") if delta is None else float(delta)
    return {
        "random": random_order(RANDOM_POLICY_SEED)(data.candidates),
        "nearest_driver": nearest_driver(data.candidates),
        "weighted_rule": weighted_rule(data.candidates),
        "ml_logistic_regression": scores["logistic_regression"],
        "ml_xgboost": scores["xgboost"],
        "ml_selected": apply_eta_constraint(scores[selected["model"]], data.candidates, delta),
        "oracle": oracle_scores(data.truth),
    }


def gap_closed(policies: dict[str, dict], metric: str, policy: str) -> float | None:
    """Share of the achievable improvement that a policy captured: (policy - baseline) / (oracle - baseline).

    Only meaningful where the oracle is actually an upper bound. It is not one for completed rate: ordering by
    the true acceptance probability picks drivers who are further away and therefore cancel more, so the oracle
    can sit below the baseline (docs/evaluation.md, Phase 4 §5.3). There the ratio is left out.
    """
    baseline = policies[REFERENCE]["metrics"][metric]
    headroom = policies["oracle"]["metrics"][metric] - baseline
    if headroom <= 1e-9:
        return None
    return round((policies[policy]["metrics"][metric] - baseline) / headroom, 4)


def ml_metrics_on_test(models: dict, raw, split: str) -> dict:
    """The classification tier of docs/research.md §7.1, on the labelled offers of the test split."""
    labelled = labeled_offers(raw, split)
    report = {"n_offers": len(labelled), "n_bookings": labelled.n_bookings, "positive_rate": round(labelled.positive_rate, 6)}
    for name, model in models.items():
        p = model.predict_proba(labelled.features)[:, 1]
        metrics = {**classification_metrics(labelled.y, p), "ranking": ranking_metrics(labelled.booking_id, labelled.y, p)}
        report[name] = _round(metrics)
    return report


def hypotheses(policies: dict[str, dict], ml_metrics: dict, selected_model: str) -> list[dict]:
    """PRD §8.3, decided on the test split with the bootstrap interval of the difference to the baseline."""
    selected, baseline = policies["ml_selected"], policies[REFERENCE]
    verdicts = []

    for key, name, statement in [
        ("matching_success_rate", "H1", "ML beats Nearest Driver on matching success rate"),
        ("completed_rate", "H1b", "ML beats Nearest Driver on completed rate"),
    ]:
        interval = selected["bootstrap"][key]["diff_vs_reference_ci95"]
        verdicts.append(_verdict(name, statement, interval[0] > 0,
                                 f"{selected['metrics'][key] - baseline['metrics'][key]:+.4f} (95% CI {interval})"))

    eta_increase = selected["metrics"]["avg_eta_matched_min"] / baseline["metrics"]["avg_eta_matched_min"] - 1
    verdicts.append(_verdict("H2", "Average ETA of ML is at most +15% over the baseline",
                             eta_increase <= MAX_ETA_INCREASE, f"{eta_increase:+.1%}"))

    # H3 is a "no worse than" claim, so it fails only when the interval says the rate is *significantly* higher.
    cancellation = selected["bootstrap"]["cancellation_rate"]["diff_vs_reference_ci95"]
    verdicts.append(_verdict("H3", "Cancellation rate of ML is not higher than the baseline",
                             cancellation[0] <= 0,
                             f"{selected['metrics']['cancellation_rate'] - baseline['metrics']['cancellation_rate']:+.4f} "
                             f"(95% CI {cancellation})"))

    roc_auc = ml_metrics[selected_model]["roc_auc"]
    verdicts.append(_verdict("H4", "Test ROC-AUC is clearly above chance but not implausibly high",
                             0.5 < roc_auc < 0.95, f"{roc_auc:.4f}"))
    return verdicts


def _verdict(name: str, statement: str, holds: bool, evidence: str) -> dict:
    return {"id": name, "hypothesis": statement, "verdict": "supported" if holds else "not supported", "evidence": evidence}


def _repo_path(path: Path) -> str:
    """Record where the models came from without writing a machine-specific absolute path into the report."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _round(value, digits: int = 6):
    if isinstance(value, dict):
        return {key: _round(item, digits) for key, item in value.items()}
    return round(value, digits) if isinstance(value, float) else value


def markdown_headline(policies: dict[str, dict]) -> str:
    """The table the project brief asks for: Metric | Baseline | ML | Improvement."""
    lines = ["| Metric | Baseline (Nearest Driver) | ML (selected) | Improvement |", "|---|---|---|---|"]
    for key, label, kind in HEADLINE_ROWS:
        baseline = policies[REFERENCE]["metrics"][key]
        ml = policies["ml_selected"]["metrics"][key]
        interval = policies["ml_selected"]["bootstrap"][key]["diff_vs_reference_ci95"]
        if kind == "pct":
            improvement = f"{100 * (ml - baseline):+.1f} pp"
        else:
            improvement = f"{ml - baseline:+.2f} min ({(ml / baseline - 1):+.1%})"
        lines.append(f"| {label} | {format_value(baseline, kind)} | {format_value(ml, kind)} | "
                     f"{improvement}, 95% CI {format_interval(interval, kind)} |")
    return "\n".join(lines)


def markdown_validation_search(rows: list[dict], baseline: dict[str, float]) -> str:
    lines = ["| Policy | MSR | Completed rate | Avg ETA (min) | Cancellation |", "|---|---|---|---|---|"]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['label']} | {100 * metrics['matching_success_rate']:.1f}% | "
            f"{100 * metrics['completed_rate']:.1f}% | {metrics['avg_eta_matched_min']:.2f} "
            f"({metrics['avg_eta_matched_min'] / baseline['avg_eta_matched_min'] - 1:+.1%}) | "
            f"{100 * metrics['cancellation_rate']:.1f}% |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare ML ranking with the baselines (Phase 6).")
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    raw = load_raw(args.raw_dir)
    truth = pd.read_parquet(args.oracle_dir / "candidate_truth.parquet")
    models = load_models(args.artifact_dir)

    def split_data(split: str) -> EvaluationData:
        return build_evaluation_data(raw.bookings, raw.candidates, raw.drivers, truth, split)

    # Stage 1: choose on validation.
    validation = split_data("validation")
    validation_scores = {name: model_scores(model, validation.candidates) for name, model in models.items()}
    baseline_validation = business_metrics(simulate_offers(validation, nearest_driver(validation.candidates), args.max_offers))
    search = validation_search(validation, validation_scores, args.max_offers)
    selected = select_policy(search, baseline_validation)
    print(f"validation: {len(validation.bookings):,} bookings, {len(validation.candidates):,} candidate rows\n")
    print(markdown_validation_search(search, baseline_validation))
    print(f"\nselected: {selected['label']} ({selected['rule']}, {selected['n_eligible']} eligible)\n")

    # Stage 2: measure once on test.
    test = split_data("test")
    test_scores = {name: model_scores(model, test.candidates) for name, model in models.items()}
    outcomes = {
        name: simulate_offers(test, scores, args.max_offers)
        for name, scores in test_policy_scores(test, test_scores, selected).items()
    }
    intervals = paired_bootstrap(outcomes, reference=REFERENCE, n_resamples=args.resamples, seed=args.seed)
    policies = {
        name: {"metrics": {key: round(value, 6) for key, value in business_metrics(frame).items()},
               "bootstrap": intervals[name]}
        for name, frame in outcomes.items()
    }
    ml_metrics = ml_metrics_on_test(models, raw, "test")

    data_metadata = json.loads((args.raw_dir / "metadata.json").read_text(encoding="utf-8"))
    training_metadata = json.loads((REPO_ROOT / "ml" / "training" / "results" / "training.json").read_text(encoding="utf-8"))
    payload = {
        "experiment": "phase6-ml-vs-baseline",
        "data": {"generator_version": data_metadata["generator_version"], "seed": data_metadata["config"]["seed"]},
        "settings": {
            "max_offers": args.max_offers,
            "reference_policy": REFERENCE,
            "bootstrap": {"method": "paired percentile bootstrap over bookings", "resamples": args.resamples, "seed": args.seed},
            "random_policy_seed": RANDOM_POLICY_SEED,
            "eta_margin_grid_min": [delta if np.isfinite(delta) else None for delta in DELTA_GRID_MIN],
            "selection_metric": SELECTION_METRIC,
            "max_eta_increase": MAX_ETA_INCREASE,
            "models_from": _repo_path(args.artifact_dir),
            "model_features": training_metadata["settings"]["features"],
        },
        "validation_selection": {
            "n_bookings": len(validation.bookings),
            "baseline": {key: round(value, 6) for key, value in baseline_validation.items()},
            "candidates": search,
            "selected": selected,
        },
        "test": {
            "n_bookings": len(test.bookings),
            "n_bookings_with_candidates": int((outcomes[REFERENCE]["n_candidates"] > 0).sum()),
            "n_candidate_rows": len(test.candidates),
            "policies": policies,
            "gap_closed": {
                metric: {policy: gap_closed(policies, metric, policy)
                         for policy in ("weighted_rule", "ml_logistic_regression", "ml_xgboost", "ml_selected")}
                for metric in ("matching_success_rate", "completed_rate", "acceptance_rate")
            },
            "ml_metrics": ml_metrics,
            "hypotheses": hypotheses(policies, ml_metrics, selected["model"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"test: {payload['test']['n_bookings']:,} bookings "
          f"({payload['test']['n_bookings_with_candidates']:,} with candidates), "
          f"{payload['test']['n_candidate_rows']:,} candidate rows\n")
    print(markdown_headline(policies))
    print()
    print(policy_table(policies, REFERENCE))
    print()
    for verdict in payload["test"]["hypotheses"]:
        print(f"{verdict['id']}: {verdict['verdict']} - {verdict['evidence']}")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
