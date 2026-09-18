"""Shared formatting for the offline evaluation reports (Phase 4 baselines, Phase 6 ML comparison).

One definition of the results table means the baseline numbers of Phase 4 and the ML numbers of Phase 6 are
always printed - and read - the same way.
"""

from __future__ import annotations

TABLE_ROWS = [
    ("matching_success_rate", "Matching success rate", "pct"),
    ("acceptance_rate", "Acceptance rate (offer-level)", "pct"),
    ("avg_eta_matched_min", "Average ETA of matched driver (min)", "min"),
    ("p90_eta_matched_min", "P90 ETA of matched driver (min)", "min"),
    ("cancellation_rate", "Cancellation rate", "pct"),
    ("completed_rate", "Completed rate (matched, not cancelled)", "pct"),
    ("first_offer_acceptance_rate", "First-offer acceptance rate", "pct"),
    ("avg_offers_per_booking_with_candidates", "Offers per booking with candidates", "num"),
]


def format_value(value: float, kind: str) -> str:
    return {"pct": f"{100 * value:.1f}%", "min": f"{value:.2f}", "num": f"{value:.3f}"}[kind]


def format_interval(interval: list[float], kind: str) -> str:
    low, high = interval
    if kind == "pct":
        return f"[{100 * low:+.1f}, {100 * high:+.1f}] pp"
    digits = 2 if kind == "min" else 3
    return f"[{low:+.{digits}f}, {high:+.{digits}f}]"


def policy_table(policies: dict[str, dict], reference: str) -> str:
    """One column per policy; every cell but the reference carries the bootstrap interval of the difference."""
    names = list(policies)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for key, label, kind in TABLE_ROWS:
        cells = []
        for name in names:
            cell = format_value(policies[name]["metrics"][key], kind)
            if name != reference and key in policies[name].get("bootstrap", {}):
                cell += f" ({format_interval(policies[name]['bootstrap'][key]['diff_vs_reference_ci95'], kind)})"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
