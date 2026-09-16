"""The decision-time view: everything a policy may look at when ranking one booking, and nothing else.

Both offline evaluation (Phase 4) and training (Phase 5) assemble their input through this module, so a
feature built here is available online exactly the same way (docs/architecture.md §4.1). Labels and log
artifacts (accepted, cancelled, offer_rank, logging_policy) and the simulator's hidden truth are excluded
by construction: they are simply never selected.
"""

from __future__ import annotations

import pandas as pd

# candidates.idle_time_min is the value at request time; drivers.idle_time_min is a current snapshot and must not be used.
DECISION_TIME_CANDIDATE_COLUMNS = [
    "booking_id", "driver_id", "driver_lat", "driver_lon", "distance_km", "estimated_eta_min", "idle_time_min",
]
BOOKING_CONTEXT_COLUMNS = [
    "booking_id", "request_time", "pickup_lat", "pickup_lon", "destination_lat", "destination_lon",
    "passenger_type", "traffic_level", "weather", "time_of_day",
]
DRIVER_PROFILE_COLUMNS = ["driver_id", "vehicle_type", "rating", "acceptance_rate", "cancellation_rate", "completed_trips"]

VIEW_COLUMNS = DECISION_TIME_CANDIDATE_COLUMNS + [
    column for column in BOOKING_CONTEXT_COLUMNS + DRIVER_PROFILE_COLUMNS if column not in ("booking_id", "driver_id")
]

# Never allowed downstream, whatever the caller passes in (docs/research.md §5.3, data/README.md §8).
FORBIDDEN_COLUMNS = ["accepted", "cancelled", "offer_rank", "logging_policy", "p_accept", "u_accept", "p_cancel", "u_cancel"]


def build_decision_time_view(bookings: pd.DataFrame, candidates: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Join candidates with their booking context and driver profile, keeping the candidate row order.

    The caller decides which candidate rows to pass in (a split, one booking, the rows of one API request).
    """
    view = (
        candidates[DECISION_TIME_CANDIDATE_COLUMNS]
        .merge(bookings[BOOKING_CONTEXT_COLUMNS], on="booking_id", how="left")
        .merge(drivers[DRIVER_PROFILE_COLUMNS], on="driver_id", how="left")
        .reset_index(drop=True)
    )
    missing_context = view["request_time"].isna()
    if missing_context.any():
        raise ValueError(f"{int(missing_context.sum())} candidate rows have no booking context")
    return view[VIEW_COLUMNS]
