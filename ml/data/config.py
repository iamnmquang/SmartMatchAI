"""Configuration of the synthetic data generator.

A generated dataset is fully described by (generator version, this config). The effective config is
written to data/raw/metadata.json so every dataset can be reproduced.
Coefficients of the hidden behaviour model live in ml/data/behavior.py, not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hotspot:
    """A concentration of demand and supply, in km offsets from the city center."""

    x_km: float
    y_km: float
    sigma_km: float
    weight: float


DEFAULT_HOTSPOTS: tuple[Hotspot, ...] = (
    Hotspot(x_km=0.0, y_km=0.0, sigma_km=2.0, weight=0.30),  # downtown
    Hotspot(x_km=5.0, y_km=4.0, sigma_km=1.5, weight=0.15),  # business district
    Hotspot(x_km=-6.0, y_km=-3.0, sigma_km=2.0, weight=0.15),  # residential area
    Hotspot(x_km=7.0, y_km=-7.0, sigma_km=0.8, weight=0.10),  # transport hub
)

# Relative booking demand for each hour of the day (0..23): morning and evening peaks.
DEFAULT_HOURLY_DEMAND: tuple[float, ...] = (
    0.25, 0.15, 0.10, 0.10, 0.15, 0.35, 0.80, 1.40, 1.60, 1.10, 0.90, 0.95,
    1.00, 0.95, 0.90, 1.00, 1.30, 1.70, 1.60, 1.20, 0.90, 0.75, 0.60, 0.40,
)


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 42
    n_drivers: int = 20_000
    n_bookings: int = 100_000

    # Fictional city: a square of side 2 * half_extent_km. The (0, 0) center makes it obvious
    # that coordinates are synthetic and not tied to any real city.
    center_lat: float = 0.0
    center_lon: float = 0.0
    half_extent_km: float = 10.0
    hotspots: tuple[Hotspot, ...] = DEFAULT_HOTSPOTS
    background_weight: float = 0.30  # share of points spread uniformly over the city

    # Demand
    start_date: str = "2026-06-01"
    n_days: int = 56
    hourly_demand: tuple[float, ...] = DEFAULT_HOURLY_DEMAND
    weekend_demand_factor: float = 0.85
    min_trip_km: float = 1.0
    group_booking_share: float = 0.08  # group bookings need a 7-seat car

    # Supply around each booking
    car7_share: float = 0.25
    search_radius_km: float = 5.0
    mean_nearby_drivers: float = 12.0
    max_candidates: int = 30

    # Logging policy that produced the historical offers (see data/README.md)
    max_offers: int = 3
    exploration_rate: float = 0.2

    def __post_init__(self) -> None:
        if min(self.n_drivers, self.n_bookings, self.n_days) <= 0:
            raise ValueError("n_drivers, n_bookings and n_days must be positive")
        if len(self.hourly_demand) != 24:
            raise ValueError("hourly_demand needs exactly 24 values")
        for name in ("group_booking_share", "car7_share", "exploration_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_offers < 1 or self.max_candidates < 1:
            raise ValueError("max_offers and max_candidates must be >= 1")
