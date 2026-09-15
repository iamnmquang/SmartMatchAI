"""Hidden behaviour model of the simulated world (the "oracle").

Given everything about an offer - including latent driver traits and per-offer noise that no real
system could observe - these functions return the true probability that the driver accepts, and
that an accepted trip is later cancelled.

Only the data generator and offline evaluation may use this module. Training and serving code must
never import it (oracle leakage, docs/research.md §5.3).

Coefficients are chosen up front for plausible behaviour and documented in data/README.md.
They are NOT tuned to make the ML model beat the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class BehaviorParams:
    # Acceptance: logit of P(driver accepts the offer)
    accept_intercept: float = 2.1  # calibrated against marginal plausibility bands only (data/README.md)
    accept_eta: float = -0.18  # per minute of pickup ETA
    accept_eta_hinge: float = -0.10  # extra per minute above eta_hinge_min
    eta_hinge_min: float = 8.0
    accept_log_trip: float = 0.45  # per unit of log(1 + trip km): longer trips pay more
    accept_pickup_ratio: float = -0.9  # per unit of pickup_km / (trip_km + 0.5), capped at 3
    accept_selectivity: float = -1.0  # per SD of latent driver selectivity
    accept_idle: float = 0.9  # times (1 - exp(-idle_min / idle_scale_min)): saturating effect
    idle_scale_min: float = 15.0
    accept_rain: float = -0.25
    accept_heavy_rain: float = -0.60
    accept_heavy_rain_eta: float = -0.06  # interaction: heavy rain x ETA
    accept_traffic_medium: float = -0.10
    accept_traffic_high: float = -0.35
    accept_high_traffic_log_trip: float = -0.15  # interaction: high traffic x log(1 + trip km)
    accept_night_eta: float = -0.05  # interaction: night x ETA
    accept_new_driver: float = 0.35  # completed_trips < new_driver_trips
    new_driver_trips: int = 100
    accept_group: float = 0.25
    accept_quality: float = -0.10  # per SD of latent quality: top drivers are slightly pickier
    accept_noise_sd: float = 0.6  # unobserved per-offer factors

    # Cancellation: logit of P(trip is cancelled | driver accepted)
    cancel_intercept: float = -3.0
    cancel_eta: float = 0.09  # passengers cancel when the pickup takes long
    cancel_unreliability: float = 0.8  # per SD of latent driver unreliability
    cancel_rain: float = 0.15
    cancel_heavy_rain: float = 0.35
    cancel_night: float = 0.20
    cancel_noise_sd: float = 0.4


def acceptance_probability(
    p: BehaviorParams,
    *,
    eta_min,
    pickup_km,
    trip_km,
    idle_min,
    weather,
    traffic,
    time_of_day,
    is_group,
    completed_trips,
    selectivity,
    quality,
    noise,
) -> np.ndarray:
    heavy_rain = weather == "heavy_rain"
    high_traffic = traffic == "high"
    night = time_of_day == "night"
    log_trip = np.log1p(trip_km)
    z = (
        p.accept_intercept
        + p.accept_eta * eta_min
        + p.accept_eta_hinge * np.maximum(eta_min - p.eta_hinge_min, 0.0)
        + p.accept_log_trip * log_trip
        + p.accept_pickup_ratio * np.minimum(pickup_km / (trip_km + 0.5), 3.0)
        + p.accept_selectivity * selectivity
        + p.accept_idle * (1.0 - np.exp(-idle_min / p.idle_scale_min))
        + p.accept_rain * (weather == "rain")
        + p.accept_heavy_rain * heavy_rain
        + p.accept_heavy_rain_eta * heavy_rain * eta_min
        + p.accept_traffic_medium * (traffic == "medium")
        + p.accept_traffic_high * high_traffic
        + p.accept_high_traffic_log_trip * high_traffic * log_trip
        + p.accept_night_eta * night * eta_min
        + p.accept_new_driver * (completed_trips < p.new_driver_trips)
        + p.accept_group * is_group
        + p.accept_quality * quality
        + noise
    )
    return sigmoid(z)


def cancellation_probability(
    p: BehaviorParams,
    *,
    eta_min,
    weather,
    time_of_day,
    unreliability,
    noise,
) -> np.ndarray:
    z = (
        p.cancel_intercept
        + p.cancel_eta * eta_min
        + p.cancel_unreliability * unreliability
        + p.cancel_rain * (weather == "rain")
        + p.cancel_heavy_rain * (weather == "heavy_rain")
        + p.cancel_night * (time_of_day == "night")
        + noise
    )
    return sigmoid(z)
