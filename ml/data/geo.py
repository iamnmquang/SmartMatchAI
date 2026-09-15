"""Vectorized geographic helpers."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE = EARTH_RADIUS_KM * np.pi / 180.0  # same sphere as haversine_km


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km between points given in degrees."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def offset_to_latlon(center_lat: float, center_lon: float, x_km, y_km) -> tuple[np.ndarray, np.ndarray]:
    """Convert east (x) / north (y) offsets in km around a center into lat/lon degrees.

    Accurate enough for city-scale offsets (tens of km).
    """
    lat = center_lat + np.asarray(y_km, dtype=float) / KM_PER_DEGREE
    lon = center_lon + np.asarray(x_km, dtype=float) / (KM_PER_DEGREE * np.cos(np.radians(center_lat)))
    return lat, lon
