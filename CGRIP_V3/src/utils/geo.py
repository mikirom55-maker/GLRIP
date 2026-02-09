import numpy as np
from typing import Tuple


EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def spatiotemporal_distance(
    lat1: float, lon1: float, t1: float,
    lat2: float, lon2: float, t2: float,
    alpha: float = 0.5,
    spatial_scale: float = 100.0,
    temporal_scale: float = 24.0,
) -> float:
    spatial_dist = haversine_distance(lat1, lon1, lat2, lon2)
    temporal_dist = abs(t1 - t2)
    norm_spatial = spatial_dist / spatial_scale
    norm_temporal = temporal_dist / temporal_scale
    return alpha * norm_spatial + (1 - alpha) * norm_temporal


def distance_weight(distance: float, beta: float = 1.0) -> float:
    return np.exp(beta * distance)


def compute_centroid(lats: np.ndarray, lons: np.ndarray,
                     weights: np.ndarray = None) -> Tuple[float, float]:
    if weights is None:
        weights = np.ones(len(lats))
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.mean(lats), np.mean(lons)
    centroid_lat = np.sum(weights * lats) / total_weight
    centroid_lon = np.sum(weights * lons) / total_weight
    return centroid_lat, centroid_lon


def compute_displacement(
    centroid_prev: Tuple[float, float],
    centroid_curr: Tuple[float, float],
    time_delta_hours: float = 1.0,
) -> dict:
    lat1, lon1 = centroid_prev
    lat2, lon2 = centroid_curr
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    speed = distance / time_delta_hours if time_delta_hours > 0 else 0
    direction = compute_bearing(lat1, lon1, lat2, lon2)
    return {
        'distance_km': distance,
        'speed_kmh': speed,
        'direction': direction,
    }


def direction_to_text(direction: float) -> str:
    directions = [
        (22.5, 'N'), (67.5, 'NE'), (112.5, 'E'), (157.5, 'SE'),
        (202.5, 'S'), (247.5, 'SW'), (292.5, 'W'), (337.5, 'NW'),
        (360, 'N'),
    ]
    for threshold, name in directions:
        if direction < threshold:
            return name
    return 'N'
