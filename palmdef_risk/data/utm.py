from __future__ import annotations


def detect_utm_zones(bbox_4326: tuple[float, float, float, float]) -> list[str]:
    """Return EPSG codes for every UTM zone that the bbox overlaps."""
    xmin, ymin, xmax, ymax = bbox_4326
    lat_c = (ymin + ymax) / 2.0
    zones = []
    for z in range(1, 61):
        lon_left = -180 + (z - 1) * 6
        lon_right = lon_left + 6
        if lon_right > xmin and lon_left < xmax:
            epsg = 32600 + z if lat_c > 0 else 32700 + z
            zones.append(f"EPSG:{epsg}")
    return zones


def primary_utm_zone(bbox_4326: tuple[float, float, float, float]) -> str:
    """Return EPSG code of UTM zone containing the bbox centroid."""
    xmin, ymin, xmax, ymax = bbox_4326
    lon_c = (xmin + xmax) / 2.0
    lat_c = (ymin + ymax) / 2.0
    z = int((lon_c + 180) / 6) + 1
    epsg = 32600 + z if lat_c > 0 else 32700 + z
    return f"EPSG:{epsg}"
