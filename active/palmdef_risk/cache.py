from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _hash(*parts) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _covers(cached: list, needed: list) -> bool:
    """True if cached bbox entirely contains needed bbox."""
    return (cached[0] <= needed[0] and cached[1] <= needed[1]
            and cached[2] >= needed[2] and cached[3] >= needed[3])


class CacheManager:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

    # ── Mill ────────────────────────────────────────────────
    def mill_dir(self, t2: int, t3: int) -> Path:
        return self.cache_dir / "mill" / f"{t2}_{t3}"

    def mill_valid(self, t2: int, t3: int) -> bool:
        d = self.mill_dir(t2, t3)
        return (d / "mill_t2.gpkg").exists() and (d / "mill_t3.gpkg").exists()

    # ── Forest ──────────────────────────────────────────────
    def forest_key(self, aoi_bbox, buffer, source, years, perc) -> str:
        return _hash(aoi_bbox, buffer, source, years, perc)

    def forest_dir(self, key: str) -> Path:
        return self.cache_dir / "forest" / key

    def forest_valid(self, key: str, needed_bbox) -> bool:
        meta = self.forest_dir(key) / "metadata.json"
        if not meta.exists():
            return False
        data = json.loads(meta.read_text())
        stored = data.get("downloaded_extent")
        return bool(stored and _covers(stored, needed_bbox))

    # ── Variables ────────────────────────────────────────────
    def variables_key(self, aoi_bbox, buffer, use_ghsl, ghsl_years, timeout,
                      river_source="big") -> str:
        return _hash(aoi_bbox, buffer, use_ghsl, ghsl_years, timeout, river_source)

    def variables_dir(self, key: str) -> Path:
        return self.cache_dir / "variables" / key

    def variables_valid(self, key: str, needed_bbox) -> bool:
        meta = self.variables_dir(key) / "metadata.json"
        if not meta.exists():
            return False
        data = json.loads(meta.read_text())
        stored = data.get("downloaded_extent")
        return bool(stored and _covers(stored, needed_bbox))

    def status_report(self, t2, t3, needed_bbox, forest_key, vars_key) -> dict:
        return {
            "mill": "hit" if self.mill_valid(t2, t3) else "miss",
            "forest": "hit" if self.forest_valid(forest_key, needed_bbox) else "miss",
            "variables": "hit" if self.variables_valid(vars_key, needed_bbox) else "miss",
        }
