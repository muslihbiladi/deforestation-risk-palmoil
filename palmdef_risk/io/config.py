from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import yaml


VALID_VARIANTS = {"A", "B", "C", "D", "E", "F", "G"}


@dataclass
class RunConfig:
    # ── Run identity ───────────────────────────────────────
    project: str
    area: str
    task: str

    # ── AOI ────────────────────────────────────────────────
    aoi_source: str         # path to vector file or "xmin,ymin,xmax,ymax"
    aoi_buffer: float

    # ── CRS ────────────────────────────────────────────────
    crs: str                # e.g. "EPSG:32750"

    # ── Forest ─────────────────────────────────────────────
    forest_source: str      # "tmf" or "gfc"
    forest_years: List[int]
    forest_perc: int

    # ── Variables ──────────────────────────────────────────
    use_ghsl_towns: bool
    ghsl_years: Optional[List[int]]
    osm_timeout: int

    # ── User inputs ────────────────────────────────────────
    peatland_path: str
    peatland_type: str      # "binary" or "continuous"
    hgu_path: str
    plantation_t2: str
    plantation_t3: Optional[str]
    plantation_industrial_value: int
    plantation_smallholder_value: int

    # ── Mill ───────────────────────────────────────────────
    mill_source: str        # "trase" or "gfw"

    # ── Process ────────────────────────────────────────────
    kde_bandwidth_km: float
    lq_epsilon: float
    lq_direction: str       # "mp" or "pm"

    # ── Model ──────────────────────────────────────────────
    model_variants: List[str]
    csize: int
    burnin: int
    mcmc: int
    thin: int
    run_gwr: bool
    gwr_bandwidth: str

    # ── Output ─────────────────────────────────────────────
    project_future: bool
    projection_year: int
    risk_classes: int
    scenarios: List[str]

    # ───────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            d = yaml.safe_load(f)

        ui = d.get("user_inputs", {})
        peat = ui.get("peatland", {})
        hgu = ui.get("hgu", {})
        plant = ui.get("plantation", {})
        proc = d.get("process", {})
        mod = d.get("model", {})
        out = d.get("output", {})

        return cls(
            project=d["run"]["project"],
            area=d["run"]["area"],
            task=d["run"]["task"],
            aoi_source=str(d["aoi"]["source"]),
            aoi_buffer=float(d["aoi"].get("buffer", 0.0)),
            crs=d["crs"],
            forest_source=d["forest"]["source"],
            forest_years=list(d["forest"]["years"]),
            forest_perc=int(d["forest"].get("perc", 75)),
            use_ghsl_towns=bool(d.get("variables", {}).get("use_ghsl_towns", False)),
            ghsl_years=d.get("variables", {}).get("ghsl_years"),
            osm_timeout=int(d.get("variables", {}).get("osm_timeout", 180)),
            peatland_path=str(peat["path"]),
            peatland_type=str(peat.get("type", "binary")),
            hgu_path=str(hgu["path"]),
            plantation_t2=str(plant["t2"]),
            plantation_t3=str(plant["t3"]) if plant.get("t3") else None,
            plantation_industrial_value=int(plant.get("industrial_value", 1)),
            plantation_smallholder_value=int(plant.get("smallholder_value", 2)),
            mill_source=str(d.get("mill", {}).get("source", "trase")),
            kde_bandwidth_km=float(proc.get("kde_bandwidth_km", 35.0)),
            lq_epsilon=float(proc.get("lq_epsilon", 0.001)),
            lq_direction=str(proc.get("lq_direction", "mp")),
            model_variants=list(mod.get("variants", ["A", "B", "E", "F"])),
            csize=int(mod.get("csize", 10)),
            burnin=int(mod.get("burnin", 1000)),
            mcmc=int(mod.get("mcmc", 1000)),
            thin=int(mod.get("thin", 1)),
            run_gwr=bool(mod.get("run_gwr", False)),
            gwr_bandwidth=str(mod.get("gwr_bandwidth", "adaptive")),
            project_future=bool(out.get("project_future", False)),
            projection_year=int(out.get("projection_year", 2035)),
            risk_classes=int(out.get("risk_classes", 5)),
            scenarios=[s["name"] if isinstance(s, dict) else s
                       for s in out.get("scenarios", [])],
        )

    def validate(self) -> List[str]:
        """Return list of error strings. Empty list means valid."""
        errors = []
        if len(self.forest_years) < 2:
            errors.append("forest.years must have at least 2 entries")
        if self.project_future and self.projection_year <= self.forest_years[-1]:
            errors.append(
                f"output.projection_year ({self.projection_year}) must be "
                f"> forest.years[-1] ({self.forest_years[-1]})"
            )
        if self.use_ghsl_towns and not self.ghsl_years:
            errors.append("variables.ghsl_years is required when use_ghsl_towns: true")
        for v in self.model_variants:
            if v not in VALID_VARIANTS:
                errors.append(f"model.variants: unknown variant '{v}'")
        if self.peatland_type not in ("binary", "continuous"):
            errors.append("user_inputs.peatland.type must be 'binary' or 'continuous'")
        if self.lq_direction not in ("mp", "pm"):
            errors.append("process.lq_direction must be 'mp' or 'pm'")
        if self.forest_source not in ("tmf", "gfc"):
            errors.append("forest.source must be 'tmf' or 'gfc'")
        if self.mill_source not in ("trase", "gfw"):
            errors.append("mill.source must be 'trase' or 'gfw'")
        return errors

    def run_folder_name(self, timestamp: str) -> str:
        """Return run folder name: {project}_{area}_{task}_{timestamp}."""
        return f"{self.project}_{self.area}_{self.task}_{timestamp}"
