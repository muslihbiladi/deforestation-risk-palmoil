from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import yaml

logger = logging.getLogger(__name__)
VALID_VARIANTS = {"A", "B", "C"}


@dataclass
class RunConfig:
    # Run identity
    project: str
    area: str
    task: str
    # AOI
    aoi_source: str
    aoi_buffer: float
    # CRS (None = auto-detect UTM at create_run time)
    crs: Optional[str]
    cache_dir: str
    # Forest
    forest_source: str
    forest_years: List[int]
    forest_perc: int
    # Variables
    use_ghsl_towns: bool
    ghsl_years: Optional[List[int]]
    osm_timeout: int
    # User inputs
    peatland_path: str
    peatland_type: str
    hgu_path: str
    plantation_t2: str
    plantation_t3: Optional[str]
    plantation_industrial_value: int
    plantation_smallholder_value: int
    # Mill
    mill_source: str          # "trase" or "user"
    mill_path: Optional[str]  # required when source="user"
    # Process — gravity
    sigma_km: float
    radius_km: float
    sensitivity_sigmas: List[float]
    # Model
    model_variants: List[str]
    nsamp: int
    csize: int
    Vbeta: float
    burnin: int
    mcmc: int
    thin: int
    seed: int
    # Parallel
    max_workers: Optional[int]
    cpu_fraction: float
    ram_per_dist_gb: float
    ram_per_icar_gb: float
    ram_per_predict_gb: float
    # Output
    project_future: bool
    projection_year: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            d = yaml.safe_load(f)

        ui = d.get("user_inputs", {})
        peat = ui.get("peatland", {})
        hgu = ui.get("hgu", {})
        plant = ui.get("plantation", {})
        proc = d.get("process", {})
        grav = proc.get("gravity", {})
        sens = proc.get("sensitivity", {})
        mod = d.get("model", {})
        par = d.get("parallel", {})
        out = d.get("output", {})

        cfg = cls(
            project=d["run"]["project"],
            area=d["run"]["area"],
            task=d["run"]["task"],
            aoi_source=str(d["aoi"]["source"]),
            aoi_buffer=float(d["aoi"].get("buffer", 0.0)),
            crs=d.get("crs"),
            cache_dir=str(d.get("cache_dir", "cache/")),
            forest_source=d["forest"]["source"],
            forest_years=list(d["forest"]["years"]),
            forest_perc=int(d["forest"].get("perc", 75)),
            use_ghsl_towns=bool(d.get("variables", {}).get("use_ghsl_towns", False)),
            ghsl_years=d.get("variables", {}).get("ghsl_years"),
            osm_timeout=int(d.get("variables", {}).get("osm_timeout", 180)),
            peatland_path=str(peat.get("path", "")),
            peatland_type=str(peat.get("type", "binary")),
            hgu_path=str(hgu.get("path", "")),
            plantation_t2=str(plant.get("t2", "")),
            plantation_t3=str(plant["t3"]) if plant.get("t3") else None,
            plantation_industrial_value=int(plant.get("industrial_value", 1)),
            plantation_smallholder_value=int(plant.get("smallholder_value", 2)),
            mill_source=str(d.get("mill", {}).get("source", "trase")),
            mill_path=d.get("mill", {}).get("path"),
            sigma_km=float(grav.get("sigma_km", 25.0)),
            radius_km=float(grav.get("radius_km", 80.0)),
            sensitivity_sigmas=[float(s) for s in sens.get("sigmas_km", [15.0, 25.0, 40.0])],
            model_variants=list(mod.get("variants", ["A", "B", "C"])),
            nsamp=int(mod.get("nsamp", 10000)),
            csize=int(mod.get("csize", 10)),
            Vbeta=float(mod.get("Vbeta", 1000)),
            burnin=int(mod.get("burnin", 1000)),
            mcmc=int(mod.get("mcmc", 1000)),
            thin=int(mod.get("thin", 1)),
            seed=int(mod.get("seed", 42)),
            max_workers=par.get("max_workers"),
            cpu_fraction=float(par.get("cpu_fraction", 0.9)),
            ram_per_dist_gb=float(par.get("ram_per_dist_gb", 0.5)),
            ram_per_icar_gb=float(par.get("ram_per_icar_gb", 1.0)),
            ram_per_predict_gb=float(par.get("ram_per_predict_gb", 0.75)),
            project_future=bool(out.get("project_future", False)),
            projection_year=int(out.get("projection_year", 2035)),
        )
        cfg._warn_if_needed()
        return cfg

    def _warn_if_needed(self) -> None:
        if self.Vbeta > 100:
            logger.warning(
                "Vbeta=%.0f > 100: risk of divergent MCMC chain under spatial confounding. "
                "Consider reducing to ~10 if residual autocorrelation is high.", self.Vbeta
            )

    def validate(self) -> List[str]:
        errors = []
        if len(self.forest_years) < 2:
            errors.append("forest.years must have at least 2 entries")
        if self.project_future and self.projection_year <= self.forest_years[-1]:
            errors.append(
                f"output.projection_year ({self.projection_year}) must be "
                f"> forest.years[-1] ({self.forest_years[-1]})"
            )
        if self.use_ghsl_towns and not self.ghsl_years:
            errors.append("variables.ghsl_years required when use_ghsl_towns: true")
        for v in self.model_variants:
            if v not in VALID_VARIANTS:
                errors.append(f"model.variants: unknown variant '{v}' (valid: A, B, C)")
        if self.peatland_type not in ("binary", "continuous"):
            errors.append("user_inputs.peatland.type must be 'binary' or 'continuous'")
        if self.forest_source not in ("tmf", "gfc"):
            errors.append("forest.source must be 'tmf' or 'gfc'")
        if self.mill_source not in ("trase", "user"):
            errors.append("mill.source must be 'trase' or 'user'")
        if self.mill_source == "user" and not self.mill_path:
            errors.append("mill.path required when mill.source: 'user'")
        if self.sigma_km <= 0:
            errors.append("process.gravity.sigma_km must be > 0")
        if self.radius_km <= self.sigma_km:
            errors.append("process.gravity.radius_km must be > sigma_km")
        if self.Vbeta <= 0:
            errors.append("model.Vbeta must be > 0")
        if self.nsamp <= 0:
            errors.append("model.nsamp must be > 0")
        return errors

    def run_folder_name(self, timestamp: str) -> str:
        return f"{self.project}_{self.area}_{self.task}_{timestamp}"
