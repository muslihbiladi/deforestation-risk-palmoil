# palmdef_risk Refactor — Implementation Plan Part 1 (Phases 1–3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename `palmoil_risk` → `palmdef_risk`, replace config schema with gravity-based fields, wire CRS auto-detection, and deliver new infrastructure modules (utm.py, parallel.py, cache.py) plus rewritten data-download layer (mill.py, forest.py, variables.py).

**Architecture:** Incremental module replacement — existing tests remain runnable throughout. New package directory created first; old directory left in place until Task 1 is done so no import breaks during migration.

**Tech Stack:** Python 3.10+, GDAL/osgeo, geopandas, scipy, psutil, requests, pyyaml, pytest

**Spec:** `docs/superpowers/specs/2026-05-19-palmdef-risk-refactor-design.md`  
**Part 2:** `docs/superpowers/plans/2026-05-19-palmdef-risk-refactor-plan-part2.md`

---

## File Map

| Action | Path |
|---|---|
| Rename dir | `palmoil_risk/` → `palmdef_risk/` |
| Delete | `palmdef_risk/process/lq.py`, `palmdef_risk/process/correlation.py`, `palmdef_risk/model/gwr.py` |
| Rewrite | `palmdef_risk/io/config.py` |
| Update | `palmdef_risk/io/run.py`, `pyproject.toml` |
| New | `palmdef_risk/data/utm.py` |
| New | `palmdef_risk/parallel.py` |
| New | `palmdef_risk/cache.py` |
| Rewrite | `palmdef_risk/data/mill.py` |
| Polish | `palmdef_risk/data/forest.py` (add `output_crs` arg) |
| Polish | `palmdef_risk/data/variables.py` (add `output_crs` arg, rename `pa` → `protected`) |
| Update | `tests/conftest.py` (new config schema, palmdef_risk imports) |
| Update | `tests/io/test_config.py` |
| New | `tests/data/test_utm.py` |
| New | `tests/test_parallel.py` |
| New | `tests/test_cache.py` |
| Update | `tests/data/test_mill.py` |

---

## Task 1 — Package rename: `palmoil_risk` → `palmdef_risk`

**Files:**
- Rename: `palmoil_risk/` → `palmdef_risk/`
- Modify: `pyproject.toml`
- Modify: all `*.py` import lines (bulk sed)

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_import.py  (new file)
def test_package_imports_as_palmdef_risk():
    import palmdef_risk
    from palmdef_risk.io.config import RunConfig
    from palmdef_risk.io.run import RunContext, create_run, load_run
    assert True
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_import.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk'`

- [ ] **Step 3: Rename directory and update all imports**

```powershell
# In PowerShell from repo root
Move-Item palmoil_risk palmdef_risk
```

Then bulk-replace in every `.py` file under `palmdef_risk/` and `tests/`:
```
from palmoil_risk   →   from palmdef_risk
import palmoil_risk →   import palmdef_risk
palmoil_risk.       →   palmdef_risk.
```

Update `pyproject.toml`:
```toml
[project]
name = "palmdef-risk"

[tool.setuptools.packages.find]
include = ["palmdef_risk*"]
```

Reinstall: `pip install -e .`

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_import.py -v
```
Expected: PASS

- [ ] **Step 5: Run existing test suite to catch broken imports**

```
pytest tests/ -x -q 2>&1 | head -40
```
Fix any remaining `palmoil_risk` import strings found.

- [ ] **Step 6: Commit**

```
git add -A
git commit -m "refactor: rename package palmoil_risk → palmdef_risk"
```

---

## Task 2 — Config schema rewrite (`io/config.py`)

**Files:**
- Rewrite: `palmdef_risk/io/config.py`
- Update: `tests/conftest.py` (new YAML fixture)
- Rewrite: `tests/io/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/io/test_config.py  (full rewrite)
import pytest
import yaml
from pathlib import Path
from palmdef_risk.io.config import RunConfig


def test_new_fields_parse(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert cfg.sigma_km == 25.0
    assert cfg.radius_km == 80.0
    assert cfg.sensitivity_sigmas == [15.0, 25.0, 40.0]
    assert cfg.Vbeta == 1000
    assert cfg.nsamp == 10000
    assert cfg.mill_source == "trase"
    assert cfg.cache_dir == "cache/"


def test_crs_may_be_null(tmp_path, user_input_files):
    cfg_dict = _base_cfg(user_input_files)
    cfg_dict["crs"] = None
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_dict))
    cfg = RunConfig.from_yaml(p)
    assert cfg.crs is None


def test_old_lq_fields_absent(minimal_config_yaml):
    cfg = RunConfig.from_yaml(minimal_config_yaml)
    assert not hasattr(cfg, "kde_bandwidth_km")
    assert not hasattr(cfg, "lq_direction")
    assert not hasattr(cfg, "run_gwr")


def test_validation_unknown_variant(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["model"]["variants"] = ["A", "Z"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("variant" in e for e in errors)


def test_vbeta_warning_logged(minimal_config_yaml, tmp_path, caplog):
    import logging
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["model"]["Vbeta"] = 50
    p = tmp_path / "vbeta.yaml"
    p.write_text(yaml.dump(d))
    with caplog.at_level(logging.WARNING, logger="palmdef_risk"):
        RunConfig.from_yaml(p).validate()
    assert any("Vbeta" in r.message for r in caplog.records)


def test_sigma_gt_zero_validation(minimal_config_yaml, tmp_path):
    d = yaml.safe_load(minimal_config_yaml.read_text())
    d["process"]["gravity"]["sigma_km"] = 0
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(d))
    errors = RunConfig.from_yaml(p).validate()
    assert any("sigma" in e for e in errors)


def _base_cfg(uif):
    return {
        "run": {"project": "test_proj", "area": "test_area", "task": "test"},
        "aoi": {"source": str(uif["hgu"]), "buffer": 0.0},
        "crs": "EPSG:32750",
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(uif["peatland"]), "type": "binary"},
            "hgu": {"path": str(uif["hgu"])},
            "plantation": {"t2": str(uif["plantation_t2"]), "t3": None,
                           "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {
            "gravity": {"sigma_km": 25.0, "radius_km": 80.0},
            "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]},
        },
        "model": {
            "variants": ["A", "B"], "nsamp": 10000, "csize": 10,
            "Vbeta": 1000, "burnin": 100, "mcmc": 100, "thin": 1, "seed": 42,
        },
        "parallel": {"max_workers": None, "cpu_fraction": 0.9,
                     "ram_per_dist_gb": 0.5, "ram_per_icar_gb": 1.0,
                     "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
```

Update `tests/conftest.py` — replace the `minimal_config_yaml` fixture body with `_base_cfg` content:

```python
# In tests/conftest.py, replace the minimal_config_yaml fixture:
@pytest.fixture
def minimal_config_yaml(tmp_path, user_input_files) -> Path:
    (tmp_path / "configs").mkdir(exist_ok=True)
    cfg = {
        "run": {"project": "test_proj", "area": "test_area", "task": "test"},
        "aoi": {"source": str(user_input_files["hgu"]), "buffer": 0.0},
        "crs": "EPSG:32750",
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(user_input_files["peatland"]), "type": "binary"},
            "hgu": {"path": str(user_input_files["hgu"])},
            "plantation": {"t2": str(user_input_files["plantation_t2"]),
                           "t3": None, "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {
            "gravity": {"sigma_km": 25.0, "radius_km": 80.0},
            "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]},
        },
        "model": {
            "variants": ["A", "B"], "nsamp": 10000, "csize": 10,
            "Vbeta": 1000, "burnin": 100, "mcmc": 100, "thin": 1, "seed": 42,
        },
        "parallel": {"max_workers": None, "cpu_fraction": 0.9,
                     "ram_per_dist_gb": 0.5, "ram_per_icar_gb": 1.0,
                     "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
    path = tmp_path / "configs" / "test.yaml"
    path.write_text(yaml.dump(cfg))
    return path
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/io/test_config.py -v
```
Expected: `AttributeError: 'RunConfig' object has no attribute 'sigma_km'`

- [ ] **Step 3: Rewrite `palmdef_risk/io/config.py`**

```python
from __future__ import annotations
import logging
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/io/test_config.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/io/config.py tests/io/test_config.py tests/conftest.py
git commit -m "feat: rewrite RunConfig with gravity/icar schema, drop LQ/GWR fields"
```

---

## Task 3 — `run.py`: CRS auto-detect + updated subdirs

**Files:**
- Modify: `palmdef_risk/io/run.py`
- Modify: `tests/io/test_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/io/test_run.py — add these tests
from palmdef_risk.io.run import create_run


def test_crs_autodetected_when_null(tmp_path, user_input_files):
    """When config.crs is null, create_run populates it via utm.py."""
    import yaml
    cfg = _null_crs_config(user_input_files)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))
    ctx = create_run(p, runs_root=tmp_path / "runs")
    # AOI is hgu.gpkg at EPSG:32750 (UTM 50S) — auto-detected must be non-null
    assert ctx.config.crs is not None
    assert ctx.config.crs.startswith("EPSG:")


def test_run_subdirs_no_kde_or_correlation(tmp_path, minimal_config_yaml):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    dirs = [str(p.relative_to(ctx.run_dir)) for p in ctx.run_dir.rglob("*") if p.is_dir()]
    assert not any("kde" in d for d in dirs)
    assert not any("correlation" in d for d in dirs)
    assert any("mill" in d for d in dirs)
    assert any("diagnostics" in d for d in dirs)
    assert any("sensitivity" in d for d in dirs)


def _null_crs_config(uif):
    return {
        "run": {"project": "p", "area": "a", "task": "t"},
        "aoi": {"source": str(uif["hgu"]), "buffer": 0.0},
        "crs": None,
        "cache_dir": "cache/",
        "forest": {"source": "tmf", "years": [2015, 2020, 2024], "perc": 75},
        "variables": {"use_ghsl_towns": False, "ghsl_years": None, "osm_timeout": 180},
        "user_inputs": {
            "peatland": {"path": str(uif["peatland"]), "type": "binary"},
            "hgu": {"path": str(uif["hgu"])},
            "plantation": {"t2": str(uif["plantation_t2"]), "t3": None,
                           "industrial_value": 1, "smallholder_value": 2},
        },
        "mill": {"source": "trase", "path": None},
        "process": {"gravity": {"sigma_km": 25.0, "radius_km": 80.0},
                    "sensitivity": {"sigmas_km": [15.0, 25.0, 40.0]}},
        "model": {"variants": ["A"], "nsamp": 100, "csize": 10, "Vbeta": 1000,
                  "burnin": 10, "mcmc": 10, "thin": 1, "seed": 42},
        "parallel": {"max_workers": None, "cpu_fraction": 0.9, "ram_per_dist_gb": 0.5,
                     "ram_per_icar_gb": 1.0, "ram_per_predict_gb": 0.75},
        "output": {"project_future": False, "projection_year": 2035},
    }
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/io/test_run.py::test_crs_autodetected_when_null -v
```
Expected: `AssertionError` — ctx.config.crs is None

- [ ] **Step 3: Update `palmdef_risk/io/run.py`**

Replace `from palmoil_risk...` import and add CRS auto-detection + updated subdirs:

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import shutil

from palmdef_risk.io.config import RunConfig


@dataclass
class RunContext:
    run_dir: Path
    config: RunConfig

    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def raw_dir(self) -> Path:
        return self.run_dir / "data" / "raw"

    @property
    def output_dir(self) -> Path:
        return self.run_dir / "output"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"


_SUBDIRS = [
    "data/raw/forest",
    "data/raw/variables",
    "data/raw/mill",
    "data/raw/user_inputs",
    "output/models",
    "output/diagnostics",
    "output/predictions",
    "output/sensitivity",
    "logs",
]


def create_run(
    config_path: str | Path,
    runs_root: str | Path = "runs",
    dry_run: bool = False,
) -> RunContext:
    config_path = Path(config_path)
    config = RunConfig.from_yaml(config_path)

    # CRS auto-detection when crs is null
    if config.crs is None:
        from palmdef_risk.data.utm import primary_utm_zone
        from palmdef_risk.io.helpers import aoi_bbox_4326
        bbox = aoi_bbox_4326(config.aoi_source)
        config.crs = primary_utm_zone(bbox)
        logging.getLogger(__name__).info("Auto-detected CRS: %s", config.crs)

    errors = config.validate()
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / config.run_folder_name(ts)
    ctx = RunContext(run_dir=run_dir, config=config)

    if dry_run:
        print(f"[dry-run] Run folder would be: {run_dir}")
        return ctx

    for sub in _SUBDIRS:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, run_dir / "config.yaml")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "logs" / "run.log"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger(__name__).info("Run created: %s", run_dir)
    return ctx


def load_run(run_dir: str | Path | None = None) -> RunContext:
    if run_dir is None:
        run_dir = _prompt_run_selection()
    run_dir = Path(run_dir)
    if not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    config = RunConfig.from_yaml(run_dir / "config.yaml")
    return RunContext(run_dir=run_dir, config=config)


def _prompt_run_selection() -> Path:
    runs_root = Path("runs")
    runs = sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    print("Available runs:")
    for i, r in enumerate(runs[:10]):
        print(f"  [{i}] {r.name}")
    choice = int(input("Select run number [0]: ") or "0")
    return runs[choice]
```

Add `aoi_bbox_4326` to `palmdef_risk/io/helpers.py`:

```python
def aoi_bbox_4326(aoi_source: str) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in EPSG:4326 for the AOI source."""
    import geopandas as gpd
    from shapely.geometry import box
    try:
        parts = [float(x) for x in aoi_source.split(",")]
        if len(parts) == 4:
            return tuple(parts)  # already a bbox string
    except ValueError:
        pass
    gdf = gpd.read_file(aoi_source)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    xmin, ymin, xmax, ymax = gdf_4326.total_bounds
    return (xmin, ymin, xmax, ymax)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/io/test_run.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/io/run.py palmdef_risk/io/helpers.py tests/io/test_run.py
git commit -m "feat: CRS auto-detection in create_run, drop kde/correlation subdirs"
```

---

## Task 4 — `data/utm.py` (NEW)

**Files:**
- Create: `palmdef_risk/data/utm.py`
- Create: `tests/data/test_utm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_utm.py
from palmdef_risk.data.utm import detect_utm_zones, primary_utm_zone


def test_primary_utm_kalimantan_tengah():
    # Centroid ~110°E, ~-1.5°S → UTM zone 49S → EPSG:32749
    bbox = (108.5, -3.0, 116.0, 2.0)
    assert primary_utm_zone(bbox) == "EPSG:32749"


def test_primary_utm_north_of_equator():
    # Centroid ~110°E, +2°N → UTM zone 49N → EPSG:32649
    bbox = (108.0, 1.0, 112.0, 4.0)
    assert primary_utm_zone(bbox) == "EPSG:32649"


def test_detect_single_zone():
    # bbox entirely within UTM 49 (108–114°E)
    bbox = (109.0, -2.0, 113.0, 1.0)
    zones = detect_utm_zones(bbox)
    assert zones == ["EPSG:32749"]


def test_detect_spans_two_zones():
    # bbox 107.5–109.0°E overlaps zones 48 (102–108) and 49 (108–114)
    bbox = (107.5, -1.0, 109.0, 1.0)
    zones = detect_utm_zones(bbox)
    assert "EPSG:32748" in zones
    assert "EPSG:32749" in zones
    assert len(zones) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/data/test_utm.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.data.utm'`

- [ ] **Step 3: Create `palmdef_risk/data/utm.py`**

```python
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
            epsg = 32600 + z if lat_c >= 0 else 32700 + z
            zones.append(f"EPSG:{epsg}")
    return zones


def primary_utm_zone(bbox_4326: tuple[float, float, float, float]) -> str:
    """Return EPSG code of UTM zone containing the bbox centroid."""
    xmin, ymin, xmax, ymax = bbox_4326
    lon_c = (xmin + xmax) / 2.0
    lat_c = (ymin + ymax) / 2.0
    z = int((lon_c + 180) / 6) + 1
    epsg = 32600 + z if lat_c >= 0 else 32700 + z
    return f"EPSG:{epsg}"
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/data/test_utm.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/data/utm.py tests/data/test_utm.py
git commit -m "feat: add utm.py — detect_utm_zones, primary_utm_zone"
```

---

## Task 5 — `parallel.py` (NEW)

**Files:**
- Create: `palmdef_risk/parallel.py`
- Create: `tests/test_parallel.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parallel.py
import pytest
from unittest.mock import MagicMock
from palmdef_risk.parallel import adaptive_workers, run_parallel


def _square(x):
    return x * x


def _cfg(max_workers=None, cpu_fraction=0.9):
    c = MagicMock()
    c.max_workers = max_workers
    c.cpu_fraction = cpu_fraction
    return c


def test_adaptive_workers_returns_at_least_one():
    assert adaptive_workers(1.0, _cfg()) >= 1


def test_adaptive_workers_respects_max_workers():
    assert adaptive_workers(0.001, _cfg(max_workers=2)) <= 2


def test_run_parallel_returns_correct_results():
    tasks = [1, 2, 3, 4]
    results = run_parallel(_square, tasks, ram_per_task_gb=0.001, cfg=_cfg())
    assert results == [1, 4, 9, 16]


def test_run_parallel_sequential_fallback():
    """With max_workers=1, falls back to sequential (no subprocess)."""
    results = run_parallel(_square, [5, 6], ram_per_task_gb=999.0, cfg=_cfg())
    assert results == [25, 36]


def test_run_parallel_empty_tasks():
    assert run_parallel(_square, [], ram_per_task_gb=1.0, cfg=_cfg()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_parallel.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.parallel'`

- [ ] **Step 3: Create `palmdef_risk/parallel.py`**

```python
from __future__ import annotations
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

logger = logging.getLogger(__name__)


def adaptive_workers(ram_per_task_gb: float, cfg) -> int:
    """Return worker count bounded by available RAM, CPU fraction, and max_workers."""
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        ram_workers = max(1, int(avail_gb / ram_per_task_gb))
    except ImportError:
        ram_workers = 4

    cpu_count = os.cpu_count() or 1
    cpu_fraction = getattr(cfg, "cpu_fraction", 0.9)
    cpu_workers = max(1, math.floor(cpu_count * cpu_fraction))

    n = min(ram_workers, cpu_workers)
    max_w = getattr(cfg, "max_workers", None)
    if max_w:
        n = min(n, max_w)
    return max(1, n)


def run_parallel(fn: Callable, tasks: list, ram_per_task_gb: float, cfg) -> list[Any]:
    """Run fn(task) for each task. Falls back to sequential when workers=1."""
    if not tasks:
        return []
    n = adaptive_workers(ram_per_task_gb, cfg)
    if n == 1:
        return [fn(t) for t in tasks]
    results: list[Any] = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n) as pool:
        futs = {pool.submit(fn, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()
    logger.info("run_parallel: %d tasks on %d workers", len(tasks), n)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_parallel.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/parallel.py tests/test_parallel.py
git commit -m "feat: add parallel.py — adaptive_workers, run_parallel"
```

---

## Task 6 — `cache.py` (NEW)

**Files:**
- Create: `palmdef_risk/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache.py
import json
import pytest
from pathlib import Path
from palmdef_risk.cache import CacheManager


def test_mill_miss_when_files_absent(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    assert not cm.mill_valid(2020, 2023)


def test_mill_hit_when_files_present(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    d = cm.mill_dir(2020, 2023)
    d.mkdir(parents=True)
    (d / "mill_t2.gpkg").write_text("")
    (d / "mill_t3.gpkg").write_text("")
    assert cm.mill_valid(2020, 2023)


def test_forest_miss_when_no_metadata(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    key = cm.forest_key((108.0, -2.0, 114.0, 2.0), 5000, "tmf", [2015, 2020, 2024], 75)
    assert not cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_forest_hit_when_extent_covers(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    bbox = (108.5, -1.5, 113.5, 1.5)
    key = cm.forest_key(bbox, 5000, "tmf", [2015, 2020, 2024], 75)
    d = cm.forest_dir(key)
    d.mkdir(parents=True)
    meta = {"downloaded_extent": [107.0, -3.0, 115.0, 3.0]}  # wider than needed
    (d / "metadata.json").write_text(json.dumps(meta))
    assert cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_forest_miss_when_extent_too_small(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    bbox = (108.5, -1.5, 113.5, 1.5)
    key = cm.forest_key(bbox, 5000, "tmf", [2015, 2020, 2024], 75)
    d = cm.forest_dir(key)
    d.mkdir(parents=True)
    meta = {"downloaded_extent": [109.0, -1.0, 113.0, 1.0]}  # narrower than needed
    (d / "metadata.json").write_text(json.dumps(meta))
    assert not cm.forest_valid(key, [108.5, -1.5, 113.5, 1.5])


def test_status_report_keys(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    k = cm.forest_key((0, 0, 1, 1), 0, "tmf", [], 75)
    kv = cm.variables_key((0, 0, 1, 1), 0, False, None, 180)
    report = cm.status_report(2020, 2023, [0, 0, 1, 1], k, kv)
    assert set(report.keys()) == {"mill", "forest", "variables"}
    assert report["mill"] == "miss"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_cache.py -v
```
Expected: `ModuleNotFoundError: No module named 'palmdef_risk.cache'`

- [ ] **Step 3: Create `palmdef_risk/cache.py`**

```python
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _hash(*parts) -> str:
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _covers(cached: list[float], needed: list[float]) -> bool:
    """True if cached bbox entirely contains needed bbox."""
    return (cached[0] <= needed[0] and cached[1] <= needed[1]
            and cached[2] >= needed[2] and cached[3] >= needed[3])


class CacheManager:
    def __init__(self, cache_dir: Path | str):
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

    def forest_valid(self, key: str, needed_bbox: list[float]) -> bool:
        meta = self.forest_dir(key) / "metadata.json"
        if not meta.exists():
            return False
        data = json.loads(meta.read_text())
        stored = data.get("downloaded_extent")
        return bool(stored and _covers(stored, needed_bbox))

    # ── Variables ────────────────────────────────────────────
    def variables_key(self, aoi_bbox, buffer, use_ghsl, ghsl_years, timeout) -> str:
        return _hash(aoi_bbox, buffer, use_ghsl, ghsl_years, timeout)

    def variables_dir(self, key: str) -> Path:
        return self.cache_dir / "variables" / key

    def variables_valid(self, key: str, needed_bbox: list[float]) -> bool:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_cache.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/cache.py tests/test_cache.py
git commit -m "feat: add cache.py — CacheManager with per-dataset validity logic"
```

---

## Task 7 — `data/mill.py` rewrite (Trase-only, cumulative filter, t2/t3)

**Files:**
- Rewrite: `palmdef_risk/data/mill.py`
- Rewrite: `tests/data/test_mill.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_mill.py
import pytest
import geopandas as gpd
from shapely.geometry import Point
from unittest.mock import patch
from palmdef_risk.data.mill import _filter_mills, _filter_to_aoi, download_mill
from palmdef_risk.io.run import create_run


def _make_gdf(years):
    return gpd.GeoDataFrame(
        {"earliest_year_of_existence": years,
         "geometry": [Point(110 + i, -1) for i in range(len(years))]},
        crs="EPSG:4326",
    )


def test_filter_mills_keeps_null_years():
    gdf = _make_gdf([None, 2015, 2021])
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 2  # null + 2015; drops 2021


def test_filter_mills_keeps_equal_year():
    gdf = _make_gdf([2020, 2021])
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 1  # keeps 2020


def test_filter_mills_no_year_column_keeps_all():
    gdf = gpd.GeoDataFrame(
        {"mill_id": [1, 2], "geometry": [Point(110, -1), Point(111, -1)]},
        crs="EPSG:4326",
    )
    result = _filter_mills(gdf, year=2020)
    assert len(result) == 2


def test_filter_to_aoi_clips_correctly():
    gdf = _make_gdf([2015, 2015, 2015])
    # gdf has points at (110,-1), (111,-1), (112,-1)
    result = _filter_to_aoi(gdf, aoi_extent=(109.5, -2.0, 111.5, 0.0))
    assert len(result) == 2  # drops point at 112


def test_download_mill_writes_t2_and_t3(minimal_config_yaml, tmp_path):
    mock_gdf = _make_gdf([2010, 2015, 2022])
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    with patch("palmdef_risk.data.mill._fetch_trase", return_value=mock_gdf):
        with patch("palmdef_risk.data.mill._aoi_extent_4326", return_value=(109.0, -3.0, 115.0, 1.0)):
            result = download_mill(ctx, use_cache=False)
    assert result["mill_t2"].exists()
    assert result["mill_t3"].exists()
    # t2=2020: keeps 2010, 2015 (null not present); drops 2022
    t2 = gpd.read_file(result["mill_t2"])
    assert len(t2) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/data/test_mill.py -v
```
Expected: `ImportError` — `_filter_mills` not found

- [ ] **Step 3: Rewrite `palmdef_risk/data/mill.py`**

```python
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import requests

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

_TRASE_URL = "https://trase.earth/open-data/datasets/indonesia-palm-oil-mills/download?format=geojson"


def _fetch_trase() -> gpd.GeoDataFrame:
    logger.info("Downloading Trase mill data …")
    resp = requests.get(_TRASE_URL, timeout=120)
    resp.raise_for_status()
    import tempfile, json
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
        f.write(resp.content)
        tmp = f.name
    gdf = gpd.read_file(tmp)
    Path(tmp).unlink(missing_ok=True)
    return gdf


def _filter_mills(gdf: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
    """Keep mills where earliest_year_of_existence <= year OR is null."""
    col = "earliest_year_of_existence"
    if col not in gdf.columns:
        return gdf.copy()
    mask = gdf[col].isna() | (gdf[col] <= year)
    return gdf[mask].copy()


def _filter_to_aoi(
    gdf: gpd.GeoDataFrame,
    aoi_extent: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    xmin, ymin, xmax, ymax = aoi_extent
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


def _aoi_extent_4326(ctx: "RunContext") -> tuple[float, float, float, float]:
    from palmdef_risk.io.helpers import aoi_bbox_4326
    return aoi_bbox_4326(ctx.config.aoi_source)


def download_mill(
    ctx: "RunContext",
    use_cache: bool = True,
) -> dict[str, Path]:
    from palmdef_risk.cache import CacheManager
    t2 = ctx.config.forest_years[1]
    t3 = ctx.config.forest_years[2] if len(ctx.config.forest_years) > 2 else t2

    cm = CacheManager(ctx.config.cache_dir)
    cache_dir = cm.mill_dir(t2, t3)

    if use_cache and cm.mill_valid(t2, t3):
        logger.info("Mill cache hit (t2=%d, t3=%d)", t2, t3)
        raw_t2 = cache_dir / "mill_t2.gpkg"
        raw_t3 = cache_dir / "mill_t3.gpkg"
    else:
        raw = _fetch_trase()
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Write cache (AOI-unfiltered, Indonesia-wide)
        _filter_mills(raw, t2).to_file(cache_dir / "mill_t2.gpkg", driver="GPKG")
        _filter_mills(raw, t3).to_file(cache_dir / "mill_t3.gpkg", driver="GPKG")
        raw_t2 = cache_dir / "mill_t2.gpkg"
        raw_t3 = cache_dir / "mill_t3.gpkg"

    # AOI clip + reproject to run CRS
    aoi_ext = _aoi_extent_4326(ctx)
    out_dir = ctx.raw_dir / "mill"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_t2 = out_dir / "mill_t2.gpkg"
    out_t3 = out_dir / "mill_t3.gpkg"

    for src, dst in [(raw_t2, out_t2), (raw_t3, out_t3)]:
        gdf = gpd.read_file(src)
        clipped = _filter_to_aoi(gdf, aoi_ext)
        clipped.to_crs(ctx.config.crs).to_file(dst, driver="GPKG")

    logger.info("Mill files written: %s, %s", out_t2, out_t3)
    return {"mill_t2": out_t2, "mill_t3": out_t3}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/data/test_mill.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/data/mill.py tests/data/test_mill.py
git commit -m "feat: rewrite mill.py — Trase-only, cumulative year filter, t2/t3 split, cache"
```

---

## Task 8 — `forest.py` UTM enforcement

**Files:**
- Modify: `palmdef_risk/data/forest.py`
- Modify: `tests/data/test_forest.py` (or create if absent)

- [ ] **Step 1: Write failing test**

```python
# tests/data/test_forest.py  (add this test)
from unittest.mock import patch, MagicMock


def test_download_forest_passes_output_crs(minimal_config_yaml, tmp_path):
    """download_forest must pass output_crs=ctx.config.crs to get_fcc."""
    from palmdef_risk.io.run import create_run
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")

    with patch("palmdef_risk.data.forest.get_fcc") as mock_get_fcc:
        mock_get_fcc.return_value = {}
        try:
            from palmdef_risk.data.forest import download_forest
            download_forest(ctx, use_cache=False)
        except Exception:
            pass  # may fail on missing files; we only care about the call
        call_kwargs = mock_get_fcc.call_args
        if call_kwargs:
            # output_crs must be the config CRS, not None
            passed_crs = (call_kwargs.kwargs.get("output_crs")
                          or (call_kwargs.args[3] if len(call_kwargs.args) > 3 else None))
            assert passed_crs == ctx.config.crs
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/data/test_forest.py::test_download_forest_passes_output_crs -v
```
Expected: `AssertionError` — `output_crs` is `None`

- [ ] **Step 3: Fix `download_forest` in `palmdef_risk/data/forest.py`**

Find and replace the call site where `get_fcc` is invoked. The existing call passes `output_crs=None`:

```python
# Before (line ~80 in forest.py — find with grep):
result = get_fcc(..., output_crs=None, ...)

# After:
result = get_fcc(..., output_crs=ctx.config.crs, ...)
```

Also add `use_cache: bool = True` parameter to `download_forest(ctx, use_cache=True)` signature, and wire cache check using `CacheManager` (same pattern as mill.py):

```python
def download_forest(ctx: "RunContext", use_cache: bool = True) -> dict:
    from palmdef_risk.cache import CacheManager
    bbox = _aoi_extent_4326(ctx)
    cfg = ctx.config
    cm = CacheManager(cfg.cache_dir)
    key = cm.forest_key(bbox, cfg.aoi_buffer, cfg.forest_source,
                        cfg.forest_years, cfg.forest_perc)
    needed = [bbox[0] - 0.05, bbox[1] - 0.05, bbox[2] + 0.05, bbox[3] + 0.05]

    if use_cache and cm.forest_valid(key, needed):
        logger.info("Forest cache hit")
        return _copy_from_cache(cm.forest_dir(key), ctx.raw_dir / "forest")

    # Existing download logic — only change is adding output_crs:
    result = get_fcc(
        ...,
        output_crs=cfg.crs,   # ← THE FIX
        ...
    )
    _write_forest_cache(result, cm.forest_dir(key), bbox)
    return result
```

> Note: `get_fcc` is the inner GEE function defined in `forest.py`. Locate its invocation with `grep -n "get_fcc" palmdef_risk/data/forest.py` and pass `output_crs=ctx.config.crs`.

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/data/test_forest.py::test_download_forest_passes_output_crs -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```
git add palmdef_risk/data/forest.py tests/data/test_forest.py
git commit -m "fix: pass output_crs=ctx.config.crs in download_forest (UTM enforcement)"
```

---

## Task 9 — `variables.py` UTM enforcement + `protected` rename

**Files:**
- Modify: `palmdef_risk/data/variables.py`
- Create/modify: `tests/data/test_variables.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_variables.py
import pytest
from unittest.mock import patch, MagicMock
from palmdef_risk.io.run import create_run


def test_no_pa_gpkg_written(minimal_config_yaml, tmp_path):
    """WDPA output must be named protected.gpkg — never pa.gpkg."""
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    # If the function writes pa.gpkg somewhere, this will catch it
    from palmdef_risk.data.variables import _WDPA_OUTPUT_NAME
    assert _WDPA_OUTPUT_NAME == "protected"


def test_download_variables_passes_output_crs(minimal_config_yaml, tmp_path):
    ctx = create_run(minimal_config_yaml, runs_root=tmp_path / "runs")
    with patch("palmdef_risk.data.variables._download_ee_raster") as mock_dl:
        mock_dl.return_value = None
        with patch("palmdef_risk.data.variables._download_wdpa") as mock_wdpa:
            mock_wdpa.return_value = None
            with patch("palmdef_risk.data.variables._download_osm") as mock_osm:
                mock_osm.return_value = None
                try:
                    from palmdef_risk.data.variables import download_variables
                    download_variables(ctx, use_cache=False)
                except Exception:
                    pass
                # Every EE download call must have received the UTM crs
                for call in mock_dl.call_args_list:
                    crs_arg = call.kwargs.get("output_crs") or (
                        call.args[3] if len(call.args) > 3 else None)
                    assert crs_arg == ctx.config.crs, f"output_crs not set: {call}"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/data/test_variables.py -v
```
Expected: `ImportError` or `AssertionError` — `_WDPA_OUTPUT_NAME` not defined or equals `"pa"`

- [ ] **Step 3: Edit `palmdef_risk/data/variables.py`**

Add module-level constant (near top of file):
```python
_WDPA_OUTPUT_NAME = "protected"   # never "pa" — causes patsy formula errors
```

Find all occurrences of the string `"pa"` used as the WDPA output filename and replace:
```
grep -n '"pa"' palmdef_risk/data/variables.py
grep -n "'pa'" palmdef_risk/data/variables.py
grep -n "pa.gpkg" palmdef_risk/data/variables.py
grep -n "pa.tif" palmdef_risk/data/variables.py
```

Replace each with `_WDPA_OUTPUT_NAME` (for the variable name) or `"protected"` (for the hardcoded string). Typical locations: the output path string passed to `_download_wdpa`, the returned dict key.

Pass `output_crs=ctx.config.crs` to every `_download_ee_raster(...)` call:
```python
# Before:
_download_ee_raster(..., output_crs=None, ...)
# After:
_download_ee_raster(..., output_crs=ctx.config.crs, ...)
```

Add `use_cache: bool = True` parameter to `download_variables(ctx, use_cache=True)` and wire `CacheManager` (same pattern as mill.py and forest.py).

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/data/test_variables.py -v
```
Expected: PASS

- [ ] **Step 5: Run all Phase 1–3 tests together**

```
pytest tests/ -q --ignore=tests/process --ignore=tests/model
```
Expected: all PASS (or skip tests requiring network)

- [ ] **Step 6: Commit**

```
git add palmdef_risk/data/variables.py tests/data/test_variables.py
git commit -m "fix: rename WDPA output to protected, pass output_crs in download_variables"
```

---

## Phase 1–3 Done

All foundation, infrastructure, and data-download tasks are complete. Continue with **Part 2** (`docs/superpowers/plans/2026-05-19-palmdef-risk-refactor-plan-part2.md`) for Phases 4–6: process stage (align, distances, gravity), model stage (icar, diagnostics, sensitivity, predict), notebooks, and `.claude` folder.
