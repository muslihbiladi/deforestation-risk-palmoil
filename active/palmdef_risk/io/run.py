from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import os
import sys
import shutil

from palmdef_risk.io.config import RunConfig


def _fix_proj_path() -> None:
    """Override stale PROJ db (e.g. from PostgreSQL/PostGIS) with the one
    from the active Python environment.  Safe no-op if the db is not found.
    """
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",   # Windows conda
        Path(sys.exec_prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "share" / "proj",                # Linux/macOS conda
        Path(sys.exec_prefix) / "share" / "proj",
    ]
    for proj_dir in candidates:
        if (proj_dir / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(proj_dir)
            os.environ["PROJ_DATA"] = str(proj_dir)
            logging.getLogger(__name__).debug("PROJ path set to %s", proj_dir)
            return


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
    "data/intermediate",
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
    _fix_proj_path()
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
        _print_dry_run_summary(config)
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
    print(f"run_dir={run_dir.resolve()}")
    return ctx


def create_or_resume_run(
    config_path: str | Path,
    resume: bool = False,
    runs_root: str | Path = "runs",
) -> RunContext:
    """Create a new run or resume the most recent matching one.

    When resume=True, searches runs_root for the most recently modified folder
    whose name starts with ``project_area_task_`` (from the config).  If a
    match is found that folder is reloaded; otherwise a new run is created.
    When resume=False (default), always creates a fresh timestamped folder.
    """
    if resume:
        cfg_tmp = RunConfig.from_yaml(config_path)
        prefix = f"{cfg_tmp.project}_{cfg_tmp.area}_{cfg_tmp.task}_"
        runs_root_path = Path(runs_root)
        if runs_root_path.exists():
            candidates = sorted(
                [p for p in runs_root_path.iterdir()
                 if p.is_dir() and p.name.startswith(prefix)],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                run_dir = candidates[0]
                logging.getLogger(__name__).info("Resuming run: %s", run_dir)
                print(f"Resuming existing run: {run_dir.name}")
                print(f"run_dir={run_dir.resolve()}")
                return load_run(run_dir)
        logging.getLogger(__name__).warning(
            "No existing run found matching '%s*', creating new run.", prefix
        )
        print(f"No existing run found for '{prefix}*', creating new run.")
    return create_run(config_path, runs_root=runs_root)


def load_run(run_dir: str | Path | None = None) -> RunContext:
    _fix_proj_path()
    if run_dir is None:
        run_dir = _prompt_run_selection()
    run_dir = Path(run_dir)
    if not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    config = RunConfig.from_yaml(run_dir / "config.yaml")
    if config.crs is None:
        from palmdef_risk.data.utm import primary_utm_zone
        from palmdef_risk.io.helpers import aoi_bbox_4326
        bbox = aoi_bbox_4326(config.aoi_source)
        config.crs = primary_utm_zone(bbox)
        logging.getLogger(__name__).info("Auto-detected CRS: %s", config.crs)
    return RunContext(run_dir=run_dir, config=config)


def _prompt_run_selection() -> Path:
    runs_root = Path("runs")
    if not runs_root.exists():
        raise RuntimeError("No runs/ folder found. Create a run first.")
    runs = sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise RuntimeError("runs/ folder is empty.")
    print("Available runs:")
    for i, r in enumerate(runs[:10]):
        print(f"  [{i}] {r.name}")
    choice = int(input("Select run number [0]: ") or "0")
    return runs[choice]


def _print_dry_run_summary(config: RunConfig) -> None:
    print(f"  project      : {config.project}")
    print(f"  area         : {config.area}")
    print(f"  task         : {config.task}")
    print(f"  forest       : {config.forest_source} {config.forest_years}")
    print(f"  crs          : {config.crs}")
    print(f"  peatland     : {config.peatland_path} ({config.peatland_type})")
    print(f"  hgu          : {config.hgu_path}")
    print(f"  plantation   : source={config.plantation_source} "
          f"t2={config.plantation_t2}, t3={config.plantation_t3}")
    print(f"  mill source  : {config.mill_source}")
    print(f"  variants     : {config.model_variants}")
    print(f"  sigma_km     : {config.sigma_km}")
    print(f"  radius_km    : {config.radius_km}")
