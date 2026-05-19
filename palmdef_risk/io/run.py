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
    "data/intermediate/kde",
    "output/models",
    "output/diagnostics",
    "output/predictions",
    "output/correlation",
    "output/scenarios",
    "output/maps",
    "logs",
]


def create_run(
    config_path: str | Path,
    runs_root: str | Path = "runs",
    dry_run: bool = False,
) -> RunContext:
    config_path = Path(config_path)
    config = RunConfig.from_yaml(config_path)

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
    print(f"  plantation   : t2={config.plantation_t2}, t3={config.plantation_t3}")
    print(f"  mill source  : {config.mill_source}")
    print(f"  variants     : {config.model_variants}")
    print(f"  sigma_km     : {config.sigma_km}")
    print(f"  radius_km    : {config.radius_km}")
