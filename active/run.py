"""CLI runner for the palmdef_risk pipeline.

Usage
-----
  python run.py --config configs/my_run.yaml
  python run.py --config configs/my_run.yaml --notebook 01_download
  python run.py --config configs/my_run.yaml --dry-run

All three notebooks are executed in sequence by default.  Pass --notebook to
run a single stage.  The run folder is created by the first notebook that
executes; subsequent notebooks reuse it via the injected run_dir parameter.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path


_NOTEBOOKS = ["01_download", "02_process", "03_model"]

_NB_DIR = Path(__file__).parent / "notebooks"
_OUTPUT_DIR = Path(__file__).parent / "notebook_outputs"


def _run_notebook(
    name: str,
    config_path: str,
    runs_root: str,
    run_dir: str | None,
    dry_run: bool,
) -> str | None:
    """Execute one notebook via papermill. Returns run_dir string."""
    import papermill as pm

    nb_in = _NB_DIR / f"{name}.ipynb"
    if not nb_in.exists():
        print(f"[error] notebook not found: {nb_in}", file=sys.stderr)
        sys.exit(1)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nb_out = _OUTPUT_DIR / f"{name}_output.ipynb"

    params: dict = {
        "config_path": config_path,
        "runs_root": runs_root,
    }
    if run_dir:
        params["run_dir"] = run_dir

    if dry_run:
        print(f"[dry-run] would execute: {nb_in}")
        print(f"          parameters:    {params}")
        return run_dir

    print(f"\n{'='*60}")
    print(f"  Executing {name}")
    print(f"{'='*60}")
    t0 = time.time()

    result = pm.execute_notebook(
        str(nb_in),
        str(nb_out),
        parameters=params,
        kernel_name="conda-far",
    )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s  ->  {nb_out}")

    # Extract run_dir from notebook output cells (printed as "run_dir=<path>")
    if run_dir is None:
        for cell in result.cells:
            for output in cell.get("outputs", []):
                text = output.get("text", "")
                for line in (text if isinstance(text, list) else [text]):
                    if line.startswith("run_dir="):
                        return line.split("=", 1)[1].strip()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="palmdef_risk pipeline runner"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file (e.g. configs/my_run.yaml)"
    )
    parser.add_argument(
        "--notebook",
        choices=_NOTEBOOKS + [n.replace("0", "", 1) for n in _NOTEBOOKS],
        default=None,
        help="Run a single notebook stage. Omit to run all three in sequence.",
    )
    parser.add_argument(
        "--runs-root", default="runs",
        help="Root directory for run folders (default: runs/)"
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Reuse an existing run folder (skips folder creation)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be executed without running anything"
    )
    args = parser.parse_args()

    config_path_obj = Path(args.config).resolve()
    if not config_path_obj.exists():
        print(f"[error] config not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    config_path = str(config_path_obj)
    runs_root = args.runs_root
    run_dir: str | None = args.run_dir

    # Validate config fields (skipped for dry-run — file existence already checked)
    if not args.dry_run:
        from palmdef_risk.io.config import RunConfig
        cfg = RunConfig.from_yaml(config_path)
        errors = cfg.validate()
        if errors:
            print("[error] Config validation failed:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)

    if args.notebook:
        # Normalise "1" → "01_download" etc.
        nb_name = args.notebook if args.notebook.startswith("0") else f"0{args.notebook}"
        _run_notebook(nb_name, config_path, runs_root, run_dir, args.dry_run)
    else:
        for nb_name in _NOTEBOOKS:
            run_dir = _run_notebook(
                nb_name, config_path, runs_root, run_dir, args.dry_run
            )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
