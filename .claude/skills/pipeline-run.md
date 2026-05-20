---
name: pipeline-run
description: Run the full palmdef_risk pipeline (all 3 notebooks) via papermill for a given config file.
---

# pipeline-run

Execute all three pipeline notebooks in sequence using papermill.

## Usage

```
/pipeline-run configs/my_run.yaml
```

## What this does

1. Runs `notebooks/01_download.ipynb` with `config_path` injected
2. Runs `notebooks/02_process.ipynb` loading the run created in step 1
3. Runs `notebooks/03_model.ipynb` loading the same run

## Implementation

When the user invokes `/pipeline-run <config_path>`, execute:

```bash
conda activate conda-far
python run.py --config <config_path>
```

Or, to run a single stage:
```bash
python run.py --config <config_path> --notebook 01_download
python run.py --config <config_path> --notebook 02_process --run-dir runs/<run_dir>
python run.py --config <config_path> --notebook 03_model --run-dir runs/<run_dir>
```

Progress is logged to `runs/<run_dir>/logs/run.log`.
