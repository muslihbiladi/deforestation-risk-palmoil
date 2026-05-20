---
name: pipeline-check
description: Inspect a palmdef_risk run folder and report which stages are complete, partial, or missing.
---

# pipeline-check

Check the status of a run folder.

## Usage

```
/pipeline-check runs/wri_kalteng_20250501_120000
```

## What this does

Inspects the run folder and reports per-stage completion:

| Stage | Done if | Key files |
|---|---|---|
| Stage 1 (download) | raw/forest/ + raw/variables/ + raw/mill/ populated | `forest_t2.tif`, `protected.gpkg`, `mill_t2.gpkg` |
| Stage 2 (process) | data/ flat rasters present | `dist_road.tif`, `gravity_resid.tif`, `hgu_signed_dist.tif` |
| Stage 3 (model) | output/models/ populated | `mod_A.pkl`, `vif.json`, `moran.json` |

## Implementation

When the user invokes `/pipeline-check <run_dir>`, use Read and Glob tools to check:

```python
from pathlib import Path

raw = Path(run_dir) / "data" / "raw"
s1_ok = all([
    (raw / "forest" / "forest_t2.tif").exists(),
    (raw / "variables" / "protected.gpkg").exists(),
    (raw / "mill" / "mill_t2.gpkg").exists(),
])

data = Path(run_dir) / "data"
s2_ok = all([
    (data / "dist_road.tif").exists(),
    (data / "gravity_resid.tif").exists(),
    (data / "hgu_signed_dist.tif").exists(),
])

out = Path(run_dir) / "output"
s3_ok = (out / "models").exists() and any((out / "models").glob("*/mod_*.pkl"))
```

Report as a table:
```
Stage 1 (download):  ✓ complete
Stage 2 (process):   ✓ complete
Stage 3 (model):     ✗ missing  — no mod_*.pkl found
```
