"""Diagnostic report exports for fitted iCAR models.

One function per artifact, plus `export_all_diagnostics(ctx)` that runs them
all. Per-variant outputs land in `<output>/diagnostics/<variant>/`; run-level
outputs (csize, historical fcc map) in `<output>/diagnostics/`.

ROC, calibration, classification report and confusion matrix follow
forestatrisk's tutorial pattern: predict on the same balanced sample used
for fitting (sample.csv) and threshold at 0.5. This matches what
`far.accuracy_indices` is typically invoked with.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from palmdef_risk.io.run import RunContext

logger = logging.getLogger(__name__)

# (output_dir, variant) → (p_hat, y_obs, beta_names). The three per-variant
# exporters each call _predict_in_sample, so without this each run rebuilt the
# patsy design matrices 3x per variant (9x total). The result is deterministic
# per (run output dir, variant) — sample.csv and the fitted state are fixed
# within a run — so caching is safe.
_IN_SAMPLE_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str]]] = {}


def _predict_in_sample(
    ctx: "RunContext", state: dict, variant: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (p_hat, y_obs, beta_names) for variant's training subset."""
    from palmdef_risk.model.icar import load_design_matrix

    cache_key = (str(ctx.output_dir), variant)
    cached = _IN_SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    data, y, x = load_design_matrix(ctx, variant, state["formula"], dropna="base")
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    col_names = list(x.design_info.column_names)
    cell_pos = col_names.index("cell")
    cell_idx = x_arr[:, cell_pos].astype(int)
    x_fixed = np.delete(x_arr, cell_pos, axis=1)
    beta_names = [n for n in col_names if n != "cell"]

    betas = np.asarray(state["betas"]).ravel()
    rho = np.asarray(state["rho"]).ravel()
    eta = x_fixed @ betas + rho[cell_idx]
    p_hat = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-12, 1.0 - 1e-12)
    y_obs = y_arr[:, 0].astype(int)
    result = (p_hat, y_obs, beta_names)
    _IN_SAMPLE_CACHE[cache_key] = result
    return result


def _load_state(ctx: "RunContext", variant: str) -> Optional[dict]:
    pkl = ctx.output_dir / "models" / f"model_{variant}" / f"mod_{variant}.pkl"
    if not pkl.exists():
        logger.warning("Missing %s — skipping variant %s", pkl, variant)
        return None
    with open(pkl, "rb") as fh:
        return pickle.load(fh)


def _autocorr(chain: np.ndarray, nlags: int) -> np.ndarray:
    """FFT-based autocorrelation up to nlags."""
    x = chain - chain.mean()
    n = len(x)
    if x.std() == 0:
        return np.zeros(nlags + 1)
    f = np.fft.fft(x, n=2 * n)
    acf = np.real(np.fft.ifft(f * np.conjugate(f)))[: nlags + 1]
    acf /= acf[0] if acf[0] != 0 else 1.0
    return acf


def _effective_sample_size(chain: np.ndarray) -> float:
    """ESS via cumulative autocorrelation sum, truncated at first negative lag."""
    n = len(chain)
    if np.asarray(chain).std() == 0:
        return float(n)
    nlags = min(n // 3, 1000)
    acf = _autocorr(chain, nlags)
    s = 0.0
    for k in range(1, len(acf)):
        if acf[k] < 0:
            break
        s += acf[k]
    ess = n / (1.0 + 2.0 * s)
    return float(min(max(ess, 1.0), n))


# ---------------------------------------------------------------------------
# Run-level exports

def export_csize(ctx: "RunContext") -> Path:
    """Write iCAR cell size (km) to diagnostics/csize_icar.txt."""
    out = ctx.output_dir / "diagnostics" / "csize_icar.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{float(ctx.config.csize)}\n")
    return out


def export_fcc_map(ctx: "RunContext") -> Optional[Path]:
    """Plot historical forest-cover change (fcc123.tif) via far.plot.fcc123."""
    import forestatrisk as far
    fcc = ctx.data_dir / "fcc123.tif"
    if not fcc.exists():
        logger.warning("fcc123.tif not found — skipping fcc map")
        return None
    out = ctx.output_dir / "diagnostics" / "fcc_history.png"
    if out.exists():
        logger.info("fcc_history.png exists — skipping")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Colors: class 1 (first defor period) = red, class 2 (second) = orange,
    # class 3 (forest at t3) = green — matches the project's legend style.
    far.plot.fcc123(
        input_fcc_raster=str(fcc),
        output_file=str(out),
        col=[(227, 26, 28, 255), (255, 165, 0, 255), (34, 139, 34, 255)],
    )
    return out


# ---------------------------------------------------------------------------
# Per-variant exports

def export_summary_table(ctx: "RunContext", variant: str) -> Optional[Path]:
    """Write summary_icar.txt with posterior mean/std/CI per parameter."""
    out = ctx.output_dir / "diagnostics" / variant / "summary_icar.txt"
    if out.exists():
        logger.info("summary_icar.txt [%s] exists — skipping", variant)
        return out
    state = _load_state(ctx, variant)
    if state is None:
        return None
    _, _, beta_names = _predict_in_sample(ctx, state, variant)

    mcmc = np.asarray(state["mcmc"])  # (nsamp, npar+2): betas, Vrho, Deviance
    param_names = beta_names + ["Vrho", "Deviance"]
    if mcmc.shape[1] != len(param_names):
        logger.warning(
            "Variant %s: mcmc has %d cols but %d param names — truncating",
            variant, mcmc.shape[1], len(param_names),
        )
        param_names = param_names[: mcmc.shape[1]]

    mean = mcmc.mean(axis=0)
    std = mcmc.std(axis=0)
    ci_lo = np.percentile(mcmc, 2.5, axis=0)
    ci_hi = np.percentile(mcmc, 97.5, axis=0)

    width = max(len(n) for n in param_names) + 2
    lines = [
        "Binomial logistic regression with iCAR process",
        f"  Model: {state['formula']}",
        "  Posteriors:",
        f"  {'':>{width}} {'Mean':>10} {'Std':>10} {'CI_low':>10} {'CI_high':>10}",
    ]
    for i, n in enumerate(param_names):
        lines.append(
            f"  {n:>{width}} {mean[i]:>10.4g} {std[i]:>10.4g} "
            f"{ci_lo[i]:>10.4g} {ci_hi[i]:>10.4g}"
        )
    out = ctx.output_dir / "diagnostics" / variant / "summary_icar.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def export_accuracy(ctx: "RunContext", variant: str) -> Optional[tuple[Path, Path]]:
    """Write ROC+calibration plot and accuracy_summary.txt (AUC + report + CM)."""
    out_dir = ctx.output_dir / "diagnostics" / variant
    plot_path = out_dir / "roc_calibration.png"
    txt_path = out_dir / "accuracy_summary.txt"
    if plot_path.exists() and txt_path.exists():
        logger.info("accuracy outputs [%s] exist — skipping", variant)
        return plot_path, txt_path
    from sklearn.metrics import (
        roc_curve, auc, classification_report, confusion_matrix,
    )
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    state = _load_state(ctx, variant)
    if state is None:
        return None
    p_hat, y_obs, _ = _predict_in_sample(ctx, state, variant)

    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_obs, p_hat)
    auc_val = auc(fpr, tpr)
    obs_freq, mean_pred = calibration_curve(y_obs, p_hat, n_bins=10, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve — iCAR model {variant}")
    ax1.legend(loc="lower right")

    ax2.plot(mean_pred, obs_freq, "o-")
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Observed deforestation rate")
    ax2.set_title("Calibration plot")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    y_pred = (p_hat >= 0.5).astype(int)
    report = classification_report(
        y_obs, y_pred, target_names=["Forest", "Deforested"], digits=4
    )
    cm = confusion_matrix(y_obs, y_pred)
    txt_path.write_text(f"AUC: {auc_val:.3f}\n\n{report}\nConfusion matrix:\n{cm}\n")
    return plot_path, txt_path


def export_mcmc_diagnostics(
    ctx: "RunContext", variant: str
) -> Optional[tuple[Path, Path, Path]]:
    """Write trace plots, autocorrelation plots, and ESS table for the MCMC chain."""
    out_dir = ctx.output_dir / "diagnostics" / variant
    traces_path = out_dir / "mcmc_traces.png"
    autocorr_path = out_dir / "mcmc_autocorr.png"
    ess_path = out_dir / "mcmc_ess.txt"
    if traces_path.exists() and autocorr_path.exists() and ess_path.exists():
        logger.info("mcmc diagnostics [%s] exist — skipping", variant)
        return traces_path, autocorr_path, ess_path
    import matplotlib.pyplot as plt

    state = _load_state(ctx, variant)
    if state is None:
        return None
    _, _, beta_names = _predict_in_sample(ctx, state, variant)

    mcmc = np.asarray(state["mcmc"])
    # Plot all betas + Vrho (skip Deviance — uninformative for mixing diagnostics)
    n_betas = len(beta_names)
    if mcmc.shape[1] >= n_betas + 1:
        param_names = beta_names + ["Vrho"]
        cols = list(range(n_betas)) + [n_betas]
    else:
        param_names = beta_names[: mcmc.shape[1]]
        cols = list(range(len(param_names)))

    out_dir.mkdir(parents=True, exist_ok=True)

    ncols = 3
    nrows = (len(param_names) + ncols - 1) // ncols

    # Trace plots
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, (name, col) in enumerate(zip(param_names, cols)):
        ax = axes_flat[i]
        ax.plot(mcmc[:, col], lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("Iteration")
    for ax in axes_flat[len(param_names):]:
        ax.axis("off")
    fig.suptitle(f"Trace Plots — iCAR MCMC chain ({variant})")
    fig.tight_layout()
    fig.savefig(traces_path, dpi=120)
    plt.close(fig)

    # Autocorrelation plots
    nlags = 40
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 3))
    axes_flat = np.atleast_1d(axes).flatten()
    for i, (name, col) in enumerate(zip(param_names, cols)):
        ax = axes_flat[i]
        acf = _autocorr(mcmc[:, col], nlags)
        ax.stem(range(nlags + 1), acf)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("Lag")
    for ax in axes_flat[len(param_names):]:
        ax.axis("off")
    fig.suptitle(f"Autocorrelation Plots — iCAR MCMC chain ({variant})")
    fig.tight_layout()
    fig.savefig(autocorr_path, dpi=120)
    plt.close(fig)

    # ESS
    total = mcmc.shape[0]
    width = max(len(n) for n in param_names) + 2
    lines = [
        "Effective Sample Size (ESS) per parameter:",
        f"  {'Parameter':<{width}} {'ESS':>10} {'ESS / total':>14}",
        "  " + "-" * (width + 28),
    ]
    for name, col in zip(param_names, cols):
        ess = _effective_sample_size(mcmc[:, col])
        lines.append(f"  {name:<{width}} {ess:>10.1f} {ess / total * 100:>13.1f}%")
    ess_path.write_text("\n".join(lines) + "\n")

    return traces_path, autocorr_path, ess_path


def export_risk_map(ctx: "RunContext", variant: str) -> Optional[Path]:
    """Plot the predicted risk raster (green→red colormap)."""
    import forestatrisk as far
    risk = ctx.output_dir / "predictions" / f"risk_{variant}.tif"
    if not risk.exists():
        logger.warning("risk_%s.tif not found — skipping risk map", variant)
        return None
    out = ctx.output_dir / "diagnostics" / variant / "risk_map.png"
    if out.exists():
        logger.info("risk_map.png [%s] exists — skipping", variant)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    far.plot.prob(input_prob_raster=str(risk), output_file=str(out), legend=True)
    return out


def export_rho_maps(
    ctx: "RunContext", variant: str
) -> Optional[tuple[Optional[Path], Optional[Path]]]:
    """Plot interpolated (smooth) and raw (cell-grid) rho maps."""
    import forestatrisk as far
    model_dir = ctx.output_dir / "models" / f"model_{variant}"
    rho_smooth = model_dir / "rho.tif"
    rho_raw = model_dir / "rho_orig.tif"  # written automatically by far.interpolate_rho

    out_dir = ctx.output_dir / "diagnostics" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_smooth = out_dir / "rho_interpolated.png"
    out_raw = out_dir / "rho_raw.png"

    p1: Optional[Path] = None
    p2: Optional[Path] = None
    if out_smooth.exists():
        logger.info("rho_interpolated.png [%s] exists — skipping", variant)
        p1 = out_smooth
    elif rho_smooth.exists():
        far.plot.rho(input_rho_raster=str(rho_smooth), output_file=str(out_smooth))
        p1 = out_smooth
    else:
        logger.warning("rho.tif not found for variant %s — run prediction first", variant)
    if out_raw.exists():
        logger.info("rho_raw.png [%s] exists — skipping", variant)
        p2 = out_raw
    elif rho_raw.exists():
        far.plot.rho(input_rho_raster=str(rho_raw), output_file=str(out_raw))
        p2 = out_raw
    else:
        logger.warning("rho_orig.tif not found for variant %s", variant)

    if p1 is None and p2 is None:
        return None
    return p1, p2


def export_risk_histogram(ctx: "RunContext", variant: str) -> Optional[Path]:
    """Scatter-style histogram of risk raster pixel counts vs scaled probability."""
    import matplotlib.pyplot as plt
    from osgeo import gdal

    risk = ctx.output_dir / "predictions" / f"risk_{variant}.tif"
    if not risk.exists():
        return None
    out = ctx.output_dir / "diagnostics" / variant / "freq_prob.png"
    if out.exists():
        logger.info("freq_prob.png [%s] exists — skipping", variant)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(str(risk))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    ds = None

    valid = arr[arr != (nodata if nodata is not None else 0)]
    if valid.size == 0:
        logger.warning("Risk raster %s has no valid pixels", risk)
        return None

    # Bins span the UInt16 scaled probability range [1, 65535].
    counts, edges = np.histogram(valid, bins=256, range=(1, 65535))
    centers = (edges[:-1] + edges[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(centers, counts, s=20, color="blue")
    ax.set_title("Frequencies of deforestation probabilities")

    # If config defines risk thresholds (in [0, 1]), overlay dashed lines.
    thresholds = getattr(ctx.config, "risk_thresholds", None) or []
    for t in thresholds:
        ax.axvline(t * 65535, ls="--", color="black")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Runner

def export_all_diagnostics(ctx: "RunContext") -> dict[str, list[Path]]:
    """Run every diagnostic export. Returns {variant_or_'_run': [paths]}.

    Per-artifact failures are logged but do not abort the rest.
    """
    from tqdm.auto import tqdm
    results: dict[str, list[Path]] = {"_run": []}
    run_level = (export_csize, export_fcc_map)
    for fn in tqdm(run_level, desc="Run-level diagnostics", unit="artifact"):
        try:
            p = fn(ctx)
            if p is not None:
                results["_run"].append(p)
        except Exception:
            import traceback
            logger.error("%s failed:\n%s", fn.__name__, traceback.format_exc())

    per_variant = (
        export_summary_table,
        export_accuracy,
        export_mcmc_diagnostics,
        export_risk_map,
        export_rho_maps,
        export_risk_histogram,
    )
    variants = list(ctx.config.model_variants)
    total = len(variants) * len(per_variant)
    with tqdm(total=total, desc="Per-variant diagnostics", unit="artifact") as pbar:
        for variant in variants:
            paths: list[Path] = []
            for fn in per_variant:
                pbar.set_postfix_str(f"{variant}:{fn.__name__}")
                try:
                    out = fn(ctx, variant)
                except Exception:
                    import traceback
                    logger.error(
                        "%s failed for variant %s:\n%s",
                        fn.__name__, variant, traceback.format_exc(),
                    )
                    pbar.update(1)
                    continue
                pbar.update(1)
                if out is None:
                    continue
                if isinstance(out, tuple):
                    paths.extend(p for p in out if p is not None)
                else:
                    paths.append(out)
            results[variant] = paths

    return results
