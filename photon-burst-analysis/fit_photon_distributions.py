"""
fit_photon_distributions.py
============================
For each of 4 CSV files and each photon channel:
  1. Fits Poisson, Negative Binomial, Log-normal, and Gamma distributions via MLE
  2. Ranks fits by AIC (lower = better)
  3. Plots fitted PMF/PDF overlaid on histogram (full distribution, no outlier removal)
  4. Compares skewness, kurtosis, and Jensen-Shannon divergence across files
  5. Prints a clean summary table

Usage:
    python fit_photon_distributions.py file1.csv file2.csv file3.csv file4.csv
"""

import sys
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.special import gammaln
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
PALETTE   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CHANNELS  = ["total_photons", "donor_photons", "acceptor_photons"]
CH_LABELS = ["Total", "Donor", "Acceptor"]

# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    df = df[df["entry_time"] != "SUMMARY"].copy()
    for col in CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=CHANNELS)
    return df

def clean_series(series):
    """Drop NaNs only — no outlier removal."""
    return series.dropna()

def short_name(path):
    import os
    return os.path.splitext(os.path.basename(path))[0]

# ── Distribution fitting ──────────────────────────────────────────────────────

def fit_poisson(data):
    """MLE for Poisson: lambda = mean."""
    try:
        lam = data.mean()
        ll  = np.sum(stats.poisson.logpmf(data.astype(int), lam))
        aic = 2 * 1 - 2 * ll
        return {"name": "Poisson", "params": {"λ": lam}, "aic": aic, "ll": ll}
    except Exception:
        return {"name": "Poisson", "params": {}, "aic": np.inf, "ll": -np.inf}

def fit_negbinom(data):
    """MLE for Negative Binomial via numerical optimisation."""
    try:
        data = data.astype(int)
        mu0  = data.mean()
        var0 = data.var()
        r0   = max(mu0**2 / (var0 - mu0 + 1e-9), 0.5)

        def neg_ll(log_r):
            r = np.exp(log_r)
            p = r / (r + mu0)
            return -np.sum(stats.nbinom.logpmf(data, r, p))

        res = minimize(neg_ll, x0=[np.log(r0)], method="Nelder-Mead")
        r   = np.exp(res.x[0])
        p   = r / (r + mu0)
        ll  = -res.fun
        aic = 2 * 2 - 2 * ll
        return {"name": "Neg. Binomial", "params": {"r": r, "p": p, "μ": mu0}, "aic": aic, "ll": ll}
    except Exception:
        return {"name": "Neg. Binomial", "params": {}, "aic": np.inf, "ll": -np.inf}

def fit_lognormal(data):
    """MLE for Log-normal (continuous approximation)."""
    try:
        pos = data[data > 0]
        if len(pos) < 5:
            return {"name": "Log-normal", "params": {}, "aic": np.inf, "ll": -np.inf}
        mu, sigma = np.mean(np.log(pos)), np.std(np.log(pos))
        ll  = np.sum(stats.lognorm.logpdf(pos, s=sigma, scale=np.exp(mu)))
        aic = 2 * 2 - 2 * ll
        return {"name": "Log-normal", "params": {"μ_log": mu, "σ_log": sigma}, "aic": aic, "ll": ll}
    except Exception:
        return {"name": "Log-normal", "params": {}, "aic": np.inf, "ll": -np.inf}

def fit_gamma(data):
    """MLE for Gamma distribution."""
    pos = data[data > 0]
    if len(pos) < 5 or pos.var() == 0:
        return {"name": "Gamma", "params": {}, "aic": np.inf, "ll": -np.inf}
    try:
        a, loc, scale = stats.gamma.fit(pos, floc=0)
        ll  = np.sum(stats.gamma.logpdf(pos, a, loc=loc, scale=scale))
        aic = 2 * 2 - 2 * ll
        return {"name": "Gamma", "params": {"α": a, "scale": scale}, "aic": aic, "ll": ll}
    except Exception:
        return {"name": "Gamma", "params": {}, "aic": np.inf, "ll": -np.inf}

def fit_all(data):
    results = [fit_poisson(data), fit_negbinom(data),
               fit_lognormal(data), fit_gamma(data)]
    results.sort(key=lambda x: x["aic"])
    return results

# ── PMF/PDF for overlay ───────────────────────────────────────────────────────

def dist_pmf(fit_result, x_vals):
    name   = fit_result["name"]
    params = fit_result["params"]
    x = np.array(x_vals)
    if not params:
        return np.zeros_like(x, dtype=float)
    try:
        if name == "Poisson":
            return stats.poisson.pmf(x.astype(int), params["λ"])
        elif name == "Neg. Binomial":
            return stats.nbinom.pmf(x.astype(int), params["r"], params["p"])
        elif name == "Log-normal":
            return stats.lognorm.pdf(x, s=params["σ_log"], scale=np.exp(params["μ_log"]))
        elif name == "Gamma":
            return stats.gamma.pdf(x, params["α"], scale=params["scale"])
    except Exception:
        pass
    return np.zeros_like(x, dtype=float)

# ── Jensen-Shannon divergence matrix ─────────────────────────────────────────

def js_matrix(datasets):
    """Build NxN JS divergence matrix from a list of Series."""
    n    = len(datasets)
    top  = int(max(d.max() for d in datasets)) + 1
    bins = np.arange(top + 1)
    hists = []
    for d in datasets:
        h, _ = np.histogram(d, bins=bins, density=True)
        h   += 1e-10        # smooth zeros
        hists.append(h / h.sum())

    mat = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        js  = jensenshannon(hists[i], hists[j])
        mat[i, j] = mat[j, i] = js
    return mat

# ── Shape stats ───────────────────────────────────────────────────────────────

def shape_stats(data):
    return {
        "mean":     data.mean(),
        "std":      data.std(),
        "skewness": stats.skew(data),
        "kurtosis": stats.kurtosis(data),   # excess kurtosis (normal = 0)
    }

# ── Printing ──────────────────────────────────────────────────────────────────

def print_summary(all_fits, all_shapes, all_labels, channel_label, js_mat):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Channel: {channel_label}")
    print(sep)

    # Best-fit table
    print(f"\n{'Dataset':<32} {'Best Fit':<16} {'AIC':>9}  {'Skewness':>9}  {'Kurtosis':>9}")
    print("-" * 72)
    for lbl, fits, shape in zip(all_labels, all_fits, all_shapes):
        best = fits[0]
        print(f"  {lbl:<30} {best['name']:<16} {best['aic']:>9.1f}"
              f"  {shape['skewness']:>9.3f}  {shape['kurtosis']:>9.3f}")

    # AIC breakdown for all distributions
    print(f"\n  AIC breakdown (lower = better fit):")
    dist_names = [f["name"] for f in all_fits[0]]
    print(f"  {'Dataset':<30}" + "".join(f"  {n:<16}" for n in dist_names))
    print("  " + "-" * 68)
    for lbl, fits in zip(all_labels, all_fits):
        aic_by_name = {f["name"]: f["aic"] for f in fits}
        row = f"  {lbl:<30}" + "".join(
            f"  {aic_by_name.get(n, float('nan')):>16.1f}" for n in dist_names
        )
        print(row)

    # JS divergence matrix
    print(f"\n  Jensen-Shannon divergence  (0=identical shape, 1=completely different):")
    print(f"  {'':30}" + "".join(f"  {l:<14}" for l in all_labels))
    for i, lbl in enumerate(all_labels):
        row = f"  {lbl:<30}" + "".join(f"  {js_mat[i,j]:<14.4f}" for j in range(len(all_labels)))
        print(row)

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all(all_data, all_fits, all_labels, file_paths):
    n_ch  = len(CHANNELS)
    n_ds  = len(all_labels)

    fig = plt.figure(figsize=(5 * n_ds, 5 * n_ch))
    outer = gridspec.GridSpec(n_ch, 1, figure=fig, hspace=0.55)
    fig.suptitle("Distribution Fitting — Photon Count Channels",
                 fontsize=15, fontweight="bold", y=1.01)

    for ch_idx, (channel, ch_label) in enumerate(zip(CHANNELS, CH_LABELS)):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_ds, subplot_spec=outer[ch_idx], wspace=0.35
        )
        fig.text(0.01,
                 outer[ch_idx].get_position(fig).y1 + 0.005,
                 f"▌ {ch_label} Photons",
                 fontsize=12, fontweight="bold", va="bottom", color="#333")

        for ds_idx, (label, color) in enumerate(zip(all_labels, PALETTE)):
            ax   = fig.add_subplot(inner[ds_idx])
            data = all_data[channel][ds_idx]
            fits = all_fits[channel][ds_idx]

            # Histogram (density)
            max_val = int(data.max())
            bins    = np.arange(0, max_val + 2) - 0.5
            ax.hist(data, bins=bins, density=True, color=color,
                    alpha=0.35, edgecolor="white", linewidth=0.3, label="Data")

            # Overlay top-3 fitted distributions
            x_cont = np.linspace(max(0.1, data.min()), data.max(), 400)
            x_disc = np.arange(0, max_val + 1)
            line_styles = ["-", "--", ":"]
            line_colors = ["#1a1a1a", "#555555", "#999999"]
            for rank, (fit, ls, lc) in enumerate(zip(fits[:3], line_styles, line_colors)):
                y = dist_pmf(fit, x_disc if fit["name"] in ("Poisson", "Neg. Binomial") else x_cont)
                x = x_disc if fit["name"] in ("Poisson", "Neg. Binomial") else x_cont
                marker = "o" if fit["name"] in ("Poisson", "Neg. Binomial") else None
                ms     = 2 if marker else None
                lw     = 2 if rank == 0 else 1.2
                ax.plot(x, y, ls=ls, color=lc, lw=lw,
                        marker=marker, markersize=ms,
                        label=f"#{rank+1} {fit['name']} (AIC {fit['aic']:.0f})")

            ax.set_title(f"{label}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Photon count", fontsize=8)
            ax.set_ylabel("Density" if ds_idx == 0 else "", fontsize=8)
            ax.legend(fontsize=6.5, framealpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.25, linestyle="--")

    return fig

# ── JS divergence heatmap ─────────────────────────────────────────────────────

def plot_js_heatmaps(js_mats, all_labels):
    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(5 * len(CHANNELS), 4.5))
    fig.suptitle("Jensen-Shannon Divergence Between Datasets\n(0 = identical shape, 1 = completely different)",
                 fontsize=13, fontweight="bold")

    for ax, mat, ch_label in zip(axes, js_mats, CH_LABELS):
        im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.5, aspect="auto")
        ax.set_xticks(range(len(all_labels)))
        ax.set_yticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_title(f"{ch_label} Photons", fontsize=11, fontweight="bold")
        for i in range(len(all_labels)):
            for j in range(len(all_labels)):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if mat[i,j] > 0.25 else "black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────

def main(file_paths):
    if len(file_paths) != 4:
        print("Please supply exactly 4 CSV file paths.")
        sys.exit(1)

    all_labels = [short_name(p) for p in file_paths]
    all_data   = {ch: [] for ch in CHANNELS}
    all_fits   = {ch: [] for ch in CHANNELS}
    all_shapes = {ch: [] for ch in CHANNELS}
    js_mats    = []

    print("\nLoading & fitting distributions...")
    for path in file_paths:
        df = load_csv(path)
        for ch in CHANNELS:
            cleaned = clean_series(df[ch])
            all_data[ch].append(cleaned)
            all_fits[ch].append(fit_all(cleaned.values))
            all_shapes[ch].append(shape_stats(cleaned.values))

    # Print summary tables + JS matrices
    print("\n" + "━" * 72)
    print("  DISTRIBUTION FITTING RESULTS")
    print("━" * 72)
    for ch, ch_label in zip(CHANNELS, CH_LABELS):
        js_mat = js_matrix(all_data[ch])
        js_mats.append(js_mat)
        print_summary(all_fits[ch], all_shapes[ch], all_labels, ch_label, js_mat)

    # Plots
    print("\n\nGenerating plots...")
    fig1 = plot_all(all_data, all_fits, all_labels, file_paths)
    fig1.savefig("photon_fits.png", dpi=150, bbox_inches="tight")
    print("Saved → photon_fits.png")

    fig2 = plot_js_heatmaps(js_mats, all_labels)
    fig2.savefig("photon_js_divergence.png", dpi=150, bbox_inches="tight")
    print("Saved → photon_js_divergence.png")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
