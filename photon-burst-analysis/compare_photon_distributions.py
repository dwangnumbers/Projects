"""
compare_photon_distributions.py
================================
Compares photon count distributions across 4 burst CSV files using:
  - Overlaid KDE plots
  - Empirical CDF plots
  - Q-Q plots (pairwise)
  - KS tests (pairwise, Bonferroni-corrected)
  - Kruskal-Wallis test (all 4 simultaneously)

Usage:
    python compare_photon_distributions.py file1.csv file2.csv file3.csv file4.csv

Each CSV must have columns: total_photons, donor_photons, acceptor_photons
SUMMARY rows are skipped automatically.
"""

import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde

# ── Colour palette (one per dataset) ────────────────────────────────────────
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CHANNELS = ["total_photons", "donor_photons", "acceptor_photons"]
CHANNEL_LABELS = ["Total", "Donor", "Acceptor"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    df = pd.read_csv(path)
    df = df[df["entry_time"] != "SUMMARY"].copy()
    for col in CHANNELS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def iqr_filter(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return series[(series >= lo) & (series <= hi)].dropna()

def short_name(path):
    """Return just the filename without extension."""
    import os
    return os.path.splitext(os.path.basename(path))[0]

# ── Statistical tests ────────────────────────────────────────────────────────

def run_tests(datasets, labels, channel):
    """
    datasets : list of filtered pd.Series (one per file)
    Returns a formatted summary string + raw result dicts.
    """
    n = len(datasets)
    pairs = list(itertools.combinations(range(n), 2))
    n_comparisons = len(pairs)               # 6 for 4 datasets
    bonferroni_alpha = 0.05 / n_comparisons

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  Channel: {channel}")
    lines.append(f"{'='*60}")

    # Kruskal-Wallis (all 4 at once)
    kw_stat, kw_p = stats.kruskal(*[d.values for d in datasets])
    sig = "★ SIGNIFICANT" if kw_p < 0.05 else "not significant"
    lines.append(f"\nKruskal-Wallis (all groups):")
    lines.append(f"  H = {kw_stat:.3f},  p = {kw_p:.4e}  →  {sig}")

    # Pairwise KS tests with Bonferroni correction
    lines.append(f"\nPairwise KS tests  (Bonferroni α = {bonferroni_alpha:.4f}):")
    lines.append(f"  {'Pair':<30} {'KS stat':>8}  {'p-value':>12}  Result")
    lines.append(f"  {'-'*60}")

    ks_results = []
    for i, j in pairs:
        ks_stat, ks_p = stats.ks_2samp(datasets[i].values, datasets[j].values)
        sig_ks = "★ DIFFERENT" if ks_p < bonferroni_alpha else "same"
        pair_label = f"{labels[i]} vs {labels[j]}"
        lines.append(f"  {pair_label:<30} {ks_stat:>8.4f}  {ks_p:>12.4e}  {sig_ks}")
        ks_results.append((i, j, ks_stat, ks_p, ks_p < bonferroni_alpha))

    return "\n".join(lines), ks_results

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_channel(fig, outer_gs_cell, datasets, labels, channel_label, ks_results, pairs):
    """
    For one channel, draw: KDE | CDF | Q-Q grid (upper triangle)
    inside the given GridSpecFromSubplotSpec cell.
    """
    n = len(datasets)
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs_cell, wspace=0.38
    )

    # ── 1. Overlaid KDE ──────────────────────────────────────────────────────
    ax_kde = fig.add_subplot(inner[0])
    for idx, (data, label, color) in enumerate(zip(datasets, labels, PALETTE)):
        kde = gaussian_kde(data, bw_method="scott")
        x = np.linspace(data.min(), data.max(), 500)
        ax_kde.fill_between(x, kde(x), alpha=0.18, color=color)
        ax_kde.plot(x, kde(x), color=color, lw=2, label=label)
    ax_kde.set_title("KDE", fontsize=10, fontweight="bold")
    ax_kde.set_xlabel("Photon count")
    ax_kde.set_ylabel("Density")
    ax_kde.legend(fontsize=7, framealpha=0.6)
    ax_kde.spines[["top", "right"]].set_visible(False)
    ax_kde.grid(axis="y", alpha=0.25, linestyle="--")

    # ── 2. Empirical CDF ─────────────────────────────────────────────────────
    ax_cdf = fig.add_subplot(inner[1])
    for data, label, color in zip(datasets, labels, PALETTE):
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax_cdf.step(sorted_d, cdf, color=color, lw=2, label=label, where="post")
    ax_cdf.set_title("Empirical CDF", fontsize=10, fontweight="bold")
    ax_cdf.set_xlabel("Photon count")
    ax_cdf.set_ylabel("Cumulative probability")
    ax_cdf.legend(fontsize=7, framealpha=0.6)
    ax_cdf.spines[["top", "right"]].set_visible(False)
    ax_cdf.grid(alpha=0.25, linestyle="--")

    # ── 3. Q-Q plot grid (pairwise upper triangle, 2×3 for 4 datasets) ───────
    qq_inner = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=inner[2], hspace=0.55, wspace=0.45
    )
    sig_lookup = {(i, j): is_sig for i, j, _, _, is_sig in ks_results}

    for plot_idx, (i, j) in enumerate(pairs):
        row, col = divmod(plot_idx, 3)
        ax_qq = fig.add_subplot(qq_inner[row, col])

        q_i = np.quantile(datasets[i], np.linspace(0, 1, 200))
        q_j = np.quantile(datasets[j], np.linspace(0, 1, 200))
        is_sig = sig_lookup.get((i, j), False)
        dot_color = "#C44E52" if is_sig else "#55A868"

        ax_qq.scatter(q_i, q_j, s=4, alpha=0.6, color=dot_color, rasterized=True)
        mn = min(q_i.min(), q_j.min())
        mx = max(q_i.max(), q_j.max())
        ax_qq.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.6)
        ax_qq.set_title(f"{labels[i]} vs\n{labels[j]}", fontsize=6.5, pad=2)
        ax_qq.tick_params(labelsize=6)
        ax_qq.spines[["top", "right"]].set_visible(False)

    # Add a tiny legend for Q-Q colour meaning
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52",
               markersize=5, label="KS: different"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#55A868",
               markersize=5, label="KS: same"),
    ]
    # place legend below the qq grid using the last used axes
    ax_qq.legend(handles=legend_els, fontsize=6, loc="lower right",
                 framealpha=0.6, bbox_to_anchor=(1.0, -0.55))


def make_figure(file_paths, all_data, all_labels):
    pairs = list(itertools.combinations(range(len(all_data[CHANNELS[0]])), 2))
    n_channels = len(CHANNELS)

    fig = plt.figure(figsize=(18, 5 * n_channels))
    outer = gridspec.GridSpec(n_channels, 1, figure=fig, hspace=0.55)

    for ch_idx, (channel, ch_label) in enumerate(zip(CHANNELS, CHANNEL_LABELS)):
        datasets = all_data[channel]
        _, ks_results = run_tests(datasets, all_labels, ch_label)

        # Channel row label
        ax_title = fig.add_subplot(outer[ch_idx])
        ax_title.set_visible(False)
        fig.text(
            0.01, ax_title.get_position().y1 - 0.005,
            f"▌ {ch_label} Photons",
            fontsize=13, fontweight="bold", va="top", color="#333333",
        )

        plot_channel(fig, outer[ch_idx], datasets, all_labels,
                     ch_label, ks_results, pairs)

    fig.suptitle(
        "Photon Distribution Comparison across 4 Datasets",
        fontsize=15, fontweight="bold", y=1.01,
    )
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main(file_paths):
    if len(file_paths) != 4:
        print("Please supply exactly 4 CSV file paths.")
        sys.exit(1)

    print("\nLoading & filtering CSVs...")
    all_labels = [short_name(p) for p in file_paths]
    all_data   = {ch: [] for ch in CHANNELS}

    for path in file_paths:
        df = load_csv(path)
        for ch in CHANNELS:
            all_data[ch].append(iqr_filter(df[ch]))

    # Print statistical summary to console
    print("\n" + "━"*60)
    print("  STATISTICAL TEST RESULTS")
    print("━"*60)
    for channel, ch_label in zip(CHANNELS, CHANNEL_LABELS):
        summary, _ = run_tests(all_data[channel], all_labels, ch_label)
        print(summary)

    print("\n\nGenerating plots...")
    fig = make_figure(file_paths, all_data, all_labels)
    out = "photon_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
