"""
plot_photon_histograms.py
=========================
Density histograms of photons per burst for the total, donor and acceptor
channels, with a KDE curve and mean/median markers. Outliers are removed per
channel with the IQR x 1.5 rule before plotting.

Usage:
    python plot_photon_histograms.py photon_bursts_nanosecond.csv

The input is the burst CSV written by the FRET simulations
(columns: entry_time, total_photons, donor_photons, acceptor_photons;
SUMMARY rows are skipped).
"""

import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

csv_path = sys.argv[1] if len(sys.argv) > 1 else "photon_bursts_nanosecond.csv"

# --- Load data, skipping SUMMARY rows ---
df = pd.read_csv(csv_path, comment=None)
df = df[df["entry_time"] != "SUMMARY"].copy()
df["total_photons"]    = pd.to_numeric(df["total_photons"])
df["donor_photons"]    = pd.to_numeric(df["donor_photons"])
df["acceptor_photons"] = pd.to_numeric(df["acceptor_photons"])

# --- Remove outliers per channel using IQR ×1.5 fence ---
def iqr_filter(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series[(series >= q1 - 1.5 * iqr) & (series <= q3 + 1.5 * iqr)]

filtered = {col: iqr_filter(df[col]) for col, *_ in [
    ("total_photons",),
    ("donor_photons",),
    ("acceptor_photons",),
]}

# --- Plot config ---
channels = [
    ("total_photons",    "Total",    "#4C72B0"),
    ("donor_photons",    "Donor",    "#55A868"),
    ("acceptor_photons", "Acceptor", "#C44E52"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle("Photon Count Density Histograms (outliers removed via IQR ×1.5)", fontsize=14, fontweight="bold", y=1.02)

for ax, (col, label, color) in zip(axes, channels):
    data = filtered[col].values

    # Bin width = 1 for integer photon counts; use KDE-friendly density=True
    max_val = int(data.max())
    bins = np.arange(0, max_val + 2) - 0.5   # centres on integers

    counts, edges, patches = ax.hist(
        data,
        bins=bins,
        density=True,
        color=color,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.4,
    )

    # Overlay a smoothed KDE curve
    kde = gaussian_kde(data, bw_method="scott")
    x = np.linspace(data.min(), data.max(), 500)
    ax.plot(x, kde(x), color=color, linewidth=2.0, label="KDE")

    # Vertical lines for mean and median
    ax.axvline(data.mean(),   color="black",  linestyle="--", linewidth=1.2, label=f"Mean  {data.mean():.1f}")
    ax.axvline(np.median(data), color="grey", linestyle=":",  linewidth=1.2, label=f"Median {np.median(data):.0f}")

    ax.set_title(f"{label} Photons", fontsize=13, fontweight="semibold")
    ax.set_xlabel("Photon Count per Burst", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("photon_histograms.png", dpi=150, bbox_inches="tight")
print("Saved → photon_histograms.png")
plt.show()
