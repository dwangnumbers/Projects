# Photon Burst Analysis

Scripts for analyzing the burst CSVs produced by the [FRET simulations](../fret-simulation). Each burst row has total, donor and acceptor photon counts. These scripts look at how those counts are distributed and whether different simulation runs produce the same distribution.

## Scripts

### `plot_photon_histograms.py`

Density histograms of photons per burst for the total, donor and acceptor channels. Each panel has integer-width bins, a KDE curve, and mean and median lines. Outliers are removed per channel with the IQR × 1.5 rule.

```
python plot_photon_histograms.py photon_bursts_nanosecond.csv
```

Output: `photon_histograms.png`

### `compare_photon_distributions.py`

Tests whether four runs have the same distribution, channel by channel:

- Kruskal–Wallis test across all four
- Pairwise two-sample KS tests with a Bonferroni-corrected α (0.05 / 6)
- Plots: overlaid KDEs, empirical CDFs, and a grid of pairwise Q-Q plots colored by the KS result

```
python compare_photon_distributions.py run1.csv run2.csv run3.csv run4.csv
```

Output: test results printed to the console, plus `photon_comparison.png`. Outliers are removed with the IQR × 1.5 rule.

### `fit_photon_distributions.py`

Tests whether four runs have the same *shape* of distribution, even if the parameters differ:

- Fits Poisson, negative binomial, log-normal and gamma distributions by maximum likelihood and ranks them by AIC
- Reports skewness and excess kurtosis
- Computes a Jensen–Shannon divergence matrix between the runs
- Uses the full distributions: only NaNs are dropped, with no outlier removal

```
python fit_photon_distributions.py run1.csv run2.csv run3.csv run4.csv
```

Output: summary tables printed to the console, plus `photon_fits.png` (fitted curves over each histogram) and `photon_js_divergence.png` (heatmaps).

## Input format

The burst CSV written by the simulations. The scripts need the columns `entry_time`, `total_photons`, `donor_photons` and `acceptor_photons`. Rows whose `entry_time` is `SUMMARY` are skipped.

## Requirements

```
pip install numpy pandas scipy matplotlib
```
