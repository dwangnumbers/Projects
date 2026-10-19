# smFRET Photon Burst Simulation

Monte Carlo simulations of single-molecule FRET (smFRET) experiments. A donor and an acceptor dye are tethered to a cargo carried by a diffusing molecular motor. The simulation follows the dyes through a confocal detection volume and records every detected donor or acceptor photon, then groups the photons into bursts.

Each script is a self-contained version of the same simulation with a different model of the dye dynamics or a different burst analysis.

## Shared features

- Overdamped Langevin (Ornstein–Uhlenbeck) dynamics for the motor, cargo anchor and dye linkers
- Gaussian confocal detection volume (ellipsoidal) with an intensity threshold
- Photon emission resolved per time step as a race between FRET transfer and donor fluorescence, with non-radiative decay and optical losses as stochastic filters
- Adaptive time stepping: fine steps inside the confocal volume (100 ns cargo, 10 µs motor), coarse steps outside (10 µs cargo, 1 ms motor)
- Reflecting-wall confinement box around the confocal volume, so the motor keeps revisiting it
- Runs until a target number of bursts (`TARGET_BURSTS`) is collected
- Float32 state, event-only storage and Numba JIT compilation for speed

## Versions

| Script | Dye model | Notes |
|---|---|---|
| `fret_isotropic.py` | Radial OU spring plus a Cartesian orientation vector per dye | κ² is resampled from random dipoles every step (isotropic average). Photon emission uses a continuous-time Markov chain (CTMC). |
| `fret_elastic_pendulum_kappa.py` | Elastic pendulum in spherical coordinates (r, θ, φ) with dipoles that rotate by Stokes rotational diffusion | κ² follows the dipole orientations, so it stays correlated within a burst. Speed-optimized: cached sin/cos, Cartesian dipole vectors, precomputed constants. |
| `fret_elastic_pendulum_kappa_parallel.py` | Same physics as above | Adds batched random numbers, parallel independent trajectories via Numba `prange`, an integer motor countdown, early rejection of undetected photons and inlined helpers. Writes `*_epk.csv` output. |
| `fret_smoothed_burst_selection.py` | Cartesian Langevin springs (original model) | Replaces the fixed-gap burst finder with the smoothed inter-photon-time method of Fries et al. (1998), Eqs. 24–26, including the false-positive rate α. |
| `fret_fries_paper_update.py` | Cartesian Langevin springs (original model) | Updated to the experimental setup in Fries et al. (1998): ω₀ = 620 nm, z₀ = 12.4 µm confocal volume, 3D Gaussian profile, NA = 1.3, BIFL burst detection, burst size distribution analysis and the detection factor g. |

## Requirements

Python 3.9+ with:

```
pip install numpy pandas numba scipy
```

`scipy` is only needed by the two Fries et al. versions.

## Running

```
python fret_isotropic.py
```

All parameters are constants at the top of each script. Lower `TARGET_BURSTS` for a quick test. The first run spends a minute or two on JIT compilation.

Each run writes to the working directory:

- `photon_events_nanosecond.csv`: one row per detected photon (time, donor/acceptor, both dye positions, interdye distance, FRET efficiency, intensities)
- `photon_bursts_nanosecond.csv`: `SUMMARY` rows with run parameters, then one row per burst (entry/exit time, duration, total/donor/acceptor photons)

`fret_fries_paper_update.py` also writes `burst_size_distribution.csv` and `burst_fret_analysis.csv`. The parallel version adds an `_epk` suffix to its file names.

## Known limitation

In `fret_smoothed_burst_selection.py`, the background lag is estimated from the median of all inter-photon lags. The simulation records photons only inside the confocal volume, so there are no background photons. The median is then dominated by short within-burst lags, and the method can find zero bursts. One fix is to estimate the background lag from the upper percentiles of the lag distribution, which reflect the gaps between bursts.

## Reference

Fries et al., *J. Phys. Chem. A* **102**, 6601–6613 (1998).
