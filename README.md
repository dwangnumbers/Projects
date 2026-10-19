# Projects

Code from my research on single-molecule FRET (smFRET), from simulating the experiment to analyzing its output.

| Project | What it does |
|---|---|
| [fret-simulation](fret-simulation) | Numba-accelerated Monte Carlo simulations of donor/acceptor dyes on a diffusing molecular motor passing through a confocal volume. Several dye-dynamics models (isotropic, elastic pendulum with κ² dynamics) and burst-selection methods based on Fries et al. (1998). |
| [photon-burst-analysis](photon-burst-analysis) | Histograms, two-sample tests (KS, Kruskal–Wallis) and distribution fitting (Poisson, negative binomial, log-normal, gamma, with AIC and Jensen–Shannon divergence) for the simulated photon bursts. |
| [rl-photon-model](rl-photon-model) | Tabular Q-learning model that predicts photon emission rates over time from the simulated photon stream and compares it against Poisson and persistence baselines. |

The simulations write the CSVs that the other two projects read. Each folder has its own README with details and run instructions.
