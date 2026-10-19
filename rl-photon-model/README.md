# Reinforcement Learning for Photon Emission Rates

A tabular Q-learning model that asks whether the rate of photon emission in a simulated smFRET experiment can be predicted from its recent history. The data comes from the elastic pendulum + κ² [FRET simulation](../fret-simulation).

## Hypothesis

An agent that observes recent emission history can learn to predict the emission rate in the next time window. It should outperform a constant-rate (Poisson) predictor and a persistence predictor, which would show that emission rates have learnable temporal structure.

## Method

- **Time windows:** The photon timeline is split into 5 ns windows. Each window's rate is labeled as one of six categories: silent, very low, low, medium, high or very high. The non-zero categories are quintiles of the non-zero rates.
- **State (270 states):** current rate category (6) × rate trend over the last 3 windows (3) × time epoch (5) × fraction of recent silent windows (3).
- **Actions:** predict the next window's rate category (6 actions).
- **Reward:** +1 for an exact match, +0.4 if off by one category, −0.3 if off by two, −1 otherwise.
- **Training:** Q-learning with α = 0.15 and γ = 0.9, and ε-greedy exploration decaying from 1.0 to 0.05. Each episode samples a contiguous segment of the timeline.
- **Baselines:** a constant predictor (always the most common category) and a persistence predictor (next window = current window).

## Results from the original run

| Model | Exact accuracy | Within ±1 category | Reward per step |
|---|---|---|---|
| RL agent | 42.6% | 74.1% | +0.393 |
| Persistence | 28.9% | 55.7% | +0.088 |
| Constant | 29.8% | 44.1% | −0.106 |

The agent's advantage over persistence was significant (z = 7.01, p ≈ 2.4 × 10⁻¹²). The emission rate decays with a timescale of about 1,970 ns. The learned policy reflects this: it predicts a range of high-rate categories early in the timeline and "silent" for every state in the final epoch.

## Files

- `rl_emission_rate.py`: the full pipeline (loading, binning, Q-learning, baselines, statistics, conclusions)
- `rl_emission_dashboard.jsx`: a React dashboard of the original run's results, with the results embedded in the file

## Running

```
pip install numpy pandas
python rl_emission_rate.py photon_events_nanosecond_epk.csv
```

The input is the photon event CSV from `fret_elastic_pendulum_kappa_parallel.py`. Results are written to `rl_results_v2.json` in the working directory.
