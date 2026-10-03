# Limnologen Hus 6 — physics-informed environmental normalization of modal frequencies

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23122001.svg)](https://doi.org/10.5281/zenodo.23122001)

Code, results and processed data accompanying

> C. Amaddeo, A. R. Barbosa, R. W. Karlsson, M. Dorn, *Physics-informed machine learning with environmental effects lag for structural health monitoring of mass timber buildings*, submitted to Mechanical Systems and Signal Processing (2026). (Reference to be updated on acceptance.)

The repository reproduces the three-level methodology of the paper on 18 months of continuous ambient-vibration monitoring of Limnologen Hus 6 (Växjö, Sweden), an eight-story CLT building:

| Level | What it does | Script | Results read by the paper |
|---|---|---|---|
| 0 | Equilibrium moisture content (Hailwood–Horrobin) and exponential-moving-average lagged covariates from the SMHI record | `analysis/level0_build_dataset.py` | `Limnologen_continuous_OMA_aligned.csv` |
| 1 | Adimensional twin: per-mode sensitivities (α, β) and diffusion lag times (τ_EMC, τ_T), emcee | `analysis/level1_adimensional_twin.py` | `aux/level1_adim/v3_per_mode_tau/posterior.json` → Table 1, Fig. 4 |
| 1b | Stiffness–mass decomposition α_eff ≈ α_k − α_m | `analysis/level1b_split_model.py` | `aux/level1b_split/decomposition_results.json` → Table 2, Fig. 6 |
| 2 | 7-DOF shear stack, directional story stiffnesses k_X, k_Y (NumPyro NUTS) | `analysis/level2_mdof_nuts.py` | `aux/level2_mdof_nuts/posterior.json` → Eq. (6), Fig. 7 |
| 3 | Bounded Matérn-5/2 GP residual layer (tinygp), CRPS/coverage on the held-out validation set | `analysis/level3_gp_residual.py` | `aux/level3_gp/level3_Mode{1,2,3}_result.json` → Table 3, Fig. 8, Fig. S1 |
| 3b | Same GP on the Level-2 (dimensional) physics mean — physics-equivalence check | `analysis/level3b_gp_dimensional.py` | `aux/level3b_gp_dimensional/` → Table 3, bottom row |
| SHM | Rolling 30-day stiffness estimator and synthetic-damage detection | `analysis/fig10_damage_v2.py` | `aux/checks/fig10_v2_summary.json` → Section 4.5, Fig. 9 |

Supporting checks (Supplementary Material): `analysis/checks/synthetic_lag_identifiability.py` (S8), `analysis/checks/alpha_prior_sensitivity.py` (S2), `analysis/mode_shape_sensitivity.py` (S9), `analysis/checks/indoor_emc_check.py` (indoor-EMC proxy, Section 5).

## Layout

```
Limnologen_continuous_OMA_aligned.csv   processed dataset (see data/README.md)
analysis/        all scripts (levels, checks, figure generators); analysis/common/ shared loaders, priors, forward model
aux/             results of the runs reported in the paper (posteriors and validation metrics)
figures/         the manuscript figures (PNG)
data/            auxiliary data and the data statement
reproduce.sh     end-to-end reproduction; REPRODUCE.md explains each stage and the expected values
```

## Quick start

```bash
bash setup_env.sh            # creates .venv and installs requirements.txt (Python >= 3.10)
source .venv/bin/activate
bash reproduce.sh            # Levels 0-3b + figures, 20-60 min (Level-2 NUTS dominates)
```

Random seeds are fixed (`rng_seed=20260609` for Level 1, `20260608` for Level 2). With the same library versions the posterior summaries reproduce to the printed precision; with other versions of NumPy, emcee or JAX they agree within Monte Carlo error (the weakly identified lag times can shift by a few percent). GP hyperparameters are found by multi-start marginal-likelihood optimization (12 restarts).

## Data

`Limnologen_continuous_OMA_aligned.csv` contains the identified modal frequencies, damping ratios and pole counts for one 15-minute ambient-vibration record every 6 hours (Nov 2024 – May 2026), together with the SMHI Växjö weather variables and the derived EMC and lagged covariates. The raw acceleration records are not distributed; see `data/README.md`.

## Software

Python, with `emcee` (Level 1), `NumPyro`/`JAX` (Level 2) and `tinygp` (Level 3); see `requirements.txt`.

## Licence and citation

Code: MIT License (see `LICENSE`). Processed data: see `data/README.md`. Release v1.0.1 is archived on Zenodo (https://doi.org/10.5281/zenodo.23122001). Please cite the paper and the Zenodo record (see `CITATION.cff`) when using the code or data.
