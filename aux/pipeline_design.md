# Analysis pipeline — design

This document describes how the analysis code is organized. `README.md` maps each
level to the paper, and `REPRODUCE.md` gives the commands and the expected values.

---

## 1. Folder layout

```
./
├── Limnologen_continuous_OMA_aligned.csv   # processed dataset; entry point of every level
├── analysis/
│   ├── common/                     # shared infrastructure
│   │   ├── data_loader.py          # loaders, record filters, training/validation split
│   │   ├── priors.py               # prior definitions; single source of truth
│   │   ├── forward_model.py        # adimensional model and MDOF eigenvalue solvers
│   │   ├── plotting.py             # consistent figure style
│   │   └── io.py                   # helpers that save posteriors and run summaries
│   ├── level0_build_dataset.py     # EMC and lagged covariates
│   ├── level1_adimensional_twin.py # Level 1: adimensional twin (emcee)
│   ├── level1b_split_model.py      # Level 1b: stiffness–mass decomposition
│   ├── level1b_validation.py       # Level 1b: prior sensitivity and exact-model check
│   ├── level2_mdof_nuts.py         # Level 2: 7-DOF shear stack (NumPyro NUTS)
│   ├── level3_gp_residual.py       # Level 3: bounded GP residual layer
│   ├── level3b_gp_dimensional.py   # Level 3b: same GP on the Level 2 physics mean
│   ├── mode_shape_sensitivity.py   # sensitivity to the stiffness of the concrete story
│   ├── checks/                     # supporting checks reported in the Supplementary Material
│   ├── fig*.py                     # figure scripts
│   └── wdd_loader.py, wdd_oma.py   # processing of the raw acceleration files (not distributed)
├── aux/                            # results; one subfolder per level
├── figures/                        # manuscript figures
└── data/                           # auxiliary data and the data statement
```

---

## 2. Levels and data flow

| Level | Reads | Writes |
|---|---|---|
| 0 | aligned CSV (`T_out`, `RH_out`) | `EMC` and the lagged columns `EMC_tau*h`, `T_tau*h` in the aligned CSV |
| 1 | aligned CSV | `aux/level1_adim/v3_per_mode_tau/posterior.json` |
| 1b | Level 1 posterior | `aux/level1b_split/decomposition_results.json` |
| 2 | aligned CSV; Level 1 lag-time medians | `aux/level2_mdof_nuts/posterior.json` |
| 3 | aligned CSV; Level 1 posterior medians | `aux/level3_gp/level3_Mode{1,2,3}_result.json` |
| 3b | aligned CSV; Level 2 posterior medians | `aux/level3b_gp_dimensional/level3b_Mode{1,2}_result.json` |

Each level is a script that can be run on its own once the levels it reads from
have been run. The figure scripts read only the files in `aux/` and the aligned CSV.

---

## 3. Output convention

| File | Contents |
|---|---|
| `posterior.json` | Posterior medians, credible intervals and convergence diagnostics per parameter; thinned draws for the downstream levels |
| `level3*_result.json` | GP hyperparameters and validation metrics for each value of the variance cap |
| `diagnostic.png` | One diagnostic figure per level (posterior marginals, predictive overlay) |

The scripts also write run logs and short run summaries next to these files;
those are not tracked in the repository.

---

## 4. Record filters and data split

- A record is used for a mode only if that mode has at least 5 stable SSI poles
  and a damping ratio of at most 6 %.
- Training window: start of the record to October 31, 2025. Validation window:
  November 1, 2025 to May 27, 2026.
- Lagged covariates are pre-computed on a grid of 14 lag times (1 h to 2160 h)
  and interpolated in log(tau).
- In Level 3, the two wind bursts in the training window are down-sampled to the
  nominal 6-hour cadence, and the April 2026 windstorm in the validation window
  is reported separately.
