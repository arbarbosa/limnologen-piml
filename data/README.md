# Data

## Shipped with the repository

| File | Content | Source |
|---|---|---|
| `../Limnologen_continuous_OMA_aligned.csv` (repository root) | One row per 6-hour ambient-vibration record (Nov 20, 2024 – May 28, 2026; 2,371 rows): identified natural frequencies `f1,f2,f3` (Hz), damping ratios and SSI-Cov pole counts for Modes 1–3, outdoor temperature `T_out` (°C), relative humidity `RH_out` (%), wind speed, the equilibrium moisture content `EMC` (%, Hailwood–Horrobin) and its exponential-moving-average lagged versions `EMC_tau*h`, `T_tau*h`. | Continuous OMA of Limnologen Hus 6 (Linnaeus University) + SMHI Växjö weather station |
| `smhi_T_64510_corrected-archive.csv` | Hourly outdoor temperature, SMHI station 64510 (Växjö), corrected archive | SMHI open data (CC BY 4.0) |

The aligned CSV is the entry point for every analysis stage (Levels 1–3, the rolling-window estimator and all data-dependent figures). The `--relag` option of `analysis/level0_build_dataset.py` recomputes EMC and the lagged covariates from `T_out`/`RH_out` in place.

## Not distributed

The raw acceleration time histories (continuous `.wdd` files and the 12-channel ambient-vibration test campaigns, about 1.2 GB) are not included. They belong to the building monitoring programme of Linnaeus University and the building owner; the published manuscript states that permission to share them has not been obtained. `analysis/wdd_loader.py` and `analysis/wdd_oma.py` document how the raw records were processed into the aligned CSV.

The drawings used as inputs for the building figure (Fig. 2 of the manuscript, `analysis/input/fig01/`) are reproduced from Cantisani (2026) and Martinsons Byggsystem AB and are likewise not redistributed; the final PNG panels are in `figures/`.
