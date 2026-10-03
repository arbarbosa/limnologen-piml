#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — end-to-end reproduction of the Limnologen Hus 6 PIML analysis
# by Andre R. Barbosa, April - October 2026
#
# Runs Levels 0–3b, the rolling-window estimator and the manuscript figures.
# Run from the repository root after `bash setup_env.sh`:
#
#   bash reproduce.sh
#
# Entry point is the shipped aligned CSV (Limnologen_continuous_OMA_aligned.csv).
# Wall time: 20–60 min, dominated by the Level-2 NUTS run.
# =============================================================================
set -euo pipefail
source .venv/bin/activate
ALIGNED="Limnologen_continuous_OMA_aligned.csv"

echo "=== Level 0: recompute EMC (Hailwood–Horrobin) and lagged covariates in place ==="
python analysis/level0_build_dataset.py --relag "$ALIGNED" --out "$ALIGNED"

cd analysis
echo "=== Level 1: adimensional twin, per-mode lag times (emcee, pinned seed) ==="
mkdir -p ../aux/level1_adim/v3_per_mode_tau
python - <<'PY'
from pathlib import Path
import level1_adimensional_twin as s
s.run(Path("../aux/level1_adim/v3_per_mode_tau"), per_mode_tau=True, rng_seed=20260609)
PY

echo "=== Level 1b: stiffness–mass decomposition (Table 2) ==="
python level1b_split_model.py

echo "=== Level 2: 7-DOF shear stack, NumPyro NUTS (Eq. 6, Fig. 7) ==="
python level2_mdof_nuts.py --warmup 1000 --samples 10000 --chains 4

echo "=== Level 3: GP residual layer, per mode (Table 3) ==="
for m in 1 2 3; do python level3_gp_residual.py --mode "$m"; done
echo "=== Level 3b: GP on the dimensional (Level 2) physics mean — physics equivalence ==="
for m in 1 2; do python level3b_gp_dimensional.py --mode "$m"; done

echo "=== Figures ==="
for f in fig02_oma_timeseries fig04_posteriors fig05_split_decomposition \
         fig06_nuts_diagnostic fig07_piml_composite fig08_ksweep \
         fig09_baseline fig10_damage_v2 fig03_architecture_v2; do
  echo "  -> $f"; python "${f}.py"
done
cd ..

echo "=== Export figures under the manuscript file names ==="
O=analysis/output; F=figures
cp "$O/Fig_Architecture_PIML_v2.png"        "$F/fig01.png"
cp "$O/Fig_OMA_TimeSeries.png"              "$F/fig02.png"
cp "$O/Fig_Level1_Posteriors_Violin.png"    "$F/fig04.png"
cp "$O/Fig_Level1b_Split_Decomposition.png" "$F/fig05.png"
cp "$O/Fig_Level2_NUTS_Diagnostic.png"      "$F/fig06.png"
for m in 1 2 3; do
  cp "$O/Fig_Level3_PIML_Mode$m.png"  "$F/fig07$(printf "\\$(printf '%03o' $((96+m)))").png"
  cp "$O/Fig_Level3_kSweep_Mode$m.png" "$F/fig08$(printf "\\$(printf '%03o' $((96+m)))").png"
done
cp "$O/Fig_Damage_Detection_v2.png"          "$F/fig10.png"
# fig09a/b are written directly to figures/ by fig09_baseline.py;
# fig03a–e (building photo, key-map, test set-ups) need inputs that are not redistributed.
echo "DONE — compare aux/*.json with the values in manuscript/main.tex (see REPRODUCE.md)."
