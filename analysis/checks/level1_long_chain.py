"""Run Level 1 (per-mode tau, same priors/seed) with a longer chain to check convergence.
by Andre R. Barbosa, April - October 2026
Writes to aux/checks/level1_long_chain/ only; the reported run in aux/level1_adim/ is untouched."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from analysis.level1_adimensional_twin import run
run(ROOT / "aux" / "checks" / "level1_long_chain", n_walkers=64, n_warmup=4000, n_production=16000,
    rng_seed=20260609, init_prior_scale=1.0, per_mode_tau=True)
