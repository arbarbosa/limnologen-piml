"""Run Level 3 exactly as analysis/level3_gp_residual.py, but on the per-mode-tau Level 1
by Andre R. Barbosa, April - October 2026
posterior reported in Table 1 (aux/level1_adim/v3_per_mode_tau).  Outputs go to
aux/checks/level3_v3/ and output/checks_level3_v3/ only; nothing reported is overwritten.
Usage: python analysis/checks/level3_with_v3_posterior.py --mode 1"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT))
import level3_gp_residual as L3
L3.STEP01_PATH = ROOT / "aux" / "level1_adim" / "v3_per_mode_tau" / "posterior.json"
L3.AUX_DIR = ROOT / "aux" / "checks" / "level3_v3"; L3.AUX_DIR.mkdir(parents=True, exist_ok=True)
L3.OUTPUT_DIR = ROOT / "output" / "checks_level3_v3"; L3.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import os
if os.environ.get('K1_ONLY'): L3.K_VALUES = [1.0]
L3.main()
