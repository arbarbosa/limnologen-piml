#!/bin/bash
# setup_env.sh — one-time environment setup for Limnologen PIML
# by Andre R. Barbosa, April - October 2026
# Run from the repository root:
#   cd <repository root>
#   bash setup_env.sh

set -e

ENV_NAME=".venv"
PYTHON_MIN="3.10"

echo "=== Limnologen PIML — Python environment setup ==="

# ── 1. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment in $ENV_NAME ..."
    python3 -m venv "$ENV_NAME"
else
    echo "Virtual environment already exists at $ENV_NAME — skipping creation."
fi

# ── 2. Activate it ────────────────────────────────────────────────────────────
source "$ENV_NAME/bin/activate"
echo "Activated: $(which python)"

# ── 3. Upgrade pip first ──────────────────────────────────────────────────────
pip install --upgrade pip

# ── 4. Install all dependencies ───────────────────────────────────────────────
echo ""
echo "Installing packages from requirements.txt ..."
pip install -r requirements.txt

# ── 5. Register as a Jupyter kernel (so VS Code notebooks can find it) ────────
python -m ipykernel install --user --name limnologen-piml --display-name "Python (Limnologen PIML)"

echo ""
echo "=== Done! ==="
echo ""
echo "To activate the environment in any new terminal:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the diffusion-lag figure:"
echo "  python code/plot_diffusion_lag_correlation.py"
echo ""
echo "To open VS Code in this folder:"
echo "  code ."
