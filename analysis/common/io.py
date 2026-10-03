"""Output-saving helpers for the per-level output folders.
by Andre R. Barbosa, April - October 2026
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def to_json_safe(obj):
    """Recursively convert numpy / pandas types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_posterior_summary(
    out_path: Path,
    param_names: list[str],
    samples: np.ndarray,                 # (n_draws, n_params)
    chain_diagnostics: dict,
    config: dict,
    notes: str = "",
) -> None:
    """Save posterior medians, 95 % CIs, ESS, R̂, and thinned draws."""
    quantiles = np.quantile(samples, [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975], axis=0)
    summary = {}
    for i, name in enumerate(param_names):
        summary[name] = {
            "median":   float(quantiles[3, i]),
            "mean":     float(np.mean(samples[:, i])),
            "std":      float(np.std(samples[:, i])),
            "q025":     float(quantiles[0, i]),
            "q10":      float(quantiles[1, i]),
            "q25":      float(quantiles[2, i]),
            "q75":      float(quantiles[4, i]),
            "q90":      float(quantiles[5, i]),
            "q975":     float(quantiles[6, i]),
        }

    # Thin draws to reasonable size for downstream warm-start
    n_keep = min(2000, samples.shape[0])
    thin_idx = np.linspace(0, samples.shape[0] - 1, n_keep).astype(int)
    thinned = samples[thin_idx]

    payload = {
        "param_names": param_names,
        "summary": summary,
        "chain_diagnostics": to_json_safe(chain_diagnostics),
        "config": to_json_safe(config),
        "thinned_draws": to_json_safe(thinned),
        "n_thinned": int(n_keep),
        "n_total_post_warmup": int(samples.shape[0]),
        "notes": notes,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_posterior_summary(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def write_decision_file(
    out_path: Path,
    step_id: str,
    headline_result: str,
    sanity_status: str,         # "PASS" | "PASS-WITH-NOTE" | "FAIL"
    notes: str = "",
) -> None:
    """Write the short run-summary file."""
    lines = [
        f"# {step_id} — Run summary",
        "",
        f"**Headline result:** {headline_result}",
        f"**Sanity-check status:** {sanity_status}",
        "",
        "## Notes",
        notes,
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def append_status_log(status_md_path: Path, entry: str) -> None:
    """Append one line to aux/pipeline_status.md."""
    if not status_md_path.exists():
        with open(status_md_path, "w") as f:
            f.write("# Pipeline — Status Log\n\nOne line per step transition.\n\n")
    with open(status_md_path, "a") as f:
        f.write(entry.rstrip() + "\n")
