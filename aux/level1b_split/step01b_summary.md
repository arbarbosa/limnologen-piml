# Level 1b — Stiffness/Mass EMC Decomposition Results

Run: 2026-10-03 13:45
N Monte Carlo samples: 20,000
Reference EMC: 19.0 %

## Wood-Science Prior on α_m

- **Prior**: TruncatedNormal(μ=0.0050, σ=0.0020, lower=0) per %EMC
- **Derivation**: Norway spruce CLT, ρ_dry ≈ 440 kg/m³, r_CLT ≈ 0.55
  → mass gain per %EMC = 0.55 × (440/524) × 0.01 ≈ 0.0046 per %EMC
- **Posterior** (prior-dominated): 5.04 ± 1.95 ×10⁻³ per %EMC

## Per-Mode Decomposition

| Mode | α_eff (Level 1) | α_m (wood sci.) | α_k = α_eff + α_m* | α_k/α_m | P(α_k>0) |
|------|----------------|----------------|---------------------|---------|----------|
| Mode 1 | 4.027±0.285 | 5.044±1.948 | 9.457±2.122 | 2.24 | 1.0000 |
| Mode 2 | 1.589±0.175 | 5.044±1.948 | 6.786±2.017 | 1.49 | 1.0000 |
| Mode 3 | 1.366±0.317 | 5.044±1.948 | 6.541±2.029 | 1.41 | 1.0000 |

*×10⁻³ per %EMC; α_k recovery uses nonlinear inversion at EMC_ref=19%.*

## Note

All posterior samples of α_k are positive. The sign follows from α_eff > 0 and
the non-negative mass effect (α_m ≥ 0); the magnitude of α_k depends on the prior
on α_m, because the frequency data constrain only the difference α_k − α_m.
