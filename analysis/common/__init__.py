"""Shared infrastructure for the analysis pipeline.
by Andre R. Barbosa, April - October 2026

Modules
-------
data_loader   : canonical data loaders with the record filters
                (pole-count >= 5, damping outlier removal, etc.).
priors        : JCSS-derived prior definitions; single source of truth.
forward_model : MDOF eigenvalue solvers (adimensional + dimensional).
plotting      : consistent plot style; reusable diagnostic figures.
io            : save_step_outputs() and load_step_outputs().
"""
