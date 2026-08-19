# Proof-of-concept code

- `engine.py` — 3PL IRT, Bayesian EAP, adaptive testing (CAT), Bradley–Terry
- `exp_static.py` — Experiments 1–2: efficiency/saturation, contamination
- `exp_arena.py` — Experiment 3: manipulation red-team on 57,477 real LMArena votes
- `robustness.py` — five-seed robustness checks
- `results_static.json`, `results_arena.json` — raw outputs behind the paper's claims

Requires: numpy, scipy, pandas (and duckdb + network for re-downloading arena votes).
Master seed 20260818. Reference thresholds are illustrative; production values are
withheld per the paper's disclosure policy.
