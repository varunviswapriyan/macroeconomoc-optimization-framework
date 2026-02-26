# Policy Shock Emulator (Single Script) — FRED-MD + DOE + Surrogate Optimization

This repository contains a **single Python script** that trains a macroeconomic emulator on FRED-MD style data (with transformation codes), simulates future paths under **policy shock vectors**, runs a **Sobol design of experiments (DOE)**, fits **surrogate models** for fast shock→outcome mapping, and solves a **composite desirability optimization** with realism penalties and validation/robustness tests.


---

## What this script does

### Inputs (policy levers)
- `FEDFUNDS` (Fed Funds rate)
- `USGOVT` (Govt spending/series as provided by FRED-MD file)
- `BOGMBASE` (Monetary base)
- `NONBORRES` (Nonborrowed reserves)

### Outputs (targets)
- `BAAFFM_T` (transformed BAA-FF spread series from FRED-MD)
- `CPI_YOY` (constructed YoY inflation)
- `UNRATE_L` (unemployment rate level)

### Main stages
1. **Read FRED-MD CSV + tcodes** (expects a `Transform:` row on line 2)
2. **Apply transformations** and build supervised lagged features (lags: 1,3,6,12,18,24)
3. **Train emulator** (XGBoost if installed; otherwise RandomForest fallback)
4. **Baseline simulation** (zero shocks) and benchmark baselines (Persistence / AR / VAR via sklearn)
5. **Sobol DOE** over `shock_bounds` and simulate each point
6. **Fit surrogate models** mapping shocks → mean outcome deltas
7. **Composite desirability optimization** (L-BFGS-B multistart + penalties)
8. **Validations & robustness** (surrogate validation, horizon sensitivity, anchor robustness, regime splits, placebo checks, decay shocks, forecast degradation plots, bootstrap uncertainty, multi-objective baselines)

---

## Repo contents

- `main.py` (or whatever you name the script): **the entire pipeline**
- `data/` (not committed): place your FRED-MD CSV here
- `reports_dynamic_fixed_v4/` (generated): all outputs (CSV/JSON/PNG)

Recommended structure:
