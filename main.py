import os, json, warnings, copy
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance

from scipy.optimize import minimize
from scipy.stats import qmc

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)


DATA_FILE = "2025-08-md.csv"

MODELS_DIR  = "models_dynamic_fixed_v4"
REPORTS_DIR = "reports_dynamic_fixed_v4"
PLOTS_DIR   = os.path.join(REPORTS_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


input_vars = ["FEDFUNDS", "USGOVT", "BOGMBASE", "NONBORRES"]

output_vars = ["BAAFFM_T", "CPI_YOY", "UNRATE_L"]

lags = [1, 3, 6, 12, 18, 24]

shock_bounds = {
    "FEDFUNDS":  (-2.0,  3.0),
    "USGOVT":    (-1200, 1200),
    "BOGMBASE":  (-0.05, 0.05),
    "NONBORRES": (-350,  350),
}

DOE_M = 8
SIM_HORIZON = 12

EXTREMENESS_WEIGHT = 0.25
PENALTY_WEIGHT = 0.30

DOE_M_PAPER = DOE_M  # set to 9 or 10 if desired (512/1024 DOE points)



def safe_log(x: pd.Series) -> pd.Series:
    x = x.copy()
    x[x <= 0] = np.nan
    return np.log(x)

def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    s = series.astype(float)
    if tcode == 1:
        return s
    elif tcode == 2:
        return s.diff(1)
    elif tcode == 3:
        return s.diff(2)
    elif tcode == 4:
        return safe_log(s)
    elif tcode == 5:
        return safe_log(s).diff(1)
    elif tcode == 6:
        return safe_log(s).diff(2)
    elif tcode == 7:
        gr = s.pct_change(1)
        return gr.diff(1)
    else:
        raise ValueError(f"Unsupported tcode: {tcode}")

def read_fredmd_with_tcodes(path: str):
    with open(path, "r", newline="") as f:
        first = f.readline().rstrip("\n")
        second = f.readline().rstrip("\n")

    delim = "\t" if ("\t" in first and "\t" in second) else ","
    header = first.split(delim)
    trow   = second.split(delim)

    if len(trow) != len(header) or not str(trow[0]).strip().lower().startswith("transform"):
        raise ValueError("Missing valid 'Transform:' row on line 2.")

    tcodes = {}
    for col, code in zip(header[1:], trow[1:]):
        code = str(code).strip()
        if code and code.lower() != "nan":
            tcodes[col] = int(float(code))

    df_levels = pd.read_csv(path, sep=delim, skiprows=[1])

    if "sasdate" in df_levels.columns:
        df_levels["sasdate"] = pd.to_datetime(df_levels["sasdate"])
        df_levels = df_levels.sort_values("sasdate").set_index("sasdate")
    else:
        df_levels.iloc[:, 0] = pd.to_datetime(df_levels.iloc[:, 0])
        df_levels = df_levels.sort_values(df_levels.columns[0]).set_index(df_levels.columns[0])

    df_levels = df_levels.apply(pd.to_numeric, errors="coerce").ffill().bfill()
    return df_levels, tcodes



def shock_calibration_report(df_levels, input_vars, path_out):
    """
    Produces shock distribution diagnostics using historical changes:
    - FEDFUNDS : monthly difference
    - USGOVT   : monthly difference
    - NONBORRES: monthly difference
    - BOGMBASE : log monthly change
    This does NOT override shock bounds, but outputs justification diagnostics.
    """
    rows = []
    for v in input_vars:
        s = df_levels[v].astype(float).dropna()
        if v == "BOGMBASE":
            changes = safe_log(s).diff(1).dropna()
        else:
            changes = s.diff(1).dropna()

        qs = np.percentile(changes, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        rows.append({
            "var": v,
            "mean": float(changes.mean()),
            "std": float(changes.std()),
            "q01": float(qs[0]),
            "q05": float(qs[1]),
            "q10": float(qs[2]),
            "q25": float(qs[3]),
            "q50": float(qs[4]),
            "q75": float(qs[5]),
            "q90": float(qs[6]),
            "q95": float(qs[7]),
            "q99": float(qs[8]),
        })

    df_rep = pd.DataFrame(rows)
    df_rep.to_csv(path_out, index=False)
    print("\n Saved historical shock calibration report to:", path_out)
    return df_rep



def build_base_frame(df_levels: pd.DataFrame, tcodes: dict) -> pd.DataFrame:
    needed_raw = input_vars + ["BAAFFM", "CPIAUCSL", "UNRATE", "INDPRO"]
    missing = [c for c in needed_raw if c not in df_levels.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = pd.DataFrame(index=df_levels.index)

    for v in input_vars:
        df[v] = apply_tcode(df_levels[v], tcodes.get(v, 1))

    df["BAAFFM_T"] = apply_tcode(df_levels["BAAFFM"], tcodes.get("BAAFFM", 1))
    df["CPI_YOY"]  = 100.0 * (df_levels["CPIAUCSL"] / df_levels["CPIAUCSL"].shift(12) - 1.0)
    df["UNRATE_L"] = df_levels["UNRATE"].astype(float)
    df["INDPRO_G6"] = 100.0 * (safe_log(df_levels["INDPRO"]) - safe_log(df_levels["INDPRO"]).shift(6))

    return df

def make_supervised(df_base: pd.DataFrame, lags: list):
    df_feat = df_base.copy()
    for col in (input_vars + output_vars):
        for L in lags:
            df_feat[f"{col}_lag{L}"] = df_base[col].shift(L)

    df_feat = df_feat.dropna().copy()
    X_cols = input_vars + [f"{col}_lag{L}" for col in (input_vars + output_vars) for L in lags]
    return df_feat, X_cols



def get_model():
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=1200,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ), "XGBoost"
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=900,
            max_depth=14,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ), "RandomForest"



def build_future_levels_from_shocks(baseline_levels: pd.Series, shocks: dict, future_dates: list):
    paths = pd.DataFrame(index=future_dates, columns=input_vars, dtype=float)
    last = baseline_levels.copy()

    for t in future_dates:
        last["FEDFUNDS"] = baseline_levels["FEDFUNDS"] + shocks["FEDFUNDS"]
        last["USGOVT"]   = last["USGOVT"] + shocks["USGOVT"]
        last["NONBORRES"] = max(0.0, last["NONBORRES"] + shocks["NONBORRES"])
        last["BOGMBASE"]  = max(1e-6, last["BOGMBASE"] * float(np.exp(shocks["BOGMBASE"])))
        paths.loc[t] = last.values

    return paths



def simulate_path_shocks(
    shocks: dict,
    df_levels: pd.DataFrame,
    tcodes: dict,
    base_history_trans: pd.DataFrame,
    anchor_date: pd.Timestamp,
    X_cols: list,
    scaler_X: StandardScaler,
    models: dict,
    scaler_Y: dict,
    horizon: int = 12
):
    hist_trans = base_history_trans.copy()
    hist_levels = df_levels[input_vars].loc[:anchor_date].copy()
    baseline_levels = hist_levels.loc[anchor_date].copy()

    future_dates = []
    last = anchor_date
    for _ in range(horizon):
        last = last + pd.offsets.MonthBegin(1)
        future_dates.append(last)

    future_paths = build_future_levels_from_shocks(baseline_levels, shocks, future_dates)

    for t in future_dates:
        hist_levels.loc[t] = future_paths.loc[t].values

    out_rows = []
    for step, t in enumerate(future_dates, start=1):
        lever_current = {}
        for v in input_vars:
            lever_current[v] = float(apply_tcode(hist_levels[v], tcodes.get(v, 1)).loc[t])

        row = {v: lever_current[v] for v in input_vars}
        for col in (input_vars + output_vars):
            for L in lags:
                row[f"{col}_lag{L}"] = float(hist_trans[col].iloc[-L])

        x = np.array([row[c] for c in X_cols], dtype=float).reshape(1, -1)
        x_s = scaler_X.transform(x)

        y_pred = {}
        for y in output_vars:
            y_s = float(models[y].predict(x_s)[0])
            y_pred[y] = float(scaler_Y[y].inverse_transform([[y_s]])[0, 0])

        new_row = pd.Series({**{v: lever_current[v] for v in input_vars},
                             **{y: y_pred[y] for y in output_vars}}, name=t)
        hist_trans = pd.concat([hist_trans, new_row.to_frame().T], axis=0)

        out_rows.append({
            "step": step,
            **y_pred,
            **{f"{v}_LEVEL": float(hist_levels.loc[t, v]) for v in input_vars}
        })

    return pd.DataFrame(out_rows)


df_levels, tcodes = read_fredmd_with_tcodes(DATA_FILE)

shock_calibration_path = os.path.join(REPORTS_DIR, "historical_shock_calibration.csv")
shock_calibration_report(df_levels, input_vars, shock_calibration_path)

df_base = build_base_frame(df_levels, tcodes)
df_feat, X_cols = make_supervised(df_base, lags)

X = df_feat[X_cols].values
Y = df_feat[output_vars].values

split = int(0.8 * len(df_feat))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

scaler_X = StandardScaler().fit(X_train)
X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)

scaler_Y = {}
Y_train_s = np.zeros_like(Y_train)
Y_test_s  = np.zeros_like(Y_test)

for i, t in enumerate(output_vars):
    sc = StandardScaler().fit(Y_train[:, [i]])
    scaler_Y[t] = sc
    Y_train_s[:, i] = sc.transform(Y_train[:, [i]]).ravel()
    Y_test_s[:, i]  = sc.transform(Y_test[:, [i]]).ravel()



models = {}
metrics = []
model_obj, model_name = get_model()
print(f"\nUsing model family: {model_name} for ALL targets")

for i, t in enumerate(output_vars):
    m = copy.deepcopy(model_obj)
    m.fit(X_train_s, Y_train_s[:, i])
    models[t] = m

    yhat = m.predict(X_test_s)
    r2 = float(r2_score(Y_test_s[:, i], yhat))
    rmse = float(np.sqrt(mean_squared_error(Y_test_s[:, i], yhat)))
    metrics.append({"target": t, "R2": r2, "RMSE": rmse})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(REPORTS_DIR, "test_metrics.csv"), index=False)
print(metrics_df)

r2_map = {row["target"]: max(0.0, float(row["R2"])) for _, row in metrics_df.iterrows()}



anchor_date = df_feat.index[:split][-1]
print(f"\nAnchor date used for simulation: {anchor_date}")

history_start = df_feat.index[df_feat.index.get_loc(anchor_date) - max(lags)]
base_history_trans = df_base.loc[history_start:anchor_date, input_vars + output_vars].copy()

baseline_shocks = {v: 0.0 for v in input_vars}
baseline_path = simulate_path_shocks(
    baseline_shocks, df_levels, tcodes, base_history_trans, anchor_date,
    X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
)
baseline_path.to_csv(os.path.join(REPORTS_DIR, "baseline_path.csv"), index=False)

print("\nBase simulation (first 5 steps):")
print(baseline_path.head())

baseline_means = baseline_path[output_vars].mean().to_dict()



from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor



def make_future_dates(anchor_date, horizon):
    future_dates = []
    last = anchor_date
    for _ in range(horizon):
        last = last + pd.offsets.MonthBegin(1)
        future_dates.append(last)
    return future_dates



def persistence_baseline(df_base, anchor_date, horizon, output_vars):
    """
    Forecast each output as constant equal to last observed value at anchor_date.
    """
    last_vals = df_base.loc[anchor_date, output_vars].astype(float).to_dict()

    out = pd.DataFrame({"step": np.arange(1, horizon + 1)})
    for y in output_vars:
        out[y] = last_vals[y]
    return out


persistence_path = persistence_baseline(df_base, anchor_date, SIM_HORIZON, output_vars)
persistence_means = persistence_path[output_vars].mean().to_dict()

persistence_path.to_csv(os.path.join(REPORTS_DIR, "baseline_persistence_path.csv"), index=False)
with open(os.path.join(REPORTS_DIR, "baseline_persistence_means.json"), "w") as f:
    json.dump(persistence_means, f, indent=2)

print(" Persistence baseline created.")



def make_ar_design(series: pd.Series, p: int):
    """
    Build AR(p) supervised data:
      X_t = [y_{t-1}, ..., y_{t-p}], y_t = y_t
    Returns (X, y) aligned.
    """
    s = series.dropna().astype(float).values
    X, y = [], []
    for t in range(p, len(s)):
        X.append(s[t - p:t][::-1])  # reverse so lag1 is first
        y.append(s[t])
    return np.array(X), np.array(y)


def ar_forecast_sklearn(series: pd.Series, anchor_date, horizon, p=12):
    """
    Fit AR(p) using sklearn LinearRegression and simulate forward.
    """
    train = series.loc[:anchor_date].dropna()
    X, y = make_ar_design(train, p)

    if len(X) < 5:
        # fallback to persistence if too few points
        last_val = float(train.iloc[-1])
        return np.array([last_val] * horizon)

    model = LinearRegression().fit(X, y)

    history = train.astype(float).values.tolist()

    preds = []
    for _ in range(horizon):
        x = np.array(history[-p:][::-1]).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        history.append(yhat)

    return np.array(preds)


def ar_baseline(df_base, anchor_date, horizon, output_vars, p=12):
    out = pd.DataFrame({"step": np.arange(1, horizon + 1)})
    for y in output_vars:
        out[y] = ar_forecast_sklearn(df_base[y], anchor_date, horizon, p=p)
    return out


ar_path = ar_baseline(df_base, anchor_date, SIM_HORIZON, output_vars, p=12)
ar_means = ar_path[output_vars].mean().to_dict()

ar_path.to_csv(os.path.join(REPORTS_DIR, "baseline_ar12_path.csv"), index=False)
with open(os.path.join(REPORTS_DIR, "baseline_ar12_means.json"), "w") as f:
    json.dump(ar_means, f, indent=2)

print(" AR(12) baseline (sklearn) created.")



def make_var_design(df: pd.DataFrame, p: int):
    """
    Build VAR(p) supervised data:
      X_t = [y_{t-1},...,y_{t-p}] flattened across all vars
      Y_t = y_t vector
    """
    data = df.dropna().astype(float).values
    k = data.shape[1]

    X, Y = [], []
    for t in range(p, len(data)):
        lag_block = data[t - p:t][::-1].reshape(p * k)
        X.append(lag_block)
        Y.append(data[t])
    return np.array(X), np.array(Y)


def var_forecast_sklearn(df_train: pd.DataFrame, horizon: int, p=6):
    """
    Fit VAR(p) using sklearn MultiOutputRegressor(LinearRegression) and forecast forward.
    """
    X, Y = make_var_design(df_train, p)

    if len(X) < 5:
        # fallback to persistence if too few points
        last = df_train.iloc[-1].astype(float).values
        return np.tile(last, (horizon, 1))

    model = MultiOutputRegressor(LinearRegression()).fit(X, Y)

    history = df_train.astype(float).values.tolist()
    k = df_train.shape[1]

    preds = []
    for _ in range(horizon):
        lag_block = np.array(history[-p:][::-1]).reshape(1, p * k)
        yhat = model.predict(lag_block)[0]
        preds.append(yhat)
        history.append(yhat.tolist())

    return np.array(preds)


def var_baseline(df_base, anchor_date, horizon, vars_list, p=6):
    train = df_base[vars_list].dropna().loc[:anchor_date].copy()
    fc = var_forecast_sklearn(train, horizon, p=p)

    out = pd.DataFrame(fc, columns=vars_list)
    out.insert(0, "step", np.arange(1, horizon + 1))
    return out


vars_for_var = output_vars
var_path = var_baseline(df_base, anchor_date, SIM_HORIZON, vars_for_var, p=6)
var_means = var_path[output_vars].mean().to_dict()

var_path.to_csv(os.path.join(REPORTS_DIR, "baseline_var6_path.csv"), index=False)
with open(os.path.join(REPORTS_DIR, "baseline_var6_means.json"), "w") as f:
    json.dump(var_means, f, indent=2)

print(" VAR(6) baseline (sklearn) created.")


def compare_forecasts_to_actual(df_base, anchor_date, horizon, forecast_path, output_vars):
    future_dates = make_future_dates(anchor_date, horizon)

    actual = df_base.loc[future_dates, output_vars].copy()
    actual.reset_index(drop=True, inplace=True)

    pred = forecast_path[output_vars].copy().reset_index(drop=True)

    rows = []
    for y in output_vars:
        rmse = float(np.sqrt(mean_squared_error(actual[y], pred[y])))
        mae  = float(mean_absolute_error(actual[y], pred[y]))
        rows.append({"target": y, "RMSE": rmse, "MAE": mae})
    return pd.DataFrame(rows)


# Emulator baseline path (your baseline simulation output)
emulator_forecast_df = baseline_path[["step"] + output_vars].copy()

eval_emulator = compare_forecasts_to_actual(df_base, anchor_date, SIM_HORIZON, emulator_forecast_df, output_vars)
eval_persist  = compare_forecasts_to_actual(df_base, anchor_date, SIM_HORIZON, persistence_path, output_vars)
eval_ar       = compare_forecasts_to_actual(df_base, anchor_date, SIM_HORIZON, ar_path, output_vars)
eval_var      = compare_forecasts_to_actual(df_base, anchor_date, SIM_HORIZON, var_path, output_vars)

eval_emulator["model"] = "Emulator (zero shocks)"
eval_persist["model"]  = "Persistence"
eval_ar["model"]       = "AR(12)-sklearn"
eval_var["model"]      = "VAR(6)-sklearn"

eval_all = pd.concat([eval_emulator, eval_persist, eval_ar, eval_var], ignore_index=True)
eval_all = eval_all[["model", "target", "RMSE", "MAE"]].sort_values(["target", "model"])

eval_all.to_csv(os.path.join(REPORTS_DIR, "baseline_forecast_eval.csv"), index=False)

print(eval_all)



baseline_summary = pd.DataFrame([
    {"model": "Emulator (zero shocks)", **baseline_means},
    {"model": "Persistence", **persistence_means},
    {"model": "AR(12)-sklearn", **ar_means},
    {"model": "VAR(6)-sklearn", **var_means},
])

baseline_summary.to_csv(os.path.join(REPORTS_DIR, "baseline_model_comparison_means.csv"), index=False)

print(baseline_summary)
print(" Saved:", os.path.join(REPORTS_DIR, "baseline_model_comparison_means.csv"))

print("\n Benchmark baselines complete (no statsmodels).")



sampler = qmc.Sobol(d=len(input_vars), scramble=True, seed=42)
X_doe_unit = sampler.random_base2(m=DOE_M_PAPER)
N_DOE = X_doe_unit.shape[0]

X_doe_real = np.zeros_like(X_doe_unit)
for j, v in enumerate(input_vars):
    lo, hi = shock_bounds[v]
    X_doe_real[:, j] = lo + (hi - lo) * X_doe_unit[:, j]

doe_rows = []
for i in range(N_DOE):
    shocks = {v: float(X_doe_real[i, j]) for j, v in enumerate(input_vars)}
    path = simulate_path_shocks(
        shocks, df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    means = path[output_vars].mean().to_dict()
    deltas = {y: means[y] - baseline_means[y] for y in output_vars}

    doe_rows.append({
        "doe_id": i,
        **{f"{v}_shock": shocks[v] for v in input_vars},
        **{f"{y}_delta_mean": deltas[y] for y in output_vars},
        **{f"{y}_mean": means[y] for y in output_vars},
    })

df_doe = pd.DataFrame(doe_rows)

# Add baseline point explicitly
baseline_row = {
    "doe_id": -1,
    **{f"{v}_shock": 0.0 for v in input_vars},
    **{f"{y}_delta_mean": 0.0 for y in output_vars},
    **{f"{y}_mean": baseline_means[y] for y in output_vars},
}
df_doe = pd.concat([df_doe, pd.DataFrame([baseline_row])], ignore_index=True)

doe_path = os.path.join(REPORTS_DIR, "doe_dynamic_delta_results.csv")
df_doe.to_csv(doe_path, index=False)
print(f"\nSaved DOE results to: {doe_path}")

# Pareto export
pareto_cols = [f"{y}_delta_mean" for y in output_vars] + [f"{v}_shock" for v in input_vars]
df_doe[pareto_cols].to_csv(os.path.join(REPORTS_DIR, "pareto_frontier_dataset.csv"), index=False)
print(" Saved Pareto dataset for plotting to:", os.path.join(REPORTS_DIR, "pareto_frontier_dataset.csv"))



feasible = {}
for y in output_vars:
    vals = df_doe[f"{y}_delta_mean"].values
    feasible[y] = {
        "min": float(np.min(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p10": float(np.percentile(vals, 10)),
        "p50": float(np.percentile(vals, 50)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }

print("\nFeasible Δ mean ranges (DOE):")
print(json.dumps(feasible, indent=2))
with open(os.path.join(REPORTS_DIR, "feasible_ranges.json"), "w") as f:
    json.dump(feasible, f, indent=2)



rules = {}
for y in output_vars:
    vals = df_doe[f"{y}_delta_mean"].values
    U = max(abs(feasible[y]["min"]), abs(feasible[y]["max"]))

    if y == "CPI_YOY":
        T = float(np.percentile(vals, 10))
        T = min(0.0, T)
        band = max(0.10, abs(T) * 0.35)
        rules[y] = {"typ": "target", "T": T, "band": band, "U": U}

    elif y in ["UNRATE_L", "BAAFFM_T"]:
        T = float(np.percentile(vals, 25))
        band = max(0.10, abs(T) * 0.35)
        rules[y] = {"typ": "target", "T": T, "band": band, "U": U}

    else:
        rules[y] = {"typ": "smaller", "T": feasible[y]["p10"], "U": feasible[y]["p95"]}

with open(os.path.join(REPORTS_DIR, "rules_used.json"), "w") as f:
    json.dump(rules, f, indent=2)

print("\nRules used:")
print(json.dumps(rules, indent=2))

def desirability(val, rule):
    typ = rule["typ"]

    if typ == "smaller":
        T, U = rule["T"], rule["U"]
        if val >= U: return 0.0
        if val <= T: return 1.0
        return (U - val) / (U - T)

    if typ == "target":
        T, band, U = rule["T"], rule["band"], abs(rule["U"])
        dist = abs(val - T)
        if dist <= band: return 1.0
        if dist >= U: return 0.0
        return (U - dist) / (U - band)

    raise ValueError("Unknown typ")



X_surr = df_doe[[f"{v}_shock" for v in input_vars]].values

surrogate_models = {}
try:
    import xgboost as xgb
    surrogate_name = "XGBoost"
    for y in output_vars:
        m = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_surr, df_doe[f"{y}_delta_mean"].values)
        surrogate_models[y] = m
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    surrogate_name = "RandomForest"
    for y in output_vars:
        m = RandomForestRegressor(
            n_estimators=800,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        m.fit(X_surr, df_doe[f"{y}_delta_mean"].values)
        surrogate_models[y] = m

print(f"\nSurrogate family used: {surrogate_name}")

zero = np.zeros((1, len(input_vars)))
print("\nSurrogate deltas at ZERO shocks (should be ~0):")
print({y: float(surrogate_models[y].predict(zero)[0]) for y in output_vars})



imp_rows = []
for y in output_vars:
    try:
        r = permutation_importance(
            surrogate_models[y],
            X_surr,
            df_doe[f"{y}_delta_mean"].values,
            n_repeats=10,
            random_state=42
        )
        for i, v in enumerate(input_vars):
            imp_rows.append({"target": y, "var": v, "importance_mean": float(r.importances_mean[i])})
    except Exception as e:
        print(f"⚠️ Permutation importance failed for {y}: {e}")

if imp_rows:
    df_imp = pd.DataFrame(imp_rows)
    df_imp.to_csv(os.path.join(REPORTS_DIR, "surrogate_permutation_importance.csv"), index=False)
    print(" Saved surrogate permutation importance to:", os.path.join(REPORTS_DIR, "surrogate_permutation_importance.csv"))



def composite_surrogate(shocks):
    ds, ws = [], []
    deltas = {}
    x = np.array([[shocks[v] for v in input_vars]])

    for y in output_vars:
        dval = float(surrogate_models[y].predict(x)[0])
        deltas[y] = dval
        d = max(1e-9, desirability(dval, rules[y]))
        w = max(r2_map.get(y, 0.1), 0.1)
        ds.append(d); ws.append(w)

    ds = np.array(ds); ws = np.array(ws)
    comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))
    return comp, deltas

def policy_penalty(shocks):
    fed, usg, base, nres = shocks["FEDFUNDS"], shocks["USGOVT"], shocks["BOGMBASE"], shocks["NONBORRES"]
    penalty = 0.0
    if fed > 2.0 and base < -0.02: penalty += 1.0
    if fed > 2.0 and nres < -150: penalty += 1.0
    if usg > 700 and base < -0.02: penalty += 1.0
    return penalty

def extremeness_penalty(shocks):
    pen = 0.0
    for v in input_vars:
        lo, hi = shock_bounds[v]
        mid = 0.5*(lo+hi)
        width = hi-lo
        dist = abs(shocks[v] - mid)
        thresh = 0.40*width
        if dist > thresh:
            pen += ((dist - thresh)/(0.60*width))**2
    return pen

def objective(theta):
    shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
    comp, _ = composite_surrogate(shocks)
    return -comp + PENALTY_WEIGHT*policy_penalty(shocks) + EXTREMENESS_WEIGHT*extremeness_penalty(shocks)

bounds = [shock_bounds[v] for v in input_vars]

def run_opt(x0):
    return minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")

best_res = None
best_val = 1e18

starts = [np.zeros(len(input_vars))]
for _ in range(25):
    starts.append(np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds]))

for x0 in starts:
    res_try = run_opt(x0)
    if res_try.fun < best_val:
        best_val = res_try.fun
        best_res = res_try

res = best_res
best_shocks = {v: float(res.x[i]) for i, v in enumerate(input_vars)}
best_comp, best_deltas = composite_surrogate(best_shocks)

opt_path = simulate_path_shocks(
    best_shocks, df_levels, tcodes, base_history_trans, anchor_date,
    X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
)
opt_means = opt_path[output_vars].mean().to_dict()

print("Optimal shocks:")
print(json.dumps(best_shocks, indent=2))
print("\nPredicted Δ mean outcomes (scenario - baseline):")
print(json.dumps(best_deltas, indent=2))
print("\nAbsolute mean outcomes at optimum:")
print(json.dumps(opt_means, indent=2))

with open(os.path.join(REPORTS_DIR, "optimal_shocks.json"), "w") as f:
    json.dump(best_shocks, f, indent=2)
with open(os.path.join(REPORTS_DIR, "optimal_delta_outcomes.json"), "w") as f:
    json.dump(best_deltas, f, indent=2)
with open(os.path.join(REPORTS_DIR, "optimal_absolute_outcomes.json"), "w") as f:
    json.dump(opt_means, f, indent=2)

opt_path.to_csv(os.path.join(REPORTS_DIR, "optimal_path.csv"), index=False)



def local_sensitivity(best_shocks, eps_frac=0.02):
    base_comp, base_deltas = composite_surrogate(best_shocks)
    rows = []

    for v in input_vars:
        lo, hi = shock_bounds[v]
        width = hi - lo
        eps = eps_frac * width

        shocks_up = best_shocks.copy()
        shocks_dn = best_shocks.copy()
        shocks_up[v] = min(hi, shocks_up[v] + eps)
        shocks_dn[v] = max(lo, shocks_dn[v] - eps)

        comp_up, deltas_up = composite_surrogate(shocks_up)
        comp_dn, deltas_dn = composite_surrogate(shocks_dn)

        row = {
            "var": v,
            "eps_used": eps,
            "comp_base": base_comp,
            "comp_up": comp_up,
            "comp_dn": comp_dn,
            "comp_slope": (comp_up - comp_dn) / (2*eps)
        }

        for y in output_vars:
            row[f"{y}_slope"] = (deltas_up[y] - deltas_dn[y]) / (2*eps)

        rows.append(row)

    return pd.DataFrame(rows)

df_sens = local_sensitivity(best_shocks, eps_frac=0.02)
df_sens.to_csv(os.path.join(REPORTS_DIR, "local_sensitivity_optimum.csv"), index=False)
print(" Saved local sensitivity report to:", os.path.join(REPORTS_DIR, "local_sensitivity_optimum.csv"))



indiv_rows = []

for target in output_vars:
    print(f"\nOptimizing for {target}...")

    def obj_single(theta):
        shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
        delta = float(surrogate_models[target].predict(np.array([[shocks[v] for v in input_vars]]))[0])

        rule = rules[target]
        typ = rule["typ"]

        if typ == "smaller":
            core = delta
        elif typ == "target":
            core = abs(delta - rule["T"])
        else:
            core = delta

        pen = PENALTY_WEIGHT * policy_penalty(shocks) + EXTREMENESS_WEIGHT * extremeness_penalty(shocks)
        return core + pen

    best_res_t, best_val_t = None, 1e18

    for x0 in starts:
        res_t = minimize(obj_single, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res_t.fun < best_val_t:
            best_val_t, best_res_t = res_t.fun, res_t

    best_t = {v: float(best_res_t.x[i]) for i, v in enumerate(input_vars)}
    delta_t = float(surrogate_models[target].predict(np.array([[best_t[v] for v in input_vars]]))[0])

    indiv_rows.append({"target": target, "delta_mean_pred": delta_t, **best_t})

    print("Optimal shocks:")
    print(json.dumps(best_t, indent=2))
    print(f"Predicted delta mean for {target}: {delta_t:.4f}")

df_indiv = pd.DataFrame(indiv_rows)
df_indiv.to_csv(os.path.join(REPORTS_DIR, "individual_opt_results.csv"), index=False)

print("\nSaved individual optimizations to:", os.path.join(REPORTS_DIR, "individual_opt_results.csv"))
print(df_indiv)

print("\nSaved outputs in:", REPORTS_DIR)




VALID_DIR = os.path.join(REPORTS_DIR, "validations")
os.makedirs(VALID_DIR, exist_ok=True)

def evaluate_surrogate_models(df_doe, input_vars, output_vars, test_size=0.25, seed=42):
    X = df_doe[[f"{v}_shock" for v in input_vars]].values

    rows = []
    for y in output_vars:
        y_true = df_doe[f"{y}_delta_mean"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=test_size, random_state=seed)

        try:
            import xgboost as xgb
            m = xgb.XGBRegressor(
                n_estimators=800,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1
            )
        except Exception:
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=-1
            )

        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        rows.append({
            "target": y,
            "R2": float(r2_score(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "n_test": len(y_test)
        })

    return pd.DataFrame(rows)

def compute_diagnostics(shocks, name="scenario"):
    comp, deltas = composite_surrogate(shocks)
    pol = policy_penalty(shocks)
    ext = extremeness_penalty(shocks)
    total_obj = -comp + PENALTY_WEIGHT*pol + EXTREMENESS_WEIGHT*ext

    return {
        "name": name,
        "composite": comp,
        "policy_penalty": pol,
        "extreme_penalty": ext,
        "objective_value": total_obj,
        **{f"delta_{k}": v for k, v in deltas.items()},
        **{f"shock_{k}": v for k, v in shocks.items()}
    }

# (A) SURROGATE VALIDATION
print("Running surrogate validation...")
surrogate_val_df = evaluate_surrogate_models(df_doe, input_vars, output_vars, test_size=0.25, seed=42)
surrogate_val_path = os.path.join(VALID_DIR, "surrogate_validation.csv")
surrogate_val_df.to_csv(surrogate_val_path, index=False)

print(surrogate_val_df)
print("Saved:", surrogate_val_path)

# (B) OPT SANITY CHECKS
print("\nRunning optimization sanity checks...")
baseline_diag = compute_diagnostics({v: 0.0 for v in input_vars}, name="baseline")
opt_diag      = compute_diagnostics(best_shocks, name="composite_opt")

sanity_df = pd.DataFrame([baseline_diag, opt_diag])
sanity_path = os.path.join(VALID_DIR, "optimization_sanity_checks.csv")
sanity_df.to_csv(sanity_path, index=False)

print(sanity_df[["name", "composite", "policy_penalty", "extreme_penalty", "objective_value"]])
print("Saved:", sanity_path)

# (C) HORIZON SENSITIVITY
print("\nRunning horizon sensitivity (baseline + optimal)...")
horizons = [6, 12, 18]
rows = []

for H in horizons:
    baseH = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, horizon=H
    )
    optH = simulate_path_shocks(
        best_shocks,
        df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, horizon=H
    )

    base_means = baseH[output_vars].mean().to_dict()
    opt_meansH = optH[output_vars].mean().to_dict()

    rows.append({
        "horizon": H,
        **{f"baseline_mean_{y}": base_means[y] for y in output_vars},
        **{f"opt_mean_{y}": opt_meansH[y] for y in output_vars},
        **{f"delta_mean_{y}": opt_meansH[y] - base_means[y] for y in output_vars}
    })

horizon_df = pd.DataFrame(rows)
horizon_path = os.path.join(VALID_DIR, "horizon_sensitivity.csv")
horizon_df.to_csv(horizon_path, index=False)

print(horizon_df[["horizon"] + [f"delta_mean_{y}" for y in output_vars]])
print("Saved:", horizon_path)

# (D) ANCHOR ROBUSTNESS
print("\nRunning anchor-date robustness (fast)...")

anchor_indices = [
    df_feat.index[:split][-1],
    df_feat.index[:split][-1 - 12] if split-12 > 0 else df_feat.index[:split][-1],
    df_feat.index[:split][-1 - 24] if split-24 > 0 else df_feat.index[:split][-1]
]

anchor_rows = []

for a in anchor_indices:
    history_start_a = df_feat.index[df_feat.index.get_loc(a) - max(lags)]
    base_history_a = df_base.loc[history_start_a:a, input_vars + output_vars].copy()

    baseA = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels, tcodes, base_history_a, a,
        X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON
    )
    optA = simulate_path_shocks(
        best_shocks,
        df_levels, tcodes, base_history_a, a,
        X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON
    )

    base_meansA = baseA[output_vars].mean().to_dict()
    opt_meansA  = optA[output_vars].mean().to_dict()

    anchor_rows.append({
        "anchor_date": str(a.date()),
        **{f"delta_mean_{y}": opt_meansA[y] - base_meansA[y] for y in output_vars}
    })

anchor_df = pd.DataFrame(anchor_rows)
anchor_path = os.path.join(VALID_DIR, "anchor_robustness.csv")
anchor_df.to_csv(anchor_path, index=False)

print(anchor_df)
print("Saved:", anchor_path)

print("\n Paper-grade validations complete. Outputs saved in:", VALID_DIR)




SENS_DIR = VALID_DIR
os.makedirs(SENS_DIR, exist_ok=True)



def composite_surrogate_with_weights(shocks, weight_map):
    ds, ws = [], []
    deltas = {}
    x = np.array([[shocks[v] for v in input_vars]])

    for y in output_vars:
        dval = float(surrogate_models[y].predict(x)[0])
        deltas[y] = dval

        d = max(1e-9, desirability(dval, rules[y]))
        w = max(float(weight_map.get(y, 1.0)), 1e-6)

        ds.append(d)
        ws.append(w)

    ds = np.array(ds)
    ws = np.array(ws)

    comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))
    return comp, deltas



def make_objective_weighted(weight_map):
    def obj(theta):
        shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
        comp, _ = composite_surrogate_with_weights(shocks, weight_map)
        return -comp + PENALTY_WEIGHT * policy_penalty(shocks) + EXTREMENESS_WEIGHT * extremeness_penalty(shocks)
    return obj


def optimize_with_objective(obj_func, bounds):
    def run_opt(x0):
        return minimize(obj_func, x0=x0, bounds=bounds, method="L-BFGS-B")

    best_res = None
    best_val = 1e18

    starts_local = [np.zeros(len(input_vars))]
    for _ in range(25):
        starts_local.append(np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds]))

    for x0 in starts_local:
        res_try = run_opt(x0)
        if res_try.fun < best_val:
            best_val = res_try.fun
            best_res = res_try

    return best_res



print("Running weight sensitivity tests...")

# Define paper weight schemes
weight_schemes = {
    "R2_based": {y: max(r2_map.get(y, 0.1), 0.1) for y in output_vars},
    "Equal": {y: 1.0 for y in output_vars},
    "Inflation_priority": {"CPI_YOY": 2.0, "UNRATE_L": 1.0, "BAAFFM_T": 1.0},
    "Unemployment_priority": {"UNRATE_L": 2.0, "CPI_YOY": 1.0, "BAAFFM_T": 1.0},
}

bounds_default = [shock_bounds[v] for v in input_vars]

weight_rows = []

for name, wmap in weight_schemes.items():
    obj_func = make_objective_weighted(wmap)
    resw = optimize_with_objective(obj_func, bounds_default)

    shocks_w = {v: float(resw.x[i]) for i, v in enumerate(input_vars)}
    comp_w, deltas_w = composite_surrogate_with_weights(shocks_w, wmap)

    # Validate using full emulator simulation
    opt_path_w = simulate_path_shocks(
        shocks_w, df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    opt_means_w = opt_path_w[output_vars].mean().to_dict()

    # Baseline means already computed
    delta_actual = {y: opt_means_w[y] - baseline_means[y] for y in output_vars}

    weight_rows.append({
        "weight_scheme": name,
        "composite_score": comp_w,
        **{f"shock_{v}": shocks_w[v] for v in input_vars},
        **{f"delta_mean_{y}": delta_actual[y] for y in output_vars},
    })

df_weight_sens = pd.DataFrame(weight_rows)
weight_sens_path = os.path.join(SENS_DIR, "weight_sensitivity_results.csv")
df_weight_sens.to_csv(weight_sens_path, index=False)

print("\n Weight sensitivity saved to:", weight_sens_path)
print(df_weight_sens)

print("\nRunning shock-bound sensitivity tests...")

def scaled_bounds(scale):
    out = {}
    for v in input_vars:
        lo, hi = shock_bounds[v]
        out[v] = (lo * scale, hi * scale)
    return out

bound_scales = [0.50, 1.00, 1.25]
bound_rows = []

for scale in bound_scales:
    shock_bounds_scaled = scaled_bounds(scale)
    bounds_scaled = [shock_bounds_scaled[v] for v in input_vars]

    # New optimization objective must use NEW extremeness penalty based on scaled bounds
    def extremeness_penalty_scaled(shocks):
        pen = 0.0
        for v in input_vars:
            lo, hi = shock_bounds_scaled[v]
            mid = 0.5*(lo+hi)
            width = hi-lo
            dist = abs(shocks[v] - mid)
            thresh = 0.40*width
            if dist > thresh:
                pen += ((dist - thresh)/(0.60*width))**2
        return pen

    def objective_scaled(theta):
        shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
        comp, _ = composite_surrogate(shocks)
        return -comp + PENALTY_WEIGHT*policy_penalty(shocks) + EXTREMENESS_WEIGHT*extremeness_penalty_scaled(shocks)

    resb = optimize_with_objective(objective_scaled, bounds_scaled)

    shocks_b = {v: float(resb.x[i]) for i, v in enumerate(input_vars)}
    comp_b, deltas_b = composite_surrogate(shocks_b)

    # Validate using full emulator simulation
    opt_path_b = simulate_path_shocks(
        shocks_b, df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    opt_means_b = opt_path_b[output_vars].mean().to_dict()

    delta_actual_b = {y: opt_means_b[y] - baseline_means[y] for y in output_vars}

    bound_rows.append({
        "bound_scale": scale,
        "composite_score": comp_b,
        **{f"shock_{v}": shocks_b[v] for v in input_vars},
        **{f"delta_mean_{y}": delta_actual_b[y] for y in output_vars},
    })

df_bound_sens = pd.DataFrame(bound_rows)
bound_sens_path = os.path.join(SENS_DIR, "bound_sensitivity_results.csv")
df_bound_sens.to_csv(bound_sens_path, index=False)

print("\n Bound sensitivity saved to:", bound_sens_path)
print(df_bound_sens)

print("\n Weight + bound sensitivity tests complete.")



print("\nRunning bootstrapped uncertainty (moving block bootstrap)...")

BOOT_DIR = VALID_DIR
BOOT_PATH = os.path.join(BOOT_DIR, "bootstrap_opt_uncertainty.csv")


BOOT_M = 30               # number of bootstrap draws (increase to 50-100 for publication)
BLOCK_LEN = 12            # moving block length (months)
SEED_BOOT = 42
np.random.seed(SEED_BOOT)

def moving_block_bootstrap_indices(n, block_len):
    """
    Returns indices of length n sampled using moving blocks of fixed length.
    """
    idx = []
    while len(idx) < n:
        start = np.random.randint(0, n - block_len + 1)
        idx.extend(range(start, start + block_len))
    return np.array(idx[:n])

def fit_emulator_on_bootstrap(df_feat, X_cols, output_vars, lags, split_frac=0.8):
    """
    Fits emulator models on a bootstrap resample of the TRAIN portion only.
    Returns fitted models + scalers.
    """
    n = len(df_feat)
    split = int(split_frac * n)

    # Training sample only (chronological)
    train_df = df_feat.iloc[:split].copy()

    # Bootstrap resample indices with blocks
    boot_idx = moving_block_bootstrap_indices(len(train_df), BLOCK_LEN)
    boot_df = train_df.iloc[boot_idx].copy()

    # Extract matrices
    Xb = boot_df[X_cols].values
    Yb = boot_df[output_vars].values

    scaler_Xb = StandardScaler().fit(Xb)
    Xb_s = scaler_Xb.transform(Xb)

    scaler_Yb = {}
    Yb_s = np.zeros_like(Yb)

    for i, t in enumerate(output_vars):
        sc = StandardScaler().fit(Yb[:, [i]])
        scaler_Yb[t] = sc
        Yb_s[:, i] = sc.transform(Yb[:, [i]]).ravel()

    # Train models
    models_b = {}
    model_obj_b, model_name_b = get_model()

    for i, t in enumerate(output_vars):
        m = copy.deepcopy(model_obj_b)
        m.fit(Xb_s, Yb_s[:, i])
        models_b[t] = m

    return models_b, scaler_Xb, scaler_Yb, split

def run_pipeline_given_emulator(models_b, scaler_Xb, scaler_Yb, split, df_feat, df_base, df_levels,
                                tcodes, anchor_date, X_cols):
    """
    Runs baseline simulation, DOE simulation, surrogate fitting, and composite optimization.
    Returns optimum shocks + optimum deltas (scenario-baseline mean).
    """

    # Base history for simulation
    history_start = df_feat.index[df_feat.index.get_loc(anchor_date) - max(lags)]
    base_history_trans_b = df_base.loc[history_start:anchor_date, input_vars + output_vars].copy()

    # Baseline simulation
    baseline_path_b = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels, tcodes, base_history_trans_b, anchor_date,
        X_cols, scaler_Xb, models_b, scaler_Yb, SIM_HORIZON
    )
    baseline_means_b = baseline_path_b[output_vars].mean().to_dict()

    # DOE simulation
    sampler_b = qmc.Sobol(d=len(input_vars), scramble=True, seed=42)
    X_doe_unit_b = sampler_b.random_base2(m=DOE_M_PAPER)
    X_doe_real_b = np.zeros_like(X_doe_unit_b)

    for j, v in enumerate(input_vars):
        lo, hi = shock_bounds[v]
        X_doe_real_b[:, j] = lo + (hi - lo) * X_doe_unit_b[:, j]

    doe_rows_b = []
    for i in range(X_doe_real_b.shape[0]):
        shocks = {v: float(X_doe_real_b[i, j]) for j, v in enumerate(input_vars)}
        path = simulate_path_shocks(
            shocks, df_levels, tcodes, base_history_trans_b, anchor_date,
            X_cols, scaler_Xb, models_b, scaler_Yb, SIM_HORIZON
        )
        means = path[output_vars].mean().to_dict()
        deltas = {y: means[y] - baseline_means_b[y] for y in output_vars}

        doe_rows_b.append({
            **{f"{v}_shock": shocks[v] for v in input_vars},
            **{f"{y}_delta_mean": deltas[y] for y in output_vars},
        })

    df_doe_b = pd.DataFrame(doe_rows_b)

    # ---- Fit surrogate models (same as your code) ----
    X_surr_b = df_doe_b[[f"{v}_shock" for v in input_vars]].values
    surrogate_models_b = {}

    try:
        import xgboost as xgb
        for y in output_vars:
            m = xgb.XGBRegressor(
                n_estimators=1500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1
            )
            m.fit(X_surr_b, df_doe_b[f"{y}_delta_mean"].values)
            surrogate_models_b[y] = m
    except Exception:
        from sklearn.ensemble import RandomForestRegressor
        for y in output_vars:
            m = RandomForestRegressor(
                n_estimators=800,
                max_depth=12,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            )
            m.fit(X_surr_b, df_doe_b[f"{y}_delta_mean"].values)
            surrogate_models_b[y] = m

    # ---- Feasible ranges + desirability rules (same logic) ----
    feasible_b = {}
    for y in output_vars:
        vals = df_doe_b[f"{y}_delta_mean"].values
        feasible_b[y] = {
            "min": float(np.min(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
        }

    rules_b = {}
    for y in output_vars:
        vals = df_doe_b[f"{y}_delta_mean"].values
        U = max(abs(feasible_b[y]["min"]), abs(feasible_b[y]["max"]))

        if y == "CPI_YOY":
            T = float(np.percentile(vals, 10))
            T = min(0.0, T)
            band = max(0.10, abs(T) * 0.35)
            rules_b[y] = {"typ": "target", "T": T, "band": band, "U": U}
        elif y in ["UNRATE_L", "BAAFFM_T"]:
            T = float(np.percentile(vals, 25))
            band = max(0.10, abs(T) * 0.35)
            rules_b[y] = {"typ": "target", "T": T, "band": band, "U": U}
        else:
            rules_b[y] = {"typ": "smaller", "T": feasible_b[y]["p10"], "U": feasible_b[y]["p95"]}

    def desirability_b(val, rule):
        typ = rule["typ"]
        if typ == "smaller":
            T, U = rule["T"], rule["U"]
            if val >= U: return 0.0
            if val <= T: return 1.0
            return (U - val) / (U - T)
        if typ == "target":
            T, band, U = rule["T"], rule["band"], abs(rule["U"])
            dist = abs(val - T)
            if dist <= band: return 1.0
            if dist >= U: return 0.0
            return (U - dist) / (U - band)
        raise ValueError("Unknown typ")

    # ---- Composite objective same form ----
    def composite_surrogate_b(shocks):
        ds, ws = [], []
        deltas = {}
        x = np.array([[shocks[v] for v in input_vars]])

        for y in output_vars:
            dval = float(surrogate_models_b[y].predict(x)[0])
            deltas[y] = dval
            d = max(1e-9, desirability_b(dval, rules_b[y]))
            w = max(r2_map.get(y, 0.1), 0.1)
            ds.append(d); ws.append(w)

        ds = np.array(ds); ws = np.array(ws)
        comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))
        return comp, deltas

    def objective_b(theta):
        shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
        comp, _ = composite_surrogate_b(shocks)
        return -comp + PENALTY_WEIGHT*policy_penalty(shocks) + EXTREMENESS_WEIGHT*extremeness_penalty(shocks)

    bounds_b = [shock_bounds[v] for v in input_vars]

    def run_opt_b(x0):
        return minimize(objective_b, x0=x0, bounds=bounds_b, method="L-BFGS-B")

    best_res_b = None
    best_val_b = 1e18

    starts_b = [np.zeros(len(input_vars))]
    for _ in range(25):
        starts_b.append(np.array([np.random.uniform(lo, hi) for (lo, hi) in bounds_b]))

    for x0 in starts_b:
        res_try = run_opt_b(x0)
        if res_try.fun < best_val_b:
            best_val_b = res_try.fun
            best_res_b = res_try

    resb = best_res_b
    best_shocks_b = {v: float(resb.x[i]) for i, v in enumerate(input_vars)}
    best_comp_b, best_deltas_b = composite_surrogate_b(best_shocks_b)

    return best_shocks_b, best_deltas_b, best_comp_b

# ---- Bootstrap Loop ----
boot_rows = []
n_total = len(df_feat)
split0 = int(0.8 * n_total)
anchor_date0 = df_feat.index[:split0][-1]

for m in range(BOOT_M):
    try:
        models_b, scaler_Xb, scaler_Yb, split_b = fit_emulator_on_bootstrap(df_feat, X_cols, output_vars, lags)

        best_shocks_b, best_deltas_b, best_comp_b = run_pipeline_given_emulator(
            models_b, scaler_Xb, scaler_Yb, split_b, df_feat, df_base, df_levels,
            tcodes, anchor_date0, X_cols
        )

        boot_rows.append({
            "boot_id": m,
            "composite": best_comp_b,
            **{f"shock_{v}": best_shocks_b[v] for v in input_vars},
            **{f"delta_{y}": best_deltas_b[y] for y in output_vars},
        })

        print(f"  Bootstrap {m+1}/{BOOT_M} done.")

    except Exception as e:
        print(f"  ⚠️ Bootstrap {m+1}/{BOOT_M} failed: {e}")

boot_df = pd.DataFrame(boot_rows)

if len(boot_df) > 5:
    boot_df.to_csv(BOOT_PATH, index=False)

    # Summary with CI bands
    summary_rows = []
    for col in boot_df.columns:
        if col in ["boot_id"]:
            continue
        vals = boot_df[col].dropna().values
        if len(vals) < 5:
            continue
        summary_rows.append({
            "metric": col,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p05": float(np.percentile(vals, 5)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p95": float(np.percentile(vals, 95)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(BOOT_DIR, "bootstrap_opt_uncertainty_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n Bootstrap uncertainty results saved:")
    print("   Raw bootstrap draws:", BOOT_PATH)
    print("   Summary statistics :", summary_path)

else:
    print("\n⚠️ Not enough bootstrap samples succeeded; check errors above.")




REAL_DIR = os.path.join(REPORTS_DIR, "realism_validation_tests")
os.makedirs(REAL_DIR, exist_ok=True)


def build_future_levels_from_shocks_decaying(baseline_levels: pd.Series, shocks: dict, future_dates: list, rho=0.85):
    """
    Apply policy shock vector with exponential decay:
      shock_{t+h} = rho^h * shock
    - FEDFUNDS additive shift in level
    - USGOVT additive monthly increments
    - NONBORRES additive monthly increments (floored at 0)
    - BOGMBASE multiplicative monthly scaling using exp(shock)
    """
    paths = pd.DataFrame(index=future_dates, columns=input_vars, dtype=float)
    last = baseline_levels.copy()

    for h, t in enumerate(future_dates, start=1):
        scale = rho**(h-1)

        # Decayed shocks
        fed_shock  = shocks["FEDFUNDS"]  * scale
        usg_shock  = shocks["USGOVT"]    * scale
        nres_shock = shocks["NONBORRES"] * scale
        base_shock = shocks["BOGMBASE"]  * scale

        # Apply to levels
        last["FEDFUNDS"]  = baseline_levels["FEDFUNDS"] + fed_shock
        last["USGOVT"]    = last["USGOVT"] + usg_shock
        last["NONBORRES"] = max(0.0, last["NONBORRES"] + nres_shock)
        last["BOGMBASE"]  = max(1e-6, last["BOGMBASE"] * float(np.exp(base_shock)))

        paths.loc[t] = last.values

    return paths



def simulate_path_shocks_with_decay(
    shocks: dict,
    df_levels: pd.DataFrame,
    tcodes: dict,
    base_history_trans: pd.DataFrame,
    anchor_date: pd.Timestamp,
    X_cols: list,
    scaler_X: StandardScaler,
    models: dict,
    scaler_Y: dict,
    horizon: int = 12,
    decay: bool = False,
    rho: float = 0.85
):
    hist_trans = base_history_trans.copy()
    hist_levels = df_levels[input_vars].loc[:anchor_date].copy()
    baseline_levels = hist_levels.loc[anchor_date].copy()

    future_dates = []
    last = anchor_date
    for _ in range(horizon):
        last = last + pd.offsets.MonthBegin(1)
        future_dates.append(last)

    if decay:
        future_paths = build_future_levels_from_shocks_decaying(baseline_levels, shocks, future_dates, rho=rho)
    else:
        future_paths = build_future_levels_from_shocks(baseline_levels, shocks, future_dates)

    for t in future_dates:
        hist_levels.loc[t] = future_paths.loc[t].values

    out_rows = []
    for step, t in enumerate(future_dates, start=1):
        lever_current = {}
        for v in input_vars:
            lever_current[v] = float(apply_tcode(hist_levels[v], tcodes.get(v, 1)).loc[t])

        row = {v: lever_current[v] for v in input_vars}
        for col in (input_vars + output_vars):
            for L in lags:
                row[f"{col}_lag{L}"] = float(hist_trans[col].iloc[-L])

        x = np.array([row[c] for c in X_cols], dtype=float).reshape(1, -1)
        x_s = scaler_X.transform(x)

        y_pred = {}
        for y in output_vars:
            y_s = float(models[y].predict(x_s)[0])
            y_pred[y] = float(scaler_Y[y].inverse_transform([[y_s]])[0, 0])

        new_row = pd.Series({**{v: lever_current[v] for v in input_vars},
                             **{y: y_pred[y] for y in output_vars}}, name=t)
        hist_trans = pd.concat([hist_trans, new_row.to_frame().T], axis=0)

        out_rows.append({
            "step": step,
            **y_pred,
            **{f"{v}_LEVEL": float(hist_levels.loc[t, v]) for v in input_vars}
        })

    return pd.DataFrame(out_rows)


print("TEST (1): Persistent vs Decaying shock structure...")

shock_struct_results = []

# baseline persistent
base_persist = simulate_path_shocks_with_decay(
    {v: 0.0 for v in input_vars}, df_levels, tcodes, base_history_trans,
    anchor_date, X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON,
    decay=False
)
base_means_persist = base_persist[output_vars].mean().to_dict()

# optimum persistent
opt_persist = simulate_path_shocks_with_decay(
    best_shocks, df_levels, tcodes, base_history_trans,
    anchor_date, X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON,
    decay=False
)
opt_means_persist = opt_persist[output_vars].mean().to_dict()

# optimum decaying (test rho=0.85 + rho=0.65)
for rho in [0.85, 0.65]:
    opt_decay = simulate_path_shocks_with_decay(
        best_shocks, df_levels, tcodes, base_history_trans,
        anchor_date, X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON,
        decay=True, rho=rho
    )
    opt_means_decay = opt_decay[output_vars].mean().to_dict()

    shock_struct_results.append({
        "shock_structure": f"decay_rho_{rho}",
        **{f"delta_mean_{y}": opt_means_decay[y] - base_means_persist[y] for y in output_vars},
        **{f"opt_mean_{y}": opt_means_decay[y] for y in output_vars}
    })

shock_struct_results.append({
    "shock_structure": "persistent",
    **{f"delta_mean_{y}": opt_means_persist[y] - base_means_persist[y] for y in output_vars},
    **{f"opt_mean_{y}": opt_means_persist[y] for y in output_vars}
})

df_shock_struct = pd.DataFrame(shock_struct_results)
df_shock_struct.to_csv(os.path.join(REAL_DIR, "shock_structure_robustness.csv"), index=False)

# Save paths for plotting
opt_persist.to_csv(os.path.join(REAL_DIR, "optimal_path_persistent.csv"), index=False)
for rho in [0.85, 0.65]:
    opt_decay = simulate_path_shocks_with_decay(
        best_shocks, df_levels, tcodes, base_history_trans,
        anchor_date, X_cols, scaler_X, models, scaler_Y, horizon=SIM_HORIZON,
        decay=True, rho=rho
    )
    opt_decay.to_csv(os.path.join(REAL_DIR, f"optimal_path_decay_rho{rho}.csv"), index=False)

print(" Saved shock structure robustness:")
print("   ", os.path.join(REAL_DIR, "shock_structure_robustness.csv"))



print("\nTEST (2): Regime split robustness...")

REGIME_SPLITS = [
    ("1959_1979", "1959-01-01", "1979-12-01"),
    ("1980_2007", "1980-01-01", "2007-12-01"),
    ("2008_2019", "2008-01-01", "2019-12-01"),
    ("2020_2025", "2020-01-01", "2025-07-01"),
]

regime_rows = []

def run_regime_pipeline(start_date, end_date, label):
    df_levels_r = df_levels.loc[start_date:end_date].copy()
    df_base_r   = build_base_frame(df_levels_r, tcodes)
    df_feat_r, X_cols_r = make_supervised(df_base_r, lags)

    if len(df_feat_r) < 200:
        print(f"⚠️ Regime {label}: too few observations -> skipped.")
        return None

    Xr = df_feat_r[X_cols_r].values
    Yr = df_feat_r[output_vars].values

    split_r = int(0.8 * len(df_feat_r))
    X_train_r, X_test_r = Xr[:split_r], Xr[split_r:]
    Y_train_r, Y_test_r = Yr[:split_r], Yr[split_r:]

    scaler_Xr = StandardScaler().fit(X_train_r)
    X_train_rs = scaler_Xr.transform(X_train_r)
    X_test_rs  = scaler_Xr.transform(X_test_r)

    scaler_Yr = {}
    Y_train_rs = np.zeros_like(Y_train_r)
    Y_test_rs  = np.zeros_like(Y_test_r)

    for i, t in enumerate(output_vars):
        sc = StandardScaler().fit(Y_train_r[:, [i]])
        scaler_Yr[t] = sc
        Y_train_rs[:, i] = sc.transform(Y_train_r[:, [i]]).ravel()
        Y_test_rs[:, i]  = sc.transform(Y_test_r[:, [i]]).ravel()

    models_r = {}
    model_obj_r, model_name_r = get_model()

    perf = {}
    for i, t in enumerate(output_vars):
        m = copy.deepcopy(model_obj_r)
        m.fit(X_train_rs, Y_train_rs[:, i])
        models_r[t] = m
        yhat = m.predict(X_test_rs)
        perf[t] = float(r2_score(Y_test_rs[:, i], yhat))

    # anchor at end of regime train
    anchor_r = df_feat_r.index[:split_r][-1]
    history_start_r = df_feat_r.index[df_feat_r.index.get_loc(anchor_r) - max(lags)]
    base_hist_r = df_base_r.loc[history_start_r:anchor_r, input_vars + output_vars].copy()

    baseline_r = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels_r, tcodes, base_hist_r, anchor_r,
        X_cols_r, scaler_Xr, models_r, scaler_Yr, SIM_HORIZON
    )
    baseline_means_r = baseline_r[output_vars].mean().to_dict()

    opt_r = simulate_path_shocks(
        best_shocks,
        df_levels_r, tcodes, base_hist_r, anchor_r,
        X_cols_r, scaler_Xr, models_r, scaler_Yr, SIM_HORIZON
    )
    opt_means_r = opt_r[output_vars].mean().to_dict()

    row = {
        "regime": label,
        "anchor_date": str(anchor_r.date()),
        **{f"R2_{y}": perf[y] for y in output_vars},
        **{f"delta_mean_{y}": opt_means_r[y] - baseline_means_r[y] for y in output_vars},
    }
    return row

for label, sd, ed in REGIME_SPLITS:
    out = run_regime_pipeline(sd, ed, label)
    if out is not None:
        regime_rows.append(out)

df_regime = pd.DataFrame(regime_rows)
df_regime.to_csv(os.path.join(REAL_DIR, "regime_split_robustness.csv"), index=False)

print(" Saved regime split robustness:")
print("   ", os.path.join(REAL_DIR, "regime_split_robustness.csv"))

print("\nTEST (3): Placebo policy event checks...")

# Choose 3 anchor dates where tightening / easing direction should be reasonable.
# You can change these.
PLACEBO_ANCHORS = [
    ("Volcker_1979", "1979-10-01"),
    ("GFC_2008", "2008-10-01"),
    ("Inflation_2022", "2022-06-01"),
]

placebo_rows = []

def placebo_check(date_str, label):
    a = pd.Timestamp(date_str)
    if a not in df_feat.index:
        # use nearest available index
        a = df_feat.index[df_feat.index.get_indexer([a], method="nearest")[0]]

    history_start_a = df_feat.index[df_feat.index.get_loc(a) - max(lags)]
    base_hist_a = df_base.loc[history_start_a:a, input_vars + output_vars].copy()

    # baseline
    baseA = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels, tcodes, base_hist_a, a,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    base_meansA = baseA[output_vars].mean().to_dict()

    # tightening placebo shock
    placebo_shock = {
        "FEDFUNDS": +2.0,
        "USGOVT": 0.0,
        "BOGMBASE": -0.02,
        "NONBORRES": -150.0
    }

    shockA = simulate_path_shocks(
        placebo_shock,
        df_levels, tcodes, base_hist_a, a,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    shock_meansA = shockA[output_vars].mean().to_dict()

    row = {
        "label": label,
        "anchor_used": str(a.date()),
        **{f"delta_mean_{y}": shock_meansA[y] - base_meansA[y] for y in output_vars},
        **{f"baseline_mean_{y}": base_meansA[y] for y in output_vars},
        **{f"shock_mean_{y}": shock_meansA[y] for y in output_vars},
    }

    return row

for label, date_str in PLACEBO_ANCHORS:
    placebo_rows.append(placebo_check(date_str, label))

df_placebo = pd.DataFrame(placebo_rows)
df_placebo.to_csv(os.path.join(REAL_DIR, "placebo_policy_event_checks.csv"), index=False)

print("Saved placebo sanity checks:")
print("   ", os.path.join(REAL_DIR, "placebo_policy_event_checks.csv"))


print("\nTEST (4): Multi-step forecast degradation...")

H_LIST = list(range(1, 19))  # horizons 1 to 18 months
degrade_rows = []

for H in H_LIST:
    # Emulator recursive baseline for horizon H
    emu_path_H = simulate_path_shocks(
        {v: 0.0 for v in input_vars},
        df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, horizon=H
    )

    emu_forecast_df = emu_path_H[["step"] + output_vars].copy()
    eval_emu = compare_forecasts_to_actual(df_base, anchor_date, H, emu_forecast_df, output_vars)
    eval_emu["model"] = "Emulator (recursive)"

    # Persistence
    persist_path_H = persistence_baseline(df_base, anchor_date, H, output_vars)
    eval_pers = compare_forecasts_to_actual(df_base, anchor_date, H, persist_path_H, output_vars)
    eval_pers["model"] = "Persistence"

    # AR(12)
    ar_path_H = ar_baseline(df_base, anchor_date, H, output_vars, p=12)
    eval_arH = compare_forecasts_to_actual(df_base, anchor_date, H, ar_path_H, output_vars)
    eval_arH["model"] = "AR(12)"

    # VAR(6)
    var_path_H = var_baseline(df_base, anchor_date, H, output_vars, p=6)
    eval_varH = compare_forecasts_to_actual(df_base, anchor_date, H, var_path_H, output_vars)
    eval_varH["model"] = "VAR(6)"

    allH = pd.concat([eval_emu, eval_pers, eval_arH, eval_varH], ignore_index=True)
    allH["horizon"] = H
    degrade_rows.append(allH)

df_degrade = pd.concat(degrade_rows, ignore_index=True)
df_degrade.to_csv(os.path.join(REAL_DIR, "multistep_forecast_degradation.csv"), index=False)

# Plot RMSE vs horizon for each target
for target in output_vars:
    plt.figure()
    for model in df_degrade["model"].unique():
        sub = df_degrade[(df_degrade["target"] == target) & (df_degrade["model"] == model)]
        plt.plot(sub["horizon"], sub["RMSE"], marker="o", label=model)
    plt.title(f"Multi-step Forecast Degradation (RMSE): {target}")
    plt.xlabel("Forecast horizon (months)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REAL_DIR, f"degradation_rmse_{target}.png"), dpi=200)
    plt.close()

print(" Saved multi-step forecast degradation:")
print("   ", os.path.join(REAL_DIR, "multistep_forecast_degradation.csv"))
print("   PNG plots:", [f"degradation_rmse_{t}.png" for t in output_vars])

print("\n All 4 realism + validation tests complete.")
print("Outputs saved in:", REAL_DIR)



MO_DIR = os.path.join(REPORTS_DIR, "mo_baselines")
os.makedirs(MO_DIR, exist_ok=True)


def objective_losses_from_rules(deltas: dict, rules: dict):
    """
    Convert predicted delta outcomes into objective "losses" for multi-objective optimization.
    Smaller is better.

    - typ == "smaller": loss = delta (assumes you want delta as low as possible)
    - typ == "target": loss = |delta - T|
    """
    losses = {}
    for y, rule in rules.items():
        val = float(deltas[y])
        if rule["typ"] == "smaller":
            losses[y] = float(val)
        elif rule["typ"] == "target":
            losses[y] = float(abs(val - float(rule["T"])))
        else:
            losses[y] = float(val)
    return losses

def constraint_violation(shocks: dict, extreme_soft_cap: float = 0.25):
    """
    Define feasibility for baseline comparisons.
    - policy_penalty must be 0
    - extremeness_penalty should be <= extreme_soft_cap (soft constraint)
    Return a nonnegative violation value.
    """
    pol = float(policy_penalty(shocks))
    ext = float(extremeness_penalty(shocks))
    vio = 0.0
    if pol > 0:
        vio += pol
    if ext > extreme_soft_cap:
        vio += (ext - extreme_soft_cap)
    return float(max(0.0, vio))

def evaluate_candidate_via_emulator(theta):
    """
    theta: np.array of shape (d,)
    Returns:
      shocks dict
      deltas dict (actual emulator deltas vs baseline)
      composite score (actual, computed from emulator deltas using desirability rules)
      losses dict (rule-based losses from actual emulator deltas)
      violation float
    """
    shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}
    vio = constraint_violation(shocks)

    # Run full emulator simulation (ground truth for evaluation)
    path = simulate_path_shocks(
        shocks, df_levels, tcodes, base_history_trans, anchor_date,
        X_cols, scaler_X, models, scaler_Y, SIM_HORIZON
    )
    means = path[output_vars].mean().to_dict()
    deltas = {y: float(means[y] - baseline_means[y]) for y in output_vars}

    # Composite score using the SAME desirability + weights concept,
    # but computed on emulator deltas (not surrogate)
    ds, ws = [], []
    for y in output_vars:
        d = max(1e-9, float(desirability(deltas[y], rules[y])))
        w = max(r2_map.get(y, 0.1), 0.1)
        ds.append(d); ws.append(w)
    ds = np.array(ds); ws = np.array(ws)
    comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))

    losses = objective_losses_from_rules(deltas, rules)
    return shocks, deltas, comp, losses, vio


def random_search_baseline(n_evals: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    d = len(input_vars)
    lo = np.array([shock_bounds[v][0] for v in input_vars], dtype=float)
    hi = np.array([shock_bounds[v][1] for v in input_vars], dtype=float)

    rows = []
    feasible_count = 0

    for i in range(n_evals):
        theta = lo + (hi - lo) * rng.random(d)
        shocks, deltas, comp, losses, vio = evaluate_candidate_via_emulator(theta)
        feasible = (vio <= 0.0)
        feasible_count += int(feasible)

        rows.append({
            "method": "RandomSearch",
            "eval_id": i,
            "feasible": int(feasible),
            "violation": vio,
            "composite": comp,
            **{f"shock_{v}": shocks[v] for v in input_vars},
            **{f"delta_{y}": deltas[y] for y in output_vars},
            **{f"loss_{y}": losses[y] for y in output_vars},
        })

    df = pd.DataFrame(rows)
    return df, feasible_count / max(1, n_evals)

def weighted_sum_multistart(n_weights: int = 12, n_starts: int = 15, seed: int = 7):
    """
    Standard baseline: optimize different weightings of the rule-based losses.
    Uses L-BFGS-B with multi-start on the SURROGATE (fast),
    but evaluates best candidates with the FULL emulator (fair comparison).
    """
    rng = np.random.default_rng(seed)
    bounds = np.array([shock_bounds[v] for v in input_vars], dtype=float)

    # create weights on simplex
    W = rng.random((n_weights, len(output_vars)))
    W = W / W.sum(axis=1, keepdims=True)

    rows = []

    for wi in range(n_weights):
        w = W[wi].copy()

        def weighted_loss_surrogate(theta):
            shocks = {v: float(theta[i]) for i, v in enumerate(input_vars)}

            # surrogate deltas
            x = np.array([[shocks[v] for v in input_vars]])
            deltas_s = {y: float(surrogate_models[y].predict(x)[0]) for y in output_vars}
            losses_s = objective_losses_from_rules(deltas_s, rules)

            # constraint as penalty (same form as your main objective)
            vio = constraint_violation(shocks)

            # weighted sum of losses (smaller better)
            L = 0.0
            for j, y in enumerate(output_vars):
                L += float(w[j]) * float(losses_s[y])

            # add soft penalties so optimizer prefers feasible regions
            return float(L + 2.0 * vio)

        # multi-start on surrogate
        best_theta = None
        best_val = 1e18

        starts = [np.zeros(len(input_vars), dtype=float)]
        for _ in range(n_starts):
            x0 = np.array([rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(len(input_vars))], dtype=float)
            starts.append(x0)

        for x0 in starts:
            res = minimize(weighted_loss_surrogate, x0=x0, bounds=bounds.tolist(), method="L-BFGS-B")
            if float(res.fun) < best_val:
                best_val = float(res.fun)
                best_theta = np.array(res.x, dtype=float)

        # evaluate that candidate via emulator
        shocks, deltas, comp, losses, vio = evaluate_candidate_via_emulator(best_theta)
        feasible = int(vio <= 0.0)

        rows.append({
            "method": "WeightedSumMultiStart",
            "eval_id": wi,
            "feasible": feasible,
            "violation": vio,
            "composite": comp,
            "w0": float(w[0]), "w1": float(w[1]), "w2": float(w[2]),
            **{f"shock_{v}": shocks[v] for v in input_vars},
            **{f"delta_{y}": deltas[y] for y in output_vars},
            **{f"loss_{y}": losses[y] for y in output_vars},
        })

    df = pd.DataFrame(rows)
    feas_rate = df["feasible"].mean() if len(df) else 0.0
    return df, float(feas_rate)


def doe_only_ablation(df_doe: pd.DataFrame):
    """
    df_doe already contains emulator-evaluated deltas for DOE points.
    We'll compute composite score on those deltas and pick the best feasible (by our constraint check).
    """
    rows = []
    for i, r in df_doe.iterrows():
        shocks = {v: float(r[f"{v}_shock"]) for v in input_vars}
        deltas = {y: float(r[f"{y}_delta_mean"]) for y in output_vars}

        vio = constraint_violation(shocks)
        feasible = int(vio <= 0.0)

        # composite from deltas
        ds, ws = [], []
        for y in output_vars:
            d = max(1e-9, float(desirability(deltas[y], rules[y])))
            w = max(r2_map.get(y, 0.1), 0.1)
            ds.append(d); ws.append(w)
        ds = np.array(ds); ws = np.array(ws)
        comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))

        losses = objective_losses_from_rules(deltas, rules)

        rows.append({
            "method": "DOE_only",
            "eval_id": int(r.get("doe_id", i)),
            "feasible": feasible,
            "violation": float(vio),
            "composite": comp,
            **{f"shock_{v}": shocks[v] for v in input_vars},
            **{f"delta_{y}": deltas[y] for y in output_vars},
            **{f"loss_{y}": losses[y] for y in output_vars},
        })

    out = pd.DataFrame(rows)
    feas_rate = out["feasible"].mean() if len(out) else 0.0
    return out, float(feas_rate)


def your_method_row(best_shocks, opt_means):
    deltas = {y: float(opt_means[y] - baseline_means[y]) for y in output_vars}
    vio = constraint_violation(best_shocks)
    feasible = int(vio <= 0.0)

    ds, ws = [], []
    for y in output_vars:
        d = max(1e-9, float(desirability(deltas[y], rules[y])))
        w = max(r2_map.get(y, 0.1), 0.1)
        ds.append(d); ws.append(w)
    ds = np.array(ds); ws = np.array(ws)
    comp = float(np.exp((ws * np.log(ds)).sum() / ws.sum()))

    losses = objective_losses_from_rules(deltas, rules)

    return pd.DataFrame([{
        "method": "Surrogate+LBFGSB",
        "eval_id": 0,
        "feasible": feasible,
        "violation": float(vio),
        "composite": comp,
        **{f"shock_{v}": float(best_shocks[v]) for v in input_vars},
        **{f"delta_{y}": deltas[y] for y in output_vars},
        **{f"loss_{y}": losses[y] for y in output_vars},
    }])

try:
    N_DOE_CURRENT = int(len(df_doe[df_doe["doe_id"] >= 0]))
except Exception:
    N_DOE_CURRENT = 2 ** DOE_M_PAPER

RAND_EVALS = int(N_DOE_CURRENT)  # fair-ish comparison


# (1) Random Search
df_rand, rand_feas = random_search_baseline(n_evals=RAND_EVALS, seed=101)
df_rand.to_csv(os.path.join(MO_DIR, "random_search_results.csv"), index=False)
print(f" RandomSearch done. Feasible rate={rand_feas:.3f}. Saved random_search_results.csv")

# (2) Weighted-sum Multi-start
df_wsum, wsum_feas = weighted_sum_multistart(n_weights=12, n_starts=15, seed=202)
df_wsum.to_csv(os.path.join(MO_DIR, "weighted_sum_multistart_results.csv"), index=False)
print(f" WeightedSumMultiStart done. Feasible rate={wsum_feas:.3f}. Saved weighted_sum_multistart_results.csv")

# (3) DOE-only ablation
df_doe_only, doe_feas = doe_only_ablation(df_doe)
df_doe_only.to_csv(os.path.join(MO_DIR, "doe_only_ablation_results.csv"), index=False)
print(f" DOE-only ablation done. Feasible rate={doe_feas:.3f}. Saved doe_only_ablation_results.csv")

# (4) Your method row
df_yours = your_method_row(best_shocks, opt_means)
df_yours.to_csv(os.path.join(MO_DIR, "your_method_result.csv"), index=False)
print(" Saved your method row to your_method_result.csv")

# Combine all candidate points for plotting and table
df_all = pd.concat([df_rand, df_wsum, df_doe_only, df_yours], ignore_index=True)
df_all.to_csv(os.path.join(MO_DIR, "pareto_dataset_all.csv"), index=False)


def summarize_method(df, method_name: str):
    sub = df[df["method"] == method_name].copy()
    if len(sub) == 0:
        return None

    feas = sub[sub["feasible"] == 1].copy()
    feas_rate = float(sub["feasible"].mean())

    # best composite among feasible if possible
    if len(feas) > 0:
        best_comp = float(feas["composite"].max())
        best_row = feas.loc[feas["composite"].idxmax()]
    else:
        best_comp = float(sub["composite"].max())
        best_row = sub.loc[sub["composite"].idxmax()]

    out = {
        "method": method_name,
        "n_evals": int(len(sub)),
        "feasible_rate": feas_rate,
        "best_composite": best_comp,
        "best_violation": float(best_row["violation"]),
        **{f"best_shock_{v}": float(best_row[f"shock_{v}"]) for v in input_vars},
        **{f"best_delta_{y}": float(best_row[f"delta_{y}"]) for y in output_vars},
    }
    return out

summary_rows = []
for mname in ["Surrogate+LBFGSB", "DOE_only", "RandomSearch", "WeightedSumMultiStart"]:
    r = summarize_method(df_all, mname)
    if r:
        summary_rows.append(r)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(MO_DIR, "mo_baseline_table.csv"), index=False)
print(" Saved MO baseline summary table:", os.path.join(MO_DIR, "mo_baseline_table.csv"))
print(df_summary[["method", "n_evals", "feasible_rate", "best_composite", "best_violation"]])


x_col = "loss_CPI_YOY"
y_col = "loss_UNRATE_L"

plt.figure()
for mname in df_all["method"].unique():
    sub = df_all[df_all["method"] == mname]
    sub = sub[sub["feasible"] == 1]  # plot feasible only (cleaner for paper)
    if len(sub) == 0:
        continue
    plt.scatter(sub[x_col], sub[y_col], label=mname, alpha=0.8)

plt.xlabel("Loss (CPI): |ΔCPI - target|")
plt.ylabel("Loss (UNRATE): |ΔUNRATE - target|")
plt.title("Feasible Pareto Scatter (CPI vs UNRATE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MO_DIR, "pareto_cpi_unrate.png"), dpi=250)
plt.close()

print(" Saved Pareto plot:", os.path.join(MO_DIR, "pareto_cpi_unrate.png"))


best_points = []
for mname in df_all["method"].unique():
    sub = df_all[(df_all["method"] == mname) & (df_all["feasible"] == 1)]
    if len(sub) == 0:
        sub = df_all[(df_all["method"] == mname)]
    if len(sub) == 0:
        continue
    row = sub.loc[sub["composite"].idxmax()]
    best_points.append(row)

df_best = pd.DataFrame(best_points)
df_best.to_csv(os.path.join(MO_DIR, "best_point_each_method.csv"), index=False)
print(" Saved best_point_each_method.csv")

print("\n MO baselines + ablations complete. Outputs in:", MO_DIR)
