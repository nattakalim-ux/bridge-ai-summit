"""
Train Biological Age (PhenoAge) prediction model — age-stratified ensemble.
Trains 3 separate XGBoost models (young/middle/old) then routes at inference.
Features: 9 blood markers → Target: phenoage
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

DATA_PATH = "/Users/ming/Desktop/MDCU/Data/phenoage_nhanes_complete.csv"
MODEL_DIR = "/Users/ming/Desktop/MDCU/Model"

FEATURES     = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
                "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]
TARGET       = "phenoage"

# Tuned XGBoost params (from Optuna)
XGB_PARAMS = dict(
    n_estimators=782, learning_rate=0.01115, max_depth=6,
    min_child_weight=9, subsample=0.589, colsample_bytree=0.728,
    colsample_bylevel=0.948, reg_alpha=7.3e-8, reg_lambda=1.66e-5,
    gamma=0.823, random_state=42, verbosity=0, n_jobs=-1,
)

AGE_GROUPS = {
    "young":  (0,  45),
    "middle": (45, 65),
    "old":    (65, 999),
}

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X_all = df[FEATURES].copy()
X_all[LOG_FEATURES] = np.log1p(X_all[LOG_FEATURES])
y_all = df[TARGET]
age_all = df["age"]

print(f"Dataset: {len(df):,} rows | Log1p on: {LOG_FEATURES}\n")

# ── 2. Global train/test split (stratified by age group for fair eval) ────────
X_train_g, X_test_g, y_train_g, y_test_g, age_train_g, age_test_g = \
    train_test_split(X_all, y_all, age_all, test_size=0.2, random_state=42)

print(f"{'Group':<8} {'Train':>7} {'Test':>7}")
print("─" * 24)
strat_models = {}
group_results = {}

# ── 3. Train one model per age group ─────────────────────────────────────────
for name, (lo, hi) in AGE_GROUPS.items():
    mask_tr = (age_train_g >= lo) & (age_train_g < hi)
    mask_te = (age_test_g  >= lo) & (age_test_g  < hi)

    Xtr, ytr = X_train_g[mask_tr], y_train_g[mask_tr]
    Xte, yte = X_test_g[mask_te],  y_test_g[mask_te]

    print(f"  {name:<8} {len(Xtr):>6}   {len(Xte):>5}", end="")

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(Xtr, ytr)
    strat_models[name] = model

    y_pred = model.predict(Xte)
    r2  = r2_score(yte, y_pred)
    mae = mean_absolute_error(yte, y_pred)

    cv_scores = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2", n_jobs=-1)
    group_results[name] = {
        "R2": round(r2, 4), "MAE": round(mae, 4),
        "CV_R2": round(cv_scores.mean(), 4),
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "age_range": [lo, hi],
    }
    print(f"   R²={r2:.4f}  MAE={mae:.4f}")

# ── 4. Ensemble evaluation (route each test sample to its group model) ────────
def route_predict(X_feat, ages):
    preds = np.zeros(len(X_feat))
    for name, (lo, hi) in AGE_GROUPS.items():
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() > 0:
            preds[mask] = strat_models[name].predict(X_feat[mask])
    return preds

y_pred_ensemble = route_predict(X_test_g.values, age_test_g.values)
ens_r2  = r2_score(y_test_g, y_pred_ensemble)
ens_mae = mean_absolute_error(y_test_g, y_pred_ensemble)

# Baseline single-model (re-train for comparison)
baseline = XGBRegressor(**XGB_PARAMS)
baseline.fit(X_train_g, y_train_g)
base_r2  = r2_score(y_test_g, baseline.predict(X_test_g))
base_mae = mean_absolute_error(y_test_g, baseline.predict(X_test_g))

print(f"\n{'─'*48}")
print(f"{'':20} {'R²':>8}  {'MAE':>10}")
print(f"{'─'*48}")
print(f"{'Single XGBoost':<20} {base_r2:>8.4f}  {base_mae:>8.4f} yrs")
print(f"{'Stratified Ensemble':<20} {ens_r2:>8.4f}  {ens_mae:>8.4f} yrs")
print(f"{'Improvement':<20} {ens_r2-base_r2:>+8.4f}  {base_mae-ens_mae:>+8.4f} yrs")
print(f"{'─'*48}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
for name, model in strat_models.items():
    joblib.dump(model, f"{MODEL_DIR}/phenoage_model_{name}.pkl")

meta = {
    "mode":         "stratified_ensemble",
    "age_groups":   {k: list(v) for k, v in AGE_GROUPS.items()},
    "features":     FEATURES,
    "log_features": LOG_FEATURES,
    "target":       TARGET,
    "group_results": group_results,
    "ensemble": {
        "R2": round(ens_r2, 4), "MAE": round(ens_mae, 4),
    },
    "baseline": {
        "R2": round(base_r2, 4), "MAE": round(base_mae, 4),
    },
}
with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → {MODEL_DIR}/phenoage_model_{{young,middle,old}}.pkl")
print(f"Saved → {MODEL_DIR}/model_meta.json")
