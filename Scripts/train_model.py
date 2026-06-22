"""
Phase 5: Age-Stratified Ensemble Model Training
Trains three distinct XGBoost sub-models stratified across specific age brackets.
Features: 9 clinical biomarkers -> Target: PhenotypicAge
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

print("=" * 70)
print(" PHASE 5: AGE-STRATIFIED ENSEMBLE MODEL TRAINING")
print("=" * 70)

# ── 1. Path Configuration (Absolute Root Alignment) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# Ensure target environment directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")
PARAMS_PATH = os.path.join(MODEL_DIR, "best_hpo_params.json")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# Verify asset existence prior to script execution
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Target input file not found: {DATA_PATH}")
    print("[STOP] Ensemble training halted due to missing upstream assets.")
    sys.exit(1)

# ── 2. Hyperparameter Resolution ─────────────────────────────────────────────
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, "r") as f:
        XGB_PARAMS = json.load(f)
    print("[INFO] Successfully resolved optimal hyperparameters from Optuna tuning registry.")
else:
    # Fallback baseline configurations if tuning metrics do not exist
    XGB_PARAMS = dict(
        n_estimators=782, learning_rate=0.01115, max_depth=6,
        min_child_weight=9, subsample=0.589, colsample_bytree=0.728,
        colsample_bylevel=0.948, reg_alpha=7.3e-8, reg_lambda=1.66e-5,
        random_state=42, verbosity=0, n_jobs=-1
    )
    print("[WARNING] Optimal hyperparameter registry not located. Utilizing baseline configurations.")

# Define structural age segments
AGE_GROUPS = {
    "young":  (0,  45),
    "middle": (45, 65),
    "old":    (65, 999),
}

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]
TARGET = "PhenotypicAge"  # Aligned with Phase 3 array schemas

# ── 3. Data Loading and Preprocessing ────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X_all = df[FEATURES].copy()

# Apply log1p mathematical transformation to heavily skewed clinical markers
X_all[LOG_FEATURES] = np.log1p(X_all[LOG_FEATURES])
y_all = df[TARGET]
age_all = df["age"]

print(f"[SUCCESS] Dataset loaded: {len(df):,} rows | Transformations applied to: {LOG_FEATURES}\n")

# Global stratified split for comprehensive cross-validation and benchmarking
X_train_g, X_test_g, y_train_g, y_test_g, age_train_g, age_test_g = \
    train_test_split(X_all, y_all, age_all, test_size=0.2, random_state=42)

print(f"{'Age Segment':<15} {'Train Set':>10} {'Test Set':>10}")
print("─" * 40)

strat_models = {}
group_results = {}

# ── 4. Independent Stratified Sub-Model Evaluation ──────────────────────────
for name, (lo, hi) in AGE_GROUPS.items():
    mask_tr = (age_train_g >= lo) & (age_train_g < hi)
    mask_te = (age_test_g  >= lo) & (age_test_g  < hi)

    Xtr, ytr = X_train_g[mask_tr], y_train_g[mask_tr]
    Xte, yte = X_test_g[mask_te],  y_test_g[mask_te]

    print(f"  {name:<13} {len(Xtr):>10,} {len(Xte):>10,}", end="")

    if len(Xtr) == 0 or len(Xte) == 0:
        print(" -> [SKIP] Insufficient statistical samples available in this partition.")
        continue

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(Xtr, ytr)
    strat_models[name] = model

    y_pred = model.predict(Xte)
    r2 = r2_score(yte, y_pred)
    mae = mean_absolute_error(yte, y_pred)

    cv_scores = cross_val_score(model, Xtr, ytr, cv=5, scoring="r2", n_jobs=-1)
    
    group_results[name] = {
        "R2": round(r2, 4), 
        "MAE": round(mae, 4),
        "CV_R2": round(cv_scores.mean(), 4),
        "n_train": int(len(Xtr)), 
        "n_test": int(len(Xte)),
        "age_range": [lo, hi],
    }
    print(f"   R2={r2:.4f}  MAE={mae:.4f}")

# ── 5. Integrated Routing Ensemble Verification ─────────────────────────────
def route_predict(X_feat, ages):
    preds = np.zeros(len(X_feat))
    for name, (lo, hi) in AGE_GROUPS.items():
        mask = (ages >= lo) & (ages < hi)
        if mask.sum() > 0 and name in strat_models:
            preds[mask] = strat_models[name].predict(X_feat[mask])
    return preds

y_pred_ensemble = route_predict(X_test_g.values, age_test_g.values)
ens_r2 = r2_score(y_test_g, y_pred_ensemble)
ens_mae = mean_absolute_error(y_test_g, y_pred_ensemble)

# Establish unified benchmark baseline model
baseline = XGBRegressor(**XGB_PARAMS)
baseline.fit(X_train_g, y_train_g)
base_r2 = r2_score(y_test_g, baseline.predict(X_test_g))
base_mae = mean_absolute_error(y_test_g, baseline.predict(X_test_g))

print("\n" + "─" * 55)
print(f"{'Architecture Framework':<25} {'R2 Score':>10}  {'MAE (Years)':>15}")
print("─" * 55)
print(f"{'Standard Unified XGBoost':<25} {base_r2:>10.4f}  {base_mae:>15.4f}")
print(f"{'Stratified Ensemble Model':<25} {ens_r2:>10.4f}  {ens_mae:>15.4f}")
print(f"{'Performance delta':<25} {ens_r2 - base_r2:>+10.4f}  {ens_mae - base_mae:>15.4f}")
print("─" * 55)

# ── 6. Serialization and Exportation ─────────────────────────────────────────
# Export independent localized model arrays
for name, model in strat_models.items():
    sub_model_path = os.path.join(MODEL_DIR, f"phenoage_model_{name}.pkl")
    joblib.dump(model, sub_model_path)

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

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[EXPORT] Sub-models successfully serialized inside folder: {MODEL_DIR}")
print(f"[EXPORT] Evaluation telemetry log written to: {META_PATH}")
print("=" * 70)