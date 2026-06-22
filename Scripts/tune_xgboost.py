"""
Phase 4: Hyperparameter Tuning for XGBoost using Optuna
Optimizes the ensemble regressor against the PhenotypicAge baseline estimates.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

# Suppress detailed tuning logs to maintain clean terminal outputs
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print(" MACHINE LEARNING PIPELINE: XGBOOST HYPERPARAMETER TUNING")
print("=" * 70)

# ── 1. Path Configuration (Absolute Root Alignment) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# Ensure the output model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_FILE = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")
PARAMS_PATH = os.path.join(MODEL_DIR, "best_hpo_params.json")
MODEL_PATH = os.path.join(MODEL_DIR, "phenoage_model.pkl")

# Verify source asset availability prior to execution
if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] Target input file not found: {INPUT_FILE}")
    print("[STOP] Tuning pipeline execution halted due to missing upstream assets.")
    sys.exit(1)

# ── 2. Data Ingestion & Variable Mapping ─────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
print(f"[SUCCESS] Dataset loaded successfully: {len(df):,} rows")

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
TARGET = "PhenotypicAge"  # Aligned with Phase 3 output array schema

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Baseline Model Assessment ──────────────────────────────────────────────
baseline = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
baseline.fit(X_train, y_train)
base_r2 = r2_score(y_test, baseline.predict(X_test))
base_mae = mean_absolute_error(y_test, baseline.predict(X_test))

print(f" Baseline Configuration Execution -> R2: {base_r2:.4f} | MAE: {base_mae:.4f} years\n")

# ── 4. Optuna Objective Definition ───────────────────────────────────────────
N_TRIALS = 100

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "random_state":      42,
        "verbosity":         0,
        "n_jobs":            -1,
    }
    model = XGBRegressor(**params)
    # Evaluate using 5-fold cross-validation optimized for R-squared
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    return scores.mean()

# ── 5. Optimization Strategy Execution ───────────────────────────────────────
print(f"Executing {N_TRIALS} Bayesian optimization trials via Optuna...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_cv_r2 = study.best_value
print(f"\n Optimization Cross-Validation Optimal R2: {best_cv_r2:.4f}")
print(" Optimal Hyperparameters Selected:")
for key, value in best_params.items():
    print(f"  - {key}: {value}")

# ── 6. Final Model Retraining & Evaluation ──────────────────────────────────
best_model = XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
best_model.fit(X_train, y_train)

tuned_r2 = r2_score(y_test, best_model.predict(X_test))
tuned_mae = mean_absolute_error(y_test, best_model.predict(X_test))

print("\n" + "─" * 55)
print(f"{'Configuration':<25} {'R2':>10}  {'MAE (Years)':>15}")
print("─" * 55)
print(f"{'Baseline XGBoost':<25} {base_r2:>10.4f}  {base_mae:>15.4f}")
print(f"{'Tuned XGBoost':<25} {tuned_r2:>10.4f}  {tuned_mae:>15.4f}")
print(f"{'Variance delta':<25} {tuned_r2 - base_r2:>+10.4f}  {tuned_mae - base_mae:>15.4f}")
print("─" * 55)

# ── 7. Feature Importance Extractor ──────────────────────────────────────────
importances = dict(zip(FEATURES, best_model.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\nFeature Importance Metrics:")
for feat, imp in sorted_imp:
    marker_length = int(imp * 40)
    bar = "█" * marker_length
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# ── 8. Asset Exportation ─────────────────────────────────────────────────────
# Save tuned model binary weights
joblib.dump(best_model, MODEL_PATH)
print(f"\n[EXPORT] Model parameters serialized to: {MODEL_PATH}")

# Export best hyperparameter arguments
with open(PARAMS_PATH, "w") as f:
    json.dump(best_params, f, indent=4)
print(f"[EXPORT] Optimal configurations saved to: {PARAMS_PATH}")

print("=" * 70)