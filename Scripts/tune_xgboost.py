"""
Hyperparameter tuning for XGBoost using Optuna (Bayesian optimization).
Tries 100 trials → picks best → saves updated model.
"""

import pandas as pd
import numpy as np
import joblib
import json
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH  = "/Users/ming/Desktop/MDCU/Data/phenoage_nhanes_complete.csv"
MODEL_DIR  = "/Users/ming/Desktop/MDCU/Model"
FEATURES   = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
               "mcv", "rdw", "wbc", "lymph_pct"]
TARGET     = "phenoage"
N_TRIALS   = 100

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES]
y  = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Baseline (current model) ─────────────────────────────────────────────────
baseline = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
baseline.fit(X_train, y_train)
base_r2  = r2_score(y_test, baseline.predict(X_test))
base_mae = mean_absolute_error(y_test, baseline.predict(X_test))
print(f"Baseline XGBoost  →  R²={base_r2:.4f}  MAE={base_mae:.4f} years\n")

# ── Optuna objective ─────────────────────────────────────────────────────────
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
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    # 5-fold CV on training set → optimise for R²
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    return scores.mean()

# ── Run tuning ───────────────────────────────────────────────────────────────
print(f"Running {N_TRIALS} Optuna trials (this takes ~2-4 min)...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_cv_r2  = study.best_value
print(f"\nBest CV R²: {best_cv_r2:.4f}")
print("Best params:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ── Retrain on full train set with best params ───────────────────────────────
best_model = XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
best_model.fit(X_train, y_train)

tuned_r2  = r2_score(y_test, best_model.predict(X_test))
tuned_mae = mean_absolute_error(y_test, best_model.predict(X_test))

print(f"\n{'':─<48}")
print(f"{'':20} {'R²':>8}  {'MAE':>10}")
print(f"{'':─<48}")
print(f"{'Baseline XGBoost':<20} {base_r2:>8.4f}  {base_mae:>8.4f} yrs")
print(f"{'Tuned XGBoost':<20} {tuned_r2:>8.4f}  {tuned_mae:>8.4f} yrs")
print(f"{'Improvement':<20} {tuned_r2 - base_r2:>+8.4f}  {base_mae - tuned_mae:>+8.4f} yrs")
print(f"{'':─<48}")

# ── Feature importance ────────────────────────────────────────────────────────
importances = dict(zip(FEATURES, best_model.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("\nFeature importances (tuned):")
for feat, imp in sorted_imp:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# ── Save ─────────────────────────────────────────────────────────────────────
joblib.dump(best_model, f"{MODEL_DIR}/phenoage_model.pkl")

with open(f"{MODEL_DIR}/model_meta.json") as f:
    meta = json.load(f)

meta["best_model"]    = "XGBoost (tuned)"
meta["best_params"]   = best_params
meta["results"]["XGBoost (tuned)"] = {
    "R2":    round(tuned_r2, 4),
    "MAE":   round(tuned_mae, 4),
    "CV_R2": round(best_cv_r2, 4),
}

with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → {MODEL_DIR}/phenoage_model.pkl  (tuned model)")
print(f"Saved → {MODEL_DIR}/model_meta.json")
