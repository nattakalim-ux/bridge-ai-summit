"""
Train mortality risk classifier on NHANES 2007-2010.
Input:  9 blood markers
Output: probability of dying within 10 years (0–1)
"""

import pandas as pd
import numpy as np
import joblib
import json
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, classification_report,
                             RocCurveDisplay)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_PATH = "/Users/ming/Desktop/MDCU/Data/mortality_nhanes_complete.csv"
MODEL_DIR = "/Users/ming/Desktop/MDCU/Model"

FEATURES     = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
                "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]
TARGET       = "died_10yr"
N_TRIALS     = 80

# ── 1. Load & preprocess ─────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df[TARGET]

X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])

# Hold out test set, then split remaining into train_fit + calibration
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos = n_neg / n_pos
print(f"Train: {len(X_train):,}  |  Cal: {len(X_cal):,}  |  Test: {len(X_test):,}")
print(f"Mortality rate: {y.mean()*100:.1f}%  |  scale_pos_weight: {scale_pos:.1f}\n")

# ── 2. Optuna tuning ─────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth":         trial.suggest_int("max_depth", 3, 8),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight":  scale_pos,
        "eval_metric":       "auc",
        "random_state": 42, "verbosity": 0, "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print(f"Running {N_TRIALS} Optuna trials...")
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({"scale_pos_weight": scale_pos,
                    "eval_metric": "auc",
                    "random_state": 42, "verbosity": 0, "n_jobs": -1})

print(f"\nBest CV AUC: {study.best_value:.4f}")

# ── 3. Train base model, then calibrate with isotonic on X_cal ───────────────
base_model = XGBClassifier(**best_params)
base_model.fit(X_train, y_train)

# Calibrate: fit IsotonicRegression on raw probabilities from X_cal
raw_cal = base_model.predict_proba(X_cal)[:, 1]
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(raw_cal, y_cal)

# Wrapper so eval_model works uniformly
class CalibratedWrapper:
    def predict_proba(self, X):
        raw = base_model.predict_proba(X)[:, 1]
        cal = ir.predict(raw)
        return np.column_stack([1 - cal, cal])

calibrated = CalibratedWrapper()

# Evaluate both on held-out test set
def eval_model(name, predictor, X, y):
    prob = predictor.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "name":   name,
        "prob":   prob,
        "pred":   pred,
        "auc":    roc_auc_score(y, prob),
        "ap":     average_precision_score(y, prob),
        "brier":  brier_score_loss(y, prob),
    }

res_base = eval_model("Base XGBoost",      base_model,  X_test, y_test)
res_cal  = eval_model("+ Isotonic Calib.", calibrated,  X_test, y_test)

print(f"\n{'─'*52}")
print(f"{'':22} {'AUC':>7}  {'AP':>7}  {'Brier':>7}")
print(f"{'─'*52}")
for r in [res_base, res_cal]:
    print(f"  {r['name']:<20} {r['auc']:>7.4f}  {r['ap']:>7.4f}  {r['brier']:>7.4f}")
print(f"  {'Improvement':<20} "
      f"{res_cal['auc']-res_base['auc']:>+7.4f}  "
      f"{res_cal['ap']-res_base['ap']:>+7.4f}  "
      f"{res_base['brier']-res_cal['brier']:>+7.4f}")
print(f"{'─'*52}")

print("\nClassification report (calibrated, threshold=0.5):")
print(classification_report(y_test, res_cal["pred"], target_names=["alive", "dead"]))

# Pick the better model for saving
best_model  = calibrated if res_cal["brier"] < res_base["brier"] else base_model
best_result = res_cal    if res_cal["brier"] < res_base["brier"] else res_base
auc, ap, brier = best_result["auc"], best_result["ap"], best_result["brier"]
y_prob = best_result["prob"]

# ── 4. Feature importance (from base model) ───────────────────────────────────
importances = dict(zip(FEATURES, base_model.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("Feature importances:")
for feat, imp in sorted_imp:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# ── 5. Calibration plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve — both models
RocCurveDisplay.from_predictions(y_test, res_base["prob"], ax=axes[0],
                                  name=f"Base (AUC={res_base['auc']:.3f})")
RocCurveDisplay.from_predictions(y_test, res_cal["prob"],  ax=axes[0],
                                  name=f"Calibrated (AUC={res_cal['auc']:.3f})")
axes[0].plot([0,1],[0,1], 'k--', alpha=0.4)
axes[0].set_title("ROC Curve — 10-Year Mortality")

# Calibration curve — both models
for r, marker in [(res_base, "o--"), (res_cal, "s-")]:
    pt, pp = calibration_curve(y_test, r["prob"], n_bins=5, strategy="uniform")
    axes[1].plot(pp, pt, marker, label=f"{r['name']} (Brier={r['brier']:.4f})")
axes[1].plot([0,1],[0,1], 'k--', alpha=0.4, label="Perfect")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction of positives")
axes[1].set_title("Calibration Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/mortality_model_eval.png", dpi=150)
print(f"\nPlot saved → {MODEL_DIR}/mortality_model_eval.png")

# ── 6. Risk interpretation ───────────────────────────────────────────────────
# Bucket predicted risk into 3 tiers
thresholds = [0.10, 0.25]
def risk_label(p):
    if p < thresholds[0]: return "Low"
    if p < thresholds[1]: return "Moderate"
    return "High"

risk_df = pd.DataFrame({"prob": y_prob, "actual": y_test.values})
risk_df["tier"] = risk_df["prob"].apply(risk_label)
summary = risk_df.groupby("tier").agg(
    n=("prob", "count"),
    actual_mortality=("actual", "mean"),
    mean_risk=("prob", "mean")
).round(3)
print(f"\nRisk tier summary:\n{summary.to_string()}")

# ── 7. Save artifacts ────────────────────────────────────────────────────────
# Save base model and isotonic regression separately (avoids pickle class-ref issues)
joblib.dump(base_model, f"{MODEL_DIR}/mortality_base_model.pkl")
joblib.dump(ir,         f"{MODEL_DIR}/mortality_ir.pkl")

meta = {
    "model":         "XGBoost + Isotonic Calibration",
    "target":        "died within 10 years",
    "features":      FEATURES,
    "log_features":  LOG_FEATURES,
    "risk_thresholds": {"low_to_moderate": 0.10, "moderate_to_high": 0.25},
    "metrics": {
        "AUC_ROC":         round(auc, 4),
        "Avg_Prec":        round(ap, 4),
        "Brier":           round(brier, 4),
        "CV_AUC":          round(study.best_value, 4),
        "base_Brier":      round(res_base["brier"], 4),
        "calibrated_Brier": round(res_cal["brier"], 4),
    },
    "best_params":   best_params,
    "train_size":    len(X_train),
    "cal_size":      len(X_cal),
    "test_size":     len(X_test),
    "mortality_rate": round(float(y.mean()), 4),
}
with open(f"{MODEL_DIR}/mortality_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → {MODEL_DIR}/mortality_base_model.pkl  +  mortality_ir.pkl")
print(f"Saved → {MODEL_DIR}/mortality_meta.json")
