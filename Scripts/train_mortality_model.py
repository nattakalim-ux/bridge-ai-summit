"""
Phase 6: Train Mortality Risk Classifier
Input: 9 blood markers
Output: Probability of dying within the follow-up window (0-1)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
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

# Suppress hyperparameter tuning internal logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print(" PHASE 6: MORTALITY RISK MODEL TRAINING WITH CALIBRATION")
print("=" * 70)

# ── 1. Path Configuration (Absolute Root Alignment) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# Create output model folder registry if non-existent
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")
PLOT_PATH = os.path.join(MODEL_DIR, "mortality_model_eval.png")

BASE_MODEL_PATH = os.path.join(MODEL_DIR, "mortality_base_model.pkl")
IR_MODEL_PATH = os.path.join(MODEL_DIR, "mortality_ir.pkl")
META_PATH = os.path.join(MODEL_DIR, "mortality_meta.json")

# Verify asset existence prior to script execution
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Target input file not found: {DATA_PATH}")
    print("[STOP] Mortality training pipeline halted due to missing assets.")
    sys.exit(1)

# ── 2. Data Loading & Feature Target Extraction ──────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"[SUCCESS] Dataset loaded successfully: {len(df):,} rows")

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]
TARGET = "died_followup"  # Utilizing the validated target constructed in Phase 1

X = df[FEATURES].copy()
y = df[TARGET]

# Apply natural log transformation to skewed continuous lab distributions
X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])

# Split strategy: Hold out 20% for testing, use remaining for train and calibration
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

print(f"  Partition Split -> Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")
print(f"  Baseline Cohort Mortality Rate: {y.mean() * 100:.2f}% | Scale Position Weight: {scale_pos:.2f}\n")

# ── 3. Optuna Hyperparameter Optimization ────────────────────────────────────
N_TRIALS = 80
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
        "random_state":      42,
        "verbosity":         0,
        "n_jobs":            -1,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

print(f"Executing {N_TRIALS} Bayesian optimization trials via Optuna...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_params.update({
    "scale_pos_weight": scale_pos,
    "eval_metric":       "auc",
    "random_state":      42,
    "verbosity":         0,
    "n_jobs":            -1
})

print(f"\nOptimization Optimal Cross-Validation AUC: {study.best_value:.4f}")

# ── 4. Core Classifier Training & Isotonic Post-Calibration ──────────────────
base_model = XGBClassifier(**best_params)
base_model.fit(X_train, y_train)

# Fit Isotonic Regression mapping over raw training outputs using calibration set
raw_cal = base_model.predict_proba(X_cal)[:, 1]
ir = IsotonicRegression(out_of_bounds="clip")
ir.fit(raw_cal, y_cal)

class CalibratedWrapper:
    def predict_proba(self, X_matrix):
        raw_probabilities = base_model.predict_proba(X_matrix)[:, 1]
        calibrated_probabilities = ir.predict(raw_probabilities)
        return np.column_stack([1 - calibrated_probabilities, calibrated_probabilities])

calibrated = CalibratedWrapper()

def eval_model(label, predictor, X_mat, y_true):
    prob = predictor.predict_proba(X_mat)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "name":  label,
        "prob":  prob,
        "pred":  pred,
        "auc":   roc_auc_score(y_true, prob),
        "ap":    average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
    }

res_base = eval_model("Base XGBoost", base_model, X_test, y_test)
res_cal = eval_model("+ Isotonic Calib.", calibrated, X_test, y_test)

print("\n" + "─" * 55)
print(f"{'Model Framework':<22} {'AUC Score':>10}  {'Avg Prec':>10}  {'Brier Loss':>10}")
print("─" * 55)
for r in [res_base, res_cal]:
    print(f"  {r['name']:<20} {r['auc']:>10.4f}  {r['ap']:>10.4f}  {r['brier']:>10.4f}")
print("─" * 55)

print("\nClassification Matrix Report (Calibrated Framework, Threshold=0.50):")
print(classification_report(y_test, res_cal["pred"], target_names=["alive", "dead"]))

# Choose model architecture variant maximizing reliability indices
best_result = res_cal if res_cal["brier"] < res_base["brier"] else res_base
auc, ap, brier = best_result["auc"], best_result["ap"], best_result["brier"]
y_prob = best_result["prob"]

# ── 5. Feature Importance Calculations ──────────────────────────────────────
importances = dict(zip(FEATURES, base_model.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
print("Calculated Model Feature Importance Metrics:")
for feat, imp in sorted_imp:
    bar = "█" * int(imp * 40)
    print(f"  {feat:<15} {imp:.4f}  {bar}")

# ── 6. Graphics Generation & Asset Storage ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

RocCurveDisplay.from_predictions(y_test, res_base["prob"], ax=axes[0], name=f"Base (AUC={res_base['auc']:.3f})")
RocCurveDisplay.from_predictions(y_test, res_cal["prob"], ax=axes[0], name=f"Calibrated (AUC={res_cal['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_title("ROC Curve - Follow-Up Mortality")

for r, marker in [(res_base, "o--"), (res_cal, "s-")]:
    pt, pp = calibration_curve(y_test, r["prob"], n_bins=5, strategy="uniform")
    axes[1].plot(pp, pt, marker, label=f"{r['name']} (Brier={r['brier']:.4f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction of positives")
axes[1].set_title("Calibration Curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f"\n[EXPORT] Evaluation curves saved to: {PLOT_PATH}")

# ── 7. Risk Stratification Evaluation ────────────────────────────────────────
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
print(f"\nRisk Tier Stratification Summary Table:\n{summary.to_string()}")

# ── 8. Serialization ─────────────────────────────────────────────────────────
joblib.dump(base_model, BASE_MODEL_PATH)
joblib.dump(ir, IR_MODEL_PATH)

meta = {
    "model":           "XGBoost + Isotonic Calibration",
    "target":          "died within follow up window",
    "features":        FEATURES,
    "log_features":    LOG_FEATURES,
    "risk_thresholds": {"low_to_moderate": thresholds[0], "moderate_to_high": thresholds[1]},
    "metrics": {
        "AUC_ROC":          round(auc, 4),
        "Avg_Prec":         round(ap, 4),
        "Brier":            round(brier, 4),
        "CV_AUC":           round(study.best_value, 4),
        "base_Brier":       round(res_base["brier"], 4),
        "calibrated_Brier": round(res_cal["brier"], 4),
    },
    "best_params":     best_params,
    "train_size":      len(X_train),
    "cal_size":        len(X_cal),
    "test_size":       len(X_test),
    "mortality_rate":  round(float(y.mean()), 4),
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n[EXPORT] Binary weight models stored inside: {MODEL_DIR}")
print(f"[EXPORT] Evaluation telemetry log written to: {META_PATH}")
print("=" * 70)