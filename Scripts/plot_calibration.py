"""
Phase 7: Calibration Curve Re-plotting and Evaluation
Regenerates evaluation plots from saved models using diverse binning strategies.
Consumes cleaned datasets derived from Phase 3 processing.
"""

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
import os
import sys

print("=" * 70)
print(" PHASE 7: CALIBRATION AND RELIABILITY CURVE VISUALIZATION")
print("=" * 70)

# ── 1. Path Configuration (Absolute Root Alignment) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

DATA_PATH = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")

# Verify source dataset availability prior to script execution
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Target input file not found: {DATA_PATH}")
    print("[STOP] Plotting pipeline halted due to missing upstream assets.")
    sys.exit(1)

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]
TARGET = "died_followup"  # Aligned with Phase 1 and Phase 6 schema designations

# ── 2. Data Loading & Reproducible Partition Splitting ───────────────────────
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].copy()
y = df[TARGET]

X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])

# Execute identical stratified replication split matching train_mortality_model.py
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[SUCCESS] Test validation data verified: {len(X_test):,} cases")

# ── 3. Model Weight Restoration ──────────────────────────────────────────────
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "mortality_base_model.pkl")
IR_MODEL_PATH = os.path.join(MODEL_DIR, "mortality_ir.pkl")

if not os.path.exists(BASE_MODEL_PATH) or not os.path.exists(IR_MODEL_PATH):
    print("[ERROR] Serialized binary model weights missing inside Models registry.")
    print("[STOP] Please ensure Phase 6 (train_mortality_model.py) is executed beforehand.")
    sys.exit(1)

base_model = joblib.load(BASE_MODEL_PATH)
ir = joblib.load(IR_MODEL_PATH)
print("[INFO] Serialized classifier weights and Isotonic mappers successfully loaded.")

# ── 4. Telemetry Inference Calculations ──────────────────────────────────────
prob_base = base_model.predict_proba(X_test)[:, 1]
prob_cal = ir.predict(prob_base)

# ── 5. Side-by-Side Binning Multi-Plot Generation ───────────────────────────
bins_to_test = [5, 7, 10]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, n_bins in enumerate(bins_to_test):
    ax = axes[i]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect Calibration")
    
    # Base XGBoost Calculation
    pt_b, pp_b = calibration_curve(y_test, prob_base, n_bins=n_bins, strategy="uniform")
    brier_b = brier_score_loss(y_test, prob_base)
    ax.plot(pp_b, pt_b, "s--", color="steelblue", label=f"Base (Brier={brier_b:.4f})")
    
    # Calibrated XGBoost Calculation
    pt_c, pp_c = calibration_curve(y_test, prob_cal, n_bins=n_bins, strategy="uniform")
    brier_c = brier_score_loss(y_test, prob_cal)
    ax.plot(pp_c, pt_c, "o-", color="tomato", label=f"Calibrated (Brier={brier_c:.4f})")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Evaluation (Bins = {n_bins})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
OUT_GRID_PATH = os.path.join(MODEL_DIR, "mortality_calibration_comparison.png")
plt.savefig(OUT_GRID_PATH, dpi=150)
print(f"[EXPORT] Multi-bin matrix visualization exported to: {OUT_GRID_PATH}")

# ── 6. Comprehensive Standard Output Plot Generation ────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Plot ROC Area Under Curve metrics
RocCurveDisplay.from_predictions(y_test, prob_base, ax=axes2[0], name=f"Base (AUC={roc_auc_score(y_test, prob_base):.3f})")
axes2[0].lines[-1].set(color="steelblue", linestyle="--")

RocCurveDisplay.from_predictions(y_test, prob_cal, ax=axes2[0], name=f"Calibrated (AUC={roc_auc_score(y_test, prob_cal):.3f})")
axes2[0].lines[-1].set(color="tomato")

axes2[0].plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
axes2[0].set_title("ROC Curve: Follow-Up Mortality Assessment")
axes2[0].grid(True, alpha=0.2)
axes2[0].legend()

# Standardized Reliability Curve (N_BINS=5 Default)
axes2[1].plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
for label, prob, style in [("Base Model", prob_base, ("steelblue", "s--")),
                             ("Calibrated Framework", prob_cal, ("tomato", "o-"))]:
    pt, pp = calibration_curve(y_test, prob, n_bins=5, strategy="uniform")
    br = brier_score_loss(y_test, prob)
    axes2[1].plot(pp, pt, style[1], color=style[0], label=f"{label} (Brier={br:.4f})")

axes2[1].set_xlabel("Mean Predicted Probability")
axes2[1].set_ylabel("Fraction of Positives")
axes2[1].set_title("Reliability Calibration Curve (Bins = 5)")
axes2[1].legend(loc="lower right")
axes2[1].grid(True, alpha=0.2)

plt.tight_layout()
OUT_EVAL_PATH = os.path.join(MODEL_DIR, "mortality_model_eval.png")
plt.savefig(OUT_EVAL_PATH, dpi=150)
print(f"[EXPORT] Standardized framework evaluation layout saved to: {OUT_EVAL_PATH}")
print("=" * 70)