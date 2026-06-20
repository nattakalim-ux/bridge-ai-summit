"""
Regenerate evaluation plots from saved models — no retraining needed.
Tries n_bins = 5, 7, 10 side-by-side so you can pick the best look.
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

MODEL_DIR = "/Users/ming/Desktop/MDCU/Model"
DATA_PATH = "/Users/ming/Desktop/MDCU/Data/mortality_nhanes_complete.csv"

FEATURES     = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
                "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = ["crp_mgl", "glucose", "wbc", "alp"]

# ── Reload data with same splits ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURES].copy()
y  = df["died_10yr"]
X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)

# ── Load saved models ─────────────────────────────────────────────────────────
base_model = joblib.load(f"{MODEL_DIR}/mortality_base_model.pkl")
ir         = joblib.load(f"{MODEL_DIR}/mortality_ir.pkl")

prob_base = base_model.predict_proba(X_test)[:, 1]
prob_cal  = ir.predict(prob_base)

# ── Figure: ROC + calibration curves at 3 different n_bins ───────────────────
fig = plt.figure(figsize=(16, 5))
fig.suptitle("Mortality Model Evaluation", fontsize=13, fontweight="bold")

# ── Left: ROC curve ───────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(1, 4, 1)
RocCurveDisplay.from_predictions(
    y_test, prob_base, ax=ax_roc,
    name=f"Base  (AUC={roc_auc_score(y_test, prob_base):.3f})")
ax_roc.lines[-1].set(color="steelblue", linestyle="--")
RocCurveDisplay.from_predictions(
    y_test, prob_cal, ax=ax_roc,
    name=f"Calib (AUC={roc_auc_score(y_test, prob_cal):.3f})")
ax_roc.lines[-1].set(color="tomato")
ax_roc.plot([0,1],[0,1], "k--", alpha=0.3, linewidth=1)
ax_roc.set_title("ROC Curve")
ax_roc.legend(fontsize=8)

# ── Right: calibration curves at 3 bin settings ───────────────────────────────
bin_options = [5, 7, 10]
colors = {"Base": "steelblue", "Calib": "tomato"}

for i, n_bins in enumerate(bin_options):
    ax = fig.add_subplot(1, 4, i + 2)
    ax.plot([0,1],[0,1], "k--", alpha=0.3, linewidth=1, label="Perfect")

    for label, prob in [("Base", prob_base), ("Calib", prob_cal)]:
        pt, pp = calibration_curve(y_test, prob, n_bins=n_bins, strategy="uniform")
        brier  = brier_score_loss(y_test, prob)
        ax.plot(pp, pt, "o-" if label == "Calib" else "s--",
                color=colors[label], linewidth=2, markersize=5,
                label=f"{label} (Brier={brier:.4f})")

    ax.set_title(f"Calibration  n_bins={n_bins}")
    ax.set_xlabel("Mean predicted prob")
    if i == 0:
        ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

plt.tight_layout()
out = f"{MODEL_DIR}/mortality_calibration_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")

# ── Also update the main eval plot with n_bins=5 ─────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

RocCurveDisplay.from_predictions(y_test, prob_base, ax=axes[0],
    name=f"Base  (AUC={roc_auc_score(y_test, prob_base):.3f})")
axes[0].lines[-1].set(color="steelblue", linestyle="--")
RocCurveDisplay.from_predictions(y_test, prob_cal, ax=axes[0],
    name=f"Calibrated (AUC={roc_auc_score(y_test, prob_cal):.3f})")
axes[0].lines[-1].set(color="tomato")
axes[0].plot([0,1],[0,1], "k--", alpha=0.3, linewidth=1)
axes[0].set_title("ROC Curve — 10-Year Mortality")
axes[0].legend()

for label, prob, style in [("Base", prob_base, ("steelblue","s--")),
                             ("Calibrated", prob_cal, ("tomato","o-"))]:
    pt, pp = calibration_curve(y_test, prob, n_bins=5, strategy="uniform")
    brier  = brier_score_loss(y_test, prob)
    axes[1].plot(pp, pt, style[1], color=style[0], linewidth=2,
                 markersize=6, label=f"{label} (Brier={brier:.4f})")

axes[1].plot([0,1],[0,1], "k--", alpha=0.3, linewidth=1, label="Perfect")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction of positives")
axes[1].set_title("Calibration Curve (n_bins=5)")
axes[1].legend()
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)

plt.tight_layout()
out2 = f"{MODEL_DIR}/mortality_model_eval.png"
plt.savefig(out2, dpi=150)
print(f"Updated → {out2}")
