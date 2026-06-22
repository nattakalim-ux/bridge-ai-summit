"""
Phase 8: Feature Importance Comparison and Distribution Plots
Generates publication-quality charts contrasting PhenoAge and Mortality prediction drivers.
Consumes cleaned datasets derived from Phase 3 processing.
"""

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
import sys

print("=" * 70)
print(" PHASE 8: FEATURE IMPORTANCE AND METRIC VISUALIZATION")
print("=" * 70)

# ── 1. Path Configuration (Absolute Root Alignment) ─────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

DATA_PATH = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = {"crp_mgl", "glucose", "wbc", "alp"}

FEATURE_LABELS = {
    "albumin":    "Albumin",
    "creatinine": "Creatinine",
    "glucose":    "Glucose",
    "alp":        "ALP",
    "crp_mgl":    "CRP",
    "mcv":        "MCV",
    "rdw":        "RDW",
    "wbc":        "WBC",
    "lymph_pct":  "Lymphocyte %",
}

# Verify source dataset availability prior to script execution
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Target input file not found: {DATA_PATH}")
    print("[STOP] Plotting pipeline halted due to missing upstream assets.")
    sys.exit(1)

# ── 2. Load Models & Metadata ────────────────────────────────────────────────
MORT_MODEL_PATH = os.path.join(MODEL_DIR, "mortality_base_model.pkl")
MORT_META_PATH = os.path.join(MODEL_DIR, "mortality_meta.json")
PHENO_MODEL_PATH = os.path.join(MODEL_DIR, "phenoage_model.pkl")
PHENO_META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

if not os.path.exists(MORT_MODEL_PATH) or not os.path.exists(MORT_META_PATH):
    print(f"[ERROR] Core mortality assets missing inside: {MODEL_DIR}")
    print("[STOP] Please ensure Phase 6 has run successfully before executing Phase 8.")
    sys.exit(1)

mort_base = joblib.load(MORT_MODEL_PATH)
with open(MORT_META_PATH, "r") as f:
    mort_meta = json.load(f)

# Optional handling for PhenoAge regressor assets
pheno_available = os.path.exists(PHENO_MODEL_PATH) and os.path.exists(PHENO_META_PATH)
if pheno_available:
    pheno_base = joblib.load(PHENO_MODEL_PATH)
    with open(PHENO_META_PATH, "r") as f:
        pheno_meta = json.load(f)
    print("[INFO] Both Mortality and PhenoAge regression assets loaded successfully.")
else:
    print("[WARNING] PhenoAge regression assets missing. Defaulting focus to Mortality arrays.")

# ── 3. Data Processing & Replication Splits ─────────────────────────────────
df = pd.read_csv(DATA_PATH)
X = df[FEATURES].copy()
y_mort = df["died_followup"]

X[list(LOG_FEATURES)] = np.log1p(X[list(LOG_FEATURES)])

# Replicate identical partition splits mapping back to model states
_, X_test, _, y_test_mort = train_test_split(
    X, y_mort, test_size=0.2, random_state=42, stratify=y_mort
)
prob_mort = mort_base.predict_proba(X_test)[:, 1]

# ── 4. Chart 1: Feature Importance Contrast Plot ────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 6))

mort_imp = mort_base.feature_importances_
y_pos = np.arange(len(FEATURES))

if pheno_available:
    pheno_imp = pheno_base.feature_importances_
    width = 0.35
    ax.barh(y_pos - width/2, mort_imp, width, label="Mortality Risk Classifier", color="tomato", alpha=0.85)
    ax.barh(y_pos + width/2, pheno_imp, width, label="PhenoAge Regressor", color="steelblue", alpha=0.85)
else:
    ax.barh(y_pos, mort_imp, 0.5, label="Mortality Risk Classifier", color="tomato", alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels([FEATURE_LABELS[f] for f in FEATURES], fontsize=11)
ax.invert_yaxis()  # Top-down feature layout
ax.set_xlabel("Relative Feature Importance (Gini Multi-split Fractional Weight)", fontsize=11)
ax.set_title("Biomarker Importance Comparison Matrix", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
ax.grid(True, axis="x", alpha=0.2)

plt.tight_layout()
OUT_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance_comparison.png")
plt.savefig(OUT_IMPORTANCE_PATH, dpi=150)
print(f"[EXPORT] Feature importance comparison layout saved to: {OUT_IMPORTANCE_PATH}")

# ── 5. Chart 2: Outcome Probability Distribution Matrix ─────────────────────
fig2, ax_mort = plt.subplots(figsize=(6, 5))

probs = [prob_mort[y_test_mort == 0], prob_mort[y_test_mort == 1]]
colors = ["#4682B4", "#FF6347"]
labels = ["Survived", "Died"]

for x_center, (prob, color, label) in enumerate(zip(probs, colors, labels)):
    # Jitter scatter overlay to expose distribution density patterns
    jitter = np.random.normal(x_center, 0.04, size=len(prob))
    ax_mort.scatter(jitter, prob, color=color, alpha=0.15, s=6, rasterized=True)
    
    # Standardized clinical boxplot overlay
    ax_mort.boxplot(prob, positions=[x_center], widths=0.25,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.55, edgecolor=color),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="", alpha=0))
    ax_mort.plot([], [], color=color, linewidth=6, alpha=0.7, label=label)

# Render customized decision and risk thresholds
risk_thresholds = [0.10, 0.25]
for thresh, lbl in [(risk_thresholds[0], "Low / Moderate"), (risk_thresholds[1], "Moderate / High")]:
    ax_mort.axhline(thresh, color="#888888", linestyle=":", linewidth=1.2)
    ax_mort.text(1.22, thresh + 0.005, lbl, fontsize=8, color="#666666", va="bottom")

ax_mort.set_xticks([0, 1])
ax_mort.set_xticklabels(["Survived", "Died"], fontsize=11)
ax_mort.set_ylabel("Predicted Mortality Probability", fontsize=11)
auc_val = mort_meta["metrics"]["AUC_ROC"]
ax_mort.set_title(f"Mortality Model Risk Distribution\n(Test Cohort AUC = {auc_val:.3f})", fontsize=12, fontweight="bold")
ax_mort.set_xlim(-0.5, 1.5)
ax_mort.set_ylim(-0.02, 1.02)
ax_mort.grid(True, axis="y", alpha=0.2)

plt.tight_layout()
OUT_DIST_PATH = os.path.join(MODEL_DIR, "prediction_distribution_analysis.png")
plt.savefig(OUT_DIST_PATH, dpi=150)
print(f"[EXPORT] Probability distribution scatter metrics saved to: {OUT_DIST_PATH}")
print("=" * 70)