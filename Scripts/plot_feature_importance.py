"""
Generate two publication-quality PNGs:
  1. Model/feature_importance.png  — feature importance for both models
  2. Model/prediction_scatter.png  — predicted vs actual for PhenoAge +
                                     predicted probability by outcome for Mortality
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

MODEL_DIR = "/Users/ming/Desktop/MDCU/Model"
DATA_DIR  = "/Users/ming/Desktop/MDCU/Data"

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]
LOG_FEATURES = {"crp_mgl", "glucose", "wbc", "alp"}

FEATURE_LABELS = {
    "albumin":    "Albumin",
    "creatinine": "Creatinine",
    "glucose":    "Glucose",
    "alp":        "ALP",
    "crp_mgl":   "CRP",
    "mcv":        "MCV",
    "rdw":        "RDW",
    "wbc":        "WBC",
    "lymph_pct":  "Lymphocyte %",
}

GROUP_COLORS = {"young": "#4C9BE8", "middle": "#F4A92B", "old": "#E05C5C"}

# ── Load models & metadata ────────────────────────────────────────────────────
mort_base  = joblib.load(f"{MODEL_DIR}/mortality_base_model.pkl")
mort_ir    = joblib.load(f"{MODEL_DIR}/mortality_ir.pkl")

with open(f"{MODEL_DIR}/mortality_meta.json") as f:
    mort_meta = json.load(f)
with open(f"{MODEL_DIR}/model_meta.json") as f:
    bio_meta = json.load(f)

bio_models = {g: joblib.load(f"{MODEL_DIR}/phenoage_model_{g}.pkl")
              for g in ["young", "middle", "old"]}

AGE_GROUPS = {"young": (0, 45), "middle": (45, 65), "old": (65, 999)}

# ── Load & preprocess data ────────────────────────────────────────────────────
df_mort = pd.read_csv(f"{DATA_DIR}/mortality_nhanes_complete.csv")
df_bio  = pd.read_csv(f"{DATA_DIR}/phenoage_nhanes_complete.csv")

def preprocess(df):
    X = df[FEATURES].copy()
    for f in LOG_FEATURES:
        X[f] = np.log1p(X[f])
    return X

X_mort = preprocess(df_mort)
X_bio  = preprocess(df_bio)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
mort_imp = dict(zip(FEATURES, mort_base.feature_importances_))

# Aggregate bio-age importance across 3 group models (weighted by n_train)
bio_imp_agg = np.zeros(len(FEATURES))
total_n = sum(bio_meta["group_results"][g]["n_train"] for g in AGE_GROUPS)
for g in AGE_GROUPS:
    w = bio_meta["group_results"][g]["n_train"] / total_n
    bio_imp_agg += w * bio_models[g].feature_importances_
bio_imp = dict(zip(FEATURES, bio_imp_agg))

labels = [FEATURE_LABELS[f] for f in FEATURES]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Feature Importance", fontsize=15, fontweight="bold", y=1.01)

for ax, imp_dict, title, color in [
    (axes[0], mort_imp, "Mortality Risk Model\n(XGBoost — 10-yr death)", "#E05C5C"),
    (axes[1], bio_imp,  "PhenoAge Model\n(Stratified Ensemble — weighted avg)", "#4C9BE8"),
]:
    vals   = [imp_dict[f] for f in FEATURES]
    order  = np.argsort(vals)
    y_pos  = np.arange(len(FEATURES))

    bars = ax.barh(y_pos, [vals[i] for i in order],
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[i] for i in order], fontsize=10)
    ax.set_xlabel("Feature Importance (F-score)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels
    for bar, idx in zip(bars, order):
        v = vals[idx]
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8.5, color="#333333")

    ax.set_xlim(0, max(vals) * 1.22)

plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/feature_importance.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved → feature_importance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Prediction Scatter
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Panel A: PhenoAge predicted vs actual ─────────────────────────────────────
ax_bio = fig.add_subplot(gs[0])

y_actual = df_bio["phenoage"].values
y_pred   = np.zeros(len(df_bio))
group_col = np.full(len(df_bio), "", dtype=object)

for g, (lo, hi) in AGE_GROUPS.items():
    mask = (df_bio["age"] >= lo) & (df_bio["age"] < hi)
    if mask.sum() > 0:
        y_pred[mask]   = bio_models[g].predict(X_bio[mask])
        group_col[mask] = g

r2  = r2_score(y_actual, y_pred)
mae = mean_absolute_error(y_actual, y_pred)

for g, (lo, hi) in AGE_GROUPS.items():
    mask = group_col == g
    ax_bio.scatter(y_actual[mask], y_pred[mask],
                   color=GROUP_COLORS[g], alpha=0.25, s=6, label=g.capitalize(),
                   rasterized=True)

lims = [min(y_actual.min(), y_pred.min()) - 2, max(y_actual.max(), y_pred.max()) + 2]
ax_bio.plot(lims, lims, "k--", linewidth=1.2, alpha=0.6, label="Perfect (y=x)")
ax_bio.set_xlim(lims); ax_bio.set_ylim(lims)
ax_bio.set_xlabel("Actual PhenoAge (years)", fontsize=10)
ax_bio.set_ylabel("Predicted PhenoAge (years)", fontsize=10)
ax_bio.set_title(f"PhenoAge — Predicted vs Actual\nR²={r2:.3f}  MAE={mae:.2f} yrs",
                 fontsize=11, fontweight="bold")
ax_bio.legend(fontsize=8.5, markerscale=2)
ax_bio.spines[["top", "right"]].set_visible(False)
ax_bio.text(0.04, 0.94, f"n = {len(df_bio):,}", transform=ax_bio.transAxes,
            fontsize=8, color="#555555")

# ── Panel B: Mortality predicted probability by outcome ───────────────────────
ax_mort = fig.add_subplot(gs[1])

y_true  = df_mort["died_10yr"].values
raw_prob = mort_base.predict_proba(X_mort)[:, 1]
cal_prob = mort_ir.predict(raw_prob)

alive_prob = cal_prob[y_true == 0]
dead_prob  = cal_prob[y_true == 1]

# Jitter + scatter
rng = np.random.default_rng(42)
for probs, x_center, color, label in [
    (alive_prob, 0, "#4C9BE8", f"Survived  (n={len(alive_prob):,})"),
    (dead_prob,  1, "#E05C5C", f"Died  (n={len(dead_prob):,})"),
]:
    jitter = rng.uniform(-0.18, 0.18, len(probs))
    ax_mort.scatter(x_center + jitter, probs,
                    color=color, alpha=0.18, s=5, rasterized=True)
    ax_mort.boxplot(probs, positions=[x_center], widths=0.25,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.55),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="", alpha=0))
    ax_mort.plot([], [], color=color, linewidth=6, alpha=0.7, label=label)

# Risk threshold lines
for thresh, lbl in [(0.10, "Low / Moderate"), (0.25, "Moderate / High")]:
    ax_mort.axhline(thresh, color="#888888", linestyle=":", linewidth=1.2)
    ax_mort.text(1.22, thresh + 0.005, lbl, fontsize=7.5, color="#666666", va="bottom")

ax_mort.set_xticks([0, 1])
ax_mort.set_xticklabels(["Survived", "Died"], fontsize=10)
ax_mort.set_ylabel("Predicted 10-yr Mortality Probability", fontsize=10)
auc = mort_meta["metrics"]["AUC_ROC"]
ax_mort.set_title(f"Mortality Model — Predicted Probability by Outcome\nAUC={auc:.3f}",
                  fontsize=11, fontweight="bold")
ax_mort.set_ylim(-0.03, 1.05)
ax_mort.set_xlim(-0.5, 1.7)
ax_mort.legend(fontsize=8.5, loc="upper left")
ax_mort.spines[["top", "right"]].set_visible(False)

fig.suptitle("Model Prediction Quality", fontsize=14, fontweight="bold", y=1.02)
plt.savefig(f"{MODEL_DIR}/prediction_scatter.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved → prediction_scatter.png")
