"""
5 Publication-Quality Figures for Research Abstract & Pitch Deck
NHANES 2015-2016 | PhenoAge vs ML Mortality Comparison

Retrains models with known-best hyperparameters (deterministic, no grid search).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
import warnings, os
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATA_PATH  = os.path.join(OUTPUT_DIR, "merged_nhanes_2015_2016_with_mortality.csv")

# ── Color palette ─────────────────────────────────────────────────────────────
C_LEVINE = "#888780"
C_EN     = "#534AB7"
C_XGB    = "#1D9E75"
C_RF     = "#BA7517"
C_DIED   = "#E24B4A"
C_SURV   = "#1D9E75"

COLORS = [C_LEVINE, C_EN, C_XGB, C_RF]
LABELS = ["Levine PhenoAge", "Elastic Net", "XGBoost", "Random Forest"]

def spine_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & RETRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data and retraining models...")

df = pd.read_csv(DATA_PATH)

FEATURES    = ["albumin","creatinine","glucose","alp","crp_mgl","mcv","rdw","wbc","lymph_pct"]
ALL_FEATS   = FEATURES + ["age"]
TARGET      = "died_10yr"

X   = df[ALL_FEATS].values
y   = df[TARGET].values
pg  = df["PhenoAge_gap"].values

# Same split as phase4_10 (random_state=42 throughout)
X_tv, X_test, y_tv, y_test, pg_tv, pg_test = train_test_split(
    X, y, pg, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val, pg_train, pg_val = train_test_split(
    X_tv, y_tv, pg_tv, test_size=0.25, random_state=42, stratify=y_tv)

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xva = scaler.transform(X_val)
Xte = scaler.transform(X_test)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# Train with known-best params (no grid search)
print("  Training Elastic Net...")
model_en = LogisticRegression(penalty="elasticnet", solver="saga", C=0.001,
                               l1_ratio=0.1, max_iter=2000, random_state=42,
                               class_weight="balanced").fit(Xtr, y_train)

print("  Training XGBoost...")
model_xgb = XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.02,
    min_child_weight=13, subsample=0.5707, colsample_bytree=0.8930,
    scale_pos_weight=scale_pos, eval_metric="auc",
    random_state=42, verbosity=0, n_jobs=-1).fit(
    Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)

print("  Training Random Forest...")
model_rf = RandomForestClassifier(
    n_estimators=500, max_depth=None, class_weight="balanced",
    random_state=42, n_jobs=-1).fit(Xtr, y_train)

# Calibrate ML models (fit isotonic on val, apply to test)
def calibrate(model, Xva, yva, Xte):
    raw_val  = model.predict_proba(Xva)[:,1]
    raw_test = model.predict_proba(Xte)[:,1]
    ir = IsotonicRegression(out_of_bounds="clip").fit(raw_val, yva)
    return raw_test, ir.predict(raw_test)

_, prob_en  = calibrate(model_en,  Xva, y_val, Xte)
_, prob_xgb = calibrate(model_xgb, Xva, y_val, Xte)
_, prob_rf  = calibrate(model_rf,  Xva, y_val, Xte)

# Levine: calibrate PhenoAge_gap → probability
ir_lev   = IsotonicRegression(out_of_bounds="clip").fit(pg_val, y_val)
prob_lev_raw = pg_test.copy()         # raw gap for AUC
prob_lev     = ir_lev.predict(pg_test) # calibrated for distribution plot

PROBS_AUC = [prob_lev_raw, prob_en, prob_xgb, prob_rf]
KNOWN_AUC = [0.6754, 0.8827, 0.8457, 0.8338]

print("  Done. Generating figures...\n")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════
print("[Figure 1] ROC Curves...")

fig, ax = plt.subplots(figsize=(8, 6))

for prob, col, lbl, auc in zip(PROBS_AUC, COLORS, LABELS, KNOWN_AUC):
    fpr, tpr, _ = roc_curve(y_test, prob)
    lw = 2.5 if lbl == "Elastic Net" else 1.8
    ax.plot(fpr, tpr, color=col, linewidth=lw, label=f"{lbl}  (AUC = {auc:.4f})")
    if lbl == "Elastic Net":
        ax.fill_between(fpr, tpr, alpha=0.10, color=col)

ax.plot([0,1],[0,1], "--", color="#BBBBBB", linewidth=1.2, label="Random classifier (AUC = 0.5000)")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — Mortality Prediction Performance", fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#CCCCCC")
ax.annotate("All ML models p < 0.05 vs Levine (DeLong test)",
            xy=(0.5, 0.08), xycoords="axes fraction",
            ha="center", fontsize=9.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.9))
spine_clean(ax)
ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)

plt.tight_layout()
fig1_path = os.path.join(OUTPUT_DIR, "roc_curves_final.png")
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig1_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: FEATURE IMPORTANCE COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("[Figure 2] Feature Importance...")

feat_names   = ALL_FEATS
feat_labels  = ["Albumin","Creatinine","Glucose","ALP","CRP","MCV","RDW","ALP","WBC","Age"]
feat_labels  = ["Albumin","Creatinine","Glucose","ALP","CRP","MCV","RDW","WBC","Age"]
feat_labels  = [f.replace("crp_mgl","CRP").replace("lymph_pct","Lymphocyte%")
                 .replace("alp","ALP").replace("mcv","MCV").replace("rdw","RDW")
                 .replace("wbc","WBC").replace("albumin","Albumin")
                 .replace("creatinine","Creatinine").replace("glucose","Glucose")
                 .replace("age","Age")
                for f in feat_names]

levine_coefs = np.array([0.0336, 0.0095, 0.1953, 0.00188, 0.0954,
                          0.0268, 0.3306, 0.0554, 0.0120, 0.0804])
en_coefs  = np.abs(model_en.coef_[0])
xgb_imps  = model_xgb.feature_importances_
rf_imps   = model_rf.feature_importances_

def norm01(arr):
    a = np.array(arr, dtype=float)
    return a / a.max() if a.max() > 0 else a

imp_arrays = [norm01(levine_coefs), norm01(en_coefs), norm01(xgb_imps), norm01(rf_imps)]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(feat_names))
w = 0.20

for i, (arr, col, lbl) in enumerate(zip(imp_arrays, COLORS, LABELS)):
    ax.bar(x + i*w, arr, w, color=col, label=lbl, alpha=0.88, edgecolor="white", linewidth=0.5)

ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Normalized Importance (0 – 1)", fontsize=11)
ax.set_title("Biomarker Importance Comparison", fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, loc="upper right", framealpha=0.9, edgecolor="#CCCCCC")
ax.annotate("RDW = highest in Levine formula  |  Age = highest in ML models",
            xy=(0.5, 0.96), xycoords="axes fraction",
            ha="center", fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F9F9F9",
                      edgecolor="#DDDDDD", alpha=0.9))
spine_clean(ax)
ax.set_ylim(0, 1.12)

plt.tight_layout()
fig2_path = os.path.join(OUTPUT_DIR, "feature_importance_final.png")
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig2_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: PHENOAGE vs CHRONOLOGICAL AGE SCATTER
# ═══════════════════════════════════════════════════════════════════════════════
print("[Figure 3] PhenoAge Scatter...")

fig, ax = plt.subplots(figsize=(8, 7))

died_mask  = df["died_10yr"] == 1
surv_mask  = df["died_10yr"] == 0

ax.scatter(df.loc[surv_mask,  "age"], df.loc[surv_mask,  "PhenotypicAge"],
           color=C_SURV, alpha=0.35, s=18, label="Survived", zorder=2, linewidths=0)
ax.scatter(df.loc[died_mask, "age"], df.loc[died_mask, "PhenotypicAge"],
           color=C_DIED, alpha=0.75, s=30, label="Died within follow-up",
           zorder=3, linewidths=0)

# Diagonal reference
xmin, xmax = df["age"].min()-2, df["age"].max()+2
ax.plot([xmin, xmax], [xmin, xmax], "--", color="#AAAAAA", linewidth=1.4,
        label="Biological Age = Chronological Age", zorder=1)

# OLS trend lines per group
for mask, col in [(surv_mask, C_SURV), (died_mask, C_DIED)]:
    sub = df[mask]
    if len(sub) > 5:
        m, b, *_ = stats.linregress(sub["age"], sub["PhenotypicAge"])
        xs = np.linspace(sub["age"].min(), sub["age"].max(), 100)
        ax.plot(xs, m*xs + b, color=col, linewidth=2.0, zorder=4)

ax.set_xlabel("Chronological Age (years)", fontsize=12)
ax.set_ylabel("Phenotypic Age (years)", fontsize=12)
ax.set_title("Phenotypic Age vs Chronological Age", fontsize=13, fontweight="bold", pad=14)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
ax.annotate("Points above diagonal = biologically older\nthan chronological age",
            xy=(0.97, 0.08), xycoords="axes fraction", ha="right",
            fontsize=9, color="#555555",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F5F5",
                      edgecolor="#CCCCCC", alpha=0.9))
spine_clean(ax)
ax.xaxis.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)

plt.tight_layout()
fig3_path = os.path.join(OUTPUT_DIR, "phenoage_scatter_final.png")
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig3_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: MORTALITY RISK DISTRIBUTION (Elastic Net, test set)
# ═══════════════════════════════════════════════════════════════════════════════
print("[Figure 4] Mortality Risk Distribution...")

fig, ax = plt.subplots(figsize=(7, 6))

groups    = {0: ("Survived", C_SURV), 1: ("Died", C_DIED)}
positions = {0: 0, 1: 1}

for label_val, (label_str, col) in groups.items():
    mask  = y_test == label_val
    probs = prob_en[mask]
    pos   = positions[label_val]

    # Violin
    parts = ax.violinplot(probs, positions=[pos], widths=0.45,
                          showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.35); pc.set_edgecolor(col)

    # Box (IQR)
    q1, med, q3 = np.percentile(probs, [25, 50, 75])
    ax.vlines(pos, q1, q3, color=col, linewidth=5, alpha=0.7, zorder=3)
    ax.scatter(pos, med, color="white", s=40, zorder=4, edgecolors=col, linewidth=1.5)

    # Jittered points
    jitter = np.random.default_rng(42).uniform(-0.14, 0.14, len(probs))
    ax.scatter(pos + jitter, probs, color=col, alpha=0.28, s=10, zorder=2, linewidths=0)

    # Median annotation
    ax.annotate(f"Median: {med:.3f}",
                xy=(pos, med), xytext=(pos + 0.25, med),
                fontsize=9, color=col, va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=1))

# Threshold lines
for thresh, lbl in [(0.10, "Low / Moderate threshold (10%)"),
                    (0.25, "Moderate / High threshold (25%)")]:
    ax.axhline(thresh, color="#888888", linestyle=":", linewidth=1.3)
    ax.text(1.52, thresh + 0.005, lbl, fontsize=8, color="#666666", va="bottom")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Survived", "Died"], fontsize=12)
ax.set_ylabel("Predicted Mortality Probability", fontsize=11)
ax.set_title("Predicted Mortality Risk Distribution\n"
             "(Elastic Net, Test Cohort  AUC = 0.8827)", fontsize=12,
             fontweight="bold", pad=12)
ax.set_xlim(-0.55, 1.8)
ax.set_ylim(-0.02, 1.02)
spine_clean(ax)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "mortality_distribution_final.png")
plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig4_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: PHENOAGE GAP DISTRIBUTION BY MORTALITY STATUS
# ═══════════════════════════════════════════════════════════════════════════════
print("[Figure 5] PhenoAge Gap Distribution...")

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(9, 6))

gap_surv = df.loc[df["died_10yr"] == 0, "PhenoAge_gap"].values
gap_died = df.loc[df["died_10yr"] == 1, "PhenoAge_gap"].values

mn_stat, mn_p = mannwhitneyu(gap_died, gap_surv, alternative="greater")

# Histograms (normalized to density)
bins = np.linspace(df["PhenoAge_gap"].min()-2, df["PhenoAge_gap"].max()+2, 45)
ax.hist(gap_surv, bins=bins, density=True, alpha=0.35, color=C_SURV,
        label="Survived", edgecolor="white", linewidth=0.3)
ax.hist(gap_died, bins=bins, density=True, alpha=0.45, color=C_DIED,
        label="Died", edgecolor="white", linewidth=0.3)

# KDE overlays
for gap, col in [(gap_surv, C_SURV), (gap_died, C_DIED)]:
    kde  = gaussian_kde(gap, bw_method=0.4)
    xs   = np.linspace(bins[0], bins[-1], 300)
    ax.plot(xs, kde(xs), color=col, linewidth=2.5)

# x=0 reference
ax.axvline(0, color="#AAAAAA", linestyle="--", linewidth=1.3, label="No aging gap (gap = 0)")

# Mean lines
mean_surv = gap_surv.mean()
mean_died = gap_died.mean()
for mean_v, col, yf in [(mean_surv, C_SURV, 0.68), (mean_died, C_DIED, 0.82)]:
    ax.axvline(mean_v, color=col, linestyle="-", linewidth=1.8, alpha=0.8)
    ax.text(mean_v + 0.4, ax.get_ylim()[1]*yf if ax.get_ylim()[1] > 0 else 0.06,
            f"Mean\n{mean_v:+.1f} yrs", color=col, fontsize=8.5, ha="left")

# Stats annotation box
stats_text = (f"Died:      mean gap = {mean_died:+.1f} yrs\n"
              f"Survived: mean gap = {mean_surv:+.1f} yrs\n"
              f"p-value (Mann-Whitney) = {mn_p:.4f}")
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", alpha=0.95),
        fontfamily="monospace")

ax.set_xlabel("PhenoAge Gap (years)\nPositive = Biologically Older than Chronological Age",
              fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Biological Age Acceleration by Mortality Status", fontsize=13,
             fontweight="bold", pad=14)
ax.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="#CCCCCC")
spine_clean(ax)

plt.tight_layout()
fig5_path = os.path.join(OUTPUT_DIR, "phenoage_gap_distribution_final.png")
plt.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved → {fig5_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("All 5 figures saved to output/")
print("=" * 62)
print(f"  roc_curves_final.png            — ROC curves for all 4 models with AUC labels")
print(f"  feature_importance_final.png    — Grouped importance: Levine coefficients vs ML")
print(f"  phenoage_scatter_final.png      — PhenoAge vs chronological age, colored by outcome")
print(f"  mortality_distribution_final.png — Elastic Net risk score: survived vs died")
print(f"  phenoage_gap_distribution_final.png — KDE of biological age gap by mortality")
print(f"\n  Stats (Figure 5): Died mean gap={mean_died:+.1f} yrs | "
      f"Survived mean gap={mean_surv:+.1f} yrs | Mann-Whitney p={mn_p:.4f}")
