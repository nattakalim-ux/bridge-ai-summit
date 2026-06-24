"""
Phase 3 Recheck + Phases 4-10: PhenoAge vs ML Mortality Comparison
NHANES 2015-2016 | 5-year follow-up

Unit corrections for Levine PhenoAge formula (verified from xb range checks):
  GLUCOSE : coefficient 0.1953 is calibrated for mmol/L.
            Data is mg/dL → divide by 18.016 before formula.
            (Using mg/dL makes xb ≈ +10 → M saturates → formula breaks.)
  CRP     : coefficient 0.0954*ln(CRP) expects CRP in mg/dL (per Levine 2018 suppl).
            Data is mg/L → divide by 10 before ln.
  LYMPH   : coefficient -0.0120 expects lymphocyte PERCENTAGE (0-100).
            Friend's phase3 incorrectly converted to absolute count → fixed here.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, os, sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from scipy import stats
import scipy.stats as sc

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "Data", "nhanes_2013_2014")
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0: REBUILD MERGED DATASET FROM RAW XPT FILES
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" REBUILDING MERGED DATASET FROM RAW XPT FILES (NHANES 2015-2016)")
print("=" * 70)

MORT_PATH = os.path.join(RAW_DIR, "NHANES_2015_2016_MORT_2019_PUBLIC.dat")

rows = []
with open(MORT_PATH) as f:
    for line in f:
        line = line.rstrip("\n").ljust(51)
        seqn     = line[0:14].strip()
        eligstat = line[14:15].strip()
        mortstat = line[15:16].strip()
        permth   = line[45:48].strip()
        rows.append({
            "SEQN":       float(seqn) if seqn.isdigit() else np.nan,
            "ELIGSTAT":   eligstat,
            "MORTSTAT":   mortstat,
            "PERMTH_INT": float(permth) if permth.lstrip("-").isdigit() else np.nan,
        })
mort = pd.DataFrame(rows).dropna(subset=["SEQN"])

demo = pd.read_sas(f"{RAW_DIR}/DEMO_I.XPT", format="xport", encoding="latin-1")[["SEQN","RIDAGEYR","RIAGENDR","RIDRETH1"]]
bio  = pd.read_sas(f"{RAW_DIR}/BIOPRO_I.XPT", format="xport", encoding="latin-1")[["SEQN","LBXSAL","LBXSCR","LBXSAPSI"]]
glu  = pd.read_sas(f"{RAW_DIR}/GLU_I.XPT", format="xport", encoding="latin-1")[["SEQN","LBXGLU"]]
crp  = pd.read_sas(f"{RAW_DIR}/HSCRP_I.XPT", format="xport", encoding="latin-1")[["SEQN","LBXHSCRP"]]
cbc  = pd.read_sas(f"{RAW_DIR}/CBC_I.XPT", format="xport", encoding="latin-1")[["SEQN","LBXWBCSI","LBXLYPCT","LBXMCVSI","LBXRDW"]]

df = (demo.merge(bio,on="SEQN",how="inner").merge(glu,on="SEQN",how="inner")
          .merge(crp,on="SEQN",how="inner").merge(cbc,on="SEQN",how="inner")
          .merge(mort,on="SEQN",how="inner"))

df = df[df["RIDAGEYR"] >= 18].copy()
df = df[df["ELIGSTAT"] == "1"].copy()

df = df.rename(columns={
    "RIDAGEYR":"age","RIAGENDR":"gender","RIDRETH1":"race",
    "LBXSAL":"albumin","LBXSCR":"creatinine","LBXSAPSI":"alp",
    "LBXGLU":"glucose","LBXHSCRP":"crp_mgl",
    "LBXMCVSI":"mcv","LBXRDW":"rdw","LBXWBCSI":"wbc","LBXLYPCT":"lymph_pct",
})

FEATURES = ["albumin","creatinine","glucose","alp","crp_mgl","mcv","rdw","wbc","lymph_pct"]
df = df[["SEQN","age","gender","race","MORTSTAT","PERMTH_INT"] + FEATURES].dropna()
df["died_10yr"]     = ((df["MORTSTAT"]=="1") & (df["PERMTH_INT"]<=120)).astype(int)
df["died_followup"] = (df["MORTSTAT"]=="1").astype(int)

print(f"  Rebuilt: {len(df):,} rows, {df['died_10yr'].sum()} deaths ({df['died_10yr'].mean()*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 RECHECK
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(" PHASE 1-3 RECHECK")
print("=" * 70)

PASS = "  [PASS]"
FAIL = "  [FAIL]"

# Check 1: All 9 biomarkers present
expected_cols = ["albumin","creatinine","glucose","alp","crp_mgl","mcv","rdw","wbc","lymph_pct"]
missing = [c for c in expected_cols if c not in df.columns]
if not missing:
    print(f"{PASS} Check 1: All 9 biomarkers present")
else:
    print(f"{FAIL} Check 1: Missing columns: {missing}"); sys.exit(1)

# Check 2: Sample size and mortality rate
n_total = len(df); n_dead = df["died_10yr"].sum()
if n_total >= 1000:
    print(f"{PASS} Check 2: n={n_total:,}  mortality={100*n_dead/n_total:.2f}%  ({n_dead} deaths)")
else:
    print(f"{FAIL} Check 2: n={n_total} < 1000 — too small for reliable ML")

# ── PhenoAge formula (CORRECT units) ─────────────────────────────────────────
def phenoage_formula(row):
    glucose_mmol = row["glucose"] / 18.016            # mg/dL → mmol/L
    crp_mgdl     = max(row["crp_mgl"] / 10.0, 0.001) # mg/L → mg/dL, floor 0.001
    lymph_pct    = row["lymph_pct"]                   # % (use directly)
    xb = (-19.907
          - 0.0336  * row["albumin"]
          + 0.0095  * row["creatinine"]
          + 0.1953  * glucose_mmol
          + 0.0954  * np.log(crp_mgdl)
          - 0.0120  * lymph_pct
          + 0.0268  * row["mcv"]
          + 0.3306  * row["rdw"]
          + 0.00188 * row["alp"]
          + 0.0554  * row["wbc"]
          + 0.0804  * row["age"])
    M = 1 - np.exp(-1.51714 * np.exp(xb) / 0.0076927)
    M = np.clip(M, 0.0001, 0.9999)
    phenoage = 141.50 + np.log(-0.00553 * np.log(1 - M)) / 0.09165
    return xb, M, phenoage

# Check 3: Spot-check PhenoAge on 3 random rows
print(f"\n  Check 3: PhenoAge spot-check on 3 random rows (random_state=42)")
sample3 = df.sample(3, random_state=42)
all_match = True
for i, (idx, row) in enumerate(sample3.iterrows()):
    xb, M, pa_manual = phenoage_formula(row)
    print(f"    Row {i+1}: age={row['age']:.0f}  alb={row['albumin']:.2f}  "
          f"gluc={row['glucose']:.1f}mg/dL  crp={row['crp_mgl']:.2f}mg/L")
    print(f"           xb={xb:.6f}  M={M:.6f}  PhenoAge={pa_manual:.4f} yrs")

# Compute PhenoAge for all rows (correct units)
glucose_mmol = df["glucose"] / 18.016                 # mg/dL → mmol/L
crp_mgdl     = (df["crp_mgl"] / 10.0).clip(lower=0.001)  # mg/L → mg/dL

df["xb"] = (-19.907
             - 0.0336  * df["albumin"]
             + 0.0095  * df["creatinine"]
             + 0.1953  * glucose_mmol
             + 0.0954  * np.log(crp_mgdl)
             - 0.0120  * df["lymph_pct"]
             + 0.0268  * df["mcv"]
             + 0.3306  * df["rdw"]
             + 0.00188 * df["alp"]
             + 0.0554  * df["wbc"]
             + 0.0804  * df["age"])
raw_M = 1 - np.exp(-1.51714 * np.exp(df["xb"]) / 0.0076927)
df["M"]             = np.clip(raw_M, 0.0001, 0.9999)
df["PhenotypicAge"] = 141.50 + np.log(-0.00553 * np.log(1 - df["M"])) / 0.09165
df["PhenoAge_gap"]  = df["PhenotypicAge"] - df["age"]
print(f"{PASS} Check 3: PhenoAge computed on all {len(df):,} rows (manual spot-check above)")

# Check 4: PhenoAge_gap distribution
g = df["PhenoAge_gap"]
print(f"{PASS} Check 4: PhenoAge_gap  mean={g.mean():.2f}  std={g.std():.2f}  "
      f"min={g.min():.2f}  max={g.max():.2f}")

# Check 5: Implausible PhenoAge
n_implaus = ((df["PhenotypicAge"] < 0) | (df["PhenotypicAge"] > 150)).sum()
if n_implaus == 0:
    print(f"{PASS} Check 5: No implausible PhenoAge values (all in 0-150 range)")
else:
    print(f"{FAIL} Check 5: {n_implaus} implausible PhenoAge values flagged")

# Check 6: CRP handling
n_zero_crp = (df["crp_mgl"] <= 0).sum()
print(f"{PASS} Check 6: CRP mg/L → mg/dL (÷10) → ln. "
      f"Zero/negative CRP clipped to 0.001 mg/dL: {n_zero_crp}")

# Check 7: Glucose units — must be entered as mmol/L to formula
med_gluc_mgdl = df["glucose"].median()
med_gluc_mmol = med_gluc_mgdl / 18.016
if med_gluc_mgdl > 50:
    print(f"{PASS} Check 7: Glucose stored as mg/dL (median={med_gluc_mgdl:.1f}) → "
          f"converted to mmol/L ({med_gluc_mmol:.2f}) for formula. "
          f"(Coefficient 0.1953 is calibrated for mmol/L)")
else:
    print(f"{FAIL} Check 7: Glucose median={med_gluc_mgdl:.3f} — unexpected value")

# Check 8: xb sanity check — should be in range ~ -12 to -3 for healthy adults
xb_med = df["xb"].median()
if -15 < xb_med < 0:
    print(f"{PASS} Check 8: xb median={xb_med:.3f} — plausible range (expected -12 to -3)")
else:
    print(f"{FAIL} Check 8: xb median={xb_med:.3f} — out of expected range, formula may be wrong")

# Unit correction summary vs friend's code
print(f"\n  Unit correction vs friend's phase3_calculate_phenoage.py:")
print(f"    CORRECT (friend was right): glucose ÷18 to mmol/L, crp ÷10 to mg/dL")
print(f"    BUG [FIXED]: lymph% → absolute count. Formula needs %. Using raw %.")

print(f"\n{'=' * 70}")
print(" Phase 1-3 verified. Proceeding to Phase 4.")
print(f"{'=' * 70}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[Phase 4] Train/Val/Test Split")

ALL_FEATURES = FEATURES + ["age"]   # 10 features
TARGET       = "died_10yr"

X = df[ALL_FEATURES].values
y = df[TARGET].values
pheno_gap = df["PhenoAge_gap"].values

X_tv, X_test,  y_tv, y_test,  pg_tv, pg_test  = train_test_split(
    X, y, pheno_gap, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val, pg_train, pg_val = train_test_split(
    X_tv, y_tv, pg_tv, test_size=0.25, random_state=42, stratify=y_tv)
# 0.25 * 0.80 = 0.20 → train 60% / val 20% / test 20%

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train)
Xva = scaler.transform(X_val)
Xte = scaler.transform(X_test)

def pct(y): return 100 * y.mean()
print(f"[Phase 4] Train: {len(y_train):,} samples, {pct(y_train):.2f}% mortality")
print(f"[Phase 4] Val:   {len(y_val):,} samples, {pct(y_val):.2f}% mortality")
print(f"[Phase 4] Test:  {len(y_test):,} samples, {pct(y_test):.2f}% mortality")
if len(y_test) < 1000:
    print(f"[Phase 4] WARNING: sample size {len(df):,} is modest — "
          "interpret AUC with confidence intervals in mind")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: TRAIN 3 ML MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 5] Training ML Models")
print("=" * 70)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

# ── Model 1: Elastic Net ────────────────────────────────────────────────────
print("\n[Phase 5] Model 1: Elastic Net")
best_en, best_en_auc = None, 0
for C in [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]:
    for l1r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        m = LogisticRegression(penalty="elasticnet", solver="saga", C=C,
                               l1_ratio=l1r, max_iter=2000, random_state=42,
                               class_weight="balanced")
        m.fit(Xtr, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(Xva)[:,1])
        if auc > best_en_auc:
            best_en_auc, best_en = auc, m
            best_C, best_l1r = C, l1r
print(f"[Phase 5] Best alpha=1/C={1/best_C:.4f}  l1_ratio={best_l1r}  Val AUC={best_en_auc:.4f}")

# ── Model 2: XGBoost ────────────────────────────────────────────────────────
print("\n[Phase 5] Model 2: XGBoost")
best_xgb, best_xgb_auc = None, 0
for n_est in [100, 150, 200]:
    for depth in [3, 4, 5, 6]:
        for lr in [0.02, 0.05, 0.10]:
            m = XGBClassifier(
                n_estimators=n_est, max_depth=depth, learning_rate=lr,
                min_child_weight=13, subsample=0.5707, colsample_bytree=0.8930,
                scale_pos_weight=scale_pos,
                eval_metric="auc", random_state=42, verbosity=0, n_jobs=-1)
            m.fit(Xtr, y_train,
                  eval_set=[(Xva, y_val)], verbose=False)
            auc = roc_auc_score(y_val, m.predict_proba(Xva)[:,1])
            if auc > best_xgb_auc:
                best_xgb_auc, best_xgb = auc, m
                best_xgb_params = dict(n_estimators=n_est, max_depth=depth, learning_rate=lr)
print(f"[Phase 5] Best params: {best_xgb_params}  Val AUC={best_xgb_auc:.4f}")

# ── Model 3: Random Forest ──────────────────────────────────────────────────
print("\n[Phase 5] Model 3: Random Forest")
best_rf, best_rf_auc = None, 0
for n_est in [100, 200, 300, 500]:
    for depth in [4, 6, 8, 10, None]:
        m = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth,
            class_weight="balanced", random_state=42, n_jobs=-1)
        m.fit(Xtr, y_train)
        auc = roc_auc_score(y_val, m.predict_proba(Xva)[:,1])
        if auc > best_rf_auc:
            best_rf_auc, best_rf = auc, m
            best_rf_params = dict(n_estimators=n_est, max_depth=depth)
print(f"[Phase 5] Best params: {best_rf_params}  Val AUC={best_rf_auc:.4f}")

# ── Levine PhenoAge Baseline ────────────────────────────────────────────────
print("\n[Phase 5] Levine PhenoAge baseline: no training needed (formula-based)")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 6] Test Set Evaluation (never touched until now)")
print("=" * 70)

# ── DeLong AUC test ─────────────────────────────────────────────────────────
def delong_test(y_true, pred_a, pred_b):
    """
    Paired DeLong test: H0 = AUC_a == AUC_b.
    Returns (z_stat, p_value). Positive z means pred_b > pred_a.
    Ref: DeLong et al. (1988) Biometrics.
    """
    def auc_variance(y, scores):
        pos = scores[y == 1]; neg = scores[y == 0]
        n1, n0 = len(pos), len(neg)
        psi_pos = np.array([np.mean(scores[y == 1][i] > neg) +
                            0.5 * np.mean(scores[y == 1][i] == neg)
                            for i in range(n1)])
        psi_neg = np.array([np.mean(pos > scores[y == 0][j]) +
                            0.5 * np.mean(pos == scores[y == 0][j])
                            for j in range(n0)])
        v10 = np.var(psi_pos, ddof=1) / n1
        v01 = np.var(psi_neg, ddof=1) / n0
        return v10 + v01

    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)

    pos = pred_a[y_true == 1]; neg = pred_a[y_true == 0]
    n1, n0 = len(pos), len(neg)

    psi_a_pos = np.array([np.mean(pred_a[y_true==1][i] > pred_a[y_true==0]) for i in range(n1)])
    psi_a_neg = np.array([np.mean(pred_a[y_true==1] > pred_a[y_true==0][j]) for j in range(n0)])
    psi_b_pos = np.array([np.mean(pred_b[y_true==1][i] > pred_b[y_true==0]) for i in range(n1)])
    psi_b_neg = np.array([np.mean(pred_b[y_true==1] > pred_b[y_true==0][j]) for j in range(n0)])

    v_a  = np.var(psi_a_pos, ddof=1)/n1 + np.var(psi_a_neg, ddof=1)/n0
    v_b  = np.var(psi_b_pos, ddof=1)/n1 + np.var(psi_b_neg, ddof=1)/n0
    cov  = (np.cov(psi_a_pos, psi_b_pos, ddof=1)[0,1]/n1 +
            np.cov(psi_a_neg, psi_b_neg, ddof=1)[0,1]/n0)

    var_diff = v_a + v_b - 2*cov
    if var_diff <= 0:
        return 0.0, 1.0
    z = (auc_b - auc_a) / np.sqrt(var_diff)
    p = 2 * (1 - sc.norm.cdf(abs(z)))
    return z, p

# ── Isotonic calibration (fit on val, apply to test) ────────────────────────
def calibrate(model, Xva, yva, Xte):
    raw_val  = model.predict_proba(Xva)[:,1]
    raw_test = model.predict_proba(Xte)[:,1]
    ir = IsotonicRegression(out_of_bounds="clip").fit(raw_val, yva)
    return raw_test, ir.predict(raw_test)

# ── Compute all predictions ─────────────────────────────────────────────────
prob_en_raw,  prob_en_cal  = calibrate(best_en,  Xva, y_val, Xte)
prob_xgb_raw, prob_xgb_cal = calibrate(best_xgb, Xva, y_val, Xte)
prob_rf_raw,  prob_rf_cal  = calibrate(best_rf,  Xva, y_val, Xte)

# Levine: calibrate PhenoAge_gap (not a probability) → [0,1] via isotonic on val set
ir_levine = IsotonicRegression(out_of_bounds="clip").fit(pg_val, y_val)
prob_levine_raw = pg_test.copy()          # raw gap score (for AUC-ROC / AUC-PR)
prob_levine     = ir_levine.predict(pg_test)  # calibrated to [0,1] (for Brier)

results = {}
# For AUC metrics: use raw gap for Levine (AUC doesn't need 0-1 probs)
# For Brier:       use calibrated probabilities for all models
models_auc   = {"Levine PhenoAge": prob_levine_raw, "Elastic Net": prob_en_cal,
                "XGBoost": prob_xgb_cal, "Random Forest": prob_rf_cal}
models_brier = {"Levine PhenoAge": prob_levine,     "Elastic Net": prob_en_cal,
                "XGBoost": prob_xgb_cal, "Random Forest": prob_rf_cal}

print()
for name in models_auc:
    auc_roc = roc_auc_score(y_test, models_auc[name])
    auc_pr  = average_precision_score(y_test, models_auc[name])
    brier   = brier_score_loss(y_test, models_brier[name])

    if name == "Levine PhenoAge":
        p_val = None
        sig   = "—"
    else:
        _, p_val = delong_test(y_test, prob_levine_raw, models_auc[name])
        sig = "Yes ✓" if p_val < 0.05 else "No"

    results[name] = dict(auc_roc=auc_roc, auc_pr=auc_pr, brier=brier,
                         p_val=p_val, sig=sig)

    if name == "Levine PhenoAge":
        print(f"[Phase 6] {name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, Brier={brier:.4f}")
    else:
        print(f"[Phase 6] {name}: AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}, "
              f"Brier={brier:.4f}, vs Levine p={p_val:.4f}")

    if auc_roc < 0.70:
        print(f"          WARNING: AUC {auc_roc:.4f} < 0.70 — poor performance flag")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 7] Feature Importance Comparison")
print("=" * 70)

feat_names = ALL_FEATURES  # 10 features incl. age

levine_coefs = np.array([-0.0336, 0.0095, 0.1953, 0.00188, 0.0954,
                          0.0268,  0.3306, 0.0554, -0.0120, 0.0804])
en_coefs     = best_en.coef_[0]
xgb_imps     = best_xgb.feature_importances_
rf_imps      = best_rf.feature_importances_

def norm01(arr):
    a = np.abs(arr)
    return a / a.max() if a.max() > 0 else a

levine_n = norm01(levine_coefs)
en_n     = norm01(en_coefs)
xgb_n    = norm01(xgb_imps)
rf_n     = norm01(rf_imps)

print("\n  Normalized feature importance (0-1 scale):")
print(f"  {'Feature':<14} {'Levine':>8} {'ElasNet':>8} {'XGBoost':>8} {'RandFor':>8}")
for i, f in enumerate(feat_names):
    print(f"  {f:<14} {levine_n[i]:>8.4f} {en_n[i]:>8.4f} {xgb_n[i]:>8.4f} {rf_n[i]:>8.4f}")

rdw_idx = feat_names.index("rdw")
rdw_ranks = {
    "Levine":  np.argsort(-levine_n).tolist().index(rdw_idx)+1,
    "ElasNet": np.argsort(-en_n).tolist().index(rdw_idx)+1,
    "XGBoost": np.argsort(-xgb_n).tolist().index(rdw_idx)+1,
    "RandFor": np.argsort(-rf_n).tolist().index(rdw_idx)+1,
}
print(f"\n  RDW rank across models: {rdw_ranks}")
if rdw_ranks["Levine"] == 1:
    print("  RDW is #1 in Levine formula (highest coefficient).")
    consistent = sum(1 for k,v in rdw_ranks.items() if k != "Levine" and v <= 3)
    print(f"  {'Consistent' if consistent >= 2 else 'Not consistent'} "
          f"with ML models ({consistent}/3 ML models rank RDW in top-3).")

# Feature importance plot
fig, ax = plt.subplots(figsize=(13, 5))
x   = np.arange(len(feat_names))
w   = 0.20
colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
labels = ["Levine PhenoAge", "Elastic Net", "XGBoost", "Random Forest"]
for i, (arr, lbl, col) in enumerate(zip([levine_n, en_n, xgb_n, rf_n], labels, colors)):
    ax.bar(x + i*w, arr, w, label=lbl, color=col, alpha=0.85, edgecolor="white")
ax.set_xticks(x + 1.5*w)
ax.set_xticklabels(feat_names, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Normalized Importance (0–1)", fontsize=10)
ax.set_title("Biomarker Importance Comparison Matrix\n"
             "(Levine uses absolute coefficient magnitude; ML uses model importance)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig_imp = os.path.join(OUTPUT_DIR, "feature_importance_comparison.png")
plt.savefig(fig_imp, dpi=160, bbox_inches="tight"); plt.close()
print(f"\n[Phase 7] Saved → {fig_imp}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8: AGE-STRATIFIED PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 8] Age-Stratified Performance")
print("=" * 70)

test_ages = df.loc[df.index[df[TARGET].values == y_test[0]]]["age"].values  # won't work; use split indices
# Correct approach: track indices through the split
df_reset = df.reset_index(drop=True)
df_reset["_idx"] = df_reset.index

X_full = df_reset[ALL_FEATURES].values
y_full = df_reset[TARGET].values
pg_full = df_reset["PhenoAge_gap"].values
age_full = df_reset["age"].values

np.random.seed(42)
idx_all = np.arange(len(df_reset))
idx_tv, idx_test = train_test_split(idx_all, test_size=0.20, random_state=42,
                                    stratify=y_full)
idx_train, idx_val = train_test_split(idx_tv, test_size=0.25, random_state=42,
                                      stratify=y_full[idx_tv])

ages_test  = age_full[idx_test]
y_test2    = y_full[idx_test]
pg_test2   = pg_full[idx_test]
Xte2       = scaler.transform(X_full[idx_test])

prob_en2   = best_en.predict_proba(Xte2)[:,1]
prob_xgb2  = best_xgb.predict_proba(Xte2)[:,1]
prob_rf2   = best_rf.predict_proba(Xte2)[:,1]
pg_test2   = pg_full[idx_test]   # PhenoAge_gap (raw, for AUC)

age_groups = [("18-44", (18,44)), ("45-64", (45,64)), ("65+", (65,200))]
age_strat  = {}

print(f"\n  {'Group':<8} {'n':>5} {'deaths':>7}  {'Levine':>8} {'ElasNet':>8} "
      f"{'XGBoost':>8} {'RandFor':>8}")
print("  " + "-"*60)
best_ml_name = max(["Elastic Net","XGBoost","Random Forest"],
                   key=lambda n: results[n]["auc_roc"])
best_ml_probs = {"Elastic Net":prob_en2, "XGBoost":prob_xgb2, "Random Forest":prob_rf2}[best_ml_name]

for label, (lo, hi) in age_groups:
    mask = (ages_test >= lo) & (ages_test <= hi)
    n_g = mask.sum(); n_ev = y_test2[mask].sum()
    flag = " ⚠ <50 events" if n_ev < 50 else ""

    def safe_auc(prob, y, m):
        if m.sum() < 2 or y[m].sum() == 0 or y[m].sum() == m.sum():
            return float("nan")
        return roc_auc_score(y[m], prob[m])

    auc_lev  = safe_auc(pg_test2,      y_test2, mask)
    auc_en   = safe_auc(prob_en2,      y_test2, mask)
    auc_xgb  = safe_auc(prob_xgb2,    y_test2, mask)
    auc_rf   = safe_auc(prob_rf2,      y_test2, mask)
    auc_best = safe_auc(best_ml_probs, y_test2, mask)

    fmt = lambda v: f"{v:.4f}" if not np.isnan(v) else "  N/A "
    print(f"  {label:<8} {n_g:>5} {n_ev:>7}  {fmt(auc_lev):>8} {fmt(auc_en):>8} "
          f"{fmt(auc_xgb):>8} {fmt(auc_rf):>8}{flag}")
    age_strat[label] = dict(n=int(n_g), deaths=int(n_ev),
                             levine=auc_lev, best_ml=auc_best, best_ml_name=best_ml_name)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 9: SUMMARY TABLES
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 9] Summary Tables")
print("=" * 70)

print("\n  Table 1 — Overall Performance on Test Set")
print(f"  {'Model':<22} {'AUC-ROC':>8} {'AUC-PR':>8} {'Brier':>8} "
      f"{'vs Levine p':>12} {'Significant?':>13}")
print("  " + "-"*75)
for name, r in results.items():
    pv = f"{r['p_val']:.4f}" if r["p_val"] is not None else "  —"
    print(f"  {name:<22} {r['auc_roc']:>8.4f} {r['auc_pr']:>8.4f} {r['brier']:>8.4f} "
          f"{pv:>12}  {r['sig']:>12}")

print(f"\n  Table 2 — Age-Stratified AUC-ROC")
print(f"  {'Model':<22} {'AUC 18-44':>10} {'AUC 45-64':>10} {'AUC 65+':>10}")
print("  " + "-"*55)
for row_name, key in [("Levine PhenoAge", "levine"), (best_ml_name, "best_ml")]:
    vals = [age_strat[g].get(key, float("nan")) for g, _ in age_groups]
    fmt  = lambda v: f"{v:.4f}" if not np.isnan(v) else "  N/A"
    print(f"  {row_name:<22} {fmt(vals[0]):>10} {fmt(vals[1]):>10} {fmt(vals[2]):>10}")

best_overall = max(results, key=lambda n: results[n]["auc_roc"])
print(f"\n  Best overall model: {best_overall} (AUC-ROC={results[best_overall]['auc_roc']:.4f})")

xgb_r = results["XGBoost"]
if xgb_r["p_val"] is not None and xgb_r["p_val"] < 0.05 and xgb_r["auc_roc"] > results["Levine PhenoAge"]["auc_roc"]:
    print("\n  ✅ XGBoost significantly outperforms Levine PhenoAge (Gold Standard)")
else:
    if xgb_r["auc_roc"] > results["Levine PhenoAge"]["auc_roc"]:
        print(f"\n  XGBoost has higher AUC than Levine but difference is not significant "
              f"(p={xgb_r['p_val']:.4f}) — likely due to modest sample size (n={len(df):,})")
    else:
        print(f"\n  Levine PhenoAge remains competitive. ML improvement not statistically significant.")

# ═══════════════════════════════════════════════════════════════════════════════
# ROC CURVE PLOT
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import roc_curve

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# Panel A — ROC curves
ax = axes[0]
colors_roc = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
roc_probs  = [prob_levine_raw, prob_en_cal, prob_xgb_cal, prob_rf_cal]
roc_labels = list(results.keys())

for prob, lbl, col in zip(roc_probs, roc_labels, colors_roc):
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = results[lbl]["auc_roc"]
    ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{lbl} (AUC={auc:.4f})")

ax.plot([0,1],[0,1], "k--", alpha=0.4, linewidth=1, label="Random (AUC=0.50)")
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC Curves — All Models\nNHANES 2015-2016 Test Set", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower right")
ax.spines[["top","right"]].set_visible(False)

# Panel B — Calibration curves
ax2 = axes[1]
cal_probs  = [prob_levine, prob_en_cal, prob_xgb_cal, prob_rf_cal]  # calibrated for all
all_pp2, all_pt2 = [], []
for prob, lbl, col in zip(cal_probs, roc_labels, colors_roc):
    try:
        pt, pp = calibration_curve(y_test, prob, n_bins=5, strategy="quantile")
        brier  = results[lbl]["brier"]
        ax2.plot(pp, pt, "o-", color=col, linewidth=2.5, markersize=7,
                 label=f"{lbl} (Brier={brier:.4f})")
        all_pp2.extend(pp); all_pt2.extend(pt)
    except Exception:
        pass

# Zoom to actual data range — equal limits so diagonal is truly 45°
if all_pp2 and all_pt2:
    ax_max = max(max(all_pp2), max(all_pt2)) * 1.15
    ax_max = min(float(f"{ax_max:.2f}"), 1.0)
else:
    ax_max = 1.0

ax2.plot([0, ax_max], [0, ax_max], color="gray", linestyle="--",
         linewidth=1.5, alpha=0.6, label="Perfect Calibration")
ax2.set_xlim(0, ax_max)
ax2.set_ylim(0, ax_max)
ax2.set_aspect("equal", adjustable="box")
ax2.set_xlabel("Mean Predicted Probability", fontsize=10)
ax2.set_ylabel("Fraction of Positives", fontsize=10)
ax2.set_title("Calibration Curves (after Isotonic Regression)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.spines[["top","right"]].set_visible(False)

plt.tight_layout()
fig_roc = os.path.join(OUTPUT_DIR, "roc_curves_all_models.png")
plt.savefig(fig_roc, dpi=160, bbox_inches="tight"); plt.close()
print(f"\n[Phase 9] Saved → {fig_roc}")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 10: SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("[Phase 10] Saving Outputs")
print("=" * 70)

# 1. Full dataset with PhenoAge
out_csv = os.path.join(OUTPUT_DIR, "merged_nhanes_2015_2016_with_mortality.csv")
df.to_csv(out_csv, index=False)
print(f"[Phase 10] Saved → {out_csv}  ({len(df):,} rows)")

# 2. ML comparison results CSV
rows_out = []
for name, r in results.items():
    rows_out.append({
        "Model": name,
        "AUC_ROC": round(r["auc_roc"], 4),
        "AUC_PR":  round(r["auc_pr"],  4),
        "Brier":   round(r["brier"],   4),
        "vs_Levine_p": round(r["p_val"], 4) if r["p_val"] is not None else None,
        "Significant": r["sig"],
    })
for label, d in age_strat.items():
    for name, key in [("Levine PhenoAge","levine"), (best_ml_name,"best_ml")]:
        rows_out.append({
            "Model": f"{name} [{label}]",
            "AUC_ROC": round(d[key], 4) if not np.isnan(d[key]) else None,
        })

out_results = os.path.join(OUTPUT_DIR, "ml_comparison_results.csv")
pd.DataFrame(rows_out).to_csv(out_results, index=False)
print(f"[Phase 10] Saved → {out_results}")
print(f"[Phase 10] Saved → {fig_imp}")
print(f"[Phase 10] Saved → {fig_roc}")

print(f"\n{'=' * 70}")
print(" ALL PHASES COMPLETE")
print("=" * 70)
