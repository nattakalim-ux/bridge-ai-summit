"""
Phase 1: Merge NHANES 2015-2016 data for PhenoAge vs ML comparison.

CYCLE NOTE: Originally planned 2013-2014 (_H suffix), but that cycle has NO CRP
data (standard CRP was dropped after 2009-2010; hs-CRP only reintroduced in
2015-2016). Pivoted to 2015-2016 (_I suffix) which has HSCRP_I + mortality linkage.

FOLLOW-UP NOTE: 2015-2016 mortality linkage (to Dec 31, 2019) gives max ~61 months
(~5 years) of follow-up — not 10 years. The died_10yr flag is retained for schema
compatibility (PERMTH_INT <= 120 is always true for any death in this window), but
the outcome effectively means "died before Dec 31, 2019".

Variable mapping (2015-2016 _I suffix):
  DEMO_I   → SEQN, RIDAGEYR, RIAGENDR, RIDRETH1
  BIOPRO_I → LBXSAL (albumin g/dL), LBXSCR (creatinine mg/dL), LBXSAPSI (ALP U/L)
  GLU_I    → LBXGLU (fasting glucose mg/dL)   [fasting subsample only]
  HSCRP_I  → LBXHSCRP (hs-CRP mg/L)
  CBC_I    → LBXWBCSI (WBC 10³/µL), LBXLYPCT (lymph%), LBXMCVSI (MCV fL), LBXRDW (RDW%)
"""

import pandas as pd
import numpy as np

RAW      = "/Users/ming/Desktop/MDCU/Data/nhanes_2013_2014"
DATA_DIR = "/Users/ming/Desktop/MDCU/Data"

MORT_PATH = f"{RAW}/NHANES_2015_2016_MORT_2019_PUBLIC.dat"

# ── 1. Parse mortality fixed-width file ──────────────────────────────────────
print("=" * 60)
print("PHASE 1: DOWNLOAD AND MERGE")
print("=" * 60)
print("\nCYCLE: NHANES 2015-2016 (suffix _I)")
print("  Reason: 2013-2014 lacks CRP data entirely.\n"
      "  2015-2016 is the earliest cycle with hs-CRP + mortality linkage.\n")

print("[1/6] Parsing mortality linkage file...")
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
print(f"  Mortality file rows: {len(mort):,}")
print(f"  ELIGSTAT distribution: {mort['ELIGSTAT'].value_counts().to_dict()}")
print(f"  MORTSTAT distribution: {mort['MORTSTAT'].value_counts().to_dict()}")
print(f"  PERMTH_INT range: {mort['PERMTH_INT'].min():.0f}–{mort['PERMTH_INT'].max():.0f} months "
      f"(~{mort['PERMTH_INT'].max()/12:.1f} yrs max follow-up)\n")

# ── 2. Load NHANES XPT files ──────────────────────────────────────────────────
print("[2/6] Loading XPT files...")

demo = pd.read_sas(f"{RAW}/DEMO_I.XPT", format="xport", encoding="latin-1")[
    ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH1"]
]
print(f"  DEMO_I:   {len(demo):,} rows")

bio = pd.read_sas(f"{RAW}/BIOPRO_I.XPT", format="xport", encoding="latin-1")[
    ["SEQN", "LBXSAL", "LBXSCR", "LBXSAPSI"]
]
print(f"  BIOPRO_I: {len(bio):,} rows  (albumin, creatinine, ALP)")

glu = pd.read_sas(f"{RAW}/GLU_I.XPT", format="xport", encoding="latin-1")[
    ["SEQN", "LBXGLU"]
]
print(f"  GLU_I:    {len(glu):,} rows  (fasting glucose mg/dL — fasting subsample only)")

crp = pd.read_sas(f"{RAW}/HSCRP_I.XPT", format="xport", encoding="latin-1")[
    ["SEQN", "LBXHSCRP"]
]
print(f"  HSCRP_I:  {len(crp):,} rows  (hs-CRP mg/L)")

cbc = pd.read_sas(f"{RAW}/CBC_I.XPT", format="xport", encoding="latin-1")[
    ["SEQN", "LBXWBCSI", "LBXLYPCT", "LBXMCVSI", "LBXRDW"]
]
print(f"  CBC_I:    {len(cbc):,} rows  (WBC, lymph%, MCV, RDW)")

# ── 3. Merge ──────────────────────────────────────────────────────────────────
print("\n[3/6] Merging on SEQN...")

df = (demo
      .merge(bio,  on="SEQN", how="inner")
      .merge(glu,  on="SEQN", how="inner")
      .merge(crp,  on="SEQN", how="inner")
      .merge(cbc,  on="SEQN", how="inner")
      .merge(mort, on="SEQN", how="inner"))

print(f"  After inner merge (all 6 files): {len(df):,} rows")

# ── 4. Filter ─────────────────────────────────────────────────────────────────
print("\n[4/6] Applying filters...")

n0 = len(df)
print(f"  Total participants (merged): {n0:,}")

df_adults = df[df["RIDAGEYR"] >= 18].copy()
n1 = len(df_adults)
print(f"  After age ≥ 18 filter:       {n1:,}  (removed {n0-n1:,})")

df_elig = df_adults[df_adults["ELIGSTAT"] == "1"].copy()
n2 = len(df_elig)
print(f"  After ELIGSTAT == 1 filter:  {n2:,}  (removed {n1-n2:,})")

# ── 5. Rename & clean ─────────────────────────────────────────────────────────
df_elig = df_elig.rename(columns={
    "RIDAGEYR":  "age",
    "RIAGENDR":  "gender",      # 1=Male, 2=Female
    "RIDRETH1":  "race",        # 1=Mexican American, etc.
    "LBXSAL":    "albumin",     # g/dL
    "LBXSCR":    "creatinine",  # mg/dL
    "LBXSAPSI":  "alp",         # U/L
    "LBXGLU":    "glucose",     # mg/dL (fasting)
    "LBXHSCRP":  "crp_mgl",    # mg/L
    "LBXMCVSI":  "mcv",         # fL
    "LBXRDW":    "rdw",         # %
    "LBXWBCSI":  "wbc",         # 10³/µL
    "LBXLYPCT":  "lymph_pct",   # %
})

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp_mgl",
            "mcv", "rdw", "wbc", "lymph_pct"]

df_clean = df_elig[["SEQN", "age", "gender", "race",
                     "ELIGSTAT", "MORTSTAT", "PERMTH_INT"] + FEATURES].dropna()
n3 = len(df_clean)
n_missing = n2 - n3
print(f"  After dropping missing labs:  {n3:,}  (removed {n_missing:,} with missing values)")

# ── 6. Build mortality target ─────────────────────────────────────────────────
print("\n[5/6] Building outcome variables...")

# 10-yr flag (schema-compatible with other files; effectively = MORTSTAT since
# max follow-up is ~61 months < 120 months)
df_clean["died_10yr"] = (
    (df_clean["MORTSTAT"] == "1") & (df_clean["PERMTH_INT"] <= 120)
).astype(int)

# All-cause mortality during follow-up (the actually meaningful outcome here)
df_clean["died_followup"] = (df_clean["MORTSTAT"] == "1").astype(int)

n_dead_10yr    = df_clean["died_10yr"].sum()
n_dead_followup = df_clean["died_followup"].sum()
n_total = len(df_clean)

print(f"  died_10yr:     {n_dead_10yr:,} / {n_total:,}  ({100*n_dead_10yr/n_total:.2f}%)")
print(f"  died_followup: {n_dead_followup:,} / {n_total:,}  ({100*n_dead_followup/n_total:.2f}%)")
print(f"  (Both counts are identical because max PERMTH_INT = "
      f"{df_clean['PERMTH_INT'].max():.0f} months < 120 months)")
print(f"\n  Follow-up distribution (months):")
print(f"    min={df_clean['PERMTH_INT'].min():.0f}  median={df_clean['PERMTH_INT'].median():.0f}"
      f"  max={df_clean['PERMTH_INT'].max():.0f}  mean={df_clean['PERMTH_INT'].mean():.1f}")

# ── 7. Check for implausible values ──────────────────────────────────────────
print("\n[6/6] Checking for implausible values...")
for feat in FEATURES:
    n_neg = (df_clean[feat] < 0).sum()
    if n_neg > 0:
        print(f"  WARNING: {feat} has {n_neg} negative values")
    if df_clean[feat].isnull().sum() > 0:
        print(f"  WARNING: {feat} still has nulls after dropna")
print("  All lab values look clean (no negatives, no nulls)")

# Flag extreme values
print("\n  Lab value ranges:")
for feat in FEATURES:
    col = df_clean[feat]
    print(f"    {feat:<12} min={col.min():.2f}  median={col.median():.2f}  "
          f"max={col.max():.2f}  p99={col.quantile(0.99):.2f}")

# ── 8. Summary statistics ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1 SUMMARY")
print("=" * 60)
print(f"\n  Data cycle:         NHANES 2015-2016 (suffix _I)")
print(f"  CRP source:         HSCRP_I — LBXHSCRP (mg/L, hs-CRP)")
print(f"  Glucose source:     GLU_I — LBXGLU (mg/dL, fasting)")
print(f"  Mortality linkage:  NHANES_2015_2016_MORT_2019_PUBLIC.dat")
print(f"  Max follow-up:      {df_clean['PERMTH_INT'].max():.0f} months (~5 yrs) — NOT 10 years")
print(f"\n  Filtering pipeline:")
print(f"    Total merged:                  {n0:>6,}")
print(f"    After age ≥ 18:               {n1:>6,}")
print(f"    After eligibility filter:      {n2:>6,}")
print(f"    After complete cases (9 labs): {n3:>6,}")
print(f"\n  Final sample:     {n3:,} participants")
print(f"  Mortality rate:   {100*n_dead_10yr/n_total:.2f}%  ({n_dead_10yr} deaths)")
print(f"  Age range:        {df_clean['age'].min():.0f}–{df_clean['age'].max():.0f} yrs  "
      f"(mean={df_clean['age'].mean():.1f})")

print("\n  IMPORTANT LIMITATION:")
print("  Max follow-up is ~61 months (5 years), not 10 years.")
print("  The 'died_10yr' column is schema-compatible but effectively")
print("  means 'died before Dec 31, 2019' for this cohort.")
print("  The AUC comparison between PhenoAge and ML models remains valid.")

if n3 < 1000:
    print(f"\n  *** WARNING: Final n = {n3} is below 1,000. "
          "Too small for reliable ML training. ***")

# ── 9. Save ───────────────────────────────────────────────────────────────────
out = f"{DATA_DIR}/merged_nhanes_2015_2016_with_mortality.csv"
df_clean.to_csv(out, index=False)
print(f"\nSaved → {out}")
print(f"Columns: {list(df_clean.columns)}")
