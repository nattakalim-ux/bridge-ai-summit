"""
PhenoAge Calculator — NHANES 2017-March 2020 (P_ prefix cycle)
Based on: Levine et al. (2018) An epigenetic biomarker of aging for lifespan
          and healthspan. Aging Cell. https://doi.org/10.1111/acel.12774
"""

import os
import math
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ── Directory setup ─────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# CDC reorganised their data URLs in 2024. Each entry is (url, local_filename).
# Primary = 2017-2020 pre-pandemic (new URL); fallbacks = 2015-2016 / 2013-2014.
NEW_BASE  = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
OLD_BASE  = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"

FILE_CANDIDATES = {
    "demo": [
        (f"{NEW_BASE}/2017/DataFiles/P_DEMO.XPT",  "P_DEMO.XPT"),
        (f"{OLD_BASE}/2015/DataFiles/DEMO_I.XPT",  "DEMO_I.XPT"),
        (f"{OLD_BASE}/2013/DataFiles/DEMO_H.XPT",  "DEMO_H.XPT"),
    ],
    "biopro": [
        (f"{NEW_BASE}/2017/DataFiles/P_BIOPRO.XPT","P_BIOPRO.XPT"),
        (f"{OLD_BASE}/2015/DataFiles/BIOPRO_I.XPT","BIOPRO_I.XPT"),
        (f"{OLD_BASE}/2013/DataFiles/BIOPRO_H.XPT","BIOPRO_H.XPT"),
    ],
    "cbc": [
        (f"{NEW_BASE}/2017/DataFiles/P_CBC.XPT",   "P_CBC.XPT"),
        (f"{OLD_BASE}/2015/DataFiles/CBC_I.XPT",   "CBC_I.XPT"),
        (f"{OLD_BASE}/2013/DataFiles/CBC_H.XPT",   "CBC_H.XPT"),
    ],
    "crp": [
        (f"{NEW_BASE}/2017/DataFiles/P_HSCRP.XPT", "P_HSCRP.XPT"),
        (f"{OLD_BASE}/2015/DataFiles/HSCRP_I.XPT", "HSCRP_I.XPT"),
        (f"{OLD_BASE}/2013/DataFiles/HSCRP_H.XPT", "HSCRP_H.XPT"),
    ],
}

# ── Download with fallback ───────────────────────────────────────────────────
def _is_real_xpt(data: bytes) -> bool:
    """XPT files start with 'HEADER RECORD*' — reject HTML/redirect pages."""
    return data[:14] == b"HEADER RECORD*"

def download_file(key: str) -> tuple[Path, str]:
    """Try each candidate URL; return (local_path, actual_url) for the first hit."""
    for url, fname in FILE_CANDIDATES[key]:
        local = DATA_DIR / fname
        if local.exists() and _is_real_xpt(local.read_bytes()[:14]):
            print(f"  [cached] {fname}")
            return local, url
        elif local.exists():
            local.unlink()   # stale HTML from old download
        print(f"  Trying {url} …", end=" ", flush=True)
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and _is_real_xpt(r.content):
            local.write_bytes(r.content)
            print(f"OK ({len(r.content)//1024} KB)")
            return local, url
        print(f"HTTP {r.status_code} / not XPT")
    raise FileNotFoundError(f"None of the candidate URLs for '{key}' returned a valid XPT.")

print("=" * 60)
print("STEP 1 — Downloading NHANES files")
print("=" * 60)
paths = {}
for key in FILE_CANDIDATES:
    print(f"\n[{key.upper()}]")
    paths[key], _ = download_file(key)

# ── Load XPT files ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading XPT files …")
print("=" * 60)

demo   = pd.read_sas(paths["demo"],   format="xport", encoding="utf-8")
biopro = pd.read_sas(paths["biopro"], format="xport", encoding="utf-8")
cbc    = pd.read_sas(paths["cbc"],    format="xport", encoding="utf-8")
crp    = pd.read_sas(paths["crp"],    format="xport", encoding="utf-8")

for name, df in [("DEMO", demo), ("BIOPRO", biopro), ("CBC", cbc), ("CRP/HSCRP", crp)]:
    print(f"  {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Variable inspection — print relevant columns ─────────────────────────────
print("\n── BIOPRO columns (subset) ─────────────────")
bio_cols = [c for c in biopro.columns if any(k in c.upper() for k in
            ["ALB","SCR","SGL","SAP","LBXS"])]
print(bio_cols)

print("\n── CBC columns (subset) ─────────────────────")
cbc_cols = [c for c in cbc.columns if any(k in c.upper() for k in
            ["MCV","RDW","WBC","LYM","LYMPH","LBXWB","LBXL","LBX"])]
print(cbc_cols)

print("\n── CRP columns ─────────────────────────────")
crp_cols = [c for c in crp.columns if "CRP" in c.upper() or "LBXHS" in c.upper()]
print(crp_cols)

# ── Map variable names ───────────────────────────────────────────────────────
# We try known names in priority order and pick the first match.
def pick_var(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            print(f"  {label}: using '{c}'")
            return c
    raise KeyError(f"Could not find {label} — tried {candidates}\nAvailable: {list(df.columns)}")

print("\n── Mapping variable names ──────────────────")

AGE_VAR  = pick_var(demo,   ["RIDAGEYR"],                         "Age")
ALB_VAR  = pick_var(biopro, ["LBXSAL", "LBDSAL", "LB2SAL"],      "Albumin")
CRE_VAR  = pick_var(biopro, ["LBXSCR", "LBDSCR", "LB2SCR"],      "Creatinine")
GLU_VAR  = pick_var(biopro, ["LBXSGL", "LBDSGL", "LB2SGL"],      "Glucose")
ALP_VAR  = pick_var(biopro, ["LBXSAPSI","LBXSAP","LB2SAP","LBDSAPSI"], "Alk Phos")
CRP_VAR  = pick_var(crp,    ["LBXHSCRP","LBXCRP","LBDHSCRP"],    "CRP (hs)")
MCV_VAR  = pick_var(cbc,    ["LBXMCVSI","LBXMCV","LBDMCV"],      "MCV")
RDW_VAR  = pick_var(cbc,    ["LBXRDW",  "LBDRDW"],               "RDW")
WBC_VAR  = pick_var(cbc,    ["LBXWBCSI","LBXWBC","LBDWBC"],      "WBC")
LYM_VAR  = pick_var(cbc,    ["LBXLYPCT","LBXLYPB","LBDLYPCT","LBXLY%","LBXLY"], "Lymphocyte %")

# ── STEP 2 — Merge on SEQN ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Merging on SEQN")
print("=" * 60)

# Keep only columns we need
demo_sub   = demo[[  "SEQN", AGE_VAR]].copy()
biopro_sub = biopro[["SEQN", ALB_VAR, CRE_VAR, GLU_VAR, ALP_VAR]].copy()
cbc_sub    = cbc[[   "SEQN", MCV_VAR, RDW_VAR, WBC_VAR, LYM_VAR]].copy()
crp_sub    = crp[[   "SEQN", CRP_VAR]].copy()

merged = (demo_sub
          .merge(biopro_sub, on="SEQN", how="inner")
          .merge(cbc_sub,    on="SEQN", how="inner")
          .merge(crp_sub,    on="SEQN", how="inner"))

print(f"  After inner merge: {merged.shape[0]:,} rows")

# Standardise column names
merged.rename(columns={
    AGE_VAR: "age",
    ALB_VAR: "albumin",
    CRE_VAR: "creatinine",
    GLU_VAR: "glucose",
    ALP_VAR: "alp",
    CRP_VAR: "crp_mgl",    # mg/L — formula expects mg/L then ln-transform
    MCV_VAR: "mcv",
    RDW_VAR: "rdw",
    WBC_VAR: "wbc",
    LYM_VAR: "lymph_pct",
}, inplace=True)

BIOMARKERS = ["albumin","creatinine","glucose","alp","crp_mgl","mcv","rdw","wbc","lymph_pct"]
ALL_VARS   = ["age"] + BIOMARKERS

# ── STEP 3 — Complete cases ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Filtering to complete cases")
print("=" * 60)

n_before = len(merged)
complete = merged.dropna(subset=ALL_VARS).copy()
n_after  = len(complete)
pct      = 100 * n_after / n_before

print(f"  Rows before filter : {n_before:,}")
print(f"  Rows after filter  : {n_after:,}  ({pct:.1f}% of merged sample)")

# ── Unit diagnostics ─────────────────────────────────────────────────────────
print("\n── Raw variable medians (for unit sanity check) ────────────────")
for col in ALL_VARS:
    med = complete[col].median()
    print(f"  {col:<14}: {med:.3f}")

# Glucose unit fix — the PhenoAge coefficient 0.1953 was derived from NHANES III
# where glucose was in mmol/L (NHANES reports the variable in mg/dL in later cycles).
# Dividing LBXSGL (mg/dL) by 18.02 converts to mmol/L.
gluc_median = complete["glucose"].median()
if gluc_median > 10:          # clearly mg/dL, not mmol/L
    print(f"\n  ⚠ UNIT FIX — glucose: {gluc_median:.1f} mg/dL → "
          f"{gluc_median/18.02:.2f} mmol/L")
    print("    Reason: PhenoAge coefficient 0.1953 is calibrated for glucose in mmol/L.")
    print("    Converting: glucose_mmol = LBXSGL / 18.02")
    complete["glucose"] = complete["glucose"] / 18.02

# CRP unit check — NHANES P_ cycle (LBXHSCRP) reports in mg/L; formula uses mg/L.
crp_median = complete["crp_mgl"].median()
print(f"\n  CRP median: {crp_median:.3f} mg/L (expect ~1-3 for healthy adult population)")
if crp_median < 0.1:
    print("  ⚠ CRP looks like mg/dL — multiplying × 10 to get mg/L")
    complete["crp_mgl"] = complete["crp_mgl"] * 10

# Guard: replace CRP ≤0 with LoD floor before log transform
n_zero_crp = (complete["crp_mgl"] <= 0).sum()
if n_zero_crp > 0:
    print(f"  ⚠ {n_zero_crp} rows with CRP ≤ 0 — replacing with 0.1 mg/L (LoD floor)")
    complete.loc[complete["crp_mgl"] <= 0, "crp_mgl"] = 0.1

# ── STEP 4 — PhenoAge formula ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Calculating PhenoAge")
print("=" * 60)

def calc_phenoage(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with intermediate xb, M, and final PhenoAge columns."""
    xb = (
        -19.907
        - 0.0336  * df["albumin"]
        + 0.0095  * df["creatinine"]
        + 0.1953  * df["glucose"]        # glucose must be in mmol/L
        + 0.0954  * np.log(df["crp_mgl"])  # CRP in mg/L
        - 0.0120  * df["lymph_pct"]
        + 0.0268  * df["mcv"]
        + 0.3306  * df["rdw"]
        + 0.00188 * df["alp"]
        + 0.0554  * df["wbc"]
        + 0.0804  * df["age"]
    )
    M  = 1 - np.exp(-1.51714 * np.exp(xb) / 0.0076927)
    # clip M away from 1 to avoid log(0); values hitting this clip are extreme outliers
    M  = M.clip(upper=1 - 1e-15)
    PA = 141.50 + np.log(-0.00553 * np.log(1 - M)) / 0.09165
    return xb, M, PA

xb_series, M_series, pa_series = calc_phenoage(complete)
complete["xb"]      = xb_series
complete["M"]       = M_series
complete["phenoage"] = pa_series

print(f"  xb      — median: {complete['xb'].median():.3f}  "
      f"range: [{complete['xb'].min():.2f}, {complete['xb'].max():.2f}]")
print(f"  M       — median: {complete['M'].median():.4f}  "
      f"range: [{complete['M'].min():.4f}, {complete['M'].max():.4f}]")
print(f"  PhenoAge — mean: {complete['phenoage'].mean():.1f} yr  "
      f"std: {complete['phenoage'].std():.1f} yr  "
      f"range: [{complete['phenoage'].min():.1f}, {complete['phenoage'].max():.1f}]")

# ── STEP 5 — Validation ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Validation")
print("=" * 60)

# 5a — Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(complete["age"], complete["phenoage"],
           alpha=0.15, s=5, color="steelblue", rasterized=True)
ax.set_xlabel("Chronological Age (years)")
ax.set_ylabel("PhenoAge (years)")
ax.set_title("PhenoAge vs Chronological Age — NHANES 2017-2020")

# Best-fit line
m, b = np.polyfit(complete["age"], complete["phenoage"], 1)
x_line = np.linspace(complete["age"].min(), complete["age"].max(), 200)
ax.plot(x_line, m * x_line + b, color="firebrick", linewidth=1.5, label=f"OLS y={m:.2f}x+{b:.1f}")
ax.legend(fontsize=9)
fig.tight_layout()
plot_path = OUT_DIR / "phenoage_vs_age_scatter.png"
fig.savefig(plot_path, dpi=150)
print(f"  Scatter plot saved → {plot_path}")

# 5b — Correlation (exclude non-finite rows)
finite_mask = np.isfinite(complete["phenoage"]) & np.isfinite(complete["age"])
r, p = stats.pearsonr(complete.loc[finite_mask, "age"],
                      complete.loc[finite_mask, "phenoage"])
print(f"\n  Pearson r (PhenoAge ~ chronological age) = {r:.4f}  (p = {p:.2e})")

# 5c — Implausible values
implausible = complete[
    (complete["phenoage"] < 0) |
    (complete["phenoage"] > 150) |
    (~np.isfinite(complete["phenoage"]))
]
print(f"\n  Implausible PhenoAge rows (< 0, > 150, or non-finite): {len(implausible)}")
if len(implausible) > 0:
    print("  Sample of implausible rows:")
    print(implausible[ALL_VARS + ["phenoage"]].head(5).to_string(index=False))

# 5d — Example rows
print("\n── 5d: 5 example rows ─────────────────────────────────────────────")
sample = complete[ALL_VARS + ["phenoage"]].sample(5, random_state=42)
print(sample.to_string(index=False))

# ── STEP 6 — Save output ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Saving complete-case dataset")
print("=" * 60)

out_csv = OUT_DIR / "phenoage_nhanes_complete.csv"
complete[["SEQN"] + ALL_VARS + ["phenoage"]].to_csv(out_csv, index=False)
# Note: 'glucose' column in the CSV is already in mmol/L after conversion
print(f"  Saved {len(complete):,} rows → {out_csv}")
print(f"  Columns: {list(complete[['SEQN']+ALL_VARS+['phenoage']].columns)}")

print("\nDone.")
