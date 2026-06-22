"""
Build mortality dataset from NHANES 2007-2010.
- Blood markers: same 9 features as PhenoAge
- Target: died within 10 years (binary)
- Follow-up: interview date → Dec 31 2019 (11-12 yrs for 2007-08, 9-10 for 2009-10)
"""

import pandas as pd
import numpy as np
import os

# ── Path: ถอยออกจากโฟลเดอร์ Scripts ไปยังโฟลเดอร์หลักของโปรเจกต์ ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
RAW = os.path.join(DATA_DIR, "nhanes_raw")

# ── Helper: parse CDC fixed-width mortality file ─────────────────────────────
def parse_mortality(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n").ljust(48)
            seqn      = line[0:14].strip()
            eligstat  = line[14:15].strip()
            mortstat  = line[15:16].strip()
            permth    = line[45:48].strip()
            rows.append({
                "SEQN":      float(seqn),
                "ELIGSTAT":  eligstat,
                "MORTSTAT":  mortstat,
                "PERMTH_INT": float(permth) if permth.lstrip("-").isdigit() else np.nan,
            })
    return pd.DataFrame(rows)


# ── Helper: load one NHANES cycle ────────────────────────────────────────────
def load_cycle(suffix, mort_path):
    demo  = pd.read_sas(f"{RAW}/DEMO_{suffix}.XPT",  format="xport",
                        encoding="latin-1")[["SEQN", "RIDAGEYR"]]
    bio   = pd.read_sas(f"{RAW}/BIOPRO_{suffix}.XPT", format="xport",
                        encoding="latin-1")[["SEQN", "LBXSAL", "LBXSCR",
                                             "LBDSGLSI", "LBXSAPSI"]]
    crp   = pd.read_sas(f"{RAW}/CRP_{suffix}.XPT",   format="xport",
                        encoding="latin-1")[["SEQN", "LBXCRP"]]
    cbc   = pd.read_sas(f"{RAW}/CBC_{suffix}.XPT",   format="xport",
                        encoding="latin-1")[["SEQN", "LBXMCVSI", "LBXRDW",
                                             "LBXWBCSI", "LBXLYPCT"]]
    mort  = parse_mortality(mort_path)

    df = (demo
          .merge(bio,  on="SEQN", how="inner")
          .merge(crp,  on="SEQN", how="inner")
          .merge(cbc,  on="SEQN", how="inner")
          .merge(mort, on="SEQN", how="inner"))

    df["cycle"] = suffix
    return df


# ── Load both cycles ──────────────────────────────────────────────────────────
# ตรวจสอบความพร้อมของไฟล์ข้อมูลก่อนเริ่มรันย่อย
if not os.path.exists(RAW):
    print(f"❌ Error: ไม่พบโฟลเดอร์ {RAW} กรุณาตรวจสอบตำแหน่งโฟลเดอร์ nhanes_raw")
    exit()

e = load_cycle("E", os.path.join(RAW, "NHANES_2007_2008_MORT_2019_PUBLIC.dat"))
f = load_cycle("F", os.path.join(RAW, "NHANES_2009_2010_MORT_2019_PUBLIC.dat"))
df = pd.concat([e, f], ignore_index=True)
print(f"Combined rows before filtering: {len(df):,}")

# ── Keep only eligible adults (≥18, eligstat=1) ───────────────────────────────
df = df[(df["ELIGSTAT"] == "1") & (df["RIDAGEYR"] >= 18)].copy()
print(f"After eligibility filter:       {len(df):,}")

# ── Standardise column names & units (แก้ไขชื่อคอลัมน์ให้ตรงตาม config.py) ──
df = df.rename(columns={
    "RIDAGEYR":  "age",
    "LBXSAL":    "albumin",       # g/dL
    "LBXSCR":    "creatinine",    # mg/dL
    "LBDSGLSI":  "glucose",       # mmol/L
    "LBXSAPSI":  "alp",           # U/L
    "LBXCRP":    "crp",           # เปลี่ยนจาก crp_mgl -> crp เพื่อให้ตรงคู่มือกลาง
    "LBXMCVSI":  "mcv",           # fL
    "LBXRDW":    "rdw",           # %
    "LBXWBCSI":  "wbc",           # 10³/μL
    "LBXLYPCT":  "lymphocyte",    # เปลี่ยนจาก lymph_pct -> lymphocyte เพื่อให้ตรงคู่มือกลาง
})

# CRP: LBXCRP คือ mg/dL → × 10 → mg/L
df["crp"] = df["crp"] * 10

# ── Build 10-year mortality target (แก้ไขชื่อตัวแปรจาก 5yr เป็น 10yr ให้ถูกต้อง) ──
df["died_5yr"] = (
    (df["MORTSTAT"] == "1") & (df["PERMTH_INT"] <= 120)
).astype(int)

# Drop rows where follow-up is unknown (MORTSTAT missing)
df = df[df["MORTSTAT"] != "."].copy()
print(f"After removing missing MORTSTAT: {len(df):,}")

FEATURES = ["albumin", "creatinine", "glucose", "alp", "crp", "mcv", "rdw", "wbc", "lymphocyte"]

# Drop rows with any missing lab value
df_clean = df[["SEQN", "age", "cycle", "died_5yr", "PERMTH_INT"] + FEATURES].dropna()
print(f"After dropping missing labs:     {len(df_clean):,}")

# Summary 
n_dead = df_clean["died_5yr"].sum()
n_total = len(df_clean)
print(f"\n10-yr mortality: {n_dead:,} dead / {n_total:,} total  "
      f"({100*n_dead/n_total:.1f}%)")
print(f"Age range: {df_clean['age'].min():.0f} – {df_clean['age'].max():.0f} yrs")

by_cycle = df_clean.groupby("cycle")["died_5yr"].agg(["sum","count"])
by_cycle["pct"] = (by_cycle["sum"]/by_cycle["count"]*100).round(1)
print(f"\nBy cycle:\n{by_cycle.to_string()}")

print("\nFeature stats:")
print(df_clean[FEATURES].describe().round(2).to_string())

# ── Save ผลลัพธ์ไปยังโฟลเดอร์ Data หลักข้างนอก
out = os.path.join(DATA_DIR, "mortality_nhanes_complete.csv")
df_clean.to_csv(out, index=False)
print(f"\nSaved → {out}")