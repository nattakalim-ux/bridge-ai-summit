# -*- coding: utf-8 -*-
"""
Phase 3: PhenoAge and Baseline Mortality Score Computation
Calculates biological age indicators to serve as a baseline for ML model comparison.
"""

import numpy as np
import pandas as pd
import os

print("=" * 70)
print(" PHASE 3: PHENOAGE COMPUTATION AND BASELINE MORTALITY SCORING")
print("=" * 70)

# Resolve absolute pathways to the local project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")

INPUT_CSV = os.path.join(PROJECT_DATA_DIR, "mortality_nhanes_complete_5yr.csv")
OUTPUT_CSV = os.path.join(PROJECT_DATA_DIR, "phenoage_completed_phase3.csv")

if not os.path.exists(INPUT_CSV):
    print(f"[ERROR] Target input file not found: {INPUT_CSV}")
    print("[STOP] Phase 3 execution halted due to missing upstream data.")
else:
    df = pd.read_csv(INPUT_CSV)
    print(f"[SUCCESS] Loaded dataset successfully: {len(df):,} rows")

    # Phase 2 Alignment: Establish outcome variables
    df["mortality_10yr"] = df["died_5yr"]
    df["follow_up_months"] = df["PERMTH_INT"]

    # ── SCIENTIFIC UNIT RE-ALIGNMENT ──

    # 1. Scale Glucose: Convert from mg/dL to mmol/L to prevent mathematical overflow
    glucose_mmol_l = df['glucose'] / 18.0

    # 2. Scale CRP: Adjust from mg/L to cm-relative scale for original Klemera-Doubal formulation
    crp_real = df['crp_mgl'] / 10.0
    crp_clean = np.where(crp_real <= 0, 0.001, crp_real)
    ln_crp = np.log(crp_clean)

    # 3. Scale Lymphocyte: Convert percentage value to an absolute cell count (10³ cells/µL)
    lymph_absolute = (df['wbc'] * df['lymph_pct']) / 100.0

    # Step 1: Compute linear predictor (xb) using standardized bio-clinical parameters
    df['xb'] = (
        -19.907
        - 0.0336  * df['albumin']
        + 0.0095  * df['creatinine']
        + 0.1953  * glucose_mmol_l
        + 0.0954  * ln_crp
        - 0.0120  * lymph_absolute
        + 0.0268  * df['mcv']
        + 0.3306  * df['rdw']
        + 0.00188 * df['alp']
        + 0.0554  * df['wbc']
        + 0.0804  * df['age']
    )

    # Step 2: Calculate parametric Mortality Score (M)
    raw_M = 1 - np.exp(-1.51714 * np.exp(df['xb']) / 0.0076927)
    df['M'] = np.clip(raw_M, 0.0001, 0.9999)

    # Step 3: Map parametric risk scores to clinical PhenotypicAge equivalent
    df['PhenotypicAge'] = 141.50 + (np.log(-0.00553 * np.log(1 - df['M'])) / 0.09165)
    df['PhenoAge_gap'] = df['PhenotypicAge'] - df['age']

    # Quality Control: Flag mathematically implausible estimates
    implausible_mask = (df['PhenotypicAge'] < 0) | (df['PhenotypicAge'] > 150)
    df['implausible_phenoage'] = np.where(implausible_mask, 1, 0)

    # Execution Summary Report
    print("\n" + "=" * 55)
    print(" PHASE 3 EXECUTION SUMMARY REPORT (SUCCESS)")
    print("=" * 55)
    print(f"  Total Sample Size (N):            {len(df):,}")
    print(f"  Mean Chronological Age:           {df['age'].mean():.2f} years")
    print(f"  Mean Phenotypic Age Baseline:     {df['PhenotypicAge'].mean():.2f} years")
    print(f"  Mean PhenoAge-Gap Discrepancy:    {df['PhenoAge_gap'].mean():.2f} years")
    print(f"  Implausible Estimates Flagged:    {implausible_mask.sum()} cases")
    print("=" * 55)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nOutput saved successfully to: {OUTPUT_CSV}")