import pandas as pd
import os

INPUT_CSV = "d:/bridge-ai-summit-main/Data/merged_nhanes_2015_2016_with_mortality.csv"
OUTPUT_CSV = "d:/bridge-ai-summit-main/Data/phenoage_ready_for_ml.csv"

if not os.path.exists(INPUT_CSV):
    print(f"Error: ไม่พบไฟล์ {INPUT_CSV} กรุณาตรวจสอบโฟลเดอร์ Data")
    exit()

df = pd.read_csv(INPUT_CSV)
print(f"Loaded dataset: {len(df):,} rows")

df["mortality_10yr"] = df["died_10yr"]
df["follow_up_months"] = df["PERMTH_INT"]

n_total = len(df)
n_dead = df["mortality_10yr"].sum()
n_survived = n_total - n_dead

print("\n[Report: Class Balance]")
print(f"  Total participants:       {n_total:,}")
print(f"  Survived 10 years (0):    {n_survived:,} ({(n_survived/n_total)*100:.2f}%)")
print(f"  Died within 10 years (1): {n_dead:,} ({(n_dead/n_total)*100:.2f}%)")

print("\n[Report: Follow-up Months]")
print(f"  Min:    {df['follow_up_months'].min():.0f} months")
print(f"  Median: {df['follow_up_months'].median():.0f} months")
print(f"  Max:    {df['follow_up_months'].max():.0f} months")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nPhase 2 complete. File saved to: {OUTPUT_CSV}")