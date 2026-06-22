import pandas as pd
import os
from config import DATA_DIR

# Resolve the absolute path to the project root directory (stepping out of 'Scripts')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA_DIR = os.path.join(BASE_DIR, "Data")

# Define standardized input and output file pathways
INPUT_FILE = os.path.join(PROJECT_DATA_DIR, "phase1_merged_5yr.csv")
OUTPUT_FILE = os.path.join(PROJECT_DATA_DIR, "mortality_nhanes_complete_5yr.csv")

# 3. เช็คความพร้อมของโฟลเดอร์และไฟล์
if not os.path.exists(DATA_DIR):
    print(f"Error: ไม่พบโฟลเดอร์ {DATA_DIR}")
    exit()

if not os.path.exists(INPUT_FILE):
    print(f"Error: ไม่พบไฟล์ {INPUT_FILE} กรุณาตรวจสอบว่ารัน Phase 1 จบแล้ว")
    exit()

# 4. โหลดข้อมูล
df = pd.read_csv(INPUT_FILE)
print(f"Loaded dataset: {len(df):,} rows")

# 5. จัดการ Target Variable (ปรับปรุงให้สอดคล้องกับ 5 ปี)
df["mortality_target"] = df["died_5yr"]
df["follow_up_months"] = df["PERMTH_INT"]

# 6. แสดงผลการวิเคราะห์
n_total = len(df)
n_dead = df["mortality_target"].sum()
n_survived = n_total - n_dead

print("\n[Report: Class Balance (5-Year Mortality)]")
print(f"  Total participants:       {n_total:,}")
print(f"  Survived 5 years (0):     {n_survived:,} ({(n_survived/n_total)*100:.2f}%)")
print(f"  Died within 5 years (1):  {n_dead:,} ({(n_dead/n_total)*100:.2f}%)")

print("\n[Report: Follow-up Months]")
print(f"  Min:    {df['follow_up_months'].min():.0f} months")
print(f"  Median: {df['follow_up_months'].median():.0f} months")
print(f"  Max:    {df['follow_up_months'].max():.0f} months")

# 7. เซฟไฟล์โดยใช้ตัวแปรที่ประกาศไว้
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nPhase 2 complete. File saved to: {OUTPUT_FILE}")