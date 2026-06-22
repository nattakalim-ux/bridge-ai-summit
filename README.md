# Mortality Risk Prediction Model (Inspired by Levine et al., 2018)

## Overview
โปรเจกต์นี้เป็นการพัฒนาระบบทำนาย **Mortality Risk** (ความเสี่ยงในการเสียชีวิต) โดยประยุกต์ใช้โมเดล Machine Learning (XGBoost) เพื่อสร้างแบบจำลองที่มีความแม่นยำและยืดหยุ่นสูงกว่าแบบจำลองทางสถิติดั้งเดิม โดยอ้างอิงพื้นฐานตัวแปรชีวภาพ (Biomarkers) 9 ชนิดตามแนวคิด **PhenoAge** ของ Levine et al. (2018) เพื่อประเมินสถานะสุขภาพและพยากรณ์ความเสี่ยงในเชิงลึก

## Project Directory Structure
โครงสร้างโปรเจกต์ถูกออกแบบเป็น Pipeline เพื่อรองรับการทำงานร่วมกัน:

.
├── Data/                   # เก็บข้อมูลดิบ (.dat, .XPT) และข้อมูลที่ประมวลผลแล้ว (.csv)
├── Model/                  # เก็บไฟล์โมเดล (.pkl) และไฟล์ Configuration (.json)
├── prepare_mortality_data.py
├── phase1_download_merge.py
├── phase2_create_target.py
├── phase3_calculate_phenoage.py
├── tune_xgboost.py
├── train_model.py
├── train_mortality_model.py
├── plot_calibration.py
├── plot_feature_importance.py
├── predict.py
├── requirements.txt
└── README.md

# File Functions Description

1. Data Preparation (การเตรียมข้อมูล)

**prepare_mortality_data.py**: ประมวลผลข้อมูลดิบจาก NHANES (2007-2010) ทำการคลีนข้อมูล (Clean) และกำหนด Target Variable คือสถานะการเสียชีวิต (Died/Survived) ภายในระยะเวลาติดตามผล

**phase1_download_merge.py**: ดึงข้อมูล NHANES ชุดใหม่ (2015-2016) เพื่อใช้เป็น Validation Set โดยแมปข้อมูลจากหลายไฟล์ (Demographic, Lab, Mortality) เข้าด้วยกัน

**phase2_create_target.py**: จัดรูปแบบคอลัมน์ผลลัพธ์ (Target) และคำนวณสถิติเบื้องต้น (Class Balance) เพื่อตรวจสอบความถูกต้องของข้อมูลก่อนนำไปใช้จริง

**phase3_calculate_phenoage.py**: คำนวณค่า Phenotypic Age ตามสูตรมาตรฐานของ Levine เพื่อใช้เป็นหนึ่งในตัวแปรต้นในการเปรียบเทียบประสิทธิภาพกับโมเดล Machine Learning

2. Model Training & Tuning (การจูนและเทรนโมเดล)
**tune_xgboost.py**: ใช้เทคนิค Bayesian Optimization (Optuna) เพื่อหา Hyperparameters ที่ดีที่สุดสำหรับโมเดล XGBoost

**train_model.py**: เทรนโมเดล PhenoAge แบบ Ensemble โดยแบ่งการเรียนรู้ตามกลุ่มอายุ (Young, Middle, Old) เพื่อเพิ่มความแม่นยำเฉพาะช่วงวัย

**train_mortality_model.py**: เทรนโมเดล XGBoost Classifier สำหรับทำนายความเสี่ยงการเสียชีวิต และทำ Probability Calibration ด้วย Isotonic Regression เพื่อให้ค่าความเสี่ยงมีความสมจริงเชิงการแพทย์

3. Evaluation & Visualization (การประเมินผล)
**plot_calibration.py**: สร้างกราฟ ROC Curve และ Calibration Curve เพื่อแสดงประสิทธิภาพการทำนายเทียบกับความเป็นจริง

**plot_feature_importance.py**: วิเคราะห์และแสดงภาพความสำคัญของตัวแปร (Feature Importance) เพื่อระบุว่าสารชีวภาพใดมีผลต่อความเสี่ยงมากที่สุด

4. Inference (การนำไปใช้)
predict.py: ไฟล์สำหรับทำนายผลโดยรับค่าผลตรวจเลือด แล้วคำนวณออกมาเป็นความเสี่ยงต่อการเสียชีวิตและอายุชีวภาพ

# Getting Started
Prerequisites
ตรวจสอบให้แน่ใจว่าเครื่องคอมพิวเตอร์ของคุณติดตั้ง Python 3.9+ แล้วรันคำสั่ง:

        Bash
        pip install -r requirements.txt

Execution Pipeline
เพื่อให้ระบบทำงานได้อย่างถูกต้อง ควรดำเนินการรันสคริปต์ตามลำดับ ดังนี้:

Data: prepare → phase1 → phase2 → phase3
Training: tune → train_model → train_mortality_model
Evaluation: plot_calibration → plot_feature_importance
Prediction: predict.py

Methodology Note
โปรเจกต์นี้มุ่งเน้นการตรวจจับความสัมพันธ์แบบ Non-linear ระหว่างตัวแปรทางชีวภาพ โดยใช้ XGBoost แทนสมการ Linear แบบดั้งเดิม ซึ่งช่วยให้คะแนนความเสี่ยง (Mortality Risk Score) มีความเป็นพลวัตและแม่นยำสูงขึ้น