"""
Mortality Risk Prediction Pipeline Orchestrator
Automates data acquisition, preprocessing, model training, and evaluation.
"""

import subprocess
import sys
import os
import time

# Resolve absolute pathways to prevent directory execution mismatches
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "Scripts")

# Execution order of pipeline components
PIPELINE_SCRIPTS = [
    os.path.join(BASE_DIR, "download_raw_data.py"), # Located at Root
    os.path.join(SCRIPTS_DIR, "prepare_mortality_data.py"), # Inside Scripts/
    os.path.join(SCRIPTS_DIR, "phase1_download_merge.py"),
    os.path.join(SCRIPTS_DIR, "phase2_create_target.py"),
    os.path.join(SCRIPTS_DIR, "phase3_calculate_phenoage.py"),
    os.path.join(SCRIPTS_DIR, "tune_xgboost.py"),
    os.path.join(SCRIPTS_DIR, "train_model.py"),
    os.path.join(SCRIPTS_DIR, "train_mortality_model.py"),
    os.path.join(SCRIPTS_DIR, "plot_calibration.py"),
    os.path.join(SCRIPTS_DIR, "plot_feature_importance.py")
]

def run_script(script_path):
    script_name = os.path.basename(script_path)
    print("=" * 70)
    print(f"[Pipeline] Initiating script execution: {script_name}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Execute script using the current active Python environment executable
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"[SUCCESS] {script_name} completed execution in {duration:.2f} seconds.\n")
        return True
    else:
        print(f"[FAILURE] {script_name} terminated with Exit Code: {result.returncode}\n")
        return False

def main():
    print("=" * 70)
    print(" Execution initiated: Mortality Risk Prediction Automated Pipeline")
    print("=" * 70)
    
    pipeline_start = time.time()
    
    for script_path in PIPELINE_SCRIPTS:
        # Prevent failure if a script path cannot be found prior to execution
        if not os.path.exists(script_path):
            print(f"[ERROR] Target script file not found: {script_path}")
            print("[STOP] Pipeline pipeline execution halted due to missing assets.")
            sys.exit(1)
            
        success = run_script(script_path)
        if not success:
            print(f"[STOP] Pipeline pipeline execution halted due to error in: {os.path.basename(script_path)}")
            print("Action Required: Please debug the script error displayed above before re-running.")
            sys.exit(1)
            
    total_duration = time.time() - pipeline_start
    
    print("=" * 70)
    print("[SUCCESS] All pipeline modules completed successfully.")
    print(f"Total Pipeline Runtime: {total_duration/60:.2f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()