"""
NHANES Raw Data Downloader and Verifier
Uses the verified URL structure to fetch raw .XPT files directly from CDC servers.
"""

import os
import urllib.request
import ssl
import sys

# 1. Setup target directory pathways
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR = os.path.join(BASE_DIR, "Data", "nhanes_raw")
os.makedirs(DEST_DIR, exist_ok=True)

# 2. Verified URL repository for NHANES Cycle E, Cycle F, and Cycle I
XPT_URLS = {
    # NHANES 2007-2008 (Cycle E)
    "DEMO_E.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/DEMO_E.xpt",
    "BIOPRO_E.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/BIOPRO_E.xpt",
    "CRP_E.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CRP_E.xpt",
    "CBC_E.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles/CBC_E.xpt",
    
    # NHANES 2009-2010 (Cycle F)
    "DEMO_F.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/DEMO_F.xpt",
    "BIOPRO_F.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/BIOPRO_F.xpt",
    "CRP_F.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/CRP_F.xpt",
    "CBC_F.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles/CBC_F.xpt",
    
    # NHANES 2015-2016 (Cycle I)
    "DEMO_I.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/DEMO_I.xpt",
    "BIOPRO_I.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/BIOPRO_I.xpt",
    "HSCRP_I.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/HSCRP_I.xpt",
    "GLU_I.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/GLU_I.xpt",
    "CBC_I.XPT": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/CBC_I.xpt"
}

def download_file(filename, url):
    target_path = os.path.join(DEST_DIR, filename)
    
    # Skip download if the file exists and its size is valid (> 50 KB)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 50 * 1024:
        print(f"[SKIPPED] File '{filename}' already exists and is valid.")
        return True

    # Remove corrupted files from previous failed script attempts
    if os.path.exists(target_path):
        os.remove(target_path)

    print(f"[DOWNLOADING] Fetching data for: {filename}...")
    
    # Bypass local SSL certificate verification issues if applicable
    context = ssl._create_unverified_context()
    
    # Configure request headers to mimic a standard browser request session
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=context, timeout=45) as response:
            data = response.read()
            
            # Content Integrity Verification: Detect firewall HTML responses
            if data.startswith(b"<!DOCTYPE") or data.startswith(b"<html"):
                print(f"[BLOCKED] CDC server restricted programmatic access for {filename}. (HTML response returned)")
                return False
                
            with open(target_path, 'wb') as f:
                f.write(data)
                
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[SUCCESS] Download completed for '{filename}' | Size: {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return False

def main():
    print("=" * 70)
    print(" NHANES Dataset Downloader: Standardized Data Acquisition Pipeline")
    print(f" Target Directory: {DEST_DIR}")
    print("=" * 70)
    
    success_count = 0
    for filename, url in XPT_URLS.items():
        if download_file(filename, url):
            success_count += 1
            
    print("\n" + "=" * 70)
    print(f" Pipeline Execution Summary: {success_count}/{len(XPT_URLS)} files acquired.")
    print("=" * 70)
    
    if success_count == len(XPT_URLS):
        print(" Status: Environment preparation completed successfully.")
        print(" Next Step: You may now execute the pipeline via 'python run_pipeline.py'.")
    else:
        print(" Status: Pipeline terminated due to data ingestion errors.")
        print(" Recommendation: Verify network configurations and re-execute the script.")
        sys.exit(1)

if __name__ == "__main__":
    main()