import os

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
MODEL_DIR = os.path.join(BASE_DIR, "Model")   # root/Model/
DATA_DIR  = os.path.join(BASE_DIR, "Data")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
