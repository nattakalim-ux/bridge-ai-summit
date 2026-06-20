"""
Predict Biological Age and 10-Year Mortality Risk from blood test values.
Usage: python predict.py
"""

import joblib
import json
import numpy as np

MODEL_DIR = "/Users/ming/Desktop/MDCU/Model"

# ── Load bio-age models (stratified by age group) ────────────────────────────
with open(f"{MODEL_DIR}/model_meta.json") as f:
    bio_meta = json.load(f)

AGE_GROUPS  = bio_meta["age_groups"]         # {"young":[0,45], "middle":[45,65], ...}
FEATURES    = bio_meta["features"]
LOG_FEATURES = set(bio_meta["log_features"])

bio_models = {
    name: joblib.load(f"{MODEL_DIR}/phenoage_model_{name}.pkl")
    for name in AGE_GROUPS
}

# ── Load mortality model + calibrator ────────────────────────────────────────
with open(f"{MODEL_DIR}/mortality_meta.json") as f:
    mort_meta = json.load(f)

_mort_base = joblib.load(f"{MODEL_DIR}/mortality_base_model.pkl")
_mort_ir   = joblib.load(f"{MODEL_DIR}/mortality_ir.pkl")

class _MortalityModel:
    """Thin wrapper: XGBoost raw probs → isotonic calibration."""
    def predict_proba(self, X):
        raw = _mort_base.predict_proba(X)[:, 1]
        cal = _mort_ir.predict(raw)
        return np.column_stack([1 - cal, cal])

mortality_model = _MortalityModel()
RISK_THRESHOLDS = mort_meta["risk_thresholds"]  # {"low_to_moderate":0.10, ...}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _preprocess(values: dict) -> np.ndarray:
    """Apply log1p to skewed features, return (1, 9) array."""
    return np.array([[
        np.log1p(values[f]) if f in LOG_FEATURES else values[f]
        for f in FEATURES
    ]])


def _age_group(age: float) -> str:
    for name, (lo, hi) in AGE_GROUPS.items():
        if lo <= age < hi:
            return name
    return "old"


def _risk_label(prob: float) -> str:
    lo = RISK_THRESHOLDS["low_to_moderate"]
    hi = RISK_THRESHOLDS["moderate_to_high"]
    if prob < lo:  return "Low"
    if prob < hi:  return "Moderate"
    return "High"


# ── Public API ────────────────────────────────────────────────────────────────
def predict_biological_age(values: dict, age: float) -> dict:
    """
    values : dict of 9 raw lab values
    age    : chronological age (for routing to the right model)
    returns: {"biological_age": float, "group": str, "mae": float}
    """
    group = _age_group(age)
    x     = _preprocess(values)
    bio_age = float(bio_models[group].predict(x)[0])
    mae     = bio_meta["group_results"][group]["MAE"]
    return {"biological_age": round(bio_age, 1), "group": group, "mae": mae}


def predict_mortality_risk(values: dict) -> dict:
    """
    values : dict of 9 raw lab values
    returns: {"risk_prob": float, "risk_label": str}
    """
    x    = _preprocess(values)
    prob = float(mortality_model.predict_proba(x)[0, 1])
    return {"risk_prob": round(prob, 4), "risk_label": _risk_label(prob)}


def predict_full(values: dict, age: float) -> dict:
    """Run both models and return combined result."""
    bio  = predict_biological_age(values, age)
    mort = predict_mortality_risk(values)
    return {
        "chronological_age":  age,
        "biological_age":     bio["biological_age"],
        "age_difference":     round(bio["biological_age"] - age, 1),
        "model_group":        bio["group"],
        "model_mae":          bio["mae"],
        "mortality_10yr_prob": mort["risk_prob"],
        "mortality_risk":     mort["risk_label"],
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLES = [
        ("Average healthy adult, 45 yrs", 45.0, {
            "albumin": 4.1, "creatinine": 0.85, "glucose": 5.1,
            "alp": 77.0,  "crp_mgl": 1.7,   "mcv": 88.4,
            "rdw": 13.5,  "wbc": 6.9,         "lymph_pct": 31.2,
        }),
        ("High inflammation, 60 yrs", 60.0, {
            "albumin": 3.5, "creatinine": 1.2,  "glucose": 6.8,
            "alp": 110.0, "crp_mgl": 25.0,  "mcv": 92.0,
            "rdw": 15.0,  "wbc": 10.5,        "lymph_pct": 22.0,
        }),
        ("Elderly with poor kidney, 72 yrs", 72.0, {
            "albumin": 3.2, "creatinine": 2.1,  "glucose": 7.5,
            "alp": 95.0,  "crp_mgl": 8.0,   "mcv": 95.0,
            "rdw": 16.5,  "wbc": 8.2,         "lymph_pct": 18.0,
        }),
    ]

    print("=" * 60)
    for label, age, values in SAMPLES:
        r = predict_full(values, age)
        diff = r["age_difference"]
        sign = "+" if diff >= 0 else ""
        print(f"\n{label}")
        print(f"  Chronological age : {r['chronological_age']:.0f} yrs")
        print(f"  Biological age    : {r['biological_age']} yrs  ({sign}{diff} yrs)")
        print(f"  Model MAE         : ±{r['model_mae']:.1f} yrs")
        print(f"  10-yr mortality   : {r['mortality_10yr_prob']*100:.1f}%  [{r['mortality_risk']}]")
    print("\n" + "=" * 60)
    print(f"\nBio-age model  → R²={bio_meta['ensemble']['R2']}  "
          f"MAE={bio_meta['ensemble']['MAE']} yrs")
    print(f"Mortality model → AUC={mort_meta['metrics']['AUC_ROC']}  "
          f"Brier={mort_meta['metrics']['Brier']}")
