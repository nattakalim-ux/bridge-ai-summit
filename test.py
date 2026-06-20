import joblib
import numpy as np

model = joblib.load("Model/mortality_model.pkl")

test_input = np.array([[4.0, 0.97, 90, 0.6, 35, 95, 14, 47, 4]])

prob = model.predict_proba(test_input)[0][1]
print(f"Mortality risk: {prob:.1%}")

if prob < 0.10:
    print("Risk tier: LOW")
elif prob < 0.25:
    print("Risk tier: MODERATE")
else:
    print("Risk tier: HIGH")
