"""
SHAP Explainability for the Diabetes DNN Model
Generates feature importance summary using SHAP KernelExplainer (sampled for speed)
"""
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import shap
import json

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load model + preprocessing
model = keras.models.load_model("diabetes_dnn_model.keras")
with open("preprocessing.pkl", "rb") as f:
    prep = pickle.load(f)

scaler = prep["scaler"]
feature_cols = prep["feature_cols"]

# Load data, take a background sample + explain sample (SHAP is slow on 100k rows)
df = pd.read_csv("diabetes_prediction_dataset.csv")
df['gender_enc'] = prep["gender_encoder"].transform(df['gender'])
df['smoking_enc'] = prep["smoking_encoder"].transform(df['smoking_history'])

X = df[feature_cols].values
X_scaled = scaler.transform(X)

# background: 100 random samples; explain: 200 random samples
rng = np.random.RandomState(RANDOM_STATE)
bg_idx = rng.choice(len(X_scaled), 100, replace=False)
explain_idx = rng.choice(len(X_scaled), 200, replace=False)

background = X_scaled[bg_idx]
X_explain = X_scaled[explain_idx]

def predict_fn(x):
    return model.predict(x, verbose=0).flatten()

print("Running SHAP KernelExplainer (this may take a minute)...")
explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(X_explain, nsamples=100)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = dict(zip(feature_cols, mean_abs_shap.tolist()))
shap_importance_sorted = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))

print("\nSHAP Mean |Impact| per feature:")
for k, v in shap_importance_sorted.items():
    print(f"  {k}: {v:.4f}")

with open("shap_importance.json", "w") as f:
    json.dump(shap_importance_sorted, f, indent=2)

print("\nSaved to shap_importance.json")
