# ============================================================
# GDHRA System — Google Colab Notebook (Part 1: Setup & Data)
# Generative Dynamic Health Risk Assessment System
# ============================================================
# Copy each cell block (marked with # %% [markdown] or # %%)
# into separate Colab cells.
# ============================================================

# %% [markdown]
# # 🏥 Generative Dynamic Health Risk Assessment System (GDHRA)
#
# **A research-grade, multimodal, uncertainty-aware health risk prediction system**
#
# ### Key Innovations:
# 1. **Cross-Modal Attention Fusion** — learnable modality reliability weighting
# 2. **Uncertainty-Aware Prediction** — Monte Carlo Dropout confidence intervals
# 3. **Generative Preventive Insight Engine** — Gemini API + medical RAG
# 4. **What-If Risk Simulator** — interactive parameter modification
# 5. **Temporal Risk Modeling** — LSTM-based health trajectory prediction
#
# ### Architecture:
# ```
# Dataset 1 (CVD)       → Tabular Branch (XGBoost/LightGBM)  ─┐
# Dataset 2 (MIMIC)     → Temporal Branch (BiLSTM+Attention)  ─┼→ Cross-Attention Fusion → Risk + Uncertainty
# Dataset 3 (Clinical)  → Text Branch (TF-IDF+Embedding)     ─┘
# ```

# %% — Cell 1: Install Dependencies
print("📦 Installing dependencies...")
import subprocess, sys
pkgs = [
    "pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
    "imbalanced-learn", "torch", "matplotlib", "plotly", "shap",
    "google-generativeai", "ipywidgets"
]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("✅ All dependencies installed!")

# %% — Cell 2: Imports
import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

import joblib
warnings.filterwarnings("ignore")

print("✅ All imports successful!")
print(f"   PyTorch: {torch.__version__}")
print(f"   Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 📊 Step 1: Dataset Generation
# We generate synthetic datasets simulating:
# 1. **CVD Dataset** — Cardiovascular disease structured EHR data
# 2. **MIMIC Temporal** — ICU temporal vitals/labs with patient deterioration
# 3. **Clinical Notes** — Synthetic clinical text with risk labels
# 4. **Wearable Data** — Synthetic wearable sensor signals

# %% — Cell 3: Generate CVD Dataset
def generate_synthetic_cvd(n_samples=5000):
    """Generate synthetic cardiovascular disease data matching Kaggle CVD schema."""
    np.random.seed(42)
    ages_days = np.random.randint(10000, 25000, n_samples)
    gender = np.random.choice([1, 2], n_samples)
    height = np.where(gender == 1,
                      np.random.normal(162, 8, n_samples),
                      np.random.normal(175, 9, n_samples)).astype(int)
    weight = np.where(gender == 1,
                      np.random.normal(70, 15, n_samples),
                      np.random.normal(82, 17, n_samples)).round(1)
    ap_hi = np.clip(np.random.normal(125, 18, n_samples).astype(int), 80, 220)
    ap_lo = np.clip(np.random.normal(80, 12, n_samples).astype(int), 50, 140)
    cholesterol = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
    gluc = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15])
    smoke = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    alco = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    active = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])

    age_years = ages_days / 365.25
    bmi = weight / ((height / 100) ** 2)
    risk_score = (
        (age_years - 40) * 0.03 + (ap_hi - 120) * 0.02 +
        (cholesterol - 1) * 0.3 + (gluc - 1) * 0.2 +
        smoke * 0.3 + (bmi - 25) * 0.02 + alco * 0.1 - active * 0.2
    )
    prob = 1 / (1 + np.exp(-risk_score))
    cardio = (np.random.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "id": range(1, n_samples + 1), "age": ages_days, "gender": gender,
        "height": height, "weight": weight, "ap_hi": ap_hi, "ap_lo": ap_lo,
        "cholesterol": cholesterol, "gluc": gluc, "smoke": smoke,
        "alco": alco, "active": active, "cardio": cardio
    })

cvd_df = generate_synthetic_cvd()
print(f"✅ CVD Dataset: {cvd_df.shape}")
print(f"   Positive rate: {cvd_df['cardio'].mean():.2%}")
cvd_df.head()

# %% — Cell 4: Generate MIMIC Temporal Dataset
def generate_mimic_temporal(n_patients=1000, max_visits=10):
    """Generate synthetic temporal EHR data simulating MIMIC-III/IV."""
    np.random.seed(123)
    records = []
    for pid in range(1, n_patients + 1):
        n_visits = np.random.randint(3, max_visits + 1)
        age = np.random.randint(25, 85)
        gender = np.random.choice([0, 1])
        base_hr = np.random.normal(75, 10)
        base_sbp = np.random.normal(125, 15)
        base_temp = np.random.normal(37.0, 0.3)
        base_resp = np.random.normal(16, 3)
        base_spo2 = np.random.normal(97, 1.5)
        base_glucose = np.random.normal(100, 25)
        base_creat = np.random.normal(1.0, 0.3)
        base_wbc = np.random.normal(7.5, 2.0)
        base_hgb = np.random.normal(13.5, 1.5)
        deteriorating = np.random.random() < 0.3

        for v in range(n_visits):
            t = v * 0.5 if deteriorating else 0
            records.append({
                "patient_id": pid, "visit_num": v,
                "hour_offset": v * np.random.randint(4, 48),
                "age": age, "gender": gender,
                "heart_rate": round(np.clip(base_hr + np.random.normal(0, 5) + t*2, 40, 180), 1),
                "systolic_bp": round(np.clip(base_sbp + np.random.normal(0, 8) - t*3, 70, 220), 1),
                "temperature": round(np.clip(base_temp + np.random.normal(0, 0.2) + t*0.1, 35, 41), 1),
                "respiratory_rate": round(np.clip(base_resp + np.random.normal(0, 2) + t, 8, 40), 1),
                "spo2": round(np.clip(base_spo2 + np.random.normal(0, 1) - t*0.5, 80, 100), 1),
                "glucose": round(np.clip(base_glucose + np.random.normal(0, 10) + t*5, 40, 400), 1),
                "creatinine": round(np.clip(base_creat + np.random.normal(0, 0.1) + t*0.05, 0.3, 10), 2),
                "wbc": round(np.clip(base_wbc + np.random.normal(0, 1) + t*0.5, 1, 30), 1),
                "hemoglobin": round(np.clip(base_hgb + np.random.normal(0, 0.5) - t*0.2, 5, 20), 1),
                "mortality": int(deteriorating and v == n_visits - 1)
            })
    return pd.DataFrame(records)

mimic_df = generate_mimic_temporal()
print(f"✅ MIMIC Temporal: {mimic_df.shape}")
print(f"   Patients: {mimic_df['patient_id'].nunique()}, Mortality rate: {mimic_df.groupby('patient_id')['mortality'].max().mean():.2%}")

# %% — Cell 5: Generate Clinical Notes
def generate_clinical_notes(n_patients=1000):
    """Generate synthetic clinical notes from patient characteristics."""
    np.random.seed(456)
    templates = {
        "low": [
            "Patient presents with stable vital signs. No acute complaints. {age}-year-old {gender} with normal blood pressure and regular heart rhythm.",
            "Follow-up visit. {age}-year-old {gender} reports feeling well. Well-controlled blood pressure. Routine screening normal.",
        ],
        "moderate": [
            "{age}-year-old {gender} presenting with elevated blood pressure {bp} and borderline cholesterol. Sedentary lifestyle. Recommend dietary modifications.",
            "{age}-year-old {gender} with pre-hypertension and elevated glucose. BMI {bmi}. Occasional chest discomfort with exertion.",
        ],
        "high": [
            "URGENT: {age}-year-old {gender} with significantly elevated BP {bp}, tachycardia, chest pain. History of smoking and diabetes. Elevated troponin.",
            "{age}-year-old {gender} with uncontrolled hypertension {bp}, diabetic complications, CKD stage 3. Creatinine elevated.",
        ]
    }
    records = []
    for pid in range(1, n_patients + 1):
        age = np.random.randint(25, 85)
        gender = np.random.choice(["male", "female"])
        bp = f"{np.random.randint(110, 200)}/{np.random.randint(60, 120)}"
        bmi = round(np.random.normal(28, 5), 1)
        risk_level = np.random.choice(["low", "moderate", "high"], p=[0.4, 0.35, 0.25])
        note = np.random.choice(templates[risk_level]).format(age=age, gender=gender, bp=bp, bmi=bmi)
        records.append({
            "patient_id": pid, "age": age, "gender": 1 if gender == "male" else 0,
            "clinical_note": note, "risk_level": risk_level,
            "risk_label": {"low": 0, "moderate": 1, "high": 2}[risk_level]
        })
    return pd.DataFrame(records)

notes_df = generate_clinical_notes()
print(f"✅ Clinical Notes: {notes_df.shape}")
print(f"   Risk distribution: {notes_df['risk_level'].value_counts().to_dict()}")

# %% [markdown]
# ## ✅ All Datasets Ready
# | Dataset | Shape | Purpose |
# |---------|-------|---------|
# | CVD | 5000 × 13 | Structured EHR (tabular branch) |
# | MIMIC Temporal | ~6500 × 15 | Temporal vitals/labs (LSTM branch) |
# | Clinical Notes | 1000 × 6 | Text modality (text branch) |

print("\n📊 Dataset Summary:")
print(f"  CVD:    {cvd_df.shape} — Tabular Branch")
print(f"  MIMIC:  {mimic_df.shape} — Temporal Branch")
print(f"  Notes:  {notes_df.shape} — Text Branch")
