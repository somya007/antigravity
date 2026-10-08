"""
Dataset Download & Synthetic Data Generation
=============================================
Downloads public health datasets and generates synthetic data for modalities
that require protected/unavailable datasets.

Datasets:
  1. Cardiovascular Disease (Kaggle) - structured clinical features
  2. Synthetic MIMIC-like EHR data - temporal vitals/labs
  3. Synthetic clinical notes - text modality
  4. Synthetic wearable data - time-series signals
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def ensure_data_dir():
    """Create data/raw/ directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[INFO] Data directory: {DATA_DIR}")


# ─────────────────────────────────────────────
# 1. Cardiovascular Disease Dataset
# ─────────────────────────────────────────────

def download_cvd_dataset():
    """
    Download Cardiovascular Disease dataset.
    Source: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    
    If the file already exists or download fails, generates synthetic CVD data.
    """
    filepath = os.path.join(DATA_DIR, "cardio_train.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, sep=";")
        print(f"[OK] CVD dataset already exists: {df.shape}")
        return df

    print("[INFO] CVD dataset not found locally. Generating synthetic CVD data...")
    df = generate_synthetic_cvd()
    df.to_csv(filepath, index=False, sep=";")
    print(f"[OK] Synthetic CVD dataset generated: {df.shape}")
    return df


def generate_synthetic_cvd(n_samples=5000):
    """Generate synthetic cardiovascular disease data matching Kaggle schema."""
    np.random.seed(42)
    
    ages_days = np.random.randint(10000, 25000, n_samples)  # age in days
    gender = np.random.choice([1, 2], n_samples)  # 1=female, 2=male
    height = np.where(gender == 1,
                      np.random.normal(162, 8, n_samples),
                      np.random.normal(175, 9, n_samples)).astype(int)
    weight = np.where(gender == 1,
                      np.random.normal(70, 15, n_samples),
                      np.random.normal(82, 17, n_samples)).astype(float).round(1)
    
    # Systolic/diastolic blood pressure
    ap_hi = np.random.normal(125, 18, n_samples).astype(int)
    ap_lo = np.random.normal(80, 12, n_samples).astype(int)
    ap_hi = np.clip(ap_hi, 80, 220)
    ap_lo = np.clip(ap_lo, 50, 140)
    
    cholesterol = np.random.choice([1, 2, 3], n_samples, p=[0.5, 0.3, 0.2])
    gluc = np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.25, 0.15])
    smoke = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    alco = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    active = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    
    # Target: cardiovascular disease (correlated with risk factors)
    age_years = ages_days / 365.25
    bmi = weight / ((height / 100) ** 2)
    risk_score = (
        (age_years - 40) * 0.03 +
        (ap_hi - 120) * 0.02 +
        (cholesterol - 1) * 0.3 +
        (gluc - 1) * 0.2 +
        smoke * 0.3 +
        (bmi - 25) * 0.02 +
        alco * 0.1 -
        active * 0.2
    )
    prob = 1 / (1 + np.exp(-risk_score))
    cardio = (np.random.random(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        "id": range(1, n_samples + 1),
        "age": ages_days,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "cardio": cardio
    })
    return df


# ─────────────────────────────────────────────
# 2. Synthetic MIMIC-like Temporal EHR Data
# ─────────────────────────────────────────────

def generate_mimic_temporal(n_patients=1000, max_visits=10):
    """
    Generate synthetic temporal EHR data simulating MIMIC-III/IV structure.
    Each patient has multiple time-stamped visits with vitals and lab values.
    """
    filepath = os.path.join(DATA_DIR, "mimic_temporal.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[OK] MIMIC temporal data already exists: {df.shape}")
        return df

    print("[INFO] Generating synthetic MIMIC-like temporal data...")
    np.random.seed(123)
    
    records = []
    for patient_id in range(1, n_patients + 1):
        n_visits = np.random.randint(3, max_visits + 1)
        age = np.random.randint(25, 85)
        gender = np.random.choice([0, 1])
        
        # Base vitals with patient-specific variation
        base_hr = np.random.normal(75, 10)
        base_sbp = np.random.normal(125, 15)
        base_temp = np.random.normal(37.0, 0.3)
        base_resp = np.random.normal(16, 3)
        base_spo2 = np.random.normal(97, 1.5)
        
        # Base labs
        base_glucose = np.random.normal(100, 25)
        base_creatinine = np.random.normal(1.0, 0.3)
        base_wbc = np.random.normal(7.5, 2.0)
        base_hemoglobin = np.random.normal(13.5, 1.5)
        
        # Determine if patient deteriorates over time
        deteriorating = np.random.random() < 0.3
        
        for visit in range(n_visits):
            hour_offset = visit * np.random.randint(4, 48)
            
            # Add trend for deteriorating patients
            trend = visit * 0.5 if deteriorating else 0
            
            hr = base_hr + np.random.normal(0, 5) + trend * 2
            sbp = base_sbp + np.random.normal(0, 8) - trend * 3
            temp = base_temp + np.random.normal(0, 0.2) + trend * 0.1
            resp = base_resp + np.random.normal(0, 2) + trend * 1
            spo2 = base_spo2 + np.random.normal(0, 1) - trend * 0.5
            glucose = base_glucose + np.random.normal(0, 10) + trend * 5
            creatinine = base_creatinine + np.random.normal(0, 0.1) + trend * 0.05
            wbc = base_wbc + np.random.normal(0, 1) + trend * 0.5
            hemoglobin = base_hemoglobin + np.random.normal(0, 0.5) - trend * 0.2
            
            records.append({
                "patient_id": patient_id,
                "visit_num": visit,
                "hour_offset": hour_offset,
                "age": age,
                "gender": gender,
                "heart_rate": round(np.clip(hr, 40, 180), 1),
                "systolic_bp": round(np.clip(sbp, 70, 220), 1),
                "temperature": round(np.clip(temp, 35, 41), 1),
                "respiratory_rate": round(np.clip(resp, 8, 40), 1),
                "spo2": round(np.clip(spo2, 80, 100), 1),
                "glucose": round(np.clip(glucose, 40, 400), 1),
                "creatinine": round(np.clip(creatinine, 0.3, 10), 2),
                "wbc": round(np.clip(wbc, 1, 30), 1),
                "hemoglobin": round(np.clip(hemoglobin, 5, 20), 1),
                "mortality": int(deteriorating and visit == n_visits - 1)
            })
    
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"[OK] Synthetic MIMIC temporal data generated: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 3. Synthetic Clinical Notes
# ─────────────────────────────────────────────

def generate_clinical_notes(n_patients=1000):
    """Generate synthetic clinical notes from patient characteristics."""
    filepath = os.path.join(DATA_DIR, "clinical_notes.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[OK] Clinical notes already exist: {df.shape}")
        return df

    print("[INFO] Generating synthetic clinical notes...")
    np.random.seed(456)
    
    templates_healthy = [
        "Patient presents with stable vital signs. No acute complaints. {age}-year-old {gender} with normal blood pressure and regular heart rhythm. Routine labs within normal limits. Continue current management.",
        "Follow-up visit. Patient reports feeling well. {age}-year-old {gender} with well-controlled blood pressure. Labs show normal glucose and cholesterol levels. No medication changes needed.",
        "Annual wellness exam for {age}-year-old {gender}. BMI within normal range. Blood pressure {bp}. Active lifestyle, non-smoker. Routine screening normal. Continue preventive care.",
    ]
    
    templates_moderate = [
        "Patient is a {age}-year-old {gender} presenting with elevated blood pressure {bp} and borderline cholesterol. History of sedentary lifestyle. Recommend dietary modifications and increased physical activity. Follow-up in 3 months.",
        "{age}-year-old {gender} with pre-hypertension and elevated glucose. BMI {bmi}. Reports occasional chest discomfort with exertion. Ordered cardiac workup. Started on lifestyle modification program.",
        "Follow-up for {age}-year-old {gender} with metabolic syndrome risk factors. Blood pressure {bp}, glucose elevated. Mild obesity with BMI {bmi}. Initiated dietary counseling and exercise program.",
    ]
    
    templates_high_risk = [
        "URGENT: {age}-year-old {gender} presenting with significantly elevated blood pressure {bp}, tachycardia, and chest pain. History of smoking and diabetes. Elevated troponin. Admit for cardiac monitoring and further evaluation.",
        "{age}-year-old {gender} with uncontrolled hypertension {bp}, diabetic complications, and chronic kidney disease stage 3. Creatinine elevated. Adjusting medications. Nephrology referral placed.",
        "Critical: {age}-year-old {gender} with acute decompensation. Blood pressure {bp}, respiratory distress, elevated WBC. History of heart failure and diabetes. Initiated aggressive management protocol.",
    ]
    
    records = []
    for pid in range(1, n_patients + 1):
        age = np.random.randint(25, 85)
        gender = np.random.choice(["male", "female"])
        bp = f"{np.random.randint(110, 200)}/{np.random.randint(60, 120)}"
        bmi = round(np.random.normal(28, 5), 1)
        
        risk_level = np.random.choice(["low", "moderate", "high"], p=[0.4, 0.35, 0.25])
        
        if risk_level == "low":
            template = np.random.choice(templates_healthy)
        elif risk_level == "moderate":
            template = np.random.choice(templates_moderate)
        else:
            template = np.random.choice(templates_high_risk)
        
        note = template.format(age=age, gender=gender, bp=bp, bmi=bmi)
        
        risk_label = {"low": 0, "moderate": 1, "high": 2}[risk_level]
        
        records.append({
            "patient_id": pid,
            "age": age,
            "gender": 1 if gender == "male" else 0,
            "clinical_note": note,
            "risk_level": risk_level,
            "risk_label": risk_label
        })
    
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"[OK] Synthetic clinical notes generated: {df.shape}")
    return df


# ─────────────────────────────────────────────
# 4. Synthetic Wearable Time-Series Data
# ─────────────────────────────────────────────

def generate_wearable_data(n_patients=500, days=7):
    """Generate synthetic wearable sensor data (heart rate, steps, sleep)."""
    filepath = os.path.join(DATA_DIR, "wearable_data.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"[OK] Wearable data already exists: {df.shape}")
        return df

    print("[INFO] Generating synthetic wearable data...")
    np.random.seed(789)
    
    records = []
    for pid in range(1, n_patients + 1):
        base_rhr = np.random.normal(70, 10)  # resting heart rate
        base_steps = np.random.normal(8000, 3000)
        base_sleep = np.random.normal(7, 1.5)
        is_unhealthy = np.random.random() < 0.3
        
        for day in range(days):
            rhr = base_rhr + np.random.normal(0, 3) + (5 if is_unhealthy else 0)
            steps = max(0, base_steps + np.random.normal(0, 1000) - (3000 if is_unhealthy else 0))
            sleep = max(2, base_sleep + np.random.normal(0, 0.5) - (1.5 if is_unhealthy else 0))
            hrv = max(10, np.random.normal(50, 15) - (15 if is_unhealthy else 0))
            
            records.append({
                "patient_id": pid,
                "day": day,
                "resting_hr": round(np.clip(rhr, 45, 120), 1),
                "daily_steps": int(np.clip(steps, 0, 30000)),
                "sleep_hours": round(np.clip(sleep, 2, 12), 1),
                "hrv_ms": round(np.clip(hrv, 10, 100), 1),
                "health_risk": 1 if is_unhealthy else 0
            })
    
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)
    print(f"[OK] Synthetic wearable data generated: {df.shape}")
    return df


# ─────────────────────────────────────────────
# Main: Run all downloads/generation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("GDHRA System — Dataset Preparation")
    print("=" * 60)
    
    ensure_data_dir()
    
    print("\n--- Dataset 1: Cardiovascular Disease ---")
    cvd = download_cvd_dataset()
    
    print("\n--- Dataset 2: Synthetic MIMIC Temporal ---")
    mimic = generate_mimic_temporal()
    
    print("\n--- Dataset 3: Synthetic Clinical Notes ---")
    notes = generate_clinical_notes()
    
    print("\n--- Dataset 4: Synthetic Wearable Data ---")
    wearable = generate_wearable_data()
    
    print("\n" + "=" * 60)
    print("All datasets ready!")
    print(f"  CVD:       {cvd.shape}")
    print(f"  MIMIC:     {mimic.shape}")
    print(f"  Notes:     {notes.shape}")
    print(f"  Wearable:  {wearable.shape}")
    print("=" * 60)
