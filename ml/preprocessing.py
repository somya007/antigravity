"""
Data Preprocessing Pipeline
============================
Handles loading, cleaning, feature engineering, and splitting for all datasets.
Supports: CVD, MIMIC temporal, synthetic clinical data.

Key Features:
  - KNN imputation for missing values
  - Feature engineering (interaction terms, polynomial features)
  - SMOTE oversampling for imbalanced classes
  - StandardScaler normalization
  - Stratified train/test split
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

# Path to data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def load_and_preprocess(dataset_name, test_size=0.2, random_state=42):
    """
    Load and preprocess a dataset by name.
    
    Args:
        dataset_name: One of 'cvd', 'mimic', 'clinical_notes', 'wearable'
        test_size: Test split fraction
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    print(f"[INFO] Loading dataset: {dataset_name}")
    
    if dataset_name == "cvd":
        return _preprocess_cvd(test_size, random_state)
    elif dataset_name == "mimic":
        return _preprocess_mimic(test_size, random_state)
    elif dataset_name == "clinical_notes":
        return _preprocess_clinical_notes(test_size, random_state)
    elif dataset_name == "wearable":
        return _preprocess_wearable(test_size, random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: cvd, mimic, clinical_notes, wearable")


def _preprocess_cvd(test_size, random_state):
    """Preprocess Cardiovascular Disease dataset."""
    filepath = os.path.join(DATA_DIR, "cardio_train.csv")
    df = pd.read_csv(filepath, sep=";")
    
    # Drop ID column
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    
    # Convert age from days to years
    df["age"] = (df["age"] / 365.25).astype(int)
    
    # Remove outliers in blood pressure
    df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 250)]
    df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 160)]
    
    # Feature engineering
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["age_bp_interaction"] = df["age"] * df["ap_hi"]
    df["bmi_chol_interaction"] = df["bmi"] * df["cholesterol"]
    
    # Target
    target = "cardio"
    y = df[target].values
    X = df.drop(target, axis=1)
    
    feature_names = list(X.columns)
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # SMOTE if imbalanced
    X_train, y_train = _apply_smote_if_needed(X_train, y_train, random_state)
    
    print(f"[OK] CVD preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def _preprocess_mimic(test_size, random_state):
    """Preprocess MIMIC temporal data (aggregated per patient)."""
    filepath = os.path.join(DATA_DIR, "mimic_temporal.csv")
    df = pd.read_csv(filepath)
    
    # Aggregate per patient: use last visit stats + temporal features
    agg_funcs = {
        "age": "first",
        "gender": "first",
        "heart_rate": ["mean", "std", "min", "max"],
        "systolic_bp": ["mean", "std", "min", "max"],
        "temperature": ["mean", "std", "max"],
        "respiratory_rate": ["mean", "std", "max"],
        "spo2": ["mean", "min"],
        "glucose": ["mean", "std", "max"],
        "creatinine": ["mean", "max"],
        "wbc": ["mean", "max"],
        "hemoglobin": ["mean", "min"],
        "mortality": "max"
    }
    
    df_agg = df.groupby("patient_id").agg(agg_funcs)
    df_agg.columns = ["_".join(col).strip("_") for col in df_agg.columns]
    df_agg = df_agg.reset_index(drop=True)
    
    # Add number of visits as feature
    visit_counts = df.groupby("patient_id")["visit_num"].max().values + 1
    df_agg["num_visits"] = visit_counts[:len(df_agg)]
    
    # Add temporal trend features (slope of heart rate and BP over visits)
    trends = []
    for pid in df["patient_id"].unique():
        patient_data = df[df["patient_id"] == pid].sort_values("visit_num")
        if len(patient_data) >= 2:
            hr_slope = np.polyfit(range(len(patient_data)), patient_data["heart_rate"].values, 1)[0]
            bp_slope = np.polyfit(range(len(patient_data)), patient_data["systolic_bp"].values, 1)[0]
        else:
            hr_slope, bp_slope = 0, 0
        trends.append({"hr_trend": hr_slope, "bp_trend": bp_slope})
    
    trend_df = pd.DataFrame(trends)
    df_agg["hr_trend"] = trend_df["hr_trend"].values[:len(df_agg)]
    df_agg["bp_trend"] = trend_df["bp_trend"].values[:len(df_agg)]
    
    # Target
    target = "mortality_max"
    y = df_agg[target].values
    X = df_agg.drop(target, axis=1)
    
    feature_names = list(X.columns)
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # SMOTE
    X_train, y_train = _apply_smote_if_needed(X_train, y_train, random_state)
    
    print(f"[OK] MIMIC preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def _preprocess_clinical_notes(test_size, random_state):
    """Preprocess clinical notes dataset (text features + structured)."""
    filepath = os.path.join(DATA_DIR, "clinical_notes.csv")
    df = pd.read_csv(filepath)
    
    # Extract text features using TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    text_features = tfidf.fit_transform(df["clinical_note"]).toarray()
    text_feature_names = [f"text_{name}" for name in tfidf.get_feature_names_out()]
    
    # Structured features
    structured = df[["age", "gender"]].values
    structured_names = ["age", "gender"]
    
    # Combine
    X = np.hstack([structured, text_features])
    feature_names = structured_names + text_feature_names
    
    # Binary target: high risk vs not
    y = (df["risk_label"] >= 2).astype(int).values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, y_train = _apply_smote_if_needed(X_train, y_train, random_state)
    
    print(f"[OK] Clinical notes preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def _preprocess_wearable(test_size, random_state):
    """Preprocess wearable time-series data (aggregated per patient)."""
    filepath = os.path.join(DATA_DIR, "wearable_data.csv")
    df = pd.read_csv(filepath)
    
    # Aggregate per patient
    agg_funcs = {
        "resting_hr": ["mean", "std", "min", "max"],
        "daily_steps": ["mean", "std", "min", "max"],
        "sleep_hours": ["mean", "std", "min"],
        "hrv_ms": ["mean", "std", "min"],
        "health_risk": "max"
    }
    
    df_agg = df.groupby("patient_id").agg(agg_funcs)
    df_agg.columns = ["_".join(col) for col in df_agg.columns]
    df_agg = df_agg.reset_index(drop=True)
    
    # Add trend features
    trends = []
    for pid in df["patient_id"].unique():
        patient_data = df[df["patient_id"] == pid].sort_values("day")
        if len(patient_data) >= 2:
            hr_slope = np.polyfit(range(len(patient_data)), patient_data["resting_hr"].values, 1)[0]
            steps_slope = np.polyfit(range(len(patient_data)), patient_data["daily_steps"].values, 1)[0]
        else:
            hr_slope, steps_slope = 0, 0
        trends.append({"hr_trend": hr_slope, "steps_trend": steps_slope})
    
    trend_df = pd.DataFrame(trends)
    df_agg["hr_trend"] = trend_df["hr_trend"].values[:len(df_agg)]
    df_agg["steps_trend"] = trend_df["steps_trend"].values[:len(df_agg)]
    
    # Target
    target = "health_risk_max"
    y = df_agg[target].values
    X = df_agg.drop(target, axis=1)
    
    feature_names = list(X.columns)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, y_train = _apply_smote_if_needed(X_train, y_train, random_state)
    
    print(f"[OK] Wearable preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler


def _apply_smote_if_needed(X_train, y_train, random_state, threshold=0.35):
    """Apply SMOTE if positive class ratio is below threshold."""
    pos_ratio = np.mean(y_train)
    if pos_ratio < threshold:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  [SMOTE] Applied. Positive ratio: {pos_ratio:.2f} → {np.mean(y_train):.2f}")
        except ImportError:
            print("  [WARN] imbalanced-learn not installed. Skipping SMOTE.")
    else:
        print(f"  [INFO] Classes balanced (positive ratio: {pos_ratio:.2f}). Skipping SMOTE.")
    return X_train, y_train


def get_temporal_sequences(dataset_name="mimic", seq_len=8):
    """
    Get temporal data as sequences for LSTM input.
    Returns: X_sequences (n_patients, seq_len, n_features), y, patient_ids
    """
    if dataset_name == "mimic":
        filepath = os.path.join(DATA_DIR, "mimic_temporal.csv")
    elif dataset_name == "wearable":
        filepath = os.path.join(DATA_DIR, "wearable_data.csv")
    else:
        raise ValueError(f"No temporal data for: {dataset_name}")
    
    df = pd.read_csv(filepath)
    
    if dataset_name == "mimic":
        feature_cols = ["heart_rate", "systolic_bp", "temperature", 
                       "respiratory_rate", "spo2", "glucose", 
                       "creatinine", "wbc", "hemoglobin"]
        target_col = "mortality"
        id_col = "patient_id"
        time_col = "visit_num"
    else:
        feature_cols = ["resting_hr", "daily_steps", "sleep_hours", "hrv_ms"]
        target_col = "health_risk"
        id_col = "patient_id"
        time_col = "day"
    
    sequences = []
    targets = []
    patient_ids = []
    
    for pid in df[id_col].unique():
        patient_data = df[df[id_col] == pid].sort_values(time_col)
        features = patient_data[feature_cols].values
        target = patient_data[target_col].max()
        
        # Pad or truncate to seq_len
        if len(features) >= seq_len:
            features = features[-seq_len:]
        else:
            padding = np.zeros((seq_len - len(features), len(feature_cols)))
            features = np.vstack([padding, features])
        
        sequences.append(features)
        targets.append(target)
        patient_ids.append(pid)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    # Normalize
    scaler = StandardScaler()
    n_patients, n_steps, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_flat).reshape(n_patients, n_steps, n_features)
    
    return X_scaled, y, patient_ids, scaler, feature_cols
