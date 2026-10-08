# ============================================================
# GDHRA System — Google Colab Notebook (Part 2: Preprocessing)
# ============================================================

# %% [markdown]
# ## 🔧 Step 2: Data Preprocessing Pipeline
# - KNN Imputation for missing values
# - Feature engineering (interaction terms, BMI, pulse pressure)
# - SMOTE oversampling for imbalanced classes
# - StandardScaler normalization

# %% — Cell 6: Preprocessing Functions
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer

def apply_smote_if_needed(X_train, y_train, threshold=0.35, random_state=42):
    """Apply SMOTE if positive class ratio is below threshold."""
    pos_ratio = np.mean(y_train)
    if pos_ratio < threshold:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  [SMOTE] Positive ratio: {pos_ratio:.2f} → {np.mean(y_train):.2f}")
    else:
        print(f"  [INFO] Balanced (positive: {pos_ratio:.2f}). Skipping SMOTE.")
    return X_train, y_train

def preprocess_cvd(df, test_size=0.2, random_state=42):
    """Preprocess CVD dataset with feature engineering."""
    df = df.copy()
    if "id" in df.columns:
        df = df.drop("id", axis=1)
    df["age"] = (df["age"] / 365.25).astype(int)
    df = df[(df["ap_hi"] > 60) & (df["ap_hi"] < 250)]
    df = df[(df["ap_lo"] > 30) & (df["ap_lo"] < 160)]

    # Feature engineering
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["age_bp_interaction"] = df["age"] * df["ap_hi"]
    df["bmi_chol_interaction"] = df["bmi"] * df["cholesterol"]

    y = df["cardio"].values
    X = df.drop("cardio", axis=1)
    feature_names = list(X.columns)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, y_train = apply_smote_if_needed(X_train, y_train, random_state=random_state)
    print(f"✅ CVD preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler

def preprocess_mimic(df, test_size=0.2, random_state=42):
    """Preprocess MIMIC temporal data (aggregated per patient)."""
    agg_funcs = {
        "age": "first", "gender": "first",
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

    # Temporal trend features
    visit_counts = df.groupby("patient_id")["visit_num"].max().values + 1
    df_agg["num_visits"] = visit_counts[:len(df_agg)]
    trends = []
    for pid in df["patient_id"].unique():
        pd_data = df[df["patient_id"] == pid].sort_values("visit_num")
        if len(pd_data) >= 2:
            hr_slope = np.polyfit(range(len(pd_data)), pd_data["heart_rate"].values, 1)[0]
            bp_slope = np.polyfit(range(len(pd_data)), pd_data["systolic_bp"].values, 1)[0]
        else:
            hr_slope, bp_slope = 0, 0
        trends.append({"hr_trend": hr_slope, "bp_trend": bp_slope})
    trend_df = pd.DataFrame(trends)
    df_agg["hr_trend"] = trend_df["hr_trend"].values[:len(df_agg)]
    df_agg["bp_trend"] = trend_df["bp_trend"].values[:len(df_agg)]

    y = df_agg["mortality_max"].values
    X = df_agg.drop("mortality_max", axis=1)
    feature_names = list(X.columns)

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, y_train = apply_smote_if_needed(X_train, y_train, random_state=random_state)
    print(f"✅ MIMIC preprocessed: train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, scaler

def get_temporal_sequences(df, seq_len=8):
    """Get temporal data as sequences for LSTM input."""
    feature_cols = ["heart_rate", "systolic_bp", "temperature",
                   "respiratory_rate", "spo2", "glucose",
                   "creatinine", "wbc", "hemoglobin"]
    sequences, targets, pids = [], [], []
    for pid in df["patient_id"].unique():
        patient_data = df[df["patient_id"] == pid].sort_values("visit_num")
        features = patient_data[feature_cols].values
        target = patient_data["mortality"].max()
        if len(features) >= seq_len:
            features = features[-seq_len:]
        else:
            padding = np.zeros((seq_len - len(features), len(feature_cols)))
            features = np.vstack([padding, features])
        sequences.append(features)
        targets.append(target)
        pids.append(pid)

    X = np.array(sequences)
    y = np.array(targets)
    scaler = StandardScaler()
    n_p, n_s, n_f = X.shape
    X_scaled = scaler.fit_transform(X.reshape(-1, n_f)).reshape(n_p, n_s, n_f)
    return X_scaled, y, pids, scaler, feature_cols

# %% — Cell 7: Run Preprocessing
print("=" * 60)
print("PREPROCESSING ALL DATASETS")
print("=" * 60)

print("\n--- CVD Dataset ---")
X_train_cvd, X_test_cvd, y_train_cvd, y_test_cvd, feat_cvd, scaler_cvd = preprocess_cvd(cvd_df)

print("\n--- MIMIC Dataset ---")
X_train_mimic, X_test_mimic, y_train_mimic, y_test_mimic, feat_mimic, scaler_mimic = preprocess_mimic(mimic_df)

print("\n--- Temporal Sequences ---")
X_seq, y_seq, seq_pids, seq_scaler, seq_feat_cols = get_temporal_sequences(mimic_df)
X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42,
    stratify=y_seq if len(np.unique(y_seq)) > 1 else None
)
print(f"✅ Temporal sequences: train={X_seq_train.shape}, test={X_seq_test.shape}")

print("\n--- Clinical Notes (TF-IDF) ---")
tfidf = TfidfVectorizer(max_features=50, stop_words="english")
text_features = tfidf.fit_transform(notes_df["clinical_note"]).toarray()
text_labels = (notes_df["risk_label"] >= 2).astype(int).values
X_text = np.hstack([notes_df[["age", "gender"]].values, text_features])
text_scaler = StandardScaler()
X_text_scaled = text_scaler.fit_transform(X_text)
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(
    X_text_scaled, text_labels, test_size=0.2, random_state=42, stratify=text_labels
)
print(f"✅ Text features: train={X_train_text.shape}, test={X_test_text.shape}")

print("\n✅ All preprocessing complete!")
