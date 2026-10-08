"""
==============================================================================
  Feature Engineering for ML Pipeline
==============================================================================
  Transforms the raw simulation dataset into ML-ready features:
    - Encode categorical variables (polynomial_type, noise_type)
    - Create interaction features
    - Normalize/scale numerical features
    - Split into train/test sets
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict

import config


class FeatureEngineer:
    """
    Prepares the simulation dataset for ML model training.

    Attributes
    ----------
    label_encoders : dict
        Fitted label encoders for categorical columns.
    scaler : StandardScaler
        Fitted scaler for numerical features.
    feature_columns : list
        Final list of feature column names used by the model.
    """

    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_columns = []

    # ──────────────────── Public API ─────────────────────────────────

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Full feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw simulation dataset.

        Returns
        -------
        tuple
            (X, y, feature_names)
            - X: numpy array of shape (n_samples, n_features)
            - y: numpy array of shape (n_samples,) — binary target
            - feature_names: list of feature column names
        """
        df = df.copy()

        # ── 1. Encode categorical features ──
        categorical_cols = ["polynomial_type", "noise_type"]
        for col in categorical_cols:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # ── 2. Create interaction features ──
        df["burst_exceeds_degree"] = (
            df["burst_length_actual"] > df["polynomial_degree"]
        ).astype(int)

        df["error_weight_ratio"] = (
            df["num_errors_injected"] / df["polynomial_degree"]
        ).clip(upper=10.0)

        df["frame_poly_ratio"] = df["frame_length"] / df["polynomial_degree"]

        df["ber_x_frame_length"] = df["ber"] * df["frame_length"]

        # ── 3. Select feature columns ──
        self.feature_columns = [
            "frame_length",
            "codeword_length",
            "polynomial_degree",
            "polynomial_type_encoded",
            "noise_type_encoded",
            "ber",
            "burst_length_actual",
            "num_errors_injected",
            "error_density",
            "error_to_poly_ratio",
            "burst_exceeds_degree",
            "error_weight_ratio",
            "frame_poly_ratio",
            "ber_x_frame_length",
        ]

        X = df[self.feature_columns].values.astype(np.float64)
        y = df["detected"].values.astype(int)

        return X, y, self.feature_columns

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = config.ML_TEST_SIZE,
        seed: int = config.RANDOM_SEED,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split into stratified train/test sets.

        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

    def scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler on training data and transform both sets.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def get_feature_importance_names(self) -> list:
        """Return the feature column names for importance plots."""
        return list(self.feature_columns)
