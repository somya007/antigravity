"""
==============================================================================
  ML Model Training — Random Forest & XGBoost
==============================================================================
  Trains two classifiers to predict whether a burst error will bypass
  the CRC check (undetected error / aliasing) based on channel conditions.

  Models:
    1. Random Forest (sklearn)
    2. XGBoost Classifier

  Tuning: GridSearchCV with stratified K-fold cross-validation.

  Outputs:
    - Trained model objects
    - Best hyperparameters
    - Cross-validation scores
    - Saved model files (.joblib)
==============================================================================
"""

import os
import time
from typing import Dict, Tuple, Any

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

import config


class ModelTrainer:
    """
    Trains and tunes Random Forest and XGBoost classifiers.

    Attributes
    ----------
    rf_model : RandomForestClassifier or None
        Trained Random Forest model.
    xgb_model : XGBClassifier or None
        Trained XGBoost model.
    rf_best_params : dict
        Best hyperparameters for Random Forest.
    xgb_best_params : dict
        Best hyperparameters for XGBoost.
    training_times : dict
        Wall-clock training time for each model.
    """

    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.rf_best_params: Dict = {}
        self.xgb_best_params: Dict = {}
        self.rf_cv_results = None
        self.xgb_cv_results = None
        self.training_times: Dict[str, float] = {}

    # ──────────────────── Random Forest ──────────────────────────────

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict = None,
        cv_folds: int = config.ML_CV_FOLDS,
        seed: int = config.RANDOM_SEED,
        use_quick_grid: bool = True,
    ) -> RandomForestClassifier:
        """
        Train a Random Forest with GridSearchCV.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data.
        param_grid : dict or None
            Hyperparameter grid.  Defaults to config.RF_PARAM_GRID.
        cv_folds : int
            Number of CV folds.
        use_quick_grid : bool
            If True, use a reduced grid for faster training.

        Returns
        -------
        RandomForestClassifier
            The best estimator from grid search.
        """
        if param_grid is None:
            if use_quick_grid:
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                }
            else:
                param_grid = config.RF_PARAM_GRID

        print("=" * 60)
        print("Training Random Forest with GridSearchCV")
        print(f"  CV Folds: {cv_folds}")
        print(f"  Grid size: {self._grid_size(param_grid)} combinations")
        print("=" * 60)

        rf = RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            class_weight="balanced",
        )

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed
        )

        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train, y_train)

        elapsed = time.time() - start_time

        self.rf_model = grid_search.best_estimator_
        self.rf_best_params = grid_search.best_params_
        self.rf_cv_results = grid_search.cv_results_
        self.training_times["random_forest"] = elapsed

        print(f"\n  Best params: {self.rf_best_params}")
        print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
        print(f"  Training time: {elapsed:.1f}s")

        return self.rf_model

    # ──────────────────── XGBoost ────────────────────────────────────

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict = None,
        cv_folds: int = config.ML_CV_FOLDS,
        seed: int = config.RANDOM_SEED,
        use_quick_grid: bool = True,
    ) -> XGBClassifier:
        """
        Train an XGBoost classifier with GridSearchCV.

        Returns
        -------
        XGBClassifier
            The best estimator from grid search.
        """
        if param_grid is None:
            if use_quick_grid:
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8, 1.0],
                }
            else:
                param_grid = config.XGB_PARAM_GRID

        # Compute scale_pos_weight for class imbalance
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        scale_pos_weight = n_negative / max(n_positive, 1)

        print("\n" + "=" * 60)
        print("Training XGBoost with GridSearchCV")
        print(f"  CV Folds: {cv_folds}")
        print(f"  Grid size: {self._grid_size(param_grid)} combinations")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
        print("=" * 60)

        xgb = XGBClassifier(
            random_state=seed,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=1,
        )

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed
        )

        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=1,
            verbose=1,
            return_train_score=True,
        )
        grid_search.fit(X_train, y_train)

        elapsed = time.time() - start_time

        self.xgb_model = grid_search.best_estimator_
        self.xgb_best_params = grid_search.best_params_
        self.xgb_cv_results = grid_search.cv_results_
        self.training_times["xgboost"] = elapsed

        print(f"\n  Best params: {self.xgb_best_params}")
        print(f"  Best CV F1:  {grid_search.best_score_:.4f}")
        print(f"  Training time: {elapsed:.1f}s")

        return self.xgb_model

    # ──────────────────── Save / Load ────────────────────────────────

    def save_models(self, output_dir: str = config.MODELS_DIR):
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)

        if self.rf_model:
            path = os.path.join(output_dir, "random_forest_model.joblib")
            joblib.dump(self.rf_model, path)
            print(f"  Random Forest saved → {path}")

        if self.xgb_model:
            path = os.path.join(output_dir, "xgboost_model.joblib")
            joblib.dump(self.xgb_model, path)
            print(f"  XGBoost saved → {path}")

    def load_models(self, output_dir: str = config.MODELS_DIR):
        """Load previously trained models."""
        rf_path = os.path.join(output_dir, "random_forest_model.joblib")
        xgb_path = os.path.join(output_dir, "xgboost_model.joblib")

        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            print(f"  Random Forest loaded ← {rf_path}")

        if os.path.exists(xgb_path):
            self.xgb_model = joblib.load(xgb_path)
            print(f"  XGBoost loaded ← {xgb_path}")

    # ──────────────────── Utility ────────────────────────────────────

    @staticmethod
    def _grid_size(param_grid: Dict) -> int:
        """Compute total number of hyperparameter combinations."""
        size = 1
        for vals in param_grid.values():
            size *= len(vals)
        return size
