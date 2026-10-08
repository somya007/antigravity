"""
Tabular Model Branch
====================
Gradient boosting models (XGBoost, LightGBM) for structured EHR data.
Includes GridSearchCV hyperparameter tuning.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


def get_tabular_models():
    """Return dict of models with their hyperparameter grids."""
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf"]
            }
        }
    }
    
    # Try adding XGBoost
    try:
        import xgboost as xgb
        models["XGBoost"] = {
            "model": xgb.XGBClassifier(
                random_state=42, 
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0
            ),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0]
            }
        }
    except ImportError:
        print("[WARN] XGBoost not installed. Skipping.")
    
    # Try adding LightGBM
    try:
        import lightgbm as lgb
        models["LightGBM"] = {
            "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [15, 31]
            }
        }
    except ImportError:
        print("[WARN] LightGBM not installed. Skipping.")
    
    return models


def train_tabular_models(X_train, y_train, cv=5, scoring="f1"):
    """
    Train all tabular models with GridSearchCV.
    
    Returns:
        results: dict of {model_name: {best_model, best_params, cv_scores, best_score}}
    """
    models = get_tabular_models()
    results = {}
    
    for name, config in models.items():
        print(f"  Training {name}...")
        try:
            grid = GridSearchCV(
                config["model"],
                config["params"],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                refit=True
            )
            grid.fit(X_train, y_train)
            
            results[name] = {
                "best_model": grid.best_estimator_,
                "best_params": grid.best_params_,
                "cv_scores": grid.cv_results_["mean_test_score"],
                "best_score": grid.best_score_
            }
            print(f"    Best {scoring}: {grid.best_score_:.4f} | Params: {grid.best_params_}")
        except Exception as e:
            print(f"    [ERROR] {name} failed: {e}")
    
    return results


def get_best_tabular_model(results):
    """Select the best model from training results."""
    best_name = max(results, key=lambda k: results[k]["best_score"])
    best_model = results[best_name]["best_model"]
    print(f"  Best tabular model: {best_name} (score={results[best_name]['best_score']:.4f})")
    return best_name, best_model


def get_tabular_embedding(model, X):
    """
    Extract embedding from tabular model for fusion.
    For tree models, uses leaf node indicators.
    For linear models, uses the raw features.
    """
    if hasattr(model, "apply"):
        # Tree models: use leaf indices as embedding
        leaves = model.apply(X)
        if len(leaves.shape) == 3:
            leaves = leaves.reshape(leaves.shape[0], -1)
        return leaves.astype(np.float32)
    elif hasattr(model, "predict_proba"):
        # Use prediction probabilities + original features as embedding
        probs = model.predict_proba(X)
        return np.hstack([X, probs]).astype(np.float32)
    else:
        return X.astype(np.float32)
