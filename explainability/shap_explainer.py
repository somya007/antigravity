"""
SHAP Explainability Module
============================
Feature importance and patient-level explanations using SHAP.
Provides interpretable model outputs for clinical decision support.
"""

import numpy as np


def compute_shap_values(model, X_test, feature_names, max_samples=100):
    """
    Compute SHAP values for model predictions.
    
    Args:
        model: trained sklearn model
        X_test: test data (numpy array)
        feature_names: list of feature names
        max_samples: max samples for SHAP computation
    
    Returns:
        dict with shap_values, feature_importance, feature_names
    """
    try:
        import shap
        
        X_sample = X_test[:max_samples]
        
        # Choose appropriate explainer
        model_type = type(model).__name__
        
        if model_type in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier", 
                          "GradientBoostingClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            # Kernel SHAP for other models (slower)
            background = shap.sample(X_test, min(50, len(X_test)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Compute mean absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Build feature importance dict
        feature_importance = {}
        for i, name in enumerate(feature_names):
            if i < len(mean_abs_shap):
                feature_importance[name] = float(mean_abs_shap[i])
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "feature_names": feature_names,
            "X_sample": X_sample,
            "explainer": explainer,
            "success": True
        }
    
    except ImportError:
        print("[WARN] SHAP not installed. Using built-in feature importance.")
        return _fallback_importance(model, feature_names)
    except Exception as e:
        print(f"[WARN] SHAP computation failed: {e}. Using fallback.")
        return _fallback_importance(model, feature_names)


def _fallback_importance(model, feature_names):
    """Get feature importance from model's built-in method."""
    importance = {}
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for i, name in enumerate(feature_names):
            if i < len(importances):
                importance[name] = float(importances[i])
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_).flatten()
        for i, name in enumerate(feature_names):
            if i < len(coefs):
                importance[name] = float(coefs[i])
    else:
        # Default: equal importance
        for name in feature_names:
            importance[name] = 1.0 / len(feature_names)
    
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "shap_values": None,
        "feature_importance": importance,
        "feature_names": feature_names,
        "X_sample": None,
        "explainer": None,
        "success": False,
        "note": "Using model built-in feature importance (SHAP unavailable)"
    }


def explain_patient(model, patient_features, feature_names, X_background=None):
    """
    Generate patient-level explanation.
    
    Args:
        model: trained model
        patient_features: 1D array of patient features
        feature_names: list of feature names
        X_background: background dataset for SHAP
    
    Returns:
        dict with patient-level explanation
    """
    patient_features = np.array(patient_features).reshape(1, -1)
    
    try:
        import shap
        
        model_type = type(model).__name__
        
        if model_type in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                          "GradientBoostingClassifier"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(patient_features)
        else:
            if X_background is not None:
                background = shap.sample(X_background, min(50, len(X_background)))
            else:
                background = patient_features
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(patient_features, nsamples=100)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Build contribution list
        contributions = []
        for i, name in enumerate(feature_names):
            if i < shap_values.shape[1]:
                contributions.append({
                    "feature": name,
                    "shap_value": float(shap_values[0][i]),
                    "feature_value": float(patient_features[0][i]) if i < patient_features.shape[1] else 0,
                    "direction": "increases risk" if shap_values[0][i] > 0 else "decreases risk"
                })
        
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "contributions": contributions,
            "top_risk_factors": contributions[:5],
            "top_protective": [c for c in contributions if c["shap_value"] < 0][:3],
            "shap_values": shap_values[0],
            "success": True
        }
    
    except Exception as e:
        # Fallback: use feature importance
        importance = _fallback_importance(model, feature_names)
        contributions = [
            {
                "feature": name,
                "shap_value": importance["feature_importance"].get(name, 0),
                "feature_value": float(patient_features[0][i]) if i < patient_features.shape[1] else 0,
                "direction": "contributes to prediction"
            }
            for i, name in enumerate(feature_names)
        ]
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "contributions": contributions,
            "top_risk_factors": contributions[:5],
            "top_protective": [],
            "shap_values": None,
            "success": False,
            "note": f"SHAP unavailable: {e}"
        }
