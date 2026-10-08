"""
What-If Risk Simulator
=======================
Interactive parameter modification to simulate risk changes.
Allows clinicians to model "what if the patient..." scenarios.

Examples:
  - "What if the patient reduces BMI by 5?"
  - "What if blood pressure drops to 120/80?"
  - "What if the patient quits smoking?"
"""

import numpy as np
import joblib
import os


SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")


def simulate_risk_change(patient_data, modifications, dataset_name="cvd"):
    """
    Simulate risk score changes when patient parameters are modified.
    
    Args:
        patient_data: dict of original patient features
        modifications: dict of {feature_name: new_value}
        dataset_name: which model to use
    
    Returns:
        dict with original_risk, modified_risk, risk_change, impact_details
    """
    model_path = os.path.join(SAVE_DIR, f"model_{dataset_name}.joblib")
    
    if not os.path.exists(model_path):
        return _mock_simulation(patient_data, modifications)
    
    try:
        artifacts = joblib.load(model_path)
        model = artifacts["best_model"]
        scaler = artifacts["scaler"]
        feature_names = artifacts["feature_names"]
        
        # Create original feature vector
        original_features = _patient_to_features(patient_data, feature_names)
        
        # Create modified feature vector
        modified_data = patient_data.copy()
        modified_data.update(modifications)
        
        # Recalculate derived features if needed
        if "weight" in modifications or "height" in modifications:
            w = modified_data.get("weight", patient_data.get("weight", 70))
            h = modified_data.get("height", patient_data.get("height", 170))
            modified_data["bmi"] = w / ((h / 100) ** 2)
        
        if "ap_hi" in modifications or "ap_lo" in modifications:
            modified_data["pulse_pressure"] = (
                modified_data.get("ap_hi", patient_data.get("ap_hi", 120)) - 
                modified_data.get("ap_lo", patient_data.get("ap_lo", 80))
            )
        
        modified_features = _patient_to_features(modified_data, feature_names)
        
        # Scale and predict
        original_scaled = scaler.transform([original_features])
        modified_scaled = scaler.transform([modified_features])
        
        if hasattr(model, "predict_proba"):
            original_prob = model.predict_proba(original_scaled)[0][1]
            modified_prob = model.predict_proba(modified_scaled)[0][1]
        else:
            original_prob = float(model.predict(original_scaled)[0])
            modified_prob = float(model.predict(modified_scaled)[0])
        
        original_risk = round(original_prob * 100, 1)
        modified_risk = round(modified_prob * 100, 1)
        risk_change = round(modified_risk - original_risk, 1)
        
        # Detailed impact per modification
        impact_details = []
        for feature, new_value in modifications.items():
            old_value = patient_data.get(feature, 0)
            single_mod = patient_data.copy()
            single_mod[feature] = new_value
            single_features = _patient_to_features(single_mod, feature_names)
            single_scaled = scaler.transform([single_features])
            
            if hasattr(model, "predict_proba"):
                single_prob = model.predict_proba(single_scaled)[0][1]
            else:
                single_prob = float(model.predict(single_scaled)[0])
            
            impact = round((single_prob - original_prob) * 100, 1)
            
            impact_details.append({
                "feature": feature,
                "old_value": old_value,
                "new_value": new_value,
                "risk_impact": impact,
                "direction": "⬇️ Decreased" if impact < 0 else ("⬆️ Increased" if impact > 0 else "➡️ No change")
            })
        
        return {
            "original_risk": original_risk,
            "modified_risk": modified_risk,
            "risk_change": risk_change,
            "direction": "improved" if risk_change < 0 else ("worsened" if risk_change > 0 else "unchanged"),
            "impact_details": impact_details,
            "category_original": _categorize_risk(original_risk),
            "category_modified": _categorize_risk(modified_risk)
        }
    
    except Exception as e:
        print(f"[WARN] Simulation error: {e}")
        return _mock_simulation(patient_data, modifications)


def _patient_to_features(patient_data, feature_names):
    """Convert patient dict to feature vector aligned with training features."""
    features = []
    for name in feature_names:
        value = patient_data.get(name, 0)
        if isinstance(value, (int, float)):
            features.append(float(value))
        else:
            features.append(0.0)
    return features


def _categorize_risk(score):
    """Categorize risk score."""
    if score < 30:
        return "Low"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "High"
    else:
        return "Critical"


def _mock_simulation(patient_data, modifications):
    """Mock simulation when no trained model is available."""
    # Estimate risk changes based on medical knowledge
    original_risk = 50.0
    risk_change = 0.0
    impact_details = []
    
    risk_weights = {
        "ap_hi": 0.3,
        "cholesterol": 5.0,
        "gluc": 4.0,
        "smoke": 15.0,
        "bmi": 0.8,
        "weight": 0.2,
        "active": -10.0,
        "alco": 5.0,
        "age": 0.3
    }
    
    for feature, new_value in modifications.items():
        old_value = patient_data.get(feature, 0)
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            weight = risk_weights.get(feature, 0.5)
            impact = (new_value - old_value) * weight
            risk_change += impact
            
            impact_details.append({
                "feature": feature,
                "old_value": old_value,
                "new_value": new_value,
                "risk_impact": round(impact, 1),
                "direction": "⬇️ Decreased" if impact < 0 else ("⬆️ Increased" if impact > 0 else "➡️ No change")
            })
    
    modified_risk = max(0, min(100, original_risk + risk_change))
    
    return {
        "original_risk": original_risk,
        "modified_risk": round(modified_risk, 1),
        "risk_change": round(risk_change, 1),
        "direction": "improved" if risk_change < 0 else ("worsened" if risk_change > 0 else "unchanged"),
        "impact_details": impact_details,
        "category_original": _categorize_risk(original_risk),
        "category_modified": _categorize_risk(modified_risk),
        "note": "Simulated with heuristic model (no trained model found). Train a model first for accurate results."
    }


def get_preset_scenarios():
    """Get preset what-if scenarios for the UI."""
    return [
        {
            "name": "🚭 Quit Smoking",
            "description": "Patient stops smoking completely",
            "modifications": {"smoke": 0}
        },
        {
            "name": "🏃 Start Exercising",
            "description": "Patient begins regular physical activity",
            "modifications": {"active": 1}
        },
        {
            "name": "🥗 Lower Cholesterol",
            "description": "Cholesterol reduced to normal levels through diet",
            "modifications": {"cholesterol": 1}
        },
        {
            "name": "💊 Blood Pressure Control",
            "description": "Blood pressure reduced to 120/80 with treatment",
            "modifications": {"ap_hi": 120, "ap_lo": 80}
        },
        {
            "name": "⚖️ Weight Loss (10kg)",
            "description": "Patient loses 10kg through diet and exercise",
            "modifications": {"weight": -10}  # Relative change
        },
        {
            "name": "🍷 Stop Alcohol",
            "description": "Patient stops alcohol consumption",
            "modifications": {"alco": 0}
        },
        {
            "name": "🌟 Comprehensive Lifestyle Change",
            "description": "Quit smoking + exercise + diet improvement",
            "modifications": {"smoke": 0, "active": 1, "cholesterol": 1, "gluc": 1}
        }
    ]
