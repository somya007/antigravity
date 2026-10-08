"""
GenAI Health Advisor
====================
Generates personalized health advice using Google Gemini API.
Falls back to a comprehensive rule-based advisor when API is unavailable.

Features:
  - Risk factor explanation
  - Personalized preventive recommendations
  - Lifestyle modification suggestions
  - Medical follow-up priorities
"""

import os


def generate_health_advice(risk_score, risk_category, patient_data, dataset_name="cvd", feature_importances=None):
    """
    Generate personalized health advice.
    
    Args:
        risk_score: 0-100 risk score
        risk_category: "Low" / "Moderate" / "High" / "Critical"
        patient_data: dict of patient features
        dataset_name: which dataset context
        feature_importances: optional dict of {feature: importance}
    
    Returns:
        dict with keys: advice, risk_explanation, recommendations, warnings
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if api_key:
        return _gemini_advice(risk_score, risk_category, patient_data, 
                            dataset_name, feature_importances, api_key)
    else:
        return _fallback_advice(risk_score, risk_category, patient_data,
                               dataset_name, feature_importances)


def _gemini_advice(risk_score, risk_category, patient_data, dataset_name, feature_importances, api_key):
    """Generate advice using Google Gemini API."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Build medical context from knowledge base
        from genai.knowledge_base import get_relevant_knowledge
        medical_context = get_relevant_knowledge(patient_data, risk_category)
        
        # Build importance context
        importance_text = ""
        if feature_importances:
            top_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            importance_text = "\n".join([f"  - {f}: importance={v:.3f}" for f, v in top_features])
        
        prompt = f"""You are a clinical decision support system providing health risk analysis.

PATIENT DATA:
{_format_patient_data(patient_data)}

RISK ASSESSMENT:
- Risk Score: {risk_score}/100
- Risk Category: {risk_category}
- Dataset Context: {dataset_name}

TOP RISK CONTRIBUTING FACTORS:
{importance_text or 'Not available'}

MEDICAL KNOWLEDGE CONTEXT:
{medical_context}

Provide a structured response with EXACTLY these sections:

## Risk Explanation
Explain why this patient has a {risk_category} risk score of {risk_score}/100 based on their data.

## Personalized Recommendations
Provide 5 specific, actionable recommendations tailored to this patient's risk factors.

## Lifestyle Modifications
Suggest 3-4 specific lifestyle changes with measurable goals.

## Medical Follow-up
Recommend specific tests, screenings, or specialist referrals.

## Warnings
List any urgent concerns or red flags from the patient data.

Keep language clear, professional, and patient-appropriate. Do not make diagnoses."""

        response = model.generate_content(prompt)
        
        return {
            "advice": response.text,
            "risk_explanation": _extract_section(response.text, "Risk Explanation"),
            "recommendations": _extract_section(response.text, "Personalized Recommendations"),
            "lifestyle": _extract_section(response.text, "Lifestyle Modifications"),
            "followup": _extract_section(response.text, "Medical Follow-up"),
            "warnings": _extract_section(response.text, "Warnings"),
            "source": "gemini"
        }
    
    except Exception as e:
        print(f"[WARN] Gemini API failed: {e}. Using fallback advisor.")
        return _fallback_advice(risk_score, risk_category, patient_data, 
                               dataset_name, feature_importances)


def _fallback_advice(risk_score, risk_category, patient_data, dataset_name, feature_importances):
    """Comprehensive rule-based health advisor (no API needed)."""
    
    advice_parts = {}
    
    # ─── Risk Explanation ───
    explanation_lines = [f"Your health risk score is **{risk_score}/100** ({risk_category} Risk)."]
    
    if risk_category == "Low":
        explanation_lines.append("This indicates your current health indicators are within a healthy range. "
                               "Continue maintaining your current healthy habits.")
    elif risk_category == "Moderate":
        explanation_lines.append("Some of your health indicators suggest elevated risk. "
                               "Targeted lifestyle modifications can significantly improve your outlook.")
    elif risk_category == "High":
        explanation_lines.append("Multiple risk factors indicate elevated cardiovascular or metabolic risk. "
                               "Prompt medical attention and lifestyle changes are recommended.")
    else:
        explanation_lines.append("Your risk profile indicates urgent attention is required. "
                               "Please consult a healthcare provider immediately.")
    
    # Add specific factor explanations
    if feature_importances:
        top_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        explanation_lines.append("\n**Key contributing factors:**")
        for feat, imp in top_features:
            direction = "increases" if imp > 0 else "decreases"
            explanation_lines.append(f"  • **{_humanize_feature(feat)}** — {direction} your risk (impact: {abs(imp):.1%})")
    
    advice_parts["risk_explanation"] = "\n".join(explanation_lines)
    
    # ─── Personalized Recommendations ───
    recommendations = []
    
    # Age-based
    age = patient_data.get("age", 0)
    if isinstance(age, (int, float)):
        if age > 10000:  # Age in days
            age = int(age / 365.25)
        if age > 50:
            recommendations.append("🔬 **Annual Comprehensive Screening** — Given your age, schedule yearly cardiovascular and metabolic health screenings including lipid panel, fasting glucose, and blood pressure monitoring.")
        if age > 65:
            recommendations.append("🩺 **Geriatric Assessment** — Consider a comprehensive geriatric assessment to evaluate functional status and fall risk.")
    
    # Blood pressure
    sbp = patient_data.get("ap_hi", patient_data.get("systolic_bp", 0))
    if isinstance(sbp, (int, float)) and sbp > 140:
        recommendations.append(f"🫀 **Blood Pressure Management** — Your systolic BP ({sbp} mmHg) is elevated. Target <130/80 mmHg through dietary sodium reduction (<2g/day), regular aerobic exercise (150 min/week), and stress management.")
    elif isinstance(sbp, (int, float)) and sbp > 130:
        recommendations.append(f"⚠️ **Pre-Hypertension Alert** — Your BP ({sbp} mmHg) is borderline high. Reduce sodium intake, increase potassium-rich foods, and monitor regularly.")
    
    # Cholesterol
    chol = patient_data.get("cholesterol", 0)
    if isinstance(chol, (int, float)):
        if chol >= 3 or chol > 240:
            recommendations.append("🥗 **Cholesterol Reduction Plan** — High cholesterol detected. Adopt a heart-healthy diet: increase fiber (25-30g/day), omega-3 fatty acids, plant sterols. Limit saturated fats to <7% of calories.")
        elif chol >= 2 or (chol > 200 and chol <= 240):
            recommendations.append("📊 **Cholesterol Monitoring** — Borderline cholesterol levels. Schedule a full lipid panel test. Consider increasing dietary fiber and omega-3 intake.")
    
    # Glucose
    glucose = patient_data.get("gluc", patient_data.get("glucose", 0))
    if isinstance(glucose, (int, float)):
        if glucose >= 3 or glucose > 126:
            recommendations.append("🩸 **Glucose Management** — Elevated glucose indicates diabetes risk. Monitor fasting glucose regularly. Consider HbA1c testing. Reduce simple carbohydrate intake and increase whole grains.")
        elif glucose >= 2 or (glucose > 100 and glucose <= 126):
            recommendations.append("📈 **Pre-Diabetes Prevention** — Borderline glucose levels. Implement dietary changes focusing on low glycemic index foods, regular physical activity, and weight management.")
    
    # Smoking
    smoke = patient_data.get("smoke", 0)
    if isinstance(smoke, (int, float)) and smoke > 0:
        recommendations.append("🚭 **Smoking Cessation** — Smoking is a major cardiovascular risk factor. Quitting reduces heart disease risk by 50% within 1 year. Consider nicotine replacement therapy or counseling support.")
    
    # Activity
    active = patient_data.get("active", 1)
    if isinstance(active, (int, float)) and active == 0:
        recommendations.append("🏃 **Physical Activity Program** — Sedentary lifestyle detected. Start with 30 minutes of moderate-intensity exercise 5 days/week. Walking, swimming, or cycling are excellent starting points.")
    
    # BMI
    bmi = patient_data.get("bmi", 0)
    weight = patient_data.get("weight", 0)
    height = patient_data.get("height", 0)
    if bmi == 0 and weight > 0 and height > 50:
        bmi = weight / ((height / 100) ** 2)
    if isinstance(bmi, (int, float)) and bmi > 30:
        recommendations.append(f"⚖️ **Weight Management** — BMI of {bmi:.1f} indicates obesity. A 5-10% weight reduction can significantly lower cardiovascular risk. Consult a nutritionist for a personalized plan.")
    elif isinstance(bmi, (int, float)) and bmi > 25:
        recommendations.append(f"📉 **Weight Optimization** — BMI of {bmi:.1f} is slightly elevated. Target a moderate caloric deficit and increased physical activity.")
    
    # Default recommendations if few specific ones
    if len(recommendations) < 3:
        recommendations.extend([
            "💧 **Hydration** — Drink 8-10 glasses of water daily for optimal cardiovascular function.",
            "😴 **Sleep Quality** — Aim for 7-9 hours of quality sleep. Poor sleep is linked to increased cardiovascular risk.",
            "🧘 **Stress Management** — Practice daily relaxation techniques: deep breathing, meditation, or yoga for 15-20 minutes."
        ])
    
    advice_parts["recommendations"] = "\n\n".join(recommendations[:6])
    
    # ─── Lifestyle Modifications ───
    lifestyle = [
        "🥗 **Dietary Changes** — Follow a Mediterranean or DASH diet pattern. Target: 5+ servings of fruits/vegetables daily, whole grains, lean proteins, and healthy fats.",
        "🏃 **Exercise Routine** — 150 minutes of moderate aerobic activity OR 75 minutes of vigorous activity per week, plus 2 sessions of strength training.",
        "😴 **Sleep Optimization** — Maintain consistent sleep schedule (7-9 hours). Limit screen time before bed. Create a cool, dark sleep environment.",
        "🧘 **Stress Reduction** — Practice mindfulness meditation 15-20 minutes daily. Consider journaling, nature walks, or social activities for mental well-being."
    ]
    advice_parts["lifestyle"] = "\n\n".join(lifestyle)
    
    # ─── Medical Follow-up ───
    followup = []
    if risk_category in ["High", "Critical"]:
        followup.extend([
            "🔴 **Urgent: Schedule appointment with primary care physician within 1 week**",
            "🫀 **Cardiac evaluation** — ECG, echocardiogram, and stress test recommended",
            "🩸 **Complete blood panel** — CBC, metabolic panel, lipid profile, HbA1c, thyroid function"
        ])
    elif risk_category == "Moderate":
        followup.extend([
            "🟡 **Schedule follow-up within 1 month**",
            "📊 **Lipid panel and fasting glucose test** recommended",
            "🩺 **Blood pressure monitoring** — check weekly for 1 month"
        ])
    else:
        followup.extend([
            "🟢 **Routine annual checkup** is sufficient",
            "📊 **Standard screening tests** at recommended intervals",
            "🩺 **Monitor blood pressure** at home monthly"
        ])
    advice_parts["followup"] = "\n\n".join(followup)
    
    # ─── Warnings ───
    warnings = []
    if risk_category in ["High", "Critical"]:
        warnings.append("⚠️ **Your risk score indicates significant health concerns. This is a decision-support tool—please consult a healthcare professional for proper diagnosis and treatment.**")
    if isinstance(sbp, (int, float)) and sbp > 180:
        warnings.append("🔴 **HYPERTENSIVE CRISIS WARNING** — Systolic BP >180 mmHg requires immediate medical attention.")
    if isinstance(glucose, (int, float)) and glucose > 300:
        warnings.append("🔴 **SEVERE HYPERGLYCEMIA** — Glucose >300 mg/dL requires immediate medical evaluation.")
    if not warnings:
        warnings.append("ℹ️ No immediate warnings. Continue routine health monitoring.")
    
    advice_parts["warnings"] = "\n\n".join(warnings)
    
    # Compile full advice
    full_advice = f"""## Risk Explanation

{advice_parts['risk_explanation']}

## Personalized Recommendations

{advice_parts['recommendations']}

## Lifestyle Modifications

{advice_parts['lifestyle']}

## Medical Follow-up

{advice_parts['followup']}

## Warnings

{advice_parts['warnings']}

---
*This assessment is generated by an AI health advisory system and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment decisions.*
"""
    
    advice_parts["advice"] = full_advice
    advice_parts["source"] = "rule-based"
    
    return advice_parts


def _format_patient_data(patient_data):
    """Format patient data for prompt."""
    lines = []
    for key, value in patient_data.items():
        lines.append(f"  {_humanize_feature(key)}: {value}")
    return "\n".join(lines)


def _humanize_feature(feature_name):
    """Convert feature names to human-readable labels."""
    mapping = {
        "age": "Age",
        "gender": "Gender",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "ap_hi": "Systolic Blood Pressure",
        "ap_lo": "Diastolic Blood Pressure",
        "cholesterol": "Cholesterol Level",
        "gluc": "Glucose Level",
        "smoke": "Smoking Status",
        "alco": "Alcohol Consumption",
        "active": "Physical Activity",
        "bmi": "Body Mass Index",
        "cardio": "Cardiovascular Disease",
        "heart_rate": "Heart Rate",
        "systolic_bp": "Systolic Blood Pressure",
        "temperature": "Body Temperature",
        "respiratory_rate": "Respiratory Rate",
        "spo2": "Blood Oxygen (SpO2)",
        "glucose": "Blood Glucose",
        "creatinine": "Creatinine",
        "wbc": "White Blood Cell Count",
        "hemoglobin": "Hemoglobin",
        "pulse_pressure": "Pulse Pressure",
        "age_bp_interaction": "Age × Blood Pressure",
        "bmi_chol_interaction": "BMI × Cholesterol"
    }
    return mapping.get(feature_name, feature_name.replace("_", " ").title())


def _extract_section(text, section_name):
    """Extract a section from structured text."""
    lines = text.split("\n")
    section_lines = []
    in_section = False
    
    for line in lines:
        if section_name.lower() in line.lower() and line.strip().startswith("#"):
            in_section = True
            continue
        elif line.strip().startswith("#") and in_section:
            break
        elif in_section:
            section_lines.append(line)
    
    return "\n".join(section_lines).strip()
