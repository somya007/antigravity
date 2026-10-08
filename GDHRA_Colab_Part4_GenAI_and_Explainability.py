# ============================================================
# GDHRA System — Google Colab (Part 4: GenAI, SHAP, Visuals)
# ============================================================

# %% [markdown]
# ## 🤖 Step 4: Generative AI Health Advisor + Explainability

# %% — Cell 13: Medical Knowledge Base (RAG)
MEDICAL_KNOWLEDGE = {
    "cardiovascular": {
        "description": "Cardiovascular disease encompasses conditions affecting the heart and blood vessels.",
        "risk_factors": ["Hypertension (BP >140)", "High cholesterol", "Diabetes", "Smoking", "Obesity (BMI >30)", "Sedentary lifestyle"],
        "guidelines": ["BP target <130/80 for high-risk", "150 min/week moderate exercise", "Sodium <2300 mg/day", "LDL target <70 for very high risk"],
        "interventions": ["DASH diet", "Mediterranean diet", "5-10% weight loss", "Smoking cessation", "Stress management"]
    },
    "diabetes": {
        "description": "Metabolic disorder characterized by chronic hyperglycemia.",
        "risk_factors": ["Fasting glucose >126", "HbA1c >6.5%", "Obesity", "Family history", "Sedentary lifestyle"],
        "guidelines": ["HbA1c target <7%", "Fasting glucose 80-130", "Annual kidney function monitoring"],
        "interventions": ["Low glycemic index diet", "150 min/week exercise + resistance training", "Carb counting"]
    },
    "general_wellness": {
        "description": "General wellness guidelines for healthy individuals.",
        "risk_factors": [],
        "guidelines": ["Annual physical exam", "Age-appropriate screenings", "Mental health check annually"],
        "interventions": ["Balanced diet", "150-300 min/week moderate activity", "7-9 hours sleep", "Stress management"]
    }
}

def get_relevant_knowledge(patient_data, risk_category):
    """RAG: Retrieve relevant medical knowledge based on patient data."""
    conditions = ["cardiovascular"]
    glucose = patient_data.get("gluc", patient_data.get("glucose", 0))
    if isinstance(glucose, (int, float)) and (glucose >= 2 or glucose > 100):
        conditions.append("diabetes")
    if risk_category == "Low":
        conditions.append("general_wellness")
    parts = []
    for cond in conditions:
        if cond in MEDICAL_KNOWLEDGE:
            kb = MEDICAL_KNOWLEDGE[cond]
            parts.append(f"**{cond.replace('_', ' ').title()}**: {kb['description']}")
            parts.append("Guidelines: " + "; ".join(kb["guidelines"][:3]))
            parts.append("Interventions: " + "; ".join(kb["interventions"][:3]))
    return "\n".join(parts)

# %% — Cell 14: GenAI Health Advisor
def humanize_feature(name):
    mapping = {
        "age": "Age", "gender": "Gender", "ap_hi": "Systolic BP", "ap_lo": "Diastolic BP",
        "cholesterol": "Cholesterol", "gluc": "Glucose", "smoke": "Smoking", "alco": "Alcohol",
        "active": "Physical Activity", "bmi": "BMI", "pulse_pressure": "Pulse Pressure",
        "weight": "Weight", "height": "Height"
    }
    return mapping.get(name, name.replace("_", " ").title())

def generate_health_advice(risk_score, risk_category, patient_data, feature_importances=None):
    """Generate personalized health advice (Gemini API or fallback)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            medical_context = get_relevant_knowledge(patient_data, risk_category)
            imp_text = ""
            if feature_importances:
                top = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                imp_text = "\n".join([f"  - {humanize_feature(f)}: {v:.3f}" for f, v in top])
            prompt = f"""You are a clinical decision support system.
PATIENT: {', '.join(f'{humanize_feature(k)}: {v}' for k, v in patient_data.items())}
RISK: {risk_score}/100 ({risk_category})
TOP FACTORS:\n{imp_text or 'N/A'}
MEDICAL CONTEXT:\n{medical_context}

Provide: 1) Risk Explanation 2) 5 Personalized Recommendations 3) Lifestyle Modifications 4) Medical Follow-up 5) Warnings
Keep language clear and patient-appropriate. Do not make diagnoses."""
            response = model.generate_content(prompt)
            return {"advice": response.text, "source": "gemini"}
        except Exception as e:
            print(f"[Gemini failed: {e}] Using fallback")

    # ─── Rule-based fallback ───
    lines = [f"## Risk Analysis: {risk_score}/100 ({risk_category})\n"]

    if risk_category == "Low":
        lines.append("Your health indicators are within a healthy range. Continue maintaining your current habits.\n")
    elif risk_category == "Moderate":
        lines.append("Some indicators suggest elevated risk. Targeted lifestyle changes can significantly improve your outlook.\n")
    else:
        lines.append("⚠️ Multiple risk factors indicate elevated risk. Prompt medical attention is recommended.\n")

    if feature_importances:
        lines.append("### Top Contributing Factors")
        for f, v in sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            direction = "increases" if v > 0 else "decreases"
            lines.append(f"  • **{humanize_feature(f)}** — {direction} risk (impact: {abs(v):.1%})")

    lines.append("\n### Personalized Recommendations")
    recs = []
    sbp = patient_data.get("ap_hi", 0)
    if isinstance(sbp, (int, float)) and sbp > 140:
        recs.append(f"🫀 **Blood Pressure**: Your BP ({sbp} mmHg) is elevated. Target <130/80 through sodium reduction and exercise.")
    chol = patient_data.get("cholesterol", 0)
    if isinstance(chol, (int, float)) and chol >= 2:
        recs.append("🥗 **Cholesterol**: Increase fiber (25-30g/day), omega-3 fatty acids. Limit saturated fats.")
    if patient_data.get("smoke", 0) > 0:
        recs.append("🚭 **Smoking**: Quitting reduces heart disease risk by 50% within 1 year.")
    if patient_data.get("active", 1) == 0:
        recs.append("🏃 **Exercise**: Start with 30 min moderate exercise 5 days/week.")
    bmi = patient_data.get("bmi", 0)
    if isinstance(bmi, (int, float)) and bmi > 30:
        recs.append(f"⚖️ **Weight**: BMI {bmi:.1f} indicates obesity. 5-10% weight reduction can significantly lower risk.")
    recs += ["💧 **Hydration**: 8-10 glasses water daily.", "😴 **Sleep**: 7-9 hours quality sleep.", "🧘 **Stress**: 15-20 min meditation/yoga daily."]
    lines.extend(recs[:6])

    lines.append("\n### Medical Follow-up")
    if risk_category in ["High", "Critical"]:
        lines.extend(["🔴 Schedule physician appointment within 1 week", "🫀 Cardiac evaluation (ECG, echo, stress test)", "🩸 Full blood panel recommended"])
    else:
        lines.extend(["🟢 Routine annual checkup sufficient", "📊 Standard screening at recommended intervals"])

    lines.append("\n---\n*AI advisory system — not a substitute for professional medical advice.*")
    return {"advice": "\n".join(lines), "source": "rule-based"}

# %% — Cell 15: SHAP Explainability
print("Computing SHAP feature importance...")
try:
    import shap
    explainer = shap.TreeExplainer(best_model_cvd) if hasattr(best_model_cvd, "feature_importances_") or type(best_model_cvd).__name__ in ["XGBClassifier", "LGBMClassifier", "RandomForestClassifier", "GradientBoostingClassifier"] else None
    if explainer:
        shap_values = explainer.shap_values(X_test_cvd[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = {feat_cvd[i]: float(mean_abs_shap[i]) for i in range(min(len(feat_cvd), len(mean_abs_shap)))}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        print("✅ SHAP computed successfully!")
        # SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_cvd[:100], feature_names=feat_cvd, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ SHAP summary plot generated!")
    else:
        raise ValueError("Non-tree model")
except Exception as e:
    print(f"[SHAP TreeExplainer unavailable: {e}] Using model feature importance")
    if hasattr(best_model_cvd, "feature_importances_"):
        imp = best_model_cvd.feature_importances_
    elif hasattr(best_model_cvd, "coef_"):
        imp = np.abs(best_model_cvd.coef_).flatten()
    else:
        imp = np.ones(len(feat_cvd)) / len(feat_cvd)
    feature_importance = {feat_cvd[i]: float(imp[i]) for i in range(min(len(feat_cvd), len(imp)))}
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    print("✅ Feature importance computed (fallback)!")

print("\nTop 10 features:")
for i, (f, v) in enumerate(list(feature_importance.items())[:10], 1):
    bar = "█" * int(v * 100)
    print(f"  {i}. {humanize_feature(f):30s} {v:.4f} {bar}")

# %% — Cell 16: What-If Risk Simulator
def whatif_simulate(patient_data, modifications, model, scaler, feature_names):
    """Simulate risk changes when patient parameters are modified."""
    def to_features(data):
        feats = []
        for fn in feature_names:
            v = data.get(fn, 0)
            if fn == "bmi" and "bmi" not in data:
                v = data.get("weight", 70) / ((data.get("height", 170) / 100) ** 2)
            elif fn == "pulse_pressure" and "pulse_pressure" not in data:
                v = data.get("ap_hi", 120) - data.get("ap_lo", 80)
            elif fn == "age_bp_interaction" and "age_bp_interaction" not in data:
                v = (data.get("age", 18250) / 365.25) * data.get("ap_hi", 120)
            elif fn == "bmi_chol_interaction" and "bmi_chol_interaction" not in data:
                bmi = data.get("weight", 70) / ((data.get("height", 170) / 100) ** 2)
                v = bmi * data.get("cholesterol", 1)
            feats.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return feats

    orig_scaled = scaler.transform([to_features(patient_data)])
    mod_data = {**patient_data, **modifications}
    mod_scaled = scaler.transform([to_features(mod_data)])
    orig_risk = round(model.predict_proba(orig_scaled)[0][1] * 100, 1)
    mod_risk = round(model.predict_proba(mod_scaled)[0][1] * 100, 1)
    return {"original_risk": orig_risk, "modified_risk": mod_risk, "change": round(mod_risk - orig_risk, 1)}

# Demo what-if
demo_patient = {"age": 50 * 365, "gender": 1, "height": 170, "weight": 85, "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2, "smoke": 1, "alco": 0, "active": 0}
print("\n" + "=" * 60)
print("WHAT-IF RISK SIMULATOR")
print("=" * 60)
scenarios = [
    ("🚭 Quit Smoking", {"smoke": 0}),
    ("🏃 Start Exercising", {"active": 1}),
    ("🥗 Lower Cholesterol", {"cholesterol": 1}),
    ("💊 Control BP (120/80)", {"ap_hi": 120, "ap_lo": 80}),
    ("🌟 All Changes", {"smoke": 0, "active": 1, "cholesterol": 1, "ap_hi": 120, "ap_lo": 80}),
]
print(f"\nBase patient: Age=50, BP=150/95, Chol=High, Smoker, Sedentary\n")
for name, mods in scenarios:
    result = whatif_simulate(demo_patient, mods, best_model_cvd, scaler_cvd, feat_cvd)
    arrow = "📉" if result["change"] < 0 else ("📈" if result["change"] > 0 else "➡️")
    print(f"  {name:30s}  {result['original_risk']}% → {result['modified_risk']}%  ({result['change']:+.1f}%) {arrow}")
