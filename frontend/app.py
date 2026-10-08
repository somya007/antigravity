"""
GDHRA System — Streamlit Dashboard
====================================
Professional medical dashboard for health risk assessment.
Clean, light-themed design with 5 tabs:
  1. Patient Data — input form + CSV upload
  2. Risk Assessment — gauge, trajectory, confidence
  3. Explainability — SHAP, feature importance
  4. Insights & What-If — GenAI advice, simulator
  5. Model Dashboard — metrics, ROC, calibration
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SAVE_DIR = os.path.join(PROJECT_ROOT, "saved_models")

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="GDHRA — Health Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Clean CSS Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1a1a2e;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 16px;
    }
    
    .metric-card h3 {
        margin: 0 0 8px 0;
        font-size: 14px;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        font-size: 32px;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    .risk-low { background: #dcfce7; color: #166534; }
    .risk-moderate { background: #fef3c7; color: #92400e; }
    .risk-high { background: #fee2e2; color: #991b1b; }
    .risk-critical { background: #fecdd3; color: #881337; }
    
    /* Info card */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin: 12px 0;
    }
    
    /* Confidence bar */
    .confidence-container {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .confidence-bar {
        height: 12px;
        border-radius: 6px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
    }
    
    /* Button */
    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background: #1d4ed8;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 24px 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏥 GDHRA System")
    st.markdown("*Generative Dynamic Health Risk Assessment*")
    st.markdown("---")
    
    # Dataset Selection
    st.markdown("### Dataset")
    dataset = st.selectbox(
        "Select dataset",
        ["cvd", "mimic", "wearable"],
        format_func=lambda x: {
            "cvd": "Cardiovascular Disease",
            "mimic": "MIMIC Clinical (Temporal)",
            "wearable": "Wearable Sensor Data"
        }[x],
        key="dataset_selector"
    )
    
    # Training Section
    st.markdown("### Model Training")
    
    if st.button("Generate Data", key="gen_data_btn", use_container_width=True):
        with st.spinner("Generating datasets..."):
            try:
                from data.download_datasets import (
                    ensure_data_dir, download_cvd_dataset,
                    generate_mimic_temporal, generate_clinical_notes,
                    generate_wearable_data
                )
                ensure_data_dir()
                download_cvd_dataset()
                generate_mimic_temporal()
                generate_clinical_notes()
                generate_wearable_data()
                st.success("Datasets ready!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    if st.button("Train Model", key="train_btn", use_container_width=True):
        with st.spinner(f"Training on {dataset}... (this may take 30-60s)"):
            try:
                from ml.train import train_dataset
                result = train_dataset(dataset)
                if result:
                    st.success(f"Trained! Best: {result['best_model_name']} (F1={result['evaluation_results'][result['best_model_name']]['f1']:.3f})")
                    st.rerun()
                else:
                    st.error("Training failed.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Model Status
    st.markdown("### Model Status")
    model_path = os.path.join(SAVE_DIR, f"model_{dataset}.joblib")
    if os.path.exists(model_path):
        st.success(f"Model loaded: {dataset}")
        try:
            artifacts = joblib.load(model_path)
            st.markdown(f"**Best**: {artifacts['best_model_name']}")
            best_eval = artifacts['evaluation_results'][artifacts['best_model_name']]
            st.markdown(f"**F1**: {best_eval['f1']:.3f} | **AUROC**: {best_eval['auroc']:.3f}")
        except Exception:
            pass
    else:
        st.warning("No trained model. Click 'Train Model'.")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **Key Innovations:**
    - Cross-modal attention fusion
    - MC Dropout uncertainty
    - Generative AI insights
    - What-if risk simulator
    - SHAP explainability
    """)


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────

def load_model_artifacts(dataset_name):
    """Load trained model artifacts."""
    path = os.path.join(SAVE_DIR, f"model_{dataset_name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def get_risk_category(score):
    """Categorize risk score."""
    if score < 30: return "Low"
    elif score < 60: return "Moderate"
    elif score < 80: return "High"
    else: return "Critical"


def get_risk_color(category):
    """Get color for risk category."""
    return {
        "Low": "#16a34a",
        "Moderate": "#f59e0b",
        "High": "#ef4444",
        "Critical": "#dc2626"
    }.get(category, "#6b7280")


def create_risk_gauge(score, category):
    """Create a risk score gauge chart."""
    color = get_risk_color(category)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": "#1a1a2e"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e2e8f0"},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#f8f9fa",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 80], "color": "#fee2e2"},
                {"range": [80, 100], "color": "#fecdd3"}
            ],
        },
        domain={"x": [0, 1], "y": [0, 1]}
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"}
    )
    
    return fig


def create_feature_importance_chart(importance_dict, top_n=10):
    """Create horizontal bar chart for feature importance."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    names = [item[0].replace("_", " ").title() for item in sorted_items]
    values = [abs(item[1]) for item in sorted_items]
    colors = ["#ef4444" if item[1] > 0 else "#16a34a" for item in sorted_items]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside"
    ))
    
    fig.update_layout(
        height=max(300, len(names) * 35),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance",
        yaxis={"autorange": "reversed"},
        font={"family": "Inter", "size": 12}
    )
    
    return fig


# ─────────────────────────────────────────────
# Main Content — Tabs
# ─────────────────────────────────────────────

st.markdown("# 🏥 Health Risk Assessment Dashboard")
st.markdown("*Multimodal, uncertainty-aware risk prediction with generative AI insights*")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Patient Data",
    "📊 Risk Assessment",
    "🔍 Explainability",
    "💡 Insights & What-If",
    "📈 Model Dashboard"
])


# ═══════════════════════════════════════════════
# TAB 1: Patient Data Input
# ═══════════════════════════════════════════════

with tab1:
    st.markdown("## Patient Data Entry")
    
    input_method = st.radio("Input method", ["Manual Entry", "Upload CSV"], horizontal=True)
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
        if uploaded_file:
            patient_df = pd.read_csv(uploaded_file)
            st.dataframe(patient_df.head(10), use_container_width=True)
            st.session_state["patient_csv"] = patient_df
            st.success(f"Loaded {len(patient_df)} patients.")
    
    else:
        st.markdown("### Enter Patient Information")
        
        if dataset == "cvd":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age (years)", 18, 100, 50, key="age_input")
                gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male", key="gender_input")
                height = st.number_input("Height (cm)", 100, 220, 170, key="height_input")
                weight = st.number_input("Weight (kg)", 30, 200, 75, key="weight_input")
            
            with col2:
                ap_hi = st.number_input("Systolic BP (mmHg)", 70, 250, 125, key="ap_hi_input")
                ap_lo = st.number_input("Diastolic BP (mmHg)", 40, 160, 82, key="ap_lo_input")
                cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x], key="chol_input")
                gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x], key="gluc_input")
            
            with col3:
                smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="smoke_input")
                alco = st.selectbox("Alcohol", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="alco_input")
                active = st.selectbox("Physical Activity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="active_input")
            
            patient_data = {
                "age": age * 365,  # Convert to days for model
                "gender": gender,
                "height": height,
                "weight": weight,
                "ap_hi": ap_hi,
                "ap_lo": ap_lo,
                "cholesterol": cholesterol,
                "gluc": gluc,
                "smoke": smoke,
                "alco": alco,
                "active": active
            }
            
            # Display-friendly version
            patient_display = patient_data.copy()
            patient_display["age"] = age  # Show years
            patient_display["bmi"] = round(weight / ((height / 100) ** 2), 1)
        
        elif dataset == "mimic":
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", 18, 100, 60, key="age_m")
                gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="gender_m")
                heart_rate = st.number_input("Heart Rate (bpm)", 40, 180, 78, key="hr_m")
                sbp = st.number_input("Systolic BP", 70, 220, 130, key="sbp_m")
            
            with col2:
                temp = st.number_input("Temperature (°C)", 35.0, 41.0, 37.0, step=0.1, key="temp_m")
                resp_rate = st.number_input("Respiratory Rate", 8, 40, 16, key="rr_m")
                spo2 = st.number_input("SpO2 (%)", 80, 100, 97, key="spo2_m")
                glucose = st.number_input("Glucose (mg/dL)", 40, 400, 110, key="gluc_m")
            
            with col3:
                creatinine = st.number_input("Creatinine", 0.3, 10.0, 1.0, step=0.1, key="creat_m")
                wbc = st.number_input("WBC (×10³/µL)", 1.0, 30.0, 7.5, step=0.5, key="wbc_m")
                hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.5, step=0.5, key="hgb_m")
            
            patient_data = {
                "age": age, "gender": gender, "heart_rate": heart_rate,
                "systolic_bp": sbp, "temperature": temp, "respiratory_rate": resp_rate,
                "spo2": spo2, "glucose": glucose, "creatinine": creatinine,
                "wbc": wbc, "hemoglobin": hemoglobin
            }
            patient_display = patient_data.copy()
        
        elif dataset == "wearable":
            col1, col2 = st.columns(2)
            
            with col1:
                resting_hr = st.number_input("Resting HR (bpm)", 40, 120, 68, key="rhr_w")
                daily_steps = st.number_input("Daily Steps", 0, 30000, 7500, key="steps_w")
            
            with col2:
                sleep_hours = st.number_input("Sleep (hours)", 2.0, 12.0, 7.0, step=0.5, key="sleep_w")
                hrv = st.number_input("HRV (ms)", 10, 100, 45, key="hrv_w")
            
            patient_data = {
                "resting_hr": resting_hr, "daily_steps": daily_steps,
                "sleep_hours": sleep_hours, "hrv_ms": hrv
            }
            patient_display = patient_data.copy()
        
        st.session_state["patient_data"] = patient_data
        st.session_state["patient_display"] = patient_display
        
        # Predict button
        st.markdown("---")
        predict_clicked = st.button("🔬 Predict Risk", key="predict_btn", use_container_width=True)
        
        if predict_clicked:
            artifacts = load_model_artifacts(dataset)
            if artifacts is None:
                st.error("No trained model found. Please train a model first (sidebar).")
            else:
                with st.spinner("Computing risk assessment..."):
                    try:
                        model = artifacts["best_model"]
                        scaler = artifacts["scaler"]
                        feature_names = artifacts["feature_names"]
                        
                        # Build feature vector
                        features = []
                        for fname in feature_names:
                            val = patient_data.get(fname, 0)
                            # Calculate derived features
                            if fname == "bmi" and "bmi" not in patient_data:
                                w = patient_data.get("weight", 70)
                                h = patient_data.get("height", 170)
                                val = w / ((h / 100) ** 2)
                            elif fname == "pulse_pressure" and "pulse_pressure" not in patient_data:
                                val = patient_data.get("ap_hi", 120) - patient_data.get("ap_lo", 80)
                            elif fname == "age_bp_interaction" and "age_bp_interaction" not in patient_data:
                                val = patient_data.get("age", 50*365) / 365.25 * patient_data.get("ap_hi", 120)
                            elif fname == "bmi_chol_interaction" and "bmi_chol_interaction" not in patient_data:
                                w = patient_data.get("weight", 70)
                                h = patient_data.get("height", 170)
                                bmi_val = w / ((h / 100) ** 2)
                                val = bmi_val * patient_data.get("cholesterol", 1)
                            features.append(float(val) if isinstance(val, (int, float)) else 0.0)
                        
                        features_scaled = scaler.transform([features])
                        
                        # Predict
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(features_scaled)[0]
                            risk_prob = prob[1] if len(prob) > 1 else prob[0]
                        else:
                            risk_prob = float(model.predict(features_scaled)[0])
                        
                        risk_score = round(risk_prob * 100, 1)
                        risk_cat = get_risk_category(risk_score)
                        
                        # Uncertainty
                        from ml.models.uncertainty import tabular_uncertainty
                        unc = tabular_uncertainty(model, features_scaled)
                        
                        # SHAP
                        from explainability.shap_explainer import compute_shap_values, explain_patient
                        shap_result = compute_shap_values(model, artifacts.get("X_test", features_scaled), feature_names, max_samples=50)
                        patient_explanation = explain_patient(model, features, feature_names)
                        
                        # GenAI advice
                        from genai.advisor import generate_health_advice
                        advice = generate_health_advice(
                            risk_score, risk_cat, patient_display, dataset,
                            shap_result.get("feature_importance", {})
                        )
                        
                        # Store results
                        st.session_state["risk_score"] = risk_score
                        st.session_state["risk_category"] = risk_cat
                        st.session_state["uncertainty"] = unc
                        st.session_state["shap_result"] = shap_result
                        st.session_state["patient_explanation"] = patient_explanation
                        st.session_state["advice"] = advice
                        st.session_state["feature_names"] = feature_names
                        st.session_state["features_scaled"] = features_scaled
                        
                        st.success(f"Risk Score: {risk_score}/100 ({risk_cat})")
                        
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        import traceback
                        st.code(traceback.format_exc())


# ═══════════════════════════════════════════════
# TAB 2: Risk Assessment
# ═══════════════════════════════════════════════

with tab2:
    if "risk_score" not in st.session_state:
        st.info("Enter patient data and click **Predict Risk** in the Patient Data tab.")
    else:
        risk_score = st.session_state["risk_score"]
        risk_cat = st.session_state["risk_category"]
        uncertainty = st.session_state.get("uncertainty", {})
        
        st.markdown("## Risk Assessment Results")
        
        # Main risk metrics row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.plotly_chart(create_risk_gauge(risk_score, risk_cat), use_container_width=True)
        
        with col2:
            badge_class = f"risk-{risk_cat.lower()}"
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>Risk Category</h3>
                <span class="risk-badge {badge_class}">{risk_cat}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h3>Risk Score</h3>
                <div class="value">{risk_score}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            reliability = uncertainty.get("reliability", "N/A")
            unc_score = uncertainty.get("uncertainty_score", 0)
            confidence_pct = max(0, min(100, (1 - unc_score * 5) * 100))
            
            conf_color = "#16a34a" if confidence_pct > 70 else ("#f59e0b" if confidence_pct > 40 else "#ef4444")
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Prediction Confidence</h3>
                <div style="font-size: 18px; font-weight: 600; color: {conf_color}; margin-bottom: 8px;">
                    {reliability}
                </div>
                <div class="confidence-container">
                    <div class="confidence-bar" style="width: {confidence_pct}%; background: {conf_color};"></div>
                </div>
                <div style="font-size: 12px; color: #6b7280; margin-top: 4px;">
                    Uncertainty: {unc_score:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Trajectory (simulated)
        st.markdown("### Risk Trajectory Simulation")
        st.markdown("*Projected risk evolution based on current parameters*")
        
        trajectory_months = list(range(0, 13))
        base_risk = risk_score
        
        # Simulate trajectory based on risk level
        if risk_cat == "Low":
            trajectory = [base_risk + np.random.normal(0, 1) + i * 0.2 for i in trajectory_months]
        elif risk_cat == "Moderate":
            trajectory = [base_risk + np.random.normal(0, 2) + i * 0.5 for i in trajectory_months]
        else:
            trajectory = [base_risk + np.random.normal(0, 3) + i * 0.8 for i in trajectory_months]
        
        trajectory = [max(0, min(100, t)) for t in trajectory]
        
        # Intervention trajectory
        intervention = [base_risk - i * (base_risk * 0.03) + np.random.normal(0, 1) for i in trajectory_months]
        intervention = [max(0, min(100, t)) for t in intervention]
        
        fig_trajectory = go.Figure()
        fig_trajectory.add_trace(go.Scatter(
            x=trajectory_months, y=trajectory,
            mode="lines+markers", name="Without Intervention",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=6)
        ))
        fig_trajectory.add_trace(go.Scatter(
            x=trajectory_months, y=intervention,
            mode="lines+markers", name="With Intervention",
            line=dict(color="#16a34a", width=2, dash="dash"),
            marker=dict(size=6)
        ))
        
        fig_trajectory.add_hline(y=60, line_dash="dot", line_color="#f59e0b",
                                 annotation_text="High Risk Threshold")
        
        fig_trajectory.update_layout(
            height=350,
            xaxis_title="Months from Now",
            yaxis_title="Risk Score",
            yaxis_range=[0, 100],
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter"}
        )
        
        st.plotly_chart(fig_trajectory, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3: Explainability
# ═══════════════════════════════════════════════

with tab3:
    if "shap_result" not in st.session_state:
        st.info("Run a prediction first to see explainability results.")
    else:
        shap_result = st.session_state["shap_result"]
        patient_explanation = st.session_state.get("patient_explanation", {})
        
        st.markdown("## Feature Importance & Explainability")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Top Contributing Factors")
            if shap_result.get("feature_importance"):
                fig = create_feature_importance_chart(shap_result["feature_importance"])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Patient-Level Analysis")
            
            if patient_explanation and patient_explanation.get("top_risk_factors"):
                for factor in patient_explanation["top_risk_factors"]:
                    direction_icon = "🔴" if factor["shap_value"] > 0 else "🟢"
                    st.markdown(f"""
                    <div class="info-card">
                        {direction_icon} <strong>{factor['feature'].replace('_', ' ').title()}</strong><br>
                        <small>Impact: {abs(factor['shap_value']):.3f} — {factor['direction']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            if patient_explanation and patient_explanation.get("top_protective"):
                st.markdown("#### Protective Factors")
                for factor in patient_explanation["top_protective"]:
                    st.markdown(f"🟢 **{factor['feature'].replace('_', ' ').title()}** — reduces risk by {abs(factor['shap_value']):.3f}")
        
        # Feature importance table
        st.markdown("### Detailed Feature Importance Table")
        if shap_result.get("feature_importance"):
            imp_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Importance": round(v, 4)}
                for k, v in shap_result["feature_importance"].items()
            ])
            st.dataframe(imp_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════
# TAB 4: Insights & What-If
# ═══════════════════════════════════════════════

with tab4:
    if "advice" not in st.session_state:
        st.info("Run a prediction first to see AI-generated insights.")
    else:
        advice = st.session_state["advice"]
        
        st.markdown("## AI Health Recommendations")
        
        # Source indicator
        source = advice.get("source", "unknown")
        if source == "gemini":
            st.success("Powered by Google Gemini AI")
        else:
            st.info("Powered by Medical Knowledge Base (set GEMINI_API_KEY for AI-generated insights)")
        
        # Display advice sections
        if advice.get("advice"):
            st.markdown(advice["advice"])
        
        # What-If Simulator
        st.markdown("---")
        st.markdown("## What-If Risk Simulator")
        st.markdown("*Modify parameters to see how lifestyle changes affect your risk*")
        
        if "patient_data" in st.session_state and "risk_score" in st.session_state:
            from genai.whatif import get_preset_scenarios, simulate_risk_change
            
            # Preset scenarios
            st.markdown("### Quick Scenarios")
            scenarios = get_preset_scenarios()
            
            scenario_cols = st.columns(min(4, len(scenarios)))
            for i, scenario in enumerate(scenarios[:4]):
                with scenario_cols[i]:
                    if st.button(scenario["name"], key=f"scenario_{i}", use_container_width=True):
                        result = simulate_risk_change(
                            st.session_state["patient_data"],
                            scenario["modifications"],
                            dataset
                        )
                        st.session_state["whatif_result"] = result
                        st.session_state["whatif_scenario"] = scenario["name"]
            
            # Custom modification
            st.markdown("### Custom Parameter Modification")
            
            if dataset == "cvd":
                wif_col1, wif_col2, wif_col3 = st.columns(3)
                with wif_col1:
                    new_ap_hi = st.slider("Systolic BP", 80, 220, 
                                         int(st.session_state["patient_data"].get("ap_hi", 120)),
                                         key="wif_ap_hi")
                with wif_col2:
                    new_chol = st.selectbox("Cholesterol", [1, 2, 3],
                                           index=int(st.session_state["patient_data"].get("cholesterol", 1)) - 1,
                                           format_func=lambda x: {1: "Normal", 2: "Above", 3: "High"}[x],
                                           key="wif_chol")
                with wif_col3:
                    new_smoke = st.selectbox("Smoking", [0, 1],
                                           index=int(st.session_state["patient_data"].get("smoke", 0)),
                                           format_func=lambda x: "No" if x == 0 else "Yes",
                                           key="wif_smoke")
                
                if st.button("Simulate Changes", key="simulate_btn"):
                    modifications = {}
                    if new_ap_hi != st.session_state["patient_data"].get("ap_hi"):
                        modifications["ap_hi"] = new_ap_hi
                    if new_chol != st.session_state["patient_data"].get("cholesterol"):
                        modifications["cholesterol"] = new_chol
                    if new_smoke != st.session_state["patient_data"].get("smoke"):
                        modifications["smoke"] = new_smoke
                    
                    if modifications:
                        result = simulate_risk_change(
                            st.session_state["patient_data"], modifications, dataset
                        )
                        st.session_state["whatif_result"] = result
                        st.session_state["whatif_scenario"] = "Custom"
                    else:
                        st.warning("No changes made.")
            
            # Display what-if results
            if "whatif_result" in st.session_state:
                result = st.session_state["whatif_result"]
                
                st.markdown(f"### Simulation Result: {st.session_state.get('whatif_scenario', '')}")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Original Risk", f"{result['original_risk']}", 
                             delta=None)
                with res_col2:
                    st.metric("Modified Risk", f"{result['modified_risk']}",
                             delta=f"{result['risk_change']:+.1f}",
                             delta_color="inverse")
                with res_col3:
                    direction_emoji = "📈" if result["direction"] == "worsened" else ("📉" if result["direction"] == "improved" else "➡️")
                    st.metric("Impact", f"{direction_emoji} {result['direction'].title()}")
                
                # Per-feature impact
                if result.get("impact_details"):
                    st.markdown("#### Per-Factor Impact")
                    for detail in result["impact_details"]:
                        st.markdown(f"  {detail['direction']} **{detail['feature'].replace('_', ' ').title()}**: "
                                   f"{detail['old_value']} → {detail['new_value']} "
                                   f"(risk impact: {detail['risk_impact']:+.1f})")


# ═══════════════════════════════════════════════
# TAB 5: Model Dashboard
# ═══════════════════════════════════════════════

with tab5:
    artifacts = load_model_artifacts(dataset)
    
    if artifacts is None:
        st.info("No trained model. Train a model from the sidebar to see the dashboard.")
    else:
        st.markdown("## Model Performance Dashboard")
        
        eval_results = artifacts.get("evaluation_results", {})
        
        if not eval_results:
            st.warning("No evaluation results found.")
        else:
            # Metrics table
            st.markdown("### Model Comparison")
            
            metrics_data = []
            for name, metrics in eval_results.items():
                metrics_data.append({
                    "Model": name,
                    "Accuracy": round(metrics["accuracy"], 4),
                    "Precision": round(metrics["precision"], 4),
                    "Recall": round(metrics["recall"], 4),
                    "F1 Score": round(metrics["f1"], 4),
                    "AUROC": round(metrics["auroc"], 4),
                    "AUPRC": round(metrics["auprc"], 4),
                    "Brier Score": round(metrics["brier_score"], 4)
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Charts row
            chart_col1, chart_col2 = st.columns(2)
            
            # ROC Curves
            with chart_col1:
                st.markdown("### ROC Curves")
                fig_roc = go.Figure()
                
                colors = px.colors.qualitative.Set2
                for i, (name, metrics) in enumerate(eval_results.items()):
                    roc_data = metrics.get("roc_curve", {})
                    if roc_data:
                        fig_roc.add_trace(go.Scatter(
                            x=roc_data["fpr"],
                            y=roc_data["tpr"],
                            mode="lines",
                            name=f"{name} (AUC={metrics['auroc']:.3f})",
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Random",
                    line=dict(color="#cbd5e1", dash="dash", width=1)
                ))
                
                fig_roc.update_layout(
                    height=380,
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    legend=dict(orientation="h", yanchor="top", y=-0.2),
                    margin=dict(l=40, r=20, t=20, b=80),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter"}
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Calibration Curves
            with chart_col2:
                st.markdown("### Calibration Curves")
                fig_cal = go.Figure()
                
                for i, (name, metrics) in enumerate(eval_results.items()):
                    cal_data = metrics.get("calibration_curve", {})
                    if cal_data and cal_data.get("fraction_of_positives"):
                        fig_cal.add_trace(go.Scatter(
                            x=cal_data["mean_predicted_value"],
                            y=cal_data["fraction_of_positives"],
                            mode="lines+markers",
                            name=name,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=6)
                        ))
                
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines", name="Perfectly Calibrated",
                    line=dict(color="#cbd5e1", dash="dash", width=1)
                ))
                
                fig_cal.update_layout(
                    height=380,
                    xaxis_title="Mean Predicted Probability",
                    yaxis_title="Fraction of Positives",
                    legend=dict(orientation="h", yanchor="top", y=-0.2),
                    margin=dict(l=40, r=20, t=20, b=80),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter"}
                )
                
                st.plotly_chart(fig_cal, use_container_width=True)
            
            # Confusion Matrix for best model
            st.markdown("### Confusion Matrix (Best Model)")
            best_name = artifacts.get("best_model_name", list(eval_results.keys())[0])
            cm = eval_results[best_name].get("confusion_matrix", [[0, 0], [0, 0]])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted Negative", "Predicted Positive"],
                y=["Actual Negative", "Actual Positive"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 18},
                hoverinfo="skip"
            ))
            
            fig_cm.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"}
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Cross-validation scores
            st.markdown("### Cross-Validation Scores")
            cv_data = artifacts.get("cv_scores", {})
            if cv_data:
                cv_df = pd.DataFrame([
                    {"Model": name, "CV Score (F1)": round(score, 4)}
                    for name, score in cv_data.items()
                ])
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
            
            # Model info
            st.markdown("### Training Summary")
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.metric("Training Samples", artifacts.get("n_train", "N/A"))
                st.metric("Best Model", artifacts.get("best_model_name", "N/A"))
            with info_col2:
                st.metric("Test Samples", artifacts.get("n_test", "N/A"))
                unc = artifacts.get("uncertainty", {})
                st.metric("Model Reliability", unc.get("reliability", "N/A"))
