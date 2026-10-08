"""
Medical Knowledge Base (RAG)
=============================
Retrieval-Augmented Generation knowledge store for health recommendations.
Provides condition-specific medical guidelines for the GenAI advisor.
"""


# Medical knowledge organized by condition and risk factor
MEDICAL_KNOWLEDGE = {
    "cardiovascular": {
        "description": "Cardiovascular disease encompasses conditions affecting the heart and blood vessels.",
        "risk_factors": [
            "Hypertension (systolic BP >140 mmHg)",
            "High cholesterol (LDL >160 mg/dL)",
            "Diabetes (HbA1c >6.5%)",
            "Smoking",
            "Obesity (BMI >30)",
            "Sedentary lifestyle",
            "Family history of heart disease",
            "Age >55 (men) or >65 (women)"
        ],
        "guidelines": [
            "ACC/AHA 2019: Blood pressure target <130/80 mmHg for high-risk patients",
            "AHA recommends 150 min/week of moderate aerobic exercise",
            "Dietary sodium intake <2,300 mg/day (ideally <1,500 mg)",
            "LDL cholesterol target <70 mg/dL for very high-risk patients",
            "Statin therapy recommended for 10-year ASCVD risk >7.5%"
        ],
        "interventions": {
            "lifestyle": [
                "DASH diet: rich in fruits, vegetables, whole grains, low-fat dairy",
                "Mediterranean diet: olive oil, fish, nuts, and legumes",
                "Weight loss of 5-10% body weight for overweight patients",
                "Smoking cessation with behavioral and pharmacological support",
                "Stress management through mindfulness and relaxation techniques"
            ],
            "monitoring": [
                "Blood pressure: home monitoring 2x daily",
                "Lipid panel: every 4-6 weeks until target reached",
                "Fasting glucose: quarterly if pre-diabetic",
                "ECG: annual for high-risk patients",
                "Cardiac calcium score for intermediate risk assessment"
            ]
        }
    },
    "diabetes": {
        "description": "Diabetes mellitus is a metabolic disorder characterized by chronic hyperglycemia.",
        "risk_factors": [
            "Fasting glucose >126 mg/dL",
            "HbA1c >6.5%",
            "Obesity (BMI >30)",
            "Family history of diabetes",
            "Sedentary lifestyle",
            "Gestational diabetes history",
            "Polycystic ovary syndrome"
        ],
        "guidelines": [
            "ADA 2024: HbA1c target <7% for most adults",
            "Fasting glucose target: 80-130 mg/dL",
            "Blood pressure target: <130/80 mmHg",
            "Annual comprehensive metabolic panel",
            "Kidney function monitoring (eGFR, uACR) annually"
        ],
        "interventions": {
            "lifestyle": [
                "Low glycemic index diet focusing on whole grains and fiber",
                "Regular physical activity: 150 min/week aerobic + resistance training",
                "Weight management: 7% body weight loss for pre-diabetes prevention",
                "Carbohydrate counting and meal timing consistency",
                "Continuous glucose monitoring for better glycemic control"
            ],
            "monitoring": [
                "Self-monitoring of blood glucose: 4-10 times daily",
                "HbA1c: every 3 months until target reached",
                "Foot examination: annual comprehensive + daily self-check",
                "Eye examination: annual dilated retinal exam",
                "Lipid panel: annual"
            ]
        }
    },
    "metabolic_syndrome": {
        "description": "Metabolic syndrome is a cluster of conditions increasing heart disease, stroke, and diabetes risk.",
        "risk_factors": [
            "Waist circumference >40 inches (men) or >35 inches (women)",
            "Triglycerides ≥150 mg/dL",
            "HDL cholesterol <40 mg/dL (men) or <50 mg/dL (women)",
            "Blood pressure ≥130/85 mmHg",
            "Fasting glucose ≥100 mg/dL"
        ],
        "guidelines": [
            "At least 3 of 5 criteria must be present for diagnosis",
            "Primary treatment is therapeutic lifestyle change",
            "5-10% weight loss can resolve metabolic syndrome",
            "Regular physical activity is the cornerstone of treatment"
        ],
        "interventions": {
            "lifestyle": [
                "Caloric restriction for weight loss",
                "Increased physical activity to 60 min/day",
                "Heart-healthy dietary pattern",
                "Smoking cessation",
                "Alcohol moderation"
            ],
            "monitoring": [
                "Quarterly metabolic panel",
                "Monthly weight and waist circumference",
                "Blood pressure monitoring twice weekly"
            ]
        }
    },
    "general_wellness": {
        "description": "General wellness guidelines for healthy individuals.",
        "risk_factors": [],
        "guidelines": [
            "Annual physical examination",
            "Age-appropriate cancer screenings",
            "Immunization per CDC schedule",
            "Mental health screening annually"
        ],
        "interventions": {
            "lifestyle": [
                "Balanced diet rich in fruits, vegetables, whole grains",
                "Regular physical activity: 150-300 min/week moderate intensity",
                "7-9 hours of quality sleep",
                "Stress management and social connections",
                "Limited alcohol consumption"
            ],
            "monitoring": [
                "Annual physical exam",
                "Blood pressure check annually (more often if elevated)",
                "Cholesterol screening every 4-6 years (more often if risk factors)",
                "Blood glucose screening every 3 years after age 45"
            ]
        }
    }
}


def get_relevant_knowledge(patient_data, risk_category):
    """
    Retrieve relevant medical knowledge based on patient data.
    
    Args:
        patient_data: dict of patient features
        risk_category: "Low" / "Moderate" / "High" / "Critical"
    
    Returns:
        Formatted string of relevant medical knowledge
    """
    relevant_conditions = []
    
    # Determine relevant conditions based on patient data
    sbp = patient_data.get("ap_hi", patient_data.get("systolic_bp", 0))
    chol = patient_data.get("cholesterol", 0)
    glucose = patient_data.get("gluc", patient_data.get("glucose", 0))
    bmi = patient_data.get("bmi", 0)
    
    # Always include cardiovascular for CVD dataset
    relevant_conditions.append("cardiovascular")
    
    if isinstance(glucose, (int, float)) and (glucose >= 2 or glucose > 100):
        relevant_conditions.append("diabetes")
    
    if isinstance(bmi, (int, float)) and bmi > 25:
        relevant_conditions.append("metabolic_syndrome")
    
    if risk_category == "Low":
        relevant_conditions.append("general_wellness")
    
    # Build knowledge text
    knowledge_parts = []
    for condition in relevant_conditions:
        if condition in MEDICAL_KNOWLEDGE:
            kb = MEDICAL_KNOWLEDGE[condition]
            knowledge_parts.append(f"### {condition.replace('_', ' ').title()}")
            knowledge_parts.append(kb["description"])
            
            if kb["guidelines"]:
                knowledge_parts.append("**Clinical Guidelines:**")
                for g in kb["guidelines"][:3]:
                    knowledge_parts.append(f"  - {g}")
            
            if "lifestyle" in kb["interventions"]:
                knowledge_parts.append("**Recommended Interventions:**")
                for i in kb["interventions"]["lifestyle"][:3]:
                    knowledge_parts.append(f"  - {i}")
    
    return "\n".join(knowledge_parts)


def search_knowledge(query_terms):
    """
    Search knowledge base by terms (simple keyword matching).
    
    Args:
        query_terms: list of terms to search for
    
    Returns:
        list of relevant knowledge entries
    """
    results = []
    query_lower = [t.lower() for t in query_terms]
    
    for condition, kb in MEDICAL_KNOWLEDGE.items():
        # Check if any query term matches
        text_blob = (
            condition + " " + 
            kb["description"] + " " +
            " ".join(kb.get("risk_factors", [])) + " " +
            " ".join(kb.get("guidelines", []))
        ).lower()
        
        if any(term in text_blob for term in query_lower):
            results.append({
                "condition": condition,
                "description": kb["description"],
                "guidelines": kb["guidelines"][:3],
                "relevance": sum(1 for t in query_lower if t in text_blob)
            })
    
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results
