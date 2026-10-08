# Generative Dynamic Health Risk Assessment System (GDHRA)

A **research-grade, multimodal, uncertainty-aware** health risk prediction system that fuses structured EHR data, temporal signals, and clinical text using cross-attention, estimates prediction confidence via Monte Carlo Dropout, and generates personalized preventive insights.

## Architecture

```
Dataset 1 (CVD)      → Clinical EHR features    ─┐
Dataset 2 (MIMIC)    → Temporal vital signs      ─┼─→ Cross-Attention Fusion ─→ Risk Prediction
Dataset 3 (Synthetic)→ Clinical text embeddings  ─┘         ↓                    + Uncertainty
                                                    Modality Weighting            ↓
                                                                           GenAI Explanation
```

## Key Innovations

1. **Cross-Modal Attention Fusion** — learnable modality reliability weighting
2. **Uncertainty-Aware Prediction** — Monte Carlo Dropout confidence intervals
3. **Generative Preventive Insight Engine** — Gemini API + medical RAG
4. **What-If Risk Simulator** — interactive parameter modification
5. **Temporal Risk Modeling** — LSTM-based health trajectory prediction

## Quick Start

```bash
cd c:\RISK_ASSESSMENT

# Install dependencies
pip install -r requirements.txt

# Download/generate datasets
python data/download_datasets.py

# Train models
python ml/train.py --dataset cvd

# Launch UI
streamlit run frontend/app.py
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for GenAI insights | No (fallback advisor used) |

## Project Structure

```
RISK_ASSESSMENT/
├── data/                    # Dataset download & storage
├── ml/                      # ML pipeline
│   ├── models/              # Model architectures
│   ├── preprocessing.py     # Feature engineering
│   ├── train.py             # Training pipeline
│   └── evaluate.py          # Evaluation metrics
├── genai/                   # Generative AI module
├── explainability/          # SHAP & explanations
├── frontend/                # Streamlit dashboard
├── saved_models/            # Trained model artifacts
└── requirements.txt
```
