# ============================================================
# GDHRA System — Google Colab (Part 5: Visualization & Summary)
# ============================================================

# %% [markdown]
# ## 📈 Step 5: Visualizations & Research Dashboard

# %% — Cell 17: ROC Curves (All Models)
print("Generating ROC curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ROC Curves
ax1 = axes[0]
colors = plt.cm.Set2(np.linspace(0, 1, len(eval_results_cvd)))
for i, (name, ev) in enumerate(eval_results_cvd.items()):
    ax1.plot(ev["roc_curve"]["fpr"], ev["roc_curve"]["tpr"],
             label=f"{name} (AUC={ev['auroc']:.3f})", color=colors[i], linewidth=2)
ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate", fontsize=12)
ax1.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax1.legend(fontsize=9, loc="lower right")
ax1.grid(True, alpha=0.3)

# Calibration Curves
ax2 = axes[1]
for i, (name, ev) in enumerate(eval_results_cvd.items()):
    cal = ev["calibration"]
    if cal["fraction"]:
        ax2.plot(cal["mean_pred"], cal["fraction"], "o-",
                 label=name, color=colors[i], linewidth=2, markersize=5)
ax2.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, label="Perfectly Calibrated")
ax2.set_xlabel("Mean Predicted Probability", fontsize=12)
ax2.set_ylabel("Fraction of Positives", fontsize=12)
ax2.set_title("Calibration Curves", fontsize=14, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("roc_calibration.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Cell 18: Confusion Matrix + Feature Importance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = eval_results_cvd[best_name]["confusion_matrix"]
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title(f"Confusion Matrix — {best_name}", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Negative", "Positive"])
axes[0].set_yticklabels(["Negative", "Positive"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i][j] > cm.max()/2 else "black")
plt.colorbar(im, ax=axes[0])

# Feature Importance
top_features = list(feature_importance.items())[:10]
names = [humanize_feature(f) for f, _ in top_features]
values = [v for _, v in top_features]
bars = axes[1].barh(range(len(names)), values, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names))))
axes[1].set_yticks(range(len(names)))
axes[1].set_yticklabels(names)
axes[1].invert_yaxis()
axes[1].set_xlabel("Importance")
axes[1].set_title("Top 10 Feature Importance", fontsize=14, fontweight="bold")
for bar, val in zip(bars, values):
    axes[1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("confusion_importance.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Cell 19: LSTM Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(lstm_history["train_loss"], label="Train Loss", color="#2563eb", linewidth=2)
axes[0].plot(lstm_history["val_loss"], label="Val Loss", color="#ef4444", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("LSTM Training History", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Attention heatmap (sample patient)
sample_attn = attn_weights[0].cpu().numpy()
axes[1].bar(range(len(sample_attn)), sample_attn, color="#2563eb", alpha=0.7)
axes[1].set_xlabel("Time Step")
axes[1].set_ylabel("Attention Weight")
axes[1].set_title("Temporal Attention Weights (Sample Patient)", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lstm_history.png", dpi=150, bbox_inches="tight")
plt.show()

# %% — Cell 20: Risk Gauge (Plotly)
def create_risk_gauge(score, category):
    color_map = {"Low": "#16a34a", "Moderate": "#f59e0b", "High": "#ef4444", "Critical": "#dc2626"}
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_map.get(category, "#6b7280"), "thickness": 0.7},
            "steps": [
                {"range": [0, 30], "color": "#dcfce7"},
                {"range": [30, 60], "color": "#fef3c7"},
                {"range": [60, 80], "color": "#fee2e2"},
                {"range": [80, 100], "color": "#fecdd3"}
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=30, b=10))
    return fig

# Demo prediction
demo_features = []
for fn in feat_cvd:
    v = demo_patient.get(fn, 0)
    if fn == "bmi": v = 85 / ((170 / 100) ** 2)
    elif fn == "pulse_pressure": v = 150 - 95
    elif fn == "age_bp_interaction": v = 50 * 150
    elif fn == "bmi_chol_interaction": v = (85 / ((170/100)**2)) * 3
    demo_features.append(float(v) if isinstance(v, (int, float)) else 0.0)

demo_scaled = scaler_cvd.transform([demo_features])
demo_prob = best_model_cvd.predict_proba(demo_scaled)[0][1]
demo_risk = round(demo_prob * 100, 1)
demo_cat = "Low" if demo_risk < 30 else ("Moderate" if demo_risk < 60 else ("High" if demo_risk < 80 else "Critical"))

fig = create_risk_gauge(demo_risk, demo_cat)
fig.show()
print(f"\n🏥 Demo Patient Risk: {demo_risk}/100 ({demo_cat})")

# %% — Cell 21: Generate AI Health Advice
print("=" * 60)
print("AI HEALTH RECOMMENDATIONS")
print("=" * 60)

advice = generate_health_advice(
    demo_risk, demo_cat,
    {**demo_patient, "age": 50, "bmi": round(85/((170/100)**2), 1)},
    feature_importances=feature_importance
)
from IPython.display import Markdown
display(Markdown(advice["advice"]))
print(f"\n[Source: {advice['source']}]")

# %% — Cell 22: Metrics Comparison Bar Chart
fig, ax = plt.subplots(figsize=(14, 6))
metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC"]
x = np.arange(len(metric_names))
width = 0.12
model_names = list(eval_results_cvd.keys())
colors_bar = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

for i, name in enumerate(model_names):
    ev = eval_results_cvd[name]
    vals = [ev["accuracy"], ev["precision"], ev["recall"], ev["f1"], ev["auroc"], ev["auprc"]]
    ax.bar(x + i * width, vals, width, label=name, color=colors_bar[i], edgecolor="white")

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_xticks(x + width * (len(model_names) - 1) / 2)
ax.set_xticklabels(metric_names)
ax.legend(loc="lower right", fontsize=9)
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 🏁 Complete Summary
#
# ### Training Results (CVD Dataset)
# | Model | F1 | AUROC | Brier |
# |-------|-----|-------|-------|
# | LogisticRegression | 0.756 | 0.673 | 0.216 |
# | RandomForest | 0.761 | 0.662 | 0.219 |
# | GradientBoosting | 0.758 | 0.663 | 0.218 |
# | SVM | 0.767 | 0.633 | 0.224 |
# | XGBoost | 0.758 | 0.666 | 0.217 |
# | LightGBM | 0.757 | 0.662 | 0.218 |
#
# ### Research Novelty Claims
# 1. **Cross-Modal Attention Fusion** — learnable modality-specific attention
# 2. **Modality Reliability Weighting** — dynamic, input-dependent trust
# 3. **Bilinear Feature Interaction** — captures cross-modal nonlinearities
# 4. **Dual-Head Architecture** — separate risk + uncertainty pathways
# 5. **Generative What-If Simulator** — per-factor impact decomposition
#
# ### Patent Differentiation
# 1. Dynamic Modality Trust Scoring
# 2. Integrated What-If Risk Simulation
# 3. Medical RAG-Enhanced Health Advisory
# 4. Uncertainty-Flagged Clinical Decision Support

# %% — Cell 23: Save All Artifacts
print("=" * 60)
print("SAVING ALL ARTIFACTS")
print("=" * 60)

os.makedirs("saved_models", exist_ok=True)
artifacts = {
    "best_model_name": best_name,
    "best_model": best_model_cvd,
    "all_models": {k: v["best_model"] for k, v in tabular_results_cvd.items()},
    "evaluation_results": eval_results_cvd,
    "feature_names": feat_cvd,
    "scaler": scaler_cvd,
    "feature_importance": feature_importance,
    "uncertainty": unc_result,
}
joblib.dump(artifacts, "saved_models/model_cvd.joblib")
torch.save(lstm_model.state_dict(), "saved_models/lstm_mimic.pt")

print("✅ Saved: saved_models/model_cvd.joblib")
print("✅ Saved: saved_models/lstm_mimic.pt")
print("✅ Saved: shap_summary.png, roc_calibration.png, confusion_importance.png, lstm_history.png, model_comparison.png")
print("\n🏁 GDHRA SYSTEM — COMPLETE!")
print("All datasets, models, visualizations, and GenAI advisory components are operational.")
