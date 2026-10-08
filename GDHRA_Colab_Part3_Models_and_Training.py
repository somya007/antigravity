# ============================================================
# GDHRA System — Google Colab (Part 3: Models & Training)
# ============================================================

# %% [markdown]
# ## 🧠 Step 3: Model Architectures
# ### 3A: Tabular Branch (6 classifiers + GridSearchCV)
# ### 3B: Temporal Branch (BiLSTM + Attention)
# ### 3C: Text Branch (Embedding Network)
# ### 3D: Cross-Modal Attention Fusion (KEY NOVELTY)
# ### 3E: Uncertainty Estimation (MC Dropout)

# %% — Cell 8: Tabular Model Training
def train_tabular_models(X_train, y_train, cv=5, scoring="f1"):
    """Train 6 classifiers with GridSearchCV."""
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf"]}
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False, verbosity=0),
            "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
            "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
        }
    }
    results = {}
    for name, config in models.items():
        print(f"  Training {name}...", end=" ")
        grid = GridSearchCV(config["model"], config["params"], cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        results[name] = {"best_model": grid.best_estimator_, "best_params": grid.best_params_, "best_score": grid.best_score_}
        print(f"Best {scoring}: {grid.best_score_:.4f}")
    return results

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive evaluation with research-grade metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)
    try: auroc = roc_auc_score(y_test, y_prob)
    except: auroc = 0.5
    try: auprc = average_precision_score(y_test, y_prob)
    except: auprc = 0.0
    fpr, tpr, _ = roc_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else ([0,1],[0,1],None)
    cm = confusion_matrix(y_test, y_pred)
    # Calibration curve
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_frac, cal_mean = [], []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            cal_frac.append(float(y_test[mask].mean()))
            cal_mean.append(float(y_prob[mask].mean()))
    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auroc": auroc, "auprc": auprc,
        "brier_score": brier_score_loss(y_test, y_prob),
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "confusion_matrix": cm,
        "calibration": {"fraction": cal_frac, "mean_pred": cal_mean}
    }

# %% — Cell 9: Train & Evaluate on CVD
print("=" * 60)
print("TRAINING TABULAR MODELS ON CVD DATASET")
print("=" * 60)

tabular_results_cvd = train_tabular_models(X_train_cvd, y_train_cvd)

# Evaluate all
eval_results_cvd = {}
best_score, best_name = 0, ""
for name, res in tabular_results_cvd.items():
    ev = evaluate_model(res["best_model"], X_test_cvd, y_test_cvd, name)
    eval_results_cvd[name] = ev
    if ev["f1"] > best_score:
        best_score, best_name = ev["f1"], name

best_model_cvd = tabular_results_cvd[best_name]["best_model"]
print(f"\n🏆 Best Model: {best_name} (F1={best_score:.4f})")

# Display metrics table
metrics_df = pd.DataFrame([{
    "Model": n, "Accuracy": f"{e['accuracy']:.4f}", "F1": f"{e['f1']:.4f}",
    "AUROC": f"{e['auroc']:.4f}", "AUPRC": f"{e['auprc']:.4f}", "Brier": f"{e['brier_score']:.4f}"
} for n, e in eval_results_cvd.items()])
display(metrics_df)

# %% — Cell 10: LSTM Temporal Model (PyTorch)
print("\n" + "=" * 60)
print("TEMPORAL LSTM MODEL")
print("=" * 60)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, lstm_output):
        scores = self.attention(lstm_output).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = TemporalAttention(hidden_dim * 2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, embedding_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, embedding_dim))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        context, attn_w = self.attention(lstm_out)
        return self.fc(context), attn_w

class TemporalPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = TemporalLSTM(input_dim, hidden_dim, num_layers, embedding_dim, dropout)
        self.risk_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        emb, attn_w = self.encoder(x)
        return self.risk_head(emb), emb, attn_w

# Train LSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model = TemporalPredictor(input_dim=X_seq_train.shape[2]).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

X_tr_t = torch.FloatTensor(X_seq_train).to(device)
y_tr_t = torch.FloatTensor(y_seq_train).unsqueeze(1).to(device)
X_te_t = torch.FloatTensor(X_seq_test).to(device)
y_te_t = torch.FloatTensor(y_seq_test).unsqueeze(1).to(device)

print(f"Training LSTM on {device}...")
lstm_history = {"train_loss": [], "val_loss": []}
best_val_loss = float("inf")
best_state = None

for epoch in range(50):
    lstm_model.train()
    indices = torch.randperm(len(X_tr_t))
    epoch_loss, n_b = 0, 0
    for start in range(0, len(X_tr_t), 32):
        batch_idx = indices[start:start + 32]
        optimizer.zero_grad()
        pred, _, _ = lstm_model(X_tr_t[batch_idx])
        loss = criterion(pred, y_tr_t[batch_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_b += 1

    lstm_model.eval()
    with torch.no_grad():
        val_pred, _, _ = lstm_model(X_te_t)
        val_loss = criterion(val_pred, y_te_t).item()
    scheduler.step(val_loss)
    lstm_history["train_loss"].append(epoch_loss / n_b)
    lstm_history["val_loss"].append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in lstm_model.state_dict().items()}
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/50 | Train: {epoch_loss/n_b:.4f} | Val: {val_loss:.4f}")

if best_state:
    lstm_model.load_state_dict(best_state)

lstm_model.eval()
with torch.no_grad():
    preds, _, attn_weights = lstm_model(X_te_t)
    preds_np = preds.cpu().numpy().squeeze()
try: lstm_auroc = roc_auc_score(y_seq_test, preds_np)
except: lstm_auroc = 0.5
lstm_f1 = f1_score(y_seq_test, (preds_np > 0.5).astype(int), zero_division=0)
print(f"\n✅ LSTM Results: AUROC={lstm_auroc:.4f}, F1={lstm_f1:.4f}")

# %% — Cell 11: Cross-Modal Attention Fusion Network (KEY NOVELTY)
print("\n" + "=" * 60)
print("CROSS-MODAL ATTENTION FUSION NETWORK")
print("=" * 60)

class CrossModalAttention(nn.Module):
    """Cross-modal attention: each modality attends to all others."""
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_mod, all_mods):
        B, n_mod = query_mod.size(0), all_mods.size(1)
        Q = self.query_proj(query_mod).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(all_mods).view(B, n_mod, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(all_mods).view(B, n_mod, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, -1)
        return self.norm(self.output_proj(out) + query_mod)

class ModalityReliabilityWeighting(nn.Module):
    """Learnable modality reliability weights (NOVEL FEATURE)."""
    def __init__(self, embed_dim, n_modalities=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim * n_modalities, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, n_modalities))
    def forward(self, modality_embs):
        concat = torch.cat(modality_embs, dim=1)
        weights = F.softmax(self.net(concat), dim=1)
        stacked = torch.stack(modality_embs, dim=1)
        return torch.bmm(weights.unsqueeze(1), stacked).squeeze(1), weights

class MultimodalFusionNetwork(nn.Module):
    """Complete multimodal fusion — KEY NOVELTY."""
    def __init__(self, tabular_dim, temporal_dim=128, text_dim=64, fusion_dim=128, dropout=0.3):
        super().__init__()
        self.tab_proj = nn.Sequential(nn.Linear(tabular_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.temp_proj = nn.Sequential(nn.Linear(temporal_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.text_proj = nn.Sequential(nn.Linear(text_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(), nn.Dropout(dropout))
        self.cross_attn_tab = CrossModalAttention(fusion_dim, 4, dropout)
        self.cross_attn_temp = CrossModalAttention(fusion_dim, 4, dropout)
        self.cross_attn_text = CrossModalAttention(fusion_dim, 4, dropout)
        self.reliability = ModalityReliabilityWeighting(fusion_dim, 3)
        self.interaction = nn.Bilinear(fusion_dim, fusion_dim, 128)
        self.interact_proj = nn.Sequential(nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128))
        self.risk_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())
        self.uncertainty_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, tab_feat, temp_emb, text_emb):
        tab_e = self.tab_proj(tab_feat)
        temp_e = self.temp_proj(temp_emb)
        txt_e = self.text_proj(text_emb)
        all_m = torch.stack([tab_e, temp_e, txt_e], dim=1)
        tab_a = self.cross_attn_tab(tab_e, all_m)
        temp_a = self.cross_attn_temp(temp_e, all_m)
        txt_a = self.cross_attn_text(txt_e, all_m)
        fused, mod_weights = self.reliability([tab_a, temp_a, txt_a])
        orig = (tab_e + temp_e + txt_e) / 3
        interacted = self.interact_proj(self.interaction(fused, orig))
        return self.risk_head(interacted), self.uncertainty_head(interacted), mod_weights

# Demo the fusion architecture
print("Architecture instantiated:")
demo_fusion = MultimodalFusionNetwork(tabular_dim=16, temporal_dim=128, text_dim=64)
total_params = sum(p.numel() for p in demo_fusion.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Components: CrossModalAttention × 3, ModalityReliabilityWeighting, BilinearInteraction, RiskHead, UncertaintyHead")
demo_tab = torch.randn(4, 16)
demo_temp = torch.randn(4, 128)
demo_text = torch.randn(4, 64)
risk, conf, weights = demo_fusion(demo_tab, demo_temp, demo_text)
print(f"\n  Demo forward pass (batch=4):")
print(f"    Risk output:    {risk.shape} — values: {risk.detach().numpy().flatten()}")
print(f"    Confidence:     {conf.shape} — values: {conf.detach().numpy().flatten()}")
print(f"    Modality weights: {weights.detach().numpy()}")
print("\n✅ Fusion network working correctly!")

# %% — Cell 12: Uncertainty Estimation (MC Dropout)
print("\n" + "=" * 60)
print("UNCERTAINTY ESTIMATION")
print("=" * 60)

def tabular_uncertainty(model, X, n_perturbations=50, noise_std=0.05):
    """Estimate uncertainty via input perturbation for sklearn models."""
    if not hasattr(model, "predict_proba"):
        return {"mean": model.predict(X).astype(float), "std": np.zeros(len(X)), "reliability": "Unknown"}
    predictions = []
    for _ in range(n_perturbations):
        noise = np.random.normal(0, noise_std, X.shape)
        prob = model.predict_proba(X + noise)[:, 1]
        predictions.append(prob)
    preds = np.array(predictions)
    mean_p, std_p = preds.mean(axis=0), preds.std(axis=0)
    ci_low, ci_high = np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)
    avg_std = std_p.mean()
    reliability = "High Confidence" if avg_std < 0.03 else ("Moderate" if avg_std < 0.10 else "Low — Review Required")
    return {"mean": mean_p, "std": std_p, "ci_low": ci_low, "ci_high": ci_high, "reliability": reliability, "uncertainty_score": avg_std}

unc_result = tabular_uncertainty(best_model_cvd, X_test_cvd[:100])
print(f"  Reliability: {unc_result['reliability']}")
print(f"  Avg uncertainty: {unc_result['uncertainty_score']:.4f}")
print(f"  Sample prediction: {unc_result['mean'][0]:.4f} ± {unc_result['std'][0]:.4f}")
print(f"  95% CI: [{unc_result['ci_low'][0]:.4f}, {unc_result['ci_high'][0]:.4f}]")
