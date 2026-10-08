"""
Temporal Model Branch (LSTM)
=============================
PyTorch LSTM with attention for temporal health data sequences.
Bidirectional 2-layer LSTM with MC Dropout for uncertainty estimation.

Architecture:
  Input (seq_len, n_features) → BiLSTM (2 layers) → Temporal Attention → FC → Embedding (128-dim)
"""

import numpy as np
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Attention mechanism over LSTM time steps."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) — attention-weighted sum
            weights: (batch, seq_len) — attention weights
        """
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM with temporal attention for time-series health data.
    
    Args:
        input_dim: Number of input features per time step
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        embedding_dim: Output embedding dimension
        dropout: Dropout rate for MC Dropout
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, 
                 embedding_dim=128, dropout=0.3):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = TemporalAttention(hidden_dim * 2)  # *2 for bidirectional
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            embedding: (batch, embedding_dim)
            attention_weights: (batch, seq_len)
        """
        # Normalize input
        x = self.input_norm(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        
        # Attention
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_dim*2)
        
        # Project to embedding
        embedding = self.fc(context)  # (batch, embedding_dim)
        
        return embedding, attn_weights


class TemporalPredictor(nn.Module):
    """
    Complete temporal prediction model with risk head.
    Can be used standalone or provide embeddings for fusion.
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2,
                 embedding_dim=128, dropout=0.3):
        super().__init__()
        
        self.encoder = TemporalLSTM(
            input_dim, hidden_dim, num_layers, embedding_dim, dropout
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            risk_prob: (batch, 1)
            embedding: (batch, embedding_dim)
            attention_weights: (batch, seq_len)
        """
        embedding, attn_weights = self.encoder(x)
        risk_prob = self.risk_head(embedding)
        return risk_prob, embedding, attn_weights
    
    def get_embedding(self, x):
        """Get just the embedding for fusion."""
        embedding, _ = self.encoder(x)
        return embedding


def train_temporal_model(X_train, y_train, X_test, y_test,
                         epochs=50, batch_size=32, lr=0.001, device=None):
    """
    Train the temporal LSTM model.
    
    Args:
        X_train: (n_train, seq_len, features) numpy array
        y_train: (n_train,) numpy array
        X_test: (n_test, seq_len, features) numpy array
        y_test: (n_test,) numpy array
    
    Returns:
        model, training_history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"  [LSTM] Training on {device}")
    
    input_dim = X_train.shape[2]
    model = TemporalPredictor(input_dim=input_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    
    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        
        # Mini-batch training
        indices = torch.randperm(len(X_train_t))
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, len(X_train_t), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            
            optimizer.zero_grad()
            pred, _, _ = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _, _ = model(X_test_t)
            val_loss = criterion(val_pred, y_test_t).item()
        
        avg_train_loss = epoch_loss / n_batches
        scheduler.step(val_loss)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Load best weights
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"  [LSTM] Training complete. Best val loss: {best_val_loss:.4f}")
    return model, history
