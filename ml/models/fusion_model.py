"""
Cross-Modal Attention Fusion Network (KEY NOVELTY)
===================================================
Fuses tabular, temporal, and text modality embeddings using:
  1. Cross-modal attention — each modality attends to all others
  2. Learnable modality reliability weights — network learns trust per modality
  3. Feature interaction block — bilinear interaction between fused features
  4. Dual prediction heads — risk score + uncertainty estimate

Architecture:
  Tabular Emb ─┐
  Temporal Emb ─┼─→ Cross-Attention ─→ Modality Weighting ─→ Feature Interaction ─→ Risk Head
  Text Emb ────┘                                                                  └→ Uncertainty Head

This is the primary novel contribution of the system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: each modality computes attention over all modalities.
    Uses multi-head attention where queries come from one modality
    and keys/values come from all modalities concatenated.
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projection layers for Q, K, V per modality
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, query_modality, all_modalities):
        """
        Args:
            query_modality: (batch, embed_dim) — the modality computing attention
            all_modalities: (batch, n_modalities, embed_dim) — all modalities stacked
        Returns:
            attended: (batch, embed_dim) — cross-attention output
        """
        batch_size = query_modality.size(0)
        n_mod = all_modalities.size(1)
        
        # Project queries, keys, values
        Q = self.query_proj(query_modality).unsqueeze(1)  # (batch, 1, embed_dim)
        K = self.key_proj(all_modalities)  # (batch, n_mod, embed_dim)
        V = self.value_proj(all_modalities)  # (batch, n_mod, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_mod, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_mod, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, 1, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1)
        
        # Output projection + residual
        output = self.output_proj(attn_output)
        output = self.norm(output + query_modality)
        
        return output


class ModalityReliabilityWeighting(nn.Module):
    """
    Learnable modality reliability weights (NOVEL FEATURE).
    The network learns how much to trust each modality based on the input.
    
    Unlike fixed weights, these are input-dependent—allowing the model
    to dynamically adjust modality importance for each patient.
    """
    
    def __init__(self, embed_dim, n_modalities=3):
        super().__init__()
        
        # Reliability scorer: takes concatenated modalities, outputs weights
        self.reliability_net = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_modalities)
        )
        
        self.n_modalities = n_modalities
    
    def forward(self, modality_embeddings):
        """
        Args:
            modality_embeddings: list of (batch, embed_dim) tensors
        Returns:
            weighted_sum: (batch, embed_dim)
            weights: (batch, n_modalities) — reliability weights
        """
        # Concatenate all modalities
        concat = torch.cat(modality_embeddings, dim=1)  # (batch, embed_dim * n_mod)
        
        # Compute reliability weights (softmax to sum to 1)
        raw_weights = self.reliability_net(concat)
        weights = F.softmax(raw_weights, dim=1)  # (batch, n_modalities)
        
        # Weighted sum
        stacked = torch.stack(modality_embeddings, dim=1)  # (batch, n_mod, embed_dim)
        weighted_sum = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, n_mod)
            stacked               # (batch, n_mod, embed_dim)
        ).squeeze(1)              # (batch, embed_dim)
        
        return weighted_sum, weights


class FeatureInteractionBlock(nn.Module):
    """
    Bilinear feature interaction for capturing cross-feature relationships.
    Applies bilinear transformation followed by non-linear projection.
    """
    
    def __init__(self, embed_dim, output_dim=128):
        super().__init__()
        
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, output_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, fused_features, original_concat):
        """
        Args:
            fused_features: (batch, embed_dim) — from modality weighting
            original_concat: (batch, embed_dim) — concatenated original features
        Returns:
            interacted: (batch, output_dim)
        """
        interaction = self.bilinear(fused_features, original_concat)
        return self.projection(interaction)


class MultimodalFusionNetwork(nn.Module):
    """
    Complete multimodal fusion network — THE KEY NOVELTY.
    
    Combines:
      1. Modality-specific projection layers
      2. Cross-modal attention for each modality
      3. Learnable modality reliability weighting
      4. Bilinear feature interaction
      5. Risk prediction head
      6. Uncertainty estimation head
    
    Args:
        tabular_dim: Input dimension of tabular features
        temporal_dim: Input dimension of temporal embeddings (from LSTM)
        text_dim: Input dimension of text embeddings
        fusion_dim: Shared fusion embedding dimension
        dropout: Dropout rate
    """
    
    def __init__(self, tabular_dim, temporal_dim=128, text_dim=64,
                 fusion_dim=128, dropout=0.3):
        super().__init__()
        
        # Modality projection layers (align all to fusion_dim)
        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal attention for each modality
        self.cross_attn_tabular = CrossModalAttention(fusion_dim, num_heads=4, dropout=dropout)
        self.cross_attn_temporal = CrossModalAttention(fusion_dim, num_heads=4, dropout=dropout)
        self.cross_attn_text = CrossModalAttention(fusion_dim, num_heads=4, dropout=dropout)
        
        # Modality reliability weighting
        self.reliability_weighting = ModalityReliabilityWeighting(fusion_dim, n_modalities=3)
        
        # Feature interaction block
        self.feature_interaction = FeatureInteractionBlock(fusion_dim, output_dim=128)
        
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head (separate pathway)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 = high uncertainty, 1 = high confidence
        )
    
    def forward(self, tabular_features, temporal_embedding, text_embedding):
        """
        Args:
            tabular_features: (batch, tabular_dim)
            temporal_embedding: (batch, temporal_dim) — from LSTM encoder
            text_embedding: (batch, text_dim)
        Returns:
            risk_prob: (batch, 1) — risk probability
            confidence: (batch, 1) — prediction confidence
            modality_weights: (batch, 3) — reliability weights per modality
        """
        # Project to shared dimension
        tab_emb = self.tabular_proj(tabular_features)
        temp_emb = self.temporal_proj(temporal_embedding)
        txt_emb = self.text_proj(text_embedding)
        
        # Stack for cross-attention
        all_modalities = torch.stack([tab_emb, temp_emb, txt_emb], dim=1)  # (batch, 3, fusion_dim)
        
        # Cross-modal attention (each modality attends to all)
        tab_attended = self.cross_attn_tabular(tab_emb, all_modalities)
        temp_attended = self.cross_attn_temporal(temp_emb, all_modalities)
        txt_attended = self.cross_attn_text(txt_emb, all_modalities)
        
        # Modality reliability weighting
        fused, modality_weights = self.reliability_weighting(
            [tab_attended, temp_attended, txt_attended]
        )
        
        # Feature interaction
        original_concat = (tab_emb + temp_emb + txt_emb) / 3  # Averaged originals
        interacted = self.feature_interaction(fused, original_concat)
        
        # Prediction heads
        risk_prob = self.risk_head(interacted)
        confidence = self.uncertainty_head(interacted)
        
        return risk_prob, confidence, modality_weights


class SimpleFusionModel(nn.Module):
    """
    Simplified fusion model for when not all modalities are available.
    Works with just tabular features but uses the same architecture principles.
    """
    
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        risk = self.risk_head(features)
        confidence = self.uncertainty_head(features)
        return risk, confidence
