"""
Text Model Branch
==================
Processes clinical text features through an embedding network.
Uses TF-IDF features as input and learns medical text representations.
"""

import numpy as np
import torch
import torch.nn as nn


class TextEmbeddingNetwork(nn.Module):
    """
    Text embedding network for clinical notes.
    Transforms TF-IDF features into dense medical text embeddings.
    
    Architecture:
        TF-IDF features → FC(input, 128) → ReLU → Dropout → FC(128, 64) → ReLU → Embedding
    """
    
    def __init__(self, input_dim, embedding_dim=64, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) TF-IDF features
        Returns:
            embedding: (batch, embedding_dim)
        """
        return self.network(x)


class TextPredictor(nn.Module):
    """Complete text prediction model with risk head."""
    
    def __init__(self, input_dim, embedding_dim=64, dropout=0.3):
        super().__init__()
        
        self.encoder = TextEmbeddingNetwork(input_dim, embedding_dim, dropout)
        
        self.risk_head = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        risk_prob = self.risk_head(embedding)
        return risk_prob, embedding
    
    def get_embedding(self, x):
        return self.encoder(x)


def extract_text_features(clinical_notes, max_features=50):
    """
    Extract TF-IDF features from clinical notes.
    
    Args:
        clinical_notes: list of text strings
        max_features: number of TF-IDF features
    
    Returns:
        features: (n_samples, max_features) numpy array
        vectorizer: fitted TfidfVectorizer
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    features = vectorizer.fit_transform(clinical_notes).toarray()
    
    return features.astype(np.float32), vectorizer
