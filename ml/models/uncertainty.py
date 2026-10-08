"""
Uncertainty Estimation Module
==============================
Monte Carlo Dropout for prediction uncertainty quantification.

Implements:
  - MC Dropout: N forward passes with dropout enabled
  - Prediction mean, variance, confidence intervals
  - Risk reliability flags for clinicians
"""

import numpy as np
import torch


def mc_dropout_predict(model, X, n_forward=30, device=None):
    """
    Run Monte Carlo Dropout inference.
    
    Enables dropout during inference and runs multiple forward passes
    to estimate prediction uncertainty.
    
    Args:
        model: PyTorch model with dropout layers
        X: input tensor or numpy array
        n_forward: number of forward passes
        device: torch device
    
    Returns:
        dict with keys:
            mean_prediction: average prediction
            std_prediction: standard deviation
            confidence_interval: (low, high) tuple
            all_predictions: array of all predictions
            reliability: "High" / "Moderate" / "Low"
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X).to(device)
    
    # Enable dropout during inference
    model.train()  # This enables dropout
    
    predictions = []
    for _ in range(n_forward):
        with torch.no_grad():
            output = model(X)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            predictions.append(pred.cpu().numpy())
    
    # Restore eval mode
    model.eval()
    
    predictions = np.array(predictions).squeeze()  # (n_forward, batch_size)
    
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    
    # 95% confidence interval
    ci_low = np.percentile(predictions, 2.5, axis=0)
    ci_high = np.percentile(predictions, 97.5, axis=0)
    
    # Reliability flag based on uncertainty
    avg_std = std_pred.mean()
    if avg_std < 0.05:
        reliability = "High Confidence"
    elif avg_std < 0.15:
        reliability = "Moderate Confidence"
    else:
        reliability = "Low Confidence — Review Required"
    
    return {
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "confidence_interval": (ci_low, ci_high),
        "all_predictions": predictions,
        "reliability": reliability,
        "uncertainty_score": avg_std
    }


def tabular_uncertainty(model, X, n_perturbations=50, noise_std=0.05):
    """
    Estimate uncertainty for scikit-learn models using input perturbation.
    
    Adds Gaussian noise to input features and measures prediction variance.
    
    Args:
        model: scikit-learn model with predict_proba
        X: (n_samples, n_features) numpy array
        n_perturbations: number of perturbation rounds
        noise_std: standard deviation of Gaussian noise
    
    Returns:
        dict with uncertainty metrics
    """
    if not hasattr(model, "predict_proba"):
        return {
            "mean_prediction": model.predict(X).astype(float),
            "std_prediction": np.zeros(len(X)),
            "confidence_interval": (model.predict(X).astype(float), model.predict(X).astype(float)),
            "reliability": "Unknown (no probability support)",
            "uncertainty_score": 0
        }
    
    predictions = []
    for _ in range(n_perturbations):
        noise = np.random.normal(0, noise_std, X.shape)
        X_noisy = X + noise
        prob = model.predict_proba(X_noisy)[:, 1] if model.predict_proba(X_noisy).shape[1] > 1 else model.predict_proba(X_noisy)[:, 0]
        predictions.append(prob)
    
    predictions = np.array(predictions)  # (n_perturbations, n_samples)
    
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)
    ci_low = np.percentile(predictions, 2.5, axis=0)
    ci_high = np.percentile(predictions, 97.5, axis=0)
    
    avg_std = std_pred.mean()
    if avg_std < 0.03:
        reliability = "High Confidence"
    elif avg_std < 0.10:
        reliability = "Moderate Confidence"
    else:
        reliability = "Low Confidence — Review Required"
    
    return {
        "mean_prediction": mean_pred,
        "std_prediction": std_pred,
        "confidence_interval": (ci_low, ci_high),
        "reliability": reliability,
        "uncertainty_score": avg_std
    }
