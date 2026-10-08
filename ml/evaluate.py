"""
Model Evaluation Module
========================
Research-grade evaluation metrics for health risk prediction.

Metrics:
  - AUROC (Area Under ROC Curve)
  - AUPRC (Area Under Precision-Recall Curve)
  - Brier Score (calibration measure)
  - Calibration Curve data
  - Confusion Matrix
  - F1, Precision, Recall, Accuracy
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a classification model.
    
    Args:
        model: fitted sklearn model with predict and predict_proba
        X_test: test features
        y_test: test labels
        model_name: name for logging
    
    Returns:
        dict with all evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Probabilities for ranking metrics
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        y_prob_positive = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
    else:
        y_prob_positive = y_pred.astype(float)
    
    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # AUROC
    try:
        auroc = roc_auc_score(y_test, y_prob_positive)
    except ValueError:
        auroc = 0.5
    
    # AUPRC
    try:
        auprc = average_precision_score(y_test, y_prob_positive)
    except ValueError:
        auprc = 0.0
    
    # Brier Score
    brier = brier_score_loss(y_test, y_prob_positive)
    
    # ROC Curve data
    try:
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_positive)
    except ValueError:
        fpr, tpr, roc_thresholds = [0, 1], [0, 1], [1, 0]
    
    # Precision-Recall Curve data
    try:
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_prob_positive)
    except ValueError:
        pr_precision, pr_recall, pr_thresholds = [0], [0], [0]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Calibration Curve data
    cal_fraction, cal_mean_pred = _calibration_curve(y_test, y_prob_positive, n_bins=10)
    
    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "brier_score": float(brier),
        "roc_curve": {
            "fpr": fpr.tolist() if isinstance(fpr, np.ndarray) else fpr,
            "tpr": tpr.tolist() if isinstance(tpr, np.ndarray) else tpr
        },
        "pr_curve": {
            "precision": pr_precision.tolist() if isinstance(pr_precision, np.ndarray) else pr_precision,
            "recall": pr_recall.tolist() if isinstance(pr_recall, np.ndarray) else pr_recall
        },
        "confusion_matrix": cm.tolist(),
        "calibration_curve": {
            "fraction_of_positives": cal_fraction,
            "mean_predicted_value": cal_mean_pred
        }
    }


def _calibration_curve(y_true, y_prob, n_bins=10):
    """Compute calibration curve data."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    fractions = []
    mean_preds = []
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            fractions.append(float(y_true[mask].mean()))
            mean_preds.append(float(y_prob[mask].mean()))
    
    return fractions, mean_preds


def generate_evaluation_report(evaluation_results):
    """
    Generate a summary report comparing all models.
    
    Args:
        evaluation_results: dict of {model_name: eval_dict}
    
    Returns:
        summary string
    """
    lines = [
        "\n" + "=" * 70,
        "MODEL EVALUATION REPORT",
        "=" * 70,
        f"{'Model':<25} {'Accuracy':>10} {'F1':>8} {'AUROC':>8} {'AUPRC':>8} {'Brier':>8}",
        "-" * 70
    ]
    
    for name, metrics in evaluation_results.items():
        lines.append(
            f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['f1']:>8.4f} "
            f"{metrics['auroc']:>8.4f} {metrics['auprc']:>8.4f} {metrics['brier_score']:>8.4f}"
        )
    
    # Best model
    best_name = max(evaluation_results, key=lambda k: evaluation_results[k]['f1'])
    lines.append("-" * 70)
    lines.append(f"BEST MODEL: {best_name} (F1={evaluation_results[best_name]['f1']:.4f})")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    print(report)
    return report


def ablation_study(base_results, ablation_results):
    """
    Compare base model vs ablated version (with removed component).
    
    Args:
        base_results: evaluation dict from full model
        ablation_results: evaluation dict from ablated model
    
    Returns:
        dict with metric differences
    """
    metrics = ["accuracy", "f1", "auroc", "auprc", "brier_score"]
    diff = {}
    for m in metrics:
        base_val = base_results.get(m, 0)
        abl_val = ablation_results.get(m, 0)
        diff[m] = {
            "base": base_val,
            "ablated": abl_val,
            "change": base_val - abl_val,
            "change_pct": ((base_val - abl_val) / max(base_val, 1e-8)) * 100
        }
    return diff
