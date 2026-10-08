"""
==============================================================================
  ML Model Evaluation
==============================================================================
  Comprehensive evaluation of trained classifiers:
    - Classification report (precision, recall, F1)
    - Confusion matrix
    - ROC-AUC curve
    - Feature importance ranking
    - Prediction probability analysis
==============================================================================
"""

import os
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config


class ModelEvaluator:
    """
    Evaluates ML models for CRC error detection prediction.

    Stores evaluation results for later visualization.
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, Any]] = {}

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run full evaluation on a trained model.

        Parameters
        ----------
        model : estimator
            Trained classifier with predict() and predict_proba().
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            True labels.
        model_name : str
            Identifier for this model (e.g. "Random Forest").
        feature_names : list or None
            Feature column names for importance ranking.

        Returns
        -------
        dict
            Evaluation metrics and data.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        # ── Predictions ──
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

        # ── Metrics ──
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC (only if both classes present)
        try:
            roc_auc = roc_auc_score(y_test, y_prob)
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        except ValueError:
            roc_auc = 0.0
            fpr, tpr, thresholds = [0], [0], [0]

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=["Undetected (Aliasing)", "Detected"],
            zero_division=0,
        )

        # Feature importance
        feature_importance = None
        if feature_names and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importance = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )

        # ── Store results ──
        result = {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "feature_importance": feature_importance,
        }

        self.results[model_name] = result

        # ── Print summary ──
        print(f"\n  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    {cm}")
        print(f"\n{report}")

        if feature_importance:
            print("  Top 5 Features:")
            for fname, imp in feature_importance[:5]:
                print(f"    {fname:30s} {imp:.4f}")

        return result

    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models in a summary table.

        Returns
        -------
        pd.DataFrame
            Comparison table with key metrics.
        """
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": res["accuracy"],
                "Precision": res["precision"],
                "Recall": res["recall"],
                "F1 Score": res["f1_score"],
                "ROC-AUC": res["roc_auc"],
            })
        df = pd.DataFrame(rows)
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(df.to_string(index=False))
        return df

    def save_report(
        self, output_dir: str = config.REPORTS_DIR
    ):
        """
        Save evaluation report as a text file.
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "ml_evaluation_report.txt")

        with open(path, "w") as f:
            for name, res in self.results.items():
                f.write(f"{'='*60}\n")
                f.write(f"Model: {name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Accuracy:  {res['accuracy']:.4f}\n")
                f.write(f"Precision: {res['precision']:.4f}\n")
                f.write(f"Recall:    {res['recall']:.4f}\n")
                f.write(f"F1 Score:  {res['f1_score']:.4f}\n")
                f.write(f"ROC-AUC:   {res['roc_auc']:.4f}\n\n")
                f.write(f"Confusion Matrix:\n{res['confusion_matrix']}\n\n")
                f.write(f"Classification Report:\n{res['classification_report']}\n\n")
                if res["feature_importance"]:
                    f.write("Feature Importance:\n")
                    for fname, imp in res["feature_importance"]:
                        f.write(f"  {fname:30s} {imp:.4f}\n")
                f.write("\n")

        print(f"  Report saved → {path}")
