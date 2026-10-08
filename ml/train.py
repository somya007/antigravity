"""
Unified Training Pipeline
=========================
Trains all model branches and saves artifacts for the GDHRA system.

Usage:
    python ml/train.py --dataset cvd
    python ml/train.py --dataset mimic
    python ml/train.py --dataset all
"""

import os
import sys
import time
import argparse
import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import load_and_preprocess, get_temporal_sequences
from ml.models.tabular_model import train_tabular_models, get_best_tabular_model
from ml.models.uncertainty import tabular_uncertainty
from ml.evaluate import evaluate_model, generate_evaluation_report

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")


def train_dataset(dataset_name):
    """
    Full training pipeline for a single dataset.
    
    Steps:
      1. Load and preprocess data
      2. Train tabular models with GridSearchCV
      3. Evaluate all models
      4. Compute uncertainty estimates
      5. Save everything to saved_models/
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # ─── Step 1: Preprocess ───
    print("\n[Step 1] Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess(dataset_name)
    
    # ─── Step 2: Train tabular models ───
    print("\n[Step 2] Training tabular models with GridSearchCV...")
    tabular_results = train_tabular_models(X_train, y_train, cv=5, scoring="f1")
    
    if not tabular_results:
        print("[ERROR] No models trained successfully!")
        return None
    
    best_name, best_model = get_best_tabular_model(tabular_results)
    
    # ─── Step 3: Evaluate all models ───
    print("\n[Step 3] Evaluating models...")
    evaluation_results = {}
    for name, result in tabular_results.items():
        model = result["best_model"]
        eval_result = evaluate_model(model, X_test, y_test, name)
        evaluation_results[name] = eval_result
        
        print(f"  {name}:")
        print(f"    Accuracy: {eval_result['accuracy']:.4f}")
        print(f"    F1:       {eval_result['f1']:.4f}")
        print(f"    AUROC:    {eval_result['auroc']:.4f}")
        print(f"    AUPRC:    {eval_result['auprc']:.4f}")
        print(f"    Brier:    {eval_result['brier_score']:.4f}")
    
    # ─── Step 4: Uncertainty estimation ───
    print("\n[Step 4] Computing uncertainty estimates...")
    uncertainty = tabular_uncertainty(best_model, X_test[:min(100, len(X_test))])
    print(f"  Reliability: {uncertainty['reliability']}")
    print(f"  Avg uncertainty: {uncertainty['uncertainty_score']:.4f}")
    
    # ─── Step 5: Train LSTM if temporal data available ───
    lstm_results = None
    if dataset_name in ["mimic", "wearable"]:
        print("\n[Step 5] Training temporal LSTM model...")
        try:
            lstm_results = _train_temporal(dataset_name)
        except Exception as e:
            print(f"  [WARN] LSTM training failed: {e}")
    
    # ─── Step 6: Save artifacts ───
    print("\n[Step 6] Saving model artifacts...")
    
    artifacts = {
        "best_model_name": best_name,
        "best_model": best_model,
        "all_models": {k: v["best_model"] for k, v in tabular_results.items()},
        "model_params": {k: v["best_params"] for k, v in tabular_results.items()},
        "cv_scores": {k: v["best_score"] for k, v in tabular_results.items()},
        "evaluation_results": evaluation_results,
        "uncertainty": {
            "reliability": uncertainty["reliability"],
            "uncertainty_score": float(uncertainty["uncertainty_score"])
        },
        "feature_names": feature_names,
        "scaler": scaler,
        "dataset_name": dataset_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "lstm_results": lstm_results
    }
    
    save_path = os.path.join(SAVE_DIR, f"model_{dataset_name}.joblib")
    joblib.dump(artifacts, save_path)
    print(f"  Saved to: {save_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETED: {dataset_name.upper()} in {elapsed:.1f}s")
    print(f"Best model: {best_name} (F1={evaluation_results[best_name]['f1']:.4f}, AUROC={evaluation_results[best_name]['auroc']:.4f})")
    print(f"{'='*60}")
    
    return artifacts


def _train_temporal(dataset_name):
    """Train LSTM temporal model."""
    try:
        import torch
        from ml.models.temporal_model import train_temporal_model
        from sklearn.model_selection import train_test_split
        
        X_seq, y_seq, patient_ids, seq_scaler, feat_cols = get_temporal_sequences(dataset_name)
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, 
            stratify=y_seq if len(np.unique(y_seq)) > 1 else None
        )
        
        model, history = train_temporal_model(X_tr, y_tr, X_te, y_te, epochs=30, batch_size=32)
        
        # Evaluate
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_te_t = torch.FloatTensor(X_te).to(device)
            preds, _, _ = model(X_te_t)
            preds = preds.cpu().numpy().squeeze()
        
        from sklearn.metrics import roc_auc_score, f1_score
        pred_labels = (preds > 0.5).astype(int)
        
        try:
            auc = roc_auc_score(y_te, preds)
        except ValueError:
            auc = 0.5
        
        f1 = f1_score(y_te, pred_labels, zero_division=0)
        
        print(f"  [LSTM] Test AUROC: {auc:.4f}, F1: {f1:.4f}")
        
        # Save LSTM model
        lstm_path = os.path.join(SAVE_DIR, f"lstm_{dataset_name}.pt")
        torch.save(model.state_dict(), lstm_path)
        
        return {
            "auroc": float(auc),
            "f1": float(f1),
            "history": history,
            "model_path": lstm_path
        }
    
    except ImportError:
        print("  [WARN] PyTorch not available. Skipping LSTM.")
        return None


def train_all():
    """Train on all available datasets."""
    datasets = ["cvd", "mimic", "wearable"]
    results = {}
    
    for ds in datasets:
        try:
            result = train_dataset(ds)
            if result:
                results[ds] = result
        except Exception as e:
            print(f"\n[ERROR] Training failed for {ds}: {e}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDHRA Training Pipeline")
    parser.add_argument("--dataset", type=str, default="cvd",
                       choices=["cvd", "mimic", "wearable", "clinical_notes", "all"],
                       help="Dataset to train on")
    args = parser.parse_args()
    
    if args.dataset == "all":
        train_all()
    else:
        train_dataset(args.dataset)
