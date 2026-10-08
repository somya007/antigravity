"""
==============================================================================
  Run Visualization — Regenerate All Plots from Saved Data
==============================================================================
"""

import os
import pandas as pd
import joblib

import config
from ml.feature_engineering import FeatureEngineer
from ml.model_evaluation import ModelEvaluator
from evaluation.metrics import compute_all_metrics
from visualization.plot_detection_efficiency import (
    plot_detection_efficiency_vs_burst_length,
    plot_detection_rate_heatmap,
)
from visualization.plot_confusion_matrix import plot_confusion_matrix
from visualization.plot_performance import plot_throughput, plot_overhead, plot_crc_vs_checksum
from visualization.plot_roc_curve import plot_roc_curves, plot_feature_importance


def main():
    print("Regenerating all visualizations...\n")

    # Load dataset
    dataset_path = os.path.join(config.DATASETS_DIR, "crc_simulation_dataset.csv")
    if not os.path.exists(dataset_path):
        print(f"  ERROR: Dataset not found at {dataset_path}")
        print("  Please run main.py first.")
        return

    df = pd.read_csv(dataset_path)
    print(f"  Loaded dataset: {df.shape}")

    # Detection efficiency plots
    plot_detection_efficiency_vs_burst_length(
        df,
        save_path=os.path.join(config.PLOTS_DIR, "detection_efficiency_vs_burst.png"),
    )
    plot_detection_rate_heatmap(
        df,
        save_path=os.path.join(config.PLOTS_DIR, "detection_rate_heatmap.png"),
    )

    # Performance metrics
    metrics = compute_all_metrics(df)
    plot_throughput(
        metrics["throughput"],
        save_path=os.path.join(config.PLOTS_DIR, "throughput.png"),
    )
    plot_overhead(
        metrics["overhead"],
        save_path=os.path.join(config.PLOTS_DIR, "overhead.png"),
    )

    # CRC vs Checksum
    comp_path = os.path.join(config.DATASETS_DIR, "crc_vs_checksum.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        plot_crc_vs_checksum(
            comp_df,
            save_path=os.path.join(config.PLOTS_DIR, "crc_vs_checksum.png"),
        )

    # ML model evaluation plots
    rf_path = os.path.join(config.MODELS_DIR, "random_forest_model.joblib")
    xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.joblib")

    if os.path.exists(rf_path) and os.path.exists(xgb_path):
        rf_model = joblib.load(rf_path)
        xgb_model = joblib.load(xgb_path)

        fe = FeatureEngineer()
        X, y, feature_names = fe.prepare_features(df)
        _, X_test, _, y_test = fe.split_data(X, y)

        evaluator = ModelEvaluator()
        rf_res = evaluator.evaluate_model(rf_model, X_test, y_test, "Random Forest", feature_names)
        xgb_res = evaluator.evaluate_model(xgb_model, X_test, y_test, "XGBoost", feature_names)

        plot_confusion_matrix(
            {"Random Forest": rf_res["confusion_matrix"], "XGBoost": xgb_res["confusion_matrix"]},
            save_path=os.path.join(config.PLOTS_DIR, "confusion_matrix.png"),
        )

        roc_data = {
            "Random Forest": (rf_res["fpr"], rf_res["tpr"], rf_res["roc_auc"]),
            "XGBoost": (xgb_res["fpr"], xgb_res["tpr"], xgb_res["roc_auc"]),
        }
        plot_roc_curves(
            roc_data,
            save_path=os.path.join(config.PLOTS_DIR, "roc_curves.png"),
        )

        importance_data = {}
        for name, res in [("Random Forest", rf_res), ("XGBoost", xgb_res)]:
            if res["feature_importance"]:
                importance_data[name] = res["feature_importance"]
        if importance_data:
            plot_feature_importance(
                importance_data,
                save_path=os.path.join(config.PLOTS_DIR, "feature_importance.png"),
            )

    print(f"\n  All plots saved to: {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
