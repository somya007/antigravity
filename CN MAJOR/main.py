"""
==============================================================================
  CRC Error Detection System — Main Pipeline Orchestrator
==============================================================================
  Runs the complete pipeline:
    1. Generate frames → Encode with CRC → Inject noise → Decode
    2. Attempt BP decoding → If fail, ORBGRAND → Final CRC validation
    3. Log results to dataset
    4. Train ML models
    5. Evaluate and visualize
==============================================================================
"""

import os
import sys
import time

import numpy as np
import pandas as pd

# ── Local imports ──
import config
from core.polynomials import get_polynomial, list_polynomials
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from core.data_framing import DataFramer
from channel.noise_injector import NoiseInjector
from channel.channel_model import ChannelModel
from decoding.bp_decoder import BPDecoder
from decoding.orbgrand_decoder import ORBGRANDDecoder
from ml.dataset_generator import generate_dataset
from ml.feature_engineering import FeatureEngineer
from ml.model_training import ModelTrainer
from ml.model_evaluation import ModelEvaluator
from evaluation.metrics import compute_all_metrics
from evaluation.checksum_comparison import compare_crc_vs_checksum
from evaluation.reliability_score import (
    compare_before_after_correction,
    batch_reliability_analysis,
)
from visualization.plot_detection_efficiency import (
    plot_detection_efficiency_vs_burst_length,
    plot_detection_rate_heatmap,
)
from visualization.plot_confusion_matrix import plot_confusion_matrix
from visualization.plot_performance import (
    plot_throughput,
    plot_overhead,
    plot_crc_vs_checksum,
)
from visualization.plot_roc_curve import plot_roc_curves, plot_feature_importance
from controller.intelligent_controller import run_controller_demo


def run_advanced_decoding_demo():
    """
    Demonstrate the BP → ORBGRAND → CRC flow on a small example.
    """
    print("\n" + "=" * 70)
    print("  ADVANCED DECODING DEMO: BP → ORBGRAND → CRC Verification")
    print("=" * 70)

    poly = get_polynomial("CRC-16-CCITT")
    encoder = CRCEncoder(poly)
    decoder = CRCDecoder(poly)
    channel = ChannelModel(seed=42)
    bp = BPDecoder(poly, max_iterations=30, damping=0.5)
    orbgrand = ORBGRANDDecoder(poly, max_queries=5000, max_error_weight=4)

    framer = DataFramer(64, seed=42)
    results = []

    for trial in range(20):
        # Generate and encode
        msg = framer.generate_random_frame()
        codeword = encoder.encode(msg)

        # Transmit through noisy channel
        received, llr, error_pattern = channel.bsc_transmit(codeword, 0.02)
        num_errors = sum(error_pattern)
        if num_errors == 0:
            continue

        reliability = channel.get_reliability(llr)

        # Step 1: Check CRC directly
        crc_pass = decoder.verify(received)

        if crc_pass:
            print(f"  Trial {trial+1}: {num_errors} errors — CRC passed (undetected!)")
            continue

        # Step 2: Try BP decoding
        bp_decoded, bp_success, bp_iters = bp.decode(received, llr)

        if bp_success:
            rel = compare_before_after_correction(codeword, received, bp_decoded)
            results.append(rel)
            print(
                f"  Trial {trial+1}: {num_errors} errors — "
                f"BP corrected in {bp_iters} iterations "
                f"(integrity: {rel['integrity_before']:.4f} → {rel['integrity_after']:.4f})"
            )
            continue

        # Step 3: Try ORBGRAND
        orb_decoded, orb_success, orb_queries = orbgrand.decode(
            received, reliability
        )

        if orb_success:
            rel = compare_before_after_correction(codeword, received, orb_decoded)
            results.append(rel)
            print(
                f"  Trial {trial+1}: {num_errors} errors — "
                f"ORBGRAND corrected in {orb_queries} queries "
                f"(integrity: {rel['integrity_before']:.4f} → {rel['integrity_after']:.4f})"
            )
        else:
            rel = compare_before_after_correction(codeword, received, orb_decoded)
            results.append(rel)
            print(
                f"  Trial {trial+1}: {num_errors} errors — "
                f"FAILED after {orb_queries} queries"
            )

    if results:
        stats = batch_reliability_analysis(results)
        print(f"\n  Reliability Summary:")
        for k, v in stats.items():
            print(f"    {k}: {v}")


def main():
    """Run the full CRC Error Detection System pipeline."""
    start_time = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  CRC-Based Error Detection System with ML Error Impact Predictor  ║")
    print("║  M.Tech Level Implementation                                      ║")
    print("╚" + "═" * 68 + "╝\n")

    # ── Phase 1 & 2: Dataset Generation ──
    print("[Phase 1-2] Generating simulation dataset...\n")
    dataset_path = os.path.join(config.DATASETS_DIR, "crc_simulation_dataset.csv")
    df = generate_dataset(
        num_samples=config.NUM_SIMULATIONS,
        save_path=dataset_path,
    )

    # ── Phase 3: Advanced Decoding Demo ──
    print("\n[Phase 3] Running advanced decoding demo...")
    run_advanced_decoding_demo()

    # ── Phase 4: ML Pipeline ──
    print("\n[Phase 4] Training ML models...\n")

    fe = FeatureEngineer()
    X, y, feature_names = fe.prepare_features(df)

    # Check if both classes exist (need undetected errors for ML)
    unique_classes = np.unique(y)
    ml_trained = False
    rf_results = None
    xgb_results = None
    evaluator = ModelEvaluator()

    if len(unique_classes) < 2:
        print("  WARNING: Only one class found in dataset.")
        print("           All errors were detected — no aliasing events.")
        print("           Increase NUM_SIMULATIONS (100K+) for ML training.")
        print("           Skipping ML training...")
    else:
        X_train, X_test, y_train, y_test = fe.split_data(X, y)

        trainer = ModelTrainer()
        trainer.train_random_forest(X_train, y_train, use_quick_grid=True)
        trainer.train_xgboost(X_train, y_train, use_quick_grid=True)
        trainer.save_models()

        rf_results = evaluator.evaluate_model(
            trainer.rf_model, X_test, y_test, "Random Forest", feature_names
        )
        xgb_results = evaluator.evaluate_model(
            trainer.xgb_model, X_test, y_test, "XGBoost", feature_names
        )
        evaluator.compare_models()
        evaluator.save_report()
        ml_trained = True

    # ── Phase 5: Evaluation ──
    print("\n[Phase 5] Computing performance metrics...\n")

    metrics = compute_all_metrics(df)

    print("  Detection Accuracy:")
    for k, v in metrics["detection_accuracy"].items():
        print(f"    {k:15s}: {v:.4f}")

    print("\n  Running CRC vs Checksum comparison...")
    comparison_df = compare_crc_vs_checksum(num_trials=5000)
    comp_path = os.path.join(config.DATASETS_DIR, "crc_vs_checksum.csv")
    comparison_df.to_csv(comp_path, index=False)

    # ── Phase 6: Visualization ──
    print("\n[Phase 6] Generating visualizations...\n")

    plot_detection_efficiency_vs_burst_length(
        df,
        save_path=os.path.join(config.PLOTS_DIR, "detection_efficiency_vs_burst.png"),
        show=False,
    )

    plot_detection_rate_heatmap(
        df,
        save_path=os.path.join(config.PLOTS_DIR, "detection_rate_heatmap.png"),
        show=False,
    )

    plot_confusion_matrix_guard = ml_trained and rf_results and xgb_results
    if plot_confusion_matrix_guard:
        plot_confusion_matrix(
            {
                "Random Forest": rf_results["confusion_matrix"],
                "XGBoost": xgb_results["confusion_matrix"],
            },
            save_path=os.path.join(config.PLOTS_DIR, "confusion_matrix.png"),
            show=False,
        )

    plot_throughput(
        metrics["throughput"],
        save_path=os.path.join(config.PLOTS_DIR, "throughput.png"),
        show=False,
    )

    plot_overhead(
        metrics["overhead"],
        save_path=os.path.join(config.PLOTS_DIR, "overhead.png"),
        show=False,
    )

    plot_crc_vs_checksum(
        comparison_df,
        save_path=os.path.join(config.PLOTS_DIR, "crc_vs_checksum.png"),
        show=False,
    )

    if plot_confusion_matrix_guard:
        roc_data = {}
        for name in ["Random Forest", "XGBoost"]:
            res = evaluator.results[name]
            roc_data[name] = (res["fpr"], res["tpr"], res["roc_auc"])

        plot_roc_curves(
            roc_data,
            save_path=os.path.join(config.PLOTS_DIR, "roc_curves.png"),
            show=False,
        )

        importance_data = {}
        for name in ["Random Forest", "XGBoost"]:
            res = evaluator.results[name]
            if res["feature_importance"]:
                importance_data[name] = res["feature_importance"]

        if importance_data:
            plot_feature_importance(
                importance_data,
                save_path=os.path.join(config.PLOTS_DIR, "feature_importance.png"),
                show=False,
            )

    # ── Phase 7: Intelligent Controller Demo ──
    print("\n[Phase 7] Running Intelligent CRC Controller...\n")
    xgb_model_for_ctrl = trainer.xgb_model if ml_trained else None
    run_controller_demo(
        num_frames=config.CONTROLLER_NUM_FRAMES,
        seed=config.RANDOM_SEED,
        xgb_model=xgb_model_for_ctrl,
        save_path=os.path.join(config.PLOTS_DIR, "intelligent_controller_dashboard.png"),
        show_dashboard=False,
    )

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Pipeline completed in {elapsed:.1f} seconds")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Models:   {config.MODELS_DIR}")
    print(f"  Plots:    {config.PLOTS_DIR}")
    print(f"  Reports:  {config.REPORTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
