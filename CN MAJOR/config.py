"""
==============================================================================
  CRC-Based Error Detection System — Global Configuration
==============================================================================
  Author  : CN Major Project
  Purpose : Centralized configuration for all modules including frame sizes,
            simulation parameters, polynomial choices, ML hyperparameters,
            and ORBGRAND decoder settings.
==============================================================================
"""

import os

# ─────────────────────────────── Paths ───────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
DATASETS_DIR = os.path.join(OUTPUT_DIR, "datasets")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# ─────────────────────────── Simulation ──────────────────────────────
RANDOM_SEED = 42
NUM_SIMULATIONS = 20_000           # Total frames to generate for ML dataset

# Frame sizes (in bits) used across simulations
FRAME_SIZES = [64, 128, 256, 512, 1024]

# Bit Error Rate values for channel simulation
BER_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1]

# Burst error lengths to test
BURST_LENGTHS = [2, 4, 8, 16, 32, 64]

# Noise types for dataset generation
NOISE_TYPES = ["none", "single_bit", "multi_bit", "burst", "random"]

# ──────────────────────── CRC Polynomials ────────────────────────────
# Default polynomials to use.  Detailed specs are in core/polynomials.py
DEFAULT_POLYNOMIALS = ["CRC-16-CCITT", "CRC-16-IBM", "CRC-32"]

# ──────────────────────── Decoding Settings ──────────────────────────
# Belief Propagation
BP_MAX_ITERATIONS = 50
BP_DAMPING_FACTOR = 0.5

# ORBGRAND
ORBGRAND_MAX_QUERIES = 10_000      # Max noise patterns to try before timeout
ORBGRAND_MAX_ERROR_WEIGHT = 5      # Max Hamming weight of guessed noise patterns

# ───────────────────────── ML Settings ───────────────────────────────
ML_TEST_SIZE = 0.2                 # 80/20 train-test split
ML_CV_FOLDS = 5                    # Stratified K-Fold cross-validation

# Random Forest hyperparameter grid
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

# XGBoost hyperparameter grid
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# ──────────────────── Visualization Settings ─────────────────────────
PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-darkgrid"
PLOT_FIGSIZE = (10, 6)
PLOT_COLORS = {
    "crc16_ccitt": "#3498db",  # Blue
    "crc16_ibm":   "#e74c3c",  # Red
    "crc32":       "#2ecc71",  # Green
    "checksum":    "#9b59b6",  # Purple
    "rf_model":    "#f39c12",  # Orange
    "xgb_model":   "#1abc9c",  # Teal
}

# ──────────────── Intelligent Controller ─────────────────────────
CONTROLLER_ALIASING_THRESHOLD = 0.15     # Aliasing probability threshold for CRC upgrade
CONTROLLER_ORBGRAND_NORMAL = 5_000       # ORBGRAND queries in Good state
CONTROLLER_ORBGRAND_AGGRESSIVE = 30_000  # ORBGRAND queries in Bad state (boosted)
CONTROLLER_WINDOW_SIZE = 50              # Sliding window for channel state monitoring
CONTROLLER_ENERGY_SAVE = True            # Enable Internet Checksum for Good-state headers
CONTROLLER_NUM_FRAMES = 200              # Default simulation frame count
