"""
==============================================================================
  Visualization: ROC Curves for ML Models
==============================================================================
"""

import os
from typing import Dict, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np

import config


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot ROC curves for all evaluated models on a single figure.

    Parameters
    ----------
    roc_data : dict
        {model_name: (fpr_array, tpr_array, auc_score)}
    """
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [
        config.PLOT_COLORS["rf_model"],
        config.PLOT_COLORS["xgb_model"],
        "#e74c3c",
        "#9b59b6",
    ]

    for idx, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(
            fpr, tpr,
            linewidth=2.5,
            color=color,
            label=f"{name} (AUC = {auc:.4f})",
        )

    # Diagonal reference line (random classifier)
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", linewidth=1.5, color="gray",
        alpha=0.7, label="Random Classifier",
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curve — ML Error Detection Prediction",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    importance_data: Dict[str, List[Tuple[str, float]]],
    top_n: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot horizontal bar chart of feature importances for each model.

    Parameters
    ----------
    importance_data : dict
        {model_name: [(feature_name, importance), ...]}
    top_n : int
        Number of top features to display.
    """
    plt.style.use(config.PLOT_STYLE)

    n_models = len(importance_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    colors = [
        config.PLOT_COLORS["rf_model"],
        config.PLOT_COLORS["xgb_model"],
    ]

    for idx, (name, importances) in enumerate(importance_data.items()):
        ax = axes[idx]
        top = importances[:top_n]
        top.reverse()  # So most important is at top

        names = [t[0] for t in top]
        values = [t[1] for t in top]
        color = colors[idx % len(colors)]

        ax.barh(names, values, color=color, edgecolor="white", linewidth=0.8)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{name}\nFeature Importance", fontsize=13, fontweight="bold")

        # Value labels
        for i, v in enumerate(values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle(
        "Feature Importance — ML Error Impact Predictor",
        fontsize=15, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
