"""
==============================================================================
  Visualization: Confusion Matrix for ML Models
==============================================================================
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config


def plot_confusion_matrix(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot confusion matrices for all evaluated models side by side.

    Parameters
    ----------
    confusion_matrices : dict
        {model_name: confusion_matrix_array}
    """
    plt.style.use(config.PLOT_STYLE)

    n_models = len(confusion_matrices)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    labels = ["Undetected\n(Aliasing)", "Detected"]
    color_maps = ["Blues", "Greens"]

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        cmap = color_maps[idx % len(color_maps)]

        # Normalize for display percentages
        cm_normalized = cm.astype(float) / cm.sum() * 100

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "Count"},
            linewidths=1, linecolor="white",
        )

        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5, i + 0.75,
                    f"({cm_normalized[i, j]:.1f}%)",
                    ha="center", va="center",
                    fontsize=9, color="gray",
                )

        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")

    fig.suptitle(
        "Confusion Matrix — ML Error Detection Prediction",
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
