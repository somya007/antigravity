"""
==============================================================================
  Visualization: Detection Efficiency vs. Burst Length
==============================================================================
  Key deliverable — shows how CRC detection rate degrades as burst error
  length increases relative to the polynomial degree.
==============================================================================
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def plot_detection_efficiency_vs_burst_length(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot CRC detection efficiency as a function of burst length,
    faceted by polynomial type.

    Highlights the critical threshold where burst_length = polynomial_degree.
    """
    plt.style.use(config.PLOT_STYLE)

    burst_df = df[df["noise_type"] == "burst"].copy()
    if burst_df.empty:
        print("  No burst error data to plot.")
        return

    poly_types = burst_df["polynomial_type"].unique()
    fig, axes = plt.subplots(
        1, len(poly_types),
        figsize=(6 * len(poly_types), 5),
        sharey=True,
    )
    if len(poly_types) == 1:
        axes = [axes]

    color_map = {
        "CRC-16-CCITT": config.PLOT_COLORS["crc16_ccitt"],
        "CRC-16-IBM": config.PLOT_COLORS["crc16_ibm"],
        "CRC-32": config.PLOT_COLORS["crc32"],
    }

    for ax, ptype in zip(axes, poly_types):
        subset = burst_df[burst_df["polynomial_type"] == ptype]
        degree = subset["polynomial_degree"].iloc[0]

        # Group by burst length
        grouped = subset.groupby("burst_length_actual").agg(
            total=("detected", "count"),
            detected=("detected", "sum"),
        ).reset_index()
        grouped["detection_rate"] = grouped["detected"] / grouped["total"]

        color = color_map.get(ptype, "#333333")

        ax.plot(
            grouped["burst_length_actual"],
            grouped["detection_rate"] * 100,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=ptype,
        )

        # Critical threshold line
        ax.axvline(
            x=degree, color="red", linestyle="--", linewidth=1.5,
            alpha=0.8, label=f"Poly degree = {degree}",
        )

        ax.set_xlabel("Burst Error Length (bits)", fontsize=12)
        ax.set_ylabel("Detection Rate (%)", fontsize=12)
        ax.set_title(f"{ptype}\n(degree = {degree})", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim([-5, 105])
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CRC Detection Efficiency vs. Burst Error Length",
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


def plot_detection_rate_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Heatmap of detection rate across burst_length × polynomial_type.
    """
    import seaborn as sns
    plt.style.use(config.PLOT_STYLE)

    burst_df = df[df["noise_type"] == "burst"].copy()
    if burst_df.empty:
        return

    pivot = burst_df.pivot_table(
        values="detected",
        index="burst_length_actual",
        columns="polynomial_type",
        aggfunc="mean",
    ) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=0, vmax=100, ax=ax,
        cbar_kws={"label": "Detection Rate (%)"},
    )
    ax.set_title(
        "Burst Error Detection Rate (%)\nby Burst Length × Polynomial",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("CRC Polynomial", fontsize=12)
    ax.set_ylabel("Burst Length (bits)", fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
