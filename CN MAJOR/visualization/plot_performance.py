"""
==============================================================================
  Visualization: Performance Metrics (Throughput, Overhead, CRC vs Checksum)
==============================================================================
"""

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def plot_throughput(
    throughput_data: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot CRC encoding throughput (bits/sec) vs. frame size for each polynomial.
    """
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)

    color_map = {
        "CRC-16-CCITT": config.PLOT_COLORS["crc16_ccitt"],
        "CRC-16-IBM": config.PLOT_COLORS["crc16_ibm"],
        "CRC-32": config.PLOT_COLORS["crc32"],
    }

    for pname, data in throughput_data.items():
        sizes = sorted(data.keys())
        rates = [data[s] / 1e6 for s in sizes]  # Convert to Mbps
        color = color_map.get(pname, "#333333")
        ax.plot(
            sizes, rates, marker="s", linewidth=2.5, markersize=8,
            color=color, label=pname,
        )

    ax.set_xlabel("Frame Size (bits)", fontsize=12)
    ax.set_ylabel("Throughput (Mbits/sec)", fontsize=12)
    ax.set_title(
        "CRC Encoding Throughput vs. Frame Size",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_overhead(
    overhead_data: Dict[str, Dict[int, float]],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot CRC computational overhead (time per frame) vs. frame size.
    """
    plt.style.use(config.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)

    color_map = {
        "CRC-16-CCITT": config.PLOT_COLORS["crc16_ccitt"],
        "CRC-16-IBM": config.PLOT_COLORS["crc16_ibm"],
        "CRC-32": config.PLOT_COLORS["crc32"],
    }

    for pname, data in overhead_data.items():
        sizes = sorted(data.keys())
        times = [data[s] * 1e6 for s in sizes]  # Convert to microseconds
        color = color_map.get(pname, "#333333")
        ax.plot(
            sizes, times, marker="^", linewidth=2.5, markersize=8,
            color=color, label=pname,
        )

    ax.set_xlabel("Frame Size (bits)", fontsize=12)
    ax.set_ylabel("Encoding Time (μs per frame)", fontsize=12)
    ax.set_title(
        "CRC Computational Overhead vs. Frame Size",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_crc_vs_checksum(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Bar chart comparing CRC vs. Checksum detection rates across noise types.
    """
    plt.style.use(config.PLOT_STYLE)

    # Aggregate across frame sizes
    agg = comparison_df.groupby("noise_type").agg(
        crc_rate=("crc_detection_rate", "mean"),
        cksum_rate=("checksum_detection_rate", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(agg))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, agg["crc_rate"] * 100, width,
        label="CRC-16-CCITT", color=config.PLOT_COLORS["crc16_ccitt"],
        edgecolor="white", linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2, agg["cksum_rate"] * 100, width,
        label="Internet Checksum", color=config.PLOT_COLORS["checksum"],
        edgecolor="white", linewidth=0.8,
    )

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 1,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xlabel("Error Type", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title(
        "CRC vs. Internet Checksum — Error Detection Comparison",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(agg["noise_type"], rotation=30, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 115])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
