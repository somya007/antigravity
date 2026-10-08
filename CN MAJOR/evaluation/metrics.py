"""
==============================================================================
  Performance Metrics — Throughput, Overhead, Detection Accuracy
==============================================================================
  Tracks and computes:
    - Computational Overhead: CRC computation time vs. frame size
    - Throughput: Effective bits processed per second
    - Detection Accuracy: Fraction of injected errors caught by CRC
    - Burst Error Detection Rate (BEDR): Fraction of burst errors
      with length > polynomial degree that go undetected (aliasing)
==============================================================================
"""

import time
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from core.polynomials import CRCPolynomial, get_polynomial
from core.crc_encoder import CRCEncoder
from core.data_framing import DataFramer


def compute_throughput(
    polynomial: CRCPolynomial,
    frame_sizes: List[int],
    num_trials: int = 500,
) -> Dict[int, float]:
    """
    Measure CRC encoding throughput (bits / second) for various frame sizes.

    Parameters
    ----------
    polynomial : CRCPolynomial
        Generator polynomial to benchmark.
    frame_sizes : list of int
        Frame sizes (in bits) to test.
    num_trials : int
        Number of encoding runs per frame size (averaged).

    Returns
    -------
    dict
        {frame_size: throughput_bps}
    """
    encoder = CRCEncoder(polynomial)
    results = {}

    for fs in frame_sizes:
        framer = DataFramer(fs, seed=42)
        frames = framer.generate_frames(num_trials)

        start = time.perf_counter()
        for frame in frames:
            encoder.encode(frame)
        elapsed = time.perf_counter() - start

        total_bits = fs * num_trials
        throughput = total_bits / elapsed if elapsed > 0 else float("inf")
        results[fs] = throughput

    return results


def compute_overhead(
    polynomial: CRCPolynomial,
    frame_sizes: List[int],
    num_trials: int = 500,
) -> Dict[int, float]:
    """
    Measure average CRC encoding time (seconds) per frame.

    Returns
    -------
    dict
        {frame_size: avg_time_seconds}
    """
    encoder = CRCEncoder(polynomial)
    results = {}

    for fs in frame_sizes:
        framer = DataFramer(fs, seed=42)
        frames = framer.generate_frames(num_trials)

        start = time.perf_counter()
        for frame in frames:
            encoder.encode(frame)
        elapsed = time.perf_counter() - start

        results[fs] = elapsed / num_trials

    return results


def compute_detection_accuracy(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute overall and per-noise-type detection accuracy from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation dataset with 'detected', 'noise_type' columns.

    Returns
    -------
    dict
        {"overall": ..., "single_bit": ..., "burst": ..., ...}
    """
    results = {}

    # Overall
    total = len(df)
    detected = df["detected"].sum()
    results["overall"] = detected / total if total > 0 else 0.0

    # Per noise type
    for nt in df["noise_type"].unique():
        subset = df[df["noise_type"] == nt]
        n = len(subset)
        d = subset["detected"].sum()
        results[nt] = d / n if n > 0 else 0.0

    return results


def compute_burst_detection_rate(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute burst error detection rate segmented by burst length
    relative to polynomial degree.

    Key insight: CRC with polynomial of degree r is guaranteed to detect
    all burst errors of length ≤ r.  For length > r, the undetected
    probability is ≈ 2^{-r}.

    Returns
    -------
    pd.DataFrame
        Columns: burst_length_actual, polynomial_degree, total_count,
                 detected_count, undetected_count, detection_rate,
                 exceeds_degree
    """
    burst_df = df[df["noise_type"] == "burst"].copy()

    if burst_df.empty:
        return pd.DataFrame()

    burst_df["exceeds_degree"] = (
        burst_df["burst_length_actual"] > burst_df["polynomial_degree"]
    )

    grouped = burst_df.groupby(
        ["burst_length_actual", "polynomial_degree", "exceeds_degree"]
    ).agg(
        total_count=("detected", "count"),
        detected_count=("detected", "sum"),
    ).reset_index()

    grouped["undetected_count"] = (
        grouped["total_count"] - grouped["detected_count"]
    )
    grouped["detection_rate"] = (
        grouped["detected_count"] / grouped["total_count"]
    )

    return grouped


def compute_all_metrics(
    df: pd.DataFrame,
    polynomial_names: List[str] = None,
    frame_sizes: List[int] = None,
) -> Dict[str, any]:
    """
    Compute all performance metrics in one call.

    Returns
    -------
    dict
        Contains throughput, overhead, detection_accuracy,
        burst_detection_rate data.
    """
    from config import FRAME_SIZES, DEFAULT_POLYNOMIALS

    if polynomial_names is None:
        polynomial_names = DEFAULT_POLYNOMIALS
    if frame_sizes is None:
        frame_sizes = FRAME_SIZES

    all_results = {
        "throughput": {},
        "overhead": {},
        "detection_accuracy": compute_detection_accuracy(df),
        "burst_detection_rate": compute_burst_detection_rate(df),
    }

    for pname in polynomial_names:
        poly = get_polynomial(pname)
        all_results["throughput"][pname] = compute_throughput(poly, frame_sizes)
        all_results["overhead"][pname] = compute_overhead(poly, frame_sizes)

    return all_results
