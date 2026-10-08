"""
==============================================================================
  Reliability Score — Data Integrity Assessment
==============================================================================
  Measures how much data integrity is preserved through the
  CRC + error correction pipeline:

    1. Hamming Distance: bit-level difference between original and received
    2. Integrity Score: fraction of bits that match the original
    3. Before/After comparison: integrity before and after error correction
==============================================================================
"""

from typing import List, Dict, Tuple


def hamming_distance(
    original: List[int], received: List[int]
) -> int:
    """
    Compute the Hamming distance between two binary sequences.

    Parameters
    ----------
    original : List[int]
        Original transmitted bit sequence.
    received : List[int]
        Received (possibly corrupted) bit sequence.

    Returns
    -------
    int
        Number of positions where the bits differ.
    """
    if len(original) != len(received):
        raise ValueError(
            f"Length mismatch: {len(original)} vs {len(received)}"
        )
    return sum(a != b for a, b in zip(original, received))


def integrity_score(
    original: List[int], received: List[int]
) -> float:
    """
    Compute the data integrity score (0.0 to 1.0).

    A score of 1.0 means perfect integrity (no bit differences).

    Returns
    -------
    float
        Fraction of matching bits.
    """
    n = len(original)
    if n == 0:
        return 1.0
    hd = hamming_distance(original, received)
    return 1.0 - (hd / n)


def compare_before_after_correction(
    original: List[int],
    corrupted: List[int],
    corrected: List[int],
) -> Dict[str, float]:
    """
    Compare data integrity before and after error correction.

    Parameters
    ----------
    original : List[int]
        Original transmitted codeword.
    corrupted : List[int]
        Received codeword (with errors).
    corrected : List[int]
        Codeword after error correction attempt.

    Returns
    -------
    dict
        {
            "hamming_before": ...,
            "hamming_after": ...,
            "integrity_before": ...,
            "integrity_after": ...,
            "improvement": ...,      # (after - before) integrity difference
            "perfect_correction": bool
        }
    """
    hd_before = hamming_distance(original, corrupted)
    hd_after = hamming_distance(original, corrected)
    int_before = integrity_score(original, corrupted)
    int_after = integrity_score(original, corrected)

    return {
        "hamming_before": hd_before,
        "hamming_after": hd_after,
        "integrity_before": round(int_before, 6),
        "integrity_after": round(int_after, 6),
        "improvement": round(int_after - int_before, 6),
        "perfect_correction": hd_after == 0,
    }


def batch_reliability_analysis(
    results: List[Dict],
) -> Dict[str, float]:
    """
    Aggregate reliability statistics from a list of correction results.

    Parameters
    ----------
    results : List[Dict]
        List of dicts from ``compare_before_after_correction``.

    Returns
    -------
    dict
        Aggregated statistics.
    """
    n = len(results)
    if n == 0:
        return {}

    avg_hd_before = sum(r["hamming_before"] for r in results) / n
    avg_hd_after = sum(r["hamming_after"] for r in results) / n
    avg_int_before = sum(r["integrity_before"] for r in results) / n
    avg_int_after = sum(r["integrity_after"] for r in results) / n
    avg_improvement = sum(r["improvement"] for r in results) / n
    perfect_rate = sum(1 for r in results if r["perfect_correction"]) / n

    return {
        "num_samples": n,
        "avg_hamming_before": round(avg_hd_before, 4),
        "avg_hamming_after": round(avg_hd_after, 4),
        "avg_integrity_before": round(avg_int_before, 6),
        "avg_integrity_after": round(avg_int_after, 6),
        "avg_improvement": round(avg_improvement, 6),
        "perfect_correction_rate": round(perfect_rate, 6),
    }
