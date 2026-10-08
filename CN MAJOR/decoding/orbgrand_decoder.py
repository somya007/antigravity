"""
==============================================================================
  ORBGRAND (Ordered Reliability Bits GRAND) Decoder
==============================================================================
  Algorithm:
    ORBGRAND is a code-agnostic, soft-detection universal decoder from the
    GRAND (Guessing Random Additive Noise Decoding) family.

    Core idea:
      Instead of decoding the codeword structure, ORBGRAND "guesses" the
      noise pattern that corrupted the transmission.  It tries patterns
      in order of decreasing likelihood (most probable noise first).

    Steps:
      1. Receive hard-decision bits y and soft-decision reliability |LLR|.
      2. Sort bit positions by reliability (ascending → least reliable first).
      3. Generate noise patterns in order of increasing Hamming weight,
         within each weight ordered by combined unreliability.
      4. For each candidate noise pattern e:
           - Compute candidate codeword:  c* = y ⊕ e
           - Check CRC:  if G(x) | c*(x)  →  output c* and terminate
      5. If max queries exceeded → declare decoding failure.

    CRC as Stopping Criterion:
      The CRC remainder is the *only* way ORBGRAND validates a guess.
      This makes CRC the verifier/stopping criterion for the algorithm —
      a key M.Tech-level insight from the project flowchart.

  Reference:
    Duffy, Médard, et al., "Ordered Reliability Bits Guessing Random
    Additive Noise Decoding," IEEE Trans. Signal Processing, 2022.
==============================================================================
"""

from itertools import combinations
from typing import List, Tuple, Optional
from core.polynomials import CRCPolynomial
from core.crc_decoder import CRCDecoder


class ORBGRANDDecoder:
    """
    ORBGRAND universal decoder using CRC as the validity check.

    Parameters
    ----------
    polynomial : CRCPolynomial
        The CRC polynomial used as the code's parity check.
    max_queries : int
        Maximum number of noise patterns to try before timeout.
    max_error_weight : int
        Maximum Hamming weight of noise patterns to consider.
    """

    def __init__(
        self,
        polynomial: CRCPolynomial,
        max_queries: int = 10_000,
        max_error_weight: int = 5,
    ):
        self.poly = polynomial
        self.max_queries = max_queries
        self.max_error_weight = max_error_weight
        self.crc_decoder = CRCDecoder(polynomial)

    # ──────────────────── Public API ─────────────────────────────────

    def decode(
        self,
        received_bits: List[int],
        reliability: List[float],
    ) -> Tuple[List[int], bool, int]:
        """
        Perform ORBGRAND decoding.

        Parameters
        ----------
        received_bits : List[int]
            Hard-decision received bits (y).
        reliability : List[float]
            Per-bit reliability values (|LLR|).
            Lower values → less reliable → more likely to be in error.

        Returns
        -------
        tuple
            (corrected_bits, success, queries_used)
            - corrected_bits: the decoded codeword (or best guess)
            - success: True if a valid codeword was found
            - queries_used: number of noise patterns tested
        """
        n = len(received_bits)

        # ── Step 1: Sort positions by reliability (ascending) ──
        # Least reliable bits first — these are most likely to be in error
        sorted_positions = sorted(
            range(n), key=lambda i: reliability[i]
        )

        # ── Step 2: Check the received word itself (weight-0 pattern) ──
        queries = 0
        if self.crc_decoder.verify(received_bits):
            return list(received_bits), True, 0

        # ── Step 3: Try noise patterns by increasing Hamming weight ──
        for weight in range(1, self.max_error_weight + 1):
            # Generate combinations of 'weight' positions from
            # the sorted (least reliable first) positions.
            # We limit to the top most-unreliable positions for tractability.
            candidate_positions = sorted_positions[
                : min(n, weight * 14 + 20)
            ]

            for error_positions in combinations(candidate_positions, weight):
                queries += 1

                # Build noise pattern
                candidate = list(received_bits)
                for pos in error_positions:
                    candidate[pos] ^= 1  # Flip the bit

                # ── CRC as stopping criterion ──
                if self.crc_decoder.verify(candidate):
                    return candidate, True, queries

                # Check timeout
                if queries >= self.max_queries:
                    return list(received_bits), False, queries

        # Exhausted all patterns without finding a valid codeword
        return list(received_bits), False, queries

    # ──────────────────── Advanced: Weighted Ordering ────────────────

    def decode_weighted(
        self,
        received_bits: List[int],
        reliability: List[float],
    ) -> Tuple[List[int], bool, int]:
        """
        ORBGRAND with noise patterns ordered by total unreliability score.

        Instead of pure combinatorial ordering, patterns are scored by
        the sum of (1/reliability) for each flipped position.  Patterns
        with higher total unreliability are tried first.

        This is closer to the CDF-ORBGRAND variant described in recent
        literature.
        """
        n = len(received_bits)

        # Check received word first
        if self.crc_decoder.verify(received_bits):
            return list(received_bits), True, 0

        # Compute unreliability scores (inverse of |LLR|)
        unreliability = [
            1.0 / max(abs(r), 1e-10) for r in reliability
        ]

        # Sort positions by unreliability descending
        sorted_positions = sorted(
            range(n), key=lambda i: unreliability[i], reverse=True
        )

        queries = 0

        # Generate and score all candidate patterns, then sort
        all_candidates = []

        for weight in range(1, self.max_error_weight + 1):
            # Limit search space to top unreliable positions
            search_positions = sorted_positions[
                : min(n, weight * 10 + 15)
            ]

            for error_positions in combinations(search_positions, weight):
                # Score = sum of unreliability at error positions
                score = sum(unreliability[pos] for pos in error_positions)
                all_candidates.append((score, error_positions))

        # Sort by score descending (most likely noise patterns first)
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        # Try each candidate
        for score, error_positions in all_candidates:
            queries += 1

            candidate = list(received_bits)
            for pos in error_positions:
                candidate[pos] ^= 1

            if self.crc_decoder.verify(candidate):
                return candidate, True, queries

            if queries >= self.max_queries:
                break

        return list(received_bits), False, queries

    # ──────────────────── Statistics ─────────────────────────────────

    @staticmethod
    def estimate_complexity(n: int, max_weight: int, top_k: int) -> int:
        """
        Estimate the maximum number of queries for given parameters.

        Parameters
        ----------
        n : int
            Codeword length.
        max_weight : int
            Maximum error weight.
        top_k : int
            Number of least-reliable positions considered per weight level.

        Returns
        -------
        int
            Upper bound on the number of CRC queries.
        """
        from math import comb
        total = 0
        for w in range(1, max_weight + 1):
            k = min(n, w * 8 + 10)
            total += comb(k, w)
        return total
