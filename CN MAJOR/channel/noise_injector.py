"""
==============================================================================
  Noise Injection Module
==============================================================================
  Simulates realistic channel errors for CRC testing:
    1. Single-bit error    — exactly one random bit flipped
    2. Multi-bit error     — N randomly chosen bits flipped
    3. Burst error         — L consecutive bits flipped (starting at random)
    4. Random (BER-based)  — each bit flipped independently with probability p

  Every method returns (corrupted_frame, error_pattern) so the caller
  can track exactly which bits were affected for analysis.
==============================================================================
"""

import random
from typing import List, Tuple, Optional


class NoiseInjector:
    """
    Channel noise simulator for binary frames.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    # ──────────────────── Error Injection Methods ────────────────────

    def inject_single_bit_error(
        self, frame: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Flip exactly one randomly chosen bit.

        Returns
        -------
        tuple
            (corrupted_frame, error_pattern) where error_pattern has a
            single 1 at the flipped position.
        """
        n = len(frame)
        error_pattern = [0] * n
        pos = self.rng.randint(0, n - 1)
        error_pattern[pos] = 1
        corrupted = [b ^ e for b, e in zip(frame, error_pattern)]
        return corrupted, error_pattern

    def inject_multi_bit_error(
        self, frame: List[int], num_errors: int = 3
    ) -> Tuple[List[int], List[int]]:
        """
        Flip ``num_errors`` randomly chosen (distinct) bits.

        Parameters
        ----------
        num_errors : int
            Number of bits to corrupt (capped at frame length).
        """
        n = len(frame)
        num_errors = min(num_errors, n)
        error_positions = self.rng.sample(range(n), num_errors)

        error_pattern = [0] * n
        for pos in error_positions:
            error_pattern[pos] = 1

        corrupted = [b ^ e for b, e in zip(frame, error_pattern)]
        return corrupted, error_pattern

    def inject_burst_error(
        self, frame: List[int], burst_length: int = 8
    ) -> Tuple[List[int], List[int]]:
        """
        Flip ``burst_length`` *consecutive* bits starting at a random position.

        A burst error of length L affects bits in positions [start, start+L-1].
        The first and last bits of the burst are always 1 (by convention);
        inner bits are flipped with probability 0.5 to model realistic bursts.

        Parameters
        ----------
        burst_length : int
            Length of the burst (capped at frame length).
        """
        n = len(frame)
        burst_length = min(burst_length, n)

        # Random start position so the burst fits within the frame
        max_start = n - burst_length
        start = self.rng.randint(0, max_start) if max_start > 0 else 0

        error_pattern = [0] * n

        for i in range(burst_length):
            pos = start + i
            if i == 0 or i == burst_length - 1:
                # First and last bits of a burst are always in error
                error_pattern[pos] = 1
            else:
                # Inner bits flipped with 50% probability
                error_pattern[pos] = self.rng.randint(0, 1)

        corrupted = [b ^ e for b, e in zip(frame, error_pattern)]
        return corrupted, error_pattern

    def inject_random_errors(
        self, frame: List[int], ber: float = 0.01
    ) -> Tuple[List[int], List[int]]:
        """
        Flip each bit independently with probability ``ber`` (Bit Error Rate).

        This models a Binary Symmetric Channel (BSC).

        Parameters
        ----------
        ber : float
            Crossover probability (0 < ber < 1).
        """
        n = len(frame)
        error_pattern = [
            1 if self.rng.random() < ber else 0 for _ in range(n)
        ]
        corrupted = [b ^ e for b, e in zip(frame, error_pattern)]
        return corrupted, error_pattern

    # ──────────────────── Utility ────────────────────────────────────

    @staticmethod
    def count_errors(error_pattern: List[int]) -> int:
        """Count the number of flipped bits in the error pattern."""
        return sum(error_pattern)

    @staticmethod
    def get_burst_length(error_pattern: List[int]) -> int:
        """
        Compute the burst length: distance from first to last error bit + 1.
        Returns 0 if there are no errors.
        """
        positions = [i for i, bit in enumerate(error_pattern) if bit == 1]
        if not positions:
            return 0
        return positions[-1] - positions[0] + 1

    @staticmethod
    def classify_error(error_pattern: List[int]) -> str:
        """
        Classify an error pattern as:
          'none', 'single_bit', 'multi_bit', or 'burst'.
        """
        num_errors = sum(error_pattern)
        if num_errors == 0:
            return "none"
        if num_errors == 1:
            return "single_bit"

        # Check if errors are contiguous → burst
        positions = [i for i, bit in enumerate(error_pattern) if bit == 1]
        span = positions[-1] - positions[0] + 1
        if span <= num_errors * 1.5:  # Heuristic: burst if errors are clustered
            return "burst"
        return "multi_bit"

    def inject_error(
        self,
        frame: List[int],
        noise_type: str,
        ber: float = 0.01,
        burst_length: int = 8,
        num_errors: int = 3,
    ) -> Tuple[List[int], List[int]]:
        """
        Unified error injection dispatcher.

        Parameters
        ----------
        noise_type : str
            One of 'none', 'single_bit', 'multi_bit', 'burst', 'random'.
        """
        if noise_type == "none":
            return list(frame), [0] * len(frame)
        elif noise_type == "single_bit":
            return self.inject_single_bit_error(frame)
        elif noise_type == "multi_bit":
            return self.inject_multi_bit_error(frame, num_errors)
        elif noise_type == "burst":
            return self.inject_burst_error(frame, burst_length)
        elif noise_type == "random":
            return self.inject_random_errors(frame, ber)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
