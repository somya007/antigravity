"""
==============================================================================
  Channel Model — BSC & Gilbert-Elliott Burst Channel
==============================================================================
  Provides channel simulation with both hard-decision (bit) and
  soft-decision (log-likelihood ratio) outputs.

  Models:
    1. BSC (Binary Symmetric Channel):
       Each bit flipped with constant probability p.

    2. Gilbert-Elliott Model:
       Two-state Markov chain (Good / Bad) with different BERs.
       Models bursty channels common in wireless communications.

  Soft outputs (LLR):
    For each received bit y_i, the log-likelihood ratio is:
        LLR_i = log( P(y_i | x_i=0) / P(y_i | x_i=1) )
    For BSC with crossover probability p:
        If y_i = 0:  LLR = log((1-p)/p)
        If y_i = 1:  LLR = log(p/(1-p)) = -log((1-p)/p)
==============================================================================
"""

import math
import random
from typing import List, Tuple, Optional


class ChannelModel:
    """
    Simulated communication channel with hard and soft outputs.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    # ──────────────────── BSC ────────────────────────────────────────

    def bsc_transmit(
        self, frame: List[int], crossover_prob: float = 0.01
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Transmit through a Binary Symmetric Channel.

        Parameters
        ----------
        frame : List[int]
            Transmitted bit sequence.
        crossover_prob : float
            Probability of flipping each bit (BER).

        Returns
        -------
        tuple
            (received_hard_bits, llr_values, error_pattern)
        """
        p = crossover_prob
        n = len(frame)

        error_pattern = [
            1 if self.rng.random() < p else 0 for _ in range(n)
        ]
        received = [b ^ e for b, e in zip(frame, error_pattern)]

        # Compute LLR for each received bit
        llr_values = self._compute_bsc_llr(received, p)

        return received, llr_values, error_pattern

    def _compute_bsc_llr(
        self, received: List[int], p: float
    ) -> List[float]:
        """
        Compute log-likelihood ratios for BSC channel.

        LLR > 0 → bit more likely 0
        LLR < 0 → bit more likely 1
        |LLR| large → high reliability
        """
        # Avoid log(0)
        p = max(p, 1e-10)
        p = min(p, 1 - 1e-10)

        base_llr = math.log((1 - p) / p)
        llr_values = []
        for bit in received:
            if bit == 0:
                llr_values.append(base_llr)   # Likely correct → positive
            else:
                llr_values.append(-base_llr)  # Likely correct → negative
        return llr_values

    # ──────────────────── Gilbert-Elliott ─────────────────────────────

    def gilbert_elliott_transmit(
        self,
        frame: List[int],
        p_good: float = 0.001,
        p_bad: float = 0.1,
        p_gb: float = 0.05,
        p_bg: float = 0.5,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Transmit through a Gilbert-Elliott bursty channel.

        Two-state Markov model:
          GOOD state: BER = p_good  (low error rate)
          BAD  state: BER = p_bad   (high error rate)

          p_gb = P(Good → Bad)     — probability of entering burst
          p_bg = P(Bad → Good)     — probability of leaving burst

        Parameters
        ----------
        frame : List[int]
            Transmitted bit sequence.
        p_good : float
            BER in the GOOD state.
        p_bad : float
            BER in the BAD state.
        p_gb : float
            Transition probability Good → Bad.
        p_bg : float
            Transition probability Bad → Good.

        Returns
        -------
        tuple
            (received_hard_bits, llr_values, error_pattern)
        """
        n = len(frame)
        state = "good"  # Start in good state
        received = []
        llr_values = []
        error_pattern = []

        for i in range(n):
            # Current BER depends on state
            ber = p_good if state == "good" else p_bad

            # Introduce error with probability ber
            error = 1 if self.rng.random() < ber else 0
            error_pattern.append(error)
            received.append(frame[i] ^ error)

            # Compute LLR with state-dependent reliability
            p_eff = max(min(ber, 1 - 1e-10), 1e-10)
            base_llr = math.log((1 - p_eff) / p_eff)
            llr = base_llr if received[-1] == 0 else -base_llr
            llr_values.append(llr)

            # State transition
            if state == "good":
                if self.rng.random() < p_gb:
                    state = "bad"
            else:
                if self.rng.random() < p_bg:
                    state = "good"

        return received, llr_values, error_pattern

    # ──────────────────── Utility ────────────────────────────────────

    @staticmethod
    def hard_decision(llr_values: List[float]) -> List[int]:
        """
        Convert LLR values to hard-decision bits.
        LLR > 0 → 0, LLR ≤ 0 → 1.
        """
        return [0 if llr > 0 else 1 for llr in llr_values]

    @staticmethod
    def get_reliability(llr_values: List[float]) -> List[float]:
        """
        Compute per-bit reliability as |LLR|.
        Higher values = more reliable bits.
        """
        return [abs(llr) for llr in llr_values]
