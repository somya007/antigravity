"""
==============================================================================
  Belief Propagation (BP) Decoder for CRC Codes
==============================================================================
  Architecture:
    CRC codes can be represented as linear block codes with a parity-check
    matrix H derived from the generator polynomial G(x).

    The BP decoder performs iterative message passing on the Tanner graph
    corresponding to H, using soft channel information (LLR values).

    Key Insight:
      Each row of the parity-check matrix corresponds to a shifted version
      of the generator polynomial.  For a CRC of degree r over a codeword
      of length n, H is an r × n matrix.

  CRC as Early-Termination Criterion:
    After each BP iteration, we perform a hard decision on the updated
    LLR values and check the CRC.  If the CRC passes, we terminate early.
    This is the standard approach in 5G CRC-aided Polar decoding.

  Limitations:
    CRC parity-check matrices are generally *dense*, which limits the
    convergence properties of BP compared to LDPC codes.  If BP fails,
    the system falls back to ORBGRAND.
==============================================================================
"""

import math
from typing import List, Tuple, Optional
from core.polynomials import CRCPolynomial
from core.crc_decoder import CRCDecoder


class BPDecoder:
    """
    Soft-decision Belief Propagation decoder using CRC structure.

    Parameters
    ----------
    polynomial : CRCPolynomial
        The CRC polynomial defining the code structure.
    max_iterations : int
        Maximum number of BP iterations before declaring failure.
    damping : float
        Damping factor (0, 1] applied to check-to-variable messages
        to improve convergence on dense graphs.
    """

    def __init__(
        self,
        polynomial: CRCPolynomial,
        max_iterations: int = 50,
        damping: float = 0.5,
    ):
        self.poly = polynomial
        self.max_iterations = max_iterations
        self.damping = damping
        self.crc_decoder = CRCDecoder(polynomial)

    # ──────────────────── Parity-Check Matrix ────────────────────────

    def build_parity_check_matrix(self, codeword_length: int) -> List[List[int]]:
        """
        Construct the r × n parity-check matrix H from G(x).

        Each row i corresponds to x^i · G(x) mod x^n, represented
        as a binary vector of length n.  There are r = degree(G(x)) rows.

        Parameters
        ----------
        codeword_length : int
            Total length n of the codeword (message + CRC bits).

        Returns
        -------
        List[List[int]]
            The r × n parity-check matrix.
        """
        r = self.poly.width
        n = codeword_length
        k = n - r  # message length

        # G(x) as bit list (length r+1)
        g_bits = self.poly.poly_as_bit_list()

        H = []
        for i in range(r):
            row = [0] * n
            # Place the generator polynomial shifted by i positions
            offset = k - 1 - i  # start from the highest message position
            if offset < 0:
                # For the CRC portion, use identity
                row[k + i] = 1
            else:
                for j, bit in enumerate(g_bits):
                    if offset + j < n:
                        row[offset + j] ^= bit
            H.append(row)

        return H

    # ──────────────────── BP Decoding ────────────────────────────────

    def decode(
        self,
        received_bits: List[int],
        llr_values: List[float],
    ) -> Tuple[List[int], bool, int]:
        """
        Perform BP decoding with CRC-based early termination.

        Implements the sum-product algorithm on the Tanner graph
        defined by the CRC parity-check matrix.

        Parameters
        ----------
        received_bits : List[int]
            Hard-decision received bits (for CRC checking).
        llr_values : List[float]
            Channel log-likelihood ratios (soft information).

        Returns
        -------
        tuple
            (decoded_bits, success, iterations_used)
            - decoded_bits: hard-decision output after BP
            - success: True if CRC passes after decoding
            - iterations_used: number of iterations performed
        """
        n = len(received_bits)
        r = self.poly.width

        # Build parity-check matrix
        H = self.build_parity_check_matrix(n)
        m = len(H)  # number of check nodes (= r)

        # ── Identify non-zero entries in H ──
        # check_to_var[c] = list of variable node indices connected to check c
        # var_to_check[v] = list of check node indices connected to variable v
        check_to_var = [[] for _ in range(m)]
        var_to_check = [[] for _ in range(n)]

        for c in range(m):
            for v in range(n):
                if H[c][v] == 1:
                    check_to_var[c].append(v)
                    var_to_check[v].append(c)

        # ── Initialize messages ──
        # Channel LLR as initial variable-to-check messages
        channel_llr = list(llr_values)

        # var_to_check_msg[v][c] = message from variable v to check c
        v2c = {}
        for v in range(n):
            for c in var_to_check[v]:
                v2c[(v, c)] = channel_llr[v]

        # check_to_var_msg[c][v]
        c2v = {}
        for c in range(m):
            for v in check_to_var[c]:
                c2v[(c, v)] = 0.0

        # ── Iterative Message Passing ──
        for iteration in range(1, self.max_iterations + 1):

            # --- Check-to-variable update (min-sum approximation) ---
            for c in range(m):
                neighbors = check_to_var[c]
                for v in neighbors:
                    # Product of tanh(msg/2) for all neighbors except v
                    product = 1.0
                    for v2 in neighbors:
                        if v2 != v:
                            msg = v2c.get((v2, c), 0.0)
                            # Clip to avoid overflow
                            msg_clipped = max(min(msg, 20.0), -20.0)
                            product *= math.tanh(msg_clipped / 2.0)

                    # Clip product to avoid atanh domain issues
                    product = max(min(product, 1.0 - 1e-12), -1.0 + 1e-12)

                    new_msg = 2.0 * math.atanh(product)

                    # Apply damping for convergence on dense graphs
                    old_msg = c2v.get((c, v), 0.0)
                    c2v[(c, v)] = self.damping * new_msg + (1 - self.damping) * old_msg

            # --- Variable-to-check update ---
            for v in range(n):
                for c in var_to_check[v]:
                    total = channel_llr[v]
                    for c2 in var_to_check[v]:
                        if c2 != c:
                            total += c2v.get((c2, v), 0.0)
                    v2c[(v, c)] = total

            # --- Hard decision on posterior LLRs ---
            posterior_llr = list(channel_llr)
            for v in range(n):
                for c in var_to_check[v]:
                    posterior_llr[v] += c2v.get((c, v), 0.0)

            decoded = [0 if llr > 0 else 1 for llr in posterior_llr]

            # --- CRC early termination ---
            if self.crc_decoder.verify(decoded):
                return decoded, True, iteration

        # BP did not converge — return best effort
        return decoded, False, self.max_iterations
