"""
==============================================================================
  CRC Decoder — Zero-Remainder Verification
==============================================================================
  Decision Logic:
    The receiver divides the entire received frame T'(x) by G(x).
      • If  T'(x) mod G(x) == 0  →  frame accepted (no error detected).
      • If  T'(x) mod G(x) != 0  →  frame rejected (error detected).

  The syndrome S(x) = T'(x) mod G(x) can also indicate the error pattern
  for correctable error classes.

  Note on Aliasing (Undetected Errors):
    An error pattern E(x) goes undetected iff E(x) mod G(x) == 0,
    i.e. E(x) is a multiple of G(x).  The probability of this for a
    random error of weight ≥ r+1 is approximately 2^{-r}.
==============================================================================
"""

from typing import List, Tuple
from core.polynomials import CRCPolynomial
from core.crc_encoder import CRCEncoder


class CRCDecoder:
    """
    CRC Decoder / Verifier.

    Parameters
    ----------
    polynomial : CRCPolynomial
        Must match the polynomial used by the encoder.
    """

    def __init__(self, polynomial: CRCPolynomial):
        self.poly = polynomial
        self._encoder = CRCEncoder(polynomial)

    # ──────────────────── Public API ─────────────────────────────────

    def verify(self, received_bits: List[int]) -> bool:
        """
        Verify the integrity of a received codeword.

        Performs modulo-2 division of the received frame by G(x).
        Returns True if the remainder is all-zero (no error detected).

        Parameters
        ----------
        received_bits : List[int]
            The received codeword (message + CRC) as a bit list.

        Returns
        -------
        bool
            True if CRC check passes (no error detected), False otherwise.
        """
        syndrome = self.get_syndrome(received_bits)
        return all(bit == 0 for bit in syndrome)

    def get_syndrome(self, received_bits: List[int]) -> List[int]:
        """
        Compute the syndrome S(x) = T'(x) mod G(x).

        Parameters
        ----------
        received_bits : List[int]
            Received codeword (message + CRC).

        Returns
        -------
        List[int]
            Syndrome bits of length r (polynomial degree).
            All zeros indicates no detectable error.
        """
        divisor = self.poly.poly_as_bit_list()
        return self._encoder._modulo2_division(list(received_bits), divisor)

    def decode_and_verify(
        self, received_bits: List[int]
    ) -> Tuple[List[int], bool, List[int]]:
        """
        Extract the message, verify integrity, and return the syndrome.

        Parameters
        ----------
        received_bits : List[int]
            Full received codeword.

        Returns
        -------
        tuple
            (extracted_message, is_valid, syndrome)
            - extracted_message: bits without the CRC portion
            - is_valid: True if CRC passes
            - syndrome: the r-bit remainder
        """
        r = self.poly.width
        message = received_bits[:-r]
        syndrome = self.get_syndrome(received_bits)
        is_valid = all(bit == 0 for bit in syndrome)
        return message, is_valid, syndrome


# ═══════════════════════ Convenience Function ════════════════════════

def verify_frame(
    received_bits: List[int], polynomial: CRCPolynomial
) -> Tuple[bool, List[int]]:
    """
    Convenience wrapper: verify a frame and return (is_valid, syndrome).
    """
    dec = CRCDecoder(polynomial)
    syndrome = dec.get_syndrome(received_bits)
    is_valid = all(b == 0 for b in syndrome)
    return is_valid, syndrome
