"""
==============================================================================
  CRC Encoder — Bitwise XOR Modulo-2 Division
==============================================================================
  Mathematical Principle:
    Given message polynomial M(x) and generator polynomial G(x) of degree r,
    the transmitted codeword is:

        T(x)  =  x^r · M(x)  +  R(x)

    where R(x) = (x^r · M(x)) mod G(x)  (modulo-2 arithmetic).

    This guarantees that T(x) is exactly divisible by G(x), enabling the
    receiver to detect errors by checking whether the received polynomial
    divides evenly by G(x).

  Two implementations are provided:
    1. Bit-level:  operates on List[int] of 0/1 — useful for analysis.
    2. Register-level:  operates on bytes — fast, standard-compliant.
==============================================================================
"""

from typing import List, Tuple
from core.polynomials import CRCPolynomial


class CRCEncoder:
    """
    CRC Encoder supporting arbitrary CRC polynomials.

    Parameters
    ----------
    polynomial : CRCPolynomial
        The generator polynomial specification.
    """

    def __init__(self, polynomial: CRCPolynomial):
        self.poly = polynomial

    # ──────────────────── Public API ─────────────────────────────────

    def encode(self, message_bits: List[int]) -> List[int]:
        """
        Compute the CRC and append it to the message.

        Parameters
        ----------
        message_bits : List[int]
            Original data as a list of 0s and 1s.

        Returns
        -------
        List[int]
            The transmitted codeword  T = message_bits + crc_bits.
        """
        crc_bits = self.compute_crc_bits(message_bits)
        return message_bits + crc_bits

    def compute_crc_bits(self, message_bits: List[int]) -> List[int]:
        """
        Compute only the CRC remainder bits (R(x)).

        Uses bitwise XOR-based modulo-2 division on bit lists.
        """
        r = self.poly.width  # degree of G(x)

        # Step 1:  Multiply M(x) by x^r  →  append r zero bits
        dividend = message_bits + [0] * r

        # Step 2:  Get the divisor as a bit list (includes leading 1)
        divisor = self.poly.poly_as_bit_list()

        # Step 3:  Perform modulo-2 long division
        remainder = self._modulo2_division(dividend, divisor)

        return remainder

    def compute_crc_register(self, data: bytes) -> int:
        """
        Standard register-based CRC computation on raw bytes.

        Handles reflection, initial value, and final XOR according to
        the polynomial specification.  Useful for verifying against
        reference implementations (e.g. ``zlib.crc32``).

        Parameters
        ----------
        data : bytes
            Raw data bytes.

        Returns
        -------
        int
            CRC checksum as an integer.
        """
        p = self.poly
        crc = p.init

        for byte in data:
            if p.reflect_in:
                byte = self._reflect_bits(byte, 8)

            crc ^= byte << (p.width - 8)

            for _ in range(8):
                if crc & p.msb_mask:
                    crc = ((crc << 1) ^ p.poly) & p.mask
                else:
                    crc = (crc << 1) & p.mask

        if p.reflect_out:
            crc = self._reflect_bits(crc, p.width)

        return crc ^ p.xor_out

    # ──────────────────── Private Helpers ────────────────────────────

    @staticmethod
    def _modulo2_division(
        dividend: List[int], divisor: List[int]
    ) -> List[int]:
        """
        Bitwise XOR-based modulo-2 polynomial long division.

        Parameters
        ----------
        dividend : List[int]
            The augmented message (M(x) · x^r) as a bit list.
        divisor : List[int]
            The generator polynomial G(x) as a bit list (length = r+1).

        Returns
        -------
        List[int]
            Remainder bits of length r (= len(divisor) - 1).

        Notes
        -----
        In GF(2) arithmetic there are no carries or borrows;
        both addition and subtraction are XOR.
        """
        r = len(divisor) - 1  # degree of G(x)

        # Work on a mutable copy
        working = list(dividend)

        # Slide the divisor across the dividend
        for i in range(len(working) - r):
            if working[i] == 1:  # leading bit is 1 → perform XOR
                for j in range(len(divisor)):
                    working[i + j] ^= divisor[j]
            # If leading bit is 0, no operation (shift only)

        # The last r bits are the remainder R(x)
        remainder = working[-r:]
        return remainder

    @staticmethod
    def _reflect_bits(value: int, width: int) -> int:
        """
        Reverse the bit order of ``value`` within ``width`` bits.

        Used for CRC standards that specify input/output reflection
        (e.g. CRC-32 ISO / Ethernet).
        """
        reflected = 0
        for i in range(width):
            if value & (1 << i):
                reflected |= 1 << (width - 1 - i)
        return reflected


# ═══════════════════════ Convenience Function ════════════════════════

def encode_frame(
    message_bits: List[int], polynomial: CRCPolynomial
) -> Tuple[List[int], List[int]]:
    """
    Convenience wrapper: encode a message and return (codeword, crc_bits).
    """
    enc = CRCEncoder(polynomial)
    crc_bits = enc.compute_crc_bits(message_bits)
    codeword = message_bits + crc_bits
    return codeword, crc_bits
