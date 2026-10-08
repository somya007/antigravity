"""
==============================================================================
  CRC Polynomial Definitions
==============================================================================
  Mathematical Foundation:
    A CRC polynomial G(x) of degree r can detect:
      - All single-bit errors
      - All double-bit errors (if G(x) has ≥ 3 terms)
      - All odd-number-of-bit errors (if (x+1) is a factor of G(x))
      - All burst errors of length ≤ r
      - A fraction (1 - 2^{-r}) of burst errors of length > r

  Standard polynomials are stored as integer representations where
  bit i corresponds to the coefficient of x^i.
==============================================================================
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CRCPolynomial:
    """
    Immutable specification of a standard CRC polynomial.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g., "CRC-32").
    poly : int
        Generator polynomial in normal (MSB-first) representation,
        *excluding* the implicit leading 1.  For CRC-32 this is 0x04C11DB7.
    width : int
        Degree of the polynomial (number of CRC check bits).
    init : int
        Initial value of the shift register.
    xor_out : int
        Value XORed with the final CRC before output.
    reflect_in : bool
        If True, each input byte is bit-reflected before processing.
    reflect_out : bool
        If True, the final CRC value is bit-reflected before XOR-out.
    description : str
        Brief note on where this polynomial is commonly used.
    """
    name: str
    poly: int
    width: int
    init: int = 0
    xor_out: int = 0
    reflect_in: bool = False
    reflect_out: bool = False
    description: str = ""

    # ── derived helpers ──────────────────────────────────────────────
    @property
    def degree(self) -> int:
        """Degree r of G(x) — equal to width."""
        return self.width

    @property
    def mask(self) -> int:
        """Bit mask for the CRC register (2^width - 1)."""
        return (1 << self.width) - 1

    @property
    def msb_mask(self) -> int:
        """Mask for the MSB of the CRC register."""
        return 1 << (self.width - 1)

    @property
    def poly_with_leading_one(self) -> int:
        """
        Full polynomial with the implicit leading x^r term.
        Used in bit-level modulo-2 division.
        """
        return self.poly | (1 << self.width)

    def poly_as_bit_list(self) -> List[int]:
        """
        Return the full generator polynomial (including leading 1)
        as a list of bits [MSB, ..., LSB].  Length = width + 1.
        """
        full = self.poly_with_leading_one
        return [(full >> (self.width - i)) & 1 for i in range(self.width + 1)]

    def __str__(self) -> str:
        return (
            f"{self.name}  "
            f"poly=0x{self.poly:0{self.width // 4}X}  "
            f"width={self.width}  "
            f"init=0x{self.init:0{self.width // 4}X}  "
            f"xor_out=0x{self.xor_out:0{self.width // 4}X}  "
            f"reflect_in={self.reflect_in}  reflect_out={self.reflect_out}"
        )


# ═══════════════════════ Standard Polynomials ════════════════════════

CRC_16_CCITT = CRCPolynomial(
    name="CRC-16-CCITT",
    poly=0x1021,
    width=16,
    init=0xFFFF,
    xor_out=0x0000,
    reflect_in=False,
    reflect_out=False,
    description="ITU-T V.41; used in HDLC, X.25, Bluetooth.",
)

CRC_16_IBM = CRCPolynomial(
    name="CRC-16-IBM",
    poly=0x8005,
    width=16,
    init=0x0000,
    xor_out=0x0000,
    reflect_in=True,
    reflect_out=True,
    description="USB, Modbus; a.k.a. CRC-16/ARC.",
)

CRC_32 = CRCPolynomial(
    name="CRC-32",
    poly=0x04C11DB7,
    width=32,
    init=0xFFFFFFFF,
    xor_out=0xFFFFFFFF,
    reflect_in=True,
    reflect_out=True,
    description="Ethernet, PKZIP, PNG, MPEG-2 (ISO 3309).",
)

# Convenience lookup by name
POLYNOMIAL_REGISTRY = {
    "CRC-16-CCITT": CRC_16_CCITT,
    "CRC-16-IBM": CRC_16_IBM,
    "CRC-32": CRC_32,
}


def get_polynomial(name: str) -> CRCPolynomial:
    """
    Retrieve a polynomial specification by its canonical name.

    Parameters
    ----------
    name : str
        One of the keys in ``POLYNOMIAL_REGISTRY``.

    Returns
    -------
    CRCPolynomial

    Raises
    ------
    ValueError
        If the name is not found.
    """
    if name not in POLYNOMIAL_REGISTRY:
        available = ", ".join(POLYNOMIAL_REGISTRY.keys())
        raise ValueError(
            f"Unknown polynomial '{name}'. Available: {available}"
        )
    return POLYNOMIAL_REGISTRY[name]


def list_polynomials() -> List[CRCPolynomial]:
    """Return all registered CRC polynomials."""
    return list(POLYNOMIAL_REGISTRY.values())
