"""
Unit tests for CRC Encoder/Decoder.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zlib
from core.polynomials import get_polynomial, CRC_16_CCITT, CRC_32
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from core.data_framing import DataFramer


def test_encode_decode_roundtrip():
    """Encoding then decoding should always give zero remainder."""
    for poly_name in ["CRC-16-CCITT", "CRC-16-IBM", "CRC-32"]:
        poly = get_polynomial(poly_name)
        enc = CRCEncoder(poly)
        dec = CRCDecoder(poly)
        framer = DataFramer(64, seed=42)

        for _ in range(100):
            msg = framer.generate_random_frame()
            codeword = enc.encode(msg)
            assert dec.verify(codeword), f"Roundtrip failed for {poly_name}"
    print("  ✓ test_encode_decode_roundtrip passed")


def test_single_bit_error_detected():
    """Any single-bit error should always be detected."""
    poly = get_polynomial("CRC-16-CCITT")
    enc = CRCEncoder(poly)
    dec = CRCDecoder(poly)

    msg = [1, 0, 1, 1, 0, 0, 1, 0] * 8  # 64-bit message
    codeword = enc.encode(msg)

    for pos in range(len(codeword)):
        corrupted = list(codeword)
        corrupted[pos] ^= 1
        assert not dec.verify(corrupted), f"Single-bit error at pos {pos} not detected"
    print("  ✓ test_single_bit_error_detected passed")


def test_crc32_matches_zlib():
    """Register-based CRC-32 should match zlib.crc32()."""
    poly = get_polynomial("CRC-32")
    enc = CRCEncoder(poly)

    test_data = b"Hello, World!"
    our_crc = enc.compute_crc_register(test_data)
    zlib_crc = zlib.crc32(test_data) & 0xFFFFFFFF

    assert our_crc == zlib_crc, f"CRC-32 mismatch: {our_crc:#010x} vs {zlib_crc:#010x}"
    print(f"  ✓ test_crc32_matches_zlib passed (CRC = {our_crc:#010x})")


def test_syndrome_nonzero_on_error():
    """Syndrome should be non-zero when errors are present."""
    poly = get_polynomial("CRC-16-CCITT")
    enc = CRCEncoder(poly)
    dec = CRCDecoder(poly)

    msg = [1] * 32
    codeword = enc.encode(msg)

    corrupted = list(codeword)
    corrupted[5] ^= 1
    corrupted[10] ^= 1

    _, is_valid, syndrome = dec.decode_and_verify(corrupted)
    assert not is_valid
    assert any(b != 0 for b in syndrome)
    print("  ✓ test_syndrome_nonzero_on_error passed")


def test_data_framing():
    """Test frame generation and conversion."""
    framer = DataFramer(128, seed=42)
    frame = framer.generate_random_frame()
    assert len(frame) == 128
    assert all(b in (0, 1) for b in frame)

    # bytes ↔ bits roundtrip
    data = b"Test123"
    bits = DataFramer.bytes_to_bits(data)
    recovered = DataFramer.bits_to_bytes(bits)
    assert recovered == data
    print("  ✓ test_data_framing passed")


if __name__ == "__main__":
    print("\nRunning CRC Tests...")
    test_encode_decode_roundtrip()
    test_single_bit_error_detected()
    test_crc32_matches_zlib()
    test_syndrome_nonzero_on_error()
    test_data_framing()
    print("\nAll CRC tests passed! ✓")
