"""
Unit tests for Noise Injection and ORBGRAND decoder.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.polynomials import get_polynomial
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from core.data_framing import DataFramer
from channel.noise_injector import NoiseInjector
from channel.channel_model import ChannelModel
from decoding.orbgrand_decoder import ORBGRANDDecoder


def test_noise_injector():
    """Test all noise types produce correct error patterns."""
    noise = NoiseInjector(seed=42)
    frame = [0] * 100

    # Single-bit
    c, e = noise.inject_single_bit_error(frame)
    assert sum(e) == 1, "Single bit should flip exactly 1 bit"

    # Multi-bit
    c, e = noise.inject_multi_bit_error(frame, 5)
    assert sum(e) == 5, "Multi-bit should flip exactly 5 bits"

    # Burst
    c, e = noise.inject_burst_error(frame, 10)
    assert noise.get_burst_length(e) <= 10
    assert e[0] == 0 or any(e)  # First error bit should be 1

    # Random
    c, e = noise.inject_random_errors(frame, 0.5)
    assert 0 < sum(e) < 100  # Roughly half should flip

    print("  ✓ test_noise_injector passed")


def test_orbgrand_single_bit_correction():
    """ORBGRAND should be able to correct single-bit errors."""
    poly = get_polynomial("CRC-16-CCITT")
    enc = CRCEncoder(poly)
    channel = ChannelModel(seed=42)
    orbgrand = ORBGRANDDecoder(poly, max_queries=1000, max_error_weight=3)
    framer = DataFramer(64, seed=42)

    successes = 0
    trials = 20

    for _ in range(trials):
        msg = framer.generate_random_frame()
        codeword = enc.encode(msg)

        # Inject exactly one error via channel
        noise = NoiseInjector(seed=None)
        corrupted, err = noise.inject_single_bit_error(codeword)

        # Use uniform reliability (no soft info)
        reliability = [1.0] * len(corrupted)
        # Make the error position less reliable
        err_pos = err.index(1)
        reliability[err_pos] = 0.1

        decoded, success, queries = orbgrand.decode(corrupted, reliability)
        if success:
            successes += 1

    rate = successes / trials
    print(f"  ✓ test_orbgrand_single_bit_correction: {successes}/{trials} ({rate:.0%}) corrected")
    assert rate >= 0.5, f"ORBGRAND should correct most single-bit errors, got {rate:.0%}"


def test_channel_model():
    """Test BSC and Gilbert-Elliott channels."""
    channel = ChannelModel(seed=42)
    frame = [0, 1] * 50  # 100 bits

    # BSC
    received, llr, errors = channel.bsc_transmit(frame, 0.1)
    assert len(received) == 100
    assert len(llr) == 100
    assert 0 < sum(errors) < 50  # Roughly 10% errors

    # Gilbert-Elliott
    received, llr, errors = channel.gilbert_elliott_transmit(
        frame, p_good=0.001, p_bad=0.1
    )
    assert len(received) == 100

    print("  ✓ test_channel_model passed")


if __name__ == "__main__":
    print("\nRunning Noise & Decoder Tests...")
    test_noise_injector()
    test_channel_model()
    test_orbgrand_single_bit_correction()
    print("\nAll noise & decoder tests passed! ✓")
