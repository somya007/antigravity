"""
==============================================================================
  Checksum vs. CRC Comparison
==============================================================================
  Implements the Internet Checksum (one's complement sum) and compares
  its error detection capability against CRC methods.

  Internet Checksum (RFC 1071):
    - Split data into 16-bit words
    - Sum all words using one's complement arithmetic
    - Take the one's complement of the result
    - Detects all single-bit errors but misses many multi-bit/burst errors
==============================================================================
"""

import random
from typing import List, Tuple, Dict

import pandas as pd
from tqdm import tqdm

from core.polynomials import CRCPolynomial, get_polynomial
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from core.data_framing import DataFramer
from channel.noise_injector import NoiseInjector


class InternetChecksum:
    """
    Standard Internet Checksum (RFC 1071) — one's complement sum.
    """

    @staticmethod
    def compute(data_bits: List[int]) -> List[int]:
        """
        Compute 16-bit Internet checksum over the given data bits.

        Parameters
        ----------
        data_bits : List[int]
            Binary data as a list of 0/1.

        Returns
        -------
        List[int]
            16-bit checksum as a list of 0/1.
        """
        # Pad to multiple of 16
        padded = list(data_bits)
        while len(padded) % 16 != 0:
            padded.append(0)

        # Sum 16-bit words
        total = 0
        for i in range(0, len(padded), 16):
            word = 0
            for j in range(16):
                word = (word << 1) | padded[i + j]
            total += word
            # Fold carry bits
            while total > 0xFFFF:
                total = (total & 0xFFFF) + (total >> 16)

        # One's complement
        checksum = total ^ 0xFFFF

        # Convert to 16-bit list
        return [(checksum >> (15 - i)) & 1 for i in range(16)]

    @staticmethod
    def verify(data_with_checksum: List[int]) -> bool:
        """
        Verify data integrity using the appended checksum.

        If the one's complement sum of all 16-bit words (including checksum)
        yields 0xFFFF (all ones), the data is valid.
        """
        padded = list(data_with_checksum)
        while len(padded) % 16 != 0:
            padded.append(0)

        total = 0
        for i in range(0, len(padded), 16):
            word = 0
            for j in range(16):
                word = (word << 1) | padded[i + j]
            total += word
            while total > 0xFFFF:
                total = (total & 0xFFFF) + (total >> 16)

        return total == 0xFFFF

    @staticmethod
    def encode(data_bits: List[int]) -> List[int]:
        """Append the checksum to the data."""
        checksum = InternetChecksum.compute(data_bits)
        return data_bits + checksum


def compare_crc_vs_checksum(
    num_trials: int = 20_000,
    frame_sizes: List[int] = None,
    polynomial_name: str = "CRC-16-CCITT",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a head-to-head comparison of CRC vs. Internet Checksum.

    Parameters
    ----------
    num_trials : int
        Number of test frames per noise type.
    frame_sizes : list of int
        Frame sizes to test.
    polynomial_name : str
        CRC polynomial for comparison.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Comparison results with detection rates for each method.
    """
    if frame_sizes is None:
        frame_sizes = [128, 256, 512]

    rng = random.Random(seed)
    noise = NoiseInjector(seed=seed)
    poly = get_polynomial(polynomial_name)
    encoder = CRCEncoder(poly)
    decoder = CRCDecoder(poly)
    cksum = InternetChecksum()

    noise_configs = [
        ("single_bit", {"noise_type": "single_bit"}),
        ("multi_bit_3", {"noise_type": "multi_bit", "num_errors": 3}),
        ("multi_bit_5", {"noise_type": "multi_bit", "num_errors": 5}),
        ("burst_8", {"noise_type": "burst", "burst_length": 8}),
        ("burst_16", {"noise_type": "burst", "burst_length": 16}),
        ("burst_32", {"noise_type": "burst", "burst_length": 32}),
        ("random_0.01", {"noise_type": "random", "ber": 0.01}),
        ("random_0.05", {"noise_type": "random", "ber": 0.05}),
    ]

    records = []
    trials_per_config = num_trials // (len(frame_sizes) * len(noise_configs))
    trials_per_config = max(trials_per_config, 100)

    print(f"Comparing CRC ({polynomial_name}) vs. Internet Checksum")
    print(f"  {trials_per_config} trials per config\n")

    for fs in frame_sizes:
        for noise_name, noise_params in tqdm(
            noise_configs, desc=f"Frame size {fs}"
        ):
            crc_detected = 0
            cksum_detected = 0
            total_with_errors = 0

            for _ in range(trials_per_config):
                # Generate random frame
                msg = [rng.randint(0, 1) for _ in range(fs)]

                # CRC encode
                crc_codeword = encoder.encode(msg)

                # Checksum encode
                cksum_codeword = cksum.encode(msg)

                # Inject same error pattern to both
                corrupted_crc, err_crc = noise.inject_error(
                    crc_codeword, **noise_params
                )
                # Apply same error type to checksum codeword
                corrupted_cksum, err_cksum = noise.inject_error(
                    cksum_codeword, **noise_params
                )

                # Only count if errors were actually injected
                if sum(err_crc) > 0:
                    total_with_errors += 1
                    if not decoder.verify(corrupted_crc):
                        crc_detected += 1
                    if not cksum.verify(corrupted_cksum):
                        cksum_detected += 1

            crc_rate = crc_detected / total_with_errors if total_with_errors > 0 else 0
            cksum_rate = cksum_detected / total_with_errors if total_with_errors > 0 else 0

            records.append({
                "frame_size": fs,
                "noise_type": noise_name,
                "total_errors": total_with_errors,
                "crc_detected": crc_detected,
                "crc_detection_rate": round(crc_rate, 6),
                "checksum_detected": cksum_detected,
                "checksum_detection_rate": round(cksum_rate, 6),
                "crc_advantage": round(crc_rate - cksum_rate, 6),
            })

    df = pd.DataFrame(records)
    print("\n" + df.to_string(index=False))
    return df
