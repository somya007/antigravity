"""
==============================================================================
  Dataset Generator for ML Model Training
==============================================================================
  Generates simulation frames with varying parameters and labels each frame
  as "detected" or "undetected" (aliasing) based on the CRC check result.

  KEY INSIGHT: CRC detects virtually all random errors. Aliasing (undetected
  errors) occurs ONLY when the error polynomial E(x) is divisible by the
  generator polynomial G(x). The probability is ~2^(-r) for random errors
  of weight > r, which is extremely rare for CRC-16 (~1 in 65,536).

  To produce a meaningful ML training set with both classes, this module:
    1. Generates normal simulation frames (mostly "detected")
    2. Engineers aliasing frames by constructing E(x) = Q(x) * G(x),
       where Q(x) is a random polynomial — these errors bypass CRC
    3. Mixes both at a configurable ratio (~12-15% aliasing)

  Feature Set:
    - frame_length          : Number of data bits in the frame
    - polynomial_type       : CRC polynomial used (categorical -> encoded)
    - polynomial_degree     : Degree r of the generator polynomial
    - noise_type            : Type of injected noise (categorical -> encoded)
    - ber                   : Bit Error Rate of the channel
    - burst_length          : Length of burst error (0 if not burst)
    - num_errors_injected   : Actual number of bits flipped
    - error_density         : num_errors / frame_length
    - error_to_poly_ratio   : burst_length / polynomial_degree

  Target:
    - detected : 1 if CRC caught the error, 0 if undetected (aliasing)
==============================================================================
"""

import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.polynomials import get_polynomial, POLYNOMIAL_REGISTRY
from core.crc_encoder import CRCEncoder
from core.crc_decoder import CRCDecoder
from core.data_framing import DataFramer
from channel.noise_injector import NoiseInjector
import config


# ─────────────────────────────────────────────────────────────────────
#  Aliasing (Undetected Error) Generator
# ─────────────────────────────────────────────────────────────────────

def _poly_multiply_gf2(a: List[int], b: List[int]) -> List[int]:
    """
    Multiply two polynomials in GF(2) (coefficients mod 2).
    Input: bit lists [MSB..LSB], e.g. x^2 + 1 = [1, 0, 1].
    """
    if not a or not b:
        return [0]
    result_len = len(a) + len(b) - 1
    result = [0] * result_len
    for i, ai in enumerate(a):
        if ai == 1:
            for j, bj in enumerate(b):
                if bj == 1:
                    result[i + j] ^= 1
    return result


def _generate_aliasing_error(
    codeword_length: int,
    poly_bits: List[int],
    rng: random.Random,
) -> List[int]:
    """
    Construct an error pattern E(x) = Q(x) * G(x) that is guaranteed
    to be divisible by G(x), so CRC check will pass (aliasing).

    The quotient Q(x) is randomly chosen such that:
      - deg(E(x)) < codeword_length
      - E(x) != 0 (at least one bit flipped)
    """
    r = len(poly_bits) - 1  # degree of G(x)
    # Max degree of Q(x) so that E(x) fits in the codeword
    max_q_degree = codeword_length - 1 - r
    if max_q_degree < 0:
        return [0] * codeword_length  # Can't create aliasing error

    # Choose a random quotient degree (at least 0)
    q_degree = rng.randint(0, min(max_q_degree, 20))  # cap for speed

    # Generate random Q(x) of the chosen degree
    q_bits = [1]  # Leading coefficient = 1
    for _ in range(q_degree):
        q_bits.append(rng.randint(0, 1))

    # E(x) = Q(x) * G(x)
    e_bits = _poly_multiply_gf2(q_bits, poly_bits)

    # Pad E(x) to codeword_length (leading zeros)
    error_pattern = [0] * codeword_length
    start = codeword_length - len(e_bits)
    for i, bit in enumerate(e_bits):
        if start + i >= 0:
            error_pattern[start + i] = bit

    # Ensure at least one error
    if sum(error_pattern) == 0:
        return [0] * codeword_length  # degenerate case

    return error_pattern


def generate_dataset(
    num_samples: int = config.NUM_SIMULATIONS,
    seed: int = config.RANDOM_SEED,
    save_path: Optional[str] = None,
    aliasing_ratio: float = 0.12,
) -> pd.DataFrame:
    """
    Generate the full simulation dataset with both detected and
    undetected (aliasing) error samples.

    Strategy:
      - (1 - aliasing_ratio) of samples: normal noise injection
        (CRC almost always detects these -> detected=1)
      - aliasing_ratio of samples: engineered E(x) = Q(x)*G(x)
        (CRC provably cannot detect these -> detected=0)

    Parameters
    ----------
    num_samples : int
        Total number of frames to simulate.
    seed : int
        Random seed for reproducibility.
    save_path : str or None
        If provided, save the dataset to this CSV path.
    aliasing_ratio : float
        Fraction of samples that should be engineered aliasing cases.

    Returns
    -------
    pd.DataFrame
        The generated dataset with features and target columns.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    encoder_cache = {}
    decoder_cache = {}
    poly_bits_cache = {}
    noise = NoiseInjector(seed=seed)

    # Pre-define parameter pools
    poly_names = list(POLYNOMIAL_REGISTRY.keys())
    frame_sizes = config.FRAME_SIZES
    noise_types = ["single_bit", "multi_bit", "burst", "random"]
    ber_values = config.BER_VALUES
    burst_lengths = config.BURST_LENGTHS

    num_aliasing = int(num_samples * aliasing_ratio)
    num_normal = num_samples - num_aliasing

    records = []

    # ─────────────────────────────────────────────────────────────────
    # Part A: Normal noise simulation (detected errors)
    # ─────────────────────────────────────────────────────────────────
    print(f"Generating {num_samples:,} simulation frames "
          f"({num_normal:,} normal + {num_aliasing:,} aliasing)...\n")

    for i in tqdm(range(num_normal), desc="Normal frames", ncols=80):
        frame_length = rng.choice(frame_sizes)
        poly_name = rng.choice(poly_names)
        noise_type = rng.choice(noise_types)
        ber = rng.choice(ber_values)
        burst_len = rng.choice(burst_lengths) if noise_type == "burst" else 0
        num_multi_errors = rng.randint(2, min(10, frame_length // 4))

        poly = get_polynomial(poly_name)

        if poly_name not in encoder_cache:
            encoder_cache[poly_name] = CRCEncoder(poly)
            decoder_cache[poly_name] = CRCDecoder(poly)
            poly_bits_cache[poly_name] = poly.poly_as_bit_list()
        encoder = encoder_cache[poly_name]
        decoder = decoder_cache[poly_name]

        message_bits = [rng.randint(0, 1) for _ in range(frame_length)]
        codeword = encoder.encode(message_bits)

        corrupted, error_pattern = noise.inject_error(
            codeword,
            noise_type=noise_type,
            ber=ber,
            burst_length=burst_len,
            num_errors=num_multi_errors,
        )

        num_errors_actual = sum(error_pattern)
        actual_burst_len = NoiseInjector.get_burst_length(error_pattern)

        if num_errors_actual == 0:
            continue  # Skip no-error frames

        crc_passed = decoder.verify(corrupted)
        detected = 0 if crc_passed else 1

        codeword_length = len(codeword)
        error_density = num_errors_actual / codeword_length
        error_to_poly_ratio = actual_burst_len / poly.degree

        records.append({
            "frame_length": frame_length,
            "codeword_length": codeword_length,
            "polynomial_type": poly_name,
            "polynomial_degree": poly.degree,
            "noise_type": noise_type,
            "ber": ber,
            "burst_length_requested": burst_len,
            "burst_length_actual": actual_burst_len,
            "num_errors_injected": num_errors_actual,
            "error_density": round(error_density, 6),
            "error_to_poly_ratio": round(error_to_poly_ratio, 4),
            "detected": detected,
        })

    # ─────────────────────────────────────────────────────────────────
    # Part B: Engineered aliasing frames (undetected errors)
    # ─────────────────────────────────────────────────────────────────
    # These use E(x) = Q(x) * G(x), which provably bypasses CRC.
    # This models the theoretical aliasing phenomenon where the error
    # polynomial is a multiple of the generator polynomial.
    # ─────────────────────────────────────────────────────────────────
    aliasing_noise_labels = ["burst", "multi_bit", "random", "structured"]

    for i in tqdm(range(num_aliasing), desc="Aliasing frames", ncols=80):
        frame_length = rng.choice(frame_sizes)
        poly_name = rng.choice(poly_names)
        noise_label = rng.choice(aliasing_noise_labels)
        ber = rng.choice(ber_values)

        poly = get_polynomial(poly_name)

        if poly_name not in encoder_cache:
            encoder_cache[poly_name] = CRCEncoder(poly)
            decoder_cache[poly_name] = CRCDecoder(poly)
            poly_bits_cache[poly_name] = poly.poly_as_bit_list()
        encoder = encoder_cache[poly_name]
        decoder = decoder_cache[poly_name]

        message_bits = [rng.randint(0, 1) for _ in range(frame_length)]
        codeword = encoder.encode(message_bits)
        codeword_length = len(codeword)

        # Generate aliasing error pattern
        error_pattern = _generate_aliasing_error(
            codeword_length, poly_bits_cache[poly_name], rng
        )

        if sum(error_pattern) == 0:
            continue  # degenerate — skip

        # Apply error
        corrupted = [b ^ e for b, e in zip(codeword, error_pattern)]

        # Verify it's truly undetected (sanity check)
        crc_passed = decoder.verify(corrupted)
        if not crc_passed:
            continue  # shouldn't happen, but guard

        num_errors_actual = sum(error_pattern)
        actual_burst_len = NoiseInjector.get_burst_length(error_pattern)
        error_density = num_errors_actual / codeword_length
        error_to_poly_ratio = actual_burst_len / poly.degree
        burst_len_req = actual_burst_len if noise_label == "burst" else 0

        records.append({
            "frame_length": frame_length,
            "codeword_length": codeword_length,
            "polynomial_type": poly_name,
            "polynomial_degree": poly.degree,
            "noise_type": noise_label,
            "ber": ber,
            "burst_length_requested": burst_len_req,
            "burst_length_actual": actual_burst_len,
            "num_errors_injected": num_errors_actual,
            "error_density": round(error_density, 6),
            "error_to_poly_ratio": round(error_to_poly_ratio, 4),
            "detected": 0,  # Aliasing — undetected by design
        })

    # ── Build DataFrame & shuffle ──
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    det_count = df["detected"].sum()
    undet_count = (df["detected"] == 0).sum()
    total = len(df)

    print(f"\nDataset generated: {total:,} total frames")
    print(f"  Errors detected (CRC):  {det_count:,} ({det_count/total*100:.2f}%)")
    print(f"  Errors undetected:      {undet_count:,} ({undet_count/total*100:.2f}%)")
    print(f"  Detection rate:         {det_count/total:.4f}")
    print(f"\n  Per noise type:")
    for nt in sorted(df["noise_type"].unique()):
        sub = df[df["noise_type"] == nt]
        print(f"    {nt:15s}: {len(sub):>6,} frames, "
              f"detection rate = {sub['detected'].mean()*100:.2f}%")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n  Saved to: {save_path}")

    return df


if __name__ == "__main__":
    output_path = os.path.join(config.DATASETS_DIR, "crc_simulation_dataset.csv")
    df = generate_dataset(save_path=output_path)
    print(f"\nDataset shape: {df.shape}")
    print(df.describe())
