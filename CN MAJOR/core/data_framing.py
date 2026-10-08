"""
==============================================================================
  Data Framing Module
==============================================================================
  Divides raw binary data into fixed-size frames for CRC processing.
  Supports:
    - Random binary data generation for simulation
    - Conversion between bytes and bit lists
    - Padding and segmentation
==============================================================================
"""

import random
from typing import List, Optional


class DataFramer:
    """
    Handles creation and manipulation of binary data frames.

    Parameters
    ----------
    frame_size : int
        Number of data bits per frame (before CRC).
    seed : int or None
        Random seed for reproducible data generation.
    """

    def __init__(self, frame_size: int = 128, seed: Optional[int] = None):
        if frame_size < 8:
            raise ValueError("Frame size must be at least 8 bits.")
        self.frame_size = frame_size
        self.rng = random.Random(seed)

    # ──────────────────── Data Generation ────────────────────────────

    def generate_random_frame(self) -> List[int]:
        """
        Generate a single random binary frame of ``frame_size`` bits.

        Returns
        -------
        List[int]
            A list of 0s and 1s of length ``frame_size``.
        """
        return [self.rng.randint(0, 1) for _ in range(self.frame_size)]

    def generate_frames(self, count: int) -> List[List[int]]:
        """
        Generate multiple random frames.

        Parameters
        ----------
        count : int
            Number of frames to generate.

        Returns
        -------
        List[List[int]]
            A list of ``count`` frames.
        """
        return [self.generate_random_frame() for _ in range(count)]

    # ──────────────────── Conversion Utilities ───────────────────────

    @staticmethod
    def bytes_to_bits(data: bytes) -> List[int]:
        """
        Convert a bytes object to a list of bits (MSB first per byte).

        >>> DataFramer.bytes_to_bits(b'\\xA5')
        [1, 0, 1, 0, 0, 1, 0, 1]
        """
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def bits_to_bytes(bits: List[int]) -> bytes:
        """
        Convert a list of bits to bytes (MSB first per byte).
        Pads with zeros on the right if length is not a multiple of 8.

        >>> DataFramer.bits_to_bytes([1, 0, 1, 0, 0, 1, 0, 1])
        b'\\xa5'
        """
        # Pad to multiple of 8
        padded = bits + [0] * ((8 - len(bits) % 8) % 8)
        result = bytearray()
        for i in range(0, len(padded), 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | padded[i + j]
            result.append(byte_val)
        return bytes(result)

    @staticmethod
    def bits_to_int(bits: List[int]) -> int:
        """Convert a bit list to an integer (MSB first)."""
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return value

    @staticmethod
    def int_to_bits(value: int, width: int) -> List[int]:
        """Convert an integer to a bit list of fixed width (MSB first)."""
        return [(value >> (width - 1 - i)) & 1 for i in range(width)]

    # ──────────────────── Segmentation ───────────────────────────────

    def segment_data(self, data_bits: List[int]) -> List[List[int]]:
        """
        Divide a long bit sequence into frames of ``frame_size``.

        The final frame is zero-padded if necessary.

        Parameters
        ----------
        data_bits : List[int]
            Arbitrary-length bit sequence.

        Returns
        -------
        List[List[int]]
            List of fixed-size frames.
        """
        frames = []
        for start in range(0, len(data_bits), self.frame_size):
            frame = data_bits[start : start + self.frame_size]
            # Pad the last frame if needed
            if len(frame) < self.frame_size:
                frame = frame + [0] * (self.frame_size - len(frame))
            frames.append(frame)
        return frames
