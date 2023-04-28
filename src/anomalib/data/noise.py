from enum import Enum


class NoiseType(str, Enum):
    """Supported Noise Types"""

    PERLIN_2D = "perlin_2d"
    SIMPLEX_2D = "simplex_2d"
