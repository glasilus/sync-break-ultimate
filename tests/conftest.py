import pytest
import numpy as np

@pytest.fixture
def blank_frame():
    """480x270 RGB frame, all black."""
    return np.zeros((270, 480, 3), dtype=np.uint8)

@pytest.fixture
def noise_frame():
    """480x270 RGB frame, random noise."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (270, 480, 3), dtype=np.uint8)

@pytest.fixture
def white_frame():
    """480x270 RGB frame, all white."""
    return np.full((270, 480, 3), 255, dtype=np.uint8)
