from __future__ import annotations

import numpy as np


TWO_PI = float(2.0 * np.pi)


def normalize_rx_to_2pi(values: np.ndarray) -> np.ndarray:
    """Return a copy with Euler rx mapped from [-pi, pi] style values to [0, 2pi)."""
    result = np.asarray(values).copy()
    result[..., 3] = np.where(result[..., 3] < 0.0, result[..., 3] + TWO_PI, result[..., 3])
    return result
