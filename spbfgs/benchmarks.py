import numpy as np


def rosen(x: np.ndarray) -> float:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2
