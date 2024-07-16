import taichi as ti
import numpy as np


@ti.func
def rosen(x: ti.math.vec2) -> ti.float32:
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


@ti.func
def ackley(x: ti.math.vec2) -> ti.f32:
    return (
        -20 * ti.exp(-0.2 * ti.sqrt(0.5 * x.norm_sqr()))
        - ti.exp(
            0.5 * ti.cos(2 * ti.math.pi * x.x) + 0.5 * ti.cos(2 * ti.math.pi * x.y)
        )
        + ti.math.e
        + 20
    )


def ackley_np(x: np.ndarray) -> np.ndarray:
    if x.size == 2:
        x = x.reshape(2,1,1)
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * np.linalg.norm(x, axis=0, keepdims=True)))
        - np.exp(0.5 * np.cos(2 * np.pi * x[0,:,:]) + 0.5 * np.cos(2 * np.pi * x[1,:,:]))
        + np.e
        + 20
    ).squeeze()
