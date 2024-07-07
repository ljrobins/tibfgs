import taichi as ti

@ti.func
def rosen(x: ti.math.vec2) -> ti.float32:
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2
