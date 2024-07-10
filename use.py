import matplotlib.pyplot as plt
import os

os.environ['TI_DIM_X'] = str(2)
os.environ['TI_NUM_PARTICLES'] = str(int(1e5))

import taichi as ti

ti.init(
    arch=ti.metal,
    default_fp=ti.float32,
    fast_math=True,
    advanced_optimization=True,
    num_compile_threads=32,
    opt_level=3,
)

from tibfgs import (
    NPART,
    N,
    minimize_bfgs,
    ackley,
    ackley_np,
    set_f,
    rosen,
    VTYPE,
    MTYPE,
)
import time
import numpy as np

set_f(ackley)

res = ti.types.struct(
    fun=ti.f32, jac=VTYPE, hess_inv=MTYPE, status=ti.u8, xk=VTYPE, k=ti.u32
)

res_field = ti.Struct.field(
    dict(fun=ti.f32, jac=VTYPE, hess_inv=MTYPE, status=ti.u8, xk=VTYPE, k=ti.u32),
    shape=(NPART,),
)


@ti.kernel
def run() -> int:
    for i in range(NPART):
        x0 = ti.math.vec2([4 * ti.random() - 2, 4 * ti.random() - 2])
        fval, gfk, Hk, warnflag, xk, k = minimize_bfgs(i=i, x0=x0, gtol=1e-3, eps=1e-6)
        res_field[i] = res(fun=fval, jac=gfk, hess_inv=Hk, status=warnflag, xk=xk, k=k)
    return 0


t1 = time.time()
run()
res_dict = res_field.to_numpy()
print(1 / ((time.time() - t1) / 1e6))

status = res_field.status.to_numpy()

xx, yy = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
x = np.vstack((xx.flatten(), yy.flatten())).T
av = ackley_np(x).reshape(xx.shape)


plt.figure(figsize=(4, 6))
plt.subplot(2, 1, 1)
plt.pcolor(xx, yy, av)
plt.scatter(res_dict['xk'][:, 0], res_dict['xk'][:, 1], s=5, c='m', alpha=0.1)
plt.title('Converged particles, Ackley function')
plt.subplot(2, 1, 2)
plt.scatter(res_dict['xk'][:, 0], res_dict['xk'][:, 1], s=5, c='m', alpha=0.1)
plt.title('Magnified view of origin')
e = 1e-5
plt.xlim(-e, e)
plt.ylim(-e, e)
plt.tight_layout()
plt.show()
