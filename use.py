import matplotlib.pyplot as plt
import os
os.environ['TI_DIM_X'] = str(2)
os.environ['TI_NUM_PARTICLES'] = str(int(1e5))

import taichi as ti
from tibfgs import NPART, N, minimize_bfgs, ackley, ackley_np, set_f, rosen, VTYPE, MTYPE
import time
import numpy as np

set_f(ackley)

res = ti.types.struct(
    fun=ti.f32,
    jac=VTYPE,
    hess_inv=MTYPE,
    status=ti.u8,
    xk=VTYPE,
    k=ti.u32
)

res_field = ti.Struct.field(
    dict(fun=ti.f32,
    jac=VTYPE,
    hess_inv=MTYPE,
    status=ti.u8,
    xk=VTYPE,
    k=ti.u32),
    shape=(NPART,)
)

@ti.kernel
def run() -> int:
    for i in range(NPART):
        x0 = ti.math.vec2([6*ti.random()-3, 6*ti.random()-3])
        fval, gfk, Hk, warnflag, xk, k = minimize_bfgs(i=i, x0=x0, gtol=1e-3, eps=1e-4)
        res_field[i] = res(fun=fval, jac=gfk, hess_inv=Hk, status=warnflag, xk=xk, k=k)
    return 0
    

t1 = time.time()
run()
res_dict = res_field.to_numpy()
print(1/((time.time()-t1)/1e6))

status = res_field.status.to_numpy()

xx, yy = np.meshgrid(np.linspace(-5,5,256), np.linspace(-5,5,256))
x = np.vstack((xx.flatten(), yy.flatten())).T
av = ackley_np(x).reshape(xx.shape)


plt.pcolor(xx, yy, av)
plt.scatter(res_dict['xk'][:,0], res_dict['xk'][:,1], s=10, c='w', alpha=0.01)
plt.show()