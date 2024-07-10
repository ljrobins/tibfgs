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
import scipy
import polars as pl

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


run()
t1 = time.time()
res_dict = res_field.to_numpy()
run()
ti_per_sec = 1 / ((time.time() - t1) / NPART)
print(ti_per_sec)

n_scipy = 20

methods = [
    'BFGS',
    'L-BFGS-B',
    'Powell',
    'Nelder-Mead',
    'COBYQA',
    'TNC',
    'SLSQP',
    'trust-constr',
    'CG',
]
results_sp = []
for method in methods:
    res = scipy.optimize.minimize(
        fun=ackley_np,
        x0=[4 * np.random.rand() - 2, 4 * np.random.rand() - 2],
        method=method,
    )
    t1 = time.time()
    n_iter = []
    n_fev = []
    for i in range(n_scipy):
        sol = scipy.optimize.minimize(
            fun=ackley_np,
            x0=[4 * np.random.rand() - 2, 4 * np.random.rand() - 2],
            method=method,
        )
        n_iter.append(sol['nit'])
        n_fev.append(sol['nfev'])
    results_sp.append(
        {
            'Method': method,
            'Solutions / sec': round(1 / ((time.time() - t1) / n_scipy)),
            'Median iterations': round(np.median(n_iter)),
            'Median func evals': round(np.median(n_fev)),
        }
    )

df = pl.DataFrame(results_sp).sort('Median func evals')
print(df)
