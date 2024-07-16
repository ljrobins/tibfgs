import matplotlib.pyplot as plt
import os
import tibfgs
import time
import numpy as np
import scipy
import polars as pl

NPART = 100_000
x0 = 4 * np.random.rand(NPART,2) - 2

_ = tibfgs.minimize(tibfgs.ackley, x0)
t1 = time.time()
res_dict = tibfgs.minimize(tibfgs.ackley, x0)
ti_per_sec = 1 / ((time.time() - t1) / NPART)
print(ti_per_sec)

print(res_dict)
endd

n_scipy = 20

methods = [
    'BFGS',
    'L-BFGS-B',
    'Powell',
    'Nelder-Mead',
    'COBYQA',
    'TNC',
    'SLSQP',
    'CG',
]
results_sp = []
for method in methods:
    res = scipy.optimize.minimize(
        fun=tibfgs.ackley_np,
        x0=x0[0],
        method=method,
    )
    t1 = time.time()
    n_iter = []
    n_fev = []
    for i in range(n_scipy):
        sol = scipy.optimize.minimize(
            fun=tibfgs.ackley_np,
            x0=x0[i],
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
