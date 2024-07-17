import tibfgs
import numpy as np
import taichi as ti
import time

x0 = np.random.rand(int(1e6), 2)

t1 = time.time()
res = tibfgs.minimize(
    tibfgs.ackley, x0, arch=ti.cpu, maxiter=10, maxfeval=500, discard_failures=True
)
dt = time.time() - t1


print(res.sort('fun').drop('gradient', 'hessian_inverse'))

total_feval = res['feval'].sum()
print(total_feval / dt / 1e6)
