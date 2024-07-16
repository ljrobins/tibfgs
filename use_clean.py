import tibfgs
import numpy as np
import taichi as ti
import time

x0 = np.random.rand(10_000_000, 2)

t1 = time.time()
res = tibfgs.minimize(tibfgs.ackley, x0, arch=ti.cpu)
print(time.time() - t1)

# print(res)
