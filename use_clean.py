import tibfgs
import numpy as np


x0 = np.random.rand(100, 2)

res = tibfgs.minimize(tibfgs.ackley, x0)

print(res)
