import matplotlib.pyplot as plt
import tibfgs
import time
import numpy as np
import taichi as ti
import polars as pl

x0 = 4 * np.random.rand(int(1e5), 2) - 2
t1 = time.time()
res_dict = tibfgs.minimize(tibfgs.ackley, x0, gtol=1e-3, eps=1e-5, discard_failures=False)
print(1 / ((time.time() - t1) / 1e6))

xx, yy = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
x = np.dstack((xx.flatten(), yy.flatten())).T
av = tibfgs.ackley_np(x).reshape(xx.shape)

xk = res_dict['xk'].to_numpy()
plt.figure(figsize=(4, 6))
plt.subplot(2, 1, 1)
plt.pcolor(xx, yy, av)
plt.scatter(xk[:, 0], xk[:, 1], s=5, c='m', alpha=0.1)
plt.title('Converged particles, Ackley function')
plt.subplot(2, 1, 2)
plt.scatter(xk[:, 0], xk[:, 1], s=5, c='m', alpha=0.1)
plt.title('Magnified view of origin')
e = 1e-5
plt.xlim(-e, e)
plt.ylim(-e, e)
plt.tight_layout()
plt.show()
