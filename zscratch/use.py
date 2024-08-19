import matplotlib.pyplot as plt
import tibfgs
import time
import numpy as np
import polars as pl
import taichi as ti

# %%
# Let's solve a 2D non-linear, non-convex optimization problem from 1e6 initial conditions. First we define an objective function:

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

# %%
# Then we set up our initial conditions

n_particles = int(1e6)
x0 = 4 * np.random.rand(n_particles, 2) - 2

# %%
# And compute the solution
t1 = time.time()
solution_df = tibfgs.minimize(
    ackley, x0, gtol=1e-3, eps=1e-5, discard_failures=False
)
print(f'M1 Mac: {n_particles / 1e6 / ((time.time() - t1)):.2f} million convergences / sec')

# %%
# We can look at the returned dataframe

print(solution_df.tail()['message'][0])

# %%
# The full schema of the solution is given by:

dfs = pl.DataFrame({"Column": solution_df.schema.keys(), "Type": solution_df.schema.values()})
print(dfs)

# %%
# We can plot the final solutions of each particle

xx, yy = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
x = np.dstack((xx.flatten(), yy.flatten())).T
av = tibfgs.ackley_np(x).reshape(xx.shape)

xk = solution_df['xk'].to_numpy()
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
