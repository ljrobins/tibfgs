Taichi BFGS (TIBFGS)
....................

This package implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) non-linear optimization algorithm in Taichi, enabling massively-parallel solutions to non-convex optimization problems.

Installation
............

::

    pip install tibfgs

Use
...

To use ``tibfgs``, define an objective function as a ``ti.func``, for example the Ackley function:

.. code-block:: python

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

Initialize :math:`n` initial conditions as an :math:`n \times m` Numpy array (where :math:`m` is the dimension of the problem, in this case :math:`m=2`):

.. code-block:: python

    n_particles = int(1e6)
    x0 = 4 * np.random.rand(n_particles, 2) - 2

Run the BFGS optimizer in parallel on the GPU with:

.. code-block:: python

    df = tibfgs.minimize(
        ackley, x0, gtol=1e-3, eps=1e-5, discard_failures=False
    )

Which returns a ``polars`` DataFrame of solutions with ``df.head()`` like:

::

    shape: (5, 10)
    ┌───────┬──────────┬───────────────┬───────────────┬───┬───────┬───────┬────────────────────────┬────────────────────────┐
    │ index ┆ fun      ┆ xk            ┆ x0            ┆ … ┆ feval ┆ geval ┆ gradient               ┆ hessian_inverse        │
    │ ---   ┆ ---      ┆ ---           ┆ ---           ┆   ┆ ---   ┆ ---   ┆ ---                    ┆ ---                    │
    │ u32   ┆ f32      ┆ array[f32, 2] ┆ array[f32, 2] ┆   ┆ i32   ┆ i32   ┆ array[f32, 2]          ┆ array[f32, (2, 2)]     │
    ╞═══════╪══════════╪═══════════════╪═══════════════╪═══╪═══════╪═══════╪════════════════════════╪════════════════════════╡
    │ 0     ┆ 5.381865 ┆ [-0.982348,   ┆ [-0.968771,   ┆ … ┆ 39    ┆ 13    ┆ [0.190735, 0.190735]   ┆ [[0.617119,            │
    │       ┆          ┆ 1.964832]     ┆ 1.981125]     ┆   ┆       ┆       ┆                        ┆ -0.479458], [-0.47…    │
    │ 1     ┆ 0.000011 ┆ [-5.7371e-7,  ┆ [1.487202,    ┆ … ┆ 147   ┆ 49    ┆ [1.716614, 0.38147]    ┆ [[0.000008, 0.000002], │
    │       ┆          ┆ -0.000005]    ┆ -1.334406]    ┆   ┆       ┆       ┆                        ┆ [0.0000…               │
    │ 2     ┆ 3.574453 ┆ [-0.968429,   ┆ [-0.806571,   ┆ … ┆ 36    ┆ 12    ┆ [0.0, 0.0]             ┆ [[0.016302, -0.00451], │
    │       ┆          ┆ -0.968395]    ┆ -0.820057]    ┆   ┆       ┆       ┆                        ┆ [-0.004…               │
    │ 3     ┆ 4.884083 ┆ [0.000768,    ┆ [-0.342866,   ┆ … ┆ 33    ┆ 11    ┆ [0.0, 0.0]             ┆ [[0.020406, 0.000647], │
    │       ┆          ┆ -1.959207]    ┆ -1.756427]    ┆   ┆       ┆       ┆                        ┆ [0.0006…               │
    │ 4     ┆ 0.000011 ┆ [0.000003,    ┆ [1.534462,    ┆ … ┆ 111   ┆ 37    ┆ [2.861023, 2.861023]   ┆ [[0.000299,            │
    │       ┆          ┆ 0.000003]     ┆ 0.979372]     ┆   ┆       ┆       ┆                        ┆ -0.000029], [-0.00…    │
    └───────┴──────────┴───────────────┴───────────────┴───┴───────┴───────┴────────────────────────┴────────────────────────┘

The full schema of this dataframe is:

:: 

    ┌─────────────────┬──────────────────────────────┐
    │ Column          ┆ Type                         │
    │ ---             ┆ ---                          │
    │ str             ┆ object                       │
    ╞═════════════════╪══════════════════════════════╡
    │ index           ┆ UInt32                       │
    │ fun             ┆ Float32                      │
    │ xk              ┆ Array(Float32, shape=(2,))   │
    │ x0              ┆ Array(Float32, shape=(2,))   │
    │ message         ┆ String                       │
    │ iterations      ┆ UInt16                       │
    │ feval           ┆ Int32                        │
    │ geval           ┆ Int32                        │
    │ gradient        ┆ Array(Float32, shape=(2,))   │
    │ hessian_inverse ┆ Array(Float32, shape=(2, 2)) │
    └─────────────────┴──────────────────────────────┘

Where:

- ``index`` is the particle initial condition index in ``x0``
- ``fun`` is the objective function value at the converged solution
- ``xk`` is the converged solution
- ``x0`` is the provided initial guess
- ``message`` describes the convergence state
- ``iterations`` is the number of state update iterations used to reach convergence
- ``feval`` is the number of objective function evaluations (including evaluations during gradient evaluation) over all iterations
- ``geval`` is the number of gradient evaluations across all iterations
- ``gradient`` is the gradient at the converged ``xk``
- ``hessian_inverse`` is the inverse of the Hessian matrix maintained by BFGS internally, at the converged ``xk``

.. note::

    ``message`` will often be ``Desired error not necessarily achieved due to precision loss`` instead of ``Optimization terminated successfully`` which is often fine and does not indicate a failed solution. Running BFGS on ``float32`` instead of ``float64`` for GPU acceleration leads to much more precision loss.