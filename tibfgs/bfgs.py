from typing import Callable
import os
import numpy as np
import polars as pl
import taichi as ti

FLAG_SUCCESS_MSG = 'Optimization terminated successfully'
FLAG_MAX_ITER_MSG = 'Max number of iterations reached'
FLAG_PRECISION_LOSS_MSG = 'Desired error not necessarily achieved due to precision loss'
FLAG_NAN_MSG = 'NaN result encountered'
FLAG_ERROR_MSG = 'Error occured during optimization'
FLAG_MAX_FEVAL_MSG = 'Max function evaluations exceeded'

_default_taichi_kwargs = dict(
    arch=ti.gpu,
    default_fp=ti.float32,
    fast_math=False,
    advanced_optimization=False,
    num_compile_threads=32,
    opt_level=1,
    cfg_optimization=False,
)


def init_ti(**kwargs):
    if 'TI_BFGS_INIT' not in os.environ:
        np.finfo(np.float32)
        _default_taichi_kwargs.update(kwargs)
        ti.init(**_default_taichi_kwargs)
        os.environ['TI_BFGS_INIT'] = 'True'


def minimize(
    fun: Callable,
    x0: np.ndarray,
    gtol: float = 1e-3,
    eps: float = 1e-5,
    maxiter: int = 100,
    maxfeval: int = 1000,
    discard_failures: bool = False,
    **taichi_kwargs: dict,
) -> pl.DataFrame:
    assert x0.ndim == 2

    os.environ['TI_DIM_X'] = str(x0.shape[1])
    os.environ['TI_NUM_PARTICLES'] = str(x0.shape[0])

    init_ti(**taichi_kwargs)

    from .core import (
        set_f,
        minimize_kernel,
        res_field,
        VTYPE,
        NPART,
    )

    set_f(fun, eps=eps)

    x0s = ti.field(dtype=VTYPE, shape=NPART)
    x0_as_dtype = x0.astype(np.float32)
    x0s.from_numpy(x0_as_dtype)

    minimize_kernel(x0s, gtol=gtol, maxiter=maxiter, maxfeval=maxfeval)

    res_df = pl.DataFrame(res_field.to_numpy())

    res_df = (
        res_df.with_columns(
            pl.when(pl.col('status') == 0)
            .then(pl.lit(FLAG_SUCCESS_MSG))
            .when(pl.col('status') == 1)
            .then(pl.lit(FLAG_MAX_ITER_MSG))
            .when(pl.col('status') == 2)
            .then(pl.lit(FLAG_PRECISION_LOSS_MSG))
            .when(pl.col('status') == 3)
            .then(pl.lit(FLAG_NAN_MSG))
            .when(pl.col('status') == 4)
            .then(pl.lit(FLAG_ERROR_MSG))
            .when(pl.col('status') == 5)
            .then(pl.lit(FLAG_MAX_FEVAL_MSG))
            .alias('message'),
            x0=x0_as_dtype,
        )
        .rename({'grad': 'gradient', 'k': 'iterations', 'hess_inv': 'hessian_inverse'})
        .cast({'iterations': pl.UInt16})
        .select(
            'fun',
            'xk',
            'x0',
            'message',
            'iterations',
            'feval',
            'geval',
            'gradient',
            'hessian_inverse',
            'status',
            'task',
        )
        .with_row_index()
    )

    if discard_failures:
        res_df = res_df.filter(
            pl.col('task') < 200,
        )

    return res_df.drop('status', 'task')
