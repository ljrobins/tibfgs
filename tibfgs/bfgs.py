from typing import Callable
import os
import numpy as np
import polars as pl
import taichi as ti


def minimize(
    fun: Callable,
    x0: np.ndarray,
    gtol: float = 1e-3,
    eps: float = 1e-5,
    **taichi_kwargs: dict,
) -> pl.DataFrame:
    _default_taichi_kwargs = dict(
        arch=ti.gpu,
        default_fp=ti.float32,
        fast_math=False,
        advanced_optimization=False,
        num_compile_threads=32,
        opt_level=1,
        cfg_optimization=False,
    )

    _default_taichi_kwargs.update(taichi_kwargs)

    ti.init(**_default_taichi_kwargs)

    assert x0.ndim == 2

    os.environ['TI_DIM_X'] = str(x0.shape[1])
    os.environ['TI_NUM_PARTICLES'] = str(x0.shape[0])

    from .core import set_f, minimize_kernel, res_field, VTYPE, NPART

    set_f(fun, eps=eps)

    x0s = ti.field(dtype=VTYPE, shape=NPART)
    x0s.from_numpy(x0.astype(np.float32))

    minimize_kernel(x0s, gtol=gtol)

    res_df = pl.DataFrame(res_field.to_numpy())
    res_df = res_df.with_columns(x0=x0)
    return res_df
