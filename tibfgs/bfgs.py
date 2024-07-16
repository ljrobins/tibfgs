import taichi as ti
from typing import Callable
import os
import numpy as np
import polars as pl


def minimize(fun: Callable, x0: np.ndarray) -> pl.DataFrame:

    assert x0.ndim == 2

    os.environ['TI_DIM_X'] = str(x0.shape[1])
    os.environ['TI_NUM_PARTICLES'] = str(x0.shape[0])

    from .core import minimize_bfgs, set_f, fill_x0, minimize_kernel, res_field

    set_f(fun)
    fill_x0(x0)
    minimize_kernel()

    res_df = pl.DataFrame(res_field.to_numpy())
    res_df = res_df.with_columns(
        x0=x0
    )
    return res_df
