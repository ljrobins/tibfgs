import sys

if not 'taichi' in sys.modules:
    import taichi as ti
    ti.init(
        arch=ti.gpu,
        default_fp=ti.float32,
        fast_math=False,
        advanced_optimization=False,
        num_compile_threads=32,
        opt_level=1,
        cfg_optimization=False,
    )

from .bfgs import *
from .benchmarks import *
