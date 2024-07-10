import taichi as ti

ti.init(arch=ti.metal, default_fp=ti.float32)

import os

os.environ['TI_DIM_X'] = str(2)
os.environ['TI_NUM_PARTICLES'] = str(int(1e5))

import spbfgs, tibfgs
import numpy as np

import tests

tests.test_matnorm()
tests.test_vecnorm()
tests.test_fdiff()
tests.test_dcstep()
# tests.test_dcsearch()
# tests.test_scalar_search_wolfe1()
# tests.test_wolfe1()
tests.test_bfgs()
