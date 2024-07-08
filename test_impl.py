import os
os.environ['TI_DIM_X'] = str(2)
os.environ['TI_NUM_PARTICLES'] = str(int(1e5))
import spbfgs, tibfgs
import numpy as np
from pprint import pprint
import taichi as ti

import tests

tests.test_matnorm()
tests.test_vecnorm()
tests.test_fdiff()
tests.test_dcstep()
# tests.test_dcsearch()
# tests.test_scalar_search_wolfe1()
# tests.test_wolfe1()
tests.test_bfgs()


# @ti.dataclass
# class Circle:
#     diameter: ti.f32
#     perimeter: ti.f32

#     @ti.func
#     def add(self, a, b):
#         return a + b

#     @ti.func
#     def iter(self) -> ti.f32:
#          return ti.math.pi * self.diameter * self.add(1.0, 2.0)

# @ti.kernel
# def test():
#     c = Circle(diameter=10.0)
#     print(c.iter())

# test()