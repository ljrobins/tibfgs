from src import spbfgs, tibfgs
import numpy as np
from pprint import pprint

import src

x0 = np.array([-1.0, 1.0])
# res = spbfgs.minimize_bfgs(spbfgs.rosen, x0)

src.test_vecnorm()
src.test_fdiff()
src.test_dcstep()
src.test_dcsearch()
src.test_scalar_search_wolfe1()
src.test_wolfe1()

import taichi as ti

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