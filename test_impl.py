from src import spbfgs, tibfgs
import numpy as np
from pprint import pprint
import taichi as ti

import src

src.test_matnorm()
src.test_vecnorm()
src.test_fdiff()
src.test_dcstep()
# src.test_dcsearch()
# src.test_scalar_search_wolfe1()
# src.test_wolfe1()
src.test_bfgs()


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