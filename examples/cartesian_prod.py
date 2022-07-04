"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
from proxnlp.manifolds import SO2, SO3, SE2, CartesianProduct

space1 = SO2()
space2 = SO3()
space3 = SE2()

prod = CartesianProduct(space1, space2)


print(prod.nx)
print(prod.ndx)

x0 = prod.neutral()
x1 = prod.rand()
print("x0", x0)
print("x1", x1)


prod2 = space1 * space1 * space2
print(prod2.nx)
print(prod2.ndx)
x0 = prod2.neutral()
print(x0)

s33 = space3 * space3 * space3
print("space3^3:", s33.ndx)
print("space3^3:", s33.neutral())
print("space3^3:", s33.rand())
