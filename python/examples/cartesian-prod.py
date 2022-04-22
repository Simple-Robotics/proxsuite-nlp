from proxnlp.manifolds import SO2, SO3, CartesianProduct

import numpy as np

space1 = SO2()
space2 = SO3()

prod = CartesianProduct(space1, space2)


print(prod.nx)
print(prod.ndx)

Jout = np.eye(prod.ndx)

x0 = prod.neutral()
x1 = prod.rand()


space3 = space1 * space1 * space2
print(space3.nx)
print(space3.ndx)
x0 = space3.neutral()
print(x0)
