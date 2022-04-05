import lienlp
import numpy as np



class MyFunction(lienlp.BaseFunction):
    def __init__(self, nx, ndx, nr):
        lienlp.BaseFunction(nx, ndx, nr)

    def __call__(self, x):
        return np.log(np.abs(x))


res = MyFunction(2, 2, 2)

x0 = np.random.randn(2)
print(res(x0))
assert np.allclose(res(x0), np.log(np.abs(x0)))
