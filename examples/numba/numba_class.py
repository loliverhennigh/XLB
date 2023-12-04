
import cupy as cp
import numba
from numba import cuda
import numpy as np

class TestClass(object):
    def __init__(self, x):
        self.x = x
        self.a = np.arange(3)

    def add(self):
        x = self.x
        a = self.a
        @cuda.jit(device=True)
        def add(y):
            x = cuda.local.array(3, dtype=numba.float32)
            add_value = x[0] * y[0] + x[1] * y[1] + x[2] * y[2]
            return add_value
        return add

a = TestClass(100)
add = a.add()

@cuda.jit
def test_kernel(x, y):
    i = cuda.grid(1)
    y[i] = add(x[i: i + 3])

if __name__ == '__main__':
    x = cp.arange(10)
    y = cp.arange(10)
    test_kernel[1, 6](x, y)
    print(y)
    print(x)
