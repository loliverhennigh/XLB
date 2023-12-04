import cupy as cp
import math
import time
from numba import cuda, config

# Don't do a stream synchronize every time we launch a kernel on a CUDA Array
# Interface object (like a CuPy array)
config.CUDA_ARRAY_INTERFACE_SYNC = False

if __name__ == "__main__":

    # Extremely simple compute kernel
    @cuda.jit("void(float32[:, :, ::1], float32[:, :, ::1])")
    def compute_kernel(f0, f1):
        i, j, k = cuda.grid(3)
        result = f0[i, j, k]
        for _ in range(10):
            result = result + cp.float32(1.0)
        f1[i, j, k] = result

    # Extremely simple copy kernel
    @cuda.jit("void(float32[:, :, ::1], float32[:, :, ::1])")
    def copy_kernel(f0, f1):
        i, j, k = cuda.grid(3)
        result = f0[i, j, k]
        for _ in range(10):
            result = result + cp.float32(1.01)
        f1[i, j, k] = result

    # Allocate f
    nr = 512
    f0 = cp.random.rand(nr, nr, nr).astype(cp.float32)
    f1 = cp.random.rand(nr, nr, nr).astype(cp.float32)

    # Set up kernel launch
    nr_iter = 128
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nr) // threads_per_block[0],
        (nr) // threads_per_block[1],
        (nr) // threads_per_block[2],
    )
    _compute_kernel = compute_kernel[blocks_per_grid, threads_per_block]
    _copy_kernel = copy_kernel[blocks_per_grid, threads_per_block]

    # Run 1 iteration of each for warmup
    _compute_kernel(f0, f1)
    _copy_kernel(f0, f1)

    # Run Copy kernel
    tic = time.time()
    for i in range(nr_iter):
        _copy_kernel(f0, f1)
        f0, f1 = f1, f0
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"Numba million copy copies per second: {(nr_iter * nr**3) / (toc - tic) / 1e6}")


    # Run Compute kernel 
    tic = time.time()
    for i in range(nr_iter):
        _compute_kernel(f0, f1)
        f0, f1 = f1, f0
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"Numba million compute copies per second: {(nr_iter * nr**3) / (toc - tic) / 1e6}")


