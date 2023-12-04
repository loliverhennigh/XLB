import cupy as cp
import math
import time
from tqdm import tqdm
from numba import cuda, config
import warp as wp

wp.init()

# Don't do a stream synchronize every time we launch a kernel on a CUDA Array
# Interface object (like a CuPy array)
#config.CUDA_ARRAY_INTERFACE_SYNC = False

if __name__ == "__main__":

    # Extremely simple compute kernel
    #@cuda.jit("void(float32[:, :, ::1], float32[:, :, ::1])")
    #@cuda.jit("void(float32[:, :, :], float32[:, :, :])")
    @cuda.jit
    def compute_kernel(f0, f1):
        i, j, k = cuda.grid(3)
        result = f0[i, j, k]
        for _ in range(100):
            result = math.sin(result) * cp.float32(math.pi)
        f1[i, j, k] = result

    # Allocate f
    nr = 512 + 256
    f0 = cp.zeros((nr, nr, nr), dtype=cp.float32)
    f1 = cp.zeros((nr, nr, nr), dtype=cp.float32)

    # Run compute kernel on f
    tic = time.time()
    nr_iter = 128
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (
        (nr) // threads_per_block[0],
        (nr) // threads_per_block[1],
        (nr) // threads_per_block[2],
    )
    kernel = compute_kernel[blocks_per_grid, threads_per_block]
    for i in tqdm(range(nr_iter)):
        kernel(f0, f1)
        f0, f1 = f1, f0

    # Sync to host
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"Numba million copies per second: {(nr_iter * nr**3) / (toc - tic) / 1e6}")

    # Warp implementation
    @wp.kernel
    def wp_compute_kernel(f0: wp.array3d(dtype=wp.float32), f1: wp.array3d(dtype=wp.float32)):
        i, j, k = wp.tid()
        result = f0[i, j, k]
        for _ in range(100):
            result =  wp.sin(result) * wp.float32(math.pi)
        f1[i, j, k] = result

    # Allocate f
    wp_f0 = wp.zeros((nr, nr, nr), dtype=wp.float32)
    wp_f1 = wp.zeros((nr, nr, nr), dtype=wp.float32)

    # Run compute kernel on f
    tic = time.time()
    for i in tqdm(range(nr_iter)):
        wp.launch(
                wp_compute_kernel,
                dim=(nr, nr, nr),
                inputs=[wp_f0, wp_f1])
        wp_f0, wp_f1 = wp_f1, wp_f0
    wp.synchronize()
    toc = time.time()
    print(f"Warp million copies per second: {(nr_iter * nr**3) / (toc - tic) / 1e6}")
