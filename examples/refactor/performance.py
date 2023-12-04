# from IPython import display
import cupy as cp
import time
from tqdm import tqdm
from numba import cuda
import warp as wp

wp.init()

if __name__ == "__main__":

    # Extremely simple compute kernel
    @cuda.jit
    def compute_kernel(f0, f1):
        i = cuda.grid(1)
        f1[i] = f0[i]

    # Allocate f
    n = 512
    nr = n**3
    f0 = cp.zeros((n, n, n), dtype=cp.float32)
    f1 = cp.zeros((n, n, n), dtype=cp.float32)

    # Run compute kernel on f
    threads_per_block = 8*8*8
    blocks_per_grid = (nr + (threads_per_block - 1)) // threads_per_block
    kernel = compute_kernel[blocks_per_grid, threads_per_block]
    tic = time.time()
    nr_iter = 1024
    for i in tqdm(range(nr_iter)):
        kernel(f0, f1)
        f0, f1 = f1, f0

    # Sync to host
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"Numba million copies per second: {(nr_iter * nr) / (toc - tic) / 1e6}")

    # Warp implementation
    @wp.kernel
    def wp_compute_kernel(f0: wp.array(dtype=wp.float32), f1: wp.array(dtype=wp.float32)):
        i = wp.tid()
        f1[i] = f0[i]

    # Allocate f
    wp_f0 = wp.zeros((nr), dtype=wp.float32)
    wp_f1 = wp.zeros((nr), dtype=wp.float32)

    # Run compute kernel on f
    tic = time.time()
    for i in tqdm(range(nr_iter)):
        wp.launch(
                wp_compute_kernel,
                dim=(nr),
                inputs=[wp_f0, wp_f1])
        wp_f0, wp_f1 = wp_f1, wp_f0
    wp.synchronize()
    toc = time.time()
    print(f"Warp million copies per second: {(nr_iter * nr) / (toc - tic) / 1e6}")


