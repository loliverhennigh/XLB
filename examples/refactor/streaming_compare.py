# from IPython import display
import cupy as cp
import time
from tqdm import tqdm
from numba import cuda, config

config.CUDA_ARRAY_INTERFACE_SYNC = False

if __name__ == "__main__":

    # Extremely simple compute kernel
    @cuda.jit('void(float32[::1,::1,:,:], float32[::1,::1,:,:])')
    def numba_stream(f0, f1):
        i, j, k = cuda.grid(3)
        for l in range(19):
            f1[i, j, k, l] = f0[i, j, k, l]

    # Allocate f
    n = 256
    f0 = cp.zeros((19, n, n, n), dtype=cp.float32)
    f1 = cp.zeros((19, n, n, n), dtype=cp.float32)

    # Run compute kernel on f
    threads_per_block = (8, 8, 8)
    blocks_per_grid = (n // threads_per_block[0], n // threads_per_block[1], n // threads_per_block[2])
    kernel = numba_stream[blocks_per_grid, threads_per_block]
    tic = time.time()
    nr_iter = 128
    for i in tqdm(range(nr_iter)):
        kernel(f0, f1)
        f0, f1 = f1, f0

    # Sync to host
    cp.cuda.stream.get_current_stream().synchronize()
    toc = time.time()
    print(f"Numba million copies per second: {(nr_iter * n**3) / (toc - tic) / 1e6}")
