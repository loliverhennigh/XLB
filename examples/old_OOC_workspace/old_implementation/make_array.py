import numpy as np
import cupy as cp
import ctypes

# Allocate 100 bytes of memory
mempool = cp.get_default_pinned_memory_pool()
cp.cuda.set_pinned_memory_allocator(None)

nr_bytes = 100
p = []
for _ in range(10):
    pinned_mem = cp.cuda.alloc_pinned_memory(nr_bytes)
    p.append(pinned_mem)
    print(pinned_mem.size()) # prints 512
    print(mempool.n_free_blocks()) # prints 0



