import cupy as cp
import numpy as np

#cp.cuda.set_allocator(None)
#
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed, threshold=1024**3)
#cp.cuda.set_allocator(pool.malloc)

#pinned_memory_pool = cp.cuda.PinnedMemoryPool(cp.cuda.malloc_managed)
#pinned_memory_pool = cp.cuda.PinnedMemoryPool(cp.cuda.malloc)
#cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)


# Bytes per array
nr_bytes = 1073741824
nr_bytes += 100000 # this will cause 2x memory to by allocated


total_bytes = 0
arrays = []
for _ in range(100):
    # Allocate pinned memory
    array = cp.cuda.alloc_pinned_memory(nr_bytes)
    #array = cp.cuda.pinned_memory.PinnedMemory(nr_bytes)
    print(array.nbytes)
    arrays.append(array)

    # Print the total amount of memory allocated
    total_bytes += nr_bytes
    print("Iteration {} allocated {} GB".format(_, total_bytes / 1024**3))
