import cupy as cp
import kvikio.nvcomp
from kvikio._lib.arr import asarray
import kvikio._lib.libnvcomp as _lib

# Make sin wave data
data = cp.sin(cp.linspace(0, 2*cp.pi, 10000, dtype=cp.float32))

# Make cascaded compressor
cascaded_compressor = kvikio.nvcomp.CascadedManager(type=cp.uint32, num_RLEs=10, num_deltas=4, use_bp=False)
cascaded_compressor._manager = _lib._CascadedManager(cascaded_compressor.options, cascaded_compressor.stream, cascaded_compressor.device_id)

# Compress data
compressed = data
compressed = cascaded_compressor.compress(compressed)

# Compression ratio
print("Compression ratio: ", len(data) * data.dtype.itemsize / len(compressed))


"""
for compressor, name in compressors:
    # Compress data
    compressed = compressor.compress(data)

    # Iterate over compressed data
    for i in range(10):
        compressed = compressor.compress(compressed)
    
    # Compression ratio
    print(name)
    print("Compression ratio: ", len(data) * data.dtype.itemsize / len(compressed))
"""




