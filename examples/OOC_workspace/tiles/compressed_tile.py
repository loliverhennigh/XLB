import numpy as np
import cupy as cp
from mpi4py import MPI
import itertools
from dataclasses import dataclass
import time
from kvikio._lib.arr import asarray

from tiles.tile import Tile
from tiles.dense_tile import DenseGPUTile
from tiles.dynamic_array import DynamicPinnedArray

def _decode(comp_array, dest_array, codec):
    """
    Decompresses comp_array into dest_array.

    Parameters
    ----------
    comp_array : cupy array
        The compressed array to be decompressed. Data type uint8.
    dest_array : cupy array
        The storage array for the decompressed data.
    codec : Codec
        The codec to use for decompression. For example, `kvikio.nvcomp.CascadedManager`.

    """

    # Store needed information
    dtype = dest_array.dtype
    shape = dest_array.shape

    # Reshape dest_array to match to make into buffer
    dest_array = dest_array.view(cp.uint8).reshape(-1)

    # Configure decompression
    codec.decompression_config = (
        codec._manager.configure_decompression_with_compressed_buffer(asarray(comp_array))
    )

    # Decompress
    codec._manager.decompress(asarray(dest_array), asarray(comp_array))

    return dest_array.view(dtype).reshape(shape)

def _encode(array, dest_array, codec):
    """
    Compresses array into dest_array.

    Parameters
    ----------
    array : cupy array
        The array to be compressed.
    dest_array : cupy array
        The storage array for the compressed data. Data type uint8.
    codec : Codec
        The codec to use for compression. For example, `kvikio.nvcomp.CascadedManager`.
    """

    # Make sure array is contiguous
    array = cp.ascontiguousarray(array)

    # Configure compression
    codec._manager.configure_compression(array.nbytes)

    # Compress
    size = codec._manager.compress(asarray(array), asarray(dest_array)) 
    return size

class CompressedTile(Tile):
    """A Tile where the data is stored in compressed form."""

    def __init__(self, shape, dtype, padding, device, codec):
        super().__init__(shape, dtype, padding, device, codec)


    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        raise NotImplementedError

    def to_array(self, array):
        """Copy a tile to a full array."""
        # Only implemented for GPU tiles
        raise NotImplementedError

    def from_array(self, array):
        """Copy a full array to tile."""
        # Only implemented for GPU tiles
        raise NotImplementedError

    def compression_ratio(self):
        """Returns the compression ratio of the tile."""
        # Get total number of bytes in tile
        total_bytes = self._array.size()
        for pad_ind in self.pad_ind:
            total_bytes += self._padding[pad_ind].size()

        # Get total number of bytes in uncompressed tile
        total_bytes_uncompressed = np.prod(self.shape) * self.dtype_itemsize
        for pad_ind in self.pad_ind:
            total_bytes_uncompressed += np.prod(self._padding_shape[pad_ind]) * self.dtype_itemsize

        # Return compression ratio
        return total_bytes_uncompressed, total_bytes


class CompressedCPUTile(CompressedTile):
    """A tile with cells on the CPU."""

    def __init__(self, shape, dtype, padding, codec):
        super().__init__(shape, dtype, padding, "cpu", codec)

    def size(self):
        """Returns the size of the tile in bytes."""
        size = self._array.size()
        for pad_ind in self.pad_ind:
            size += self._padding[pad_ind].size()
        return size

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        # Make zero array
        cp_array = cp.zeros(shape, dtype=self.dtype)

        # compress array
        compressed_cp_array = self.codec.compress(cp_array)

        # Allocate array on CPU
        array =  DynamicPinnedArray(compressed_cp_array.nbytes)

        # Copy array
        compressed_cp_array.get(out=array.array)
        
        # add nbytes
        self.nbytes += cp_array.nbytes

        # delete GPU arrays
        del compressed_cp_array
        del cp_array

        return array

    def to_gpu_tile(self, dst_gpu_tile):
        """Copy tile to a GPU tile."""

        # Check tile is Compressed
        assert isinstance(dst_gpu_tile, CompressedGPUTile), "Destination tile must be a CompressedGPUTile"

        # Copy array
        dst_gpu_tile._array[:len(self._array.array)].set(self._array.array)
        dst_gpu_tile._array_bytes = self._array.nbytes

        # Copy padding
        for pad_ind in self.pad_ind:
            dst_gpu_tile._padding[pad_ind][:len(self._padding[pad_ind].array)].set(self._padding[pad_ind].array)
            dst_gpu_tile._padding_bytes[pad_ind] = self._padding[pad_ind].nbytes


class CompressedGPUTile(CompressedTile):
    """A sub-array with ghost cells on the GPU."""

    def __init__(self, shape, dtype, padding, codec):
        super().__init__(shape, dtype, padding, "cuda", codec)

        # Allocate dense GPU tile
        self.dense_gpu_tile = DenseGPUTile(shape, dtype, padding)

        # Set bytes for each array and padding
        self._array_bytes = -1
        self._padding_bytes = {}
        for pad_ind in self.pad_ind:
            self._padding_bytes[pad_ind] = -1

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        nbytes = np.prod(shape) * self.dtype_itemsize
        max_compressed_buffer = self.codec._manager.configure_compression(nbytes)['max_compressed_buffer_size']
        array =  cp.zeros((max_compressed_buffer,), dtype=np.uint8)
        return array

    def to_array(self, array):
        """Copy a tile to a full array."""

        # Copy center array
        self.dense_gpu_tile._array = _decode(self._array[:self._array_bytes], self.dense_gpu_tile._array, self.codec)
        array[self._slice_center] = self.dense_gpu_tile._array

        # Copy padding
        for pad_ind in self.pad_ind:
            self.dense_gpu_tile._padding[pad_ind] = _decode(self._padding[pad_ind][:self._padding_bytes[pad_ind]], self.dense_gpu_tile._padding[pad_ind], self.codec)
            array[self._slice_padding_to_array[pad_ind]] = self.dense_gpu_tile._padding[pad_ind]

    def from_array(self, array):
        """Copy a full array to tile."""

        # Copy center array
        self._array_bytes = _encode(array[self._slice_center], self._array, self.codec)

        # Copy padding
        for pad_ind in self.pad_ind:
            self._padding_bytes[pad_ind] = _encode(array[self._slice_array_to_padding[pad_ind]], self._padding[pad_ind], self.codec)

    def to_cpu_tile(self, dst_cpu_tile):
        """Copy tile to a CPU tile."""

        # Check tile is Compressed
        assert isinstance(dst_cpu_tile, CompressedCPUTile), "Destination tile must be a CompressedCPUTile"

        # Copy array
        dst_cpu_tile._array.resize(self._array_bytes)
        self._array[:self._array_bytes].get(out=dst_cpu_tile._array.array)

        # Copy padding
        for pad_ind in self.pad_ind:
            dst_cpu_tile._padding[pad_ind].resize(self._padding_bytes[pad_ind])
            self._padding[pad_ind][:self._padding_bytes[pad_ind]].get(out=dst_cpu_tile._padding[pad_ind].array)

