import warp as wp
import numpy as np
from mpi4py import MPI
import itertools
from dataclasses import dataclass

from warp_kernels import padded_array_to_array, array_to_padded_array

class Tile:
    """ A Tile with ghost cells. This tile is used to build a distributed array. """

    def __init__(self, shape, dtype, padding, device, pinned):
        # Store parameters
        self.shape = shape
        self.dtype = dtype
        self.padding = padding
        self.device = device
        self.pinned = pinned

        # Make center array
        self._array = wp.empty(self.shape, self.dtype, self.device, self.pinned)

        # Make padding indices
        pad_dir = []
        for i in range(len(self.shape)):
            if self.padding[i] == 0:
                pad_dir.append((0,))
            else:
                pad_dir.append((-1, 0, 1))
        self.pad_ind = list(itertools.product(*pad_dir))
        self.pad_ind.remove((0,)*len(self.shape))

        # Make padding and padding buffer arrays
        self._padding = {}
        self._buf_padding = {}
        for ind in self.pad_ind:
            # determine array shape
            shape = []
            for i in range(len(self.shape)):
                if ind[i] == -1 or ind[i] == 1:
                    shape.append(self.padding[i])
                else:
                    shape.append(self.shape[i])

            # Make padding and padding buffer
            self._padding[ind] = wp.empty(shape, self.dtype, self.device, self.pinned)
            self._buf_padding[ind] = wp.empty(shape, self.dtype, self.device, self.pinned)

        # Compute total number of values stored in the tile
        size = self._array.size
        for ind in self.pad_ind:
            size += self._padding[ind].size
            size += self._buf_padding[ind].size
        self.size = size


    @staticmethod
    def copy_tile(src_tile, dst_tile, stream=None):
        """ Copy a tile from one tile to another. """

        # Copy array
        wp.copy(dst_tile._array, src_tile._array, stream=stream)

        # Copy padding
        for (src_array, dst_array) in zip(src_tile._padding.values(), dst_tile._padding.values()):
            wp.copy(dst_array, src_array, stream=stream)

    @staticmethod
    def tile_to_array(src_tile, dst_array, stream=None):
        """ Copy a tile to a full array. """
        dst_shape = list(dst_array.shape)
        inputs = [dst_array, src_tile._array]
        inputs.extend([src_tile._padding[index] for index in src_tile._padding.keys()])
        inputs.extend(src_tile._array.shape[1:])
        inputs.append(src_tile.padding[1]) # TODO: fix this
        wp.launch(padded_array_to_array,
                  dim=list(dst_array.shape),
                  inputs=inputs,
                  stream=stream,
                  device=dst_array.device)

    @staticmethod
    def array_to_tile(src_array, dst_tile, stream=None):
        """ Copy a full array to a tile. """
        src_shape = list(src_array.shape)
        inputs = [src_array, dst_tile._array]
        inputs.extend([dst_tile._padding[index] for index in dst_tile._padding.keys()])
        inputs.extend(dst_tile._array.shape[1:])
        inputs.append(dst_tile.padding[1]) # TODO: fix this
        wp.launch(array_to_padded_array,
                  dim=list(src_array.shape),
                  inputs=inputs,
                  stream=stream,
                  device=src_array.device)

    def swap_buf_padding(self):
        """ Swap the padding buffer pointer with the padding pointer. """
        for index in self.pad_ind:
            (self._buf_padding[index], self._padding[index]) = (self._padding[index], self._buf_padding[index])

class CPUTile(Tile):
    """ A tile with cells on the CPU. """

    def __init__(self, shape, dtype, padding):
        super().__init__(shape, dtype, padding, "cpu", True)

class GPUTile(Tile):
    """ A sub-array with ghost cells on the GPU. """

    def __init__(self, shape, dtype, padding):
        super().__init__(shape, dtype, padding, "cuda", False)
