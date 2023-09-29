import numpy as np
import cupy as cp
from mpi4py import MPI
import itertools
from dataclasses import dataclass


class Tile:
    """Base class for Tile with ghost cells. This tile is used to build a distributed array.

    Attributes
    ----------
    shape : tuple
        Shape of the tile. This will be the shape of the array without padding/ghost cells.
    dtype : cp.dtype
        Data type the tile represents. Note that the data data may be stored in a different
        data type. For example, if it is stored in compressed form.
    padding : tuple
        Number of padding/ghost cells in each dimension.
    device : cp.Device
        Device the tile is stored on.
    pinned : bool
        Whether the tile is pinned in memory (CPU only).
    """

    def __init__(self, shape, dtype, padding, device, pinned):
        # Store parameters
        self.shape = shape
        self.dtype = dtype
        self.padding = padding
        self.device = device
        self.pinned = pinned
        self.dtype_itemsize = cp.dtype(self.dtype).itemsize

        # Make center array
        self._array = self.allocate_array(self.shape)

        # Make padding indices
        pad_dir = []
        for i in range(len(self.shape)):
            if self.padding[i] == 0:
                pad_dir.append((0,))
            else:
                pad_dir.append((-1, 0, 1))
        self.pad_ind = list(itertools.product(*pad_dir))
        self.pad_ind.remove((0,) * len(self.shape))

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
            self._padding[ind] = self.allocate_array(shape)
            self._buf_padding[ind] = self.allocate_array(shape)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        raise NotImplementedError

    def copy_tile(self, dst_tile):
        """Copy a tile from one tile to another."""
        raise NotImplementedError

    def to_array(self, array):
        """Copy a tile to a full array."""
        raise NotImplementedError

    def from_array(self, array):
        """Copy a full array to a tile."""
        raise NotImplementedError

    def swap_buf_padding(self):
        """Swap the padding buffer pointer with the padding pointer."""
        for index in self.pad_ind:
            (self._buf_padding[index], self._padding[index]) = (
                self._padding[index],
                self._buf_padding[index],
            )


class DenseTile(Tile):
    """A Tile where the data is stored in a dense array of the requested dtype."""

    def __init__(self, shape, dtype, padding, device, pinned):
        super().__init__(shape, dtype, padding, device, pinned)

        # Get slicing for array copies
        self._slice_center = tuple(
            [slice(pad, pad + shape) for (pad, shape) in zip(self.padding, self.shape)]
        )
        self._slice_padding_to_array = {}
        self._slice_array_to_padding = {}
        for pad_ind in self.pad_ind:
            slice_padding_to_array = []
            slice_array_to_padding = []
            for (pad, ind, s) in zip(self.padding, pad_ind, self.shape):
                if ind == -1:
                    slice_padding_to_array.append(slice(0, pad))
                    slice_array_to_padding.append(slice(pad, 2 * pad))
                elif ind == 0:
                    slice_padding_to_array.append(slice(pad, s + pad))
                    slice_array_to_padding.append(slice(pad, s + pad))
                else:
                    slice_padding_to_array.append(slice(s + pad, s + 2 * pad))
                    slice_array_to_padding.append(slice(s, s + pad))
            self._slice_padding_to_array[pad_ind] = tuple(slice_padding_to_array)
            self._slice_array_to_padding[pad_ind] = tuple(slice_array_to_padding)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        raise NotImplementedError

    def to_array(self, array):
        """Copy a tile to a full array."""
        # TODO: This can be done with a single kernel call, profile to see if it is faster and needs to be done.

        # Copy center array
        array[self._slice_center] = self._array

        # Copy padding
        for pad_ind in self.pad_ind:
            array[self._slice_padding_to_array[pad_ind]] = self._padding[pad_ind]

    def from_array(self, array):
        """Copy a full array to tile."""
        # TODO: This can be done with a single kernel call, profile to see if it is faster and needs to be done.

        # Copy center array
        self._array[...] = array[self._slice_center]

        # Copy padding
        for pad_ind in self.pad_ind:
            self._padding[pad_ind][...] = array[self._slice_array_to_padding[pad_ind]]


class DenseCPUTile(DenseTile):
    """A tile with cells on the CPU."""

    ###########################################################################
    # TODO: Figure out what to use for memory allocation. Need to do research
    # Currently uses way too much memory do to the way cupy allocates memory.
    # If allocations more than 1GB are made, it will double the amount of
    # memory used. This is because cupy allocates memory in 1GB chunks or something.
    ###########################################################################

    def __init__(self, shape, dtype, padding):
        super().__init__(shape, dtype, padding, "cpu", True)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        if self.pinned:
            # TODO: Seems hacky, but it works. Is there a better way?
            mem = cp.cuda.alloc_pinned_memory(np.prod(shape) * self.dtype_itemsize)
            array = np.frombuffer(mem, dtype=self.dtype, count=np.prod(shape)).reshape(
                shape
            )
        else:
            array = np.zeros(shape, dtype=self.dtype)
        return array

    def copy_tile(self, dst_tile):
        """Copy a tile from one tile to another."""

        # Copy array
        dst_tile._array.set(self._array)

        # Copy padding
        for (src_array, dst_array) in zip(
            self._padding.values(), dst_tile._padding.values()
        ):
            dst_array.set(src_array)


class DenseGPUTile(DenseTile):
    """A sub-array with ghost cells on the GPU."""

    def __init__(self, shape, dtype, padding):
        super().__init__(shape, dtype, padding, "cuda", False)

    def allocate_array(self, shape):
        """Returns a cupy array with the given shape."""
        return cp.zeros(shape, dtype=self.dtype)

    def copy_tile(self, dst_tile):
        """Copy a tile from one tile to another."""

        # Copy array
        self._array.get(out=dst_tile._array)

        # Copy padding
        for (src_array, dst_array) in zip(
            self._padding.values(), dst_tile._padding.values()
        ):
            src_array.get(out=dst_array)
