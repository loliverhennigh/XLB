import numpy as np
import cupy as cp
from mpi4py import MPI
import itertools
from dataclasses import dataclass

from tiles.dense_tile import DenseTile, DenseGPUTile, DenseCPUTile
from tiles.compressed_tile import CompressedTile, CompressedGPUTile, CompressedCPUTile

class OOCArray:
    """An out-of-core distributed array class."""

    def __init__(
        self,
        shape,
        dtype,
        tile_shape,
        padding=1,
        comm=None,
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=2,
    ):
        """Initialize the out-of-core array.

        Parameters
        ----------
        shape : tuple
            The shape of the array.
        dtype : cp.dtype
            The data type of the array.
        tile_shape : tuple
            The shape of the tiles. Should be a factor of the shape.
        padding : int or tuple
            The padding of the tiles.
        comm : MPI communicator
            The MPI communicator.
        devices : list of cp.cuda.Device
            The list of GPU devices to use.
        codec : Codec
            The codec to use for compression. None for no compression (Dense tiles).
        nr_compute_tiles : int
            The number of compute tiles used for asynchronous copies.
            TODO currently only 1 is supported.
        """

        self.shape = shape
        self.tile_shape = tile_shape
        self.dtype = dtype
        if isinstance(padding, int):
            padding = (padding,) * len(shape)
        self.padding = padding
        self.comm = comm
        self.devices = devices
        self.codec = codec
        self.nr_compute_tiles = nr_compute_tiles

        # Set tile class
        if self.codec is None:
            self.Tile = DenseTile
            self.ComputeTile = DenseGPUTile
            self.StoreTile = (
                DenseCPUTile  # TODO: Possibly make HardDiskTile or something
            )

        else:
            self.Tile = CompressedTile
            self.ComputeTile = CompressedGPUTile
            self.StoreTile = CompressedCPUTile

        # Get process id and number of processes
        self.pid = self.comm.Get_rank()
        self.nr_proc = self.comm.Get_size()

        # Check that the tile shape divides the array shape.
        if any([shape[i] % tile_shape[i] != 0 for i in range(len(shape))]):
            raise ValueError(f"Tile shape {tile_shape} does not divide shape {shape}.")
        self.tile_dims = tuple([shape[i] // tile_shape[i] for i in range(len(shape))])
        self.nr_tiles = np.prod(self.tile_dims)

        # Get number of tiles per process
        if self.nr_tiles % self.nr_proc != 0:
            raise ValueError(
                f"Number of tiles {self.nr_tiles} does not divide number of processes {self.nr_proc}."
            )
        self.nr_tiles_per_proc = self.nr_tiles // self.nr_proc

        # Make the tile mapppings
        self.tile_process_map = {}
        self.tile_device_map = {}
        for i, tile_index in enumerate(
            itertools.product(*[range(n) for n in self.tile_dims])
        ):
            self.tile_process_map[tile_index] = i % self.nr_proc
            self.tile_device_map[tile_index] = devices[
                i % len(devices)
            ]  # Checkoboard pattern, TODO: may not be optimal

        # Get my device
        if self.nr_proc != len(self.devices):
            raise ValueError(
                f"Number of processes {self.nr_proc} does not equal number of devices {len(self.devices)}."
            )
        self.device = self.devices[self.pid]

        # Make the tiles
        self.tiles = {}
        total_bytes = 0
        for tile_index in self.tile_process_map.keys():
            if self.pid == self.tile_process_map[tile_index]:
                self.tiles[tile_index] = self.StoreTile(
                    self.tile_shape, self.dtype, self.padding, self.codec
                )
                total_bytes += np.prod(self.tile_shape) * cp.dtype(self.dtype).itemsize

        # Make GPU tiles for copying data between CPU and GPU
        if self.nr_tiles % self.nr_compute_tiles != 0:
            raise ValueError(
                f"Number of tiles {self.nr_tiles} does not divide number of compute tiles {self.nr_compute_tiles}. This is used for asynchronous copies."
            )
        compute_array_shape = [
            s + 2 * p for (s, p) in zip(self.tile_shape, self.padding)
        ]
        self.compute_tiles = []
        self.compute_streams = []
        self.compute_arrays = []
        self.current_compute_tile_index = 0
        for i in range(self.nr_compute_tiles):
            # Make compute tile
            compute_tile = self.ComputeTile(
                    self.tile_shape, self.dtype, self.padding, self.codec
            )
            self.compute_tiles.append(compute_tile)

            # Make compute array
            self.compute_arrays.append(
                cp.empty(compute_array_shape, self.dtype, self.device)
            )

            # Make cupy stream
            self.compute_streams.append(
                cp.cuda.Stream(non_blocking=True)
            )


    def compression_ratio(self):
        if self.codec is None:
            return 1.0
        else:
            total_bytes = 0
            total_uncompressed_bytes = 0
            for tile in self.tiles.values():
                tile_total_bytes_uncompressed, tile_total_bytes_compressed = tile.compression_ratio()
                total_bytes += tile_total_bytes_compressed
                total_uncompressed_bytes += tile_total_bytes_uncompressed
            return total_uncompressed_bytes / total_bytes

    def get_stream(self):
        """Get the stream for the current compute tile."""
        return self.compute_streams[self.current_compute_tile_index]

    def update_compute_tile_index(self):
        """Update the current compute tile index."""
        self.current_compute_tile_index = (self.current_compute_tile_index + 1) % self.nr_compute_tiles

    def get_compute_array(self, tile_index):
        """Given a tile index, copy the tile to the compute array.

        Parameters
        ----------
        tile_index : tuple
            The tile index.

        Returns
        -------
        compute_array : array
            The compute array.
        global_index : tuple
            The lower bound index that the compute array corresponds to in the global array.
            For example, if the compute array is the 0th tile and has padding 1, then the
            global index will be (-1, -1, ..., -1).
        """

        # Get the tile
        tile = self.tiles[tile_index]

        # Get the compute tile
        compute_tile = self.compute_tiles[self.current_compute_tile_index]

        # Copy the tile to the compute tile
        tile.to_gpu_tile(compute_tile)

        # Concatenate the sub-arrays to make the compute array
        compute_tile.to_array(self.compute_arrays[self.current_compute_tile_index])

        # Return the compute array index in global array
        global_index = tuple(
            [i * s - p for (i, s, p) in zip(tile_index, self.tile_shape, self.padding)]
        )

        return self.compute_arrays[self.current_compute_tile_index], global_index

    def set_tile(self, compute_array, tile_index):
        """Given a tile index, copy the compute array to the tile.

        Parameters
        ----------
        compute_array : array
            The compute array.
        tile_index : tuple
            The tile index.
        """

        # Set the compute tile to the correct one
        compute_tile = self.compute_tiles[self.current_compute_tile_index]

        # Split the compute array into a tile
        compute_tile.from_array(compute_array)

        # Copy the tile from the compute tile to the store tile
        compute_tile.to_cpu_tile(self.tiles[tile_index])

    def update_padding(self):
        """Perform a padding swap between neighboring tiles."""
        ##################################
        # TODO: Currently does not use MPI
        ##################################

        # Get padding indices
        pad_ind = self.compute_tiles[0].pad_ind

        # Loop over tiles
        for tile_index in self.tile_process_map.keys():

            # if tile is on this process move padding
            if self.pid == self.tile_process_map[tile_index]:

                # Get the tile
                tile = self.tiles[tile_index]

                # Loop over all padding
                for pad_index in pad_ind:

                    # Get neighboring tile index and neighboring padding index
                    neigh_tile_index = tuple(
                        [
                            (i + p) % s
                            for (i, p, s) in zip(tile_index, pad_index, self.tile_dims)
                        ]
                    )
                    neigh_pad_index = tuple([-p for p in pad_index])  # flip

                    # Get pointer to padding from current tile
                    padding = tile._padding[pad_index]

                    # Get pointer to padding from neighboring tile
                    neigh_padding = self.tiles[neigh_tile_index]._buf_padding[
                        neigh_pad_index
                    ]

                    # Swap padding TODO: will be come more complicated with MPI
                    tile._padding[pad_index] = neigh_padding
                    self.tiles[neigh_tile_index]._buf_padding[neigh_pad_index] = padding

        # Shuffle padding with buffers
        for tile_index in self.tile_process_map.keys():
            self.tiles[tile_index].swap_buf_padding()

    def get_array(self):
        """Get the full array out from all the sub-arrays. This should only be used for testing."""

        # Get the full array
        array = np.ones(self.shape, np.float32)
        for tile_index in self.tiles.keys():
            # Get the tile
            tile = self.tiles[tile_index]

            # Copy the tile to the compute tile
            tile.to_gpu_tile(self.compute_tiles[0])

            # Get the compute array
            self.compute_tiles[0].to_array(self.compute_arrays[0])

            # Get the center array
            center_array = self.compute_arrays[0][tile._slice_center].get()

            # Set the center array in the full array
            slice_index = tuple(
                [
                    slice(i * s, (i + 1) * s)
                    for (i, s) in zip(tile_index, self.tile_shape)
                ]
            )
            array[slice_index] = center_array
        return array
