from typing import List
import warp as wp

from ds.ooc_grid import MemoryPool
from operators.operator import Operator
from subroutine.ooc_grid.ooc_grid_subroutine import OOCGridSubroutine
from operators.copy.soa_copy import SOACopy

class VoxelizerSubroutine(OOCGridSubroutine):

    def __init__(
        self,
        voxelizer: Operator,
        my_copy: Operator = SOACopy(),
        nr_streams: int = 1,
        wp_streams: List[wp.Stream] = None,
        memory_pools: List[MemoryPool] = None,
    ):
        self.voxelizer =voxelizer 
        self.my_copy = my_copy
        super().__init__(nr_streams, wp_streams, memory_pools)

    def __call__(
        self,
        amr_grid,
        id_field_name="id_field",
        clear_memory_pools=True,
    ):

        # Make stream idx
        stream_idx = 0

        # MPI communication parameters
        comm_tag = 0
        requests = []

        # Set initial conditions
        for block in amr_grid.blocks.values():

            # Set warp stream
            with wp.ScopedStream(self.wp_streams[stream_idx]):

                # Check if block matches pid 
                if block.pid == amr_grid.pid:

                    # Get compute arrays
                    id_field = self.memory_pools[stream_idx].get((1, *block.shape), wp.int16)
                    id_field_ghost = {}
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        id_field_ghost[ghost_block] = self.memory_pools[stream_idx].get(
                            (1, *ghost_boxes[id_field_name].shape),
                            wp.int16
                        )

                    # Initialize the id field
                    id_field = self.voxelizer(
                        id_field,
                        block.local_origin,
                        block.local_spacing,
                    )

                    # Copy to local ghost boxes
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():

                        # Get slice start and stop
                        slice_start = (ghost_boxes[id_field_name].offset - block.boxes[id_field_name].offset)
                        slice_stop = slice_start + ghost_boxes[id_field_name].shape
                        slice_start = tuple([int(s) for s in slice_start])
                        slice_stop = tuple([int(s) for s in slice_stop])

                        # Copy
                        self.my_copy(
                            id_field_ghost[ghost_block],
                            id_field[
                                :,
                                slice_start[0]:slice_stop[0],
                                slice_start[1]:slice_stop[1],
                                slice_start[2]:slice_stop[2],
                            ]
                        )

                    # Copy to block
                    wp.copy(block.boxes[id_field_name].data, id_field)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        wp.copy(ghost_boxes[id_field_name].data, id_field_ghost[ghost_block])

                    # Return arrays
                    self.memory_pools[stream_idx].ret(id_field, zero=True)
                    for ghost_block, ghost_boxes in block.local_ghost_boxes.items():
                        self.memory_pools[stream_idx].ret(id_field_ghost[ghost_block], zero=True)

                    # Update stream idx
                    stream_idx = (stream_idx + 1) % self.nr_streams

        # Send blocks
        wp.synchronize()
        for block in amr_grid.blocks.values():
            r, comm_tag = block.send_ghost_boxes(
                amr_grid.comm,
                comm_tag=comm_tag,
                names=[id_field_name],
            )
            requests.extend(r)

        # Wait for requests
        if amr_grid.comm is not None:
            self.comm.Barrier()
            MPI.Request.Waitall(requests)
            pass
        else:
            assert len(requests) == 0

        # Swap neighbour buffers
        for block in amr_grid.blocks.values():
            if block.pid == amr_grid.pid:
                block.swap_buffers(
                    names=[id_field_name],
                )

        # Clear memory pools
        if clear_memory_pools:
            for memory_pool in self.memory_pools:
                memory_pool.clear()
