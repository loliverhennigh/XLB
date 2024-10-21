from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.marching_cubes.marching_cubes import MarchingCubes


class MarchingCubesBoundary(MarchingCubes):

    def _construct_warp(self):

        # Construct the warp kernels
        @wp.kernel
        def nr_triangles_kernel(
            bc_mask: wp.array4d(dtype=wp.uint8),
            id_value: wp.uint8,
            origin: wp.vec3f,
            spacing: wp.vec3f,
            nr_triangles: wp.array(dtype=wp.int32),
            vertex_indices_table: wp.array2d(dtype=wp.int32),
            padding: wp.int32 = 0,
        ):
            # Get number of triangles in the boundary

            # Get pixel index
            i, j, k = wp.tid()

            # Add padding
            i += padding
            j += padding
            k += padding

            # Get id stencil
            id_0_0_0 = bc_mask[0, i, j, k]
            id_1_0_0 = bc_mask[0, i + 1, j, k]
            id_0_1_0 = bc_mask[0, i, j + 1, k]
            id_1_1_0 = bc_mask[0, i + 1, j + 1, k]
            id_0_0_1 = bc_mask[0, i, j, k + 1]
            id_1_0_1 = bc_mask[0, i + 1, j, k + 1]
            id_0_1_1 = bc_mask[0, i, j + 1, k + 1]
            id_1_1_1 = bc_mask[0, i + 1, j + 1, k + 1]

            # Get if its solid
            solid_0_0_0 = wp.uint8(id_0_0_0 == id_value)
            solid_1_0_0 = wp.uint8(id_1_0_0 == id_value)
            solid_0_1_0 = wp.uint8(id_0_1_0 == id_value)
            solid_1_1_0 = wp.uint8(id_1_1_0 == id_value)
            solid_0_0_1 = wp.uint8(id_0_0_1 == id_value)
            solid_1_0_1 = wp.uint8(id_1_0_1 == id_value)
            solid_0_1_1 = wp.uint8(id_0_1_1 == id_value)
            solid_1_1_1 = wp.uint8(id_1_1_1 == id_value)

            # Get mc id
            mc_id = MarchingCubes._get_mc_id(
                solid_0_0_0,
                solid_1_0_0,
                solid_0_1_0,
                solid_1_1_0,
                solid_0_0_1,
                solid_1_0_1,
                solid_0_1_1,
                solid_1_1_1,
            )

            # Loop over the triangles
            local_nr_triangles = wp.int32(0)
            for _ti in range(5):

                # Get vertex index
                vertex_index_0 = vertex_indices_table[mc_id, _ti * 3 + 0]

                # Break if no triangle
                if vertex_index_0 == -1:
                    break

                # Increment the number of triangles
                local_nr_triangles += wp.int32(1)

            # Store the number of triangles
            wp.atomic_add(nr_triangles, 0, local_nr_triangles)

        # Kernel for generating the boundary
        @wp.kernel
        def mc_kernel(
            bc_mask: wp.array4d(dtype=wp.uint8),
            id_value: wp.uint8,
            origin: wp.vec3f,
            spacing: wp.vec3f,
            vertex_buffer: wp.array2d(dtype=wp.float32),
            global_vertex_index: wp.array(dtype=wp.int32),
            vertex_indices_table: wp.array2d(dtype=wp.int32),
            vertex_table: wp.array2d(dtype=wp.float32),
            padding: wp.int32 = 0,
        ):

            # Get pixel index
            i, j, k = wp.tid()

            # Add padding
            i += padding
            j += padding
            k += padding

            # Get id stencil
            id_0_0_0 = bc_mask[0, i, j, k]
            id_1_0_0 = bc_mask[0, i + 1, j, k]
            id_0_1_0 = bc_mask[0, i, j + 1, k]
            id_1_1_0 = bc_mask[0, i + 1, j + 1, k]
            id_0_0_1 = bc_mask[0, i, j, k + 1]
            id_1_0_1 = bc_mask[0, i + 1, j, k + 1]
            id_0_1_1 = bc_mask[0, i, j + 1, k + 1]
            id_1_1_1 = bc_mask[0, i + 1, j + 1, k + 1]

            # Get if its solid
            solid_0_0_0 = wp.uint8(id_0_0_0 == id_value)
            solid_1_0_0 = wp.uint8(id_1_0_0 == id_value)
            solid_0_1_0 = wp.uint8(id_0_1_0 == id_value)
            solid_1_1_0 = wp.uint8(id_1_1_0 == id_value)
            solid_0_0_1 = wp.uint8(id_0_0_1 == id_value)
            solid_1_0_1 = wp.uint8(id_1_0_1 == id_value)
            solid_0_1_1 = wp.uint8(id_0_1_1 == id_value)
            solid_1_1_1 = wp.uint8(id_1_1_1 == id_value)

            # Get mc id
            mc_id = MarchingCubes._get_mc_id(
                solid_0_0_0,
                solid_1_0_0,
                solid_0_1_0,
                solid_1_1_0,
                solid_0_0_1,
                solid_1_0_1,
                solid_0_1_1,
                solid_1_1_1,
            )

            # Get cell center
            cell_origin = wp.vec3(
                (wp.float32(i) + 0.5) * spacing[0] + origin[0],
                (wp.float32(j) + 0.5) * spacing[1] + origin[1],
                (wp.float32(k) + 0.5) * spacing[2] + origin[2],
            )

            # Loop over the triangles
            for _ti in range(5):

                # Get vertex index
                vertex_index_0 = vertex_indices_table[mc_id, _ti * 3 + 0]
                vertex_index_1 = vertex_indices_table[mc_id, _ti * 3 + 1]
                vertex_index_2 = vertex_indices_table[mc_id, _ti * 3 + 2]

                # Break if no triangle
                if vertex_index_0 == -1:
                    break

                # Get triangle vertices
                vertex_0 = wp.vec3(
                    vertex_table[vertex_index_0, 0] * spacing[0] + cell_origin[0],
                    vertex_table[vertex_index_0, 1] * spacing[1] + cell_origin[1],
                    vertex_table[vertex_index_0, 2] * spacing[2] + cell_origin[2],
                )
                vertex_1 = wp.vec3(
                    vertex_table[vertex_index_1, 0] * spacing[0] + cell_origin[0],
                    vertex_table[vertex_index_1, 1] * spacing[1] + cell_origin[1],
                    vertex_table[vertex_index_1, 2] * spacing[2] + cell_origin[2],
                )
                vertex_2 = wp.vec3(
                    vertex_table[vertex_index_2, 0] * spacing[0] + cell_origin[0],
                    vertex_table[vertex_index_2, 1] * spacing[1] + cell_origin[1],
                    vertex_table[vertex_index_2, 2] * spacing[2] + cell_origin[2],
                )

                # Get local vertex index
                local_vertex_index = wp.atomic_add(global_vertex_index, 0, 3)

                # Store the vertices
                if local_vertex_index >= vertex_buffer.shape[0]:
                    print("Vertex buffer overflow")
                    break

                vertex_buffer[local_vertex_index + 0, 0] = vertex_0[0]
                vertex_buffer[local_vertex_index + 0, 1] = vertex_0[1]
                vertex_buffer[local_vertex_index + 0, 2] = vertex_0[2]
                vertex_buffer[local_vertex_index + 1, 0] = vertex_1[0]
                vertex_buffer[local_vertex_index + 1, 1] = vertex_1[1]
                vertex_buffer[local_vertex_index + 1, 2] = vertex_1[2]
                vertex_buffer[local_vertex_index + 2, 0] = vertex_2[0]
                vertex_buffer[local_vertex_index + 2, 1] = vertex_2[1]
                vertex_buffer[local_vertex_index + 2, 2] = vertex_2[2]

        return None, (nr_triangles_kernel, mc_kernel)
 

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self, 
        bc_mask,
        id_value,
        origin,
        spacing,
        padding=0,
    ):

        # First get the number of triangles
        nr_triangles = wp.zeros((1,), dtype=wp.int32)
        wp.launch(
            self.warp_kernel[0],
            inputs=[
                bc_mask,
                id_value,
                origin,
                spacing,
                nr_triangles,
                MarchingCubes.VERTEX_INDICES_TABLE,
                padding,
            ],
            dim=[s - 1 - 2 * padding for s in bc_mask.shape[1:]],
        )
        nr_triangles = nr_triangles.numpy()[0]

        # Allocate the buffers
        vertex_buffer = wp.zeros((nr_triangles * 3, 3), dtype=wp.float32)

        # Launch the kernel for the marching cubes
        global_vertex_index = wp.zeros((1,), dtype=wp.int32)
        wp.launch(
            self.warp_kernel[1],
            inputs=[
                bc_mask,
                id_value,
                origin,
                spacing,
                vertex_buffer,
                global_vertex_index,
                MarchingCubes.VERTEX_INDICES_TABLE,
                MarchingCubes.VERTEX_TABLE,
                padding,
            ],
            dim=[s - 1 - 2 * padding for s in bc_mask.shape[1:]],
        )
        return vertex_buffer
