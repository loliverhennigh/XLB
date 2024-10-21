from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
import numpy as np
import matplotlib.cm as cm

from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.marching_cubes.marching_cubes import MarchingCubes


class MarchingCubesContour(MarchingCubes):

    @wp.func
    def _scalar_to_color(
        scalar: wp.float32,
        color_map: wp.array2d(dtype=wp.float32),
        color_range: wp.vec2f
    ) -> wp.vec3f:

        # Bound the scalar
        scalar = wp.clamp(scalar, color_range[0], color_range[1])

        # Get the color index
        color_index = (scalar - color_range[0]) / (color_range[1] - color_range[0]) * wp.float32((color_map.shape[0] - 1))

        # Get the color
        color = wp.vec3f(
            color_map[wp.int32(color_index), 0],
            color_map[wp.int32(color_index), 1],
            color_map[wp.int32(color_index), 2],
        )

        return color

    @wp.func
    def _trilinear_interpolation(
        color_0_0_0: wp.vec3f,
        color_1_0_0: wp.vec3f,
        color_0_1_0: wp.vec3f,
        color_1_1_0: wp.vec3f,
        color_0_0_1: wp.vec3f,
        color_1_0_1: wp.vec3f,
        color_0_1_1: wp.vec3f,
        color_1_1_1: wp.vec3f,
        vertex: wp.vec3f,
        cell_origin: wp.vec3f,
        spacing: wp.vec3f,
    ) -> wp.vec3f:

        # Get the cell center
        cell_center = cell_origin + 0.5 * spacing

        # Get the weights
        dx = (vertex[0] - cell_center[0]) / spacing[0]
        dy = (vertex[1] - cell_center[1]) / spacing[1]
        dz = (vertex[2] - cell_center[2]) / spacing[2]

        # Interpolate
        color = (
            color_0_0_0 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
            + color_1_0_0 * dx * (1.0 - dy) * (1.0 - dz)
            + color_0_1_0 * (1.0 - dx) * dy * (1.0 - dz)
            + color_1_1_0 * dx * dy * (1.0 - dz)
            + color_0_0_1 * (1.0 - dx) * (1.0 - dy) * dz
            + color_1_0_1 * dx * (1.0 - dy) * dz
            + color_0_1_1 * (1.0 - dx) * dy * dz
            + color_1_1_1 * dx * dy * dz
        )

        return color

    def _construct_warp(self):

        # Construct the warp kernels
        @wp.kernel
        def nr_triangles_kernel(
            volume: wp.array4d(dtype=wp.float32),
            contour_value: wp.float32,
            origin: wp.vec3f,
            spacing: wp.vec3f,
            nr_triangles: wp.array(dtype=wp.int32),
            vertex_indices_table: wp.array2d(dtype=wp.int32),
            padding: wp.int32,
        ):

            # Get pixel index
            i, j, k = wp.tid()

            # Add padding
            i += padding
            j += padding
            k += padding

            # Get id stencil
            v_0_0_0 = volume[0, i, j, k]
            v_1_0_0 = volume[0, i + 1, j, k]
            v_0_1_0 = volume[0, i, j + 1, k]
            v_1_1_0 = volume[0, i + 1, j + 1, k]
            v_0_0_1 = volume[0, i, j, k + 1]
            v_1_0_1 = volume[0, i + 1, j, k + 1]
            v_0_1_1 = volume[0, i, j + 1, k + 1]
            v_1_1_1 = volume[0, i + 1, j + 1, k + 1]

            # Get if its solid
            solid_0_0_0 = wp.uint8(v_0_0_0 > contour_value)
            solid_1_0_0 = wp.uint8(v_1_0_0 > contour_value)
            solid_0_1_0 = wp.uint8(v_0_1_0 > contour_value)
            solid_1_1_0 = wp.uint8(v_1_1_0 > contour_value)
            solid_0_0_1 = wp.uint8(v_0_0_1 > contour_value)
            solid_1_0_1 = wp.uint8(v_1_0_1 > contour_value)
            solid_0_1_1 = wp.uint8(v_0_1_1 > contour_value)
            solid_1_1_1 = wp.uint8(v_1_1_1 > contour_value)

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
            volume: wp.array4d(dtype=wp.float32),
            contour_value: wp.float32,
            color_volume: wp.array4d(dtype=wp.float32),
            color_map: wp.array2d(dtype=wp.float32),
            color_range: wp.vec2f,
            origin: wp.vec3f,
            spacing: wp.vec3f,
            vertex_buffer: wp.array2d(dtype=wp.float32),
            color_buffer: wp.array2d(dtype=wp.uint8),
            global_vertex_index: wp.array(dtype=wp.int32),
            vertex_indices_table: wp.array2d(dtype=wp.int32),
            vertex_table: wp.array2d(dtype=wp.float32),
            padding: wp.int32,
        ):

            # Get pixel index
            i, j, k = wp.tid()

            # Add padding
            i += padding
            j += padding
            k += padding

            # Get volume stencil
            v_0_0_0 = volume[0, i, j, k]
            v_1_0_0 = volume[0, i + 1, j, k]
            v_0_1_0 = volume[0, i, j + 1, k]
            v_1_1_0 = volume[0, i + 1, j + 1, k]
            v_0_0_1 = volume[0, i, j, k + 1]
            v_1_0_1 = volume[0, i + 1, j, k + 1]
            v_0_1_1 = volume[0, i, j + 1, k + 1]
            v_1_1_1 = volume[0, i + 1, j + 1, k + 1]

            # Get if its solid
            solid_0_0_0 = wp.uint8(v_0_0_0 > contour_value)
            solid_1_0_0 = wp.uint8(v_1_0_0 > contour_value)
            solid_0_1_0 = wp.uint8(v_0_1_0 > contour_value)
            solid_1_1_0 = wp.uint8(v_1_1_0 > contour_value)
            solid_0_0_1 = wp.uint8(v_0_0_1 > contour_value)
            solid_1_0_1 = wp.uint8(v_1_0_1 > contour_value)
            solid_0_1_1 = wp.uint8(v_0_1_1 > contour_value)
            solid_1_1_1 = wp.uint8(v_1_1_1 > contour_value)

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
            if mc_id == 0 or mc_id == 255:
                return

            # Get color stencil
            c_0_0_0 = color_volume[0, i, j, k]
            c_1_0_0 = color_volume[0, i + 1, j, k]
            c_0_1_0 = color_volume[0, i, j + 1, k]
            c_1_1_0 = color_volume[0, i + 1, j + 1, k]
            c_0_0_1 = color_volume[0, i, j, k + 1]
            c_1_0_1 = color_volume[0, i + 1, j, k + 1]
            c_0_1_1 = color_volume[0, i, j + 1, k + 1]
            c_1_1_1 = color_volume[0, i + 1, j + 1, k + 1]

            # Get color
            color_0_0_0 = MarchingCubesContour._scalar_to_color(c_0_0_0, color_map, color_range)
            color_1_0_0 = MarchingCubesContour._scalar_to_color(c_1_0_0, color_map, color_range)
            color_0_1_0 = MarchingCubesContour._scalar_to_color(c_0_1_0, color_map, color_range)
            color_1_1_0 = MarchingCubesContour._scalar_to_color(c_1_1_0, color_map, color_range)
            color_0_0_1 = MarchingCubesContour._scalar_to_color(c_0_0_1, color_map, color_range)
            color_1_0_1 = MarchingCubesContour._scalar_to_color(c_1_0_1, color_map, color_range)
            color_0_1_1 = MarchingCubesContour._scalar_to_color(c_0_1_1, color_map, color_range)
            color_1_1_1 = MarchingCubesContour._scalar_to_color(c_1_1_1, color_map, color_range)

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

                # Get color for the vertices
                color_0 = MarchingCubesContour._trilinear_interpolation(
                    color_0_0_0,
                    color_1_0_0,
                    color_0_1_0,
                    color_1_1_0,
                    color_0_0_1,
                    color_1_0_1,
                    color_0_1_1,
                    color_1_1_1,
                    vertex_0,
                    cell_origin,
                    spacing,
                )
                color_1 = MarchingCubesContour._trilinear_interpolation(
                    color_0_0_0,
                    color_1_0_0,
                    color_0_1_0,
                    color_1_1_0,
                    color_0_0_1,
                    color_1_0_1,
                    color_0_1_1,
                    color_1_1_1,
                    vertex_1,
                    cell_origin,
                    spacing,
                )
                color_2 = MarchingCubesContour._trilinear_interpolation(
                    color_0_0_0,
                    color_1_0_0,
                    color_0_1_0,
                    color_1_1_0,
                    color_0_0_1,
                    color_1_0_1,
                    color_0_1_1,
                    color_1_1_1,
                    vertex_2,
                    cell_origin,
                    spacing,
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

                # Store the colors
                color_buffer[local_vertex_index + 0, 0] = wp.uint8(255.0 * color_0[0])
                color_buffer[local_vertex_index + 0, 1] = wp.uint8(255.0 * color_0[1])
                color_buffer[local_vertex_index + 0, 2] = wp.uint8(255.0 * color_0[2])
                color_buffer[local_vertex_index + 1, 0] = wp.uint8(255.0 * color_1[0])
                color_buffer[local_vertex_index + 1, 1] = wp.uint8(255.0 * color_1[1])
                color_buffer[local_vertex_index + 1, 2] = wp.uint8(255.0 * color_1[2])
                color_buffer[local_vertex_index + 2, 0] = wp.uint8(255.0 * color_2[0])
                color_buffer[local_vertex_index + 2, 1] = wp.uint8(255.0 * color_2[1])
                color_buffer[local_vertex_index + 2, 2] = wp.uint8(255.0 * color_2[2])

        return None, (nr_triangles_kernel, mc_kernel)


    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self, 
        volume,
        contour_value,
        color_volume,
        color_range,
        origin,
        spacing,
        color_map_name="jet",
        num_table_values=256,
        padding=0, # Removes this from edges
    ):

        # Get the color map from matplotlib
        cmap = cm.get_cmap(color_map_name, num_table_values)
        color_map_np = np.array([cmap(i) for i in range(num_table_values)], dtype=np.float32)
        color_map = wp.from_numpy(color_map_np, dtype=wp.float32)

        # First get the number of triangles
        nr_triangles = wp.zeros((1,), dtype=wp.int32)
        wp.launch(
            self.warp_kernel[0],
            inputs=[
                volume,
                contour_value,
                origin,
                spacing,
                nr_triangles,
                MarchingCubes.VERTEX_INDICES_TABLE,
                padding,
            ],
            dim=[s - 1 - 2*padding for s in volume.shape[1:]],
        )
        nr_triangles = nr_triangles.numpy()[0]

        # Allocate the buffers
        vertex_buffer = wp.zeros((nr_triangles * 3, 3), dtype=wp.float32)
        color_buffer = wp.zeros((nr_triangles * 3, 3), dtype=wp.uint8)

        # Launch the kernel for the marching cubes
        global_vertex_index = wp.zeros((1,), dtype=wp.int32)
        wp.launch(
            self.warp_kernel[1],
            inputs=[
                volume,
                contour_value,
                color_volume,
                color_map,
                color_range,
                origin,
                spacing,
                vertex_buffer,
                color_buffer,
                global_vertex_index,
                MarchingCubes.VERTEX_INDICES_TABLE,
                MarchingCubes.VERTEX_TABLE,
                padding,
            ],
            dim=[s - 1 - 2*padding for s in volume.shape[1:]],
        )
        return vertex_buffer, color_buffer
