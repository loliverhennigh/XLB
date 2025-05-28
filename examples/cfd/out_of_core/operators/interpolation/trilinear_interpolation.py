import warp as wp

from pumpkin_pulse.operator.operator import Operator

class TrilinearInterpolation(Operator):

    @wp.kernel
    def trilinear_interpolation(
        grid: wp.array4d(dtype=Any),
        points: wp.array(dtype=wp.vec3),
        point_values: wp.array(dtype=Any),
        q: wp.int32,
    ):

        # Get the global index
        i = wp.tid()

        # Get the point
        point = points[i]

        # Get lower and upper bounds
        lower_0_0_0 = wp.vec3i(wp.int32(point[0]), wp.int32(point[1]), wp.int32(point[2]))
        lower_0_0_1 = lower_0_0_0 + wp.vec3i(0, 0, 1)
        lower_0_1_0 = lower_0_0_0 + wp.vec3i(0, 1, 0)
        lower_0_1_1 = lower_0_0_0 + wp.vec3i(0, 1, 1)
        lower_1_0_0 = lower_0_0_0 + wp.vec3i(1, 0, 0)
        lower_1_0_1 = lower_0_0_0 + wp.vec3i(1, 0, 1)
        lower_1_1_0 = lower_0_0_0 + wp.vec3i(1, 1, 0)
        lower_1_1_1 = lower_0_0_0 + wp.vec3i(1, 1, 1)

        # Compute the interpolation weights
        dx = point[0] - wp.float32(lower_0_0_0[0])
        dy = point[1] - wp.float32(lower_0_0_0[1])
        dz = point[2] - wp.float32(lower_0_0_0[2])
        w_000 = (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        w_001 = (1.0 - dx) * (1.0 - dy) * dz
        w_010 = (1.0 - dx) * dy * (1.0 - dz)
        w_011 = (1.0 - dx) * dy * dz
        w_100 = dx * (1.0 - dy) * (1.0 - dz)
        w_101 = dx * (1.0 - dy) * dz
        w_110 = dx * dy * (1.0 - dz)
        w_111 = dx * dy * dz

        # Loop over values to interpolate
        for n in range(q)
        
            # Get grid values
            grid_0_0_0 = grid[n, lower_0_0_0[0], lower_0_0_0[1], lower_0_0_0[2]]
            grid_0_0_1 = grid[n, lower_0_0_1[0], lower_0_0_1[1], lower_0_0_1[2]]
            grid_0_1_0 = grid[n, lower_0_1_0[0], lower_0_1_0[1], lower_0_1_0[2]]
            grid_0_1_1 = grid[n, lower_0_1_1[0], lower_0_1_1[1], lower_0_1_1[2]]
            grid_1_0_0 = grid[n, lower_1_0_0[0], lower_1_0_0[1], lower_1_0_0[2]]
            grid_1_0_1 = grid[n, lower_1_0_1[0], lower_1_0_1[1], lower_1_0_1[2]]
            grid_1_1_0 = grid[n, lower_1_1_0[0], lower_1_1_0[1], lower_1_1_0[2]]
            grid_1_1_1 = grid[n, lower_1_1_1[0], lower_1_1_1[1], lower_1_1_1[2]]

            # Compute the interpolated value
            point_value = (
                w_000 * grid_0_0_0 + w_001 * grid_0_0_1 + w_010 * grid_0_1_0 + w_011 * grid_0_1_1 +
                w_100 * grid_1_0_0 + w_101 * grid_1_0_1 + w_110 * grid_1_1_0 + w_111 * grid_1_1_1
            )

            # Set the output
            point_values[n, i] = point_value

    def __call__(
        self,
        grid,
        points,
        point_values
    ):

        # Launch the warp kernel
        wp.launch(
            self.grid_to_point,
            inputs=[grid, points, point_values, grid.shape[0]],
            dim=[points.shape[0]],
        )

        return point_values
