# Base class for voxelizing geometries
import warp as wp
import numpy as np

from pumpkin_pulse.operator.operator import Operator

class Tube(Operator):
    """
    Base class for initializing a tube following a given path
    """

    @wp.func
    def _bounds(
        p1: wp.vec3,
        p2: wp.vec3,
        r1: wp.float32,
        r2: wp.float32,
    ):
        p1_minus_r = p1 - wp.vec3(r1, r1, r1)
        p1_plus_r = p1 + wp.vec3(r1, r1, r1)
        p2_minus_r = p2 - wp.vec3(r2, r2, r2)
        p2_plus_r = p2 + wp.vec3(r2, r2, r2)
        min_x = wp.min(
            wp.min(p1_minus_r.x, p1_plus_r.x),
            wp.min(p2_minus_r.x, p2_plus_r.x),
        )
        min_y = wp.min(
            wp.min(p1_minus_r.y, p1_plus_r.y),
            wp.min(p2_minus_r.y, p2_plus_r.y),
        )
        min_z = wp.min(
            wp.min(p1_minus_r.z, p1_plus_r.z),
            wp.min(p2_minus_r.z, p2_plus_r.z),
        )
        max_x = wp.max(
            wp.max(p1_minus_r.x, p1_plus_r.x),
            wp.max(p2_minus_r.x, p2_plus_r.x),
        )
        max_y = wp.max(
            wp.max(p1_minus_r.y, p1_plus_r.y),
            wp.max(p2_minus_r.y, p2_plus_r.y),
        )
        max_z = wp.max(
            wp.max(p1_minus_r.z, p1_plus_r.z),
            wp.max(p2_minus_r.z, p2_plus_r.z),
        )
        return wp.vec3(min_x, min_y, min_z), wp.vec3(max_x, max_y, max_z)

    @wp.func
    def _rotation_matrix(
        normal_0: wp.vec3,
        normal_1: wp.vec3,
    ):

        # Normalize the normals
        normal_0 = wp.normalize(normal_0)
        normal_1 = wp.normalize(normal_1)

        # Get the rotation axis
        axis = wp.cross(normal_0, normal_1)
        axis = wp.normalize(axis)

        # Get angle between the normals
        theta = wp.acos(wp.dot(normal_0, normal_1))

        # Rodrigues' rotation formula
        k = wp.mat33f(
            0.0, -axis.z, axis.y,
            axis.z, 0.0, -axis.x,
            -axis.y, axis.x, 0.0,
        )
        eye = wp.mat33f(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )
        rotation_matrix = eye + wp.sin(theta) * k + (1.0 - wp.cos(theta)) * k * k

        return rotation_matrix

    @wp.func
    def _cylinder(
        position: wp.vec3,
        center: wp.vec3,
        rotation_matrix: wp.mat33f,
        radius: wp.float32,
        height: wp.float32,
    ):
        # Get the position relative to the center
        centered_position = position - center

        # Rotate the position
        rotated_position = rotation_matrix * centered_position

        # Get the position
        d = wp.abs(wp.vec2(wp.length(wp.vec2(rotated_position.x, rotated_position.z)), rotated_position.y)) - wp.vec2(radius, height)
        max_d = wp.vec2(
            wp.max(d.x, 0.0),
            wp.max(d.y, 0.0),
        )
        return wp.length(max_d) + wp.min(wp.max(d.x, d.y), 0.0)

    @wp.func
    def _box(
        position: wp.vec3,
        center: wp.vec3,
        rotation_matrix: wp.mat33f,
        width: wp.float32,
        height: wp.float32,
    ):
        # Get the position relative to the center
        centered_position = position - center

        # Rotate the position
        rotated_position = rotation_matrix * centered_position

        # Get the position
        q = wp.abs(rotated_position) - wp.vec3(width, height, width)
        max_q = wp.vec3(
            wp.max(q.x, 0.0),
            wp.max(q.y, 0.0),
            wp.max(q.z, 0.0),
        )
        return wp.length(max_q) + wp.min(wp.max(q.x, wp.max(q.y, q.z)), 0.0)

    @wp.kernel
    def _voxelize_tube(
        id_field: Fielduint8,
        path: wp.array(dtype=wp.vec3),
        radius: wp.array(dtype=wp.float32),
        id_number: wp.uint8,
        center: wp.bool,
        square: wp.bool,
    ):
        # get index
        ii = wp.tid()

        # path center
        if center:
            # get path
            p1 = path[ii]
            p2 = path[ii + 1]

            # Get radius
            r1 = radius[ii]
            r2 = radius[ii + 1]

        # path edge
        else:
            # get path
            p1 = path[ii]
            p2 = path[ii + 1]
            p3 = path[ii + 2]

            # Get radius
            r1 = radius[ii]
            r2 = radius[ii + 1]
            r3 = radius[ii + 2]

            # Interpolate the path
            p1 = (p1 + p2) / 2.0
            p2 = (p2 + p3) / 2.0
            r1 = (r1 + r2) / 2.0
            r2 = (r2 + r3) / 2.0

        # Get normal of the path
        length = wp.length(p2 - p1)
        normal = (p2 - p1) / length

        # Get bounds
        min_bounds, max_bounds = Tube._bounds(p1, p2, r1, r2)

        # Get rotation matrix
        rotation_matrix = Tube._rotation_matrix(normal, wp.vec3(0.0, 1.0, 0.0))

        # Get lower and upper bounds in integer
        min_bounds_ijk = wp.vec3i(
            wp.int32((min_bounds.x - id_field.origin.x) / id_field.spacing.x),
            wp.int32((min_bounds.y - id_field.origin.y) / id_field.spacing.y),
            wp.int32((min_bounds.z - id_field.origin.z) / id_field.spacing.z),
        )
        max_bounds_ijk = wp.vec3i(
            wp.int32((max_bounds.x - id_field.origin.x) / id_field.spacing.x),
            wp.int32((max_bounds.y - id_field.origin.y) / id_field.spacing.y),
            wp.int32((max_bounds.z - id_field.origin.z) / id_field.spacing.z),
        )

        # Loop through the bounds
        for i in range(min_bounds_ijk.x, max_bounds_ijk.x):
            for j in range(min_bounds_ijk.y, max_bounds_ijk.y):
                for k in range(min_bounds_ijk.z, max_bounds_ijk.z):

                    # Get cell centered position
                    pos = wp.vec3(
                        id_field.origin.x + wp.float32(i) * id_field.spacing.x + 0.5 * id_field.spacing.x,
                        id_field.origin.y + wp.float32(j) * id_field.spacing.y + 0.5 * id_field.spacing.y,
                        id_field.origin.z + wp.float32(k) * id_field.spacing.z + 0.5 * id_field.spacing.z,
                    )

                    # Get sdf
                    if square:
                        distance = Tube._box(
                            pos,
                            (p1 + p2) / 2.0,
                            rotation_matrix,
                            (r1 + r2) / 2.0,
                            length / 2.0
                        )
                    else:
                        distance = Tube._cylinder(
                            pos,
                            (p1 + p2) / 2.0,
                            rotation_matrix,
                            (r1 + r2) / 2.0,
                            length / 2.0
                        )

                    # Get index
                    if distance < 0.0:
                        id_field.data[0, i, j, k] = id_number

    def __call__(
        self,
        id_field: Fielduint8,
        path: wp.array,
        radius: wp.array,
        id_number: wp.uint8,
        square: bool = False,
    ):

        # Launch the kernel for the center of the path
        wp.launch(
            self._voxelize_tube,
            inputs=[id_field, path, radius, id_number, True, square],
            dim=path.shape[0] - 1,
        )

        # Launch the kernel for the edge of the path
        wp.launch(
            self._voxelize_tube,
            inputs=[id_field, path, radius, id_number, False, square],
            dim=path.shape[0] - 2,
        )

        return id_field

class BezierTube(Operator):
    """
    Base class for initializing a tube following a bezier path
    """

    @staticmethod
    def _bezier_curve(
        point_0: np.array,
        point_1: np.array,
        point_2: np.array,
        point_3: np.array,
        num_points: int,
    ):
        """Generates a cubic Bezier curve given control points."""
        t = np.linspace(0, 1, num_points)[:, None]
        curve = (1 - t)**3 * point_0 + 3 * (1 - t)**2 * t * point_1 + 3 * (1 - t) * t**2 * point_2 + t**3 * point_3
        return curve

    @staticmethod
    def _compute_control_points(
        point_0: np.array,
        normal_0: np.array,
        point_3: np.array,
        normal_3: np.array,
        scale: float,
    ):
        """Compute the control points """
        P1 = point_0 + scale * normal_0
        P2 = point_3 - scale * normal_3
        return P1, P2

    def __call__(
        self,
        id_field: Fielduint8,
        point_0: tuple,
        normal_0: tuple,
        point_3: tuple,
        normal_3: tuple,
        num_points: int,
        scale: float,
        radius: float,
        id_number: wp.uint8,
    ):

        # Convert to np.array
        point_0 = np.array(point_0)
        point_3 = np.array(point_3)
        normal_0 = np.array(normal_0)
        normal_3 = - np.array(normal_3) # flip the normal so comes into point_3

        # Normalize the normals
        normal_0 = normal_0 / np.linalg.norm(normal_0)
        normal_3 = normal_3 / np.linalg.norm(normal_3)

        # Compute control points
        point_1, point_2 = BezierTube._compute_control_points(point_0, normal_0, point_3, normal_3, scale)

        # Compute the bezier curve
        path = BezierTube._bezier_curve(point_0, point_1, point_2, point_3, num_points)

        # Make the radius
        radius = np.ones(num_points) * radius

        # Launch the kernel for the center of the path
        wp.launch(
            Tube._voxelize_tube,
            inputs=[
                id_field,
                wp.from_numpy(path, dtype=wp.vec3),
                wp.from_numpy(radius, dtype=wp.float32),
                id_number,
                True,
                False,
            ],
            dim=path.shape[0] - 1,
        )

        # Launch the kernel for the edge of the path
        wp.launch(
            Tube._voxelize_tube,
            inputs=[
                id_field,
                wp.from_numpy(path, dtype=wp.vec3),
                wp.from_numpy(radius, dtype=wp.float32),
                id_number,
                False,
                False,
            ],
            dim=path.shape[0] - 2,
        )

        return id_field
