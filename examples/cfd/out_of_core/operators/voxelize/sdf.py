# Base class for setting Signed distance fields

from typing import Union
import warp as wp

from pumpkin_pulse.operator.operator import Operator

class SignedDistanceFunction(Operator):
    """
    Yee Cell electric field update operator
    """

    @staticmethod
    def sphere(
        center: wp.vec3,
        radius: float,
    ):

        if not isinstance(center, wp.vec3):
            center = wp.vec3(center)

        @wp.func
        def _sphere(
            position: wp.vec3,
        ):
            return wp.length(position - center) - radius

        return _sphere

    @staticmethod
    def cylinder(
        center: wp.vec3,
        radius: float,
        height: float,
    ):

        if not isinstance(center, wp.vec3):
            center = wp.vec3(center)

        @wp.func
        def _cylinder(
            position: wp.vec3,
        ):
            normalized_position = position - center
            d = wp.abs(wp.vec2(wp.length(wp.vec2(normalized_position.x, normalized_position.z)), normalized_position.y)) - wp.vec2(radius, height)
            max_d = wp.vec2(
                wp.max(d.x, 0.0),
                wp.max(d.y, 0.0),
            )
            return wp.length(max_d) + wp.min(wp.max(d.x, d.y), 0.0)

        return _cylinder

    @staticmethod
    def box(
        center: wp.vec3,
        size: wp.vec3,
    ):

        if not isinstance(center, wp.vec3):
            center = wp.vec3(center)
        if not isinstance(size, wp.vec3):
            size = wp.vec3(size)

        @wp.func
        def _box(
            position: wp.vec3,
        ):
            normalized_position = position - center
            qp = wp.abs(normalized_position) - size
            max_qp = wp.vec3(
                wp.max(qp.x, 0.0),
                wp.max(qp.y, 0.0),
                wp.max(qp.z, 0.0),
            )
            return wp.length(max_qp) + wp.min(wp.max(qp.x, wp.max(qp.y, qp.z)), 0.0)

        return _box

    @staticmethod
    def union(
        sdf1: wp.func,
        sdf2: wp.func,
    ):

        @wp.func
        def _union(
            position: wp.vec3,
        ):
            return wp.min(sdf1(position), sdf2(position))

        return _union

    @staticmethod
    def intersection(
        sdf1: wp.func,
        sdf2: wp.func,
    ):

        @wp.func
        def _intersection(
            position: wp.vec3,
        ):
            return wp.max(sdf1(position), sdf2(position))

        return _intersection

    @staticmethod
    def difference(
        sdf1: wp.func,
        sdf2: wp.func,
    ):

        @wp.func
        def _difference(
            position: wp.vec3,
        ):
            return wp.max(sdf1(position), -sdf2(position))

        return _difference

    @staticmethod
    def translate(
        sdf: wp.func,
        translation: wp.vec3,
    ):

        if not isinstance(translation, wp.vec3):
            translation = wp.vec3(translation)

        @wp.func
        def _translate(
            position: wp.vec3,
        ):
            return sdf(position - translation)

        return _translate

    @staticmethod
    def rotate(
        sdf: wp.func,
        center: wp.vec3,
        centeraxis: wp.vec3,
        angle: float,
    ):

        if not isinstance(center, wp.vec3):
            center = wp.vec3(center)
        if not isinstance(centeraxis, wp.vec3):
            centeraxis = wp.vec3(centeraxis)

        @wp.func
        def _rotate(
            position: wp.vec3,
        ):
            # Translate
            position -= center

            # Rotate
            c = wp.cos(angle)
            s = wp.sin(angle)
            axis = wp.normalize(centeraxis)
            temp = position * c + wp.cross(axis, position) * s + axis * wp.dot(axis, position) * (1.0 - c)

            # Translate back
            return sdf(temp + center)

        return _rotate

    def __init__(
        self,
        sdf_func: wp.func,
    ):

        # Make kernels
        @wp.kernel
        def _set_sdf(
            id_field: wp.array4d(dtype=wp.int16),
            origin: wp.vec3,
            spacing: wp.vec3,
            set_id: wp.int16,
        ):
            # get index
            i, j, k = wp.tid()

            # Get cell centered position
            position = wp.vec3(
                wp.float32(i) * spacing[0] + origin[0] + 0.5 * spacing[0],
                wp.float32(j) * spacing[1] + origin[1] + 0.5 * spacing[1],
                wp.float32(k) * spacing[2] + origin[2] + 0.5 * spacing[2],
            )

            # Compute sdf
            sdf = sdf_func(position)

            # Set id
            if sdf < 0.0:
                id_field[0, i, j, k] = set_id

        self._set_sdf = _set_sdf

    def __call__(
        self,
        id_field: wp.array4d(dtype=wp.int16),
        origin: wp.vec3,
        spacing: wp.vec3,
        set_id: int,
    ):

        # Launch kernel
        wp.launch(
            self._set_sdf,
            inputs=[
                id_field,
                origin,
                spacing,
                set_id,
            ],
            dim=id_field.shape[1:],
        )

        return id_field
