from typing import Any
import warp as wp
from operator.operator import Operator

class SecondOrderUpsampler(Operator):
    """
    Second-order upsampling operator.

    This operator upsamples a field from a lower resolution to a higher resolution
    using a second-order interpolation method. It assumes the target grid is 2^n times
    larger than the original grid.

    Methods
    -------
    __call__(input_field, output_field)
        Upsample the input field to a higher resolution using second-order interpolation.
    """

    def __init__(self):
        """
        Initialize the SecondOrderUpsampler.
        """
        pass

    @wp.func
    def _periodic_indexing_3d(
        array: wp.array4d(dtype=Any),
        shape: wp.vec3i,
        component: int,
        i: int,
        j: int,
        k: int,
    ):
        # Implement periodic indexing directly in the kernel
        i = (i + shape[0]) % shape[0]
        j = (j + shape[1]) % shape[1]
        k = (k + shape[2]) % shape[2]
        return array[component, i, j, k]

    @wp.kernel
    def upsample_kernel_3d(
        input_field: wp.array4d(dtype=Any),
        output_field: wp.array4d(dtype=Any),
        scale_factor: int,
    ):
        # Get the global index for the output field
        i, j, k = wp.tid()

        # Get current point coordinates
        px = 0.5
        py = 0.5
        pz = 0.5

        # Get the upsampled grid spacing
        upsampled_dx = 1.0 / wp.float32(scale_factor)

        # Loop over the components of the input field
        for q in range(input_field.shape[0]):

            # Get 7 points around the current point
            shape = wp.vec3i(input_field.shape[1], input_field.shape[2], input_field.shape[3])
            f_1_1_1 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i, j, k)
            f_0_1_1 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i-1, j, k)
            f_2_1_1 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i+1, j, k)
            f_1_0_1 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i, j-1, k)
            f_1_2_1 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i, j+1, k)
            f_1_1_0 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i, j, k-1)
            f_1_1_2 = SecondOrderUpsampler._periodic_indexing_3d(input_field, shape, q, i, j, k+1)

            # Calculate derivatives
            df_dx = (f_2_1_1 - f_0_1_1) / 2.0
            df_dy = (f_1_2_1 - f_1_0_1) / 2.0
            df_dz = (f_1_1_2 - f_1_1_0) / 2.0

            # Calculate the second-order upsampled value
            for ii in range(scale_factor):
                for jj in range(scale_factor):
                    for kk in range(scale_factor):

                        # Get the upsampled points coordinates
                        upsampled_px = upsampled_dx * wp.float32(ii) + upsampled_dx / 2.0
                        upsampled_py = upsampled_dx * wp.float32(jj) + upsampled_dx / 2.0
                        upsampled_pz = upsampled_dx * wp.float32(kk) + upsampled_dx / 2.0

                        # Calculate the distance between the current point and the upsampled point
                        dx = upsampled_px - px
                        dy = upsampled_py - py
                        dz = upsampled_pz - pz

                        # Calculate output field value
                        output_field[q, scale_factor * i + ii, scale_factor * j + jj, scale_factor * k + kk] = (
                            f_1_1_1
                            + dx * df_dx
                            + dy * df_dy
                            + dz * df_dz
                        )

    def __call__(self, input_field: wp.array4d, output_field: wp.array4d):
        """
        Upsample the input field to a higher resolution using second-order interpolation.

        Parameters
        ----------
        input_field : wp.array4d
            The input 3D field to be upsampled.
        output_field : wp.array4d
            The output 3D field with the higher resolution.
        """

        # Calculate the scale factor
        scale_factor_x = output_field.shape[1] // input_field.shape[1]
        scale_factor_y = output_field.shape[2] // input_field.shape[2]
        scale_factor_z = output_field.shape[3] // input_field.shape[3]

        # Check that the output shape is a multiple of the input shape
        if any(o % i != 0 for i, o in zip(input_field.shape[1:], output_field.shape[1:])):
            raise ValueError("Output shape must be a multiple of input shape in each dimension.")

        # Check that the scale factor is consistent across all dimensions
        if scale_factor_x != scale_factor_y or scale_factor_x != scale_factor_z:
            raise ValueError("Inconsistent scale factor across dimensions.")
        
        # Launch the Warp kernel
        wp.launch(
            kernel=self.upsample_kernel_3d,
            dim=input_field.shape[1:],
            inputs=[input_field, output_field, scale_factor_x]
        ) 

        return output_field
