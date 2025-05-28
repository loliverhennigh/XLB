import warp as wp
import numpy as np
from pumpkin_pulse.operator.operator import Operator

class VectorFieldColorMapper(Operator):
    """
    Operator for mapping vector fields and LIC results to RGBA volumes.
    
    This operator combines vector field magnitude (mapped to color using jet colormap)
    with LIC results (used to modulate opacity) to create a volumetric visualization.
    """
    
    @staticmethod
    @wp.func
    def jet_colormap(value: float) -> wp.vec3:
        """
        Map a value in [0,1] to RGB colors using the jet colormap.
        """
        r = wp.clamp(wp.min(4.0 * value - 1.5, -4.0 * value + 4.5), 0.0, 1.0)
        g = wp.clamp(wp.min(4.0 * value - 0.5, -4.0 * value + 3.5), 0.0, 1.0)
        b = wp.clamp(wp.min(4.0 * value + 0.5, -4.0 * value + 2.5), 0.0, 1.0)
        
        return wp.vec3(r, g, b)
    
    @staticmethod
    @wp.func
    def compute_field_magnitude(
        vector_field: wp.array4d(dtype=wp.float32),
        i: int, j: int, k: int
    ) -> float:
        """Compute magnitude of vector field at given position."""
        x = vector_field[0, i, j, k]
        y = vector_field[1, i, j, k]
        z = vector_field[2, i, j, k]
        return wp.sqrt(x*x + y*y + z*z)
    
    @wp.kernel
    def _map_to_volume(
        vector_field: wp.array4d(dtype=wp.float32),  # (3, nx, ny, nz)
        lic_result: wp.array4d(dtype=wp.float32),    # (1, nx, ny, nz)
        volume: wp.array4d(dtype=wp.float32),        # (4, nx, ny, nz)
        vmin: float,
        vmax: float,
        max_opacity: float,
        lic_threshold: float,
    ):
        """Map vector field and LIC result to RGBA volume."""
        # Get thread indices
        i, j, k = wp.tid()
        
        # Compute field magnitude and normalize
        magnitude = VectorFieldColorMapper.compute_field_magnitude(vector_field, i, j, k)
        magnitude_norm = (magnitude - vmin) / (vmax - vmin)
        magnitude_norm = wp.clamp(magnitude_norm, 0.0, 1.0)
        
        # Get color from jet colormap
        color = VectorFieldColorMapper.jet_colormap(magnitude_norm)
        
        # Store RGB values
        volume[0, i, j, k] = color[0]
        volume[1, i, j, k] = color[1]
        volume[2, i, j, k] = color[2]
        
        # Get LIC value (assumed to be in [0,1] range)
        lic_value = lic_result[0, i, j, k]

        # Apply LIC threshold
        if lic_value < lic_threshold:
            lic_value = 0.0
        else:
            lic_value = 1.0
        
        # Combine LIC and magnitude for opacity
        alpha = lic_value * magnitude_norm

        # Scale to max opacity and clamp
        volume[3, i, j, k] = alpha * max_opacity
    
    def __call__(
        self,
        vector_field: wp.array4d(dtype=wp.float32),
        lic_result: wp.array4d(dtype=wp.float32),
        volume: wp.array4d(dtype=wp.float32),
        vmin: float,
        vmax: float,
        max_opacity: float = 0.8,
        lic_threshold: float = 0.5
    ) -> wp.array4d:
        """
        Map vector field and LIC result to RGBA volume.
        
        Parameters
        ----------
        vector_field : wp.array4d(dtype=wp.float32)
            Input vector field with shape (3, nx, ny, nz)
        lic_result : wp.array4d(dtype=wp.float32)
            Line integral convolution result with shape (1, nx, ny, nz).
            Expected to be normalized to [0,1] range.
        volume : wp.array4d(dtype=wp.float32)
            Output RGBA volume with shape (4, nx, ny, nz)
        vmin : float
            Minimum value for vector magnitude normalization
        vmax : float
            Maximum value for vector magnitude normalization
        max_opacity : float, optional
            Maximum opacity value (default: 0.8)
        lic_threshold : float, optional
            Threshold for LIC result (default: 0.5)
            
        Returns
        -------
        wp.array4d(dtype=wp.float32)
            Reference to the input volume array
        """
        # Verify input shapes
        assert vector_field.shape[0] == 3, "Vector field must have 3 components"
        assert lic_result.shape[0] == 1, "LIC result must have 1 component"
        assert volume.shape[0] == 4, "Volume must have 4 components (RGBA)"
        assert vector_field.shape[1:] == lic_result.shape[1:] == volume.shape[1:], "Spatial dimensions must match"
        
        # Verify vmin/vmax
        assert isinstance(vmin, float), "vmin must be a float"
        assert isinstance(vmax, float), "vmax must be a float"
        assert vmax > vmin, f"vmax ({vmax}) must be greater than vmin ({vmin})"
        
        # Launch kernel
        wp.launch(
            kernel=self._map_to_volume,
            dim=(vector_field.shape[1], vector_field.shape[2], vector_field.shape[3]),
            inputs=[
                vector_field,
                lic_result,
                volume,
                vmin,
                vmax,
                max_opacity,
                lic_threshold
            ]
        )
        
        return volume 