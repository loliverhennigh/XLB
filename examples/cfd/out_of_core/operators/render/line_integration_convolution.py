import warp as wp
from pumpkin_pulse.operator.operator import Operator
import math

class LineIntegrationConvolution(Operator):
    """
    Operator that performs one step of line integration convolution on a vector field.
    
    Takes a 3D vector field and a noise field, performs LIC integration,
    and returns the line integral field. Uses trilinear interpolation for smooth sampling
    and a uniform kernel for streamline weighting.
    """
    
    @staticmethod
    @wp.func
    def sample_noise_trilinear(
        noise: wp.array4d(dtype=wp.float32),
        pos: wp.vec3,
        shape: wp.vec3i
    ) -> float:
        """Sample noise field using trilinear interpolation."""
        # Get base indices
        i0 = int(wp.floor(pos[0]))
        j0 = int(wp.floor(pos[1]))
        k0 = int(wp.floor(pos[2]))
        
        # Get fractional parts
        fx = pos[0] - float(i0)
        fy = pos[1] - float(j0)
        fz = pos[2] - float(k0)
        
        # Complement of fractional parts
        fx1 = wp.float32(1.0) - fx
        fy1 = wp.float32(1.0) - fy
        fz1 = wp.float32(1.0) - fz
        
        # Sample with clamping
        ix0 = wp.clamp(i0, 0, shape[0]-1)
        ix1 = wp.clamp(i0+1, 0, shape[0]-1)
        jy0 = wp.clamp(j0, 0, shape[1]-1)
        jy1 = wp.clamp(j0+1, 0, shape[1]-1)
        kz0 = wp.clamp(k0, 0, shape[2]-1)
        kz1 = wp.clamp(k0+1, 0, shape[2]-1)
        
        # Sample all corners
        c000 = noise[0, ix0, jy0, kz0]
        c001 = noise[0, ix0, jy0, kz1]
        c010 = noise[0, ix0, jy1, kz0]
        c011 = noise[0, ix0, jy1, kz1]
        c100 = noise[0, ix1, jy0, kz0]
        c101 = noise[0, ix1, jy0, kz1]
        c110 = noise[0, ix1, jy1, kz0]
        c111 = noise[0, ix1, jy1, kz1]
        
        # Interpolate along x
        c00 = c000 * fx1 + c100 * fx
        c01 = c001 * fx1 + c101 * fx
        c10 = c010 * fx1 + c110 * fx
        c11 = c011 * fx1 + c111 * fx
        
        # Interpolate along y
        c0 = c00 * fy1 + c10 * fy
        c1 = c01 * fy1 + c11 * fy
        
        # Interpolate along z
        return c0 * fz1 + c1 * fz
    
    @staticmethod
    @wp.func
    def sample_vector_field(
        vector_field: wp.array4d(dtype=wp.float32),
        pos: wp.vec3,
        shape: wp.vec3i
    ) -> wp.vec3:
        """Sample vector field using trilinear interpolation."""
        # Get base indices
        i0 = int(wp.floor(pos[0]))
        j0 = int(wp.floor(pos[1]))
        k0 = int(wp.floor(pos[2]))
        
        # Get fractional parts
        fx = pos[0] - float(i0)
        fy = pos[1] - float(j0)
        fz = pos[2] - float(k0)
        
        # Complement of fractional parts
        fx1 = wp.float32(1.0) - fx
        fy1 = wp.float32(1.0) - fy
        fz1 = wp.float32(1.0) - fz
        
        # Sample with clamping
        ix0 = wp.clamp(i0, 0, shape[0]-1)
        ix1 = wp.clamp(i0+1, 0, shape[0]-1)
        jy0 = wp.clamp(j0, 0, shape[1]-1)
        jy1 = wp.clamp(j0+1, 0, shape[1]-1)
        kz0 = wp.clamp(k0, 0, shape[2]-1)
        kz1 = wp.clamp(k0+1, 0, shape[2]-1)
        
        # Initialize result vector
        result = wp.vec3(0.0, 0.0, 0.0)
        
        # Sample each component
        for c in range(3):
            c000 = vector_field[c, ix0, jy0, kz0]
            c001 = vector_field[c, ix0, jy0, kz1]
            c010 = vector_field[c, ix0, jy1, kz0]
            c011 = vector_field[c, ix0, jy1, kz1]
            c100 = vector_field[c, ix1, jy0, kz0]
            c101 = vector_field[c, ix1, jy0, kz1]
            c110 = vector_field[c, ix1, jy1, kz0]
            c111 = vector_field[c, ix1, jy1, kz1]
            
            # Interpolate along x
            c00 = c000 * fx1 + c100 * fx
            c01 = c001 * fx1 + c101 * fx
            c10 = c010 * fx1 + c110 * fx
            c11 = c011 * fx1 + c111 * fx
            
            # Interpolate along y
            c0 = c00 * fy1 + c10 * fy
            c1 = c01 * fy1 + c11 * fy
            
            # Interpolate along z and store
            result[c] = c0 * fz1 + c1 * fz
            
        return result
    
    @staticmethod
    @wp.func
    def kernel_weight(t: float, kernel_width: int) -> float:
        """Compute weight for streamline integration using linear falloff."""
        # Linear falloff from center (1.0) to edges (0.0)
        t = wp.abs(t)  # Make symmetric around center
        if t >= float(kernel_width):
            return wp.float32(0.0)
        return wp.float32(1.0) - (t / float(kernel_width))
    
    @wp.kernel
    def _lic_step(
        vector_field: wp.array4d(dtype=wp.float32),  # (3, nx, ny, nz)
        noise: wp.array4d(dtype=wp.float32),         # (1, nx, ny, nz)
        line_integral: wp.array4d(dtype=wp.float32), # (1, nx, ny, nz)
        step_size: float,
        nr_steps: int,
    ):
        """Perform one step of line integration convolution."""
        # Get current position in grid
        i, j, k = wp.tid()
        shape = wp.vec3i(noise.shape[1], noise.shape[2], noise.shape[3])
        
        # Initialize accumulation
        sum_val = wp.float32(0.0)
        sum_weight = wp.float32(0.0)
        
        # Get initial position and sample center point first
        pos = wp.vec3(float(i), float(j), float(k))
        center_sample = noise[0, i, j, k]
        center_weight = LineIntegrationConvolution.kernel_weight(0.0, nr_steps)
        sum_val += center_sample * center_weight
        sum_weight += center_weight
        
        # Get initial vector for forward integration
        vec = LineIntegrationConvolution.sample_vector_field(vector_field, pos, shape)
        magnitude = wp.sqrt(wp.dot(vec, vec))
        if magnitude > 1e-6:
            vec = vec / magnitude
            
            # Forward integration using RK2 (Heun's method)
            pos_forward = pos
            for s in range(1, nr_steps + 1):
                # First RK2 stage
                k1 = vec * step_size
                pos_mid = pos_forward + k1 * wp.float32(0.5)
                
                # Sample vector at midpoint
                vec_mid = LineIntegrationConvolution.sample_vector_field(vector_field, pos_mid, shape)
                magnitude = wp.sqrt(wp.dot(vec_mid, vec_mid))
                if magnitude < 1e-6:
                    break
                vec_mid = vec_mid / magnitude
                
                # Second RK2 stage
                k2 = vec_mid * step_size
                pos_forward = pos_forward + k2
                
                # Sample noise at new position
                sample = LineIntegrationConvolution.sample_noise_trilinear(noise, pos_forward, shape)
                weight = LineIntegrationConvolution.kernel_weight(float(s), nr_steps)
                
                sum_val += sample * weight
                sum_weight += weight
                
                # Sample new vector for next step
                vec = LineIntegrationConvolution.sample_vector_field(vector_field, pos_forward, shape)
                magnitude = wp.sqrt(wp.dot(vec, vec))
                if magnitude < 1e-6:
                    break
                vec = vec / magnitude
        
        # Reset position and get vector for backward integration
        pos_backward = pos
        vec = LineIntegrationConvolution.sample_vector_field(vector_field, pos, shape)
        magnitude = wp.sqrt(wp.dot(vec, vec))
        if magnitude > 1e-6:
            vec = -vec / magnitude  # Note the negative for backward integration
            
            # Backward integration using RK2 (Heun's method)
            for s in range(1, nr_steps + 1):
                # First RK2 stage
                k1 = vec * step_size
                pos_mid = pos_backward + k1 * wp.float32(0.5)
                
                # Sample vector at midpoint
                vec_mid = LineIntegrationConvolution.sample_vector_field(vector_field, pos_mid, shape)
                magnitude = wp.sqrt(wp.dot(vec_mid, vec_mid))
                if magnitude < 1e-6:
                    break
                vec_mid = vec_mid / magnitude
                
                # Second RK2 stage
                k2 = vec_mid * step_size
                pos_backward = pos_backward + k2
                
                # Sample noise at new position
                sample = LineIntegrationConvolution.sample_noise_trilinear(noise, pos_backward, shape)
                weight = LineIntegrationConvolution.kernel_weight(float(s), nr_steps)
                
                sum_val += sample * weight
                sum_weight += weight
                
                # Sample new vector for next step
                vec = LineIntegrationConvolution.sample_vector_field(vector_field, pos_backward, shape)
                magnitude = wp.sqrt(wp.dot(vec, vec))
                if magnitude < 1e-6:
                    break
                vec = -vec / magnitude  # Keep negative for backward integration
        
        # Write output with enhanced contrast
        if sum_weight > 0.0:
            val = sum_val / sum_weight
            # Push values away from 0.5 to enhance contrast
            val = val * wp.float32(2.0) - wp.float32(1.0)  # Map to [-1, 1]
            val = wp.sign(val) * wp.pow(wp.abs(val), wp.float32(0.5))  # Power law with < 1 exponent increases contrast
            val = (val + wp.float32(1.0)) * wp.float32(0.5)  # Map back to [0, 1]
            line_integral[0, i, j, k] = wp.clamp(val, wp.float32(0.0), wp.float32(1.0))
        else:
            line_integral[0, i, j, k] = noise[0, i, j, k]
    
    def __call__(
        self,
        vector_field: wp.array4d(dtype=wp.float32),
        noise: wp.array4d(dtype=wp.float32),
        line_integral: wp.array4d(dtype=wp.float32),
        step_size: float = 0.5,
        nr_steps: int = 20
    ) -> wp.array4d(dtype=wp.float32):
        """
        Perform one step of line integration convolution.
        
        Parameters
        ----------
        vector_field : wp.array4d
            Vector field components (3, nx, ny, nz)
        noise : wp.array4d
            Input noise field (1, nx, ny, nz)
        line_integral : wp.array4d
            Output line integral field (1, nx, ny, nz), must be same shape as input noise
        step_size : float
            Integration step size in grid units
        nr_steps : int
            Number of integration steps in each direction
            
        Returns
        -------
        wp.array4d
            Reference to output_line_integral array
        """
        # Verify input and output shapes match
        assert noise.shape == line_integral.shape, "Input and output line integral arrays must have the same shape"
        
        wp.launch(
            kernel=self._lic_step,
            dim=(noise.shape[1], noise.shape[2], noise.shape[3]),
            inputs=[
                vector_field,
                noise,
                line_integral,
                step_size,
                nr_steps
            ]
        )
        
        return line_integral 