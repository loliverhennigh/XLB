import warp as wp
from pumpkin_pulse.operator.operator import Operator
import math
from typing import Any

@wp.struct
class RayHit:
    t_min: wp.float32
    t_max: wp.float32

class VolumeRenderer(Operator):
    """
    Operator for rendering volumetric data using ray marching.
    Assumes volumes are rendered front-to-back and don't intersect.
    
    The volume data is stored in a 4D array where:
    - channels[0:3]: RGB color components
    - channel[3]: Alpha/opacity
    - Remaining dimensions (1,2,3): Spatial grid (i,j,k)
    """
    
    @staticmethod
    @wp.func
    def create_view_matrix(eye: wp.vec3, target: wp.vec3, up: wp.vec3) -> wp.mat44:
        """Create a view matrix from camera parameters."""
        # Forward vector points from eye to target (negative z-axis in view space)
        forward = wp.normalize(target - eye)
        
        # Right vector
        right = wp.normalize(wp.cross(forward, up))
        
        # Recompute up vector to ensure orthogonality
        up = wp.normalize(wp.cross(right, forward))
        
        # Construct view matrix - note forward is negated to maintain right-handed system
        return wp.mat44(
            right[0], up[0], -forward[0], eye[0],
            right[1], up[1], -forward[1], eye[1],
            right[2], up[2], -forward[2], eye[2],
            0.0, 0.0, 0.0, 1.0
        )
    
    @staticmethod
    @wp.func
    def intersect_box(
        ray_origin: wp.vec3,
        ray_dir: wp.vec3,
        box_min: wp.vec3,
        box_max: wp.vec3
    ) -> RayHit:
        """Compute ray intersection with axis-aligned bounding box."""
        hit = RayHit()
        hit.t_min = wp.float32(-1e6)
        hit.t_max = wp.float32(1e6)
        
        for i in range(3):
            if abs(ray_dir[i]) > 1e-6:
                t1 = (box_min[i] - ray_origin[i]) / ray_dir[i]
                t2 = (box_max[i] - ray_origin[i]) / ray_dir[i]
                
                hit.t_min = wp.max(hit.t_min, wp.min(t1, t2))
                hit.t_max = wp.min(hit.t_max, wp.max(t1, t2))
            elif ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                # Ray is parallel to slab and outside box
                hit.t_min = wp.float32(1e6)
                hit.t_max = wp.float32(-1e6)
        
        return hit
    
    @staticmethod
    @wp.func
    def sample_volume(
        volume: wp.array4d(dtype=wp.float32),
        pos: wp.vec3,
        origin: wp.vec3,
        spacing: wp.vec3
    ) -> wp.vec4:
        """Sample volume RGBA at world position."""
        # Convert world position to grid coordinates, accounting for cell-centered values
        # Subtract half spacing to account for cell-centered values
        grid_pos = wp.cw_div(pos - origin, spacing) - wp.vec3(0.5, 0.5, 0.5)
        
        # Get grid dimensions
        nx = wp.int32(volume.shape[1])
        ny = wp.int32(volume.shape[2])
        nz = wp.int32(volume.shape[3])
        
        # Clamp to grid bounds
        i = wp.clamp(wp.int32(grid_pos[0]), wp.int32(0), nx - 1)
        j = wp.clamp(wp.int32(grid_pos[1]), wp.int32(0), ny - 1)
        k = wp.clamp(wp.int32(grid_pos[2]), wp.int32(0), nz - 1)
        
        # Trilinear interpolation
        fx = grid_pos[0] - wp.float32(i)
        fy = grid_pos[1] - wp.float32(j)
        fz = grid_pos[2] - wp.float32(k)
        
        # Sample all RGBA components
        result = wp.vec4(0.0)
        for c in range(4):  # RGBA channels
            # Sample corners with bounds checking
            c000 = volume[c, i, j, k]
            
            if k < nz-1:
                c001 = volume[c, i, j, k+1]
            else:
                c001 = c000
                
            if j < ny-1:
                c010 = volume[c, i, j+1, k]
                if k < nz-1:
                    c011 = volume[c, i, j+1, k+1]
                else:
                    c011 = c010
            else:
                c010 = c000
                c011 = c000
                
            if i < nx-1:
                c100 = volume[c, i+1, j, k]
                if k < nz-1:
                    c101 = volume[c, i+1, j, k+1]
                else:
                    c101 = c100
                    
                if j < ny-1:
                    c110 = volume[c, i+1, j+1, k]
                    if k < nz-1:
                        c111 = volume[c, i+1, j+1, k+1]
                    else:
                        c111 = c110
                else:
                    c110 = c100
                    c111 = c100
            else:
                c100 = c000
                c101 = c000
                c110 = c000
                c111 = c000
            
            # Interpolate along x
            c00 = c000 * (1.0 - fx) + c100 * fx
            c01 = c001 * (1.0 - fx) + c101 * fx
            c10 = c010 * (1.0 - fx) + c110 * fx
            c11 = c011 * (1.0 - fx) + c111 * fx
            
            # Interpolate along y
            c0 = c00 * (1.0 - fy) + c10 * fy
            c1 = c01 * (1.0 - fy) + c11 * fy
            
            # Interpolate along z
            result[c] = c0 * (1.0 - fz) + c1 * fz
            
        return result
    
    @wp.kernel
    def _render_volume(
        volume: wp.array4d(dtype=Any),
        pixel_buffer: wp.array3d(dtype=wp.float32),
        depth_buffer: wp.array2d(dtype=wp.float32),
        origin: wp.vec3,
        spacing: wp.vec3,
        camera_pos: wp.vec3,
        camera_target: wp.vec3,
        camera_up: wp.vec3,
        fov_degrees: float,
        opacity_threshold: float
    ):
        """Ray march through volume."""
        # Get pixel coordinates
        i, j = wp.tid()
        height = pixel_buffer.shape[0]
        width = pixel_buffer.shape[1]

        # Get step size from spacing
        step_size = wp.min(spacing[0], wp.min(spacing[1], spacing[2]))
        
        # Calculate ray direction
        aspect = float(width) / float(height)
        fov = wp.float32(fov_degrees * 3.14159 / 180.0)
        
        # Screen space to NDC
        sx = (2.0 * float(j) / float(width) - 1.0) * aspect * wp.tan(fov * 0.5)
        sy = (1.0 - 2.0 * float(i) / float(height)) * wp.tan(fov * 0.5)
        
        # Create view matrix and get ray direction
        view_matrix = VolumeRenderer.create_view_matrix(camera_pos, camera_target, camera_up)
        ray_dir = wp.normalize(wp.vec3(
            sx * view_matrix[0, 0] + sy * view_matrix[0, 1] - view_matrix[0, 2],
            sx * view_matrix[1, 0] + sy * view_matrix[1, 1] - view_matrix[1, 2],
            sx * view_matrix[2, 0] + sy * view_matrix[2, 1] - view_matrix[2, 2]
        ))
        
        # Calculate grid bounds
        grid_size = wp.vec3(
            float(volume.shape[1]) * spacing[0],
            float(volume.shape[2]) * spacing[1],
            float(volume.shape[3]) * spacing[2]
        )
        
        # Ray-box intersection
        hit = VolumeRenderer.intersect_box(
            camera_pos, ray_dir, origin, origin + grid_size
        )
        
        if hit.t_min < hit.t_max:
            # Initialize accumulated color and opacity
            accum_color = wp.vec3(0.0)
            accum_alpha = wp.float32(0.0)
            
            # Get existing depth from buffer
            existing_depth = depth_buffer[i, j]
            
            # March through volume
            t = hit.t_min
            while t < hit.t_max and t < existing_depth and accum_alpha < opacity_threshold:
                # Get sample position
                pos = camera_pos + ray_dir * t
                
                # Sample RGBA
                sample = VolumeRenderer.sample_volume(volume, pos, origin, spacing)
                
                # Front-to-back compositing
                opacity = (1.0 - accum_alpha) * sample[3]
                accum_color += wp.vec3(sample[0], sample[1], sample[2]) * opacity
                accum_alpha += opacity
                
                t += step_size
            
            # Write results if anything was accumulated
            if accum_alpha > 0.0:
                # Blend with existing color using accumulated alpha
                existing_color = wp.vec3(
                    pixel_buffer[i, j, 0],
                    pixel_buffer[i, j, 1],
                    pixel_buffer[i, j, 2]
                )
                final_color = accum_color + existing_color * (1.0 - accum_alpha)
                
                pixel_buffer[i, j, 0] = final_color[0]
                pixel_buffer[i, j, 1] = final_color[1]
                pixel_buffer[i, j, 2] = final_color[2]
                pixel_buffer[i, j, 3] = accum_alpha + pixel_buffer[i, j, 3] * (1.0 - accum_alpha)
                
                # Update depth only if we accumulated significant opacity
                if accum_alpha > 0.1:  # Threshold to avoid updating depth with very transparent regions
                    depth_buffer[i, j] = wp.min(depth_buffer[i, j], t)

    def __call__(
        self,
        volume: wp.array4d,
        pixel_buffer: wp.array3d,
        depth_buffer: wp.array2d,
        origin: wp.vec3,
        spacing: wp.vec3,
        camera_pos: wp.vec3,
        camera_target: wp.vec3,
        camera_up: wp.vec3,
        fov_degrees: float = 60.0,
        opacity_threshold: float = 0.95
    ) -> tuple[wp.array3d, wp.array2d]:
        """
        Render a volume using ray marching.
        
        Parameters
        ----------
        volume : wp.array4d
            Volume data array (4, nx, ny, nz) where:
            - channels[0:3]: RGB color
            - channel[3]: Alpha/opacity
        pixel_buffer : wp.array3d
            Output pixel buffer (height, width, 4) RGBA
        depth_buffer : wp.array2d
            Output depth buffer (height, width)
        origin : wp.vec3
            Origin of the volume grid in world space
        spacing : wp.vec3
            Grid spacing in each dimension
        camera_pos : wp.vec3
            Camera position in world space
        camera_target : wp.vec3
            Point the camera is looking at
        camera_up : wp.vec3
            Camera up vector
        fov_degrees : float
            Field of view in degrees
        opacity_threshold : float
            Stop ray marching when accumulated opacity exceeds this
            
        Returns
        -------
        tuple[wp.array3d, wp.array2d]
            Updated pixel and depth buffers
        """
        assert volume.shape[0] == 4, "Volume must have 4 channels (RGBA)"
        
        wp.launch(
            kernel=self._render_volume,
            dim=(pixel_buffer.shape[0], pixel_buffer.shape[1]),
            inputs=[
                volume,
                pixel_buffer,
                depth_buffer,
                origin,
                spacing,
                camera_pos,
                camera_target,
                camera_up,
                fov_degrees,
                opacity_threshold
            ]
        )
        
        return pixel_buffer, depth_buffer 