import warp as wp
from pumpkin_pulse.operator.operator import Operator
import math

class PointCloudRenderer(Operator):
    """
    Operator for rendering point clouds using point rasterization.
    
    This operator takes an array of 3D points and renders them as screen-space points,
    with configurable colors and point sizes. Each point undergoes perspective projection
    and depth testing.
    """
    
    @staticmethod
    @wp.func
    def create_view_matrix(camera_pos: wp.vec3, camera_target: wp.vec3, camera_up: wp.vec3) -> wp.mat44:
        """Create a view matrix from camera parameters."""
        # Forward vector points from eye to target (negative z-axis in view space)
        forward = wp.normalize(camera_target - camera_pos)
        
        # Right vector
        right = wp.normalize(wp.cross(forward, camera_up))
        
        # Recompute up vector to ensure orthogonality
        up = wp.normalize(wp.cross(right, forward))
        
        # Construct view matrix - note forward is negated to maintain right-handed system
        return wp.mat44(
            right[0], right[1], right[2], -wp.dot(right, camera_pos),
            up[0], up[1], up[2], -wp.dot(up, camera_pos),
            -forward[0], -forward[1], -forward[2], wp.dot(forward, camera_pos),
            wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0)
        )

    @staticmethod
    @wp.func
    def create_projection_matrix(fov_radians: float, aspect: float, near: float, far: float) -> wp.mat44:
        """Create a perspective projection matrix."""
        f = wp.float32(1.0) / wp.tan(fov_radians * wp.float32(0.5))
        
        result = wp.mat44()
        
        # First row
        result[0, 0] = f / aspect
        result[0, 1] = wp.float32(0.0)
        result[0, 2] = wp.float32(0.0)
        result[0, 3] = wp.float32(0.0)
        
        # Second row
        result[1, 0] = wp.float32(0.0)
        result[1, 1] = f
        result[1, 2] = wp.float32(0.0)
        result[1, 3] = wp.float32(0.0)
        
        # Third row
        result[2, 0] = wp.float32(0.0)
        result[2, 1] = wp.float32(0.0)
        result[2, 2] = -(far + near) / (far - near)
        result[2, 3] = -(wp.float32(2.0) * far * near) / (far - near)
        
        # Fourth row
        result[3, 0] = wp.float32(0.0)
        result[3, 1] = wp.float32(0.0)
        result[3, 2] = wp.float32(-1.0)
        result[3, 3] = wp.float32(0.0)
        
        return result

    @staticmethod
    @wp.func
    def project_point(point: wp.vec3, view_matrix: wp.mat44, proj_matrix: wp.mat44, width: float, height: float) -> wp.vec4:
        """Project a 3D point to screen space."""
        # Transform point to view space
        view_pos = wp.vec4(point[0], point[1], point[2], wp.float32(1.0))
        view_pos = view_matrix * view_pos
        
        # Store actual view space z for depth test
        view_z = view_pos[2]
        
        # Apply projection matrix
        clip_pos = proj_matrix * view_pos
        
        # Perspective divide
        if clip_pos[3] != wp.float32(0.0):
            inv_w = wp.float32(1.0) / clip_pos[3]
            clip_pos[0] = clip_pos[0] * inv_w
            clip_pos[1] = clip_pos[1] * inv_w
            clip_pos[2] = clip_pos[2] * inv_w
        
        # Convert to screen coordinates
        screen_pos = wp.vec4()
        screen_pos[0] = (clip_pos[0] + wp.float32(1.0)) * wp.float32(0.5) * width
        screen_pos[1] = (wp.float32(1.0) - (clip_pos[1] + wp.float32(1.0)) * wp.float32(0.5)) * height  # Flip Y
        screen_pos[2] = view_z  # Store actual view space z
        screen_pos[3] = view_z  # Store view space z for depth test
        
        return screen_pos

    @staticmethod
    @wp.func
    def draw_point(center_x: wp.int32,
                  center_y: wp.int32,
                  z: wp.float32,
                  point_size: wp.float32,
                  color: wp.vec3,
                  pixel_buffer: wp.array3d(dtype=wp.float32),
                  depth_buffer: wp.array2d(dtype=wp.float32)):
        """Draw a point as a square with the given size and color."""
        # Calculate point bounds
        half_size = wp.int32(point_size / wp.float32(2.0))
        min_x = center_x - half_size
        max_x = center_x + half_size
        min_y = center_y - half_size
        max_y = center_y + half_size
        
        # Clamp to screen bounds
        min_x = wp.max(min_x, wp.int32(0))
        max_x = wp.min(max_x, wp.int32(pixel_buffer.shape[1] - 1))
        min_y = wp.max(min_y, wp.int32(0))
        max_y = wp.min(max_y, wp.int32(pixel_buffer.shape[0] - 1))
        
        # Draw point as a filled square
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if z < depth_buffer[y, x]:
                    depth_buffer[y, x] = z
                    pixel_buffer[y, x, 0] = color[0]
                    pixel_buffer[y, x, 1] = color[1]
                    pixel_buffer[y, x, 2] = color[2]
                    pixel_buffer[y, x, 3] = wp.float32(1.0)
    
    @wp.kernel
    def _rasterize_points(points: wp.array2d(dtype=wp.float32),
                         colors: wp.array2d(dtype=wp.float32),
                         pixel_buffer: wp.array3d(dtype=wp.float32),
                         depth_buffer: wp.array2d(dtype=wp.float32),
                         camera_pos: wp.vec3,
                         camera_target: wp.vec3,
                         camera_up: wp.vec3,
                         fov_degrees: float,
                         point_size: float,
                         near: float,
                         far: float):
        """Kernel to rasterize points."""
        # Get thread ID (one per point)
        tid = wp.tid()
        
        # Get point position and color
        point = wp.vec3(points[tid, 0], points[tid, 1], points[tid, 2])
        color = wp.vec3(colors[tid, 0], colors[tid, 1], colors[tid, 2])
        
        # Create view and projection matrices
        view_matrix = PointCloudRenderer.create_view_matrix(camera_pos, camera_target, camera_up)
        aspect = float(pixel_buffer.shape[1]) / float(pixel_buffer.shape[0])
        fov = wp.float32(fov_degrees) * wp.float32(3.14159) / wp.float32(180.0)
        proj_matrix = PointCloudRenderer.create_projection_matrix(fov, aspect, near, far)
        
        # Project point to screen space
        screen_pos = PointCloudRenderer.project_point(point, view_matrix, proj_matrix, 
                                                    float(pixel_buffer.shape[1]), float(pixel_buffer.shape[0]))
        
        # Draw point if it's in front of the camera (-z is forward)
        if screen_pos[3] < wp.float32(0.0):
            PointCloudRenderer.draw_point(
                wp.int32(screen_pos[0]),
                wp.int32(screen_pos[1]),
                -screen_pos[3],  # Negate z for proper depth comparison
                point_size,
                color,
                pixel_buffer,
                depth_buffer
            )

    def __call__(self,
                points: wp.array2d,  # Shape: (num_points, 3) for positions
                colors: wp.array2d,  # Shape: (num_points, 3) for RGB
                pixel_buffer: wp.array3d,
                depth_buffer: wp.array2d,
                camera_pos: wp.vec3,
                camera_target: wp.vec3,
                camera_up: wp.vec3,
                fov_degrees: float = 60.0,
                point_size: float = 1.0,
                near: float = 0.1,
                far: float = 100.0) -> tuple[wp.array3d, wp.array2d]:
        """
        Render a point cloud.
        
        Parameters
        ----------
        points : wp.array2d
            Array of point positions, shape (num_points, 3) for [x,y,z]
        colors : wp.array2d
            Array of point colors, shape (num_points, 3) for RGB
        pixel_buffer : wp.array3d
            Output pixel buffer (height, width, 4) RGBA
        depth_buffer : wp.array2d
            Output depth buffer (height, width)
        camera_pos : wp.vec3
            Camera position in world space
        camera_target : wp.vec3
            Point the camera is looking at
        camera_up : wp.vec3
            Camera up vector
        fov_degrees : float
            Field of view in degrees
        point_size : float
            Size of points in pixels
        near : float
            Distance to the near clipping plane
        far : float
            Distance to the far clipping plane
            
        Returns
        -------
        tuple[wp.array3d, wp.array2d]
            Updated pixel and depth buffers
        """
        # Launch kernel with one thread per point
        wp.launch(
            kernel=self._rasterize_points,
            dim=points.shape[0],  # Number of points
            inputs=[
                points,
                colors,
                pixel_buffer,
                depth_buffer,
                camera_pos,
                camera_target,
                camera_up,
                fov_degrees,
                wp.float32(point_size),
                wp.float32(near),
                wp.float32(far)
            ]
        )
        
        return pixel_buffer, depth_buffer 