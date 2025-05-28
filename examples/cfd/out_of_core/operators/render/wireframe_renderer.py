import warp as wp
from pumpkin_pulse.operator.operator import Operator
import math
import numpy as np

class WireframeRenderer(Operator):
    """
    Operator for rendering wireframes using line rasterization.
    
    This operator takes an array of edges (pairs of 3D points) and renders them as wireframe,
    with configurable line color and thickness.
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
        
        # Third row - match MeshRenderer
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
        
        # Store actual view space z for depth test (not normalized)
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
    def draw_line(x0: wp.int32,
                y0: wp.int32,
                x1: wp.int32,
                y1: wp.int32,
                z0: wp.float32,
                z1: wp.float32,
                pixel_buffer: wp.array3d(dtype=wp.float32),
                depth_buffer: wp.array2d(dtype=wp.float32),
                line_color: wp.vec3,
                line_thickness: wp.float32):
        """Draw a line using Bresenham's algorithm with depth testing."""
        dx = wp.abs(x1 - x0)
        dy = wp.abs(y1 - y0)
        
        # Handle steep lines by swapping x and y
        if dy > dx:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            dx, dy = dy, dx
            swapped = True
        else:
            swapped = False
        
        # Ensure line is drawn from left to right
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            z0, z1 = z1, z0
        
        # Calculate y step direction
        if y0 < y1:
            ystep = wp.int32(1)
        else:
            ystep = wp.int32(-1)
        
        # Initial error term
        error = dx / wp.int32(2)
        y = y0
        
        # Draw the line
        for x in range(x0, x1 + 1):
            # Simple linear interpolation for z
            run = wp.float32(x1 - x0)
            if run == 0.0:
                t = wp.float32(0.0)
            else:
                t = wp.float32(x - x0) / run
            
            # Interpolate z value
            z = z0 * (wp.float32(1.0) - t) + z1 * t
            
            # Draw the point (handle steep case)
            if swapped:
                px = y
                py = x
            else:
                px = x
                py = y

            # Check bounds and draw with thickness
            half_thickness = wp.int32(line_thickness / wp.float32(2.0))
            for offset in range(-half_thickness, half_thickness + 1):
                draw_x = px
                draw_y = py + offset
                
                if (draw_x >= 0 and draw_x < pixel_buffer.shape[1] and 
                    draw_y >= 0 and draw_y < pixel_buffer.shape[0]):
                    if z < depth_buffer[draw_y, draw_x]:
                        depth_buffer[draw_y, draw_x] = z
                        pixel_buffer[draw_y, draw_x, 0] = line_color[0]
                        pixel_buffer[draw_y, draw_x, 1] = line_color[1]
                        pixel_buffer[draw_y, draw_x, 2] = line_color[2]
                        pixel_buffer[draw_y, draw_x, 3] = wp.float32(1.0)
            
            # Update error term and y coordinate
            error = error - dy
            if error < 0:
                y = y + ystep
                error = error + dx
    
    @wp.kernel
    def _rasterize_edges(edges: wp.array2d(dtype=wp.float32),
                        pixel_buffer: wp.array3d(dtype=wp.float32),
                        depth_buffer: wp.array2d(dtype=wp.float32),
                        camera_pos: wp.vec3,
                        camera_target: wp.vec3,
                        camera_up: wp.vec3,
                        fov_degrees: float,
                        near: float,
                        far: float,
                        line_color: wp.vec3,
                        line_thickness: wp.float32):
        """Kernel to rasterize edges."""
        # Get thread ID
        tid = wp.tid()
        
        # Get edge start and end points
        v0 = wp.vec3(edges[tid, 0], edges[tid, 1], edges[tid, 2])
        v1 = wp.vec3(edges[tid, 3], edges[tid, 4], edges[tid, 5])
        
        # Create view and projection matrices
        view_matrix = WireframeRenderer.create_view_matrix(camera_pos, camera_target, camera_up)
        aspect = float(pixel_buffer.shape[1]) / float(pixel_buffer.shape[0])
        fov = wp.float32(fov_degrees) * wp.float32(3.14159) / wp.float32(180.0)
        proj_matrix = WireframeRenderer.create_projection_matrix(fov, aspect, near, far)
        
        # Project points to screen space
        p0 = WireframeRenderer.project_point(v0, view_matrix, proj_matrix, float(pixel_buffer.shape[1]), float(pixel_buffer.shape[0]))
        p1 = WireframeRenderer.project_point(v1, view_matrix, proj_matrix, float(pixel_buffer.shape[1]), float(pixel_buffer.shape[0]))
        
        # Draw line if both points are in front of the near plane (-z is forward)
        if p0[3] < wp.float32(0.0) and p1[3] < wp.float32(0.0):
            WireframeRenderer.draw_line(
                wp.int32(p0[0]),
                wp.int32(p0[1]),
                wp.int32(p1[0]),
                wp.int32(p1[1]),
                -p0[3],  # Negate z for proper depth comparison
                -p1[3],  # Negate z for proper depth comparison
                pixel_buffer,
                depth_buffer,
                line_color,
                line_thickness
            )

    def __call__(self,
                edges: wp.array2d,  # Shape: (num_edges, 6) for [x0,y0,z0, x1,y1,z1]
                pixel_buffer: wp.array3d,
                depth_buffer: wp.array2d,
                camera_pos: wp.vec3,
                camera_target: wp.vec3,
                camera_up: wp.vec3,
                fov_degrees: float = 60.0,
                near: float = 0.1,
                far: float = 100.0,
                line_color: wp.vec3 = wp.vec3(1.0, 1.0, 1.0),
                line_thickness: float = 1.0) -> tuple[wp.array3d, wp.array2d]:
        """
        Render a wireframe from a list of edges.
        
        Parameters
        ----------
        edges : wp.array2d
            Array of edges, shape (num_edges, 6) where each row is [x0,y0,z0, x1,y1,z1]
            representing the start and end points of each edge
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
        near : float
            Distance to the near clipping plane
        far : float
            Distance to the far clipping plane
        line_color : wp.vec3
            Color of the wireframe lines (RGB)
        line_thickness : float
            Thickness of the lines in pixels
            
        Returns
        -------
        tuple[wp.array3d, wp.array2d]
            Updated pixel and depth buffers
        """
        # Launch kernel with one thread per edge
        wp.launch(
            kernel=self._rasterize_edges,
            dim=edges.shape[0],  # Number of edges
            inputs=[
                edges,
                pixel_buffer,
                depth_buffer,
                camera_pos,
                camera_target,
                camera_up,
                fov_degrees,
                wp.float32(near),
                wp.float32(far),
                line_color,
                wp.float32(line_thickness)
            ]
        )
        
        return pixel_buffer, depth_buffer

