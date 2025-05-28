"""
Renderer for visualizing ID fields using solid voxels.

This renderer takes an ID field and color mapping, and renders the voxels using
ray marching. Each voxel is rendered as a solid cube with its color determined
by the ID field and color mapping.
"""

import warp as wp
import math
from typing import Tuple
from pumpkin_pulse.operator.operator import Operator

class IDFieldRenderer(Operator):
    """
    Operator for rendering ID fields as solid voxels.
    
    This operator takes an ID field and color mapping, and renders each voxel
    as a solid cube. The color of each voxel is determined by looking up its
    ID in the color mapping.
    """
    
    @staticmethod
    @wp.func
    def create_view_matrix(eye: wp.vec3, target: wp.vec3, up: wp.vec3) -> wp.mat44:
        """Create a view matrix from camera parameters."""
        # Forward vector points from eye to target
        forward = wp.normalize(target - eye)
        
        # Right vector
        right = wp.normalize(wp.cross(forward, up))
        
        # Recompute up vector to ensure orthogonality
        up = wp.normalize(wp.cross(right, forward))
        
        # Construct view matrix
        return wp.mat44(
            right[0], up[0], -forward[0], eye[0],
            right[1], up[1], -forward[1], eye[1],
            right[2], up[2], -forward[2], eye[2],
            wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0)
        )

    @staticmethod
    @wp.func
    def normal_based_shading(
        normal: wp.vec3,
        view_dir: wp.vec3,
        base_color: wp.vec3,
        ambient_intensity: float,
        edge_sharpness: float,
    ) -> wp.vec3:
        """Compute lighting with simple normal-based shading."""
        # Normalize vectors
        n = wp.normalize(normal)
        v = wp.normalize(view_dir)
        
        # Check if normal is facing away from view direction
        n_dot_v = wp.dot(n, v)
        if n_dot_v < wp.float32(0.0):
            # Flip normal if it's facing away
            n = wp.vec3(-n[0], -n[1], -n[2])
            n_dot_v = -n_dot_v
        
        # Use configurable falloff for edge definition
        diffuse_factor = wp.pow(n_dot_v, edge_sharpness)
        
        # Add ambient light to prevent completely black areas
        light = wp.vec3(ambient_intensity + diffuse_factor)
        
        # Apply lighting to base color using component-wise multiplication
        return wp.cw_mul(base_color, light)

    @staticmethod
    @wp.func
    def ray_box_intersection(
        ro: wp.vec3,  # Ray origin in world space
        rd: wp.vec3,  # Ray direction in world space
        box_min: wp.vec3,  # Box minimum corner in world space
        box_max: wp.vec3,  # Box maximum corner in world space
        max_t: float,  # Maximum ray distance to check
    ):
        """
        Compute intersection between a ray and an axis-aligned box in world space.
        Uses the slab method with improved numerical stability and edge case handling.
        
        Parameters
        ----------
        ro : wp.vec3
            Ray origin in world space
        rd : wp.vec3
            Ray direction (normalized) in world space
        box_min : wp.vec3
            Minimum corner of box in world space
        box_max : wp.vec3
            Maximum corner of box in world space
        max_t : float
            Maximum distance along ray to check for intersection
            
        Returns
        -------
        bool : Whether the ray hits the box within max_t distance
        wp.vec3 : Normal at intersection point (in world space)
        float : Distance to intersection
        """
        # Initialize outputs
        hit = wp.bool(False)
        normal = wp.vec3(0.0, 0.0, 0.0)
        t = max_t
        
        # Small epsilon for numerical stability
        eps = wp.float32(1.0e-7)

        # Calculate inverse ray direction and handle division by zero
        inv_rd = wp.vec3(
            1.0 / rd[0],
            1.0 / rd[1],
            1.0 / rd[2]
        )
        
        # Calculate intersections with axis-aligned slabs
        t1 = (box_min[0] - ro[0]) * inv_rd[0]
        t2 = (box_max[0] - ro[0]) * inv_rd[0]
        t3 = (box_min[1] - ro[1]) * inv_rd[1]
        t4 = (box_max[1] - ro[1]) * inv_rd[1]
        t5 = (box_min[2] - ro[2]) * inv_rd[2]
        t6 = (box_max[2] - ro[2]) * inv_rd[2]
        
        # Find entry and exit points for each axis
        tmin = wp.max(
            wp.max(wp.min(t1, t2), wp.min(t3, t4)),
            wp.min(t5, t6)
        )
        tmax = wp.min(
            wp.min(wp.max(t1, t2), wp.max(t3, t4)),
            wp.max(t5, t6)
        )
        
        # Check if intersection occurs within bounds
        if tmax >= 0.0 and tmin <= tmax and tmin < max_t:
            hit = True
            if tmin > 0.0:
                t = tmin
            else:
                t = tmax
            
            # Calculate hit position
            hit_pos = ro + rd * t
            
            # Calculate normal based on which face was hit
            # Use relative position and small epsilon for robustness
            rel_pos = (hit_pos - box_min) / (box_max - box_min)
            
            if wp.abs(rel_pos[0]) < eps:
                normal = wp.vec3(-1.0, 0.0, 0.0)
            elif wp.abs(rel_pos[0] - 1.0) < eps:
                normal = wp.vec3(1.0, 0.0, 0.0)
            elif wp.abs(rel_pos[1]) < eps:
                normal = wp.vec3(0.0, -1.0, 0.0)
            elif wp.abs(rel_pos[1] - 1.0) < eps:
                normal = wp.vec3(0.0, 1.0, 0.0)
            elif wp.abs(rel_pos[2]) < eps:
                normal = wp.vec3(0.0, 0.0, -1.0)
            else:
                normal = wp.vec3(0.0, 0.0, 1.0)
            
            # Ensure normal points against ray direction
            if wp.dot(normal, rd) > 0.0:
                normal = wp.vec3(-normal[0], -normal[1], -normal[2])
        
        return hit, normal, t

    @wp.kernel
    def _render_id_field(
        id_field: wp.array4d(dtype=wp.int16),
        color_mapping: wp.array2d(dtype=wp.uint8),
        pixel_buffer: wp.array3d(dtype=wp.float32),
        depth_buffer: wp.array2d(dtype=wp.float32),
        camera_pos: wp.vec3f,
        camera_target: wp.vec3f,
        camera_up: wp.vec3f,
        origin: wp.vec3f,
        spacing: wp.vec3f,
        fov_degrees: float,
        ambient_intensity: float,
        edge_sharpness: float,
    ):
        """Render the ID field using ray marching."""
        # Get pixel coordinates
        i, j = wp.tid()
        height = pixel_buffer.shape[0]
        width = pixel_buffer.shape[1]

        # Get Shape of ID Field
        shape = wp.vec3i(id_field.shape[1], id_field.shape[2], id_field.shape[3])
        
        # Convert FOV to radians and calculate image plane parameters
        aspect = float(width) / float(height)
        fov = math.radians(fov_degrees)
        tan_fov = math.tan(fov * wp.float32(0.5))
        
        # Convert to NDC space with proper FOV
        sx = (wp.float32(2.0) * float(j) / float(width) - wp.float32(1.0)) * aspect * tan_fov
        sy = (wp.float32(1.0) - wp.float32(2.0) * float(i) / float(height)) * tan_fov
        
        # Create view matrix
        view = IDFieldRenderer.create_view_matrix(camera_pos, camera_target, camera_up)
        
        # Create ray in camera space
        ray_dir = wp.normalize(wp.vec3(sx, sy, wp.float32(-1.0)))
        
        # Transform ray to world space
        ro = camera_pos
        rd = wp.vec3(
            ray_dir[0] * view[0, 0] + ray_dir[1] * view[0, 1] + ray_dir[2] * view[0, 2],
            ray_dir[0] * view[1, 0] + ray_dir[1] * view[1, 1] + ray_dir[2] * view[1, 2],
            ray_dir[0] * view[2, 0] + ray_dir[1] * view[2, 1] + ray_dir[2] * view[2, 2]
        )
        rd = wp.normalize(rd)

        # Get current depth
        current_depth = depth_buffer[i, j]

        # Check if ray hits box
        hit, normal, t = IDFieldRenderer.ray_box_intersection(
            ro,
            rd,
            origin,
            origin + wp.vec3(spacing[0] * wp.float32(shape[0]), spacing[1] * wp.float32(shape[1]), spacing[2] * wp.float32(shape[2])),
            current_depth
        )

        # If ray does not hit box, return
        if not hit:
            return
        
        # Push ray origin to intersection point
        ro = ro + rd * t

        # Get step size
        step_size = wp.min(spacing[0], wp.min(spacing[1], spacing[2])) / wp.float32(2.0)

        # Get maximum distance to travel
        max_t = wp.length(wp.vec3(spacing[0] * wp.float32(shape[0]), spacing[1] * wp.float32(shape[1]), spacing[2] * wp.float32(shape[2])))
        nr_steps = wp.int32(
            wp.ceil(max_t / step_size)
        )

        # Store
        ray_hit = wp.bool(False)
        ray_normal = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
        ray_distance = wp.float32(0.0)
        color = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))

        # March through voxels
        for step in range(nr_steps):

            # Get current voxel index
            vi_current = wp.int32(wp.floor((ro[0] - origin[0]) / spacing[0]))
            vj_current = wp.int32(wp.floor((ro[1] - origin[1]) / spacing[1]))
            vk_current = wp.int32(wp.floor((ro[2] - origin[2]) / spacing[2]))

            # Check if voxel is out of bounds
            if vi_current < 0 or vi_current >= shape[0] or vj_current < 0 or vj_current >= shape[1] or vk_current < 0 or vk_current >= shape[2]:
                continue

            # Get current voxel ID
            current_id = id_field[0, vi_current, vj_current, vk_current]

            # Check if voxel is empty
            if current_id > 0:

                # Test intersection with voxel
                ray_hit, ray_normal, ray_distance = IDFieldRenderer.ray_box_intersection(
                    ro,
                    rd,
                    wp.vec3(
                        origin[0] + wp.float32(vi_current) * spacing[0],
                        origin[1] + wp.float32(vj_current) * spacing[1],
                        origin[2] + wp.float32(vk_current) * spacing[2]
                    ),
                    wp.vec3(
                        origin[0] + wp.float32(vi_current + 1) * spacing[0],
                        origin[1] + wp.float32(vj_current + 1) * spacing[1],
                        origin[2] + wp.float32(vk_current + 1) * spacing[2]
                    ),
                    step_size
                )

                # Get final ray origin
                ro = ro + rd * ray_distance

                # Get color from mapping
                color = wp.vec3(
                    float(color_mapping[current_id, 0]) / wp.float32(255.0),
                    float(color_mapping[current_id, 1]) / wp.float32(255.0),
                    float(color_mapping[current_id, 2]) / wp.float32(255.0)
                )

                # Break
                break

            # Get next voxel index
            vi_next = wp.int32(wp.floor((ro[0] - origin[0] + step_size * rd[0]) / spacing[0]))
            vj_next = wp.int32(wp.floor((ro[1] - origin[1] + step_size * rd[1]) / spacing[1]))
            vk_next = wp.int32(wp.floor((ro[2] - origin[2] + step_size * rd[2]) / spacing[2]))

            # Check if voxel is out of bounds
            if vi_next < 0 or vi_next >= shape[0] or vj_next < 0 or vj_next >= shape[1] or vk_next < 0 or vk_next >= shape[2]:
                continue

            # Get next voxel ID
            next_id = id_field[0, vi_next, vj_next, vk_next]

            # Check if voxel is empty
            if next_id > 0:

                # Test intersection with voxel
                ray_hit, ray_normal, ray_distance = IDFieldRenderer.ray_box_intersection(
                    ro,
                    rd,
                    wp.vec3(
                        origin[0] + wp.float32(vi_next) * spacing[0],
                        origin[1] + wp.float32(vj_next) * spacing[1],
                        origin[2] + wp.float32(vk_next) * spacing[2]
                    ),
                    wp.vec3(
                        origin[0] + wp.float32(vi_next + 1) * spacing[0],
                        origin[1] + wp.float32(vj_next + 1) * spacing[1],
                        origin[2] + wp.float32(vk_next + 1) * spacing[2]
                    ),
                    step_size
                )

                # Get final ray origin
                ro = ro + rd * ray_distance 

                # Get color from mapping
                color = wp.vec3(
                    float(color_mapping[next_id, 0]) / wp.float32(255.0),
                    float(color_mapping[next_id, 1]) / wp.float32(255.0),
                    float(color_mapping[next_id, 2]) / wp.float32(255.0)
                )

                # Break
                break

            # Push ray
            ro = ro + rd * step_size

        if ray_hit:

            ## Apply shading
            #color = IDFieldRenderer.normal_based_shading(
            #    normal=ray_normal,
            #    view_dir=rd,
            #    base_color=color,
            #    ambient_intensity=ambient_intensity,
            #    edge_sharpness=edge_sharpness
            #)

            # Write results
            pixel_buffer[i, j, 0] = color[0]
            pixel_buffer[i, j, 1] = color[1]
            pixel_buffer[i, j, 2] = color[2]
            pixel_buffer[i, j, 3] = wp.float32(1.0)
            depth_buffer[i, j] = wp.length(ro - camera_pos)
       
    def __call__(
        self,
        id_field: wp.array4d,  # Shape: (1, size, size, size)
        color_mapping: wp.array2d,  # Shape: (num_materials + 1, 3)
        pixel_buffer: wp.array3d,  # Shape: (height, width, 4) RGBA
        depth_buffer: wp.array2d,  # Shape: (height, width)
        camera_pos: wp.vec3f,
        camera_target: wp.vec3f = wp.vec3f(0.0, 0.0, 0.0),
        camera_up: wp.vec3f = wp.vec3f(0.0, 1.0, 0.0),
        origin: wp.vec3f = wp.vec3f(0.0, 0.0, 0.0),
        spacing: wp.vec3f = wp.vec3f(1.0, 1.0, 1.0),
        fov_degrees: float = 60.0,
        ambient_intensity: float = 0.1,
        edge_sharpness: float = 2.0,
    ):
        """
        Render an ID field as solid voxels.
        
        Parameters
        ----------
        id_field : wp.array4d
            4D array of material IDs (1, size, size, size)
        color_mapping : wp.array2d
            RGB colors for each material ID (num_materials + 1, 3)
        pixel_buffer : wp.array3d
            Output pixel buffer (height, width, 4) RGBA
        depth_buffer : wp.array2d
            Output depth buffer (height, width)
        camera_pos : wp.vec3f
            Camera position in world space
        camera_target : wp.vec3f
            Point the camera is looking at
        camera_up : wp.vec3f
            Camera up vector
        origin : wp.vec3f
            Origin of the voxel grid in world space
        spacing : wp.vec3f
            Size of each voxel in world space
        fov_degrees : float
            Field of view in degrees
        ambient_intensity : float
            Intensity of ambient light (0.0-1.0)
        edge_sharpness : float
            Controls edge definition (lower values = softer edges)
            
        Returns
        -------
        tuple[wp.array3d, wp.array2d]
            Updated pixel and depth buffers
        """
        # Launch kernel
        wp.launch(
            self._render_id_field,
            dim=(pixel_buffer.shape[0], pixel_buffer.shape[1]),
            inputs=[
                id_field,
                color_mapping,
                pixel_buffer,
                depth_buffer,
                camera_pos,
                camera_target,
                camera_up,
                origin,
                spacing,
                fov_degrees,
                ambient_intensity,
                edge_sharpness,
            ],
        )
        
        return pixel_buffer, depth_buffer 