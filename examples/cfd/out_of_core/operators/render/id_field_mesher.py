"""
Operator for converting ID fields to triangle meshes.

This operator takes an ID field and generates a triangle mesh where faces are only
created between voxels of different IDs. Each voxel face is represented by two triangles.
"""

import warp as wp
import numpy as np
from typing import Tuple

from pumpkin_pulse.operator.operator import Operator

class IDFieldMesher(Operator):
    """
    Operator that converts an ID field to a triangle mesh.
    
    This operator examines each voxel in the ID field and generates triangle faces
    between voxels of different IDs. Each face consists of two triangles with proper
    winding order for correct face culling.
    """
    
    @staticmethod
    @wp.func
    def _get_face_vertices(
        pos: wp.vec3i,
        face: int,
        origin: wp.vec3,
        spacing: wp.vec3
    ):
        """
        Get the four vertices for a given face of a voxel.
        
        Parameters
        ----------
        pos : wp.vec3i
            Voxel position (i,j,k)
        face : int
            Face index (0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z)
        origin : wp.vec3
            Grid origin in world space
        spacing : wp.vec3
            Grid spacing in world space
            
        Returns
        -------
        tuple(wp.vec3, wp.vec3, wp.vec3, wp.vec3)
            Four vertices in counter-clockwise order for the face
        """
        # Convert index space to world space
        base_pos = wp.vec3(
            float(pos[0]) * spacing[0] + origin[0],
            float(pos[1]) * spacing[1] + origin[1],
            float(pos[2]) * spacing[2] + origin[2]
        )
        
        # Get vertices based on face direction
        if face == 0:  # +x face
            v0 = base_pos + wp.vec3(spacing[0], 0.0, 0.0)
            v1 = base_pos + wp.vec3(spacing[0], spacing[1], 0.0)
            v2 = base_pos + wp.vec3(spacing[0], spacing[1], spacing[2])
            v3 = base_pos + wp.vec3(spacing[0], 0.0, spacing[2])
        elif face == 1:  # -x face
            v0 = base_pos
            v1 = base_pos + wp.vec3(0.0, 0.0, spacing[2])
            v2 = base_pos + wp.vec3(0.0, spacing[1], spacing[2])
            v3 = base_pos + wp.vec3(0.0, spacing[1], 0.0)
        elif face == 2:  # +y face
            v0 = base_pos + wp.vec3(0.0, spacing[1], 0.0)
            v1 = base_pos + wp.vec3(spacing[0], spacing[1], 0.0)
            v2 = base_pos + wp.vec3(spacing[0], spacing[1], spacing[2])
            v3 = base_pos + wp.vec3(0.0, spacing[1], spacing[2])
        elif face == 3:  # -y face
            v0 = base_pos
            v1 = base_pos + wp.vec3(0.0, 0.0, spacing[2])
            v2 = base_pos + wp.vec3(spacing[0], 0.0, spacing[2])
            v3 = base_pos + wp.vec3(spacing[0], 0.0, 0.0)
        elif face == 4:  # +z face
            v0 = base_pos + wp.vec3(0.0, 0.0, spacing[2])
            v1 = base_pos + wp.vec3(spacing[0], 0.0, spacing[2])
            v2 = base_pos + wp.vec3(spacing[0], spacing[1], spacing[2])
            v3 = base_pos + wp.vec3(0.0, spacing[1], spacing[2])
        else:  # -z face
            v0 = base_pos
            v1 = base_pos + wp.vec3(0.0, spacing[1], 0.0)
            v2 = base_pos + wp.vec3(spacing[0], spacing[1], 0.0)
            v3 = base_pos + wp.vec3(spacing[0], 0.0, 0.0)
            
        return v0, v1, v2, v3
    
    @wp.kernel
    def _generate_mesh(
        id_field: wp.array4d(dtype=wp.int16),
        color_mapping: wp.array2d(dtype=wp.uint8),
        points: wp.array2d(dtype=wp.float32),
        indices: wp.array1d(dtype=wp.int32),
        vertex_colors: wp.array2d(dtype=wp.float32),
        vertex_counter: wp.array(dtype=wp.int32),
        index_counter: wp.array(dtype=wp.int32),
        origin: wp.vec3,
        spacing: wp.vec3,
    ):
        """Generate mesh vertices, indices, and colors."""
        i, j, k, face = wp.tid()
        
        # Center voxel
        i += 1
        j += 1
        k += 1
        
        # Get current voxel's ID
        current_id = id_field[0, i, j, k]
        
        # Skip if empty
        if current_id == 0:
            return
            
        # Get neighbor ID based on face
        neighbor_id = current_id  # Default to same ID (no face needed)
        if face == 0 and i + 1 < id_field.shape[1]:      # +x
            neighbor_id = id_field[0, i + 1, j, k]
        elif face == 1 and i > 0:                        # -x
            neighbor_id = id_field[0, i - 1, j, k]
        elif face == 2 and j + 1 < id_field.shape[2]:    # +y
            neighbor_id = id_field[0, i, j + 1, k]
        elif face == 3 and j > 0:                        # -y
            neighbor_id = id_field[0, i, j - 1, k]
        elif face == 4 and k + 1 < id_field.shape[3]:    # +z
            neighbor_id = id_field[0, i, j, k + 1]
        elif face == 5 and k > 0:                        # -z
            neighbor_id = id_field[0, i, j, k - 1]
            
        # Only create face if IDs are different
        if current_id == neighbor_id:
            return
            
        # Get color for current ID
        r = float(color_mapping[current_id, 0]) / 255.0
        g = float(color_mapping[current_id, 1]) / 255.0
        b = float(color_mapping[current_id, 2]) / 255.0
        color = wp.vec3(r, g, b)
        
        # Get vertices for this face
        pos = wp.vec3i(i, j, k)
        v0, v1, v2, v3 = IDFieldMesher._get_face_vertices(pos, face, origin, spacing)
        
        # Add vertices and triangles
        idx = wp.atomic_add(vertex_counter, 0, 4)
        tri_idx = wp.atomic_add(index_counter, 0, 6)

        
        # Add vertices
        for d in range(3):
            points[idx + 0, d] = v0[d]
            points[idx + 1, d] = v1[d]
            points[idx + 2, d] = v2[d]
            points[idx + 3, d] = v3[d]
        
        # Add colors
        for d in range(3):
            vertex_colors[idx + 0, d] = color[d]
            vertex_colors[idx + 1, d] = color[d]
            vertex_colors[idx + 2, d] = color[d]
            vertex_colors[idx + 3, d] = color[d]
        
        # Add indices for two triangles
        indices[tri_idx + 0] = idx + 0
        indices[tri_idx + 1] = idx + 1
        indices[tri_idx + 2] = idx + 2
        indices[tri_idx + 3] = idx + 0
        indices[tri_idx + 4] = idx + 2
        indices[tri_idx + 5] = idx + 3
    
    def __call__(
        self,
        id_field: wp.array4d,
        color_mapping: wp.array2d,
        points: wp.array2d,
        indices: wp.array1d,
        vertex_colors: wp.array2d,
        vertex_counter: wp.array,
        index_counter: wp.array,
        origin: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        spacing: wp.vec3 = wp.vec3(1.0, 1.0, 1.0),
    ):
        """
        Convert ID field to triangle mesh.
        
        Parameters
        ----------
        id_field : wp.array4d
            Input ID field of shape (1, size, size, size)
        color_mapping : wp.array2d
            Color mapping array of shape (num_materials, 3) with RGB colors
        points : wp.array2d
            Output vertex positions (num_vertices, 3)
        indices : wp.array1d
            Output triangle indices (num_triangles * 3)
        vertex_colors : wp.array2d
            Output vertex colors (num_vertices, 3)
        vertex_counter : wp.array
            Counter for number of vertices added
        index_counter : wp.array
            Counter for number of indices added
        origin : wp.vec3, optional
            Grid origin in world space, by default (0,0,0)
        spacing : wp.vec3, optional
            Grid spacing in world space, by default (1,1,1)
        """
        # Zero out counters
        vertex_counter.zero_()
        index_counter.zero_()
        
        # Launch kernel with 6 threads per voxel (one for each face)
        wp.launch(
            kernel=self._generate_mesh,
            dim=(id_field.shape[1]-2, id_field.shape[2]-2, id_field.shape[3]-2, 6),
            inputs=[
                id_field,
                color_mapping,
                points,
                indices,
                vertex_colors,
                vertex_counter,
                index_counter,
                origin,
                spacing,
            ],
        )