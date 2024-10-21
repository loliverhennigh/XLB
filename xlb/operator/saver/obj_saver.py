# Base class for all render operators
import trimesh as mesh
import numpy as np

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.saver.saver import Saver

class OBJSaver(Saver):
    """
    Saves OBJ files of the current geometry
    """

    def _construct_warp(self):
        return None, None

    def _remove_duplicate_vertices(self, vertices, indices, colors):
        """
        Remove duplicate vertices from the mesh
    
        vertices: np.array
            Array of vertices ((3*N)x3)
        indices: np.array
            Array of indices (Nx3)
        colors: np.array
            Array of colors ((3*N)x4)
        """
    
        # Find unique vertices and get the indices mapping
        unique_vertices, index_map, inverse_indices = np.unique(
            vertices, axis=0, return_index=True, return_inverse=True
        )
    
        # Update indices to refer to the indices of unique vertices
        unique_indices = inverse_indices[indices]
    
        # Extract the colors corresponding to the unique vertices
        unique_colors = colors[index_map]
    
        return unique_vertices, unique_indices, unique_colors


    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        filename,
        vertex_buffer,
        color_buffer=None,
    ):

        #if vertex_buffer.shape[0] == 0:
        #    return

        # Get numpy array from vertex buffer
        vertices = vertex_buffer.numpy()

        # Get numpy array from color buffer
        if color_buffer is None:
            colors = np.ones((vertices.shape[0], 4), dtype=np.uint8) * 255
        elif isinstance(color_buffer, tuple):
            colors = np.ones((vertices.shape[0], 4), dtype=np.uint8) * 255
            colors[:, 0] = color_buffer[0]
            colors[:, 1] = color_buffer[1]
            colors[:, 2] = color_buffer[2]
            colors[:, 3] = color_buffer[3]
        else:
            colors = color_buffer.numpy()

        # Get array of indices
        indices = np.arange(vertices.shape[0], dtype=np.uint32).reshape(-1, 3)

        # Remove duplicate vertices
        vertices, indices, colors = self._remove_duplicate_vertices(vertices, indices, colors)

        # Create a mesh object
        if colors.shape[0] == 0:
            mesh_obj = mesh.Trimesh(vertices=vertices, faces=indices)
        else:
            mesh_obj = mesh.Trimesh(vertices=vertices, faces=indices, vertex_colors=colors)

        # Save the mesh object to a file
        mesh_obj.export(filename)
