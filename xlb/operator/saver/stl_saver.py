# Base class for all render operators
import numpy as np
import trimesh as mesh

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.saver.saver import Saver

class STLSaver(Saver):
    """
    Saves STL files of the current geometry
    """

    def _construct_warp(self):
        return None, None

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, vertex_buffer, filename):

        # Get numpy array from vertex buffer
        vertices = vertex_buffer.numpy()

        # Get array of indices
        indices = np.arange(vertices.shape[0], dtype=np.uint32).reshape(-1, 3)

        # Create mesh object
        mesh_obj = mesh.Trimesh(vertices=vertices, faces=indices)

        # Save mesh object to file
        mesh_obj.export(filename)
