import warp as wp
from build123d import Compound
from stl import mesh as np_mesh
import numpy as np
import tempfile

from pumpkin_pulse.operator.operator import Operator

class StlToMesh(Operator):

    def __call__(
        self,
        file_path: str,
    ):

        # Load the mesh
        mesh = np_mesh.Mesh.from_file(file_path)
        mesh_points = mesh.points.reshape(-1, 3)
        mesh_indices = np.arange(mesh_points.shape[0])
        mesh = wp.Mesh(
            points=wp.array(mesh_points, dtype=wp.vec3),
            indices=wp.array(mesh_indices, dtype=int),
        )
        return mesh
