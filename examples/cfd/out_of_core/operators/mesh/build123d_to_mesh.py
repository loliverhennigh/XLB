import warp as wp
from build123d import Compound
from stl import mesh as np_mesh
import numpy as np
import tempfile

from pumpkin_pulse.operator.operator import Operator

class Build123DToMesh(Operator):

    def __call__(
        self,
        compound: Compound,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.1,
    ):

        # Export build123d compound to stl
        with tempfile.NamedTemporaryFile(suffix=".stl") as f:
            compound.export_stl(
                f.name, tolerance=tolerance, angular_tolerance=angular_tolerance
            )
            mesh = np_mesh.Mesh.from_file(f.name)
            mesh_points = mesh.points.reshape(-1, 3)
            mesh_indices = np.arange(mesh_points.shape[0])
            mesh = wp.Mesh(
                points=wp.array(mesh_points, dtype=wp.vec3),
                indices=wp.array(mesh_indices, dtype=int),
            )

        return mesh
