# Base class for voxelizing geometries
import warp as wp

from pumpkin_pulse.operator.operator import Operator
from pumpkin_pulse.data.field import Fielduint8

class MeshToIdField(Operator):
    """
    Operator to set the material properties of a mesh
    """

    @wp.kernel
    def _mesh_to_id_field(
        mesh: wp.uint64,
        id_field: Fielduint8,
        id_number: int,
    ):
    
        # get spatial index
        i, j, k = wp.tid()

        # position of voxel (cell center)
        ijk = wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k))
        ijk = ijk + wp.vec3(0.5, 0.5, 0.5) # cell center
        pos = wp.cw_mul(ijk, id_field.spacing) + id_field.origin

        # Compute maximum distance to check
        max_length = wp.sqrt(
                (id_field.spacing[0] * wp.float32(id_field.shape[0]))**2.0
                + (id_field.spacing[1] * wp.float32(id_field.shape[1]))**2.0
                + (id_field.spacing[2] * wp.float32(id_field.shape[2]))**2.0
        )
    
        # evaluate distance of point
        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)
        p = wp.mesh_query_point_sign_winding_number(
            mesh, pos, max_length, sign, face_index, face_u, face_v
        )
 
        # set point to be solid
        if sign < 0.0:
            id_field.data[0, i, j, k] = wp.uint8(id_number)

    def __call__(
        self,
        mesh: wp.Mesh,
        id_field: Fielduint8,
        id_number: int,
    ):
        # Voxelize STL of mesh
        wp.launch(
            self._mesh_to_id_field,
            inputs=[
                mesh.id,
                id_field,
                id_number,
            ],
            dim=id_field.shape,
        )

        return id_field
