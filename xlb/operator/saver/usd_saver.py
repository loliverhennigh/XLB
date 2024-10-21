# Base class for all render operators
from pxr import Usd, UsdGeom, Vt, Gf, UsdUtils
import numpy as np

from xlb.compute_backend import ComputeBackend
from xlb.operator.operator import Operator
from xlb.operator.saver.saver import Saver

class USDSaver(Saver):
    """
    Saves USD files of the current geometry
    """

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


    def _construct_warp(self):
        return None, None

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(
        self,
        filename,
        vertex_buffer,
        color_buffer=None,
    ):

        if vertex_buffer.shape[0] == 0:
            return

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
        print(vertices.shape, indices.shape, colors.shape)
        vertices, indices, colors = self._remove_duplicate_vertices(vertices, indices, colors)
        print(vertices.shape, indices.shape, colors.shape)
        print("Reduced vertices from {} to {}".format(vertex_buffer.shape[0], vertices.shape[0]))

        # Create a USD stage (output file will be .usdc for binary, but can be converted to .usdz later)
        stage = Usd.Stage.CreateNew(filename)

        # Create a root Xform (transformation) for the geometry
        root_prim = UsdGeom.Xform.Define(stage, '/Root')

        # Create a mesh inside the root Xform
        mesh_prim = UsdGeom.Mesh.Define(stage, '/Root/Mesh')

        # Set the mesh vertices
        mesh_prim.GetPointsAttr().Set([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in vertices])

        # Set the mesh face indices
        mesh_prim.GetFaceVertexIndicesAttr().Set(Vt.IntArray([int(i) for tri in indices for i in tri]))

        # Set face counts (assuming triangles, so all face counts are 3)
        mesh_prim.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * indices.shape[0]))

        # Set vertex colors if available
        if colors is not None:
            mesh_prim.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in colors]))

        # Save the stage as a binary USD (.usdc) file
        stage.GetRootLayer().Save()

        # Optionally, convert the saved .usdc file to compressed .usdz using USD utilities
        #UsdUtils.CreateNewARKitUsdzPackage(stage.GetRootLayer().realPath, filename.replace('.usdc', '.usdz'))
