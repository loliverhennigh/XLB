# LES simulation of NYC with out-of-core data management

# Standard imports
import os
import time
from tqdm import tqdm
from mpi4py import MPI
import numpy as np
import cupy as cp
import warp as wp
import trimesh
import pyvista as pv
import matplotlib.pyplot as plt

wp.init()

# XLB imports
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.helper import create_nse_fields, initialize_eq
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import (
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    HalfwayBounceBackBC,
    ExtrapolationOutflowBC,
)
from xlb.operator.force.momentum_transfer import MomentumTransfer
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.boundary_masker import IndicesBoundaryMasker, MeshBoundaryMasker
from xlb.operator.q_criteria import QCriteria
from xlb.operator.marching_cubes import MarchingCubesBoundary, MarchingCubesContour
from xlb.operator.saver import OBJSaver, USDSaver
from xlb.utils import save_fields_vtk, save_image

# Local OOC implementation
from out_of_core import OOCmap
from ooc_array import OOCArray
from utils import _backend_to_cupy

# Import phantom gaze
import phantomgaze as pg

# MPI
comm = MPI.COMM_WORLD

def make_bounding_box_indices(
    shape,
    remove_edges=True,
):

    # Get the shape of the grid
    origin = np.array([0, 0, 0])
    bounds = np.array(shape)
    if remove_edges:
        origin += 1
        bounds -= 1
    slice_x = slice(origin[0], bounds[0])
    slice_y = slice(origin[1], bounds[1])
    dim = len(bounds)

    # Generate bounding box indices for each face
    grid = np.indices(shape)
    boundingBoxIndices = {}
    nx, ny, nz = shape
    slice_z = slice(origin[2], bounds[2])
    boundingBoxIndices = {
        "bottom": grid[:, slice_x, slice_y, 0].reshape(3, -1),
        "top": grid[:, slice_x, slice_y, nz - 1].reshape(3, -1),
        "front": grid[:, 0, slice_y, slice_z].reshape(3, -1),
        "back": grid[:, nx - 1, slice_y, slice_z].reshape(3, -1),
        "left": grid[:, slice_x, 0, slice_z].reshape(3, -1),
        "right": grid[:, slice_x, ny - 1, slice_z].reshape(3, -1),
    }
    return {k: v.tolist() for k, v in boundingBoxIndices.items()}

class CopyLBState():

    @wp.kernel
    def _copy_lb_state(
        source: wp.array4d(dtype=wp.float32),
        destination: wp.array4d(dtype=wp.float32),
    ):
        # Get index
        i, j, k = wp.tid()

        # Split the fields
        for c in range(27):
            destination[c, i, j, k] = source[c, i, j, k]

    def __call__(
        self,
        source,
        destination,
    ):
        # Launch kernel
        wp.launch(
            self._copy_lb_state,
            inputs=[
                source,
                destination,
            ],
            dim=source.shape[1:],
        )

        return destination

class InitializeVelocity():

    @wp.kernel
    def _initialize_velocity(
        u: wp.array4d(dtype=wp.float32),
        velocity: wp.vec3,
    ):
        # Get index
        i, j, k = wp.tid()
        for d in range(3):
            u[d, i, j, k] = velocity[d]

    def __call__(
        self,
        u,
        velocity,
    ):
        # Launch kernel
        wp.launch(
            self._initialize_velocity,
            inputs=[
                u,
                velocity,
            ],
            dim=u.shape[1:],
        )

        return u



if __name__ == '__main__':

    # Define geometric parameters
    dx = 6.0 # m
    domain_padding = 100.0
    origin = (-domain_padding, -domain_padding, -dx)
    upper_bound = (4152.0 + domain_padding, 4281.0 + domain_padding, 650)
    spacing = (dx, dx, dx)
    global_shape = [int((upper_bound[i] - origin[i]) / spacing[i]) for i in range(3)]
    stl_file = "data/final_manhattan_2.stl"

    # Define Fluid parameters
    velocity = 0.0003 # m/s
    fluid_density = 1.225 # kg/m^3
    fluid_viscosity = 1.42e-5 # kg/m/s
    solve_time = 360000000.0 # s (1 hour)

    # LBM parameters
    lbm_velocity = 0.05
    dt = dx * (lbm_velocity / velocity)
    lbm_density = 1.0
    lbm_viscosity = fluid_viscosity * dt / (dx * dx)
    tau = 0.5 + 3.0 * lbm_viscosity
    omega = 1.0 / tau
    num_steps = int(solve_time / dt)
    backend = ComputeBackend.WARP
    precision_policy = PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q27(precision_policy=precision_policy, backend=backend)

    # Initialize xlb
    xlb.init(
        velocity_set=velocity_set,
        default_backend=backend,
        default_precision_policy=precision_policy,
    )

    # Subdomain parameters
    sub_shape = [128, 128, 128]
    sub_steps = 8
    global_steps = num_steps // sub_steps
    num_steps = global_steps * sub_steps
    sub_shape_with_padding = [sub_shape[i] + 2*sub_steps for i in range(3)]

    # Padd the shape so divisible by sub_shape
    global_shape = [int(np.ceil(global_shape[i] / sub_shape[i]) * sub_shape[i]) for i in range(3)]
    nr_cells = global_shape[0] * global_shape[1] * global_shape[2]

    # Make directory if not exists
    output_dir = "output"
    boundary_dir = f"{output_dir}/boundary"
    q_criterion_dir = f"{output_dir}/q_criterion"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(boundary_dir, exist_ok=True)
    os.makedirs(q_criterion_dir, exist_ok=True)

    # Make usd parameters
    obj_saver = OBJSaver()
    usd_saver = USDSaver()

    # Compute number of subdomains
    num_subdomains = np.prod([int(global_shape[i] / sub_shape[i]) for i in range(3)])
    if num_subdomains % 2 == 0:
        nr_compute_tiles = 2
    else:
        nr_compute_tiles = 1

    # Print parameters
    print(f"Gloabl Shape: {global_shape}")
    print(f"Sub Shape: {sub_shape}")
    print(f"Nr subdomains: {[int(global_shape[i] / sub_shape[i]) for i in range(3)]}")
    print(f"Nr Million cells: {nr_cells / 1.0e6}")
    print(f"Tau: {tau}")
    print(f"Nr compute tiles: {nr_compute_tiles}")

    # Make fields for NSE
    (
        xlb_grid,
        f_0,
        f_1,
        missing_mask,
        bc_mask
    ) = create_nse_fields(
        tuple(sub_shape_with_padding),
    )
    rho = xlb_grid.create_field(cardinality=1, fill_value=1.0, dtype=precision_policy.compute_precision)
    u = xlb_grid.create_field(cardinality=velocity_set.d, fill_value=0.0, dtype=precision_policy.compute_precision)
    norm_mu = xlb_grid.create_field(cardinality=1, fill_value=0.0, dtype=precision_policy.compute_precision)
    q = xlb_grid.create_field(cardinality=1, fill_value=0.0, dtype=precision_policy.compute_precision)

    # Make the mesh vertices
    mesh = trimesh.load_mesh(stl_file, process=False)

    # Make operator for boundary condition masker
    indices_boundary_masker = IndicesBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )
    mesh_boundary_masker = MeshBoundaryMasker(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )

    # Make operator for boundary conditions
    bounding_box_indices = make_bounding_box_indices(global_shape)
    inlet_indices = [
        bounding_box_indices["left"][i]
        + bounding_box_indices["top"][i]
        for i in range(velocity_set.d)
    ]
    outlet_indices = bounding_box_indices["right"]
    wall_indices = bounding_box_indices["bottom"]
    boundary_conditions = [
        EquilibriumBC(
            rho=lbm_density,
            u=(0.0, lbm_velocity, 0.0),
            indices=inlet_indices,
        ),
        DoNothingBC(
            indices=outlet_indices,
        ),
        FullwayBounceBackBC(
            indices=wall_indices,
        ),
        HalfwayBounceBackBC(
            mesh_vertices=mesh.vertices,
        ),
    ]

    # Make operator for stepper
    stepper = IncompressibleNavierStokesStepper(
        omega,
        boundary_conditions=boundary_conditions,
        collision_type="KBC"
    )

    # Make operator for equilibrium, macroscopic and q-criterion
    equilibrium = QuadraticEquilibrium(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )
    macroscopic = Macroscopic(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )
    q_criterion = QCriteria(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )
    velocity_initializer = InitializeVelocity()

    # Make operator for marching cubes
    marching_cubes_boundary = MarchingCubesBoundary(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )
    marching_cubes_contour = MarchingCubesContour(
        velocity_set=velocity_set,
        precision_policy=precision_policy,
        compute_backend=backend,
    )

    # Apply indicies boundary masker
    bc_mask, missing_mask = indices_boundary_masker(
        boundary_conditions[:3],
        bc_mask,
        missing_mask,
    )
    bc_mask, missing_mask = mesh_boundary_masker(
        boundary_conditions[3],
        (0.0, 0.0, 0.0),
        (dx, dx, dx),
        bc_mask,
        missing_mask,
    )
    del bounding_box_indices
 
    # Make copy operator
    copy_lb_state = CopyLBState()

    # Make the fields on the host
    state_array = OOCArray(
        shape=[27] + list(global_shape),
        dtype=np.float32,
        tile_shape=[27] + list(sub_shape),
        padding=(0, sub_steps, sub_steps, sub_steps),
        comm=comm,
        devices=[cp.cuda.Device(0)],
        codec=None,
        nr_compute_tiles=nr_compute_tiles,
    )

    # Initialize the state array
    @OOCmap(comm, (0,), backend="warp")
    def ooc_initialize(
        lbm_state_array,
        f_0,
        rho,
        u,
    ):

        # Initialize the velocity
        u = velocity_initializer(
            u,
            (0.0, lbm_velocity, 0.0),
        )

        # Copy the state array to f_0
        f_0 = equilibrium(rho, u, f_0)

        # Copy the state array back
        lbm_state_array = copy_lb_state(
            f_0,
            lbm_state_array,
        )

        return lbm_state_array

    # LBM update
    @OOCmap(comm, (0,), add_index=True, backend="warp")
    def ooc_update(
        lbm_state_array,
        f_0,
        f_1,
        missing_mask,
        bc_mask,
    ):

        # Unpack the state array
        lbm_state_array, global_index = lbm_state_array

        # Nock off first dim of global index
        global_index = global_index[1:]

        # Copy the state array to f_0
        f_0 = copy_lb_state(
            lbm_state_array,
            f_0,
        )

        # Generate boundary conditions
        bc_mask.zero_()
        missing_mask.zero_()
        bc_mask, missing_mask = indices_boundary_masker(
            boundary_conditions[:3],
            bc_mask,
            missing_mask,
            start_index=global_index,
        )
        bc_mask, missing_mask = mesh_boundary_masker(
            boundary_conditions[3],
            origin,
            spacing,
            bc_mask,
            missing_mask,
            start_index=global_index,
        )

        # Run the stepper
        for step in range(sub_steps):
            f_1 = stepper(
                f_0,
                f_1,
                bc_mask,
                missing_mask,
                timestep=0,
            )

            # Swap the fields
            f_0, f_1 = f_1, f_0

        # Copy the state array back
        lbm_state_array = copy_lb_state(
            f_0,
            lbm_state_array,
        )

        return lbm_state_array

    # Make the USD saver
    @OOCmap(comm, (0,), add_index=True, backend="warp")
    def save_usd(
        lbm_state_array,
        f_0,
        rho,
        u,
        norm_mu,
        q,
        bc_mask,
        missing_mask,
        step,
    ):

        # Unpack the state array
        lbm_state_array, global_index = lbm_state_array

        # Nock off first dim of global index
        global_index = global_index[1:]

        # Copy the state array to f_0
        f_0 = copy_lb_state(
            lbm_state_array,
            f_0,
        )

        # Generate boundary conditions
        bc_mask.zero_()
        missing_mask.zero_()
        bc_mask, missing_mask = indices_boundary_masker(
            boundary_conditions[:3],
            bc_mask,
            missing_mask,
            start_index=global_index,
        )
        bc_mask, missing_mask = mesh_boundary_masker(
            boundary_conditions[3],
            origin,
            spacing,
            bc_mask,
            missing_mask,
            start_index=global_index,
        )

        # Get rho, u
        rho, u = macroscopic(f_0, rho, u)

        # Get q-cr
        norm_mu, q = q_criterion(u, norm_mu, q)

        # Get local origin
        local_origin = (
            origin[0] + global_index[0] * dx,
            origin[1] + global_index[1] * dx,
            origin[2] + global_index[2] * dx,
        )

        # Check nans
        if np.isnan(q[0, 0:32, 0:32, 0:32].numpy()).any():
            print("NaNs in q-criterion")
            exit()

        # Get marching cubes of the contour
        vertex_buffer, color_buffer = marching_cubes_contour(
            q,
            0.0000005,
            norm_mu,
            wp.vec2(0.0, 0.025),
            local_origin,
            spacing,
            "jet",
            padding=sub_steps,
        )

        # Make dir for grid
        grid_dir = f"{q_criterion_dir}/nyc_{global_index[0]}_{global_index[1]}_{global_index[2]}"
        os.makedirs(grid_dir, exist_ok=True)

        # Save to the obj
        obj_saver(
            f"{grid_dir}/step_{str(step).zfill(9)}.obj",
            vertex_buffer,
            color_buffer,
        )
        #usd_saver(
        #    f"{grid_dir}/step_{str(step).zfill(9)}.usdc",
        #    vertex_buffer,
        #    color_buffer,
        #)

        # Only save on first step
        if step == 0:

            # Get marching cubes of the boundary
            for i in range(5):

                # Get color
                if i == 0: # Inlet (seethrough blue)
                    color = (0, 0, 255, 0)
                    continue
                elif i == 1: # Outlet (seethrough red)
                    color = (255, 0, 0, 0)
                    continue
                elif i == 2: # Wall (seethrough gray)
                    color = (128, 128, 128, 255)
                elif i == 3: # Ed
                    color = (0, 255, 0, 255)
                    continue
                elif i == 4:
                    color = (255, 0, 255, 255)
                    continue

                # Get vertex buffer
                vertex_buffer = marching_cubes_boundary(
                    bc_mask,
                    wp.uint8(i + 1),
                    local_origin,
                    spacing,
                    padding=sub_steps,
                )

                # Save to the obj
                obj_saver(
                    f"{boundary_dir}/nyc_{global_index[0]}_{global_index[1]}_{global_index[2]}_{i}.obj",
                    vertex_buffer,
                    color,
                )
                #usd_saver(
                #    f"{boundary_dir}/nyc_{global_index[0]}_{global_index[1]}_{global_index[2]}_{i}.usdc",
                #    vertex_buffer,
                #    color,
                #)

        return lbm_state_array

    # Initialize the state array
    state_array = ooc_initialize(
        state_array,
        f_0,
        rho,
        u,
    )

    # Run the simulation
    wp.synchronize()
    tic = time.time()
    for step in tqdm(range(global_steps)):

        # Run the OOC update
        state_array = ooc_update(
            state_array,
            f_0,
            f_1,
            missing_mask,
            bc_mask,
        )

        # Save usd
        if step % 8 == 0:

            # Save the usd
            state_array = save_usd(
                state_array,
                f_0,
                rho,
                u,
                norm_mu,
                q,
                bc_mask,
                missing_mask,
                step,
            )

            # Print MUPS
            wp.synchronize()
            toc = time.time()
            mups = sub_steps * nr_cells * (step + 1) / (toc - tic) / 1.0e6
            print(f"MUPS: {mups}")
