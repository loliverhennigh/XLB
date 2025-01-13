# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import io
import tempfile
from collections.abc import Generator
from typing import BinaryIO

import numpy as np
import pyvista as pv
import trimesh
import warp as wp
# Clear kernel cache
wp.clear_kernel_cache()

import xlb
from pydantic import BaseModel
from tqdm import tqdm
from xlb.compute_backend import ComputeBackend
from xlb.operator import Operator
from xlb.operator.boundary_condition import (
    HalfwayBounceBackBC,
    FullwayBounceBackBC,
    EquilibriumBC,
    DoNothingBC,
    RegularizedBC,
    ExtrapolationOutflowBC,
    ZouHeBC,
)
from xlb.operator.boundary_masker import (
    IndicesBoundaryMasker,
    MeshBoundaryMasker,
)
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.force import MomentumTransfer
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.grid import grid_factory
from xlb.precision_policy import PrecisionPolicy

#import sciops

wp.init()

#app = sciops.App(name="Wind Tunnel")


class Point(BaseModel):
    """3D point"""

    x: float = 0
    y: float = 0
    z: float = 0


class Parameters(BaseModel):
    """Simulation parameters"""

    inlet_velocity: float = 27.78 # m/s
    lower_bound: Point = Point()
    upper_bound: Point = Point(x=3.5, y=1.0, z=1.0)
    dx: float = 0.01 # m
    viscosity: float = 1.48e-5 # m^2/s
    density: float = 1.225 # kg/m^3
    solve_time: float = 1.0 # s
    start_avg: float = 0.5 # s
    monitor_frequency: float = 0.01 # s
    device: str = "cuda:0"

@wp.kernel
def avg_kernel(
    current_u: wp.array4d(dtype=wp.float32),
    avg_u: wp.array4d(dtype=wp.float32),
    current_rho: wp.array4d(dtype=wp.float32),
    avg_rho: wp.array4d(dtype=wp.float32),
    nr_avg: int,
):
    """Averging Kernel"""
    i, j, k = wp.tid()
    avg_u[0, i, j, k] += current_u[0, i, j, k] / wp.float32(nr_avg)
    avg_u[1, i, j, k] += current_u[1, i, j, k] / wp.float32(nr_avg)
    avg_u[2, i, j, k] += current_u[2, i, j, k] / wp.float32(nr_avg)
    avg_rho[0, i, j, k] += current_rho[0, i, j, k] / wp.float32(nr_avg)


@wp.kernel
def get_cross_section(
    bc_mask: wp.array4d(dtype=wp.uint8),
    cross_section: wp.array2d(dtype=wp.uint8),
    id: wp.uint32,
):
    """Reduce along x-axis"""
    i, j, k = wp.tid()
    if bc_mask[0, i, j, k] == wp.uint8(id):
        cross_section[j, k] = wp.uint8(1)


def fields_to_vtk(
    fields: dict[str, wp.array4d],
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> io.BytesIO:

    # Save fields
    fields = {k: v.numpy() for k, v in fields.items()}
    pv_grid = pv.ImageData(
        dimensions=[s + 1 for s in shape], spacing=spacing, origin=origin
    )
    for key, value in fields.items():
        if value.shape[0] == 1:
            np_field = value.flatten(order="F")
        else:
            cardinality = value.shape[0]
            np_field = np.stack(
                [np.array(value[i]).flatten(order="F") for i in range(cardinality)],
                axis=1,
            ).reshape(-1, cardinality, order="F")

        pv_grid[key] = np_field

    # Save to io.BytesIO
    with tempfile.NamedTemporaryFile(suffix=".vtk") as f:
        pv_grid.save(f.name, binary=True)
        with open(f.name, "rb") as f:
            file = io.BytesIO(f.read())

    return file


def save_npy(
    array: np.ndarray,
) -> io.BytesIO:

    # Save to io.BytesIO
    with tempfile.NamedTemporaryFile(suffix=".npy") as tmp:
        np.save(tmp.name, array)
        with open(tmp.name, "rb") as f:
            file = io.BytesIO(f.read())
    return file


#@app.simulation()
def wind_tunnel_simulation(
    stl: BinaryIO, parameters: Parameters
) -> Generator[tuple[str, io.BytesIO], None, None]:
    """Wind tunnel simulation function

    Parameters
    ----------
    stl : BinaryIO
        Input STL file of bluff body to use
    parameters : Parameters
        Simulatiojn parameters

    Yields
    ------
    Generator[tuple[str, io.BytesIO], None, None]
        Yields filename, output file
    """

    # Set origin, spacing and shape
    origin = (parameters.lower_bound.x, parameters.lower_bound.y, parameters.lower_bound.z)
    spacing = (parameters.dx, parameters.dx, parameters.dx)
    shape = (
        int((parameters.upper_bound.x - parameters.lower_bound.x) / parameters.dx),
        int((parameters.upper_bound.y - parameters.lower_bound.y) / parameters.dx),
        int((parameters.upper_bound.z - parameters.lower_bound.z) / parameters.dx),
    )
    print(f"Millions of cells: {shape[0] * shape[1] * shape[2] / 1e6}")

    # Get fluid to lbm conversion factors
    base_velocity = 0.01 # inlet velocity in lattice units
    base_density = 1.0 # density in lattice units
    velocity_conversion = base_velocity / parameters.inlet_velocity
    dt = parameters.dx * velocity_conversion
    lbm_viscosity = parameters.viscosity * dt / (parameters.dx ** 2)
    tau = 0.5 + 3.0 * lbm_viscosity
    omega = 1.0 / tau

    # Get time parameters
    nr_steps = int(parameters.solve_time / dt)
    begin_averaging = int(parameters.start_avg / dt)
    monitor_frequency = int(parameters.monitor_frequency / dt)
    monitor_frequency = 1 if monitor_frequency == 0 else monitor_frequency
    nr_avg = int((nr_steps - begin_averaging))
    
    # Get mesh
    mesh = trimesh.load_mesh(
        file_obj=stl, file_type="stl", process=False
    )  # NOTE: process=False is important for vetices, I have no idea why
    vertices = mesh.vertices
    vertices = vertices - np.array([[origin[0], origin[1], origin[2]]])
    vertices = vertices / np.array([[spacing[0], spacing[1], spacing[2]]])

    # Set warp device
    with wp.ScopedDevice(parameters.device):

        # Clear XLB cache
        xlb.operator.boundary_condition.boundary_condition_registry.boundary_condition_registry.id_to_bc = (
            {}
        )
        xlb.operator.boundary_condition.boundary_condition_registry.boundary_condition_registry.bc_to_id = (
            {}
        )
        xlb.operator.boundary_condition.boundary_condition_registry.boundary_condition_registry.next_id = (
            1
        )

        # Set velocity set
        backend = ComputeBackend.WARP
        precision_policy = PrecisionPolicy.FP32FP32
        velocity_set = xlb.velocity_set.D3Q27(
        #velocity_set = xlb.velocity_set.D3Q19(
            precision_policy=precision_policy, backend=backend
        )

        # Initialize backend
        xlb.init(
            velocity_set=velocity_set,
            default_backend=backend,
            default_precision_policy=precision_policy,
        )

        # Make grid
        grid = grid_factory(shape, compute_backend=backend)

        # Make operator for boundary conditions
        boundary_conditions = []
        box = grid.bounding_box_indices()
        box_no_edge = grid.bounding_box_indices(remove_edges=True)
        inlet_indices = box_no_edge["left"]
        outlet_indices = box_no_edge["right"]
        wall_indices = [box["bottom"][i] + box["top"][i] + box["front"][i] + box["back"][i] for i in range(velocity_set.d)]
        wall_indices = np.unique(np.array(wall_indices), axis=-1).tolist()
        boundary_conditions += [
            FullwayBounceBackBC(
                indices=wall_indices
            ),
            RegularizedBC(
                "velocity",
                prescribed_value=(base_velocity, 0.0, 0.0),
                indices=inlet_indices,
            ),
            ExtrapolationOutflowBC(
                indices=outlet_indices,
            ),
            HalfwayBounceBackBC(
                mesh_vertices=vertices,
            ),
        ]

        # Make operator for stepper
        stepper = IncompressibleNavierStokesStepper(
            omega,
            boundary_conditions=boundary_conditions,
            collision_type="KBC",
            #collision_type="BGK",
            grid=grid,
        )

        # Use stepper to create initial fields
        f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

        # Make operator for getting macroscopic fields
        macroscopic = Macroscopic()

        # Make operator for computing drag and lift coefficients
        momentum_transfer = MomentumTransfer(boundary_conditions[-1], velocity_set, precision_policy, backend)

        # Get cross section
        cross_section = wp.zeros((shape[1], shape[2]), dtype=wp.uint8)
        wp.launch(
            get_cross_section,
            dim=(shape[0], shape[1], shape[2]),
            inputs=[bc_mask, cross_section, len(boundary_conditions)],
        )
        cross_section = np.sum(cross_section.numpy())

        # Initialize average fields
        current_u = wp.zeros((3, *shape), dtype=wp.float32)
        avg_u = wp.zeros((3, *shape), dtype=wp.float32)
        current_rho = wp.ones((1, *shape), dtype=wp.float32)
        avg_rho = wp.ones((1, *shape), dtype=wp.float32)

        # Make list to store drag and lift coefficients
        real_time = []
        drag_coef = []
        lift_coef = []

        # Run the simulation
        for step in tqdm(range(nr_steps)):

            # Perform a single time step
            f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, step)

            # Swap fields
            f_0, f_1 = f_1, f_0

            # Post-process
            if step >= begin_averaging:

                # Compute macroscopic fields
                current_rho, current_u = macroscopic(f_0, current_rho, current_u)

                # Update average fields
                wp.launch(
                    avg_kernel,
                    dim=shape,
                    inputs=[
                        current_u,
                        avg_u,
                        current_rho,
                        avg_rho,
                        nr_avg,
                    ],
                )

            # Monitor
            if step % monitor_frequency == 0:

                # Synchronize
                wp.synchronize()

                # Get current u and rho
                current_rho, current_u = macroscopic(f_0, current_rho, current_u)

                # Return fields
                fields = {
                    "last_u": current_u,
                    "last_rho": current_rho,
                    "avg_u": avg_u,
                    "avg_rho": avg_rho,
                    "bc_mask": bc_mask,
                    "missing_mask": missing_mask,
                }
                file = fields_to_vtk(fields, origin, spacing, shape)
                yield f"output_{str(step).zfill(5)}.vtk", file

                # Compute drag and lift coefficients
                boundary_force = momentum_transfer(
                    f_0,
                    f_1,
                    bc_mask,
                    missing_mask,
                )
                drag = boundary_force[0]
                lift = boundary_force[2]
                c_d = 2 * drag / (base_velocity ** 2 * cross_section)
                c_l = 2 * lift / (base_velocity ** 2 * cross_section)
                drag_coef.append(c_d)
                lift_coef.append(c_l)
                real_time.append(step * dt)

                # Save drag and lift coefficients
                file = save_npy(np.array(drag_coef))
                yield "drag.npy", file
                file = save_npy(np.array(lift_coef))
                yield "lift.npy", file
                file = save_npy(np.array(real_time))
                yield "time.npy", file

        # Synchronize
        wp.synchronize()

        # Save fields
        fields = {
            "avg_u": avg_u.numpy(),
            "avg_rho": avg_rho.numpy(),
            "last_u": current_u.numpy(),
            "last_rho": current_rho.numpy(),
            "bc_mask": bc_mask.numpy(),
        }
        file = fields_to_vtk(fields, origin, spacing, shape)

        yield "final.vtk", file

if __name__ == "__main__":

    params = {
        "inlet_velocity": 0.1,
        "lower_bound": {"x": -3.0, "y": -2.0, "z": -0.35},
        "upper_bound": {"x": 10.0, "y": 2.0, "z": 2.5},
        "dx": 0.02,
        "viscosity": 1.48e-5,
        "density": 1.225,
        "solve_time": 1000.0,
        "start_avg": 500.0,
        "monitor_frequency": 5.0,
        "device": "cuda:0",
    }
    stl = open("./run_1/drivaer_1_single_solid.stl", "rb")
    parameters = Parameters(**params)

    for filename, file in wind_tunnel_simulation(stl, parameters):
        with open("/home/oliver/store_xlb/" + filename, "wb") as f:
            f.write(file.read())
