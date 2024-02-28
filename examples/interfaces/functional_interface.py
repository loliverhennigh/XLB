# Simple Taylor green example using the functional interface to xlb

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import warp as wp
wp.init()

import xlb
from xlb.operator import Operator

class TaylorGreenInitializer(Operator):

    def _construct_warp(self):
        # Construct the warp kernel
        @wp.kernel
        def kernel(
            rho: self._warp_array_type,
            u: self._warp_array_type,
            vel: float,
            nr: int,
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Get real pos
            x = 2.0 * wp.pi * wp.float(i) / wp.float(nr)
            y = 2.0 * wp.pi * wp.float(j) / wp.float(nr)
            z = 2.0 * wp.pi * wp.float(k) / wp.float(nr)

            # Compute u
            u[0, i, j, k] = vel * wp.sin(x) * wp.cos(y) * wp.cos(z)
            u[1, i, j, k] = - vel * wp.cos(x) * wp.sin(y) * wp.cos(z)
            u[2, i, j, k] = 0.0

            # Compute rho
            rho[0, i, j, k] = (
                3.0
                * vel
                * vel
                * (1.0 / 16.0)
                * (
                    wp.cos(2.0 * x)
                    + (wp.cos(2.0 * y)
                    * (wp.cos(2.0 * z) + 2.0))
                )
                + 1.0
            )

        return None, kernel

    @Operator.register_backend(xlb.ComputeBackend.WARP)
    def warp_implementation(self, rho, u, vel, nr):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                rho,
                u,
                vel,
                nr,
            ],
            dim=rho.shape[1:],
        )
        return rho, u

if __name__ == "__main__":

    # Set parameters
    compute_backend = xlb.ComputeBackend.WARP
    precision_policy = xlb.PrecisionPolicy.FP32FP32
    velocity_set = xlb.velocity_set.D3Q19()

    # Make feilds
    nr = 256
    shape = (nr, nr, nr)
    grid = xlb.grid.WarpGrid(shape=shape)
    rho = grid.create_field(cardinality=1, dtype=wp.float32)
    u = grid.create_field(cardinality=velocity_set.d, dtype=wp.float32)
    f0 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    f1 = grid.create_field(cardinality=velocity_set.q, dtype=wp.float32)
    boundary_id = grid.create_field(cardinality=1, dtype=wp.uint8)
    mask = grid.create_field(cardinality=velocity_set.q, dtype=wp.bool)

    # Make operators
    initializer = TaylorGreenInitializer(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    collision = xlb.operator.collision.BGK(
            omega=1.0,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    macroscopic = xlb.operator.macroscopic.Macroscopic(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    stream = xlb.operator.stream.Stream(
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend)
    stepper = xlb.operator.stepper.IncompressibleNavierStokesStepper(
            collision=collision,
            equilibrium=equilibrium,
            macroscopic=macroscopic,
            stream=stream,
            boundary_conditions=[])

    # Parrallelize the stepper
    #stepper = grid.parallelize_operator(stepper)

    # Set initial conditions
    rho, u = initializer(rho, u, 0.1, nr)
    f0 = equilibrium(rho, u, f0)

    # Plot initial conditions
    #plt.imshow(f0[0, nr//2, :, :].numpy())
    #plt.show()

    # Time stepping
    num_steps = 1024
    start = time.time()
    for _ in tqdm(range(num_steps)):
        f1 = stepper(f0, f1, boundary_id, mask, _)
        f1, f0 = f0, f1
    wp.synchronize()
    end = time.time()

    # Print MLUPS
    print(f"MLUPS: {num_steps*nr**3/(end-start)/1e6}")