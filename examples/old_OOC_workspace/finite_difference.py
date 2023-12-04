# Description: This file contains a simple example of using the OOCmap
# decorator to apply a function to a distributed array.
# Solves Darcy flow in a 3D domain using a finite difference method.

import mpi4py.MPI as MPI
import time
import jax.numpy as jnp
from jax import jit
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

from out_of_core import OOCmap
from ooc_array import OOCArray

# Initialize MPI
comm = MPI.COMM_WORLD

@jit
def jacobi_iteration(phi: jnp.ndarray, boundary: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Apply one Jacobi iteration to the input array for Poisson equation where boundary is 0 on the boundary and 1 elsewhere."""

    # Calculate rolled arrays for each direction
    phi_0_1_1 = jnp.roll(phi, 1, axis=0)
    phi_2_1_1 = jnp.roll(phi, -1, axis=0)
    phi_1_0_1 = jnp.roll(phi, 1, axis=1)
    phi_1_2_1 = jnp.roll(phi, -1, axis=1)
    phi_1_1_0 = jnp.roll(phi, 1, axis=2)
    phi_1_1_2 = jnp.roll(phi, -1, axis=2)

    # Jacobi iteration
    phi_new = (
        phi_0_1_1 + phi_2_1_1 + phi_1_0_1 + phi_1_2_1 + phi_1_1_0 + phi_1_1_2 + 1.0
    ) / 6.0

    # Apply boundary conditions
    phi_new = phi_new * boundary

    return phi_new


@OOCmap(comm, (0,))
@jit
def apply_16_jacobi_iterations(
    phi: jnp.ndarray, boundary: jnp.ndarray, dx: float
) -> jnp.ndarray:
    """Apply 16 Jacobi iterations to the input array."""
    for _ in range(16):
        phi = jacobi_iteration(phi, boundary, dx)
    return phi

# Make OOC function for initializing boundary
@OOCmap(comm, (0,), add_index=True)
def out_of_core_boundary(boundary):
    # Boundary is a tuple of (boundary, index) because of add_index=True
    index = boundary[1]
    boundary = boundary[0]

    # Set boundary to 1 everywhere
    boundary = jnp.ones_like(boundary)

    # Set boundary to 0 on the boundary
    # x-direction
    if index[0] <= 0 and 0 < index[0] + boundary.shape[0]:
        boundary = boundary.at[-index[0], :, :].set(0.0)
    # y-direction
    if index[1] <= 0 and 0 < index[1] + boundary.shape[1]:
        boundary = boundary.at[:, -index[1], :].set(0.0)
    # z-direction
    if index[2] <= 0 and 0 < index[2] + boundary.shape[2]:
        boundary = boundary.at[:, :, -index[2]].set(0.0)

    return boundary


if __name__ == "__main__":

    # Normal jax array implementation
    n = 1024
    sub_n = 512
    padding = 16
    dx = 1.0 / n

    # Make OOC distributed array
    dist_phi = OOCArray(
        shape=(n, n, n),
        dtype=cp.float32,
        tile_shape=(sub_n, sub_n, sub_n),
        padding=(padding, padding, padding),
        comm=comm,
        devices=[cp.cuda.Device(0) for i in range(comm.size)],
    )

    # Make boundary, 0 on boundary, 1 elsewhere
    dist_boundary = OOCArray(
        shape=(n, n, n),
        dtype=cp.float32,
        tile_shape=(sub_n, sub_n, sub_n),
        padding=(padding, padding, padding),
        comm=comm,
        devices=[cp.cuda.Device(0) for i in range(comm.size)],
    )
    dist_boundary = out_of_core_boundary(dist_boundary)

    # Apply function to distributed array
    for i in tqdm(range(8)):
        dist_phi = apply_16_jacobi_iterations(dist_phi, dist_boundary, dx)

    # Plot results
    np_phi = dist_phi.get_array()
    plt.imshow(np_phi[:, :, n // 2])
    plt.colorbar()
    plt.savefig("jacobi.png")
    plt.close()
    np_boundary = dist_boundary.get_array()
    plt.imshow(np_boundary[:, :, n // 2])
    plt.colorbar()
    plt.savefig("boundary.png")
    plt.close()
