# from IPython import display
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import xlb

if __name__ == "__main__":
    # Setup
    re = 1600.0
    nr = 64
    vel = 0.04 * 32 / nr
    visc = vel * nr / re
    omega = 1.0 / (3.0 * visc + 0.5)
    print(f"visc: {visc}, omega: {omega}")

    # XLB precision policy
    precision_policy = xlb.precision_policy.Fp32Fp32()

    # XLB lattice
    velocity_set = xlb.velocity_set.D3Q27()

    # XLB equilibrium
    equilibrium = xlb.operator.equilibrium.QuadraticEquilibrium(velocity_set=velocity_set)

    # XLB macroscopic
    macroscopic = xlb.operator.macroscopic.Macroscopic(velocity_set=velocity_set)

    # XLB collision
    #collision = xlb.operator.collision.BGK(omega=omega, velocity_set=velocity_set)
    collision = xlb.operator.collision.KBC(omega=omega, velocity_set=velocity_set)

    # XLB boundary condition (solid cube in the middle)
    box_size = 8
    indices = np.arange((nr//2 - box_size//2), (nr//2 + box_size//2))
    indices = np.meshgrid(indices, indices, indices, indexing="ij")
    indices = np.stack(indices, axis=-1)
    indices = indices.reshape(-1, 3)
    #bounce_back = xlb.operator.boundary_condition.FullBounceBack(indices=indices, velocity_set=velocity_set)
    #bounce_back = xlb.operator.boundary_condition.HalfwayBounceBack(indices=indices, velocity_set=velocity_set)

    # Make XLB compute kernels
    #compute = xlb.compute_constructor.nse.JaxNSE(
    #    collision=collision,
    #    boundary_conditions=[bounce_back],
    #    forcing=None,
    #    precision_policy=precision_policy,
    #)

    # Make taylor green vortex initial condition
    lin = jnp.linspace(0, 2 * jnp.pi, nr, endpoint=False)
    X, Y, Z = jnp.meshgrid(lin, lin, lin, indexing="ij")
    X = X[..., None]
    Y = Y[..., None]
    Z = Z[..., None]
    #u = vel * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    #v = -vel * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    u = vel * jnp.ones_like(X)
    v = jnp.zeros_like(X)
    #u = vel * jnp.sin(X) * jnp.cos(Y)
    #v = -vel * jnp.cos(X) * jnp.sin(Y)
    w = jnp.zeros_like(X)
    #rho = (
    #    3.0
    #    * vel**2
    #    * (1.0 / 16.0)
    #    * (jnp.cos(2 * X) + jnp.cos(2 * Y) + jnp.cos(2 * Z))
    #    + 1.0)
    #rho = 1.0 - vel ** 2 / 12. * (np.cos(2. * X) + np.cos(2. * Y))
    rho = jnp.ones_like(X)
    u = jnp.concatenate([u, v, w], axis=-1)
    f = equilibrium(rho, u)
    print("Here")

    # Get boundary id and mask
    collision_boundary_id = jnp.zeros((nr, nr, nr), dtype=np.uint8)
    stream_boundary_id = jnp.zeros((nr, nr, nr), dtype=np.uint8)
    mask = jnp.ones((nr, nr, nr, velocity_set.q), dtype=np.uint8)
    #boundary_id = compute.set_boundary_id(boundary_id)

    # Set boundary id to be 1 in inner cube
    radius = jnp.pi / 4.0
    stream_boundary_id = jnp.zeros((nr, nr, nr), dtype=np.uint8)
    in_cylinder = (X - jnp.pi) ** 2 + (Y - jnp.pi) ** 2 + (Z - jnp.pi) ** 2 < radius ** 2
    stream_boundary_id = stream_boundary_id.at[in_cylinder[..., 0]].set(1)
    mask = compute.set_mask(mask)

    # Run simulation
    tic = time.time()
    nr_iter = 4096
    for i in tqdm(range(nr_iter)):
        f = compute.step(f, collision_boundary_id, stream_boundary_id, mask, i)

        if i % 32 == 0:
            # Get u, rho from f
            rho, u = compute.macroscopic(f)
            norm_u = jnp.linalg.norm(u, axis=-1)

            # Plot
            plt.imshow(norm_u[..., nr//2], cmap="jet")
            plt.colorbar()
            plt.savefig(f"img_{str(i).zfill(5)}.png")
            plt.close()

    # Sync to host
    f = f.block_until_ready()
    toc = time.time()
    print(f"MLUPS: {(nr_iter * nr**3) / (toc - tic) / 1e6}")
