from functools import partial
import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any

from xlb.compute_backend import ComputeBackend
from xlb.operator.equilibrium.equilibrium import Equilibrium
from xlb.operator import Operator

class QCriteria(Operator):

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0))
    def jax_implementation(self, u):

        # Compute derivatives
        u_x = u[0, ...]
        u_y = u[1, ...]
        u_z = u[2, ...]
    
        # Compute derivatives
        u_x_dx = (u_x[2:, 1:-1, 1:-1] - u_x[:-2, 1:-1, 1:-1]) / 2
        u_x_dy = (u_x[1:-1, 2:, 1:-1] - u_x[1:-1, :-2, 1:-1]) / 2
        u_x_dz = (u_x[1:-1, 1:-1, 2:] - u_x[1:-1, 1:-1, :-2]) / 2
        u_y_dx = (u_y[2:, 1:-1, 1:-1] - u_y[:-2, 1:-1, 1:-1]) / 2
        u_y_dy = (u_y[1:-1, 2:, 1:-1] - u_y[1:-1, :-2, 1:-1]) / 2
        u_y_dz = (u_y[1:-1, 1:-1, 2:] - u_y[1:-1, 1:-1, :-2]) / 2
        u_z_dx = (u_z[2:, 1:-1, 1:-1] - u_z[:-2, 1:-1, 1:-1]) / 2
        u_z_dy = (u_z[1:-1, 2:, 1:-1] - u_z[1:-1, :-2, 1:-1]) / 2
        u_z_dz = (u_z[1:-1, 1:-1, 2:] - u_z[1:-1, 1:-1, :-2]) / 2
    
        # Compute vorticity
        mu_x = u_z_dy - u_y_dz
        mu_y = u_x_dz - u_z_dx
        mu_z = u_y_dx - u_x_dy
        norm_mu = jnp.sqrt(mu_x ** 2 + mu_y ** 2 + mu_z ** 2)
    
        # Compute strain rate
        s_0_0 = u_x_dx
        s_0_1 = 0.5 * (u_x_dy + u_y_dx)
        s_0_2 = 0.5 * (u_x_dz + u_z_dx)
        s_1_0 = s_0_1
        s_1_1 = u_y_dy
        s_1_2 = 0.5 * (u_y_dz + u_z_dy)
        s_2_0 = s_0_2
        s_2_1 = s_1_2
        s_2_2 = u_z_dz
        s_dot_s = (
            s_0_0 ** 2 + s_0_1 ** 2 + s_0_2 ** 2 +
            s_1_0 ** 2 + s_1_1 ** 2 + s_1_2 ** 2 +
            s_2_0 ** 2 + s_2_1 ** 2 + s_2_2 ** 2
        )
    
        # Compute omega
        omega_0_0 = 0.0
        omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
        omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
        omega_1_0 = -omega_0_1
        omega_1_1 = 0.0
        omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
        omega_2_0 = -omega_0_2
        omega_2_1 = -omega_1_2
        omega_2_2 = 0.0
        omega_dot_omega = (
            omega_0_0 ** 2 + omega_0_1 ** 2 + omega_0_2 ** 2 +
            omega_1_0 ** 2 + omega_1_1 ** 2 + omega_1_2 ** 2 +
            omega_2_0 ** 2 + omega_2_1 ** 2 + omega_2_2 ** 2
        )
    
        # Compute q-criterion
        q = 0.5 * (omega_dot_omega - s_dot_s)
    
        return norm_mu, q

    def _construct_warp(self):

        # Construct the warp kernel
        @wp.kernel
        def kernel3d(
            u: wp.array4d(dtype=Any),
            norm_mu: wp.array4d(dtype=Any),
            q: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()

            # Add ghost cells to index
            i += 1
            j += 1
            k += 1

            # Get derivatives
            u_x_dx = (u[0, i + 1, j, k] - u[0, i - 1, j, k]) / 2.0
            u_x_dy = (u[0, i, j + 1, k] - u[0, i, j - 1, k]) / 2.0
            u_x_dz = (u[0, i, j, k + 1] - u[0, i, j, k - 1]) / 2.0
            u_y_dx = (u[1, i + 1, j, k] - u[1, i - 1, j, k]) / 2.0
            u_y_dy = (u[1, i, j + 1, k] - u[1, i, j - 1, k]) / 2.0
            u_y_dz = (u[1, i, j, k + 1] - u[1, i, j, k - 1]) / 2.0
            u_z_dx = (u[2, i + 1, j, k] - u[2, i - 1, j, k]) / 2.0
            u_z_dy = (u[2, i, j + 1, k] - u[2, i, j - 1, k]) / 2.0
            u_z_dz = (u[2, i, j, k + 1] - u[2, i, j, k - 1]) / 2.0

            # Compute vorticity
            mu_x = u_z_dy - u_y_dz
            mu_y = u_x_dz - u_z_dx
            mu_z = u_y_dx - u_x_dy
            mu = wp.sqrt(mu_x ** 2.0 + mu_y ** 2.0 + mu_z ** 2.0)

            # Compute strain rate
            s_0_0 = u_x_dx
            s_0_1 = 0.5 * (u_x_dy + u_y_dx)
            s_0_2 = 0.5 * (u_x_dz + u_z_dx)
            s_1_0 = s_0_1
            s_1_1 = u_y_dy
            s_1_2 = 0.5 * (u_y_dz + u_z_dy)
            s_2_0 = s_0_2
            s_2_1 = s_1_2
            s_2_2 = u_z_dz
            s_dot_s = (
                s_0_0 ** 2.0 + s_0_1 ** 2.0 + s_0_2 ** 2.0 +
                s_1_0 ** 2.0 + s_1_1 ** 2.0 + s_1_2 ** 2.0 +
                s_2_0 ** 2.0 + s_2_1 ** 2.0 + s_2_2 ** 2.0
            )

            # Compute omega
            omega_0_0 = 0.0
            omega_0_1 = 0.5 * (u_x_dy - u_y_dx)
            omega_0_2 = 0.5 * (u_x_dz - u_z_dx)
            omega_1_0 = -omega_0_1
            omega_1_1 = 0.0
            omega_1_2 = 0.5 * (u_y_dz - u_z_dy)
            omega_2_0 = -omega_0_2
            omega_2_1 = -omega_1_2
            omega_2_2 = 0.0
            omega_dot_omega = (
                omega_0_0 ** 2.0 + omega_0_1 ** 2.0 + omega_0_2 ** 2.0 +
                omega_1_0 ** 2.0 + omega_1_1 ** 2.0 + omega_1_2 ** 2.0 +
                omega_2_0 ** 2.0 + omega_2_1 ** 2.0 + omega_2_2 ** 2.0
            )

            # Compute q-criterion
            q_value = 0.5 * (omega_dot_omega - s_dot_s)

            # Set the output
            norm_mu[0, i, j, k] = mu
            q[0, i, j, k] = q_value

        kernel = kernel3d

        return None, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, u, norm_mu, q):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                u,
                norm_mu,
                q,
            ],
            dim=[i - 2 for i in u.shape[1:]],
        )
        return norm_mu, q
