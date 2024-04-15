import jax.numpy as jnp
from jax import jit
import warp as wp
from typing import Any
import numpy as np

from xlb.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision
from xlb.operator import Operator
from functools import partial


class BGK(Collision):
    """
    BGK collision operator for LBM.
    """

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,))
    def jax_implementation(
        self, f: jnp.ndarray, feq: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
    ):
        fneq = f - feq
        fout = f - omega * fneq
        return fout

    @Operator.register_backend(ComputeBackend.PALLAS)
    def pallas_implementation(
        self, f: jnp.ndarray, feq: jnp.ndarray, rho: jnp.ndarray, u: jnp.ndarray
    ):
        fneq = f - feq
        fout = f - self.omega * fneq
        return fout

    def _construct_warp(self):
        # Set local constants TODO: This is a hack and should be fixed with warp update
        _w = self.velocity_set.wp_w
        _omega = wp.constant(self.compute_dtype(self.omega))
        _f_vec = wp.vec(self.velocity_set.q, dtype=self.compute_dtype)

        # Construct the functional
        @wp.func
        def functional(
            f: Any,
            feq: Any,
            rho: Any,
            u: Any,
        ):
            # Compute the non-equilibrium distribution
            fneq = f - feq

            # Compute the collision
            fout = f - _omega * fneq
            return fout

        # Construct the warp kernel
        @wp.kernel
        def kernel(
            f: wp.array4d(dtype=Any),
            feq: wp.array4d(dtype=Any),
            rho: wp.array4d(dtype=Any),
            u: wp.array4d(dtype=Any),
            fout: wp.array4d(dtype=Any),
        ):
            # Get the global index
            i, j, k = wp.tid()
            index = wp.vec3i(i, j, k)  # TODO: Warp needs to fix this

            # Load needed values
            _f = _f_vec()
            _feq = _f_vec()
            for l in range(self.velocity_set.q):
                _f[l] = f[l, index[0], index[1], index[2]]
                _feq[l] = feq[l, index[0], index[1], index[2]]
            _u = self._warp_u_vec()
            for l in range(_d):
                _u[l] = u[l, index[0], index[1], index[2]]
            _rho = rho[0, index[0], index[1], index[2]]

            # Compute the collision
            _fout = functional(_f, _feq, _rho, _u)

            # Write the result
            for l in range(self.velocity_set.q):
                fout[l, index[0], index[1], index[2]] = _fout[l]

        return functional, kernel

    @Operator.register_backend(ComputeBackend.WARP)
    def warp_implementation(self, f, feq, rho, u, fout):
        # Launch the warp kernel
        wp.launch(
            self.warp_kernel,
            inputs=[
                f,
                feq,
                rho,
                u,
                fout,
            ],
            dim=f.shape[1:],
        )
        return fout
