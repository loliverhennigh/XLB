"""
KBC collision operator for LBM.
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from xlb.operator import Operator
from xlb.velocity_set import VelocitySet, D2Q9, D3Q27
from xlb.compute_backend import ComputeBackend
from xlb.operator.collision.collision import Collision


class KBC(Collision):
    """
    KBC collision operator for LBM.

    This class implements the Karlin-Bösch-Chikatamarla (KBC) model for the collision step in the Lattice Boltzmann Method.
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(
<<<<<<< HEAD
            omega=omega,
            velocity_set=velocity_set,
            precision_policy=precision_policy,
            compute_backend=compute_backend,
=======
            velocity_set=velocity_set, compute_backend=compute_backend
>>>>>>> a48510cefc7af0cb965b67c86854a609b7d8d1d4
        )
        self.epsilon = 1e-32
        self.beta = self.omega * 0.5
        self.inv_beta = 1.0 / self.beta

    @Operator.register_backend(ComputeBackend.JAX)
    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def jax_implementation(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
    ):
        """
        KBC collision step for lattice.

        Parameters
        ----------
        f : jax.numpy.array
            Distribution function.
        feq : jax.numpy.array
            Equilibrium distribution function.
        rho : jax.numpy.array
            Density.
        """
        fneq = f - feq
        if isinstance(self.velocity_set, D2Q9):
            shear = self.decompose_shear_d2q9_jax(fneq)
            delta_s = shear * rho / 4.0  # TODO: Check this
        elif isinstance(self.velocity_set, D3Q27):
            shear = self.decompose_shear_d3q27_jax(fneq)
            delta_s = shear * rho
        else:
            raise NotImplementedError(
                "Velocity set not supported: {}".format(type(self.velocity_set))
            )

        # Perform collision
        delta_h = fneq - delta_s
        gamma = self.inv_beta - (2.0 - self.inv_beta) * self.entropic_scalar_product(
            delta_s, delta_h, feq
        ) / (self.epsilon + self.entropic_scalar_product(delta_h, delta_h, feq))

        fout = f - self.beta * (2.0 * delta_s + gamma[None, ...] * delta_h)

        return fout

    @Operator.register_backend(ComputeBackend.WARP)
    @partial(jit, static_argnums=(0,), donate_argnums=(1, 2, 3))
    def warp_implementation(
        self,
        f: jnp.ndarray,
        feq: jnp.ndarray,
        rho: jnp.ndarray,
    ):
        raise NotImplementedError("Warp implementation not yet implemented")

    @partial(jit, static_argnums=(0,), inline=True)
    def entropic_scalar_product(self, x: jnp.ndarray, y: jnp.ndarray, feq: jnp.ndarray):
        """
        Compute the entropic scalar product of x and y to approximate gamma in KBC.

        Returns
        -------
        jax.numpy.array
            Entropic scalar product of x, y, and feq.
        """
        return jnp.sum(x * y / feq, axis=0)

    @partial(jit, static_argnums=(0,), donate_argnums=(1,))
    def momentum_flux_jax(
        self,
        fneq: jnp.ndarray,
    ):
        """
        This function computes the momentum flux, which is the product of the non-equilibrium
        distribution functions (fneq) and the lattice moments (cc).

        The momentum flux is used in the computation of the stress tensor in the Lattice Boltzmann
        Method (LBM).

        # TODO: probably move this to equilibrium calculation

        Parameters
        ----------
        fneq: jax.numpy.ndarray
            The non-equilibrium distribution functions.

        Returns
        -------
        jax.numpy.ndarray
            The computed momentum flux.
        """

        return jnp.tensordot(self.velocity_set.cc, fneq, axes=(0, 0))

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d3q27_jax(self, fneq):
        """
        Decompose fneq into shear components for D3Q27 lattice.

        Parameters
        ----------
        fneq : jax.numpy.ndarray
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.ndarray
            Shear components of fneq.
        """

        # Calculate the momentum flux
        Pi = self.momentum_flux_jax(fneq)
        # Calculating Nxz and Nyz with indices moved to the first dimension
        Nxz = Pi[0, ...] - Pi[5, ...]
        Nyz = Pi[3, ...] - Pi[5, ...]

        # For c = (i, 0, 0), c = (0, j, 0) and c = (0, 0, k)
        s = jnp.zeros_like(fneq)
        s = s.at[9, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[18, ...].set((2.0 * Nxz - Nyz) / 6.0)
        s = s.at[3, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[6, ...].set((-Nxz + 2.0 * Nyz) / 6.0)
        s = s.at[1, ...].set((-Nxz - Nyz) / 6.0)
        s = s.at[2, ...].set((-Nxz - Nyz) / 6.0)

        # For c = (i, j, 0)
        s = s.at[12, ...].set(Pi[1, ...] / 4.0)
        s = s.at[24, ...].set(Pi[1, ...] / 4.0)
        s = s.at[21, ...].set(-Pi[1, ...] / 4.0)
        s = s.at[15, ...].set(-Pi[1, ...] / 4.0)

        # For c = (i, 0, k)
        s = s.at[10, ...].set(Pi[2, ...] / 4.0)
        s = s.at[20, ...].set(Pi[2, ...] / 4.0)
        s = s.at[19, ...].set(-Pi[2, ...] / 4.0)
        s = s.at[11, ...].set(-Pi[2, ...] / 4.0)

        # For c = (0, j, k)
        s = s.at[8, ...].set(Pi[4, ...] / 4.0)
        s = s.at[4, ...].set(Pi[4, ...] / 4.0)
        s = s.at[7, ...].set(-Pi[4, ...] / 4.0)
        s = s.at[5, ...].set(-Pi[4, ...] / 4.0)

        return s

    @partial(jit, static_argnums=(0,), inline=True)
    def decompose_shear_d2q9_jax(self, fneq):
        """
        Decompose fneq into shear components for D2Q9 lattice.

        Parameters
        ----------
        fneq : jax.numpy.array
            Non-equilibrium distribution function.

        Returns
        -------
        jax.numpy.array
            Shear components of fneq.
        """
        Pi = self.momentum_flux_jax(fneq)
        N = Pi[0, ...] - Pi[2, ...]
        s = jnp.zeros_like(fneq)
        s = s.at[3, ...].set(N)
        s = s.at[6, ...].set(N)
        s = s.at[2, ...].set(-N)
        s = s.at[1, ...].set(-N)
        s = s.at[8, ...].set(Pi[1, ...])
        s = s.at[4, ...].set(-Pi[1, ...])
        s = s.at[5, ...].set(-Pi[1, ...])
        s = s.at[7, ...].set(Pi[1, ...])

        return s