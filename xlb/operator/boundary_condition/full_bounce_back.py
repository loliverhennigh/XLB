"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.compute_backend import ComputeBackend
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)

class FullBounceBack(BoundaryCondition):
    """
    Full Bounce-back boundary condition for a lattice Boltzmann method simulation.
    """

    def __init__(
            self,
            set_boundary_mask,
            velocity_set: VelocitySet,
            compute_backend: ComputeBackend = ComputeBackend.JAX,
        ):
        super().__init__(
            indices,
            solid=True,
            velocity_set=velocity_set,
            implementation_step=ImplementationStep.COLLISION,
        )

    @classmethod
    def from_indices(
            cls,
            indices,
            velocity_set: VelocitySet
        ):
        """
        Creates a boundary condition from a list of indices.
        """
        
        # Create a mask function
        @partial(jit)
        def set_mask(mask_id, id_number):
            return jnp.where(
                jnp.isin(indices, index),
                jnp.ones_like(indices),
                jnp.zeros_like(indices)
            )

        return cls(
            set_mask=set_mask,
            velocity_set=velocity_set,
        )

    @partial(jit, static_argnums=(0), donate_argnums=(1, 2, 3))
    def apply_jax(self, f_pre, f_post, mask):
        """
        mask is a boolean array of shape (x, y, z, 19)
        """

        indice_mask = jnp.any(mask, axis=-1)
        f_post = f_post.at[indice_mask].set(f_pre[indice_mask, self.velocity_set.opp_indices])
        return f_post
