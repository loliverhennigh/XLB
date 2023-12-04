"""
Base class for boundary conditions in a LBM simulation.
"""

import jax.numpy as jnp
from jax import jit, device_count
from functools import partial
import numpy as np

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.operator.stream.stream import Stream
from xlb.operator.boundary_condition.boundary_condition import (
    BoundaryCondition,
    ImplementationStep,
)

class HalfwayBounceBack(BoundaryCondition):
    """
    Halfway Bounce-back boundary condition for a lattice Boltzmann method simulation.

    This boundary condition is implemented as a full bounce-back followed by a streaming step.
    This is equivalent to a halfway bounce-back
    ----------------------------------------
    |     ->     |     ->     |     ->     |
    |     <-     |     <-     |     <-     |
    ----------------------------------------
    """

    def __init__(self, indices):
        super().__init__(indices, solid=True, implementation_step=ImplementationStep.STREAMING)
        self.stream = Stream(velocity_set)

    @partial(jit, static_argnums=(0, 4))
    def apply_jax(self, f_pre, f_post, mask, velocity_set: VelocitySet):
        f_full_bounce_back = f_post[velocity_set.opp_indices]
        f_halfway_bounce_back = self.stream.apply_jax(f_full_bounce_back, velocity_set)
        return f_halfway_bounce_back
