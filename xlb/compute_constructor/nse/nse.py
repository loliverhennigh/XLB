# Class that handles the construction of lattice boltzmann compute functions
# for the Navier-Stokes equations

import jax.numpy as jnp
from jax import jit
from functools import partial

from xlb.compute_constructor.compute_constructor import ComputeConstructor
from xlb.operator.stream import Stream
from xlb.operator.equilibrium import QuadraticEquilibrium
from xlb.operator.macroscopic import Macroscopic


class NSE(ComputeConstructor):
    """
    Class that handles the construction of lattice boltzmann compute functions
    for the Navier-Stokes equations
    """

    def __init__(
        self,
        velocity_set,
        collision,
        stream=Stream(),
        equilibrium=QuadraticEquilibrium(),
        macroscopic=Macroscopic(),
        boundary_conditions=[],
        forcing=None,
        precision_policy=None,
    ):
        super().__init__(
            velocity_set,
            collision,
            stream,
            equilibrium,
            macroscopic,
            boundary_conditions=boundary_conditions,
            forcing=forcing,
            precision_policy=precision_policy,
        )

        # TODO: Check if all operators are compatible with the Navier-Stokes equations
