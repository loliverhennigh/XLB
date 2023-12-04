# Used to keep track of the physics types

from enum import Enum


class PhysicsType(Enum):
    NSE = 1  # Navier-Stokes Equations
    ADE = 2  # Advection-Diffusion Equations
