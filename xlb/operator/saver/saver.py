# Base class for all render operators
import numpy as np
import warp as wp

from xlb.velocity_set.velocity_set import VelocitySet
from xlb.operator.operator import Operator


class Saver(Operator):
    """
    Base class for all render operators
    """

    def __init__(
        self,
        velocity_set: VelocitySet = None,
        precision_policy=None,
        compute_backend=None,
    ):
        super().__init__(velocity_set, precision_policy, compute_backend)
