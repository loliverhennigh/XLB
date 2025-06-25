# Base class for all stepper operators
from xlb.operator import Operator
from xlb import DefaultConfig


class Stepper(Operator):
    """
    Class that handles the construction of lattice boltzmann stepping operator
    """

    def __init__(self, grid, boundary_conditions, velocity_set=None, precision_policy=None, compute_backend=None):
        self.grid = grid
        self.boundary_conditions = boundary_conditions
        # Get velocity set, precision policy, and compute backend
        self.velocity_set = velocity_set or DefaultConfig.velocity_set
        self.precision_policy = precision_policy or DefaultConfig.default_precision_policy
        self.compute_backend = compute_backend or DefaultConfig.default_backend

        # Initialize operator
        super().__init__(self.velocity_set, self.precision_policy, self.compute_backend)

    def prepare_fields(self, initializer=None):
        """Initialize the fields required for the stepper.

        Args:
            initializer: Optional operator to initialize the distribution functions.
                        If provided, it should be a callable that takes (grid, velocity_set,
                        precision_policy, compute_backend) as arguments and returns initialized f_0.
                        If None, default equilibrium initialization is used with rho=1 and u=0.

        Returns:
            Tuple of (f_0, f_1, bc_mask, missing_mask)
        """
        raise NotImplementedError("Subclasses must implement prepare_fields()")
