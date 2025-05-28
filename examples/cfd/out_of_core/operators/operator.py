"""
Operator Base Class

This module defines the base class for all operators in the Pumpkin Pulse project. The `Operator`
class provides a common interface and structure for implementing specific computational tasks.

Classes:
    Operator: Base class for all operators.

Methods:
    __call__: Apply the operator to input data. This method should be overridden by subclasses to
              implement specific operations.

Usage:
    class CustomOperator(Operator):
        def __call__(self, *args, **kwargs):
            # Implement custom operation

Input Expectations:
    - All inputs to operators are expected to be Warp arrays or simple data types such as floats and tuples.
    - Complex data structures should be avoided to maintain simplicity and efficiency in kernel execution.
"""

# Base class for all operators

class Operator:
    """
    Base class for all operators
    """

    def __call__(self, *args, **kwargs):
        """
        Apply the operator to a input. This method will call the
        appropriate apply method based on the compute backend.
        """
