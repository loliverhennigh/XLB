"""
Operator for generating uniform random values in a tensor.

This operator uses Warp's random number generator to fill a tensor with
uniform random values in a specified range. Each element gets a unique
random stream based on thread ID and seed.
"""

import warp as wp
from typing import Optional

from pumpkin_pulse.operator.operator import Operator

class UniformRandom(Operator):
    """
    Operator that fills a tensor with uniform random values.
    
    This operator provides GPU-accelerated random number generation using Warp's
    built-in random number generator. Each element in the output tensor gets a
    unique random stream based on the thread ID and seed.
    
    Example
    -------
    >>> # Fill a 4D tensor with random values
    >>> noise = wp.zeros((1, 32, 32, 32), dtype=wp.float32)
    >>> uniform_random = UniformRandom()
    >>> uniform_random(output=noise, low=-1.0, high=1.0)  # Fill with values in [-1, 1)
    """
    
    @staticmethod
    @wp.kernel
    def _fill_uniform_random_kernel(
        output: wp.array4d(dtype=wp.float32),
        seed: wp.int32,
        low: wp.float32,
        high: wp.float32,
        c: wp.int32,
        w: wp.int32,
        h: wp.int32,
        d: wp.int32
    ):
        """
        Fill a 4D tensor with uniform random values.

        Parameters
        ----------
        output : wp.array4d
            Output tensor to fill with random values
        seed : wp.int32
            Random seed for initialization
        low : wp.float32
            Lower bound of uniform distribution
        high : wp.float32
            Upper bound of uniform distribution
        c : wp.int32
            Number of channels
        w : wp.int32
            Width of tensor
        h : wp.int32
            Height of tensor
        d : wp.int32
            Depth of tensor
        """
        i, j, k, l = wp.tid()
        
        if i < c and j < w and k < h and l < d:
            # Calculate linear index for this 4D position
            idx = ((i * w + j) * h + k) * d + l
            
            # Initialize RNG with seed and index offset to ensure unique streams
            state = wp.rand_init(seed, idx)
            
            # Generate random value in [low, high)
            output[i, j, k, l] = wp.randf(state, low, high)
    
    def __call__(
        self,
        output: wp.array4d(dtype=wp.float32),
        seed: Optional[int] = None,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        """
        Fill a 4D tensor with uniform random values.

        Parameters
        ----------
        output : wp.array4d(dtype=wp.float32)
            Output 4D tensor (c, w, h, d) to fill with random values. Must be pre-allocated.
        seed : int, optional
            Random seed for initialization. If None, uses device time counter.
        low : float
            Lower bound of uniform distribution (inclusive)
        high : float
            Upper bound of uniform distribution (exclusive)
        """
        # Use device time counter if no seed provided
        if seed is None:
            seed = int(wp.get_device().get_time_counter())
        
        # Get dimensions
        c, w, h, d = output.shape
        
        # Launch kernel with 4D thread grid
        wp.launch(
            kernel=self._fill_uniform_random_kernel,
            dim=(c, w, h, d),
            inputs=[
                output,
                wp.int32(seed),
                wp.float32(low),
                wp.float32(high),
                wp.int32(c),
                wp.int32(w),
                wp.int32(h),
                wp.int32(d)
            ]
        ) 