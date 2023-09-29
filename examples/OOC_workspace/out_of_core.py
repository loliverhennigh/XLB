# Out-of-core decorator for functions that take a lot of memory

import functools
import warp as wp
import cupy as cp
import jax.dlpack as jdlpack
import numpy as np

from ooc_array import OOCArray


def _cupy_to_backend(cupy_array, backend):
    # Convert cupy array to backend array
    dl_array = cupy_array.toDlpack()
    if backend == "jax":
        backend_array = jdlpack.from_dlpack(dl_array)
    elif backend == "warp":
        backend_array = wp.from_dlpack(dl_array)
    elif backend == "cupy":
        backend_array = cupy_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    return backend_array


def _backend_to_cupy(backend_array, backend):
    # Convert backend array to cupy array
    if backend == "jax":
        dl_array = jdlpack.to_dlpack(backend_array)
    elif backend == "warp":
        dl_array = wp.to_dlpack(backend_array)
    elif backend == "cupy":
        return backend_array
    else:
        raise ValueError(f"Backend {backend} not supported")
    cupy_array = cp.fromDlpack(dl_array)
    return cupy_array


def OOCmap(comm, ref_args, add_index=False, backend="jax"):
    """Decorator for out-of-core functions.

    Parameters
    ----------
    comm : MPI communicator
        The MPI communicator. (TODO add functionality)
    ref_args : List[int]
        The indices of the arguments that are OOC arrays to be written to by outputs of the function.
    add_index : bool, optional
        Whether to add the index of the global array to the arguments of the function. Default is False.
        If true the function will take in a tuple of (array, index) instead of just the array.
    backend : str, optional
        The backend to use for the function. Default is 'jax'.
        Options are 'jax' and 'warp'.
    """

    def decorator(func):
        def wrapper(*args):
            # Get MIP rank and size
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Get list of OOC arrays
            ooc_array_args = []
            for arg in args:
                if isinstance(arg, OOCArray):
                    ooc_array_args.append(arg)

            # Check that all ooc arrays are compatible
            # TODO: Add better checks
            for ooc_array in ooc_array_args:
                if ooc_array_args[0].tile_dims != ooc_array.tile_dims:
                    raise ValueError(
                        f"Tile dimensions of ooc arrays do not match. {ooc_array_args[0].tile_dims} != {ooc_array.tile_dims}"
                    )

            # Apply the function to each of the ooc arrays
            for tile_index in ooc_array_args[0].tiles.keys():
                # Run through args and kwargs and replace ooc arrays with their compute arrays
                new_args = []
                for arg in args:
                    if isinstance(arg, OOCArray):
                        # Get the compute array (this performs all the memory copies)
                        compute_array, global_index = arg.get_compute_array(tile_index)

                        # Convert to backend array
                        compute_array = _cupy_to_backend(compute_array, backend)

                        # Add index to the arguments if requested
                        if add_index:
                            compute_array = (compute_array, global_index)

                        new_args.append(compute_array)
                    else:
                        new_args.append(arg)

                # Run the function
                results = func(*new_args)

                # Convert the results to a tuple if not already
                if not isinstance(results, tuple):
                    results = (results,)

                # Convert the results back to cupy arrays
                results = tuple(
                    [_backend_to_cupy(result, backend) for result in results]
                )

                # Write the results back to the ooc array
                for arg_index, result in zip(ref_args, results):
                    args[arg_index].set_tile(result, tile_index)

            # Syncronize all processes
            cp.cuda.Device().synchronize()
            comm.Barrier()

            # Update the ooc arrays padding
            for ooc_array in ooc_array_args:
                ooc_array.update_padding()

            # Return OOC arrays
            if len(ref_args) == 1:
                return ooc_array_args[ref_args[0]]
            else:
                return tuple([args[arg_index] for arg_index in ref_args])

        return wrapper

    return decorator
