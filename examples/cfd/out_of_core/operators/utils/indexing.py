from typing import Any
import warp as wp

@wp.func
def _python_mod(a: wp.int32, b: wp.int32):
    """
    Python modulo function.
    """
    mod = a % b
    if mod < 0:
        mod += b
    return mod

@wp.func
def periodic_indexing(
    data: wp.array4d(dtype=Any),
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    """
    Periodic indexing for 3D data.
    """
    i = (i + shape[0]) % shape[0]
    j = (j + shape[1]) % shape[1]
    k = (k + shape[2]) % shape[2]
    return data[c, i, j, k]

#@wp.overload
#def periodic_indexing(
#    data: wp.array4d(dtype=wp.float32),
#    shape: wp.vec3i,
#    c: wp.int32,
#    i: wp.int32,
#    j: wp.int32,
#    k: wp.int32,
#) -> wp.float32:
#    ...
#
#@wp.overload
#def periodic_indexing(
#    data: wp.array4d(dtype=wp.int16),
#    shape: wp.vec3i,
#    c: wp.int32,
#    i: wp.int32,
#    j: wp.int32,
#    k: wp.int32,
#) -> wp.int16:
#    ...
#
#
#def periodic_indexing_uint32(
#    data: wp.array4d(dtype=wp.uint32),
#    shape: wp.vec3i,
#    c: wp.int32,
#    i: wp.int32,
#    j: wp.int32,
#    k: wp.int32,
#) -> wp.uint32:
#    """
#    Periodic indexing for 3D data.
#    """
#    i = _python_mod(i, shape[0])
#    j = _python_mod(j, shape[1])
#    k = _python_mod(k, shape[2])
#    return data[c, i, j, k]

@wp.func
def periodic_atomic_add(
    data: wp.array4d(dtype=Any),
    value: Any,
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    """
    Periodic indexing for 3D data.
    """
    i = _python_mod(i, shape[0])
    j = _python_mod(j, shape[1])
    k = _python_mod(k, shape[2])
    wp.atomic_add(data, c, i, j, k, value)

@wp.func
def periodic_setting(
    data: wp.array4d(dtype=Any),
    value: Any,
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    """
    Periodic indexing for 3D data.
    """
    i = _python_mod(i, shape[0])
    j = _python_mod(j, shape[1])
    k = _python_mod(k, shape[2])
    data[c, i, j, k] = value

@wp.func
def pos_to_cell_index(
    pos: wp.vec3,
    origin: wp.vec3,
    spacing: wp.vec3,
    nr_ghost_cells: wp.int32,
):
    """
    Convert a position to nearest cell index.
    """

    float_ijk = wp.cw_div(pos - origin, spacing)
    cell_index = wp.vec3i(
        wp.int32(float_ijk[0]) + nr_ghost_cells,
        wp.int32(float_ijk[1]) + nr_ghost_cells,
        wp.int32(float_ijk[2]) + nr_ghost_cells,
    )
    return cell_index


@wp.func
def pos_to_lower_cell_index(
    pos: wp.vec3,
    origin: wp.vec3,
    spacing: wp.vec3,
    nr_ghost_cells: wp.int32,
):
    """
    Convert a position to lower cell index.
    """

    float_ijk = wp.cw_div(pos - origin, spacing)
    lower_cell_index = wp.vec3i(
        wp.int32(float_ijk[0] - 0.5) + nr_ghost_cells,
        wp.int32(float_ijk[1] - 0.5) + nr_ghost_cells,
        wp.int32(float_ijk[2] - 0.5) + nr_ghost_cells,
    )
    return lower_cell_index

def periodic_indexing(array, shape, component, i, j, k):
    """
    Perform periodic indexing on a 4D array.

    Parameters:
    - array: The 4D array to index.
    - shape: The shape of the 3D grid (excluding the component dimension).
    - component: The component index (0, 1, or 2 for x, y, z).
    - i, j, k: The indices to access, which will be wrapped around the boundaries.

    Returns:
    - The value at the specified index, considering periodic boundaries.
    """
    i = i % shape[0]
    j = j % shape[1]
    k = k % shape[2]
    return array[component, i, j, k]
