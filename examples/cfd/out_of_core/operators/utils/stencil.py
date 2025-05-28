import warp as wp

from pumpkin_pulse.functional.indexing import periodic_indexing, periodic_indexing_uint8

# 7 point stencil
# 0: ( 0, 0, 0)
# 1: (-1, 0, 0)
# 2: ( 1, 0, 0)
# 3: ( 0,-1, 0)
# 4: ( 0, 1, 0)
# 5: ( 0, 0,-1)
# 6: ( 0, 0, 1)
p7_float32_stencil_type = wp.vec(7, wp.float32)
p7_uint8_stencil_type = wp.vec(7, wp.uint8)
p7_vec3f_stencil_type = wp.mat((7, 3), wp.float32)

# 4 point stencil
# 0: ( 0, 0, 0)
# 1: (-1, 0, 0)
# 2: ( 0,-1, 0)
# 3: ( 0, 0,-1)
p4_float32_stencil_type = wp.vec(4, wp.float32)
p4_uint8_stencil_type = wp.vec(4, wp.uint8)
p4_vec3f_stencil_type = wp.mat((4, 3), wp.float32)

# 2 point stencil
p2_float32_stencil_type = wp.vec(2, wp.float32)
p2_uint8_stencil_type = wp.vec(2, wp.uint8)
p2_vec3f_stencil_type = wp.mat((2, 3), wp.float32)

# lower faces
# 0: right (-1, 0, 0)
# 1: left  (-1, 0, 0)
# 2: right ( 0,-1, 0)
# 3: left  ( 0,-1, 0)
# 4: right ( 0, 0,-1)
# 5: left  ( 0, 0,-1)
faces_float32_type = wp.vec(6, wp.float32)
faces_2_float32_type = wp.vec(2, wp.float32)

@wp.func
def get_p7_float32_stencil(
    data: wp.array4d(dtype=wp.float32),
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    stencil = p7_float32_stencil_type(
        periodic_indexing(data, shape, c, i + 0, j - 0, k - 0),
        periodic_indexing(data, shape, c, i - 1, j + 0, k - 0),
        periodic_indexing(data, shape, c, i + 1, j + 0, k - 0),
        periodic_indexing(data, shape, c, i + 0, j - 1, k + 0),
        periodic_indexing(data, shape, c, i + 0, j + 1, k + 0),
        periodic_indexing(data, shape, c, i + 0, j + 0, k - 1),
        periodic_indexing(data, shape, c, i + 0, j + 0, k + 1),
    )
    return stencil

@wp.func
def get_p4_p7_float32_stencil(
    data: wp.array4d(dtype=wp.float32),
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    # Center p7
    v_2_2_2 = periodic_indexing(data, shape, c, i + 0, j - 0, k - 0)
    v_1_2_2 = periodic_indexing(data, shape, c, i - 1, j - 0, k - 0)
    v_3_2_2 = periodic_indexing(data, shape, c, i + 1, j - 0, k - 0)
    v_2_1_2 = periodic_indexing(data, shape, c, i + 0, j - 1, k - 0)
    v_2_3_2 = periodic_indexing(data, shape, c, i + 0, j + 1, k - 0)
    v_2_2_1 = periodic_indexing(data, shape, c, i + 0, j - 0, k - 1)
    v_2_2_3 = periodic_indexing(data, shape, c, i + 0, j - 0, k + 1)

    # X p7
    v_0_2_2 = periodic_indexing(data, shape, c, i - 2, j - 0, k - 0)
    v_3_2_2 = periodic_indexing(data, shape, c, i + 1, j - 0, k - 0)
    v_1_1_2 = periodic_indexing(data, shape, c, i - 1, j - 1, k - 0)
    v_1_3_2 = periodic_indexing(data, shape, c, i - 1, j + 1, k - 0)
    v_1_2_1 = periodic_indexing(data, shape, c, i - 1, j - 0, k - 1)
    v_1_2_3 = periodic_indexing(data, shape, c, i - 1, j - 0, k + 1)

    # Y p7
    v_3_1_2 = periodic_indexing(data, shape, c, i + 1, j - 1, k - 0)
    v_2_0_2 = periodic_indexing(data, shape, c, i + 0, j - 2, k - 0)
    v_2_1_1 = periodic_indexing(data, shape, c, i + 0, j - 1, k - 1)
    v_2_1_3 = periodic_indexing(data, shape, c, i + 0, j - 1, k + 1)

    # Z p7
    v_3_2_1 = periodic_indexing(data, shape, c, i + 1, j - 0, k - 1)
    v_2_3_1 = periodic_indexing(data, shape, c, i + 0, j + 1, k - 1)
    v_2_2_0 = periodic_indexing(data, shape, c, i + 0, j - 0, k - 2)

    # Make stencil
    stencil_2_2_2 = p7_float32_stencil_type(
        v_2_2_2,
        v_1_2_2,
        v_3_2_2,
        v_2_1_2,
        v_2_3_2,
        v_2_2_1,
        v_2_2_3,
    )
    stencil_1_2_2 = p7_float32_stencil_type(
        v_1_2_2,
        v_0_2_2,
        v_3_2_2,
        v_1_1_2,
        v_1_3_2,
        v_1_2_1,
        v_1_2_3,
    )
    stencil_2_1_2 = p7_float32_stencil_type(
        v_2_1_2,
        v_1_1_2,
        v_3_1_2,
        v_2_0_2,
        v_2_2_2,
        v_2_1_1,
        v_2_1_3,
    )
    stencil_2_2_1 = p7_float32_stencil_type(
        v_2_2_1,
        v_1_2_1,
        v_3_2_1,
        v_2_1_1,
        v_2_3_1,
        v_2_2_0,
        v_2_2_2,
    )
    return stencil_2_2_2, stencil_1_2_2, stencil_2_1_2, stencil_2_2_1


@wp.func
def get_p7_uint8_stencil(
    data: wp.array4d(dtype=wp.uint8),
    shape: wp.vec3i,
    c: wp.int32,
    i: wp.int32,
    j: wp.int32,
    k: wp.int32,
):
    stencil = p7_uint8_stencil_type(
        periodic_indexing_uint8(data, shape, c, i + 0, j - 0, k - 0),
        periodic_indexing_uint8(data, shape, c, i - 1, j + 0, k - 0),
        periodic_indexing_uint8(data, shape, c, i + 1, j + 0, k - 0),
        periodic_indexing_uint8(data, shape, c, i + 0, j - 1, k + 0),
        periodic_indexing_uint8(data, shape, c, i + 0, j + 1, k + 0),
        periodic_indexing_uint8(data, shape, c, i + 0, j + 0, k - 1),
        periodic_indexing_uint8(data, shape, c, i + 0, j + 0, k + 1),
    )
    return stencil

@wp.func
def p4_stencil_to_faces(
    v_stencil: p4_float32_stencil_type,
    v_stencil_dxyz: p4_vec3f_stencil_type,
    spacing: wp.vec3f,
):
    faces = faces_float32_type()
    faces[0] = v_stencil[0] + 0.5 * spacing[0] * v_stencil_dxyz[0, 0]
    faces[1] = v_stencil[1] - 0.5 * spacing[0] * v_stencil_dxyz[1, 0]
    faces[2] = v_stencil[0] + 0.5 * spacing[1] * v_stencil_dxyz[0, 1]
    faces[3] = v_stencil[2] - 0.5 * spacing[1] * v_stencil_dxyz[2, 1]
    faces[4] = v_stencil[0] + 0.5 * spacing[2] * v_stencil_dxyz[0, 2]
    faces[5] = v_stencil[3] - 0.5 * spacing[2] * v_stencil_dxyz[3, 2]
    return faces


@wp.func
def p2_stencil_to_faces(
    v_stencil: p2_float32_stencil_type,
    v_stencil_dxyz: p2_vec3f_stencil_type,
    spacing: wp.vec3f,
):
    faces = faces_2_float32_type()
    faces[0] = v_stencil[0] + 0.5 * spacing[0] * v_stencil_dxyz[0, 0]
    faces[1] = v_stencil[1] - 0.5 * spacing[0] * v_stencil_dxyz[1, 0]
    return faces
