import warp as wp

@wp.func
def centeral_difference(
    v_0_1_1: wp.float32,
    v_2_1_1: wp.float32,
    v_1_0_1: wp.float32,
    v_1_2_1: wp.float32,
    v_1_1_0: wp.float32,
    v_1_1_2: wp.float32,
    spacing: wp.vec3f,
) -> wp.vec3:
    return wp.vec3f(
        (v_2_1_1 - v_0_1_1) / (2.0 * spacing[0]),
        (v_1_2_1 - v_1_0_1) / (2.0 * spacing[1]),
        (v_1_1_2 - v_1_1_0) / (2.0 * spacing[2]),
    )
