import warp as wp

@wp.func
def minmod_slope_limiter(
    v_0: wp.float32,
    v_1: wp.float32,
    v_2: wp.float32,
    v_dx: wp.float32,
    spacing: wp.float32,
    epsilon: wp.float32,
):

    # Get minmod
    if v_dx == 0.0:
        denominator = epsilon
    else:
        denominator = v_dx
    v_dx = (
        wp.max(
            0.0,
            wp.min(
                1.0,
                ((v_1 - v_0) / spacing) / denominator,
            ),
        )
        * v_dx
    )
    if v_dx == 0.0:
        denominator = epsilon
    else:
        denominator = v_dx
    v_dx = (
        wp.max(
            0.0,
            wp.min(
                1.0,
                ((v_2 - v_1) / spacing) / denominator,
            ),
        )
        * v_dx
    )

    return v_dx
