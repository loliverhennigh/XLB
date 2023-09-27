# Useful warp kernels

import warp as wp

@wp.kernel
def padded_array_to_array(
        dst_array: wp.array4d(dtype=float),
        src_array: wp.array4d(dtype=float),
        src_pad_0_0_0: wp.array4d(dtype=float),
        src_pad_0_0_1: wp.array4d(dtype=float),
        src_pad_0_0_2: wp.array4d(dtype=float),
        src_pad_0_1_0: wp.array4d(dtype=float),
        src_pad_0_1_1: wp.array4d(dtype=float),
        src_pad_0_1_2: wp.array4d(dtype=float),
        src_pad_0_2_0: wp.array4d(dtype=float),
        src_pad_0_2_1: wp.array4d(dtype=float),
        src_pad_0_2_2: wp.array4d(dtype=float),
        src_pad_1_0_0: wp.array4d(dtype=float),
        src_pad_1_0_1: wp.array4d(dtype=float),
        src_pad_1_0_2: wp.array4d(dtype=float),
        src_pad_1_1_0: wp.array4d(dtype=float),
        src_pad_1_1_2: wp.array4d(dtype=float),
        src_pad_1_2_0: wp.array4d(dtype=float),
        src_pad_1_2_1: wp.array4d(dtype=float),
        src_pad_1_2_2: wp.array4d(dtype=float),
        src_pad_2_0_0: wp.array4d(dtype=float),
        src_pad_2_0_1: wp.array4d(dtype=float),
        src_pad_2_0_2: wp.array4d(dtype=float),
        src_pad_2_1_0: wp.array4d(dtype=float),
        src_pad_2_1_1: wp.array4d(dtype=float),
        src_pad_2_1_2: wp.array4d(dtype=float),
        src_pad_2_2_0: wp.array4d(dtype=float),
        src_pad_2_2_1: wp.array4d(dtype=float),
        src_pad_2_2_2: wp.array4d(dtype=float),
        x_size: int,
        y_size: int,
        z_size: int,
        padding: int):

    q, i, j, k = wp.tid()

    # optimal if run through
    if (i < padding):
        if (j < padding):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_0_0_0[q, i, j, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_0_0_1[q, i, j, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_0_0_2[q, i, j, k - padding - z_size]
        elif (j < padding + y_size):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_0_1_0[q, i, j - padding, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_0_1_1[q, i, j - padding, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_0_1_2[q, i, j - padding, k - padding - z_size]
        else:
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_0_2_0[q, i, j - padding - y_size, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_0_2_1[q, i, j - padding - y_size, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_0_2_2[q, i, j - padding - y_size, k - padding - z_size]
    elif (i < padding + x_size):
        if (j < padding):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_1_0_0[q, i - padding, j, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_1_0_1[q, i - padding, j, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_1_0_2[q, i - padding, j, k - padding - z_size]
        elif (j < padding + y_size):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_1_1_0[q, i - padding, j - padding, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_array[q, i - padding, j - padding, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_1_1_2[q, i - padding, j - padding, k - padding - z_size]
        else:
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_1_2_0[q, i - padding, j - padding - y_size, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_1_2_1[q, i - padding, j - padding - y_size, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_1_2_2[q, i - padding, j - padding - y_size, k - padding - z_size]
    else:
        if (j < padding):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_2_0_0[q, i - padding - x_size, j, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_2_0_1[q, i - padding - x_size, j, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_2_0_2[q, i - padding - x_size, j, k - padding - z_size]
        elif (j < padding + y_size):
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_2_1_0[q, i - padding - x_size, j - padding, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_2_1_1[q, i - padding - x_size, j - padding, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_2_1_2[q, i - padding - x_size, j - padding, k - padding - z_size]
        else:
            if (k < padding):
                dst_array[q, i, j, k] = src_pad_2_2_0[q, i - padding - x_size, j - padding - y_size, k]
            elif (k < padding + z_size):
                dst_array[q, i, j, k] = src_pad_2_2_1[q, i - padding - x_size, j - padding - y_size, k - padding]
            else:
                dst_array[q, i, j, k] = src_pad_2_2_2[q, i - padding - x_size, j - padding - y_size, k - padding - z_size]


@wp.kernel
def array_to_padded_array(
        src_array: wp.array4d(dtype=float),
        dst_array: wp.array4d(dtype=float),
        dst_pad_0_0_0: wp.array4d(dtype=float),
        dst_pad_0_0_1: wp.array4d(dtype=float),
        dst_pad_0_0_2: wp.array4d(dtype=float),
        dst_pad_0_1_0: wp.array4d(dtype=float),
        dst_pad_0_1_1: wp.array4d(dtype=float),
        dst_pad_0_1_2: wp.array4d(dtype=float),
        dst_pad_0_2_0: wp.array4d(dtype=float),
        dst_pad_0_2_1: wp.array4d(dtype=float),
        dst_pad_0_2_2: wp.array4d(dtype=float),
        dst_pad_1_0_0: wp.array4d(dtype=float),
        dst_pad_1_0_1: wp.array4d(dtype=float),
        dst_pad_1_0_2: wp.array4d(dtype=float),
        dst_pad_1_1_0: wp.array4d(dtype=float),
        dst_pad_1_1_2: wp.array4d(dtype=float),
        dst_pad_1_2_0: wp.array4d(dtype=float),
        dst_pad_1_2_1: wp.array4d(dtype=float),
        dst_pad_1_2_2: wp.array4d(dtype=float),
        dst_pad_2_0_0: wp.array4d(dtype=float),
        dst_pad_2_0_1: wp.array4d(dtype=float),
        dst_pad_2_0_2: wp.array4d(dtype=float),
        dst_pad_2_1_0: wp.array4d(dtype=float),
        dst_pad_2_1_1: wp.array4d(dtype=float),
        dst_pad_2_1_2: wp.array4d(dtype=float),
        dst_pad_2_2_0: wp.array4d(dtype=float),
        dst_pad_2_2_1: wp.array4d(dtype=float),
        dst_pad_2_2_2: wp.array4d(dtype=float),
        x_size: int,
        y_size: int,
        z_size: int,
        padding: int):

    q, i, j, k = wp.tid()

    if (i < padding):
        if (j < padding):
            if (k < padding):
                dst_pad_0_0_0[q, i, j, k] = src_array[q, i + padding, j + padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_0_0_1[q, i, j, k - padding] = src_array[q, i + padding, j + padding, k]
            else:
                dst_pad_0_0_2[q, i, j, k - padding - z_size] = src_array[q, i + padding, j + padding, k - padding]
        elif (j < padding + y_size):
            if (k < padding):
                dst_pad_0_1_0[q, i, j - padding, k] = src_array[q, i + padding, j, k + padding]
            elif (k < padding + z_size):
                dst_pad_0_1_1[q, i, j - padding, k - padding] = src_array[q, i + padding, j, k]
            else:
                dst_pad_0_1_2[q, i, j - padding, k - padding - z_size] = src_array[q, i + padding, j, k - padding]
        else:
            if (k < padding):
                dst_pad_0_2_0[q, i, j - padding - y_size, k] = src_array[q, i + padding, j - padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_0_2_1[q, i, j - padding - y_size, k - padding] = src_array[q, i + padding, j - padding, k]
            else:
                dst_pad_0_2_2[q, i, j - padding - y_size, k - padding - z_size] = src_array[q, i + padding, j - padding, k - padding]
    elif (i < padding + x_size):
        if (j < padding):
            if (k < padding):
                dst_pad_1_0_0[q, i - padding, j, k] = src_array[q, i, j + padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_1_0_1[q, i - padding, j, k - padding] = src_array[q, i, j + padding, k]
            else:
                dst_pad_1_0_2[q, i - padding, j, k - padding - z_size] = src_array[q, i, j + padding, k - padding]
        elif (j < padding + y_size):
            if (k < padding):
                dst_pad_1_1_0[q, i - padding, j - padding, k] = src_array[q, i, j, k + padding]
            elif (k < padding + z_size):
                dst_array[q, i - padding, j - padding, k - padding] = src_array[q, i, j, k]
            else:
                dst_pad_1_1_2[q, i - padding, j - padding, k - padding - z_size] = src_array[q, i, j, k - padding]
        else:
            if (k < padding):
                dst_pad_1_2_0[q, i - padding, j - padding - y_size, k] = src_array[q, i, j - padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_1_2_1[q, i - padding, j - padding - y_size, k - padding] = src_array[q, i, j - padding, k]
            else:
                dst_pad_1_2_2[q, i - padding, j - padding - y_size, k - padding - z_size] = src_array[q, i, j - padding, k - padding]
    else:
        if (j < padding):
            if (k < padding):
                dst_pad_2_0_0[q, i - padding - x_size, j, k] = src_array[q, i - padding, j + padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_2_0_1[q, i - padding - x_size, j, k - padding] = src_array[q, i - padding, j + padding, k]
            else:
                dst_pad_2_0_2[q, i - padding - x_size, j, k - padding - z_size] = src_array[q, i - padding, j + padding, k - padding]
        elif (j < padding + y_size):
            if (k < padding):
                dst_pad_2_1_0[q, i - padding - x_size, j - padding, k] = src_array[q, i - padding, j, k + padding]
            elif (k < padding + z_size):
                dst_pad_2_1_1[q, i - padding - x_size, j - padding, k - padding] = src_array[q, i - padding, j, k]
            else:
                dst_pad_2_1_2[q, i - padding - x_size, j - padding, k - padding - z_size] = src_array[q, i - padding, j, k - padding]
        else:
            if (k < padding):
                dst_pad_2_2_0[q, i - padding - x_size, j - padding - y_size, k] = src_array[q, i - padding, j - padding, k + padding]
            elif (k < padding + z_size):
                dst_pad_2_2_1[q, i - padding - x_size, j - padding - y_size, k - padding] = src_array[q, i - padding, j - padding, k]
            else:
                dst_pad_2_2_2[q, i - padding - x_size, j - padding - y_size, k - padding - z_size] = src_array[q, i - padding, j - padding, k - padding]
