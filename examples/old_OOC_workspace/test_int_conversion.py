import warp as wp
import matplotlib.pyplot as plt

wp.init()

@wp.func
def int_to_float(x: wp.int32):
    """Convert a 32-bit int to a 32-bit float using math."""
    # Extract components
    sign = (x >> 31) & 1
    exp = ((x >> 23) & 0xFF) - 127
    fraction = x & ((1 << 23) - 1)
    
    # Compute the floating point value
    f = 1.0 + float(fraction) / float(1 << 23)
    f *= (2.0 ** float(exp))
    
    if sign:
        f = -f
    
    return f

@wp.func
def float_to_int(x: wp.float32):
    """Convert a 32-bit float to a 32-bit int using math."""
    # Handle special case for 0
    if x == 0.0:
        return 0
    
    # Extract sign bit
    if x < 0:
        sign = 1
        x = -x
    else:
        sign = 0
    
    # Extract exponent and fraction
    exp = wp.int(127)  # Bias for IEEE 754 32-bit float
    while x >= 2.0:
        x /= 2.0
        exp += 1
    while x < 1.0:
        x *= 2.0
        exp -= 1
    
    fraction = int((x - 1.0) * float((1 << 23)))
    
    # Construct the int using the sign, exponent, and fraction
    i = (sign << 31) | (exp << 23) | fraction

    return wp.int32(i)

@wp.kernel
def init_float_kernel(
    x: wp.array2d(dtype=wp.float32),
):

    # Get indices
    i, j = wp.tid()

    # Convert int to float
    x[i, j] = float(i + j)

@wp.kernel
def int_to_float_kernel(
    x: wp.array2d(dtype=wp.int32),
    y: wp.array2d(dtype=wp.float32),
):

    # Get indices
    i, j = wp.tid()

    # Convert int to float
    y[i, j] = int_to_float(x[i, j])

@wp.kernel
def float_to_int_kernel(
    x: wp.array2d(dtype=wp.float32),
    y: wp.array2d(dtype=wp.int32),
):
    
    # Get indices
    i, j = wp.tid()

    # Convert float to int
    y[i, j] = float_to_int(x[i, j])



if __name__ == '__main__':

    # Make arrays
    float_array = wp.empty([10, 10], dtype=wp.float32)
    int_array = wp.empty([10, 10], dtype=wp.int32)

    # Initialize float array
    wp.launch(
        kernel=init_float_kernel,
        dim=(10, 10),
        inputs=[float_array],
    )
    np_float_array_before = float_array.numpy()

    # Convert float to int
    wp.launch(
        kernel=float_to_int_kernel,
        dim=(10, 10),
        inputs=[float_array, int_array],
    )
    np_int_array = int_array.numpy()

    # Convert int to float
    wp.launch(
        kernel=int_to_float_kernel,
        dim=(10, 10),
        inputs=[int_array, float_array],
    )
    np_float_array_after = float_array.numpy()

    # Plot with colorbars
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(np_float_array_before)
    axs[0].set_title('Before')
    axs[1].imshow(np_int_array)
    axs[1].set_title('Int')
    axs[2].imshow(np_float_array_after)
    axs[2].set_title('After')
    axs[3].imshow(np_float_array_after - np_float_array_before)
    axs[3].set_title('Difference')
    plt.savefig('test_int_conversion.png')
