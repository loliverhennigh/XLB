import warp as wp
import time
from tqdm import tqdm
import numpy as np

wp.init()

# Make push kernel
@wp.kernel
def push(
    pos: wp.array(dtype=wp.vec3),
):

    # Get particle index
    i = wp.tid()
    r = wp.rand_init(i)

    # Get particle data         
    pos[i] = wp.vec3(wp.randn(r), wp.randn(r), wp.randn(r))

if __name__ == "__main__":

    # Create particles
    nr_particles = 100000000
    pos = wp.zeros(nr_particles, dtype=wp.vec3)

    # Compute number of particles to add each cel
    wp.launch(
        push,
        inputs=[
            pos,
        ],
        dim=nr_particles,
    )
    values = pos.numpy()
    if np.any(np.isinf(values)):
        print("Infs in values")

