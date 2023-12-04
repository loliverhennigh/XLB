import numpy as np
import imageio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Parameters
    frame_resolution = 1024

    # Load data
    np_rho = np.load("triple_point.npy")

    # Zoom image
    print("Zooming image")
    zoom_factor = 63
    for i in range(zoom_factor):
        # Resize image
        resized_np_rho = np_rho[::(zoom_factor - i), ::(zoom_factor - i)]
        zoom_edges = (resized_np_rho.shape[0] - frame_resolution) // 2
        print(zoom_edges)
        resized_np_rho = resized_np_rho[zoom_edges:-zoom_edges, zoom_edges:-zoom_edges]

        # Save image
        plt.imsave(f"zoom_triple_point_{i:04d}.png", resized_np_rho, cmap="jet")
