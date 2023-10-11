import numpy as np
import imageio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':

    # Parameters
    frame_resolution = 1024

    # Load data
    np_rho = np.load("triple_point.npy")[:-2048, :-2048]
    print("np_rho.shape", np_rho.shape)

    # Normalize 0-1
    np_rho = (np_rho - np_rho.min()) / (np_rho.max() - np_rho.min())

    # Get Jet colormap
    np_rgb = np.zeros((np_rho.shape[0], np_rho.shape[1], 3)).astype(np.uint8)

    # Run through array with tiles
    tile_size = 1024
    for i in tqdm(range(0, np_rgb.shape[0], tile_size)):
        for j in range(0, np_rgb.shape[1], tile_size):
            # Get rgb tile
            np_rgb_tile = (255.0 * plt.cm.jet(np_rho[i:i + tile_size, j:j + tile_size])[:, :, :3]).astype(np.uint8)

            # Set rgb tile
            np_rgb[i:i + tile_size, j:j + tile_size] = np_rgb_tile

    # Zoom image
    print("Zooming image")
    i = 0
    while True:
        # Coarse zoom
        coarsen_factor = np_rgb.shape[0] // frame_resolution
        resized_np_rgb = np_rgb[::coarsen_factor, ::coarsen_factor]

        # Zoom
        zoom_factor = frame_resolution / resized_np_rgb.shape[0]
        resized_np_rgb = zoom(resized_np_rgb, (zoom_factor, zoom_factor, 1), order=1)

        # Save
        imageio.imwrite("zoomed_triple_point_{:04d}.png".format(i), resized_np_rgb)
        i += 1

        # Frame resolution zoom
        pixel_zoom_edges = int(np_rgb.shape[0] / 10)
        np_rgb = np_rgb[pixel_zoom_edges:-pixel_zoom_edges, pixel_zoom_edges:-pixel_zoom_edges]

        if np_rgb.shape[0] <= frame_resolution:
            break
