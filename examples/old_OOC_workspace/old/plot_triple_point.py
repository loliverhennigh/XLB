import numpy as np
import imageio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':

    # Parameters
    frame_resolution = 1024

    # Load data
    #np_rho = np.load("triple_point.npy")[:-2*2048, :-2*2048]
    np_rho = np.load("triple_point.npy")
    print("np_rho.shape", np_rho.shape)

    # Normalize 0-1
    np_rho = (np_rho - np_rho.min()) / (np_rho.max() - np_rho.min())

    # Zoom image
    print("Zooming image")
    i = 0
    while True:
        # Coarse zoom
        coarsen_factor = np_rho.shape[0] // frame_resolution
        resized_np_rho = np_rho[::coarsen_factor, ::coarsen_factor]

        # Zoom
        zoom_factor = frame_resolution / resized_np_rho.shape[0]
        resized_np_rho = zoom(resized_np_rho, (zoom_factor, zoom_factor), order=1)

        # Save
        print("Saving image")
        resized_np_rho = (resized_np_rho - resized_np_rho.min()) / (resized_np_rho.max() - resized_np_rho.min())
        resized_np_rgb = (255 * plt.cm.jet(resized_np_rho)[..., :3]).astype(np.uint8)
        imageio.imwrite("zoomed_2_triple_point_{:04d}.png".format(i), resized_np_rgb)
        i += 1

        # Frame resolution zoom
        pixel_zoom_edges = int(0.01 * np_rho.shape[0])
        pixel_zoom_edges_top = int(2.0 * 0.15 * pixel_zoom_edges)
        pixel_zoom_edges_bottom = 2 * pixel_zoom_edges - pixel_zoom_edges_top
        print("pixel_zoom_edges", pixel_zoom_edges)
        print("pixel_zoom_edges_top", pixel_zoom_edges_top)
        print("pixel_zoom_edges_bottom", pixel_zoom_edges_bottom)
        #np_rho = np_rho[pixel_zoom_edges_top:-pixel_zoom_edges_bottom, pixel_zoom_edges_top:-pixel_zoom_edges_bottom]
        #np_rho = np_rho[pixel_zoom_edges_top:-pixel_zoom_edges_bottom, pixel_zoom_edges_top:-pixel_zoom_edges_bottom]
        print(np_rho.shape)
        np_rho = np_rho[pixel_zoom_edges_top:-pixel_zoom_edges_bottom, pixel_zoom_edges:-pixel_zoom_edges]
        print(np_rho.shape)

        if np_rho.shape[0] <= frame_resolution:
            break
