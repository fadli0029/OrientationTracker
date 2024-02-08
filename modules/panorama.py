# -------------------------------------------------------------------------
# Author: muhammad fadli alim arsani
# Email: fadlialim0029[at]gmail.com
# File: jax_quaternion.py
# Description: this file contains the implementation of the
#              quaternion kinematics and the optimization
#              of the motion model using the projected gradient descent
#              (PGD) algorithm.
# Misc: this is also part of one of the projects in the course
#       "sensing and estimation in robotics" taught by prof. nikolay
#       atanasov @uc san diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import os
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.quaternions import quat2mat
from .utils import quat2rot

def build_panorama(camera_dataset: dict, R, t_ts, dataset, prefix, panorama_images_folder_path, tracker):
    cam_imgs, cam_ts = camera_dataset['cam'], camera_dataset['ts']
    N, H, W, _ = cam_imgs.shape

    screen_height = 600
    screen_width = int(np.ceil(2 * np.pi * 230)) + 2
    offset_x = screen_width // 2
    offset_y = screen_height // 2
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    for i in tqdm(range(N)):
        img_i = np.searchsorted(t_ts, cam_ts[i], side='right') - 1
        if img_i >= len(R):
            continue

        img = cam_imgs[i]
        dcm = R[img_i]

        x_img, y_img = np.meshgrid(np.arange(W) - W / 2, np.arange(H) - H / 2)
        z_img = np.ones_like(x_img) * 230

        pts_img = np.vstack((z_img.ravel(), -x_img.ravel(), -y_img.ravel()))
        pts_img_rot = np.dot(dcm, pts_img)

        a = np.arctan2(pts_img_rot[1], pts_img_rot[0])
        b = np.arctan2(pts_img_rot[2], np.sqrt(pts_img_rot[0]**2 + pts_img_rot[1]**2))

        proj_x = np.round(-230 * a + offset_x).astype(int)
        proj_y = np.round(-230 * b + offset_y).astype(int)

        # Ensure coordinates are within canvas bounds
        valid_indices = (proj_x >= 0) & (proj_x < screen_width) & (proj_y >= 0) & (proj_y < screen_height)

        # Convert to integer for indexing
        img_x_indices = (x_img.ravel()[valid_indices] + W // 2).astype(int)
        img_y_indices = (y_img.ravel()[valid_indices] + H // 2).astype(int)

        canvas[proj_y[valid_indices], proj_x[valid_indices]] = img[img_y_indices, img_x_indices]

    # Save the panorama image
    save_panorama_image(canvas, prefix, panorama_images_folder_path, dataset, tracker)

def spherical2cartesian(longitude, latitude, r=1.0):
    H, W = latitude.shape

    cartesian_coords = np.zeros((H, W, 3))
    cartesian_coords[:, :, 0] = r * np.cos(latitude) * np.cos(longitude)
    cartesian_coords[:, :, 1] = r * np.cos(latitude) * np.sin(longitude)
    cartesian_coords[:, :, 2] = -1 * r * np.sin(latitude)
    return cartesian_coords

def cartesian2spherical(cartesian_coords, r=1.0):
    X, Y, Z = cartesian_coords[:, :, 0], cartesian_coords[:, :, 1], cartesian_coords[:, :, 2]

    latitudes = np.arcsin(-Z / r)
    longitudes = np.arctan2(Y, X)
    return longitudes, latitudes

def save_panorama_image(panorama_image, prefix, folder_path, dataset, tracker):
    """
    Save the panorama image to the file path.

    Args:
        panorama_image: jnp.ndarray, shape (H, W, 3), the panorama image
        file_path:      str, the file path to save the panorama image
    """
    trackers = {
        'pgd': 'PGD',
        'ekf': 'EKF',
    }

    # If the folder path does not exist, create the folder
    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(os.path.dirname(folder_path))
    if prefix == "vicon":
        file_path = folder_path + prefix + "_" + str(dataset) + ".png"
    else:
        file_path = folder_path + prefix + "_" + str(dataset) + "_" + trackers[tracker] + ".png"
    # panorama_image = np.rot90(panorama_image, 1)
    plt.imsave(file_path, panorama_image)
    print(f"Panorama image saved to {file_path}\n")
