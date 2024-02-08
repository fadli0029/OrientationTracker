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

def build_panorama(
    camera_dataset: dict,
    R,
    t_ts,
    dataset,
    prefix,
    panorama_images_folder_path,
    tracker
):
    """
    Create a panorama image from the camera dataset using the
    optimized quaternion.

    Args:
        camera_dataset:              dict, the camera dataset, shape (N, H, W, 3)
        q_optim:                     jnp.ndarray, shape (N, 4), the optimized quaternion
        t_ts:                        jnp.ndarray, shape (N, ), the timestamps of IMU
        dataset:                     str, the dataset name
        panorama_images_folder_path: str, the folder path to save the panorama images

    Returns:
        jnp.ndarray, shape (H, W, 3), the panorama image
    """
    cam_imgs, cam_ts = camera_dataset['cam'], camera_dataset['ts']
    N, H, W, _ = cam_imgs.shape

    uctr, vctr = H // 2, W // 2

    horizontal_angle = 60.
    vertical_angle = 45.

    pan_H, pan_W = 720, 1080
    panorama_image = np.zeros((pan_H+1, pan_W+1, 3), dtype=np.uint8)

    V, U = np.meshgrid(np.arange(W), np.arange(H))

    longitudes = (-(V - vctr) * horizontal_angle / W) * np.pi / 180.
    latitudes = ((U - uctr) * vertical_angle / H) * np.pi / 180.

    cartesian_coords = spherical2cartesian(longitudes, latitudes, 1.)

    for i in tqdm(range(N)):
        # Get closest-in-the-past timestamp
        idx = np.where((cam_ts[i] - t_ts) > 0)[0][-1]
        rot = R[idx]

        cartesian_coords_rot = np.matmul(cartesian_coords, rot.T)
        longitudes_rot, latitudes_rot = cartesian2spherical(cartesian_coords_rot)

        # Convert to pixel coordinates
        u_pixel = np.round(((longitudes_rot + np.pi/2) / np.pi) * pan_H).astype(np.int32)
        v_pixel = np.round(((np.pi - latitudes_rot) / (2 * np.pi)) * pan_W).astype(np.int32)

        if np.max(u_pixel) > pan_H or np.max(u_pixel) < 0 or np.min(u_pixel) < 0 or np.min(u_pixel) > pan_H:
            continue
        if np.max(v_pixel) > pan_W or np.max(v_pixel) < 0 or np.min(v_pixel) < 0 or np.min(v_pixel) > pan_W:
            continue

        panorama_image[u_pixel, v_pixel] = np.copy(cam_imgs[i])

    # Save the panorama image
    save_panorama_image(panorama_image, prefix, panorama_images_folder_path, dataset, tracker)
    return panorama_image

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
    file_path = folder_path + prefix + "_" + str(dataset) + "_" + trackers[tracker] + ".png"
    panorama_image = np.rot90(panorama_image, 1)
    plt.imsave(file_path, panorama_image)
    print(f"Panorama image saved to {file_path}\n")
