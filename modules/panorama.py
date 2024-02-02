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
import jax.numpy as jnp
import matplotlib.pyplot as plt
from transforms3d.quaternions import quat2mat

def build_panorama(
    camera_dataset: dict,
    q_optim,
    t_ts,
    dataset,
    panorama_images_folder_path
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
    cam_img, cam_ts = camera_dataset["cam"], camera_dataset["ts"]
    N, H, W, C = cam_img.shape

    pH, pW = 720, 1080
    panorama_image = jnp.zeros((int(pH), int(pW), 3))

    def degree(x):
        return x * (180 / jnp.pi)
    theta, phi = jnp.meshgrid(
        jnp.linspace(degree(-22.5), degree(22.5), H),
        jnp.linspace(degree(-30), degree(30), W)
    )
    r = jnp.ones((H, W))
    spherical_coords = jnp.stack([theta.T, phi.T, r], axis=-1)
    cartesian_coords = spherical2cartesian(*spherical_coords.T)
    cartesian_coords = jnp.stack(cartesian_coords, axis=-1)

    pbar = tqdm(range(N), desc="==========> 📸  Building panorama images", unit="image")
    for i in pbar:
        iter_start = time.time()
        pbar.set_description(f"==========> 📸  Building panorama image for dataset {dataset}")

        # Compute the rotation matrix and apply it to cartesian coordinates
        # (we use closest-in-the-past timestamp (see project1 desc.))
        R_wc = quat2mat(q_optim[jnp.argmax(t_ts>cam_ts[i])])

        cartesian_coords_w = R_wc @ cartesian_coords.reshape((-1, 3)).T + jnp.repeat(jnp.array([0, 0, 0.1]).reshape((-1, 1)), H*W, axis=1)
        cartesian_coords_w = cartesian_coords_w.T.reshape((H, W, 3))

        spherical_coords_w = cartesian2spherical(
            cartesian_coords_w[:, :, 0], cartesian_coords_w[:, :, 1], cartesian_coords_w[:, :, 2]
        )
        spherical_coords_w = jnp.stack(spherical_coords_w, axis=-1)
        spherical_coords_w = spherical_coords_w[:, :, 0:2]

        # Project spherical coordinates to the panorama image plane
        spherical_coords_w = spherical_coords_w.at[:, :, 0].set((jnp.pi/2 + spherical_coords_w[:, :, 0]) / jnp.pi)
        spherical_coords_w = spherical_coords_w.at[:, :, 1].set((jnp.pi + spherical_coords_w[:, :, 1]) / (2 * jnp.pi))

        spherical_coords_w = spherical_coords_w.at[:, :, 0].set(spherical_coords_w[:, :, 0]*pH)
        spherical_coords_w = spherical_coords_w.at[:, :, 1].set(spherical_coords_w[:, :, 1]*pW)

        # convert to ints for indexing
        cylindrical_coords_w = spherical_coords_w.astype(int)

        # copy image to panorama
        panorama_image = panorama_image.astype(int)
        panorama_image = panorama_image.at[cylindrical_coords_w[:, :, 0], cylindrical_coords_w[:, :, 1]].set(cam_img[i, :, :, 0:3])

        iter_end = time.time()
        iter_duration = iter_end - iter_start
        pbar.set_postfix(time=f"{iter_duration:.4f}s")

    save_panorama_image(panorama_image, panorama_images_folder_path, dataset)
    return panorama_image

def spherical2cartesian(theta, phi, r):
    """
    Convert the spherical coordinate to cartesian coordinate.

    Args:
        theta: float, the azimuthal angle
        phi:   float, the polar angle
        r:     float, the radius

    Returns:
        x: float, the x coordinate
        y: float, the y coordinate
        z: float, the z coordinate
    """
    x = r * jnp.cos(phi) * jnp.cos(theta)
    y = -r * jnp.cos(theta) * jnp.sin(phi)
    z = -r * jnp.sin(theta)
    return x, y, z

def cartesian2spherical(x, y, z):
    """
    Convert the cartesian coordinate to spherical coordinate.

    Args:
        x: float, the x coordinate
        y: float, the y coordinate
        z: float, the z coordinate

    Returns:
        theta: float, the azimuthal angle
        phi:   float, the polar angle
        r:     float, the radius
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    phi = jnp.arctan2(-y, x)
    theta = jnp.arcsin(-z / r)
    return theta, phi, r

def save_panorama_image(panorama_image, folder_path, dataset):
    """
    Save the panorama image to the file path.

    Args:
        panorama_image: jnp.ndarray, shape (H, W, 3), the panorama image
        file_path:      str, the file path to save the panorama image
    """
    # If the folder path does not exist, create the folder
    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(os.path.dirname(folder_path))
    file_path = folder_path + str(dataset) + ".png"
    plt.imsave(file_path, panorama_image.astype(jnp.uint8))
    print(f"Panorama image saved to {file_path}")
